"""Append-only Topic 消息、严格前缀诊断与非破坏性 Compaction。"""

from __future__ import annotations

import hashlib
from typing import Any

from app.agentic.models import CompactionReport
from app.agentic.provider import AGENTIC_ANSWER_SYSTEM_PROMPT
from app.agentic.store import AgenticSessionStore, stable_json
from app.indexing.tokenization import token_count


def build_topic_messages(events: list[dict[str, Any]]) -> list[dict[str, str]]:
    """按 ordinal 构造确定性消息；Compaction 后从新前缀开始。"""

    ordered = sorted(events, key=lambda item: int(item["ordinal"]))
    latest_compaction = max(
        (
            index
            for index, event in enumerate(ordered)
            if event["event_type"] == "compaction"
        ),
        default=-1,
    )
    if latest_compaction >= 0:
        ordered = ordered[latest_compaction:]
    messages = [{"role": "system", "content": AGENTIC_ANSWER_SYSTEM_PROMPT}]
    for event in ordered:
        event_type = str(event["event_type"])
        role = "assistant" if event_type == "answer" else "user"
        messages.append(
            {
                "role": role,
                "content": f"[EVENT:{event_type}]\n{stable_json(event['content'])}",
            }
        )
    return messages


def prompt_cache_diagnostics(
    messages: list[dict[str, str]],
    *,
    new_message_count: int,
    usage: dict[str, Any] | None,
) -> dict[str, Any]:
    """只用真实 Provider usage 计算缓存率，其余字段保留 null。"""

    prefix_count = max(1, len(messages) - max(0, new_message_count))
    prefix = stable_json(messages[:prefix_count])
    usage = usage or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    hit = usage.get("prompt_cache_hit_tokens")
    miss = usage.get("prompt_cache_miss_tokens")
    rate: float | None = None
    if isinstance(hit, (int, float)) and isinstance(miss, (int, float)):
        denominator = hit + miss
        if denominator > 0:
            rate = round(hit / denominator, 6)
    return {
        "prompt_tokens": prompt_tokens if isinstance(prompt_tokens, (int, float)) else None,
        "completion_tokens": (
            completion_tokens
            if isinstance(completion_tokens, (int, float))
            else None
        ),
        "cache_hit_tokens": hit if isinstance(hit, (int, float)) else None,
        "cache_miss_tokens": miss if isinstance(miss, (int, float)) else None,
        "cache_hit_rate": rate,
        "prefix_hash": hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
        "message_count": len(messages),
        "new_message_count": new_message_count,
    }


def compact_topic(
    store: AgenticSessionStore,
    session_id: str,
    topic_id: str,
    *,
    recent_event_count: int = 8,
) -> CompactionReport:
    """追加 compaction 事件，保留旧事件，并构造新的稳定消息前缀。"""

    topic = store.get_topic(session_id, topic_id)
    events = store.list_events(session_id, topic_id)
    evidence = store.list_evidence(session_id, topic_id)
    before_messages = build_topic_messages(events)
    before_tokens = token_count(stable_json(before_messages))
    recent = events[-recent_event_count:]
    evidence_summaries = [
        {
            "evidence_id": item["evidence_id"],
            "chunk_id": item["chunk_id"],
            "work_id": item.get("work_id"),
            "document_id": item.get("document_id"),
            "title": item.get("title"),
            "section_path": item.get("section_path"),
            "page_start": item.get("page_start"),
            "page_end": item.get("page_end"),
            "block_ids": item.get("block_ids"),
            "evidence_summary": str(item.get("content") or "")[:600],
        }
        for item in evidence
    ]
    content = {
        "topic_summary": topic["topic_summary"],
        "user_goal": topic["user_goal"],
        "confirmed_facts": topic["confirmed_facts"],
        "open_questions": topic["open_questions"],
        "evidence_registry": evidence_summaries,
        "recent_events": [
            {
                "ordinal": event["ordinal"],
                "event_type": event["event_type"],
                "content": event["content"],
            }
            for event in recent
        ],
        "unfinished_task": topic["open_questions"],
    }
    event = store.append_event(session_id, topic_id, "compaction", content)
    after_messages = build_topic_messages(store.list_events(session_id, topic_id))
    after_tokens = token_count(stable_json(after_messages))
    discarded = sorted(
        {
            str(value["event_type"])
            for value in events[:-recent_event_count]
            if value["event_type"] not in {"answer", "user_query"}
        }
    )
    return CompactionReport(
        before_tokens=before_tokens,
        after_tokens=after_tokens,
        retained_evidence_ids=[str(item["evidence_id"]) for item in evidence],
        discarded_event_types=discarded,
        compaction_ordinal=int(event["ordinal"]),
    )
