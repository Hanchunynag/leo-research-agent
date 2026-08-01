"""来源感知、预算受控的证据上下文组装。"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Any, Sequence

from app.context.models import ContextBundle, EvidenceItem
from app.indexing.tokenization import normalize_search_text, token_count


DEFAULT_CONTEXT_TOKEN_BUDGET = 6000
DEFAULT_MAX_EVIDENCE = 8
DEFAULT_MIN_CONTENT_TOKENS = 20


def _bounded_integer(value: int, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or value < minimum or value > maximum:
        raise ValueError(f"{field} 必须在 {minimum} 到 {maximum} 之间。")
    return value


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(
        dict.fromkeys(item.strip() for item in value if isinstance(item, str) and item.strip())
    )


def _content_fingerprint(value: str) -> str:
    normalized = re.sub(r"\s+", " ", normalize_search_text(value)).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _same_source(context: dict[str, Any], work_id: str, document_id: str) -> bool:
    context_work = context.get("work_id")
    context_document = context.get("document_id")
    return (
        (context_work is None or context_work == work_id)
        and (context_document is None or context_document == document_id)
    )


def _context_label(context: dict[str, Any], kind: str) -> str:
    section = " > ".join(_string_list(context.get("section_path"))) or "Unsectioned"
    start = context.get("page_start")
    end = context.get("page_end")
    pages = f"{start}-{end}" if isinstance(start, int) and isinstance(end, int) else "unknown"
    return f"[{kind} | Section: {section} | Pages: {pages}]"


def _candidate_evidence_content(
    candidate: dict[str, Any],
    work_id: str,
    document_id: str,
    seen_content_fingerprints: set[str],
) -> tuple[str, list[str], list[int], set[str], int]:
    parts: list[str] = []
    context_block_ids: list[str] = []
    context_pages: list[int] = []
    new_fingerprints: set[str] = set()
    omitted_context_count = 0

    contexts: list[tuple[str, dict[str, Any]]] = []
    parent_contexts = candidate.get("parent_contexts")
    if isinstance(parent_contexts, list):
        contexts.extend(
            ("Parent context", value)
            for value in parent_contexts
            if isinstance(value, dict)
        )
    overlap_context = candidate.get("overlap_context")
    if isinstance(overlap_context, dict):
        contexts.append(("Previous same-section context", overlap_context))

    for label, context in contexts:
        content = context.get("content")
        if (
            not isinstance(content, str)
            or not content.strip()
            or not _same_source(context, work_id, document_id)
        ):
            omitted_context_count += 1
            continue
        fingerprint = _content_fingerprint(content)
        if fingerprint in seen_content_fingerprints or fingerprint in new_fingerprints:
            omitted_context_count += 1
            continue
        new_fingerprints.add(fingerprint)
        parts.append(f"{_context_label(context, label)}\n{content.strip()}")
        context_block_ids.extend(_string_list(context.get("block_ids")))
        for field in ("page_start", "page_end"):
            page = context.get(field)
            if isinstance(page, int) and page > 0:
                context_pages.append(page)

    primary_content = str(candidate.get("content") or "").strip()
    parts.append(f"[Primary evidence]\n{primary_content}")
    return (
        "\n\n".join(parts),
        list(dict.fromkeys(context_block_ids)),
        context_pages,
        new_fingerprints,
        omitted_context_count,
    )


def _render_header(item: EvidenceItem) -> str:
    section = " > ".join(item.section_path) or "Unsectioned"
    block_ids = ", ".join(item.block_ids)
    return (
        f"[{item.source_id}]\n"
        f"Title: {item.title}\n"
        f"Work ID: {item.work_id}\n"
        f"Document ID: {item.document_id}\n"
        f"Section: {section}\n"
        f"Pages: {item.page_start}-{item.page_end}\n"
        f"Block IDs: {block_ids}\n"
        "Evidence:\n"
    )


def render_evidence_item(item: EvidenceItem) -> str:
    return f"{_render_header(item)}{item.content}"


def _truncate_to_budget(value: str, maximum_tokens: int) -> str:
    if maximum_tokens < 1:
        return ""
    if token_count(value) <= maximum_tokens:
        return value
    suffix = " …"
    low = 0
    high = len(value)
    while low < high:
        middle = (low + high + 1) // 2
        if token_count(value[:middle].rstrip() + suffix) <= maximum_tokens:
            low = middle
        else:
            high = middle - 1
    prefix = value[:low].rstrip()
    if not prefix:
        return ""
    whitespace = max(prefix.rfind(" "), prefix.rfind("\n"))
    if whitespace >= int(len(prefix) * 0.7):
        prefix = prefix[:whitespace].rstrip()
    return prefix + suffix


def _candidate_item(
    candidate: dict[str, Any],
    source_id: str,
    content: str,
    context_block_ids: list[str],
    context_pages: list[int],
) -> EvidenceItem | None:
    required = {
        field: candidate.get(field)
        for field in ("chunk_id", "work_id", "document_id", "title")
    }
    if not all(isinstance(value, str) and value.strip() for value in required.values()):
        return None
    primary_block_ids = _string_list(candidate.get("block_ids"))
    if not primary_block_ids or not content.strip():
        return None
    page_start = candidate.get("page_start")
    page_end = candidate.get("page_end")
    if (
        not isinstance(page_start, int)
        or isinstance(page_start, bool)
        or not isinstance(page_end, int)
        or isinstance(page_end, bool)
        or page_start < 1
        or page_end < page_start
    ):
        return None
    pages = [page_start, page_end, *context_pages]
    score = candidate.get("score")
    normalized_score = (
        float(score)
        if isinstance(score, (int, float)) and not isinstance(score, bool)
        else None
    )
    year = candidate.get("year")
    rank = candidate.get("rank")
    normalized_rank = (
        rank
        if isinstance(rank, int) and not isinstance(rank, bool) and rank > 0
        else int(source_id.removeprefix("S"))
    )
    return EvidenceItem(
        source_id=source_id,
        rank=normalized_rank,
        score=normalized_score,
        retrieval_source=str(candidate.get("retrieval_source") or "unknown"),
        chunk_id=str(required["chunk_id"]),
        work_id=str(required["work_id"]),
        document_id=str(required["document_id"]),
        paper_id=(
            str(candidate["paper_id"])
            if isinstance(candidate.get("paper_id"), str)
            else None
        ),
        title=str(required["title"]),
        authors=_string_list(candidate.get("authors")),
        year=year if isinstance(year, int) and not isinstance(year, bool) else None,
        doi=str(candidate["doi"]) if isinstance(candidate.get("doi"), str) else None,
        section_path=_string_list(candidate.get("section_path")),
        page_start=min(pages),
        page_end=max(pages),
        primary_block_ids=primary_block_ids,
        block_ids=list(dict.fromkeys([*primary_block_ids, *context_block_ids])),
        content_types=_string_list(candidate.get("content_types")),
        content=content,
        truncated=False,
        token_count=0,
        evidence_id=(
            str(candidate["evidence_id"])
            if isinstance(candidate.get("evidence_id"), str)
            else None
        ),
        origin=str(candidate.get("origin") or "newly_retrieved"),
    )


def assemble_context_bundle(
    query: str,
    retrieval_mode: str,
    results: Sequence[dict[str, Any]],
    *,
    token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
    max_evidence: int = DEFAULT_MAX_EVIDENCE,
    max_evidence_per_work: int = 2,
    retrieval_diagnostics: dict[str, Any] | None = None,
) -> ContextBundle:
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("query 不能为空。")
    if retrieval_mode not in {"fast", "accurate"}:
        raise ValueError("retrieval_mode 必须是 fast 或 accurate。")
    budget = _bounded_integer(token_budget, "token_budget", 100, 100_000)
    evidence_limit = _bounded_integer(max_evidence, "max_evidence", 1, 50)
    per_work = _bounded_integer(
        max_evidence_per_work,
        "max_evidence_per_work",
        1,
        20,
    )

    evidence: list[EvidenceItem] = []
    rendered: list[str] = []
    seen_chunk_ids: set[str] = set()
    seen_content_fingerprints: set[str] = set()
    selected_document_by_work: dict[str, str] = {}
    work_counts: defaultdict[str, int] = defaultdict(int)
    skipped_reasons: Counter[str] = Counter()
    omitted_context_count = 0
    truncated_count = 0

    for candidate in results:
        if len(evidence) >= evidence_limit:
            break
        chunk_id = candidate.get("chunk_id")
        work_id = candidate.get("work_id")
        document_id = candidate.get("document_id")
        primary_content = candidate.get("content")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (chunk_id, work_id, document_id, primary_content)
        ):
            skipped_reasons["invalid_source_metadata"] += 1
            continue
        assert isinstance(chunk_id, str)
        assert isinstance(work_id, str)
        assert isinstance(document_id, str)
        assert isinstance(primary_content, str)
        if chunk_id in seen_chunk_ids:
            skipped_reasons["duplicate_chunk"] += 1
            continue
        primary_fingerprint = _content_fingerprint(primary_content)
        if primary_fingerprint in seen_content_fingerprints:
            skipped_reasons["duplicate_content"] += 1
            continue
        selected_document = selected_document_by_work.get(work_id)
        if selected_document is not None and selected_document != document_id:
            skipped_reasons["duplicate_work_document_version"] += 1
            continue
        if work_counts[work_id] >= per_work:
            skipped_reasons["per_work_limit"] += 1
            continue

        (
            content,
            context_block_ids,
            context_pages,
            context_fingerprints,
            omitted,
        ) = _candidate_evidence_content(
            candidate,
            work_id,
            document_id,
            {*seen_content_fingerprints, primary_fingerprint},
        )
        source_id = f"S{len(evidence) + 1}"
        item = _candidate_item(
            candidate,
            source_id,
            content,
            context_block_ids,
            context_pages,
        )
        if item is None:
            skipped_reasons["invalid_source_metadata"] += 1
            continue
        current_text = "\n\n".join(rendered)
        remaining = budget - token_count(current_text)
        full_block = render_evidence_item(item)
        if token_count(full_block) > remaining:
            header_tokens = token_count(_render_header(item))
            content_budget = remaining - header_tokens
            if content_budget < DEFAULT_MIN_CONTENT_TOKENS:
                skipped_reasons["token_budget_exhausted"] += 1
                break
            truncated_content = _truncate_to_budget(content, content_budget)
            if not truncated_content:
                skipped_reasons["token_budget_exhausted"] += 1
                break
            item = replace(item, content=truncated_content, truncated=True)
            full_block = render_evidence_item(item)
            truncated_count += 1

        item = replace(item, token_count=token_count(full_block))
        evidence.append(item)
        rendered.append(full_block)
        seen_chunk_ids.add(chunk_id)
        seen_content_fingerprints.add(primary_fingerprint)
        seen_content_fingerprints.update(context_fingerprints)
        selected_document_by_work.setdefault(work_id, document_id)
        work_counts[work_id] += 1
        omitted_context_count += omitted
        if item.truncated:
            break

    context_text = "\n\n".join(rendered)
    diagnostics = {
        "input_result_count": len(results),
        "selected_evidence_count": len(evidence),
        "truncated_evidence_count": truncated_count,
        "omitted_context_count": omitted_context_count,
        "skipped_candidate_reasons": dict(sorted(skipped_reasons.items())),
        "selected_document_by_work": dict(sorted(selected_document_by_work.items())),
        "token_counter": "deterministic_approximation_v1",
        "retrieval": retrieval_diagnostics or {},
    }
    return ContextBundle(
        query=cleaned_query,
        retrieval_mode=retrieval_mode,
        evidence=evidence,
        context_text=context_text,
        token_budget=budget,
        token_count=token_count(context_text),
        diagnostics=diagnostics,
    )
