"""把外部 MCP 候选安全地核验并合并到本地 paper.json。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from filelock import FileLock

from app.knowledge.catalog import PAPER_ID_PATTERN, rebuild_catalog
from app.knowledge.identity import (
    build_identity,
    rename_raw_pdf_to_verified_title,
    rollback_raw_pdf_rename,
)
from app.storage import write_json_atomic


class PaperMetadataResolver(Protocol):
    async def resolve_paper(
        self,
        title: str,
        limit: int = 5,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class MetadataAcceptancePolicy:
    min_title_score: float = 0.98
    min_source_count: int = 2
    min_score_margin: float = 0.05


@dataclass(frozen=True)
class MetadataDecision:
    status: str
    selected_index: int | None
    reason: str


@dataclass(frozen=True)
class MetadataEnrichmentResult:
    paper_id: str
    paper_json: Path
    review_path: Path | None
    extracted_title: str
    updated: bool
    decision: MetadataDecision
    selected_candidate: dict[str, Any] | None
    candidates: list[dict[str, Any]]
    provider_failures: list[dict[str, Any]]
    document_id: str | None
    work_id: str | None
    raw_pdf: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "paper_json": str(self.paper_json),
            "review_path": str(self.review_path) if self.review_path else None,
            "extracted_title": self.extracted_title,
            "updated": self.updated,
            "decision": asdict(self.decision),
            "selected_candidate": self.selected_candidate,
            "candidate_count": len(self.candidates),
            "candidates": self.candidates,
            "provider_failures": self.provider_failures,
            "document_id": self.document_id,
            "work_id": self.work_id,
            "raw_pdf": self.raw_pdf,
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_paper_path(project_root: Path, paper_id: str) -> Path:
    if PAPER_ID_PATTERN.fullmatch(paper_id) is None:
        raise ValueError("paper_id 必须符合 P_ + 12 位小写十六进制格式。")
    return (
        project_root.expanduser().resolve()
        / "data"
        / "canonical"
        / paper_id
        / "paper.json"
    )


def load_document(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("paper.json 必须是 JSON 对象。")
    return payload


def require_title(document: dict[str, Any]) -> str:
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("paper.json.metadata 必须是 JSON 对象。")
    title = metadata.get("parser_title") or metadata.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("paper.json 中没有可用于外部核验的标题。")
    return title.strip()


def valid_candidate(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    title = value.get("title")
    score = value.get("match_score")
    if not isinstance(title, str) or not title.strip():
        return None
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return None
    return value


def choose_candidate(
    candidates: list[dict[str, Any]],
    policy: MetadataAcceptancePolicy,
    selected_index: int | None = None,
) -> MetadataDecision:
    if selected_index is not None:
        if selected_index < 0 or selected_index >= len(candidates):
            raise ValueError("selected_index 超出候选范围。")
        return MetadataDecision(
            status="selected",
            selected_index=selected_index,
            reason="用户或本地 Agent 显式选择了候选。",
        )
    if not candidates:
        return MetadataDecision(
            status="no_match",
            selected_index=None,
            reason="外部 MCP 没有返回有效候选。",
        )

    top = candidates[0]
    score = float(top["match_score"])
    sources = top.get("sources")
    source_count = len(sources) if isinstance(sources, list) else 0
    authors = top.get("authors")
    has_authors = isinstance(authors, list) and bool(authors)
    year = top.get("publication_year")
    has_year = isinstance(year, int) and not isinstance(year, bool)
    doi = top.get("doi")
    has_doi = isinstance(doi, str) and bool(doi.strip())
    runner_up = float(candidates[1]["match_score"]) if len(candidates) > 1 else 0.0

    reasons: list[str] = []
    if score < policy.min_title_score:
        reasons.append("标题匹配分数不足")
    if source_count < policy.min_source_count:
        reasons.append("独立数据源不足")
    if score - runner_up < policy.min_score_margin:
        reasons.append("前两名候选分差不足")
    if not has_authors:
        reasons.append("作者缺失")
    if not has_year:
        reasons.append("年份缺失")
    if not has_doi:
        reasons.append("DOI 缺失")

    if reasons:
        return MetadataDecision(
            status="review_required",
            selected_index=None,
            reason="；".join(reasons) + "。",
        )
    return MetadataDecision(
        status="verified",
        selected_index=0,
        reason="满足标题、来源、候选分差、作者、年份和 DOI 的自动核验门槛。",
    )


def merge_verified_metadata(
    document: dict[str, Any],
    extracted_title: str,
    candidate: dict[str, Any],
    decision: MetadataDecision,
) -> None:
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("paper.json.metadata 必须是 JSON 对象。")
    metadata.update(
        {
            "parser_title": extracted_title,
            "title": candidate["title"],
            "authors": candidate.get("authors") or [],
            "abstract": candidate.get("abstract"),
            "year": candidate.get("publication_year"),
            "doi": candidate.get("doi"),
            "venue": candidate.get("venue"),
            "external_ids": candidate.get("external_ids") or {},
            "verification": {
                "status": decision.status,
                "method": "academic-discovery-mcp",
                "title_match_score": candidate.get("match_score"),
                "sources": candidate.get("sources") or [],
                "verified_at": utc_now_iso(),
                "reason": decision.reason,
            },
        }
    )


def normalize_verified_paper(
    project_root: Path,
    paper_id: str,
) -> dict[str, Any]:
    """不访问网络，为已有核验元数据补身份并规范化 raw PDF 文件名。"""

    root = project_root.expanduser().resolve()
    paper_json = canonical_paper_path(root, paper_id)
    lock = FileLock(paper_json.parent / ".metadata.lock")
    with lock:
        document = load_document(paper_json)
        metadata = document.get("metadata")
        source = document.get("source")
        if not isinstance(metadata, dict) or not isinstance(source, dict):
            raise ValueError("paper.json.metadata 和 source 必须是 JSON 对象。")
        verification = metadata.get("verification")
        status = verification.get("status") if isinstance(verification, dict) else None
        if status not in {"verified", "selected"}:
            raise ValueError("只有已核验或显式选择的标题才能用于规范化文件名。")
        title = metadata.get("title")
        sha256 = source.get("sha256")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("已核验 metadata.title 不能为空。")
        if not isinstance(sha256, str):
            raise ValueError("paper.json.source.sha256 不能为空。")
        identity = build_identity(
            paper_id=paper_id,
            sha256=sha256,
            metadata=metadata,
        )
        if identity["work_id"] is None:
            raise ValueError("已核验元数据不足以生成 work_id。")
        document["identity"] = identity
        rename = rename_raw_pdf_to_verified_title(
            project_root=root,
            paper_id=paper_id,
            document=document,
            verified_title=title,
        )
        try:
            write_json_atomic(paper_json, document)
        except Exception:
            rollback_raw_pdf_rename(rename)
            raise
    rebuild_catalog(root)
    updated_source = document["source"]
    return {
        "paper_id": paper_id,
        "document_id": identity["document_id"],
        "work_id": identity["work_id"],
        "work_id_method": identity["work_id_method"],
        "paper_json": str(paper_json),
        "old_raw_pdf": str(rename.old_path),
        "raw_pdf": updated_source["raw_pdf"],
        "renamed": rename.changed,
        "filename_policy": updated_source["filename_policy"],
    }


async def enrich_paper_metadata(
    project_root: Path,
    paper_id: str,
    resolver: PaperMetadataResolver,
    limit: int = 5,
    policy: MetadataAcceptancePolicy | None = None,
    selected_index: int | None = None,
    apply: bool = True,
) -> MetadataEnrichmentResult:
    root = project_root.expanduser().resolve()
    paper_json = canonical_paper_path(root, paper_id)
    initial_document = load_document(paper_json)
    extracted_title = require_title(initial_document)
    payload = await resolver.resolve_paper(extracted_title, limit=limit)
    raw_candidates = payload.get("results")
    candidates = (
        [candidate for value in raw_candidates if (candidate := valid_candidate(value))]
        if isinstance(raw_candidates, list)
        else []
    )
    raw_failures = payload.get("provider_failures")
    failures = (
        [value for value in raw_failures if isinstance(value, dict)]
        if isinstance(raw_failures, list)
        else []
    )
    active_policy = policy or MetadataAcceptancePolicy()
    decision = choose_candidate(
        candidates,
        active_policy,
        selected_index=selected_index,
    )
    candidate = (
        candidates[decision.selected_index]
        if decision.selected_index is not None
        else None
    )

    review_path: Path | None = None
    updated = False
    document_id: str | None = None
    work_id: str | None = None
    raw_pdf: str | None = None
    if apply:
        review_path = (
            root / "data" / "knowledge" / "metadata_reviews" / f"{paper_id}.json"
        )
        review_payload = {
            "paper_id": paper_id,
            "extracted_title": extracted_title,
            "generated_at": utc_now_iso(),
            "resolver": "academic-discovery-mcp",
            "policy": asdict(active_policy),
            "decision": asdict(decision),
            "candidates": candidates,
            "provider_failures": failures,
        }
        write_json_atomic(review_path, review_payload)

        if candidate is not None:
            lock = FileLock(paper_json.parent / ".metadata.lock")
            with lock:
                document = load_document(paper_json)
                current_title = require_title(document)
                if current_title != extracted_title:
                    raise RuntimeError(
                        "MCP 查询期间 paper.json 标题已变化，请重新执行元数据核验。"
                    )
                merge_verified_metadata(
                    document,
                    extracted_title,
                    candidate,
                    decision,
                )
                source = document.get("source")
                if not isinstance(source, dict):
                    raise ValueError("paper.json.source 必须是 JSON 对象。")
                sha256 = source.get("sha256")
                if not isinstance(sha256, str):
                    raise ValueError("paper.json.source.sha256 不能为空。")
                metadata = document.get("metadata")
                if not isinstance(metadata, dict):
                    raise ValueError("paper.json.metadata 必须是 JSON 对象。")
                identity = build_identity(
                    paper_id=paper_id,
                    sha256=sha256,
                    metadata=metadata,
                )
                document["identity"] = identity
                rename = rename_raw_pdf_to_verified_title(
                    project_root=root,
                    paper_id=paper_id,
                    document=document,
                    verified_title=str(candidate["title"]),
                )
                try:
                    write_json_atomic(paper_json, document)
                except Exception:
                    rollback_raw_pdf_rename(rename)
                    raise
                document_id = identity["document_id"]
                work_id = identity["work_id"]
                updated_source = document.get("source")
                raw_pdf = (
                    updated_source.get("raw_pdf")
                    if isinstance(updated_source, dict)
                    and isinstance(updated_source.get("raw_pdf"), str)
                    else None
                )
            rebuild_catalog(root)
            updated = True

    return MetadataEnrichmentResult(
        paper_id=paper_id,
        paper_json=paper_json,
        review_path=review_path,
        extracted_title=extracted_title,
        updated=updated,
        decision=decision,
        selected_candidate=candidate,
        candidates=candidates,
        provider_failures=failures,
        document_id=document_id,
        work_id=work_id,
        raw_pdf=raw_pdf,
    )
