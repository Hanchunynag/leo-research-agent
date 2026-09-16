"""Introduction 引用覆盖合同。

引用数量必须按不同论文统计，而不是按 Chunk、Evidence 或 BibKey 的数量
统计。同一篇论文被多个 section/chunk 支持时只能计为一篇。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping, Sequence
from typing import Any


INTRODUCTION_MINIMUM_UNIQUE_PAPERS = 5
INTRODUCTION_PAPER_RETRIEVAL_LIMIT = 20
INTRODUCTION_SECTION_RETRIEVAL_LIMIT = 30
INTRODUCTION_EVIDENCE_LIMIT = 30


def _metadata(item: Mapping[str, Any]) -> Mapping[str, Any]:
    value = item.get("metadata")
    return value if isinstance(value, Mapping) else {}


def paper_identity(item: Mapping[str, Any]) -> str | None:
    """Return a stable paper identity for one verified evidence projection.

    ``paper_id`` is preferred for local canonical papers. External literature
    may not have a local paper id, so canonical_id/DOI/work/document identity
    are accepted as fallbacks. A BibKey alone is deliberately not sufficient:
    two BibKeys can refer to the same paper and must not inflate coverage.
    """

    metadata = _metadata(item)
    for field, prefix in (
        ("paper_id", "paper"),
        ("canonical_id", "canonical"),
        ("doi", "doi"),
        ("arxiv_id", "arxiv"),
        ("work_id", "work"),
        ("document_id", "document"),
    ):
        value = item.get(field)
        if value in (None, "", []):
            value = metadata.get(field)
        if value not in (None, "", []):
            return f"{prefix}:{str(value).strip().casefold()}"
    return None


def unique_paper_identities(evidence: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    values = {
        identity
        for item in evidence
        if isinstance(item, Mapping)
        for identity in (paper_identity(item),)
        if identity
    }
    return tuple(sorted(values))


@dataclass(frozen=True, slots=True)
class CitationCoverage:
    """可审计的 Introduction 唯一论文引用覆盖结果。"""

    minimum_unique_papers: int
    available_paper_ids: tuple[str, ...]
    cited_paper_ids: tuple[str, ...]
    cited_citation_keys: tuple[str, ...]
    unresolved_citation_keys: tuple[str, ...] = ()

    @property
    def available_paper_count(self) -> int:
        return len(self.available_paper_ids)

    @property
    def cited_paper_count(self) -> int:
        return len(self.cited_paper_ids)

    @property
    def missing_paper_count(self) -> int:
        return max(0, self.minimum_unique_papers - self.cited_paper_count)

    @property
    def sufficient(self) -> bool:
        return self.cited_paper_count >= self.minimum_unique_papers

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "available_paper_count": self.available_paper_count,
            "cited_paper_count": self.cited_paper_count,
            "missing_paper_count": self.missing_paper_count,
            "sufficient": self.sufficient,
        }


def citation_coverage(
    citation_keys: Sequence[str],
    citation_catalog: Mapping[str, Sequence[str]],
    evidence: Mapping[str, Mapping[str, Any]],
    *,
    minimum_unique_papers: int = INTRODUCTION_MINIMUM_UNIQUE_PAPERS,
) -> CitationCoverage:
    """Calculate coverage from draft citation keys and verified evidence.

    Only evidence reachable through an unambiguous, runtime-built citation
    catalog entry is counted. This prevents an invented BibKey, an unresolved
    CitationRequirement, or one duplicated BibKey pointing at multiple works
    from satisfying the 5-paper contract.
    """

    if isinstance(minimum_unique_papers, bool) or minimum_unique_papers < 1:
        raise ValueError("minimum_unique_papers 必须为正整数。")
    # A paper with only ``paper_id`` is not yet citation-ready. It may be a
    # valid retrieval result, but until CitationResolutionService produces an
    # existing/proposed binding (or the legacy adapter supplies a key), it
    # cannot count toward the Introduction bibliography contract.
    available_ids: set[str] = set()
    cited_ids: set[str] = set()
    cited_keys: list[str] = []
    unresolved: list[str] = []
    catalog_identities: dict[str, frozenset[str]] = {}
    for raw_key, raw_evidence_ids in citation_catalog.items():
        identities = frozenset(
            identity
            for evidence_id in raw_evidence_ids
            for identity in (paper_identity(evidence.get(str(evidence_id), {})),)
            if identity
        )
        # A single BibKey must never be used as evidence for multiple works.
        # Same-paper multi-chunk evidence is fine because it collapses to one
        # identity here.
        if len(identities) == 1:
            catalog_identities[str(raw_key)] = identities
            available_ids.update(identities)
    for raw_key in citation_keys:
        key = str(raw_key).strip()
        if not key or key in cited_keys:
            continue
        identities = catalog_identities.get(key, frozenset())
        if not identities:
            unresolved.append(key)
            continue
        cited_keys.append(key)
        cited_ids.update(identities)
    return CitationCoverage(
        minimum_unique_papers=minimum_unique_papers,
        available_paper_ids=tuple(sorted(available_ids)),
        cited_paper_ids=tuple(sorted(cited_ids)),
        cited_citation_keys=tuple(cited_keys),
        unresolved_citation_keys=tuple(unresolved),
    )


def insufficient_introduction_message(coverage: CitationCoverage) -> str:
    return (
        "Introduction 至少需要引用 "
        f"{coverage.minimum_unique_papers} 篇不同论文；"
        f"当前 Draft 只有 {coverage.cited_paper_count} 篇，"
        f"还缺 {coverage.missing_paper_count} 篇。"
    )


def insufficient_research_message(
    coverage: CitationCoverage,
    *,
    candidate_paper_count: int | None = None,
) -> str:
    candidate = (
        f"Paper-Level 已召回 {candidate_paper_count} 篇候选，"
        if candidate_paper_count is not None
        else ""
    )
    return (
        f"{candidate}但经过内容证据验证并可绑定引用的不同论文只有 "
        f"{coverage.available_paper_count} 篇；Introduction 至少需要 "
        f"{coverage.minimum_unique_papers} 篇。请先补充相关论文或检查索引/引用元数据。"
    )
