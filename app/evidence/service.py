"""后端无关、可审计的 Evidence Intelligence Pipeline。"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from app.contracts import CandidateEvidence, EvidenceRequest, ExternalEvidenceResolution, SelectedEvidence, VerifiedEvidence, VerifiedEvidenceBundle
from app.corpus import CanonicalCorpusService, CanonicalLocator
from app.indexing.tokenization import tokenize
from app.workspaces import WorkspaceService


Reranker = Callable[[str, Sequence[CandidateEvidence]], Sequence[float]]
ExternalSourceResolver = Callable[[CandidateEvidence], ExternalEvidenceResolution | None]


def _tokens(value: str) -> frozenset[str]:
    return frozenset(tokenize(value))


def _negation_anchors(value: str) -> frozenset[str]:
    """提取否定词后的命题锚点，避免把条件描述当成证据冲突。"""

    anchors: set[str] = set()
    normalized = value.casefold()
    for match in re.finditer(
        r"\b(?:not|cannot|can't|never|doesn't|isn't|aren't)\s+([a-z0-9]+)",
        normalized,
    ):
        anchors.update(_tokens(match.group(1)))
    for match in re.finditer(r"不([\u3400-\u4dbf\u4e00-\u9fff]{1,4}|[a-z0-9]+)", normalized):
        anchors.update(_tokens(match.group(1)))
    return frozenset(anchors)


def _relevance(query: str, value: CandidateEvidence) -> float:
    query_tokens = _tokens(query)
    content_tokens = _tokens(value.content)
    overlap = len(query_tokens & content_tokens) / max(1, len(query_tokens))
    return 0.65 * float(value.score) + 0.35 * overlap


def _token_count(value: str) -> int:
    return max(1, len(tokenize(value)))


class EvidenceIntelligencePipeline:
    """严格按阶段二指定顺序治理候选证据。"""

    def __init__(
        self,
        corpus: CanonicalCorpusService,
        workspaces: WorkspaceService,
        *,
        reranker: Reranker | None = None,
        max_candidates_per_document: int = 4,
        max_selected_per_document: int = 2,
        external_resolver: ExternalSourceResolver | None = None,
    ) -> None:
        if max_candidates_per_document < 1 or max_selected_per_document < 1:
            raise ValueError("文献多样性限制必须大于 0。")
        self.corpus = corpus
        self.workspaces = workspaces
        self.reranker = reranker
        self.max_candidates_per_document = max_candidates_per_document
        self.max_selected_per_document = max_selected_per_document
        self.external_resolver = external_resolver
        self.last_diagnostics: dict[str, Any] = {}

    def verify(self, request: EvidenceRequest, candidates: Sequence[CandidateEvidence]) -> VerifiedEvidenceBundle:
        # 1. Workspace 和 Scope 过滤。
        external = tuple(value for value in candidates if value.source_type == "WEB_LITERATURE")
        local = tuple(value for value in candidates if value.source_type != "WEB_LITERATURE")
        scoped = self.workspaces.filter_candidates(request.workspace_id, request.scope_version, local)
        rejected = [value.candidate_id for value in local if value not in scoped]

        # 2. 来源有效性检查。
        located: list[tuple[CandidateEvidence, CanonicalLocator]] = []
        for value in scoped:
            locator = self.corpus.locate_chunk(value.chunk_id or "")
            if locator is None or locator.document_id != value.document_id:
                rejected.append(value.candidate_id)
                continue
            located.append((value, locator))

        # 3. 去重，以 canonical chunk + relation path 为稳定键。
        deduped: dict[tuple[str, tuple[str, ...]], tuple[CandidateEvidence, CanonicalLocator]] = {}
        for value, locator in located:
            key = (locator.chunk_id, value.relation_path)
            current = deduped.get(key)
            if current is None or value.score > current[0].score:
                if current is not None:
                    rejected.append(current[0].candidate_id)
                deduped[key] = (value, locator)
            else:
                rejected.append(value.candidate_id)

        # 4. 文献多样性限制。
        diverse: list[tuple[CandidateEvidence, CanonicalLocator]] = []
        counts: Counter[str] = Counter()
        for value, locator in sorted(deduped.values(), key=lambda item: (-item[0].score, item[0].candidate_id)):
            if counts[locator.document_id] >= self.max_candidates_per_document:
                rejected.append(value.candidate_id)
                continue
            counts[locator.document_id] += 1
            diverse.append((value, locator))

        # 5. 相关性重排。
        rerank_scores = list(self.reranker(request.query, [value for value, _ in diverse])) if self.reranker else [_relevance(request.query, value) for value, _ in diverse]
        if len(rerank_scores) != len(diverse):
            raise ValueError("Reranker 返回数量与候选数量不一致。")
        ranked = sorted(zip(diverse, rerank_scores, strict=True), key=lambda item: (-float(item[1]), item[0][0].candidate_id))

        verified: list[VerifiedEvidence] = []
        external_seen: set[tuple[str, str]] = set()
        for candidate in external:
            if candidate.workspace_id != request.workspace_id or candidate.scope_version != request.scope_version:
                rejected.append(candidate.candidate_id)
                continue
            if self.external_resolver is None:
                rejected.append(candidate.candidate_id)
                continue
            resolved = self.external_resolver(candidate)
            if resolved is None or resolved.candidate_id != candidate.candidate_id:
                rejected.append(candidate.candidate_id)
                continue
            key = (resolved.canonical_id, resolved.source_locator)
            if key in external_seen:
                rejected.append(candidate.candidate_id)
                continue
            external_seen.add(key)
            if resolved.content != candidate.content:
                rejected.append(candidate.candidate_id)
                continue
            if hashlib.sha256(resolved.content.encode("utf-8")).hexdigest() != resolved.content_hash:
                rejected.append(candidate.candidate_id)
                continue
            verified.append(
                VerifiedEvidence(
                    evidence_id=candidate.evidence_id,
                    candidate_id=candidate.candidate_id,
                    request_id=request.request_id,
                    workspace_id=request.workspace_id,
                    scope_version=request.scope_version,
                    content=resolved.content,
                    work_id=resolved.work_id,
                    verification_method="external_source_resolver",
                    content_hash=resolved.content_hash,
                    evidence_grade=candidate.evidence_grade,
                    directness=candidate.directness or "direct",
                    metadata={**dict(candidate.metadata), **dict(resolved.metadata)},
                    source_type="WEB_LITERATURE",
                    canonical_id=resolved.canonical_id,
                    source_locator=resolved.source_locator,
                    locator_type=resolved.locator_type,
                    publication_date=resolved.publication_date,
                    retrieved_at=resolved.retrieved_at,
                    provider=resolved.provider,
                    validation_status=resolved.validation_status,
                )
            )
        for ((candidate, locator), rerank_score) in ranked:
            # 6. Canonical Corpus is the only local source of truth.
            canonical_content = locator.content
            verified_content = canonical_content

            # 7. Local candidates must directly map to the canonical chunk.
            directness = candidate.directness or "direct"

            # 8. 证据等级分类；analogy 永不提升为 primary。
            grade = self.workspaces.evidence_grade(request.workspace_id, request.scope_version, locator.document_id)
            if grade == "primary" and candidate.evidence_grade in {"candidate", "analogy"}:
                grade = candidate.evidence_grade

            states = ("retrieved", "filtered", "ranked", "backfilled", "verified")
            metadata = {
                **candidate.metadata,
                "pipeline_states": states,
                "retrieval_source": candidate.retrieval_source,
                "rerank_score": float(rerank_score),
            }
            verified.append(
                VerifiedEvidence(
                    evidence_id=candidate.evidence_id,
                    candidate_id=candidate.candidate_id,
                    request_id=request.request_id,
                    workspace_id=request.workspace_id,
                    scope_version=request.scope_version,
                    content=verified_content,
                    work_id=locator.work_id or str(candidate.work_id or locator.document_id),
                    document_id=locator.document_id,
                    chunk_id=locator.chunk_id,
                    page_start=locator.page_start,
                    page_end=locator.page_end,
                    block_ids=locator.block_ids,
                    verification_method="canonical_chunk_match",
                    content_hash=hashlib.sha256(canonical_content.encode("utf-8")).hexdigest(),
                    relation_path=candidate.relation_path,
                    evidence_grade=grade,
                    directness=directness,  # type: ignore[arg-type]
                    source_type="LOCAL_CORPUS",
                    canonical_id=candidate.canonical_id,
                    source_locator=candidate.source_locator,
                    locator_type=candidate.locator_type,
                    publication_date=candidate.publication_date,
                    retrieved_at=candidate.retrieved_at,
                    provider=candidate.provider,
                    validation_status="canonical_locator_verified",
                    metadata=metadata,
                )
            )

        # 9. 问题维度覆盖分析（确定性词项代理；供后续 Planner 覆盖信号使用）。
        dimensions = [value.strip() for value in re.split(r"\?|\band\b|以及|与|和", request.query, flags=re.IGNORECASE) if value.strip()]
        coverage = {
            dimension: [value.evidence_id for value in verified if _tokens(dimension) & _tokens(value.content)]
            for dimension in dimensions
        }

        # 10. 支持/反对冲突检测：保守标记，不自动删除任一侧。
        negative = [
            (value, _negation_anchors(value.content))
            for value in verified
            if _negation_anchors(value.content)
        ]
        positive = [value for value in verified if not any(value is item for item, _ in negative)]
        conflicts = [
            (left.evidence_id, right.evidence_id)
            for left in positive
            for right, anchors in negative
            if anchors & _tokens(left.content)
        ]

        self.last_diagnostics = {
            "input_count": len(candidates),
            "workspace_scope_pass_count": len(scoped),
            "verified_count": len(verified),
            "rejected_count": len(set(rejected)),
            "coverage": coverage,
            "conflicts": conflicts,
            "stage_order": ["workspace_scope_filter", "source_validation", "deduplication", "document_diversity", "reranking", "canonical_verification", "directness", "evidence_grade", "coverage", "conflict_detection"],
        }
        issues = tuple(f"conflict:{left}:{right}" for left, right in conflicts)
        return VerifiedEvidenceBundle(request.request_id, request.workspace_id, request.scope_version, tuple(verified), tuple(dict.fromkeys(rejected)), issues)

    def select(self, request: EvidenceRequest, bundle: VerifiedEvidenceBundle) -> Sequence[SelectedEvidence]:
        # 11. Token 预算下的证据选择。
        budget = int(request.retrieval_options.get("token_budget", 4_000))
        if budget < 1:
            raise ValueError("token_budget 必须大于 0。")
        used = 0
        document_counts: Counter[str] = Counter()
        selected: list[SelectedEvidence] = []
        ordered = sorted(
            bundle.evidence,
            key=lambda value: (
                -float(value.metadata.get("rerank_score", 0.0)),
                value.evidence_id,
            ),
        )
        for value in ordered:
            if len(selected) >= request.top_k:
                break
            count = _token_count(value.content)
            source_group = value.document_id or value.canonical_id or value.evidence_id
            if used + count > budget or document_counts[source_group] >= self.max_selected_per_document:
                continue
            selected_value = replace(value, state="selected", metadata={**value.metadata, "pipeline_states": (*value.metadata.get("pipeline_states", ()), "selected")})
            selected.append(SelectedEvidence(selected_value, len(selected) + 1, "token_budget_relevance_diversity", count))
            used += count
            document_counts[source_group] += 1
        self.last_diagnostics.update({"selected_count": len(selected), "selected_tokens": used, "token_budget": budget})
        return tuple(selected)
