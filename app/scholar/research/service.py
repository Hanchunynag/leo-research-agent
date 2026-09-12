"""把现有 UnifiedKnowledgeService 投影为 Scholar Research Capability。

本模块不实现检索算法。它只负责 request normalization、结果投影、canonical
locator 读取和复用现有 EvidenceIntelligencePipeline 的验证/选择。
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date, datetime
from time import perf_counter
from typing import Any

from app.contracts import CandidateEvidence, EvidenceRequest, VerifiedEvidence
from app.corpus import CanonicalCorpusService, CanonicalLocator
from app.evidence import EvidenceIntelligencePipeline
from app.knowledge_engine import UnifiedKnowledgeService
from app.knowledge.identity import normalize_doi
from app.research.harness import BudgetExceeded
from app.scholar.models import EvidencePack
from app.scholar.research.errors import (
    EvidenceValidationError,
    InvalidEvidenceLocator,
    ResearchBudgetExceeded,
    ResearchRequestError,
    SourceNotFound,
    WebLiteratureError,
)
from app.scholar.research.freshness import FreshnessDecision, FreshnessPolicy
from app.scholar.research.models import (
    EvidenceCandidate,
    EvidenceRef,
    EvidenceSource,
    LiteratureSearchRequest,
    PaperCandidate,
    PaperSearchResult,
    ResearchRequest,
    SectionCandidate,
    SectionSearchResult,
)
from app.scholar.research.web import WebLiteratureAdapter, canonical_identity
from app.workspaces import WorkspaceService


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item.strip())


def _normalized(value: str) -> str:
    return re.sub(r"[^\w\u3400-\u4dbf\u4e00-\u9fff]+", " ", value.casefold()).strip()


def _same_text(left: str, right: str) -> bool:
    return " ".join(left.split()) == " ".join(right.split())


class ResearchCapabilityService:
    """上层 Agent 可组合使用的 paper/section/evidence 能力门面。"""

    def __init__(
        self,
        knowledge: UnifiedKnowledgeService,
        *,
        corpus: CanonicalCorpusService | None = None,
        evidence: EvidenceIntelligencePipeline | None = None,
        workspaces: WorkspaceService | None = None,
        workspace_id: str | None = None,
        scope_version: int | None = None,
        web: WebLiteratureAdapter | None = None,
    ) -> None:
        self.knowledge = knowledge
        inferred_evidence = evidence or getattr(knowledge, "evidence", None)
        inferred_corpus = corpus or getattr(inferred_evidence, "corpus", None)
        inferred_workspaces = workspaces or getattr(inferred_evidence, "workspaces", None)
        if inferred_corpus is None:
            raise TypeError("ResearchCapabilityService 需要 CanonicalCorpusService。")
        if inferred_evidence is None:
            raise TypeError("ResearchCapabilityService 需要 EvidenceIntelligencePipeline。")
        if inferred_workspaces is None:
            raise TypeError("ResearchCapabilityService 需要 WorkspaceService。")
        self.corpus = inferred_corpus
        self.evidence = inferred_evidence
        self.workspaces = inferred_workspaces
        self.workspace_id = str(workspace_id or getattr(knowledge, "workspace_id", "default"))
        self.scope_version = int(scope_version or getattr(knowledge, "scope_version", 1))
        self.last_diagnostics: dict[str, Any] = {}
        self.web = web
        if web is not None and hasattr(self.evidence, "external_resolver"):
            self.evidence.external_resolver = web.resolve

    def _request(
        self,
        value: ResearchRequest | str,
        *,
        top_k: int | None = None,
        purpose: str | None = None,
        target_claims: Sequence[str] | None = None,
        paper_ids: Sequence[str] | None = None,
        preferred_section_types: Sequence[str] | None = None,
    ) -> ResearchRequest:
        if isinstance(value, ResearchRequest):
            if value.workspace_id != self.workspace_id or value.scope_version != self.scope_version:
                raise ResearchRequestError("ResearchRequest workspace/scope 与 Capability 不一致。")
            updates: dict[str, Any] = {}
            if top_k is not None:
                updates["budget"] = replace(
                    value.budget,
                    max_papers=min(value.budget.max_papers, top_k),
                    max_sections=min(value.budget.max_sections, top_k),
                    max_evidence_items=min(value.budget.max_evidence_items, top_k),
                )
            if target_claims is not None:
                updates["target_claims"] = tuple(target_claims)
            if paper_ids is not None:
                updates["paper_ids"] = tuple(paper_ids)
            if preferred_section_types is not None:
                updates["preferred_section_types"] = tuple(preferred_section_types)
            return replace(value, **updates) if updates else value
        if not isinstance(value, str) or not value.strip():
            raise ResearchRequestError("query 不能为空。")
        query_top_k = top_k if top_k is not None else 10
        request = ResearchRequest(
            request_id=f"REQ_{secrets.token_hex(8)}",
            query=value,
            workspace_id=self.workspace_id,
            scope_version=self.scope_version,
            purpose=purpose or "background",  # type: ignore[arg-type]
            target_claims=tuple(target_claims or ()),
            paper_ids=tuple(paper_ids or ()),
            preferred_section_types=tuple(preferred_section_types or ()),
        )
        return replace(
            request,
            budget=replace(
                request.budget,
                max_papers=query_top_k,
                max_sections=query_top_k,
                max_evidence_items=query_top_k,
            ),
        )

    @staticmethod
    def _limit(request: ResearchRequest, requested: int | None, budget_name: str) -> int:
        budget = getattr(request.budget, budget_name)
        limit = budget if requested is None else requested
        if isinstance(limit, bool) or limit < 1:
            raise ResearchRequestError("top_k 必须为正整数。")
        if limit > budget:
            raise ResearchBudgetExceeded(f"{budget_name} 预算为 {budget}，请求为 {limit}。")
        return int(limit)

    def _evidence_request(self, request: ResearchRequest, limit: int) -> EvidenceRequest:
        return EvidenceRequest(
            request_id=request.request_id,
            workspace_id=request.workspace_id,
            scope_version=request.scope_version,
            query=request.query,
            top_k=limit,
            document_ids=request.document_ids,
            retrieval_options={
                "token_budget": int(request.metadata.get("token_budget", 100_000)),
                "capability": "scholar_research",
            },
        )

    def _filters(self, request: ResearchRequest) -> dict[str, Any]:
        filters: dict[str, Any] = {}
        if request.year_from is not None:
            filters["year_from"] = request.year_from
        if request.year_to is not None:
            filters["year_to"] = request.year_to
        if request.paper_ids:
            filters["paper_ids"] = list(request.paper_ids)
        if request.document_ids:
            filters["document_ids"] = list(request.document_ids)
        return filters

    def _retrieve(
        self,
        request: ResearchRequest,
        *,
        limit: int,
        paper_filters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        started = perf_counter()
        response = self.knowledge.retrieve(
            request.query,
            mode="hierarchical",
            limit=limit,
            paper_limit=max(limit, 1),
            paper_candidate_limit=min(max(limit * 3, limit), 100),
            chunk_candidate_limit=min(max(limit * 4, limit), 100),
            paper_filters=dict(paper_filters or {}),
            workspace_id=request.workspace_id,
            scope_version=request.scope_version,
        )
        if not isinstance(response, Mapping):
            raise RuntimeError("UnifiedKnowledgeService 返回了非对象结果。")
        self.last_diagnostics = {
            "request_id": request.request_id,
            "query": request.query[:160],
            "elapsed_ms": round((perf_counter() - started) * 1000, 3),
            "retriever": response.get("retriever"),
        }
        return dict(response)

    def search_papers(
        self,
        request: ResearchRequest | str,
        *,
        top_k: int | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        document_ids: Sequence[str] | None = None,
        paper_ids: Sequence[str] | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> PaperSearchResult:
        normalized = self._request(request, top_k=top_k, paper_ids=paper_ids)
        if year_from is not None or year_to is not None or document_ids is not None:
            normalized = replace(
                normalized,
                year_from=year_from if year_from is not None else normalized.year_from,
                year_to=year_to if year_to is not None else normalized.year_to,
                document_ids=tuple(document_ids) if document_ids is not None else normalized.document_ids,
            )
        limit = self._limit(normalized, top_k, "max_papers")
        response = self._retrieve(normalized, limit=limit, paper_filters={**self._filters(normalized), **dict(filters or {})})
        raw = response.get("candidate_papers")
        if not isinstance(raw, list):
            paper_stage = response.get("paper_retrieval")
            raw = paper_stage.get("results") if isinstance(paper_stage, Mapping) else []
        papers: list[PaperCandidate] = []
        allowed_documents = set(normalized.document_ids)
        scoped_documents = set(
            self.workspaces.active_document_ids(normalized.workspace_id, normalized.scope_version)
        )
        paper_documents = {
            document.paper_id: document.document_id for document in self.corpus.list_documents()
        }
        for rank, value in enumerate(raw if isinstance(raw, list) else [], 1):
            if not isinstance(value, Mapping) or not value.get("paper_id"):
                continue
            document_id = str(value.get("document_id") or "") or None
            if allowed_documents and document_id not in allowed_documents:
                continue
            scoped_document_id = document_id or paper_documents.get(str(value["paper_id"]))
            if scoped_document_id and scoped_document_id not in scoped_documents:
                continue
            papers.append(
                PaperCandidate(
                    paper_id=str(value["paper_id"]),
                    title=str(value.get("title") or ""),
                    authors=_strings(value.get("authors")),
                    year=int(value["year"]) if isinstance(value.get("year"), int) else None,
                    abstract=str(value.get("abstract")) if value.get("abstract") is not None else None,
                    score=float(value.get("score") or 0.0),
                    rank=int(value.get("rank") or rank),
                    work_id=str(value.get("work_id") or "") or None,
                    document_id=document_id,
                    doi=str(value.get("doi") or "") or None,
                    publication_date=_safe_date(value.get("publication_date"))
                    or (_safe_date(str(value.get("year"))) if value.get("year") is not None else None),
                    metadata={
                        "retrieval_source": value.get("retrieval_source"),
                        "keywords": tuple(_strings(value.get("keywords"))),
                    },
                )
            )
            if len(papers) >= limit:
                break
        return PaperSearchResult(
            normalized.request_id,
            normalized.query,
            tuple(papers),
            {"retriever": response.get("retriever"), "result_count": len(papers)},
        )

    def _section_info(self, section_id: str | None) -> tuple[str, tuple[str, ...], str | None]:
        if not section_id:
            return "", (), None
        value = self.corpus.sections.get(section_id)
        if not value:
            return "", (), None
        path = _strings(value.get("section_path"))
        title = str(value.get("title") or (path[-1] if path else ""))
        return title, path, str(value.get("content_zone") or "") or None

    @staticmethod
    def _section_matches(requested: Sequence[str], title: str, path: Sequence[str]) -> bool:
        if not requested:
            return True
        values = {_normalized(title), *(_normalized(item) for item in path)}
        return any(
            _normalized(item) in value or value in _normalized(item)
            for item in requested
            for value in values
            if value
        )

    def search_sections(
        self,
        request: ResearchRequest | str,
        *,
        paper_ids: Sequence[str] | None = None,
        section_types: Sequence[str] | None = None,
        top_k: int | None = None,
    ) -> SectionSearchResult:
        normalized = self._request(
            request,
            top_k=top_k,
            paper_ids=paper_ids,
            preferred_section_types=section_types,
        )
        limit = self._limit(normalized, top_k, "max_sections")
        response = self._retrieve(
            normalized,
            limit=min(max(limit * 5, limit), 100),
            paper_filters=self._filters(normalized),
        )
        raw = response.get("results")
        sections: list[SectionCandidate] = []
        seen: set[tuple[str, str, str]] = set()
        requested_types = tuple(section_types or normalized.preferred_section_types)
        for rank, value in enumerate(raw if isinstance(raw, list) else [], 1):
            if not isinstance(value, Mapping):
                continue
            paper_id = str(value.get("paper_id") or "")
            section_id = str(value.get("section_id") or "")
            chunk_id = str(value.get("chunk_id") or "")
            if not paper_id or not section_id or not chunk_id:
                continue
            if normalized.paper_ids and paper_id not in set(normalized.paper_ids):
                continue
            title, path, content_zone = self._section_info(section_id)
            if not self._section_matches(requested_types, title, path):
                continue
            key = (paper_id, section_id, chunk_id)
            if key in seen:
                continue
            seen.add(key)
            sections.append(
                SectionCandidate(
                    paper_id=paper_id,
                    section_id=section_id,
                    chunk_ids=(chunk_id,),
                    title=title,
                    normalized_type=title or (path[-1] if path else None),
                    preview=str(value.get("content") or "")[:500],
                    score=float(value.get("score") or 0.0),
                    rank=int(value.get("rank") or rank),
                    document_id=str(value.get("document_id") or "") or None,
                    metadata={
                        "section_path": path,
                        "content_zone": content_zone,
                        "block_ids": _strings(value.get("block_ids")),
                    },
                )
            )
            if len(sections) >= limit:
                break
        return SectionSearchResult(
            normalized.request_id,
            normalized.query,
            tuple(sections),
            {"retriever": response.get("retriever"), "result_count": len(sections)},
        )

    @staticmethod
    def _candidate_ref(ref: EvidenceRef | Mapping[str, Any]) -> EvidenceRef:
        if isinstance(ref, EvidenceRef):
            return ref
        if isinstance(ref, Mapping):
            allowed = {key: ref.get(key) for key in ("paper_id", "document_id", "section_id", "chunk_id", "block_id")}
            return EvidenceRef(**allowed)
        raise InvalidEvidenceLocator("EvidenceRef 类型不受支持。")

    def _document_for_locator(self, locator: CanonicalLocator) -> tuple[str, str | None]:
        document = self.corpus.require_document(locator.document_id)
        return document.paper_id, document.work_id

    def _validate_locator(self, ref: EvidenceRef, locator: CanonicalLocator) -> tuple[str, str | None]:
        paper_id, work_id = self._document_for_locator(locator)
        if locator.document_id not in self.workspaces.active_document_ids(self.workspace_id, self.scope_version):
            raise InvalidEvidenceLocator("EvidenceRef 指向当前 Workspace Scope 外的文档。")
        if ref.document_id and ref.document_id != locator.document_id:
            raise InvalidEvidenceLocator("EvidenceRef.document_id 与 chunk 不一致。")
        if ref.paper_id and ref.paper_id != paper_id:
            raise InvalidEvidenceLocator("EvidenceRef.paper_id 与 chunk 不一致。")
        if ref.section_id and ref.section_id != locator.section_id:
            raise InvalidEvidenceLocator("EvidenceRef.section_id 与 chunk 不一致。")
        if ref.block_id and ref.block_id not in locator.block_ids:
            raise InvalidEvidenceLocator("EvidenceRef.block_id 不属于该 chunk。")
        return paper_id, work_id

    def read_evidence(self, evidence_refs: Sequence[EvidenceRef | Mapping[str, Any]]) -> tuple[EvidenceSource, ...]:
        output: list[EvidenceSource] = []
        seen: set[str] = set()
        for raw_ref in evidence_refs:
            ref = self._candidate_ref(raw_ref)
            locators: list[CanonicalLocator] = []
            if ref.chunk_id:
                locator = self.corpus.locate_chunk(ref.chunk_id)
                if locator is None:
                    raise SourceNotFound(f"Canonical Chunk 不存在：{ref.chunk_id}")
                locators.append(locator)
            elif ref.section_id:
                for document in self.corpus.list_documents():
                    if ref.document_id and document.document_id != ref.document_id:
                        continue
                    if ref.paper_id and document.paper_id != ref.paper_id:
                        continue
                    locators.extend(
                        value
                        for value in self.corpus.locate_document_chunks(document.document_id)
                        if value.section_id == ref.section_id
                    )
                if not locators:
                    raise SourceNotFound(f"Canonical Section 不存在或没有 chunk：{ref.section_id}")
            else:
                raise InvalidEvidenceLocator("read_evidence 需要 chunk_id 或 section_id。")
            for locator in locators:
                paper_id, work_id = self._validate_locator(ref, locator)
                if locator.chunk_id in seen:
                    continue
                seen.add(locator.chunk_id)
                output.append(
                    EvidenceSource(
                        paper_id=paper_id,
                        document_id=locator.document_id,
                        section_id=locator.section_id,
                        chunk_id=locator.chunk_id,
                        block_ids=locator.block_ids,
                        text=locator.content,
                        page_start=locator.page_start or None,
                        page_end=locator.page_end or None,
                        work_id=work_id or locator.work_id,
                        metadata=dict(locator.metadata),
                    )
                )
        return tuple(output)

    def _validated_bundle(self, request: ResearchRequest, candidates: Sequence[EvidenceCandidate]) -> Any:
        evidence_request = self._evidence_request(request, request.budget.max_evidence_items)
        for candidate in candidates:
            if candidate.request_id != request.request_id:
                raise EvidenceValidationError("EvidenceCandidate.request_id 不属于当前 ResearchRequest。")
            if candidate.source_type == "WEB_LITERATURE":
                if not candidate.canonical_id or not candidate.source_locator or candidate.locator_type not in {"ABSTRACT", "FULLTEXT_SPAN"}:
                    raise EvidenceValidationError("External EvidenceCandidate 缺少真实 source locator。")
                continue
            if not candidate.document_id or not candidate.chunk_id:
                raise EvidenceValidationError(f"EvidenceCandidate 缺少 canonical locator：{candidate.candidate_id}")
            locator = self.corpus.locate_chunk(candidate.chunk_id)
            if locator is None:
                raise EvidenceValidationError(f"EvidenceCandidate chunk 不存在：{candidate.chunk_id}")
            if locator.document_id != candidate.document_id:
                raise EvidenceValidationError("EvidenceCandidate document/chunk 归属不一致。")
            if candidate.section_id and candidate.section_id != locator.section_id:
                raise EvidenceValidationError("EvidenceCandidate section/chunk 归属不一致。")
            document = self.corpus.require_document(locator.document_id)
            candidate_paper = candidate.metadata.get("paper_id")
            if candidate_paper and str(candidate_paper) != document.paper_id:
                raise EvidenceValidationError("EvidenceCandidate paper/chunk 归属不一致。")
            if candidate.work_id and candidate.work_id != document.work_id:
                raise EvidenceValidationError("EvidenceCandidate work/chunk 归属不一致。")
            if candidate.block_ids and not set(candidate.block_ids).issubset(set(locator.block_ids)):
                raise EvidenceValidationError("EvidenceCandidate block/chunk 归属不一致。")
            if candidate.retrieval_source not in {"lightrag_entity", "lightrag_relation"} and not _same_text(candidate.content, locator.content):
                raise EvidenceValidationError(f"EvidenceCandidate source text 不匹配：{candidate.candidate_id}")
        bundle = self.evidence.verify(evidence_request, candidates)
        if bundle.rejected_candidate_ids:
            raise EvidenceValidationError(f"EvidenceCandidate 未通过 canonical/workspace 校验：{list(bundle.rejected_candidate_ids)}")
        return bundle

    def validate_candidates(self, request: ResearchRequest, candidates: Sequence[EvidenceCandidate]) -> tuple[VerifiedEvidence, ...]:
        return tuple(self._validated_bundle(request, candidates).evidence)

    def _evidence_mapping(self, value: VerifiedEvidence) -> dict[str, Any]:
        # VerifiedEvidence intentionally stores canonical document/chunk
        # identity; section/paper fields are derived here for the upper
        # EvidencePack projection rather than duplicated in the core model.
        metadata = dict(value.metadata)
        if value.source_type == "WEB_LITERATURE":
            return {
                "evidence_id": value.evidence_id,
                "candidate_id": value.candidate_id,
                "request_id": value.request_id,
                "source_type": value.source_type,
                "canonical_id": value.canonical_id,
                "source_locator": value.source_locator,
                "locator_type": value.locator_type,
                "publication_date": value.publication_date.isoformat() if value.publication_date else None,
                "retrieved_at": value.retrieved_at.isoformat() if value.retrieved_at else None,
                "provider": value.provider,
                "validation_status": value.validation_status,
                "work_id": value.work_id or None,
                "paper_id": metadata.get("paper_id") or metadata.get("paperId"),
                "title": metadata.get("title"),
                "authors": metadata.get("authors", ()),
                "venue": metadata.get("venue"),
                "text": value.content,
                "content": value.content,
                "content_hash": value.content_hash,
                "verification_method": value.verification_method,
                "evidence_grade": value.evidence_grade,
                "directness": value.directness,
                "metadata": metadata,
            }
        document = self.corpus.require_document(value.document_id)
        locator = self.corpus.locate_chunk(value.chunk_id)
        return {
            "evidence_id": value.evidence_id,
            "candidate_id": value.candidate_id,
            "request_id": value.request_id,
            "paper_id": metadata.get("paper_id") or document.paper_id,
            "work_id": value.work_id,
            "document_id": value.document_id,
            "section_id": metadata.get("section_id") or (locator.section_id if locator else None),
            "chunk_id": value.chunk_id,
            "block_ids": list(value.block_ids),
            "page_start": value.page_start,
            "page_end": value.page_end,
            "text": value.content,
            "content": value.content,
            "content_hash": value.content_hash,
            "verification_method": value.verification_method,
            "evidence_grade": value.evidence_grade,
            "directness": value.directness,
            "source_type": value.source_type,
            "canonical_id": value.canonical_id,
            "source_locator": value.source_locator,
            "locator_type": value.locator_type,
            "publication_date": value.publication_date.isoformat() if value.publication_date else None,
            "retrieved_at": value.retrieved_at.isoformat() if value.retrieved_at else None,
            "provider": value.provider,
            "validation_status": value.validation_status,
            "metadata": metadata,
        }

    def build_evidence_pack(
        self,
        request: ResearchRequest,
        candidates: Sequence[EvidenceCandidate],
        *,
        web_used: bool = False,
        freshness: FreshnessDecision | None = None,
    ) -> EvidencePack:
        evidence_request = self._evidence_request(request, request.budget.max_evidence_items)
        bundle = self._validated_bundle(request, candidates)
        verified = tuple(bundle.evidence)
        selected = self.evidence.select(evidence_request, bundle)
        selected_by_id = {item.evidence.evidence_id for item in selected}
        final_evidence = tuple(value for value in verified if value.evidence_id in selected_by_id)
        evidence_maps = tuple(self._evidence_mapping(value) for value in final_evidence)
        unresolved: list[str] = []
        claims: list[dict[str, Any]] = []
        evidence_tokens = {
            value.evidence_id: set(re.findall(r"[\w\u3400-\u4dbf\u4e00-\u9fff]+", value.content.casefold()))
            for value in final_evidence
        }
        counter_maps: dict[str, Mapping[str, Any]] = {}
        for value in final_evidence:
            role = str(value.metadata.get("evidence_role") or value.metadata.get("support_type") or "support")
            if role in {"counter", "contradict", "contradicted"}:
                counter_maps[value.evidence_id] = self._evidence_mapping(value)
        for index, claim in enumerate(request.target_claims, 1):
            claim_tokens = set(re.findall(r"[\w\u3400-\u4dbf\u4e00-\u9fff]+", claim.casefold()))
            supporting = tuple(
                evidence_id
                for evidence_id, tokens in evidence_tokens.items()
                if evidence_id not in counter_maps and claim_tokens and claim_tokens & tokens
            )
            claim_id = f"{request.request_id}:C{index}"
            status = "supported" if supporting else "unresolved"
            if not supporting:
                unresolved.append(claim)
            claims.append({
                "claim_id": claim_id,
                "text": claim,
                "evidence_ids": supporting,
                "status": status,
                "support_type": "support" if supporting else None,
            })
        coverage = (len(claims) - len(unresolved)) / len(claims) if claims else None
        local_evidence = tuple(value for value in final_evidence if value.source_type == "LOCAL_CORPUS")
        local_ids = {value.evidence_id for value in local_evidence}
        local_supported = sum(
            bool(set(claim.get("evidence_ids", ())) & local_ids)
            for claim in claims
        )
        local_coverage = local_supported / len(claims) if claims else 0.0
        metadata = {
            "workspace_id": request.workspace_id,
            "scope_version": request.scope_version,
            "evidence_count": len(evidence_maps),
            "local_only": not web_used,
            "source_types": sorted({value.source_type for value in final_evidence}),
            "conflicts": tuple(self.evidence.last_diagnostics.get("conflicts", ())),
        }
        if freshness is not None:
            metadata["freshness_decision"] = freshness.to_dict()
        return EvidencePack(
            request_id=request.request_id,
            query=request.query,
            claims=tuple(claims),
            evidence=evidence_maps,
            unresolved=tuple(unresolved),
            local_coverage=local_coverage,
            coverage=coverage,
            counter_evidence=tuple(counter_maps.values()),
            web_used=web_used,
            metadata=metadata,
        )

    def _local_candidates(self, request: ResearchRequest) -> tuple[EvidenceCandidate, ...]:
        """小型组合入口；三个 capability 仍可独立调用和测试。"""
        papers = self.search_papers(request, top_k=request.budget.max_papers)
        paper_ids = request.paper_ids or tuple(value.paper_id for value in papers.papers)
        sections = self.search_sections(request, paper_ids=paper_ids, top_k=request.budget.max_sections)
        refs = tuple(
            EvidenceRef(
                paper_id=value.paper_id,
                document_id=value.document_id,
                section_id=value.section_id,
                chunk_id=value.chunk_ids[0] if value.chunk_ids else None,
            )
            for value in sections.sections
        )
        sources = self.read_evidence(refs)
        source_by_chunk = {value.chunk_id: value for value in sources}
        candidates: list[CandidateEvidence] = []
        for value in sections.sections:
            if not value.chunk_ids:
                continue
            source = source_by_chunk.get(value.chunk_ids[0])
            if source is None:
                continue
            document = self.corpus.require_document(source.document_id)
            document_metadata = dict(document.metadata)
            canonical_metadata = {
                **document_metadata,
                "title": document_metadata.get("title") or document.paper_id,
                "work_id": document.work_id,
            }
            candidates.append(
                CandidateEvidence(
                    candidate_id=f"{request.request_id}:{source.chunk_id}",
                    request_id=request.request_id,
                    workspace_id=request.workspace_id,
                    scope_version=request.scope_version,
                    content=source.text,
                    score=value.score,
                    retrieval_source="research_capability_local",
                    work_id=source.work_id,
                    document_id=source.document_id,
                    chunk_id=source.chunk_id,
                    section_id=source.section_id,
                    page_start=source.page_start,
                    page_end=source.page_end,
                    block_ids=source.block_ids,
                    canonical_id=canonical_identity(canonical_metadata),
                    publication_date=_safe_date(document_metadata.get("publication_date"))
                    or _safe_date(document_metadata.get("year"))
                    or _safe_date(source.metadata.get("publication_date"))
                    or _safe_date(source.metadata.get("year")),
                    metadata={
                        "paper_id": source.paper_id,
                        "section_id": source.section_id,
                        "source": "canonical_corpus",
                        **document_metadata,
                    },
                )
            )
        return tuple(candidates)

    @staticmethod
    def _pack_latest_date(pack: EvidencePack) -> date | None:
        dates: list[date] = []
        for value in pack.evidence:
            raw = value.get("publication_date")
            if isinstance(raw, str):
                try:
                    dates.append(date.fromisoformat(raw[:10]))
                except ValueError:
                    continue
        return max(dates) if dates else None

    def research(
        self,
        request: ResearchRequest,
        *,
        allow_web: bool = False,
        harness: Any | None = None,
    ) -> EvidencePack:
        def step(name: str, details: dict[str, Any] | None = None):
            return harness.step(name, details=details) if harness is not None else _NullStep(details)

        with step("LOCAL_RESEARCH") as trace:
            local_candidates = self._local_candidates(request)
            trace["candidate_count"] = len(local_candidates)
        local_pack = self.build_evidence_pack(request, local_candidates)
        local_time_satisfied: bool | None = None
        if request.requested_from or request.requested_to or request.year_from or request.year_to:
            local_dates = [
                parsed
                for value in local_pack.evidence
                if (parsed := _safe_date(value.get("publication_date"))) is not None
            ]
            requested_from, requested_to = FreshnessPolicy.requested_range(request)
            local_time_satisfied = bool(local_dates) and all(
                (requested_from is None or value >= requested_from)
                and (requested_to is None or value <= requested_to)
                for value in local_dates
            )
        with step("EVIDENCE_COVERAGE") as trace:
            trace.update({"coverage": local_pack.coverage, "unresolved": len(local_pack.unresolved)})
        decision = FreshnessPolicy.decide(
            request,
            local_latest_date=self._pack_latest_date(local_pack),
            local_coverage=local_pack.coverage,
            unresolved=local_pack.unresolved,
            conflicts=tuple(local_pack.metadata.get("conflicts", ())),
            local_time_satisfied=local_time_satisfied,
        )
        with step("FRESHNESS_DECIDE") as trace:
            trace.update(decision.to_dict())
        if not decision.web_required:
            return replace(local_pack, metadata={**local_pack.metadata, "freshness_decision": decision.to_dict()})
        if not allow_web:
            metadata = {**local_pack.metadata, "freshness_decision": decision.to_dict(), "error_code": "CAPABILITY_DENIED"}
            if decision.mode == "FRESH_REQUIRED":
                metadata["error_code"] = "FRESHNESS_UNAVAILABLE"
            return replace(local_pack, unresolved=tuple(dict.fromkeys((*local_pack.unresolved, *request.target_claims))), metadata=metadata)
        if request.budget.max_web_queries < 1:
            metadata = {**local_pack.metadata, "freshness_decision": decision.to_dict(), "error_code": "WEB_BUDGET_EXHAUSTED"}
            return replace(local_pack, unresolved=tuple(dict.fromkeys((*local_pack.unresolved, *request.target_claims))), metadata=metadata)
        if self.web is None:
            metadata = {**local_pack.metadata, "freshness_decision": decision.to_dict(), "error_code": "WEB_UNAVAILABLE"}
            return replace(local_pack, unresolved=tuple(dict.fromkeys((*local_pack.unresolved, *request.target_claims))), metadata=metadata)

        literature_request = LiteratureSearchRequest(
            request_id=request.request_id,
            query=request.query,
            max_results=request.budget.max_papers,
            requested_from=_safe_date(request.requested_from) or (date(request.year_from, 1, 1) if request.year_from else None),
            requested_to=_safe_date(request.requested_to) or (date(request.year_to, 12, 31) if request.year_to else None),
            metadata={"workspace_id": request.workspace_id, "scope_version": request.scope_version, "source_policy": decision.mode},
        )
        with step("WEB_DISCOVERY") as trace:
            try:
                result = self.web.search(literature_request, harness=harness)
            except WebLiteratureError as error:
                failure_code = getattr(error, "code", "WEB_PROVIDER_UNAVAILABLE")
                # Preserve the stable pack-level failure contract while
                # retaining the more specific provider classification in the
                # structured failure payload.
                error_code = "FRESHNESS_UNAVAILABLE" if decision.mode == "FRESH_REQUIRED" else "WEB_PROVIDER_UNAVAILABLE"
                trace.update({"error_code": error_code, "failure_code": failure_code, "error_type": type(error).__name__})
                metadata = {**local_pack.metadata, "freshness_decision": decision.to_dict(), "error_code": error_code, "web_failure": {"error_code": failure_code, "error_type": type(error).__name__, "message": str(error)}}
                if harness is not None:
                    harness.termination_reason = error_code
                return replace(local_pack, unresolved=tuple(dict.fromkeys((*local_pack.unresolved, *request.target_claims))), metadata=metadata)
            except BudgetExceeded as error:
                trace.update({"error_code": "WEB_BUDGET_EXHAUSTED", "error_type": type(error).__name__})
                metadata = {**local_pack.metadata, "freshness_decision": decision.to_dict(), "error_code": "WEB_BUDGET_EXHAUSTED", "web_failure": {"error_code": "WEB_BUDGET_EXHAUSTED", "error_type": type(error).__name__, "message": str(error)}}
                return replace(local_pack, unresolved=tuple(dict.fromkeys((*local_pack.unresolved, *request.target_claims))), metadata=metadata)
            except Exception as error:
                trace.update({"error_code": "WEB_PROVIDER_UNAVAILABLE", "error_type": type(error).__name__})
                metadata = {**local_pack.metadata, "freshness_decision": decision.to_dict(), "error_code": "WEB_PROVIDER_UNAVAILABLE", "web_failure": {"error_type": type(error).__name__, "message": str(error)}}
                return replace(local_pack, unresolved=tuple(dict.fromkeys((*local_pack.unresolved, *request.target_claims))), metadata=metadata)
            trace.update({"candidate_count": len(result.candidates), "provider_failures": list(result.provider_failures)})
        local_identity: set[str] = set()
        local_work_ids: set[str] = set()
        for document in self.corpus.list_documents():
            metadata = dict(document.metadata)
            metadata.update({"title": metadata.get("title") or document.paper_id, "work_id": document.work_id})
            local_identity.add(canonical_identity(metadata))
            if document.work_id:
                local_work_ids.add(str(document.work_id))
            if metadata.get("doi"):
                doi = normalize_doi(metadata["doi"])
                if doi:
                    local_identity.add(f"doi:{doi}")
            external_ids = metadata.get("external_ids")
            if isinstance(external_ids, Mapping) and external_ids.get("arxiv"):
                arxiv_id = re.sub(r"v\d+$", "", str(external_ids["arxiv"]).casefold())
                local_identity.add(f"arxiv:{arxiv_id}")
        external_result = type(result)(
            result.request_id,
            result.query,
            tuple(
                value for value in result.candidates
                if value.canonical_id not in local_identity
                and str(value.metadata.get("work_id") or "") not in local_work_ids
            ),
            result.provider_failures,
            result.metadata_conflicts,
            result.metadata,
        )
        with step("WEB_DEDUPLICATE") as trace:
            trace.update({"input_count": len(result.candidates), "deduplicated_count": len(external_result.candidates), "local_reused_count": len(result.candidates) - len(external_result.candidates)})
        web_candidates = self.web.evidence_candidates(literature_request, external_result, workspace_id=request.workspace_id, scope_version=request.scope_version)
        combined = (*local_candidates, *web_candidates)
        with step("EVIDENCE_VALIDATE") as trace:
            trace["candidate_count"] = len(combined)
            final = self.build_evidence_pack(request, combined, web_used=True, freshness=decision)
            trace.update({
                "verified_count": len(final.evidence),
                "rejected_count": max(0, len(combined) - len(final.evidence)),
                "pipeline_diagnostics": dict(self.evidence.last_diagnostics),
            })
        final = replace(
            final,
            metadata={
                **final.metadata,
                "provider_failures": list(result.provider_failures),
                "metadata_conflicts": list(result.metadata_conflicts),
            },
        )
        requested_from, requested_to = FreshnessPolicy.requested_range(request)
        freshness_candidates = tuple(
            candidate for candidate in result.candidates
            if candidate.publication_date is not None
            and (requested_from is None or candidate.publication_date >= requested_from)
            and (requested_to is None or candidate.publication_date <= requested_to)
        )
        web_verified = any(value.get("source_type") == "WEB_LITERATURE" for value in final.evidence)
        local_reused = len(result.candidates) > len(external_result.candidates)
        freshness_checked = bool(freshness_candidates) and (web_verified or local_reused)
        if decision.mode == "FRESH_REQUIRED" and not freshness_checked:
            return replace(final, unresolved=tuple(dict.fromkeys((*final.unresolved, *request.target_claims))), metadata={**final.metadata, "error_code": "FRESHNESS_UNAVAILABLE"})
        with step("EVIDENCEPACK_BUILD") as trace:
            trace.update({"verified_count": len(final.evidence), "rejected_count": max(0, len(combined) - len(final.evidence)), "web_used": final.web_used, "freshness_checked": freshness_checked})
        return final


class _NullStep:
    def __init__(self, details: dict[str, Any] | None = None) -> None:
        self.details = details if details is not None else {}

    def __enter__(self) -> dict[str, Any]:
        return self.details

    def __exit__(self, *_: Any) -> None:
        return None


def _safe_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None
