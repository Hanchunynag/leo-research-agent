"""LangChain 检索 Tool：统一执行中英文查询的 legacy RAG 召回。"""

from __future__ import annotations

from pathlib import Path
from difflib import SequenceMatcher
from typing import Any, Mapping

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class BilingualRetrievalInput(BaseModel):
    queries: list[str] = Field(min_length=1, description="已翻译并去重的中英文等价查询")
    workspace_id: str = Field(min_length=1)
    scope_version: int = Field(ge=1)
    top_k: int = Field(default=10, ge=1, le=100)
    target_document_ids: list[str] = Field(
        default_factory=list,
        description="Planner-resolved document IDs; empty means the full workspace scope",
    )


class WorkspaceDocumentsInput(BaseModel):
    workspace_id: str = Field(min_length=1)
    scope_version: int = Field(ge=1)


class DocumentOutlineInput(BaseModel):
    document_id: str = Field(min_length=1)


class DocumentReadInput(BaseModel):
    document_id: str = Field(min_length=1)
    section_id: str | None = None
    chunk_id: str | None = None
    neighbor_count: int = Field(default=1, ge=0, le=5)


class TranslateInput(BaseModel):
    text: str = Field(min_length=1)
    target_language: str = Field(pattern="^(zh|en)$")


class PublicationDateInput(BaseModel):
    title: str = Field(min_length=1, description="论文完整标题")


def _clean_queries(values: list[str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(value.strip() for value in values if isinstance(value, str) and value.strip())
    )


class BilingualRetrievalTool:
    """只向下游暴露 Selected Evidence 的检索 Tool。"""

    def __init__(self, knowledge: Any) -> None:
        self.knowledge = knowledge

    def retrieve(
        self,
        queries: list[str],
        workspace_id: str,
        scope_version: int,
        top_k: int = 10,
        target_document_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        variants = _clean_queries(queries)
        if not variants:
            raise ValueError("中英文 query 不能为空。")
        expected_workspace = str(getattr(self.knowledge, "workspace_id", workspace_id))
        if workspace_id != expected_workspace:
            raise PermissionError("Unified Knowledge Service workspace 不匹配。")
        result = self.knowledge.retrieve_multi(
            variants,
            limit=top_k,
            workspace_id=workspace_id,
            scope_version=scope_version,
        )
        raw = result.get("results") if isinstance(result, Mapping) else None
        allowed_documents = {
            value.strip()
            for value in (target_document_ids or [])
            if isinstance(value, str) and value.strip()
        }
        seen: set[str] = set()
        selected: list[dict[str, Any]] = []
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, Mapping):
                continue
            if allowed_documents and str(item.get("document_id") or "") not in allowed_documents:
                continue
            if (item.get("evidence_state") or item.get("state")) != "selected":
                continue
            key = str(item.get("evidence_id") or item.get("chunk_id") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            selected.append(dict(item))
            if len(selected) == top_k:
                break
        numbered = [
            {**item, "source_id": str(item.get("source_id") or f"S{index}")}
            for index, item in enumerate(selected, 1)
        ]
        diagnostics = dict(getattr(self.knowledge, "last_diagnostics", {}))
        intelligence = diagnostics.get("evidence_intelligence")
        if isinstance(intelligence, Mapping) and isinstance(intelligence.get("conflicts"), list):
            # 证据候选阶段的冲突不能泄漏到回答阶段；只有冲突双方都实际
            # Selected 的情况才需要回答模型承认并交由 Claim 校验处理。
            selected_ids = {str(value["evidence_id"]) for value in numbered}
            diagnostics["conflicts"] = [
                {
                    "supporting_evidence_id": str(pair[0]),
                    "opposing_evidence_id": str(pair[1]),
                }
                for pair in intelligence["conflicts"]
                if isinstance(pair, (list, tuple))
                and len(pair) == 2
                and str(pair[0]) in selected_ids
                and str(pair[1]) in selected_ids
            ]
        diagnostics["bilingual_retrieval"] = {
            "queries": list(variants),
            "query_count": len(variants),
            "selected_count": len(numbered),
        }
        return {"results": numbered, "diagnostics": diagnostics}


def build_bilingual_retrieval_tool(
    retriever: BilingualRetrievalTool,
) -> StructuredTool:
    return StructuredTool.from_function(
        func=retriever.retrieve,
        name="retrieve_bilingual",
        description=(
            "Retrieve and deduplicate evidence for the supplied Chinese and English scientific "
            "query variants. The tool returns Selected Evidence only."
        ),
        args_schema=BilingualRetrievalInput,
    )


def build_bilingual_gateway_handler(tool: StructuredTool):
    """将已有受权限/预算控制的 Gateway 接到 LangChain 检索 Tool。"""

    def retrieve(arguments: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        workspace_id = str(arguments["workspace_id"])
        scope_version = int(arguments["scope_version"])
        if workspace_id != str(context.get("workspace_id")) or scope_version != int(
            context.get("scope_version") or 0
        ):
            raise PermissionError("Tool 参数与 Run scope 不一致。")
        variants = arguments.get("query_variants")
        queries = (
            [value for value in variants if isinstance(value, str)]
            if isinstance(variants, list)
            else [str(arguments["query"])]
        )
        output = tool.invoke(
            {
                "queries": queries,
                "workspace_id": workspace_id,
                "scope_version": scope_version,
                "top_k": int(arguments.get("top_k") or 10),
                **(
                    {"target_document_ids": list(arguments["target_document_ids"])}
                    if isinstance(arguments.get("target_document_ids"), list)
                    else {}
                ),
            }
        )
        if not isinstance(output, Mapping):
            raise TypeError("LangChain retrieve_bilingual Tool 输出必须是对象。")
        return dict(output)

    return retrieve


def build_research_tool_handlers(
    project_root: Path,
    workspaces: Any,
    *,
    translation_tool: StructuredTool | None = None,
) -> dict[str, Any]:
    """Build the small, atomic document/workspace tools exposed to the graph.

    These handlers use the canonical repositories already owned by the project.  They
    never expose BM25, Dense, Qdrant, Neo4j, or LightRAG objects to an Agent.
    """

    from app.corpus import CanonicalCorpusService

    corpus = getattr(workspaces, "corpus", None) or CanonicalCorpusService(project_root)

    def list_documents(arguments: Mapping[str, Any], _context: Mapping[str, Any]) -> Mapping[str, Any]:
        workspace_id = str(arguments["workspace_id"])
        version = int(arguments["scope_version"])
        scope = workspaces.require_scope(workspace_id, version)
        active = set(scope.included_document_ids) - set(scope.excluded_document_ids)
        documents = []
        for document in corpus.list_documents():
            if document.document_id not in active:
                continue
            metadata = dict(document.metadata)
            documents.append(
                {
                    "document_id": document.document_id,
                    "title": str(metadata.get("title") or document.paper_id or document.document_id),
                    "year": metadata.get("year"),
                    "authors": list(metadata.get("authors") or []),
                    "paper_id": document.paper_id,
                }
            )
        return {"documents": documents}

    def check_document(document_id: str, arguments: Mapping[str, Any]) -> Any:
        document = corpus.require_document(document_id)
        workspace_id = arguments.get("workspace_id")
        version = arguments.get("scope_version")
        if workspace_id is not None and version is not None:
            active = workspaces.active_document_ids(str(workspace_id), int(version))
            if document_id not in active:
                raise PermissionError(f"文档不属于当前 Scope：{document_id}")
        return document

    def get_outline(arguments: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        document_id = str(arguments["document_id"])
        check_document(document_id, context)
        sections = corpus.sections.list_for_document(document_id)
        return {
            "document_id": document_id,
            "sections": [
                {
                    "section_id": str(value.get("section_id") or ""),
                    "title": str(value.get("title") or value.get("name") or ""),
                    "level": value.get("level"),
                    "parent_section_id": value.get("parent_section_id"),
                }
                for value in sections
            ],
        }

    def read_document(arguments: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        document_id = str(arguments["document_id"])
        check_document(document_id, context)
        section_id = arguments.get("section_id")
        chunk_id = arguments.get("chunk_id")
        values = list(corpus.chunks.list_for_document(document_id))
        if chunk_id:
            index = next((i for i, value in enumerate(values) if value.get("chunk_id") == chunk_id), None)
            if index is None:
                raise KeyError(f"Chunk 不存在：{chunk_id}")
            radius = int(arguments.get("neighbor_count") or 0)
            values = values[max(0, index - radius): index + radius + 1]
        elif section_id:
            values = [value for value in values if value.get("section_id") == section_id]
        return {
            "document_id": document_id,
            "chunks": [dict(value) for value in values],
            "count": len(values),
        }

    def resolve_local_publication_date(
        arguments: Mapping[str, Any], context: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Use canonical metadata when the external resolver is unavailable."""

        title = str(arguments["title"]).strip()
        active = workspaces.active_document_ids(
            str(context.get("workspace_id") or "default"),
            int(context.get("scope_version") or 1),
        )
        normalized = title.casefold()
        candidates: list[tuple[float, Any]] = []
        for document in corpus.list_documents():
            if document.document_id not in active:
                continue
            candidate_title = str(document.metadata.get("title") or "").strip()
            if not candidate_title:
                continue
            score = SequenceMatcher(None, normalized, candidate_title.casefold()).ratio()
            candidates.append((score, document))
        candidates.sort(key=lambda value: value[0], reverse=True)
        best_score, best = candidates[0] if candidates else (0.0, None)
        if best is None or best_score < 0.75:
            return {
                "query_title": title,
                "matched_title": "",
                "publication_year": None,
                "match_score": round(best_score, 4),
                "source": "canonical_metadata",
                "status": "not_found",
            }
        return {
            "query_title": title,
            "matched_title": str(best.metadata.get("title") or ""),
            "publication_year": best.metadata.get("year"),
            "doi": best.metadata.get("doi"),
            "match_score": round(best_score, 4),
            "source": "canonical_metadata",
            "status": "resolved",
        }

    def translate(arguments: Mapping[str, Any], _context: Mapping[str, Any]) -> Mapping[str, Any]:
        text = str(arguments["text"]).strip()
        target = str(arguments["target_language"])
        if translation_tool is not None:
            translated = translation_tool.invoke({"query": text})
            if isinstance(translated, Mapping):
                key = "zh_query" if target == "zh" else "en_query"
                return {
                    "original": text,
                    "translated": str(translated.get(key) or text),
                    "target_language": target,
                }
        return {"original": text, "translated": text, "target_language": target, "status": "identity_fallback"}

    return {
        "workspace.list_documents": list_documents,
        "document.get_outline": get_outline,
        "document.read": read_document,
        "literature.resolve_publication_date": resolve_local_publication_date,
        "language.translate": translate,
    }


def build_research_tools(
    project_root: Path,
    workspaces: Any,
    *,
    translation_tool: StructuredTool | None = None,
) -> tuple[StructuredTool, ...]:
    """Return the four atomic LangChain Tools for direct Agent/tool-loop use."""

    handlers = build_research_tool_handlers(
        project_root, workspaces, translation_tool=translation_tool
    )

    def invoke(name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        result = handlers[name](arguments, arguments)
        return dict(result)

    return (
        StructuredTool.from_function(
            func=lambda workspace_id, scope_version: invoke(
                "workspace.list_documents", {"workspace_id": workspace_id, "scope_version": scope_version}
            ),
            name="workspace.list_documents",
            description="List documents in the current workspace scope.",
            args_schema=WorkspaceDocumentsInput,
        ),
        StructuredTool.from_function(
            func=lambda document_id: invoke("document.get_outline", {"document_id": document_id}),
            name="document.get_outline",
            description="Read the section outline of a canonical paper.",
            args_schema=DocumentOutlineInput,
        ),
        StructuredTool.from_function(
            func=lambda document_id, section_id=None, chunk_id=None, neighbor_count=1: invoke(
                "document.read",
                {"document_id": document_id, "section_id": section_id, "chunk_id": chunk_id, "neighbor_count": neighbor_count},
            ),
            name="document.read",
            description="Read canonical chunks from a paper or section.",
            args_schema=DocumentReadInput,
        ),
        StructuredTool.from_function(
            func=lambda title: invoke(
                "literature.resolve_publication_date", {"title": title}
            ),
            name="literature.resolve_publication_date",
            description="Resolve a paper title to its publication year for timeline ordering.",
            args_schema=PublicationDateInput,
        ),
        StructuredTool.from_function(
            func=lambda text, target_language: invoke(
                "language.translate", {"text": text, "target_language": target_language}
            ),
            name="language.translate",
            description="Translate text to Chinese or English.",
            args_schema=TranslateInput,
        ),
    )
