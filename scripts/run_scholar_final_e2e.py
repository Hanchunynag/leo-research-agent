"""Run the reproducible Phase 4 ScholarHarness release evaluation.

The script is deliberately a client of the existing Production Composition
Root.  It does not provide alternate routing, research, writing, or approval
implementations.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

# Make direct ``python scripts/run_scholar_final_e2e.py`` invocation behave
# like ``python -m`` from the repository root.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

# Direct script execution intentionally adjusts ``sys.path`` before importing
# project modules.  Keep the import-order exception local to this release
# client; production modules remain Ruff-clean.
# ruff: noqa: E402
from app.agentic.config import AgenticRAGConfig
from app.embeddings.bge_m3 import BGEM3Config, BGEM3EmbeddingProvider
from app.generation.settings import load_local_llm_settings
from app.indexing.bm25 import chunks_digest
from app.indexing.dense import load_dense_manifest
from app.indexing.hierarchical import build_hierarchical_indexes
from app.indexing.paper import load_paper_records, papers_digest
from app.indexing.paper_dense import load_paper_dense_manifest
from app.scholar.approval import LatexBridgeService, PatchApprovalRequest, PatchApprovalService
from app.scholar.composition import ScholarRuntimeFactory
from app.orchestration.evaluation import (
    OrchestrationEvaluationCase,
    default_orchestration_cases,
    evaluate_backend,
)
from app.scholar.models import Contribution, ManuscriptFact
from app.retrieval.search import load_chunks
from app.web.runtime import WebRuntimeConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--env-root",
        type=Path,
        default=Path.cwd(),
        help="Directory containing the existing .env; credentials are never printed.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional existing repository data directory used by the copied demo project.",
    )
    parser.add_argument(
        "--skills-root",
        type=Path,
        default=None,
        help="Optional existing skills directory used by the copied demo project.",
    )
    parser.add_argument(
        "--accept-introduction",
        action="store_true",
        help="Apply the generated Introduction DraftPatch through Human Approval.",
    )
    parser.add_argument(
        "--fixture-provider",
        action="store_true",
        help="Use a deterministic local-compatible model provider; keep all Scholar domain services real.",
    )
    parser.add_argument(
        "--orchestration-backend",
        choices=("legacy", "crewai"),
        default=None,
        help="Top-level orchestration backend; defaults to ORCHESTRATION_BACKEND or crewai in production.",
    )
    return parser


class _DeterministicFixtureProvider:
    """Local-compatible model boundary for the offline Production E2E.

    Only model responses are deterministic. Research, writing services,
    review, persistence, approval, and artifact bridge remain the production
    composition supplied by ``ScholarRuntimeFactory``.
    """

    model_name = "fixture/local-release"

    def __init__(self) -> None:
        self._runs: dict[str, dict[str, bool]] = {}
        self._legacy_research_calls: dict[str, int] = {}
        self._legacy_review_calls: dict[str, int] = {}

    @staticmethod
    def _tool_names(tools: Any) -> set[str]:
        names: set[str] = set()
        for item in tools if isinstance(tools, (list, tuple)) else ():
            if not isinstance(item, dict):
                continue
            function = item.get("function")
            if isinstance(function, dict) and function.get("name"):
                names.add(str(function["name"]))
            elif item.get("name"):
                names.add(str(item["name"]))
        return names

    @staticmethod
    def _tool_response(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"fixture-{name}",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(
                                        arguments,
                                        ensure_ascii=False,
                                        separators=(",", ":"),
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    @staticmethod
    def _text_response(content: str) -> dict[str, Any]:
        return {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    @staticmethod
    def _introduction_response(messages: list[dict[str, Any]]) -> str:
        payload: dict[str, Any] = {}
        if messages and isinstance(messages[-1].get("content"), str):
            try:
                value = json.loads(messages[-1]["content"])
                if isinstance(value, dict):
                    payload = value
            except json.JSONDecodeError:
                pass
        current = str(payload.get("current_introduction") or "").rstrip()
        statement = (
            "The study evaluates explicit ephemeris-error compensation before "
            "receiver-state estimation."
        )
        content = current if statement in current else f"{current}\n\n{statement}"
        return json.dumps(
            {
                "content": content.strip(),
                "claim_ids": ["contribution:CONTRIB_DEMO_EPHEMERIS"],
                "evidence_ids": [],
                "citation_keys": [],
                "citation_binding_ids": [],
                "contribution_ids": ["CONTRIB_DEMO_EPHEMERIS"],
                "change_summary": "Add the confirmed contribution to the introduction.",
                "warnings": [],
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _synthesis_response(messages: list[dict[str, Any]], section: str) -> str:
        payload: dict[str, Any] = {}
        if messages and isinstance(messages[-1].get("content"), str):
            try:
                value = json.loads(messages[-1]["content"])
                if isinstance(value, dict):
                    payload = value
            except json.JSONDecodeError:
                pass
        current = str(payload.get("current_section") or "").rstrip()
        statement = (
            "The study evaluates explicit ephemeris-error compensation before "
            "receiver-state estimation."
        )
        content = current if statement in current else f"{current}\n\n{statement}"
        return json.dumps(
            {
                "content": content.strip(),
                "claim_ids": [],
                "contribution_ids": ["CONTRIB_DEMO_EPHEMERIS"],
                "change_summary": f"Add the confirmed contribution to the {section}.",
                "warnings": [],
            },
            ensure_ascii=False,
        )

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Any = None,
        **_: Any,
    ) -> dict[str, Any]:
        system = "\n".join(
            str(value.get("content") or "")
            for value in messages
            if isinstance(value, dict) and value.get("role") == "system"
        )
        if "Introduction editor" in system:
            return self._text_response(self._introduction_response(messages))
        if "scientific conclusion editor" in system:
            return self._text_response(self._synthesis_response(messages, "conclusion"))
        if "scientific abstract editor" in system:
            return self._text_response(self._synthesis_response(messages, "abstract"))
        if "Return only JSON" in system and '"issues"' in system:
            return self._text_response('{"issues":[]}')

        # CrewAI sends each specialist's role/contract in the generated
        # prompt.  Keep this provider deterministic for offline CrewAI Flow
        # validation while the actual domain providers remain exercised.
        crew_prompt = "\n".join(
            str(value.get("content") or "")
            for value in messages
            if isinstance(value, dict)
        ).casefold()
        if "scholar supervisor agent" in crew_prompt:
            selected_route = "RESEARCH"
            for field in ('"deterministic_route": "', '"selected_route": "'):
                start = crew_prompt.find(field)
                if start < 0:
                    continue
                value = crew_prompt[start + len(field) :].split('"', 1)[0].upper()
                if value in {
                    "SUPPORT_CLAIM",
                    "WRITE_INTRODUCTION",
                    "WRITE_CONCLUSION",
                    "WRITE_ABSTRACT",
                    "REVIEW",
                    "RESEARCH",
                }:
                    selected_route = value
                    break
            return self._text_response(
                json.dumps(
                    {
                        "status": "COMPLETED",
                        "selected_route": selected_route,
                        "final_answer": "fixture completed",
                        "approval_required": False,
                    },
                    ensure_ascii=False,
                )
            )
        if "research agent" in crew_prompt:
            return self._text_response(
                json.dumps(
                    {
                        "status": "COMPLETED",
                        "research_summary": "fixture research completed",
                        "evidence": [],
                        "citations": [],
                        "unresolved_claims": [],
                        "contradictions": [],
                        "freshness_status": "LOCAL_ONLY",
                    },
                    ensure_ascii=False,
                )
            )
        if "writer agent" in crew_prompt:
            return self._text_response(
                json.dumps(
                    {
                        "status": "READY",
                        "research_required": False,
                        "missing_context": [],
                        "used_evidence": [],
                        "warnings": [],
                    },
                    ensure_ascii=False,
                )
            )
        if "reviewer agent" in crew_prompt:
            return self._text_response(
                json.dumps(
                    {
                        "decision": "PASS",
                        "issues": [],
                        "unsupported_claims": [],
                        "citation_issues": [],
                        "fact_conflicts": [],
                        "revision_instructions": [],
                    },
                    ensure_ascii=False,
                )
            )

        names = self._tool_names(tools)
        if "research_evidence" in names and "Research Subagent" in system:
            # Deep Agents calls the same isolated subagent repeatedly until
            # it returns a final message.  The fixture must therefore model a
            # bounded subagent conversation instead of emitting the same tool
            # call forever.  The tool itself remains the real Legacy
            # ResearchCapability adapter and records each returned pack.
            research_key = "introduction" if "Introduction run" in system else "support"
            research_limit = 3 if research_key == "introduction" else 1
            research_calls = self._legacy_research_calls.get(research_key, 0)
            if research_calls >= research_limit:
                return self._text_response("bounded research completed")
            self._legacy_research_calls[research_key] = research_calls + 1
            return self._tool_response(
                "research_evidence",
                {"query": "LEO ephemeris-error compensation", "purpose": "background"},
            )
        if "review_draft" in names and "Reviewer Subagent" in system:
            review_prompt = "\n".join(
                str(value.get("content") or "")
                for value in messages
                if isinstance(value, dict)
            )
            review_key = str(
                next(
                    (
                        value
                        for value in (
                            "WRITE_INTRODUCTION",
                            "WRITE_CONCLUSION",
                            "WRITE_ABSTRACT",
                        )
                        if value in f"{system}\n{review_prompt}"
                    ),
                    "review",
                )
            )
            if self._legacy_review_calls.get(review_key, 0) >= 1:
                return self._text_response("bounded review completed")
            self._legacy_review_calls[review_key] = 1
            return self._tool_response(
                "review_draft",
                {"task_type": "WRITE_INTRODUCTION", "draft": "fixture draft"},
            )

        selected_task = next(
            (
                task
                for task in (
                    "WRITE_INTRODUCTION",
                    "SUPPORT_CLAIM",
                    "WRITE_CONCLUSION",
                    "WRITE_ABSTRACT",
                )
                if f"selected task is {task}" in system
            ),
        )
        if selected_task is None:
            return self._text_response("fixture completed")
        state = self._runs.setdefault(
            selected_task,
            {"context": False, "research": False, "execute": False, "review": False},
        )
        if "get_project_context" in names and not state["context"]:
            state["context"] = True
            return self._tool_response("get_project_context", {})
        if (
            "task" in names
            and not state["research"]
            and selected_task in {"WRITE_INTRODUCTION", "SUPPORT_CLAIM"}
        ):
            state["research"] = True
            return self._tool_response(
                "task",
                {
                    "description": "Research the bounded local evidence.",
                    "subagent_type": "research",
                },
            )
        if "execute_scholar_skill" in names and not state["execute"]:
            state["execute"] = True
            return self._tool_response(
                "execute_scholar_skill",
                {"task_type": selected_task, "instruction": "execute the selected skill"},
            )
        if (
            "task" in names
            and not state["review"]
            and selected_task in {"WRITE_INTRODUCTION", "WRITE_CONCLUSION", "WRITE_ABSTRACT"}
        ):
            state["review"] = True
            return self._tool_response(
                "task",
                {
                    "description": "Review the proposed draft.",
                    "subagent_type": "reviewer",
                },
            )
        return self._text_response("fixture completed")


def _load_provider_environment(env_root: Path) -> AgenticRAGConfig:
    """Use the established settings loader and expose only its values in-process."""

    root = env_root.expanduser().resolve()
    settings = load_local_llm_settings(root)
    if not settings.base_url or not settings.model:
        raise RuntimeError("E2E requires LEO_LLM_BASE_URL and LEO_LLM_MODEL in the existing configuration.")
    os.environ.setdefault("LEO_LLM_BASE_URL", settings.base_url)
    os.environ.setdefault("LEO_LLM_MODEL", settings.model)
    if settings.api_key is not None:
        os.environ.setdefault("LEO_LLM_API_KEY", settings.api_key.get_secret_value())
    os.environ.setdefault("LEO_LLM_TIMEOUT_SECONDS", str(settings.timeout_seconds))
    os.environ.setdefault("LEO_LLM_MAX_TOKENS", str(settings.max_tokens))
    return AgenticRAGConfig.from_environment(root / ".env")


def _link_optional(root: Path, name: str, source: Path | None) -> None:
    if source is None:
        return
    target = root / name
    if name == "data":
        target.mkdir(parents=True, exist_ok=True)
        source_root = source.expanduser().resolve()
        canonical = target / "canonical"
        if not canonical.exists() and not canonical.is_symlink():
            source_canonical = source_root / "canonical"
            if source_canonical.is_dir():
                # The knowledge builder records project-relative canonical
                # paths. A real copy keeps those paths valid in the isolated
                # demo project and remains small for the checked-in fixture.
                shutil.copytree(source_canonical, canonical)
        for child_name in ("indexes", "models"):
            child = source_root / child_name
            target_child = target / child_name
            if child.exists() and not target_child.exists() and not target_child.is_symlink():
                target_child.symlink_to(child, target_is_directory=True)
        index = target / "index"
        index.mkdir(exist_ok=True)
        source_index = source_root / "index"
        if source_index.is_dir() and not any((target / "index").iterdir()):
            # The checked-in/repository data index is immutable input for the
            # demo fixture. Copy only the retrieval artifacts that are
            # validated below; avoid a multi-minute embedding rebuild when
            # the source knowledge digest already matches those artifacts.
            for child_name in (
                "bm25.json",
                "dense_manifest.json",
                "paper_bm25.json",
                "paper_dense_manifest.json",
                "qdrant_dense",
                "qdrant_papers_dense",
            ):
                child = source_index / child_name
                destination = target / "index" / child_name
                if child.is_dir():
                    shutil.copytree(child, destination)
                elif child.is_file():
                    shutil.copy2(child, destination)
        if index.is_symlink():
            index.unlink()
        index.mkdir(exist_ok=True)
        knowledge = source_root / "knowledge"
        if knowledge.is_dir() and not (target / "knowledge").exists():
            shutil.copytree(knowledge, target / "knowledge")
        (target / "inbox").mkdir(exist_ok=True)
        return
    if target.exists() or target.is_symlink():
        return
    target.symlink_to(source.expanduser().resolve(), target_is_directory=True)


def _local_indexes_ready(project_root: Path) -> bool:
    """Check retrieval artifacts through the same read path as Production."""

    root = project_root.expanduser().resolve()
    try:
        digest = chunks_digest(load_chunks(root))
        bm25 = json.loads(
            (root / "data" / "index" / "bm25.json").read_text(encoding="utf-8")
        )
        dense = load_dense_manifest(root)
        paper = json.loads(
            (root / "data" / "index" / "paper_bm25.json").read_text(encoding="utf-8")
        )
        paper_dense = load_paper_dense_manifest(root)
        papers = papers_digest(load_paper_records(root))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        bm25.get("chunks_digest") == digest
        and dense.get("chunks_digest") == digest
        and paper.get("paper_bm25_schema_version")
        and paper_dense.get("papers_digest") == papers
        and (root / "data" / "index" / "qdrant_dense").is_dir()
        and (root / "data" / "index" / "qdrant_papers_dense").is_dir()
    )


def _align_legacy_chunk_projection(project_root: Path) -> bool:
    """Align a copied legacy index with its rebuildable JSON projection.

    Some production snapshots serve chunks from the structured repository,
    where ``work_id`` is absent from the legacy retrieval projection, while
    the JSON export contains the newer denormalized field. The index digest is
    authoritative for this immutable demo input. If the only mismatch is
    that field, materialize the equivalent legacy projection instead of
    needlessly embedding the whole corpus again.
    """

    root = project_root.expanduser().resolve()
    chunks_path = root / "data" / "knowledge" / "chunks.jsonl"
    manifest_path = root / "data" / "index" / "dense_manifest.json"
    if not chunks_path.is_file() or not manifest_path.is_file():
        return False
    try:
        expected = json.loads(manifest_path.read_text(encoding="utf-8")).get("chunks_digest")
        values = [
            json.loads(line)
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(expected, str) or not all(isinstance(value, dict) for value in values):
        return False
    normalized = [
        {**value, "work_id": None}
        if value.get("work_id") is not None
        else value
        for value in values
    ]
    if chunks_digest(normalized) != expected:
        return False
    chunks_path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n" for value in normalized),
        encoding="utf-8",
    )
    return True


def _prepare_local_indexes(project_root: Path) -> None:
    """Materialize coherent local indexes through the existing builder only."""

    if not (project_root / "data" / "knowledge" / "chunks.jsonl").is_file():
        return
    if _local_indexes_ready(project_root):
        return
    if _align_legacy_chunk_projection(project_root) and _local_indexes_ready(project_root):
        return
    config = WebRuntimeConfig.from_environment(project_root)
    provider = BGEM3EmbeddingProvider(
        BGEM3Config(
            model_name=config.embedding_model,
            revision=config.embedding_revision,
            device=config.device,
            cache_folder=config.model_cache,
            batch_size=config.embedding_batch_size,
            local_files_only=config.local_files_only,
            show_progress_bar=False,
        )
    )
    build_hierarchical_indexes(project_root, provider, force=True)


def _seed_project(store: Any) -> None:
    """Seed only user-confirmed fixture facts; no Agent-generated state is added."""

    store.put_fact(
        ManuscriptFact(
            "FACT_DEMO_SETUP",
            "experiment_setup",
            "131 simulated LEO satellite passes",
            confirmed=True,
        )
    )
    store.put_fact(
        ManuscriptFact(
            "FACT_DEMO_METRIC",
            "positioning_metric",
            "three-dimensional position error",
            confirmed=True,
        )
    )
    store.put_contribution(
        Contribution(
            "CONTRIB_DEMO_EPHEMERIS",
            "The study evaluates explicit ephemeris-error compensation before receiver-state estimation.",
            "confirmed",
            True,
        )
    )


def _cases(project_id: str) -> tuple[OrchestrationEvaluationCase, ...]:
    return default_orchestration_cases(project_id)


def _result_summary(result: Any) -> dict[str, Any]:
    if hasattr(result, "backend"):
        return {
            "status": result.status,
            "backend": result.backend,
            "selected_route": result.selected_route,
            "error_codes": list(result.error_codes),
            "patch_id": (
                result.pending_action.get("patch_id")
                if isinstance(result.pending_action, dict)
                else None
            ),
            "approval_required": result.approval_required,
            "trace_id": result.trace_id,
        }
    value = getattr(result, "value", None)
    patch = getattr(value, "patch", None)
    return {
        "status": result.status,
        "result_type": result.result_type,
        "task_type": result.task_type,
        "skill_name": result.skill_name,
        "error_codes": list(result.error_codes),
        "patch_id": getattr(patch, "patch_id", None),
        "termination_reason": result.metadata.get("trace", {}).get("termination_reason"),
        "error_type": result.metadata.get("error_type"),
        "error": result.metadata.get("error"),
        "selected_skill": result.metadata.get("selected_skill"),
        "visible_capabilities": result.metadata.get("visible_capabilities", []),
        "subagent_names": result.metadata.get("subagent_names", []),
        "unexpected_tool_calls": result.metadata.get("unexpected_tool_calls", []),
    }


def main() -> int:
    args = _parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    if not (project_root / "main.tex").is_file():
        raise SystemExit(f"E2E project has no main.tex: {project_root}")
    _link_optional(project_root, "data", args.data_root)
    _link_optional(project_root, "skills", args.skills_root)
    _prepare_local_indexes(project_root)

    fixture_provider = _DeterministicFixtureProvider() if args.fixture_provider else None
    base_config = (
        AgenticRAGConfig.from_environment(project_root / ".env")
        if fixture_provider is not None
        else _load_provider_environment(args.env_root)
    )
    checkpoint = project_root / "data" / "runtime" / "scholar" / "checkpoint.sqlite"
    config = replace(
        base_config,
        runtime_mode="production",
        scholar_checkpoint_path=checkpoint,
    )

    with ScholarRuntimeFactory(
        project_root,
        config=config,
        model=fixture_provider,
        orchestration_backend=args.orchestration_backend,
    ).open() as runtime:
        _seed_project(runtime.project_store)
        cases = _cases(runtime.project_store.project_id)
        results: list[Any] = []
        summaries: list[dict[str, Any]] = []
        for index, case in enumerate(cases, 1):
            result = runtime.scholar_orchestration.scholar_request(
                case.instruction,
                runtime.project_store.project_id,
                task_type=case.task_type,
                session_id=f"FINAL_E2E_SESSION_{index}",
                thread_id=f"FINAL_E2E_THREAD_{index}",
            )
            results.append(result)
            summaries.append(_result_summary(result))

        introduction = results[1]
        introduction_value = getattr(introduction, "value", None)
        patch = getattr(introduction_value, "patch", None)
        approval_summary: dict[str, Any] = {"requested": args.accept_introduction, "status": "SKIPPED"}
        if args.accept_introduction and patch is not None:
            approval = PatchApprovalService(
                project_root,
                project_store=runtime.project_store,
            )
            applied = approval.approve(
                PatchApprovalRequest(
                    patch.patch_id,
                    runtime.project_store.project_id,
                    "ACCEPT",
                    patch.base_hash,
                    "human:phase4-e2e",
                )
            )
            approval_summary = {
                "requested": True,
                "status": applied.status,
                "patch_id": applied.patch_id,
                "tex_changed": applied.apply_result is not None,
            }

            bridge = LatexBridgeService(project_root, project_store=runtime.project_store)
            build = bridge.request_build(runtime.project_store.project_id, patch_id=patch.patch_id)
            compiler = shutil.which("latexmk") or shutil.which("pdflatex") or shutil.which("tectonic")
            if compiler is None:
                build = bridge.report_build(
                    runtime.project_store.project_id,
                    build_id=build.build_id,
                    status="UNAVAILABLE",
                    message="No supported LaTeX compiler is installed; VS Code LaTeX Workshop can execute this bridge request.",
                )
            approval_summary["build"] = {
                "status": build.status,
                "build_id": build.build_id,
                "compiler_available": compiler is not None,
            }

        backend_name = (
            str(args.orchestration_backend or os.getenv("ORCHESTRATION_BACKEND") or "crewai")
            .strip()
            .lower()
        )
        report = evaluate_backend(
            backend_name,
            cases,
            lambda case: results[cases.index(case)],
        )
        payload = {
            "production": {
                "project_id": runtime.project_store.project_id,
                "checkpoint": type(runtime.checkpointer).__name__,
                "checkpoint_path": str(checkpoint),
            },
            "cases": summaries,
            "evaluation": report.to_dict(),
            "approval": approval_summary,
            "record_count": len(report.records),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
