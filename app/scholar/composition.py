"""The single production composition root for the Scholar Runtime.

This module assembles existing domain services.  It intentionally contains no
task routing, research planning, skill policy, or file-application logic.
"""

from __future__ import annotations

import asyncio
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping

from app.agentic.config import AgenticRAGConfig, RuntimeMode
from app.generation.openai_compatible import (
    OpenAICompatibleAnswerProvider,
    OpenAICompatibleConfig,
)
from app.generation.settings import load_local_llm_settings
from app.langchain_agent.checkpoint_factory import (
    CheckpointUnavailable,
    open_checkpointer,
)
from langgraph.checkpoint.memory import InMemorySaver
from app.scholar.approval import PatchApprovalService
from app.scholar.citation import CitationResolutionService
from app.scholar.context import ScholarContextBudget
from app.scholar.errors import ConfigurationError
from app.scholar.manuscript import ManuscriptSynchronizer
from app.scholar.project import ScholarProjectStore
from app.scholar.research import (
    ResearchCapabilityService,
    WebLiteratureAdapter,
)
from app.scholar.research.cache import LiteratureDiscoveryCache
from app.scholar.writing import (
    ChatCompletionIntroductionWriter,
    ChatCompletionSynthesisWriter,
    ClaimSupportService,
    IntroductionReviewer,
    SectionDraft,
    ScholarSkillRuntime,
    ScholarWritingService,
    SharedManuscriptReviewer,
    SynthesisWritingService,
)
from app.scholar.harness import ScholarHarnessService
from app.session import SessionManager


@dataclass(slots=True)
class ScholarRuntimeBundle:
    """All runtime-scoped resources created by :class:`ScholarRuntimeFactory`."""

    project_root: Path
    mode: RuntimeMode
    config: AgenticRAGConfig
    harness: ScholarHarnessService
    project_store: ScholarProjectStore
    session_manager: SessionManager
    checkpointer: Any
    _resources: ExitStack
    _closed: bool = False

    @property
    def scholar_harness(self) -> ScholarHarnessService:
        return self.harness

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._resources.close()

    def __enter__(self) -> "ScholarRuntimeBundle":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    shutdown = close


class ScholarRuntimeFactory:
    """唯一完整 Scholar Runtime 组装入口.

    Components may be overridden by tests or an embedding application, but
    Web, CLI and production code all consume the resulting bundle rather than
    manually assembling a second dependency graph.
    """

    def __init__(
        self,
        project_root: Path,
        *,
        mode: RuntimeMode | None = None,
        runtime_mode: RuntimeMode | None = None,
        config: AgenticRAGConfig | None = None,
        checkpoint_path: Path | None = None,
        component_overrides: Mapping[str, Any] | None = None,
        dependencies: Mapping[str, Any] | None = None,
        checkpointer: Any | None = None,
        model: Any | None = None,
        project_store: ScholarProjectStore | None = None,
        session_manager: SessionManager | None = None,
        research: ResearchCapabilityService | None = None,
        skill_runtime: ScholarSkillRuntime | None = None,
        writers: Mapping[str, Any] | None = None,
        reviewer: Any | None = None,
        harness: ScholarHarnessService | None = None,
        interrupt_on: Mapping[str, Any] | None = None,
    ) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.config = config or AgenticRAGConfig.from_environment(self.project_root / ".env")
        if checkpoint_path is not None:
            self.config = replace(self.config, scholar_checkpoint_path=checkpoint_path)
        if component_overrides and dependencies:
            raise ConfigurationError("component_overrides 与 dependencies 不能同时提供。")
        overrides = dict(component_overrides or dependencies or {})
        self.mode = mode or runtime_mode or self.config.runtime_mode
        if self.mode not in {"production", "test", "local-fast"}:
            raise ConfigurationError(f"不支持的 Scholar runtime mode：{self.mode}")
        self._provided = {
            "checkpointer": checkpointer,
            "model": model,
            "project_store": project_store,
            "session_manager": session_manager,
            "research": research,
            "skill_runtime": skill_runtime,
            "writers": writers,
            "reviewer": reviewer,
            "harness": harness,
        }
        self.interrupt_on = dict(interrupt_on or {}) or None
        for key, value in overrides.items():
            if key not in self._provided:
                raise ConfigurationError(f"未知 Scholar Runtime dependency override：{key}")
            if self._provided[key] is not None and self._provided[key] is not value:
                raise ConfigurationError(f"重复提供 Scholar Runtime dependency：{key}")
            self._provided[key] = value
        self._bundle: ScholarRuntimeBundle | None = None

    def _checkpoint(self, stack: ExitStack) -> Any:
        provided = self._provided["checkpointer"]
        if provided is not None:
            if self.mode == "production" and isinstance(provided, InMemorySaver):
                raise CheckpointUnavailable(
                    "CHECKPOINT_UNAVAILABLE: production 不允许注入 InMemorySaver。"
                )
            return provided
        if self.mode == "production":
            raw_path = self.config.scholar_checkpoint_path
            if raw_path is None:
                raise CheckpointUnavailable(
                    "CHECKPOINT_UNAVAILABLE: production 必须配置 LEO_AGENTIC_SCHOLAR_CHECKPOINT_PATH。"
                )
            path = raw_path if raw_path.is_absolute() else self.project_root / raw_path
            try:
                return stack.enter_context(open_checkpointer("sqlite", database_path=path))
            except (OSError, RuntimeError, ValueError) as error:
                if isinstance(error, CheckpointUnavailable):
                    raise
                raise CheckpointUnavailable(
                    f"CHECKPOINT_UNAVAILABLE: 无法打开 production checkpoint：{path}"
                ) from error
        return stack.enter_context(open_checkpointer("memory"))

    def _model(self) -> tuple[Any, Any]:
        provided = self._provided["model"]
        if provided is not None:
            return provided, None
        llm = load_local_llm_settings(self.project_root)
        if not llm.base_url or not llm.model:
            raise ConfigurationError(
                "CONFIGURATION_ERROR: 缺少 LEO_LLM_BASE_URL 或 LEO_LLM_MODEL。"
            )
        api_key = llm.api_key.get_secret_value() if llm.api_key else None
        provider = OpenAICompatibleAnswerProvider(
            OpenAICompatibleConfig(
                base_url=llm.base_url,
                model=llm.model,
                api_key=api_key,
                timeout_seconds=llm.timeout_seconds,
                max_tokens=llm.max_tokens,
                prompt_layout=llm.prompt_layout or "context_first",
                json_mode=llm.json_mode,
            )
        )
        from app.langchain_agent.provider import OpenAICompatibleChatModel

        return OpenAICompatibleChatModel(
            provider=provider,
            model_name=llm.model,
            max_tokens=llm.max_tokens,
        ), provider

    def _default_domain_components(
        self,
        stack: ExitStack,
        project_store: ScholarProjectStore,
    ) -> tuple[ScholarSkillRuntime, Any, ResearchCapabilityService, Mapping[str, Any], Any]:
        """Build the already-established Scholar domain graph and adapters."""

        model, completion_provider = self._model()
        if completion_provider is None:
            # Test callers supplying a model normally also supply domain
            # services.  Keep this failure explicit instead of guessing how to
            # turn an arbitrary model into a chat_completion provider.
            raise ConfigurationError(
                "CONFIGURATION_ERROR: default domain composition 需要 chat_completion Provider。"
            )
        stack.callback(self._close_async_resource, completion_provider)
        from app.embeddings.bge_m3 import BGEM3Config, BGEM3EmbeddingProvider
        from app.knowledge_engine import build_configured_unified_service
        from app.reranking.bge import BGERerankerConfig, BGERerankerProvider
        from app.runtime.retrieval import RetrievalRuntime
        from app.academic_mcp.service import AcademicDiscoveryService
        from app.research import build_bootstrap_provider_composition, build_default_gateway
        from app.workspaces import WorkspaceService

        from app.web.runtime import WebRuntimeConfig

        web_config = WebRuntimeConfig.from_environment(self.project_root)
        retrieval = RetrievalRuntime(
            self.project_root,
            BGEM3EmbeddingProvider(
                BGEM3Config(
                    model_name=web_config.embedding_model,
                    revision=web_config.embedding_revision,
                    device=web_config.device,
                    cache_folder=web_config.model_cache,
                    batch_size=web_config.embedding_batch_size,
                    local_files_only=web_config.local_files_only,
                    show_progress_bar=False,
                )
            ),
            BGERerankerProvider(
                BGERerankerConfig(
                    model_name=web_config.reranker_model,
                    revision=web_config.reranker_revision,
                    device=web_config.device,
                    cache_folder=web_config.model_cache,
                    batch_size=web_config.reranker_batch_size,
                    max_length=web_config.reranker_max_length,
                    local_files_only=web_config.local_files_only,
                    show_progress_bar=False,
                )
            ),
        )
        knowledge = build_configured_unified_service(
            self.project_root,
            retrieval,
            completion_provider,
            llm_model_name=str(getattr(completion_provider, "model_name", "configured-model")),
        )
        workspaces = getattr(getattr(knowledge, "evidence", None), "workspaces", None)
        if workspaces is None:
            workspaces = WorkspaceService(self.project_root)
        backend = AcademicDiscoveryService.create(self.project_root)
        stack.callback(self._close_async_resource, backend)
        bootstrap = build_bootstrap_provider_composition(
            self.project_root,
            backend,
            worker_id="scholar-runtime-provider",
        )
        gateway = build_default_gateway(bootstrap.gateway_handlers)
        web = WebLiteratureAdapter(
            gateway=gateway,
            cache=LiteratureDiscoveryCache(self.project_root),
        )
        research = ResearchCapabilityService(knowledge, web=web)
        citation = CitationResolutionService(
            self.project_root,
            project_store=project_store,
        )
        approval = PatchApprovalService(
            self.project_root,
            project_store=project_store,
        )
        intro_writer = ChatCompletionIntroductionWriter(completion_provider)
        synthesis_writers = {
            "WRITE_CONCLUSION": ChatCompletionSynthesisWriter(completion_provider, "conclusion"),
            "WRITE_ABSTRACT": ChatCompletionSynthesisWriter(completion_provider, "abstract"),
        }
        introduction = ScholarWritingService(
            self.project_root,
            research,
            intro_writer,
            project_store=project_store,
            citation_service=citation,
            approval_service=approval,
        )
        support = ClaimSupportService(
            research,
            project_store=project_store,
            citation_service=citation,
        )
        synthesis = SynthesisWritingService(
            self.project_root,
            project_store=project_store,
            approval_service=approval,
        )
        runtime = ScholarSkillRuntime(
            self.project_root,
            project_store=project_store,
            introduction=introduction,
            support_claim=support,
            synthesis=synthesis,
        )
        synthesis_reviewer = SharedManuscriptReviewer()
        introduction_reviewer = IntroductionReviewer()
        synchronizer = ManuscriptSynchronizer(self.project_root)

        def review_for_harness(context: Mapping[str, Any]) -> Any:
            task_type = str(context.get("task_type") or "")
            content = str(context.get("draft") or "")
            target = {
                "WRITE_INTRODUCTION": "introduction",
                "WRITE_CONCLUSION": "conclusion",
                "WRITE_ABSTRACT": "abstract",
            }.get(task_type, "introduction")
            draft = SectionDraft(
                target_section=target,
                base_hash="harness-review",
                content=content or "placeholder",
                claim_ids=tuple(
                    str(value)
                    for value in getattr(context.get("claim_plan"), "claim_ids", ())
                ),
                evidence_ids=tuple(
                    str(value.get("evidence_id"))
                    for value in context.get("evidence", ())
                    if isinstance(value, Mapping) and value.get("evidence_id")
                ),
            )
            facts = tuple(project_store.list_facts())
            contributions = tuple(project_store.list_contributions())
            state = synchronizer.scan()
            sections = {
                name: synchronizer.read_section(state, name)
                for name in state.sections
            }
            if task_type == "WRITE_INTRODUCTION" and context.get("claim_plan") is not None:
                evidence = {
                    str(value.get("evidence_id")): value
                    for value in context.get("evidence", ())
                    if isinstance(value, Mapping) and value.get("evidence_id")
                }
                return introduction_reviewer.review(
                    draft,
                    context["claim_plan"],
                    evidence,
                    facts,
                    contributions,
                    {},
                )
            return synthesis_reviewer.review_synthesis(
                draft,
                policy="ABSTRACT" if task_type == "WRITE_ABSTRACT" else "CONCLUSION",
                manuscript_sections=sections,
                facts=facts,
                contributions=contributions,
            )

        return runtime, model, research, {"WRITE_INTRODUCTION": intro_writer, **synthesis_writers}, review_for_harness

    @staticmethod
    def _close_async_resource(resource: Any) -> None:
        close = getattr(resource, "close", None)
        if not callable(close):
            return
        result = close()
        if not asyncio.iscoroutine(result):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(result)
        else:
            loop.create_task(result)

    def build(self) -> ScholarRuntimeBundle:
        if self._bundle is not None and not self._bundle._closed:
            return self._bundle
        stack = ExitStack()
        try:
            project_store = self._provided["project_store"] or ScholarProjectStore(self.project_root)
            session_manager = self._provided["session_manager"] or SessionManager(self.project_root)
            checkpointer = self._checkpoint(stack)
            # A newly-created process owns a new worker id.  Marking stale
            # RUNNING records before exposing the Harness makes orphan
            # detection part of runtime startup, while resume still requires
            # a real checkpoint below the Harness boundary.
            session_manager.recover_orphans()
            harness_override = self._provided["harness"]
            if harness_override is not None:
                bundle = ScholarRuntimeBundle(
                    self.project_root,
                    self.mode,
                    self.config,
                    harness_override,
                    project_store,
                    session_manager,
                    checkpointer,
                    stack,
                )
                self._bundle = bundle
                return bundle

            model = self._provided["model"]
            research = self._provided["research"]
            skill_runtime = self._provided["skill_runtime"]
            writers = self._provided["writers"]
            reviewer = self._provided["reviewer"]
            if skill_runtime is None or model is None or research is None:
                default_runtime, default_model, default_research, default_writers, default_reviewer = self._default_domain_components(stack, project_store)
                skill_runtime = skill_runtime or default_runtime
                model = model or default_model
                research = research or default_research
                writers = writers or default_writers
                reviewer = reviewer or default_reviewer
            harness = ScholarHarnessService(
                self.project_root,
                skill_runtime=skill_runtime,
                model=model,
                project_store=project_store,
                research=research,
                writers=writers,
                reviewer=reviewer,
                checkpointer=checkpointer,
                session_manager=session_manager,
                interrupt_on=self.interrupt_on,
                max_steps=self.config.scholar_max_steps,
                harness_policy=None,
                context_budget=ScholarContextBudget(
                    supervisor=self.config.scholar_supervisor_context_budget,
                    research=self.config.scholar_research_context_budget,
                    reviewer=self.config.scholar_reviewer_context_budget,
                    total=self.config.scholar_total_context_budget,
                ),
            )
            # Fail at composition time if a framework upgrade or middleware
            # default makes an unsafe tool visible.  This is an assertion over
            # the effective Skill/Capability surface, not prompt guidance.
            for definition in skill_runtime.registry.list():
                harness.permission_audit(definition.task_type)
            bundle = ScholarRuntimeBundle(
                self.project_root,
                self.mode,
                self.config,
                harness,
                project_store,
                session_manager,
                checkpointer,
                stack,
            )
            self._bundle = bundle
            return bundle
        except Exception:
            stack.close()
            raise

    create = build

    def __enter__(self) -> ScholarRuntimeBundle:
        return self.build()

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def open(self) -> Iterator[ScholarRuntimeBundle]:
        bundle = self.build()
        try:
            yield bundle
        finally:
            bundle.close()

    def close(self) -> None:
        if self._bundle is not None:
            self._bundle.close()
