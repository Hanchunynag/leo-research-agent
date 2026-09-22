"""The single production Harness for the CrewAI Scholar Runtime.

``Harness`` is the only process-level composition boundary for the production
CrewAI chain.  It assembles existing domain services and owns their lifecycle;
it intentionally contains no task routing, research planning, skill policy, or
file-application logic.  Requests still enter through ``ScholarRunManager``
and only ``ScholarRunWorker`` invokes the resulting orchestration service.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from app.runtime.config import RuntimeMode, ScholarRuntimeConfig
from app.generation.openai_compatible import (
    OpenAICompatibleAnswerProvider,
    OpenAICompatibleConfig,
)
from app.generation.settings import load_local_llm_settings
from app.scholar.approval import PatchApprovalService
from app.scholar.citation import CitationResolutionService
from app.scholar.errors import ConfigurationError
from app.scholar.manuscript import ManuscriptSynchronizer
from app.scholar.project import ScholarProjectStore
from app.scholar.research import (
    ResearchCapabilityService,
    WebLiteratureAdapter,
)
from app.scholar.research.cache import LiteratureDiscoveryCache
from app.scholar.research.gateway import build_default_gateway
from app.scholar.research.providers import build_bootstrap_provider_composition
from app.scholar.writing import (
    ChatCompletionIntroductionWriter,
    ChatCompletionSynthesisWriter,
    ClaimSupportService,
    IntroductionReviewer,
    ResearchDelegate,
    SectionDraft,
    ScholarSkillRuntime,
    ScholarWritingService,
    SharedManuscriptReviewer,
    SynthesisWritingService,
)
from app.session import SessionManager
from app.orchestration.service import CrewAIBackend, ScholarOrchestrationService
from app.scholar.events import RunEventStore

__all__ = ["Harness"]


@dataclass(slots=True)
class ScholarRuntimeBundle:
    """All runtime-scoped resources created by :class:`Harness`."""

    project_root: Path
    mode: RuntimeMode
    config: ScholarRuntimeConfig
    project_store: ScholarProjectStore
    session_manager: SessionManager
    _resources: ExitStack
    orchestration: ScholarOrchestrationService
    _closed: bool = False

    @property
    def scholar_orchestration(self) -> ScholarOrchestrationService:
        """The single Scholar orchestration entry point."""

        return self.orchestration

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


class Harness:
    """唯一完整 CrewAI Scholar Runtime 组装与生命周期入口。

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
        config: ScholarRuntimeConfig | None = None,
        component_overrides: Mapping[str, Any] | None = None,
        dependencies: Mapping[str, Any] | None = None,
        model: Any | None = None,
        project_store: ScholarProjectStore | None = None,
        session_manager: SessionManager | None = None,
        research: ResearchCapabilityService | None = None,
        skill_runtime: ScholarSkillRuntime | None = None,
        writers: Mapping[str, Any] | None = None,
        reviewer: Any | None = None,
        orchestration: ScholarOrchestrationService | None = None,
    ) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.config = config or ScholarRuntimeConfig.from_environment(self.project_root)
        if component_overrides and dependencies:
            raise ConfigurationError("component_overrides 与 dependencies 不能同时提供。")
        overrides = dict(component_overrides or dependencies or {})
        self.mode = mode or runtime_mode or self.config.runtime_mode
        if self.mode not in {"production", "test", "local-fast"}:
            raise ConfigurationError(f"不支持的 Scholar runtime mode：{self.mode}")
        self._provided = {
            "model": model,
            "project_store": project_store,
            "session_manager": session_manager,
            "research": research,
            "skill_runtime": skill_runtime,
            "writers": writers,
            "reviewer": reviewer,
            "orchestration": orchestration,
        }
        for key, value in overrides.items():
            if key not in self._provided:
                raise ConfigurationError(f"未知 Scholar Runtime dependency override：{key}")
            if self._provided[key] is not None and self._provided[key] is not value:
                raise ConfigurationError(f"重复提供 Scholar Runtime dependency：{key}")
            self._provided[key] = value
        self._bundle: ScholarRuntimeBundle | None = None

    def _model(self) -> tuple[Any, Any]:
        provided = self._provided["model"]
        if provided is not None:
            # A deterministic/local-compatible provider may be injected for
            # release validation. Keep it as the CrewAI model and also pass
            # its existing chat_completion surface to the real domain
            # writers; no domain service is replaced by this boundary.
            completion_provider = getattr(provided, "provider", provided)
            if callable(getattr(completion_provider, "chat_completion", None)):
                return provided, completion_provider
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
                auth_scheme=llm.auth_scheme,
                timeout_seconds=llm.timeout_seconds,
                max_tokens=llm.max_tokens,
                prompt_layout=llm.prompt_layout or "context_first",
                json_mode=llm.json_mode,
            )
        )
        return provider, provider

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
        from app.knowledge import build_knowledge_service
        from app.reranking.bge import BGERerankerConfig, BGERerankerProvider
        from app.runtime.retrieval import RetrievalRuntime
        from app.academic_mcp.service import AcademicDiscoveryService

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
        knowledge = build_knowledge_service(self.project_root, retrieval)
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
        try:
            research_parallelism = int(
                # BGE-M3 and the Cross-Encoder are process-local CPU/GPU
                # resources. Serial ResearchNeeds are materially faster than
                # making several workers queue behind the same model lock;
                # callers can still opt into bounded parallelism explicitly.
                os.getenv("LEO_SCHOLAR_RESEARCH_MAX_CONCURRENCY", "1")
            )
        except ValueError as error:
            raise ConfigurationError(
                "CONFIGURATION_ERROR: LEO_SCHOLAR_RESEARCH_MAX_CONCURRENCY 必须是整数。"
            ) from error
        research = ResearchCapabilityService(
            knowledge,
            web=web,
            max_parallel_retrievals=research_parallelism,
        )
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
            delegate=ResearchDelegate(
                research,
                max_parallelism=research_parallelism,
            ),
        )
        support = ClaimSupportService(
            research,
            project_store=project_store,
            citation_service=citation,
            delegate=ResearchDelegate(
                research,
                max_parallelism=research_parallelism,
            ),
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

        def review_domain_result(context: Mapping[str, Any]) -> Any:
            task_type = str(context.get("task_type") or "")
            content = str(context.get("draft") or "")
            target = {
                "WRITE_INTRODUCTION": "introduction",
                "WRITE_CONCLUSION": "conclusion",
                "WRITE_ABSTRACT": "abstract",
            }.get(task_type, "introduction")
            draft = SectionDraft(
                target_section=target,
                base_hash="orchestration-review",
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

        return runtime, model, research, {"WRITE_INTRODUCTION": intro_writer, **synthesis_writers}, review_domain_result

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
            orchestration_service = self._provided["orchestration"]
            if orchestration_service is None:
                orchestration_service = ScholarOrchestrationService(
                    backend="crewai",
                    crewai=CrewAIBackend(
                        self.project_root,
                        model=model,
                        skill_runtime=skill_runtime,
                        research=research,
                        writers=writers,
                        reviewer=reviewer,
                        project_store=project_store,
                        session_manager=session_manager,
                        event_store=RunEventStore(self.project_root),
                        max_review_rounds=2,
                        max_manager_steps=self.config.scholar_max_manager_steps,
                        max_tool_calls=16,
                    ),
                )
            bundle = ScholarRuntimeBundle(
                self.project_root,
                self.mode,
                self.config,
                project_store,
                session_manager,
                stack,
                orchestration_service,
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
