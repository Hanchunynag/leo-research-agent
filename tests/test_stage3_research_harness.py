from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Mapping

import httpx
import pytest

from app.research import (
    BudgetExceeded,
    ClaimEvidenceValidator,
    HarnessState,
    HarnessAgentService,
    ResearchBudgetPolicy,
    ResearchContextManager,
    ResearchRunHarness,
    ResearchRuntime,
    ResearchStateStore,
    RunTraceStore,
    WorkflowName,
    WorkflowRequest,
    build_default_gateway,
    unified_tool_handlers,
)
from app.agentic.store import AgenticSessionStore
from app.contracts import ResearchScope


def evidence(evidence_id: str, *, directness: str = "direct", grade: str = "primary") -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "evidence_state": "selected",
        "document_id": "D_001",
        "chunk_id": f"D_001_{evidence_id}",
        "page_start": 1,
        "page_end": 1,
        "block_ids": ["B_001"],
        "content": f"Canonical evidence {evidence_id}",
        "evidence_grade": grade,
        "directness": directness,
    }


class Generator:
    def __init__(self) -> None:
        self.calls = 0
        self.contexts: list[Any] = []

    def generate(self, context: Any) -> Mapping[str, Any]:
        self.calls += 1
        self.contexts.append(context)
        first = context.selected_evidence[0]
        return {
            "answerable": True,
            "claims": [
                {
                    "claim_id": "C1",
                    "text": "Canonical evidence supports the scientific claim.",
                    "source_ids": [first["source_id"]],
                    "evidence_ids": [first["evidence_id"]],
                }
            ],
            "refusal_reason": None,
            "usage": {"total_tokens": 20},
        }


def handlers(
    *,
    relation_gap: bool = False,
    no_evidence: bool = False,
    fail_retrieval: bool = False,
    fail_gap_retrieval: bool = False,
    fail_external_search: bool = False,
) -> tuple[dict[str, Any], dict[str, int]]:
    calls: dict[str, int] = {}

    def count(name: str) -> None:
        calls[name] = calls.get(name, 0) + 1

    def scope(arguments: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        count("workspace.read_scope")
        assert arguments["workspace_id"] == context["workspace_id"]
        return {"workspace_id": arguments["workspace_id"], "scope_version": arguments["scope_version"], "constraints": ["local-only"]}

    def retrieve(arguments: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        count("knowledge.retrieve")
        assert arguments["workspace_id"] == context["workspace_id"]
        assert arguments["scope_version"] == context["scope_version"]
        if fail_retrieval:
            raise ConnectionError("backend unavailable")
        if fail_gap_retrieval and calls["knowledge.retrieve"] > 1:
            raise ConnectionError("gap backend unavailable")
        if no_evidence:
            return {"results": []}
        if relation_gap and calls["knowledge.retrieve"] == 1:
            return {"results": [evidence("E1")]}
        return {"results": [evidence("E1"), evidence("E2", directness="indirect")]}

    def search(arguments: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        count("literature.search")
        if fail_external_search:
            raise ConnectionError("external search unavailable")
        return {"results": [{"paper_id": f"P{index}"} for index in range(5)]}

    def metadata(arguments: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        count("literature.get_metadata")
        return {"paper_id": arguments["paper_id"], "role": "candidate", "lightweight": True}

    def download(arguments: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        count("literature.download")
        return {"path": f"/candidate/{arguments['paper_id']}.pdf"}

    def parse(arguments: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        count("document.parse")
        assert arguments["mode"] == "formal"
        return {"document_id": "D_NEW"}

    def update(arguments: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        count("workspace.update_scope")
        assert arguments["scope_version"] == context["scope_version"] == 1
        return {"workspace_id": arguments["workspace_id"], "scope_version": 2}

    values = {
        "workspace.read_scope": scope,
        "knowledge.retrieve": retrieve,
        "literature.search": search,
        "literature.get_metadata": metadata,
        "literature.download": download,
        "document.parse": parse,
        "workspace.update_scope": update,
        "job.get_status": lambda arguments, context: {"status": "completed"},
    }
    return values, calls


def runtime(tmp_path: Path, *, options: dict[str, bool] | None = None) -> tuple[ResearchRuntime, Generator, dict[str, int]]:
    values, calls = handlers(**(options or {}))
    generator = Generator()
    return (
        ResearchRuntime(
            build_default_gateway(values),
            generator,
            state_store=ResearchStateStore(tmp_path),
            trace_store=RunTraceStore(tmp_path),
        ),
        generator,
        calls,
    )


def test_top_level_harness_has_only_generic_states_and_enforces_all_budgets() -> None:
    forbidden = {"lexical_retrieving", "dense_retrieving", "graph_retrieving", "community_retrieving"}
    assert forbidden.isdisjoint({value.value for value in HarnessState})
    policy = ResearchBudgetPolicy(
        max_steps=1,
        max_llm_calls=0,
        max_tool_calls=0,
        max_retrieval_rounds=0,
        max_external_searches=0,
        max_repairs=0,
        max_context_tokens=1,
        max_total_tokens=1,
    )
    harness = ResearchRunHarness("direct_qa", policy)
    harness.transition(HarnessState.CONTEXT_PREPARING)
    with harness.step("one"):
        pass
    with pytest.raises(BudgetExceeded):
        with harness.step("two"):
            pass


def test_context_manager_minimizes_router_and_rejects_candidate_for_generator() -> None:
    manager = ResearchContextManager()
    router = manager.router_pack(
        query="question",
        workspace_id="default",
        scope_version=1,
        workspace_summary={"document_count": 2},
        recent_conversation=[{"role": "user", "content": "old"}] * 8,
    )
    assert router.selected_evidence == ()
    assert len(router.recent_conversation) == 4
    with pytest.raises(ValueError, match="selected"):
        manager.generator_pack(
            query="question",
            workspace_id="default",
            scope_version=1,
            scope_constraints=[],
            evidence=[{"evidence_id": "CANDIDATE", "evidence_state": "retrieved"}],
            conflicts=[],
            output_format={},
        )


def test_tool_registry_declares_all_governance_fields() -> None:
    gateway = build_default_gateway()
    assert {value.name for value in gateway.specs()} == {
        "knowledge.retrieve",
        "workspace.read_scope",
        "workspace.update_scope",
        "literature.search",
        "literature.get_metadata",
        "literature.resolve_publication_date",
        "literature.download",
        "document.parse",
        "job.get_status",
    }
    for value in gateway.specs():
        assert value.input_schema and value.output_schema
        assert value.permission and value.timeout_seconds > 0
        assert value.allowed_workflows
        assert value.side_effect in {"none", "read", "bounded_write", "external_write"}


def test_tool_gateway_blocks_cross_workspace_scope_leakage() -> None:
    values, calls = handlers()
    gateway = build_default_gateway(values)
    harness = ResearchRunHarness("direct_qa", ResearchBudgetPolicy())
    harness.transition(HarnessState.CONTEXT_PREPARING)
    harness.transition(HarnessState.PLANNING)
    harness.transition(HarnessState.EXECUTING)
    with pytest.raises(PermissionError, match="workspace_id"):
        gateway.invoke(
            "knowledge.retrieve",
            {"query": "q", "workspace_id": "outside", "scope_version": 1},
            context={
                "workflow": "direct_qa",
                "workspace_id": "default",
                "scope_version": 1,
                "permissions": ["knowledge.read"],
            },
            harness=harness,
        )
    assert "knowledge.retrieve" not in calls


def test_unified_handler_exposes_only_selected_evidence_and_explicit_scope() -> None:
    class Knowledge:
        workspace_id = "default"
        last_diagnostics = {
            "evidence_intelligence": {"conflicts": [["E1", "E2"]]},
        }

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def retrieve(self, query: str, **kwargs: Any) -> Mapping[str, Any]:
            self.calls.append({"query": query, **kwargs})
            return {
                "results": [
                    evidence("E1"),
                    {**evidence("E2"), "evidence_state": "verified"},
                ]
            }

    class Workspaces:
        def require_scope(self, workspace_id: str, scope_version: int) -> Any:
            assert (workspace_id, scope_version) == ("default", 3)
            return object()

    knowledge = Knowledge()
    gateway = build_default_gateway(
        unified_tool_handlers(knowledge, Workspaces())
    )
    harness = ResearchRunHarness("direct_qa", ResearchBudgetPolicy())
    harness.transition(HarnessState.CONTEXT_PREPARING)
    harness.transition(HarnessState.PLANNING)
    harness.transition(HarnessState.EXECUTING)
    result = gateway.invoke(
        "knowledge.retrieve",
        {"query": "q", "workspace_id": "default", "scope_version": 3},
        context={
            "workflow": "direct_qa",
            "workspace_id": "default",
            "scope_version": 3,
            "permissions": ["knowledge.read"],
        },
        harness=harness,
    )
    assert [value["evidence_id"] for value in result["results"]] == ["E1"]
    assert result["results"][0]["source_id"] == "S1"
    assert result["diagnostics"]["conflicts"] == [
        {"supporting_evidence_id": "E1", "opposing_evidence_id": "E2"}
    ]
    assert knowledge.calls == [
        {
            "query": "q",
            "limit": 10,
            "workspace_id": "default",
            "scope_version": 3,
        }
    ]


def test_workspace_update_tool_adds_documents_without_dropping_current_scope() -> None:
    class Repository:
        def get_workspace(self, workspace_id: str) -> Any:
            return type("Workspace", (), {"scope_version": 4})()

    class Workspaces:
        repository = Repository()
        received: set[str] = set()

        def active_document_ids(
            self, workspace_id: str, scope_version: int
        ) -> frozenset[str]:
            assert (workspace_id, scope_version) == ("default", 4)
            return frozenset({"D_EXISTING"})

        def replace_scope_documents(
            self, workspace_id: str, document_ids: Any
        ) -> ResearchScope:
            assert workspace_id == "default"
            self.received = set(document_ids)
            return ResearchScope(
                workspace_id,
                5,
                included_document_ids=tuple(sorted(self.received)),
            )

    workspaces = Workspaces()
    gateway = build_default_gateway(
        unified_tool_handlers(type("Knowledge", (), {})(), workspaces)
    )
    harness = ResearchRunHarness("research_bootstrap", ResearchBudgetPolicy())
    harness.transition(HarnessState.CONTEXT_PREPARING)
    harness.transition(HarnessState.PLANNING)
    harness.transition(HarnessState.EXECUTING)
    result = gateway.invoke(
        "workspace.update_scope",
        {
            "workspace_id": "default",
            "scope_version": 4,
            "document_ids": ["D_NEW"],
        },
        context={
            "workflow": "research_bootstrap",
            "workspace_id": "default",
            "scope_version": 4,
            "permissions": ["workspace.write"],
        },
        harness=harness,
    )
    assert workspaces.received == {"D_EXISTING", "D_NEW"}
    assert result["included_document_ids"] == ("D_EXISTING", "D_NEW")


def test_direct_qa_uses_one_generation_and_no_optional_reasoning_calls(tmp_path: Path) -> None:
    service, generator, calls = runtime(tmp_path)
    result = service.run(
        WorkflowRequest("What is measured?", "default", 1, workspace_summary={"document_count": 2})
    )
    diagnostics = result["diagnostics"]
    assert result["workflow"] == "direct_qa"
    assert result["answerable"] is True
    assert generator.calls == 1
    assert calls["knowledge.retrieve"] == 1
    assert diagnostics["usage"]["llm_calls"] == 1
    assert diagnostics["usage"]["retrieval_rounds"] == 1
    assert diagnostics["state_history"] == [
        "created",
        "context_preparing",
        "planning",
        "executing",
        "evaluating",
        "committing",
        "completed",
    ]
    assert Path(result["trace_path"]).is_file()
    assert generator.contexts[0].audience == "generator"
    assert all(value["evidence_state"] == "selected" for value in result["selected_evidence"])


@pytest.mark.parametrize(
    "query",
    [
        "这些论文的研究路线如何演进？",
        "比较这些论文的发展路线",
        "What is the evolution roadmap across these papers?",
    ],
)
def test_evolution_queries_use_relation_reasoning_workflow(query: str) -> None:
    request = WorkflowRequest(
        query,
        "default",
        1,
        workspace_summary={"document_count": 2},
    )

    assert ResearchRuntime.route(request) == WorkflowName.RELATION_REASONING


def test_relation_reasoning_permits_only_one_gap_retrieval(tmp_path: Path) -> None:
    service, generator, calls = runtime(tmp_path, options={"relation_gap": True})
    result = service.run(
        WorkflowRequest(
            "How are A and B related?",
            "default",
            1,
            workflow=WorkflowName.RELATION_REASONING,
            workspace_summary={"document_count": 2},
        )
    )
    assert result["answerable"] is True
    assert generator.calls == 1
    assert calls["knowledge.retrieve"] == 2
    assert result["workflow_details"]["gap_retrievals"] == 1
    assert result["diagnostics"]["recovery_actions"] == [
        {"level": 4, "action": "one_bounded_gap_retrieval", "outcome": "continuing"}
    ]


def test_relation_gap_failure_keeps_first_round_verified_evidence(
    tmp_path: Path,
) -> None:
    service, generator, calls = runtime(
        tmp_path,
        options={"relation_gap": True, "fail_gap_retrieval": True},
    )
    result = service.run(
        WorkflowRequest(
            "How are A and B related?",
            "default",
            1,
            workflow=WorkflowName.RELATION_REASONING,
            workspace_summary={"document_count": 2},
        )
    )
    assert result["answerable"] is True
    assert generator.calls == 1
    assert calls["knowledge.retrieve"] == 3
    assert result["workflow_details"]["gap_retrieval_failed"] is True
    assert [
        value["level"] for value in result["diagnostics"]["recovery_actions"]
    ] == [4, 1, 2]


def test_deep_research_external_failure_uses_local_verified_evidence(
    tmp_path: Path,
) -> None:
    service, generator, calls = runtime(
        tmp_path,
        options={"fail_external_search": True},
    )
    result = service.run(
        WorkflowRequest(
            "explicit deep research",
            "default",
            1,
            workflow=WorkflowName.DEEP_RESEARCH,
            explicit_deep_research=True,
            workspace_summary={"document_count": 2},
        )
    )
    assert result["answerable"] is True
    assert generator.calls == 1
    assert calls["literature.search"] == 2
    assert result["workflow_details"]["external_search_failed"] is True
    assert [
        value["level"] for value in result["diagnostics"]["recovery_actions"][-2:]
    ] == [2, 3]


def test_deep_research_requires_explicit_enable(tmp_path: Path) -> None:
    service, _, _ = runtime(tmp_path)
    result = service.run(
        WorkflowRequest(
            "research deeply",
            "default",
            1,
            workflow=WorkflowName.DEEP_RESEARCH,
            workspace_summary={"document_count": 2},
        )
    )
    assert result["answerable"] is False
    assert result["diagnostics"]["state"] == "refused"
    assert "recovering" in result["diagnostics"]["state_history"]


def test_bootstrap_limits_ingestion_and_resumes_original_question(tmp_path: Path) -> None:
    service, generator, calls = runtime(tmp_path)
    result = service.run(
        WorkflowRequest(
            "new topic question",
            "default",
            1,
            workflow=WorkflowName.RESEARCH_BOOTSTRAP,
            bootstrap_confirmed=True,
            workspace_summary={"document_count": 0},
        )
    )
    assert result["answerable"] is True
    assert result["workflow_details"]["resumed_original_question"] is True
    assert calls["literature.search"] == 1
    assert calls["literature.get_metadata"] == 2
    assert calls["literature.download"] == 1
    assert calls["document.parse"] == 1
    assert calls["knowledge.retrieve"] == 1
    assert generator.calls == 1
    assert result["workflow_details"]["resumed_scope_version"] == 2


def test_backend_failure_recovers_to_safe_refusal_with_trace(tmp_path: Path) -> None:
    service, generator, _ = runtime(tmp_path, options={"fail_retrieval": True})
    result = service.run(
        WorkflowRequest("question", "default", 1, workspace_summary={"document_count": 2})
    )
    assert result["answerable"] is False
    assert generator.calls == 0
    assert "recovering" in result["diagnostics"]["state_history"]
    assert [
        value["level"] for value in result["diagnostics"]["recovery_actions"]
    ] == [1, 2]
    assert result["diagnostics"]["state"] == "refused"


def test_generation_failure_preserves_selected_evidence_and_reason(tmp_path: Path) -> None:
    class BrokenGenerator(Generator):
        def generate(self, context: Any) -> Mapping[str, Any]:
            request = httpx.Request("POST", "https://example.test/v1/chat/completions")
            response = httpx.Response(
                400,
                request=request,
                json={"error": {"message": "invalid response format"}},
            )
            raise httpx.HTTPStatusError("bad request", request=request, response=response)

    values, _ = handlers()
    service = ResearchRuntime(
        build_default_gateway(values),
        BrokenGenerator(),
        state_store=ResearchStateStore(tmp_path),
        trace_store=RunTraceStore(tmp_path),
    )

    result = service.run(
        WorkflowRequest(
            "question",
            "default",
            1,
            workspace_summary={"document_count": 2},
        )
    )

    assert result["answerable"] is False
    assert len(result["selected_evidence"]) == 2
    assert result["coverage"]["sufficient"] is True
    assert result["workflow_details"]["generation_failure"] == {
        "stage": "answer",
        "failure_kind": "HTTPStatusError",
        "http_status": 400,
        "response_body": '{"error":{"message":"invalid response format"}}',
    }
    assert result["diagnostics"]["termination_reason"] == "generation_failed:HTTPStatusError"

    presented = HarnessAgentService(
        service,
        workspace_summary={"document_count": 2},
    )._present(  # type: ignore[attr-defined]
        result,
        session_id=None,
        topic_id=None,
        session_created=False,
        include_context=True,
    )
    assert presented["outcome"]["code"] == "generation_failed"
    assert presented["selected_evidence"]
    assert "不是证据不足" in presented["outcome"]["message"]


def test_transient_tool_failure_uses_level_one_retry() -> None:
    attempts = 0

    def transient(
        arguments: Mapping[str, Any], context: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("temporary")
        return {"results": []}

    gateway = build_default_gateway({"knowledge.retrieve": transient})
    harness = ResearchRunHarness("direct_qa", ResearchBudgetPolicy())
    harness.transition(HarnessState.CONTEXT_PREPARING)
    harness.transition(HarnessState.PLANNING)
    harness.transition(HarnessState.EXECUTING)
    gateway.invoke(
        "knowledge.retrieve",
        {"query": "q", "workspace_id": "default", "scope_version": 1},
        context={
            "workflow": "direct_qa",
            "workspace_id": "default",
            "scope_version": 1,
            "permissions": ["knowledge.read"],
        },
        harness=harness,
    )
    assert attempts == 2
    assert harness.recovery_actions == [
        {"level": 1, "action": "retry:knowledge.retrieve", "outcome": "retrying"}
    ]
    assert harness.state_history[-2:] == ["recovering", "executing"]


def test_no_evidence_never_generates_deterministic_claim(tmp_path: Path) -> None:
    service, generator, _ = runtime(tmp_path, options={"no_evidence": True})
    result = service.run(
        WorkflowRequest("question", "default", 1, workspace_summary={"document_count": 2})
    )
    assert result["answerable"] is False
    assert result["claims"] == []
    assert generator.calls == 0


def test_claim_validator_rejects_graph_inference_as_direct_fact() -> None:
    validator = ClaimEvidenceValidator()
    report = validator.validate(
        {"answerable": True, "claims": [{"claim_id": "C1", "text": "A proves B.", "evidence_ids": ["G1"]}]},
        [evidence("G1", directness="inferred", grade="graph_inference")],
    )
    assert report.valid is False
    assert report.issues[0].code == "graph_inference_as_fact"
    assert validator.deterministic_repair({"claims": [{"claim_id": "C1"}]}, report)["answerable"] is False


def test_long_term_memory_accepts_only_verified_whitelist(tmp_path: Path) -> None:
    store = ResearchStateStore(tmp_path)
    store.commit_workspace("default", {"verified_conclusions": [{"claim_id": "C1"}]})
    assert store.workspace("default")["verified_conclusions"] == [{"claim_id": "C1"}]
    with pytest.raises(ValueError):
        store.commit_workspace("default", {"hidden_reasoning": "secret"})


def test_harness_agent_service_preserves_external_answer_and_session_shape(
    tmp_path: Path,
) -> None:
    service, _, _ = runtime(tmp_path)
    sessions = AgenticSessionStore(tmp_path)
    facade = HarnessAgentService(
        service,
        workspace_summary={"document_count": 2},
        session_store=sessions,
    )
    result = facade.answer("What is measured?", include_context=True)
    assert result["schema_version"] == "1.0"
    assert result["answerable"] is True
    assert result["answer"].endswith("[S1]")
    assert result["citations"][0]["evidence_id"] == "E1"
    assert result["outcome"]["code"] == "answered"
    assert result["diagnostics"]["retrieval_mode"] == "research_harness"
    assert result["context"]["selected_evidence"]
    session_id = result["session"]["session_id"]
    topic_id = result["session"]["topic_id"]
    assert [
        value["event_type"] for value in sessions.list_events(session_id, topic_id)
    ] == ["user_query", "evidence_added", "answer", "validation", "state_delta"]
    assert sessions.list_evidence(session_id, topic_id)


def test_research_agent_modules_have_no_concrete_backend_dependency() -> None:
    forbidden = (
        "is_" + "graphrag",
        "light" + "rag",
        "qdrant",
        "neo4j",
        "canonical" + "corpus",
        "data/index",
    )
    root = Path(__file__).parents[1] / "app" / "research"
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(root.glob("*.py"))
        if path.name not in {"__init__.py"}
    ).casefold()
    assert all(value not in source for value in forbidden)


def test_stage3_workflow_cost_baseline_matches_deterministic_fixture(
    tmp_path: Path,
) -> None:
    baseline = json.loads(
        (Path(__file__).parent / "baselines" / "stage3_workflow_cost_baseline.json").read_text(
            encoding="utf-8"
        )
    )
    direct, _, _ = runtime(tmp_path / "direct")
    direct_result = direct.run(
        WorkflowRequest(
            "What is measured?",
            "default",
            1,
            workspace_summary={"document_count": 2},
        )
    )
    relation, _, _ = runtime(
        tmp_path / "relation",
        options={"relation_gap": True},
    )
    relation_result = relation.run(
        WorkflowRequest(
            "How are A and B related?",
            "default",
            1,
            workflow=WorkflowName.RELATION_REASONING,
            workspace_summary={"document_count": 2},
        )
    )
    assert direct_result["diagnostics"]["policy"] == baseline["direct_qa"]["policy"]
    assert direct_result["diagnostics"]["usage"] == baseline["direct_qa"]["fixture_usage"]
    assert relation_result["diagnostics"]["policy"] == baseline["relation_reasoning_with_gap"]["policy"]
    assert relation_result["diagnostics"]["usage"] == baseline["relation_reasoning_with_gap"]["fixture_usage"]
