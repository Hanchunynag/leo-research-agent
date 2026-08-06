# 服务契约

Protocol 定义位于 [`app/contracts/protocols.py`](../../app/contracts/protocols.py)。阶段一建立接口；阶段二扩展并实现接口，同时保留阶段一兼容方法。

## KnowledgeEngine

```python
retrieve(request: EvidenceRequest) -> Sequence[CandidateEvidence]
index_documents(documents, generation=..., profile=...) -> Mapping
update_documents(documents, generation=..., profile=...) -> Mapping
delete_documents(document_ids, generation=...) -> Mapping
retrieve_candidates(request) -> Sequence[CandidateEvidence]
get_status() -> Mapping
```

上层只传领域请求，不能传 QdrantClient、Neo4j driver、FTS connection、collection 名或 `is_graphrag`。实现必须遵守 workspace/scope，返回可审计 retrieval_source 与后端 metadata。

阶段一 `LegacyKnowledgeEngineAdapter` 包装任意暴露旧 `retrieve(query, **kwargs)` 的 runtime，保持旧对象不变。多 work/document scope 由 Adapter 在结果侧再次过滤；这只是兼容方案，不是最终 workspace 授权实现。

## EvidenceIntelligenceService

```python
verify(request, candidates) -> VerifiedEvidenceBundle
select(request, bundle) -> Sequence[SelectedEvidence]
```

verify 必须回查 canonical locator/content hash；select 必须在 verified 集合内执行覆盖、直接性、多样性和预算策略。不得让 Candidate 直接进入 Context。

`LegacyEvidenceMapper` 目前只完成旧 dict -> Candidate 和 locator 结构验证，不能声称完成 canonical 内容回查或语义验证。

## ContextBuilder

```python
build(request, evidence, token_budget=...) -> ContextPack
```

只能消费 SelectedEvidence；输出中固定 request/workspace/scope 和 token 诊断。目标实现将吸收 [`app/context/assembly.py`](../../app/context/assembly.py) 的渲染/截断逻辑。

## ToolGateway

```python
invoke(tool_name, arguments, context=...) -> Mapping[str, Any]
```

这是 Agent 使用工具的唯一入口。`context` 至少应包含 workspace/scope/run/budget；工具实现可委托 Unified Knowledge Service，但不得把具体存储客户端暴露给 Agent。

## ResearchHarness

```python
run(request, context_pack=None, state=None) -> AgentRun
```

Harness 只编排 Workflow/Step/Tool/Context/Budget/State/Evaluation/Recovery。阶段一 `LegacyAgentRuntimeAdapter.answer` 原样代理 `AgenticRAGService.answer`，`run` 再将旧 dict 投影为 `AgentRun`；旧 API 返回不变。

## 兼容与错误规则

- Adapter 不修改被包装对象，不写新数据库表。
- 无法映射的 Evidence 进入 rejected 列表，不能填造 locator。
- 旧实现没有统一 LLM call/token 指标时，`AgentRun` 使用 `None`，不能用 0 冒充已测量。
- 新契约的错误在边界处用 `ValueError` 表达输入不变量；旧 Web/CLI 错误映射保持原样。
- 所有协议使用结构化 typing `Protocol`，阶段二可逐个替换实现而不做全仓命名重写。

## 阶段二实现映射

- `KnowledgeEngine`：[`LightRAGKnowledgeEngine`](../../app/knowledge_engine/lightrag_engine.py)；旧实现由 [`LegacyKnowledgeEngineAdapter`](../../app/contracts/adapters.py) 包装。
- `EvidenceIntelligenceService`：[`EvidenceIntelligencePipeline`](../../app/evidence/service.py)。
- `ContextBuilder`：[`SelectedEvidenceContextBuilder`](../../app/evidence/context.py)，只接收 state=`selected`。
- `UnifiedKnowledgeService`：[`app/knowledge_engine/unified_service.py`](../../app/knowledge_engine/unified_service.py)，生产 Agent 不直接接触旧 Dense/Graph 存储。
- `ResearchHarness`：仍由现有 [`app/agentic/harness.py`](../../app/agentic/harness.py) 提供，本阶段没有重写。
