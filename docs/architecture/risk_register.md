# 风险登记

| ID | 风险与证据 | 影响 | 阶段一处置 | 阶段二控制 |
|---|---|---|---|---|
| R1 | Web 上传只更新旧 BM25/Dense；`LocalRAGWebRuntime.parse_pdf` 不调用 `KnowledgeSyncService` | CLI GraphRAG 与 Web corpus 不一致 | 文档化并冻结，不改 Web | 统一 index lifecycle 后再切 Web |
| R2 | CLI parse、batch、Web 的自动下游步骤不同 | 新论文可处于部分可检索状态 | 生命周期矩阵记录 | 统一 `IndexGeneration` 状态机 |
| R3 | 无论文删除入口；底层 diff 删除不等于 canonical 删除 | 数据残留或误报删除成功 | 当前基线标“不支持” | 先 Workspace 解绑，再索引失效，物理删除单独审批 |
| R4 | `JobManager._jobs` 仅内存 | 进程重启后 job_id 丢失 | 新增重启基线测试 | 设计独立持久 Job repository；不改现有 schema |
| R5 | `AgenticRAGService` 多处判断 `is_graphrag` | Agent/Harness 耦合后端 | 新 Protocol + Adapter，不切业务 | 由 Unified Knowledge Service 吸收 capability |
| R6 | 检索 Evidence 是自由 dict；Graph 和 legacy 字段不完全一致 | 引用映射错误、运行期 KeyError | 新 Candidate/Verified/Selected 契约 | Canonical 回查验证后才选入 Context |
| R7 | `LegacyEvidenceMapper` 当前只能检查 locator 字段存在 | “Verified”强度不足 | 明确 `verification_method=legacy_locator_fields` | Corpus repository 校验内容 hash、页和 block |
| R8 | `chunk_id` 含切分序号 | 更新后证据复用错位 | 记录 stable_chunk_key 的现有作用 | 外部引用同时固定 generation + stable key + locator |
| R9 | GraphRAG activation 跨 SQLite/Qdrant/Neo4j，不是真正分布式事务 | 部分后端成功、epoch 失败后残留 | 旧 active epoch 保持读取；不改实现 | 幂等 replay、失败 generation 清理与计数验证 |
| R10 | Community cache 按 fingerprint，报告 LLM/model 参数未全部进入 fingerprint | 复用过期报告 | 待确认，不修改 | 把 report model/prompt/version 纳入 generation metadata/cache key |
| R11 | 当前 Graph registry 无 active epoch（本地审计快照） | Graph/关系在线基线不能在本机端到端运行 | 标记待确认；保留 core/integration tests | 在受控 Neo4j/Qdrant 环境生成可版本化图查询快照 |
| R12 | LLM 调用次数和 Token usage 未统一持久化；Web Job event 也不含 | 成本/预算基线不完整 | `AgentRun` 用 `None`，不填 0 | provider gateway 统一累计 usage/call count |
| R13 | 本地 session 历史 8 次 validation 中结构有效率 0.625，但样本含 fail-closed 拒答 | “引用完整率”容易误读 | 同时报告口径和样本量 | 分 answered/refused、claim-level、run-level 指标 |
| R14 | `IndexRegistryStore` 在构造时创建 SQLite 文件/Schema | 只读查询可能产生本地状态 | 阶段一无调用路径变化 | factory 区分 read-only/open-or-create |
| R15 | 新 Workspace 契约尚无持久化/授权来源 | Adapter 只能携带、不能真正隔离 | 不提供默认 workspace；不切外部 API | 先实现 scope repository，再开放入口 |
| R16 | 现有 evaluation baseline 在 `data/evaluation` 下被 gitignore | 团队无法仅靠 Git 复现全量 Top-K | 提交摘要、SHA-256 和代表性 Top-10 | CI 生成/归档完整 baseline artifact |
| R17 | Q001 LightRAG Recall@10=1.0，但 nDCG@10=0.630930，低于 legacy 1.0 | 过早切换可能降低首位证据质量 | 保持 legacy official，LightRAG 仅 shadow | 完成 21 题校准与 acceptance gate 后再切换 |
| R18 | LightRAG cold query 需要 keyword LLM；两题实测 1,433 tokens | 查询延迟和成本高于纯 legacy RRF | 记录 cold/warm cache 两种口径 | 阶段三评估 cache miss、预算和 provider 单价 |
| R19 | active corpus 尚未做真实破坏性删除演练 | 删除残留指标只有单元测试，无生产快照 | 不开放删除 API，不物理删除 canonical | forked generation 执行残留验收 |
| R20 | Web 已收口到 `KnowledgeIndexService`，但为保持 legacy 正式回答兼容仍会重建旧 BM25/Dense，且未自动推进 LightRAG generation | “新增论文不全库重建”目前只对 LightRAG 增量 API 成立 | 不虚构阶段二全局完成，不切正式引擎 | 阶段三把 Web add/update 接到 LightRAG incremental generation，再退役旧全量 serving 路径 |

## 待确认问题

1. Workspace 的创建者、共享/只读权限和 scope_version 递增规则。
2. 元数据更新是否应创建新 corpus version，还是只重建受影响投影。
3. Web Job 重启后 running job 应标 failed、queued 重放，还是人工恢复。
4. 物理删除的保留期、审计要求和 raw/parsed/canonical 备份策略。
5. GraphRAG 生产环境 Neo4j/Qdrant 的当前计数、延迟和关系问答 Top-K。
6. LLM 调用的“调用次数”是否包含结构修复、coverage、semantic validation 和 community report。
