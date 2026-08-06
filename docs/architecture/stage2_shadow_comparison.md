# 阶段二新旧检索对比

完整机器可读结果位于 `data/evaluation/stage2_shadow_comparison.json`，由 [`scripts/evaluate_stage2_shadow.py`](../../scripts/evaluate_stage2_shadow.py) 生成。正式引擎是 legacy RRF，LightRAG generation `IG_419c4e228f0dd1a8` 只运行 shadow。验收阈值为 graph backfill >= 0.95 且跨 Workspace 泄漏为 0。

## Q001 代表题

问题：`When ground-truth satellite ephemerides are unavailable, what target is used to train the orbit-prediction neural network?`

qrel：`D_060e764f208c_cp02_c000012`。

| 指标 | Legacy RRF | LightRAG + Evidence Intelligence |
|---|---:|---:|
| Recall@10 | 1.0 | 1.0 |
| nDCG@10 | 1.0 | 0.630930 |
| qrel rank | 1 | 2 |
| 最近一次 warm query latency | 139.6 ms | 1367.7 ms |
| 图候选 / selected relation evidence | 不适用 | 19 / 3 |
| canonical graph backfill | 不适用 | 1.0 |
| selected graph_inference | 不适用 | 0 |
| cross-workspace leakage | 0 | 0 |

LightRAG 保持 Recall，但排序低于 legacy；这是已记录的 shadow 退化，不能据此切换正式回答。修正前 Chunk/Entity/Relation 原始分数被错误地直接比较，导致 Chunk 被 Relation 挤出；[`LightRAGKnowledgeEngine.retrieve_candidates`](../../app/knowledge_engine/lightrag_engine.py) 现按来源保留候选，并在 [`EvidenceIntelligencePipeline.select`](../../app/evidence/service.py) 为直接 canonical Chunk 和图关系分别保留席位。

## REL001 关系题

问题：`How are LEO-NNPON, SGP4, and TLE data related when training an orbit-prediction neural network without ground-truth satellite ephemerides?`

本题尚无人工 qrel，因此不填写 Recall/nDCG。LightRAG 返回 20 个通过 scope 的图候选，selected 中有 4 条 relation evidence；20/20 已验证图候选均能回填 canonical 原文，回填率 1.0；selected `graph_inference=0`，跨 Workspace 泄漏为 0。没有 qrel 的指标明确标记 `pending_annotation`，不以 0 冒充已测量。

## 查询 Token 与成本

两条 cold query 共 2 次 LLM keyword 调用，1,334 prompt tokens、99 completion tokens、1,433 total tokens；Embedding 调用 2 次、文本 6 条。随后相同查询命中 LightRAG LLM cache，LLM 调用和 Token 均为 0，最近一次两题总耗时 2.785 s。Legacy RRF 查询不调用 LLM。未配置单价，所以货币成本为“待确认”，不是 0。

## 结论

预设关系回填与隔离门槛通过，Q001 Recall 未退化；但只测了 1 个带 qrel 的代表题，且 nDCG 低于 legacy。正式回答继续由 legacy 服务。切换门槛还需要阶段三完成 21 题 shadow 套件、关系题人工 qrel、删除残留和 active-answer rollback smoke。
