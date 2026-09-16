# 风险登记

| 风险 | 影响 | 当前控制 |
|---|---|---|
| Canonical 变化未被识别 | 旧 vector 内容过期 | source hash、structure fingerprint、per-paper digest 三重校验 |
| Chunk policy 漂移 | 相同文本产生不同切分 | `CHUNK_POLICY_VERSION` 写入 manifest，变化触发 rebuild |
| Embedding revision 漂移 | vector 不可比较 | model/revision/artifact/text policy manifest 门禁 |
| 局部 chunk_id 重名 | Qdrant Point 覆盖 | Point ID 使用 `paper_id:chunk_id` |
| 删除分页漏删 | 旧 Paper 仍被召回 | 删除每一页后从 filtered head 重新扫描，并做 payload 审计 |
| BM25 全局 IDF 更新过重 | 误触发 Dense 重算 | lexical lifecycle 与 Dense lifecycle 分离 |
| 全局首阶段召回粒度错误 | 跨论文噪声和成本升高 | Paper-level RRF 先选 `paper_id`，Content stage 强制 filter |
| Evidence locator 失效 | 引用幻觉 | Canonical 回查、content hash、fail closed |
| Agent 越权访问存储 | 绕过治理 | CrewAI 只接收 capability adapter 和 structured contract |
| 旧 manifest 无 per-paper digest | 无法证明增量安全 | 拒绝静默重建，要求管理员显式 force rebuild |
