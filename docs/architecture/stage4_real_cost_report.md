# Retrieval and orchestration cost notes

成本采集位于 `app/evaluation/costs.py`，按 retrieval、verification、generation、review 和 approval 阶段记录调用数、tokens、延迟和可选价格。

索引成本与单次问答成本分离：新增或变化 Paper 的 BGE-M3 文本才计入 Dense indexing；BM25 全局 IDF 更新不计作旧文献 embedding。Paper stage 和 Content stage 的查询 embedding、RRF 候选、Cross-Encoder 精排分别记录，便于判断层级检索是否减少无效细粒度召回。

真实 provider 的价格、设备和网络环境由部署方配置；没有价格时保留 `null`，不以 0 伪造成本。
