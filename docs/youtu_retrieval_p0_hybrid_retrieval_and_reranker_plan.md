# Youtu-GraphRAG P0 实施文档索引：Hybrid Retrieval 与 Chunk Reranker

当前原始合并文档已拆分为两份独立实施文档：

- [youtu_retrieval_p0_hybrid_retrieval_plan.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/youtu_retrieval_p0_hybrid_retrieval_plan.md)
  - 聚焦 `Hybrid first-stage retrieval`
  - 包含：
    - `BM25 / sparse + dense + RRF`
    - `retrieval_requirement` 元数据如何接入 query planning
    - Hybrid 路径的延迟影响与控制策略

- [youtu_retrieval_p0_chunk_reranker_plan.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/youtu_retrieval_p0_chunk_reranker_plan.md)
  - 聚焦 `Stronger chunk-level reranker`
  - 包含：
    - strong rerank 技术路线
    - `retrieval_requirement` 元数据如何接入 rerank 特征
    - reranker 路径的延迟影响与控制策略

保留本文件作为索引页，避免后续已有引用失效，同时明确两个 `P0` 任务已独立管理。
