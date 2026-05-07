# Youtu-GraphRAG P0 实施文档：Hybrid Retrieval

## 1. 文档目的

本文档只聚焦当前检索系统的第一个 `P0` 任务：

- `Hybrid first-stage retrieval`
  - 引入 `BM25 / sparse + dense + RRF`
  - 解决当前 dense-only 路径对 `Section 504`、`10.5.2`、定义术语、notice address 等精确锚点命中不稳的问题

本文档直接面向当前项目 [backend.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/backend.py) 和 [enhanced_kt_retriever.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/models/retriever/enhanced_kt_retriever.py) 的落地实施。

## 2. 当前问题概括

结合 [youtu_retrieval_requirements_smoke_analysis_and_optimization.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/youtu_retrieval_requirements_smoke_analysis_and_optimization.md) 与最近几轮 `retrieval_requirements_smoke_eval` 的结果，当前 first-stage retrieval 的短板主要是：

- `multi_hop_specific` 的 `context_recall` 已开始提升，但很多场景仍然无法稳定把 gold chunk 拉进候选池
- `Section / Article / defined term` 型 query 对精确 lexical anchor 仍然不够敏感
- `local` 字段型问题已有改善，但 clause-level 职责、角色、定义类问题仍不稳定

因此，后续最值得优先投入的是：

- 强化 first-stage retrieval 的覆盖与精确锚点能力

## 3. 目标

把当前以 dense / graph path 为主的 first-stage retrieval，升级为：

- dense retrieval
- sparse lexical retrieval
- 两者融合排序

核心目的是补上当前 dense-only 对下列场景的弱点：

- `Section 504`
- `Section 10.5.2`
- `Sensitive Customer Information`
- `Privacy Regulations`
- `notice address`
- `payment amount`
- `effective date`

## 4. 当前代码入口

优先关注以下位置：

- [enhanced_kt_retriever.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/models/retriever/enhanced_kt_retriever.py#L3527)
  - `_chunk_embedding_retrieval()`
- [enhanced_kt_retriever.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/models/retriever/enhanced_kt_retriever.py#L2304)
  - `process_retrieval_results()`
- [backend.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/backend.py#L2520)
  - `_merge_retrieval_payloads()`

## 5. 实施方案

这一节不是两条并列路线，而是同一条 `Hybrid Retrieval` 路线的两个实施阶段：

- `阶段 1`
  - 先做最小可行验证，确认 `sparse + dense + RRF` 对当前 smoke case 是否有效
- `阶段 2`
  - 在阶段 1 验证有效后，把 first-stage fusion 做成正式工程能力，补齐配置、trace 和可观测性

### 5.1 阶段 1：最小可行验证

先不改现有 FAISS 主体，只增补一条 sparse 检索路径：

- 基于 chunk 文本构建 `BM25` 索引
- query 使用当前已经改进过的 `retrieval_queries`
- sparse top-k 与 dense top-k 分别召回
- 用 `RRF` 融合成统一候选列表

这个阶段的核心目标是先验证：

- `Section / defined term / notice address` 类问题是否更容易把精确 chunk 捞进候选池
- 当前 first-stage retrieval 的主要短板，是否确实能通过 lexical + semantic 互补得到缓解

当前实施状态：

- 已完成第一版代码接入
- 具体实现为：
  - 保留原有 dense chunk retrieval
  - 新增基于 chunk 文本的 sparse lexical retrieval
  - 在 `path1.chunk_results` 上用 `RRF` 融合 dense 与 sparse 候选
  - 输出兼容现有 `chunk_ids / scores / chunk_contents`，并额外保留 dense/sparse 来源信息供后续 trace 使用
- 当前还未完成 smoke eval 验证，阶段 1 是否有效仍以评测结果为准

### 5.2 阶段 2：工程化 first-stage fusion

在阶段 1 验证有效后，在 retriever 内新增显式 first-stage fusion 能力：

- `dense_top_k`
- `sparse_top_k`
- `rrf_k`
- 输出：
  - 融合后的 `chunk_ids`
  - 每路来源 trace
  - 最终融合得分

这样后续可以直接在 trace 里看：

- 某个 gold chunk 是 dense 命中的
- 还是 sparse 命中的
- 还是两者都命中

这一阶段的重点不再是“证明路线可行”，而是：

- 把 fusion 结果接入现有检索主链路
- 保证 smoke eval、trace 复盘、后续 reranker 接口都能稳定复用这套 first-stage 输出

## 6. 推荐实现步骤

1. 新增 `SparseChunkRetriever`
   - 输入：全部 chunk 文本
   - 输出：`chunk_ids + scores`
2. 在 `KTRetriever` 中加入 sparse cache / index 生命周期
3. 在 `_process_chunk_results()` 之前，合并：
   - dense candidates
   - sparse candidates
4. 使用 `RRF` 得到统一候选
5. 保留 trace 字段，便于后续 smoke case 对照

## 7. `retrieval_requirement` 元数据如何接入 Hybrid Retrieval

当前系统在问题拆解阶段，已经会先生成结构化的 `retrieval_requirement`，包含：

- `route_type`
- `intent`
- `entities`
- `terms`
- `anchors`
- `query_keywords`
- `target_patterns`
- `left_endpoint`
- `right_endpoint`
- `bridge_relation`
- `scope`
- `themes`

这些元数据不应该只用于生成一条自然语言 `sub-question`，而应该直接参与 `Hybrid Retrieval` 的 query planning。

### 7.1 用于生成多路 query

同一个 requirement，应拆成多类 query，而不是只发一条 query：

- `anchors`
  - 主要喂给 `sparse / BM25`
  - 例如：
    - `Section 504`
    - `Section 10.5.2`
- `entities / terms / query_keywords`
  - 同时喂给 `dense` 与 `sparse`
  - 例如：
    - `Privacy Regulations`
    - `Sensitive Customer Information`
- `target_patterns / intent`
  - 用于决定 query 模板
  - 例如：
    - `what clause defines Sensitive Customer Information`
    - `which clause connects Section 504 and Section 10.5.2`

也就是说，Hybrid Retrieval 的输入不应只是：

- `retrieval_queries`

而应扩展成：

- `anchor_queries`
- `term_queries`
- `bridge_queries`
- `compiled_retrieval_queries`

### 7.2 元数据与检索路径的对应关系

推荐关系如下：

- `anchors`
  - 优先用于 `sparse`
  - 因为 section 编号、article 编号、defined term anchor 更适合 lexical exact match
- `entities / terms / query_keywords`
  - 同时用于 `dense + sparse`
  - 因为它们既有语义信息，也常常是重要 lexical phrase
- `left_endpoint / right_endpoint / bridge_relation`
  - 不建议原样直接拼成长 sparse query
  - 更适合生成：
    - 简短 bridge query
    - rerank bias 特征
- `intent`
  - 影响 query 模板与 rerank 优先级
  - 例如：
    - `definition_lookup` 更看重 `defines / means / refers to`
    - `fact_lookup` 更看重 `address / date / amount / notice`
    - `bridge_lookup` 更看重双 anchor 同现

### 7.3 用于后续 rerank bias

这些元数据不只应用于 first-stage query planning，还应直接进入后续 rerank：

- 命中 `anchors`
  - 强加分
- 命中 `terms / query_keywords`
  - 中强度加分
- 同时命中 `left_endpoint + right_endpoint`
  - 结构桥接加分
- 命中 `bridge_relation` 对应语义
  - 额外加分

因此，`retrieval_requirement` 应成为：

- first-stage query 生成的输入
- rerank score 的显式特征来源

而不是只作为调试信息保留在 trace 中。

## 8. 延迟影响与控制策略

单独看 `Hybrid Retrieval`，延迟通常只会有 `小到中等` 增量，不应成为主要风险。

原因：

- `BM25 / sparse` 检索本身通常较快
- `RRF` 融合代价几乎可以忽略
- 只要 chunk 数量不是极端大，Hybrid Retrieval 不会像强 reranker 一样显著拉高延迟

建议控制方式：

- dense 与 sparse 分别限制 top-k
- 只在 sub-question 级别做小候选池融合
- 保留 trace，但不要引入过重的在线解释逻辑

## 9. 验收标准

优先关注以下样本：

- `0043`
- `0042`
- `0002`
- `0003`
- `0001`

判定标准：

- `Section / term` 样本的 `context_recall` 提升
- 不明显恶化 `context_precision`
- 子问题 trace 中能看到 gold chunk 至少进入 first-stage 候选池

## 10. 阶段结论

当前系统已经证明：

- 图清洗不是主要方向
- query 编译与 local-first 修正已经有效
- 下一步真正最该投入的 first-stage 优化，就是：
  - `Hybrid retrieval`

它的核心目标非常明确：

- 让 gold chunk 更稳定地进入候选池
- 让 exact lexical anchor 与 dense semantic 召回互补

这也是当前从“相关但不精确”走向“稳定命中精确证据”的第一步。
