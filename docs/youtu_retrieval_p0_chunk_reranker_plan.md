# Youtu-GraphRAG P0 实施文档：Stronger Chunk Reranker

## 1. 文档目的

本文档只聚焦当前检索系统的第二个 `P0` 任务：

- `Stronger chunk-level reranker`
  - 在第一阶段召回候选之上再加一层更强的 chunk 级重排
  - 重点解决当前最核心的 `right area, wrong chunk`

本文档直接面向当前项目 [backend.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/backend.py) 和 [enhanced_kt_retriever.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/models/retriever/enhanced_kt_retriever.py) 的落地实施。

## 2. 当前问题概括

结合 [youtu_retrieval_requirements_smoke_analysis_and_optimization.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/youtu_retrieval_requirements_smoke_analysis_and_optimization.md) 与最近几轮 `retrieval_requirements_smoke_eval` 的结果，当前 second-stage selection 的短板非常明确：

- `multi_hop_specific` 的 `context_recall` 已开始提升，但 `context_precision` 仍然很低
- 这说明系统很多时候已经打到了“相关区域”，但最终没有把 gold chunk 排到前面
- 当前 `_rerank_chunks_by_relevance()` 仍是轻量启发式重排，不属于强 reranker

因此，后续最值得优先投入的是：

- 强化 second-stage reranker 的 chunk 选择能力
- 但前提是 `chunk first-stage` 链路必须先真实参与检索

## 2.1 基于 `t5` 的路线修正

结合 `retrieval_requirements_smoke_eval_t5` 的结果，当前需要把 `strong chunk reranker` 的定位说得更精确。

`t5` 说明：

- `multi_hop_abstract` 已经是当前表现最均衡的一类
- `single_hop_specific` 的 recall 也出现明显改善
- 真正没有解决的，是 `multi_hop_specific` 的 exact chunk / exact clause 命中

这意味着：

1. `strong chunk reranker` 不是为了“证明系统终于会做多跳”
   - 因为抽象型多跳已经说明系统具备一定多跳能力
2. 它的主要职责是：
   - 把已经召回到“相关区域”的候选，进一步压到更精确的 gold chunk
3. 它最该优先服务的是：
   - `single_hop_specific`
   - `multi_hop_specific`

也就是说，后续验收 reranker 是否成功，不能主要看 `abstract` 的 broad evidence collection，而要看 specific 的 exact clause precision 是否真正提升。

## 3. 目标

解决当前系统最核心的短板：

- `right area, wrong chunk`

也就是：

- first-stage 已经把相关文档、相关 section、相关 chunk 区域找到了
- 但最终排到前面的 chunk 仍不是 gold chunk

## 3.1 作用域

`Chunk reranker` 的作用域需要明确限定为：

- 它只作用于 `chunk candidate` 的排序
- 它的输入是 first-stage retrieval 已经召回的 chunk 候选池
- 它的输出是更可靠的 chunk 排名与 chunk 分数

它当前要解决的是：

- 哪些 chunk 应该进入 `support_chunk_ids`
- 哪些 chunk 应该优先进入最终上下文
- 哪些新增召回的 chunk 实际上只是噪声，应被压下去

它不直接负责：

- triple 检索或 triple 排序
- community/global 路由本身的召回策略
- 最终回答 prompt 的模板设计
- final answer 的语言生成

也就是说，`Chunk reranker` 是：

- `first-stage retrieval` 与 `sub-question / final answer evidence use` 之间的一层 chunk 级选择器

不是：

- 一个全链路 reranker
- 一个直接改写最终回答的模块

因此，后续实施与验收都应围绕以下链路来理解：

- first-stage retrieval
- chunk reranker
- support chunk selection
- final chunk selection

这里还需要强调一个前置条件：

- 如果 `first_stage_chunk_ids` 为空
- 或 `lightweight_reranked_chunk_ids` 为空

那么 `strong chunk reranker` 就没有稳定的输入候选池。  
在这种情况下，即使强行接入 reranker，也只能对后面混合出来的 chunk 列表做补救，无法准确评估它对 chunk-first 主链路的真实贡献。

而不是把它扩展理解为对 triples、IRCoT reasoning 或最终回答生成的全面替代。

## 4. 当前代码现状

当前 chunk 重排在 [enhanced_kt_retriever.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/models/retriever/enhanced_kt_retriever.py#L3632) 的 `_rerank_chunks_by_relevance()`。

当前逻辑大致是：

- query embedding vs chunk embedding cosine similarity
- 与初始 FAISS score 平均
- 加 lexical overlap bonus
- 加 doc consistency bias

这属于轻量启发式重排，不足以解决：

- `Section 504` 与 `Section 10.5.2` 这种编号锚点
- `Sensitive Customer Information` 这类定义短语
- 长 chunk 中只有一句关键、其余内容大量无关的情况

## 5. 推荐技术路线

### 5.1 路线 1：Cross-Encoder Reranker

最直接、工程成本最低的增强方式：

- 对 top `20-50` chunk 候选
- 输入 `(query, chunk)` 对
- 输出更细粒度的 relevance score

优点：

- 最容易落地
- 对 exact clause / exact phrase 匹配更敏感

缺点：

- 推理成本更高

### 5.2 路线 2：Late Interaction Reranker

例如 `ColBERT` 风格：

- query token 与 chunk token 做细粒度交互
- 更适合术语、section、定义词、短语匹配

优点：

- 比单向量 cosine 更适合合同场景

缺点：

- 实现复杂度高于 cross-encoder

## 6. 建议的实际推进顺序

建议先做：

1. 先修 `chunk first-stage` 真正生效
2. 保留当前 `_rerank_chunks_by_relevance()` 作为 lightweight rerank
3. 在 top `N` 候选上叠一层 `strong rerank`
4. 先实现可插拔接口，再决定底层模型

也就是说，推荐架构是：

- first-stage retrieval
- lightweight rerank
- strong rerank
- final chunk selection

这里的 `strong rerank` 仍然只针对 `chunk`，不改变 triple 支线的检索逻辑。

## 7. 代码落点

优先改动：

- [enhanced_kt_retriever.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/models/retriever/enhanced_kt_retriever.py#L3632)
  - `_rerank_chunks_by_relevance()`
- [backend.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/backend.py#L3043)
  - `_build_support_pairs()`
- [backend.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/backend.py#L330)
  - 子问题 support chunk 相关逻辑

建议新增：

- `StrongChunkReranker` 接口
- `reranker.trace` 输出
- 每个 chunk 的分数拆解：
  - dense score
  - sparse score
  - lexical score
  - strong rerank score

## 8. 元数据如何接入 Strong Reranker

`retrieval_requirement` 元数据在强 reranker 中也应该直接参与打分，而不是只使用 `(query, chunk)` 文本对。

推荐新增的 rerank 特征：

- `anchor_hit`
  - chunk 是否命中 `Section / Article / clause anchor`
- `term_hit`
  - chunk 是否命中关键 defined term / phrase
- `endpoint_coverage`
  - chunk 是否同时覆盖 `left_endpoint / right_endpoint`
- `bridge_relation_hit`
  - chunk 是否出现与 `bridge_relation` 相符的 clause 表达
- `intent_alignment`
  - chunk 是否更像：
    - `definition`
    - `fact`
    - `obligation`
    - `bridge`

也就是说，强 reranker 不是只看：

- semantic similarity

而应该显式看：

- requirement metadata 与 chunk 的对齐程度

## 9. 延迟影响与控制策略

`Chunk reranker` 才是更可能显著增加延迟的部分。

原因：

- cross-encoder / late interaction 都比当前 lightweight rerank 重
- 若对每个子问题都无差别 rerank 大候选池，延迟会上升明显

建议控制方式：

- 只 rerank top `20-50`
- `local` 简单问题只 rerank top `10-20`
- `structural / multi-hop specific / low-confidence` 再放大到 top `30-50`
- 高置信 local 问题可以直接停在 lightweight rerank

因此，推荐策略不是“全量强 rerank”，而是：

- 小候选池
- 条件触发
- 分 route 限流

## 10. 验收标准

这项优化的主要目标不是 broad recall，而是 exact precision。

重点验收：

- `multi_hop_specific_query_synthesizer`
  - `context_precision` 不能长期停留在 `0.0`
- `single_hop_specific_query_synthesizer`
  - 在已有 recall 的前提下，gold chunk 排名应更稳定
- `0002 / 0003 / 0042 / 0043`
  - 若 first-stage 已进入正确区域，则最终 gold chunk 应更容易进 `support_chunk_ids`

固定观察指标：

- `first_stage_chunk_hit / recall`
- `lightweight_reranked_chunk_hit / recall`
- `context_precision`
- `support_chunk_ids` 是否覆盖 gold chunk
- 子问题 trace 中 rerank 前后 chunk 排名变化

同时应明确避免误判：

- 如果 `context_recall` 上升但 `support_chunk_ids` 仍未覆盖 gold chunk，说明 reranker 仍未达标
- 如果答案文本看起来更顺，但 chunk 排名没有改善，不应算作 chunk reranker 成功

## 11. 阶段结论

当前系统已经证明：

- query 编译与 local-first 修正已经有效
- 但 exact chunk selection 仍然不够强
- 下一步真正最该投入的 second-stage 优化，就是：
  - `Stronger chunk reranker`

它的核心目标非常明确：

- 让 gold chunk 更稳定地排进最终上下文
- 把系统从“相关但不精确”推进到“精确命中关键证据”

## 12. 当前实施进展

第一版 `strong chunk reranker` 已经落地，但当前仍是 `heuristic strong rerank`，还不是 cross-encoder。

当前实现特征：

- 插入位置：
  - `lightweight rerank` 之后
  - `support_chunk_ids` 之前
- 当前输入：
  - `support_pairs`
  - `retrieval_requirement`
  - `target_doc_id`
  - `chunk_id_to_doc_id`
- 当前显式特征：
  - `target_doc match`
  - `anchor_hit`
  - `term_hit`
  - `endpoint_coverage`
  - `bridge_relation` 词面命中
  - `intent_alignment`
  - `anchor coverage diversification`

当前新增 trace：

- `strong_reranked_chunk_ids`

当前新增评估指标：

- `strong_reranked_chunk_hit`
- `strong_reranked_chunk_precision`
- `strong_reranked_chunk_recall`

因此，后续阶段不再是“从 0 到 1 接入 strong reranker”，而是：

1. 先用这版 heuristic reranker 验证 `support top-2` 是否改善
2. 如果 specific 样本仍然卡住，再升级到底层更强的模型型 reranker
