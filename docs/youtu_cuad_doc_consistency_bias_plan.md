# Youtu CUAD Doc Consistency Bias Plan

本文档用于记录当前 `youtu-graphrag` 在 `CUAD` smoke 评测中的具体问题，以及后续引入 `doc-consistency bias` 的实施方向。

相关结果文件：

- `outputs/results/ragas/edge_layer_smoke_eval_t_ircot/ragas_eval_summary.csv`
- `outputs/results/ragas/edge_layer_smoke_eval_t_ircot/ragas_eval_per_sample.jsonl`
- `outputs/results/ragas/edge_layer_youtu_smoke_t_ircot.jsonl`

相关代码：

- `youtu-graphrag/backend.py`
- `youtu-graphrag/models/retriever/enhanced_kt_retriever.py`
- `src/experiments/run_ragas_cuad_compare.py`

## 1. 当前问题不是单一问题

当前 smoke 结果显示，不同题型的问题来源不同，不能用单一解释覆盖。

### 1.1 single-hop-specific

表现：

- `answer_correctness` 最低
- 但部分样本 `faithfulness` 很高

排查结论：

- 不只是回答风格问题
- 更主要是 `local` 子问题命中了“相关但不正确”的条款
- 系统给出了看起来合理、但和 gold 不一致的答案

典型样本：

- `ragas-cuad-0002`
  - gold 关注 Motorola 在 promotion 中作为 pager 供给方的角色
  - 当前回答偏成了 trademark/IP licensor
- `ragas-cuad-0003`
  - gold 关注 PageMaster promotion period 内的具体职责
  - 当前回答混入了其他职责和错误的 period

结论：

- `local` 路由也存在 chunk 污染问题
- 只是它暂时没有像 `structural` 一样表现为大量 `NOT_FOUND`

### 1.2 multi-hop-specific

表现：

- `context_precision` / `context_recall` 经常为 `0`
- `structural` 子问题仍频繁 `NOT_FOUND`

排查结论：

- 问题由两部分组成：
  - 跨文档污染
  - 第二跳证据丢失

典型样本：

- `ragas-cuad-0042`
  - gold 是 Metavante 合同中的 Section 8 + 10.3
  - retrieved contexts 混入大量 TouchStar Reseller Agreement 内容
  - 这属于明显的错文档污染
- `ragas-cuad-0041`
  - 第一跳 expenses 能答
  - 第二跳 indemnification 证据没有稳定保住
- `ragas-cuad-0043`
  - 两个定义都能答
  - 连接关系仍答不出

结论：

- `structural` 仍是当前主问题
- 但问题不只是关系桥弱，也包括最终 evidence 明显错合同

### 1.3 multi-hop-abstract

表现：

- 本轮改造后提升最明显
- `answer_correctness / faithfulness / context_recall` 均有明显回升

排查结论：

- 当前系统更擅长“基于多个局部 findings 做综合回答”
- abstract 类题天然更容易吃到这种收益

结论：

- 当前改造不是整体失败
- 而是对 abstract 题有正收益，对 precise clause-grounded 问题副作用更大

## 2. 当前三类子问题的检索特征

基于 `edge_layer_youtu_smoke_t_ircot.jsonl` 的子问题统计：

- `local`
  - `avg_retrieved_all ≈ 11`
  - `avg_selected_final ≈ 7.17`
  - `status = answered 12/12`
- `global`
  - 样本数很少
  - `retrieved_all = 10`
  - `selected_final = 5`
  - 当前样本为 `not_found`
- `structural`
  - `avg_retrieved_all ≈ 8.86`
  - `avg_selected_final ≈ 5.29`
  - `answered 3 / not_found 4`

解释：

- `local` 并不干净，拿的 chunk 其实也不少
- `global` 天然高噪声，但当前样本太少
- `structural` 不是因为 chunk 数量一定最大，而是因为拿到的 chunk 更不像能回答该子问题的桥接证据

## 3. 为什么不建议硬 doc scope

如果直接把检索限制成“只允许一个文档”，风险很大：

1. 如果主文档判错，会导致整题彻底失败
2. 多跳题本来就可能需要跨多个片段甚至附件
3. 当前系统的问题不是“没搜到别的文档”，而是“别的文档在最终上下文里权重过高”

因此不建议做：

- hard doc-only retrieval
- first-stage retrieval 直接只保留单文档

## 4. 建议方案：doc-consistency bias

目标不是禁止跨文档，而是让最终证据选择更偏向“同一主文档”。

### 4.1 核心原则

- 第一阶段召回不做硬限制
- 只在 rerank / final selection 阶段加入文档一致性偏好
- 同 doc chunk 提升分数
- 跨 doc chunk 降权，但不完全禁止

### 4.2 推荐分级

#### 软约束

- 同 doc chunk 加权
- 不同 doc chunk 降权
- 仍允许所有 chunk 进入候选

这是首选。

#### 半硬约束

- 最终 `top_k` 中要求大部分来自主 doc
- 允许少量 cross-doc chunk 作为补充

例如：

- 最终 10 个 chunk 中至少 7 个来自主 doc
- 最多 3 个来自其他 doc

#### 硬约束

- 仅允许主 doc

只适合极确定的单合同问答，不建议作为默认策略。

## 5. 实施位置

推荐只改后排，不改第一阶段召回。

### 5.1 子问题 chunk rerank

在 `process_retrieval_results()` 内的 chunk 重排阶段引入文档一致性分数：

- 同 doc prefix：加分
- 不同 doc prefix：降权

候选 doc 的来源可以使用：

- 问题中显式合同实体
- 当前子问题前几名 chunk 的主 doc
- 已保留的 support chunk 所属 doc

### 5.2 最终 chunk selection

在子问题或总问题最终选 10 个 chunk 时：

- 优先保留来自主 doc 的高分 chunk
- 跨 doc 只占少量配额

### 5.3 不改第一阶段召回

保留跨 doc 候选，避免误 scope 造成整题失败。

## 6. 与当前 edge_layer / subquestion IRCoT 的关系

### 6.1 edge_layer

已经解决了一部分结构边噪声问题。

但它不解决：

- 错文档污染
- local 命中相关但错误条款

### 6.2 subquestion IRCoT

当前已经能触发，并保留全过程。

但本轮 smoke 表明：

- IRCoT 已经改善了 follow-up query
- 但 follow-up retrieval 仍可能拿回错文档或泛相关证据

因此：

- 下一步不应先增加 IRCoT 轮数
- 应先给 follow-up retrieval 加 doc-consistency bias

## 7. 预期收益

## 7.1 配置落点

当前实现对应配置位于 `youtu-graphrag/config/base_config.yaml`：

```yaml
retrieval:
  doc_consistency:
    enabled: false
    rerank_same_doc_bonus: 0.12
    rerank_cross_doc_penalty: 0.03
    final_same_doc_bonus: 2.0
    preferred_doc_window: 6
    max_preferred_docs: 2
```

设计原则：

- 默认关闭，便于和旧结果做 A/B 对比
- 只做软偏置，不做硬过滤
- 同文档优先同时保留少量跨文档补充证据
## 8. 预期收益

### 8.1 single-hop-specific

预期改善：

- 减少相关但错误的条款抢占 top chunks
- 提升 `answer_correctness`
- 减少 `context_precision/context_recall = 0` 的情况

### 8.2 multi-hop-specific

预期改善：

- 减少错合同 chunk 进入最终 evidence
- 提升 `context_precision`
- 让 structural follow-up 更可能在同一合同内补齐第二跳

### 8.3 multi-hop-abstract

风险：

- 过强 doc 偏置可能损失跨文档综合能力

因此 abstract 场景应只用软约束，不建议过强半硬约束。

## 8. 验证指标

引入 doc-consistency bias 后，优先关注：

1. `single_hop_specific`
   - `answer_correctness`
   - `context_precision`
2. `multi_hop_specific`
   - `context_precision`
   - `context_recall`
   - `structural` 子问题 `NOT_FOUND` 比例
3. 单样本排查
   - `retrieved_contexts` 是否仍明显混入别的合同
   - `reasoning_trace.sub_question_answers[*]`
   - `reasoning_steps`

## 9. 结论

当前 smoke 结果说明：

- 问题不只在 `structural`
- `local` 也会被错误但相关的 chunk 带偏
- `multi-hop-specific` 的主要故障之一是错文档污染

因此，下一步最合理的方向不是硬 doc scope，而是：

- 在子问题 rerank 与最终 evidence selection 中加入 `doc-consistency bias`
- 先做软约束
- 不改第一阶段召回

这条路径比继续盲目缩 `top_k` 或直接增加 IRCoT 轮数更稳。
