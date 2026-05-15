# Youtu GraphRAG 检索效果分析与优化方案

本文档基于本仓库当前 `retrieval_requirements_smoke_eval` 基线评测结果、P0 修复后的 `retrieval_requirements_smoke_eval_t` 复测结果、后续持续迭代记录，以及最新全量 `vector_test` 对照评测结果，结合对 `youtu-graphrag` 检索相关代码与 trace 的排查，分析当前系统检索效果不佳的具体原因，并提出分阶段优化方案。

## 1. 分析范围

本次分析对应的数据与代码范围如下：

- 结果文件
  - `outputs/results/ragas/retrieval_requirements_smoke_eval/ragas_eval_summary.json`
  - `outputs/results/ragas/retrieval_requirements_smoke_eval/ragas_eval_summary.csv`
  - `outputs/results/ragas/retrieval_requirements_smoke_eval/ragas_eval_per_sample.jsonl`
  - `outputs/results/ragas/retrieval_requirements_smoke.jsonl`
  - `outputs/results/ragas/retrieval_requirements_smoke_t.jsonl`
  - `outputs/results/ragas/retrieval_requirements_smoke_eval_t/ragas_eval_summary.json`
  - `outputs/results/ragas/retrieval_requirements_smoke_eval_t/ragas_eval_per_sample.jsonl`
  - `outputs/results/ragas/vector_test/ragas_eval_summary.json`
  - `outputs/results/ragas/vector_test/ragas_eval_summary.csv`
  - `outputs/results/ragas/vector_test/ragas_eval_per_sample.jsonl`
- 核心代码
  - `youtu-graphrag/models/retriever/enhanced_kt_retriever.py`
  - `youtu-graphrag/models/retriever/faiss_filter.py`
  - `youtu-graphrag/models/retriever/agentic_decomposer.py`
  - `youtu-graphrag/models/constructor/kt_gen.py`
  - `youtu-graphrag/backend.py`
  - `youtu-graphrag/config/base_config.yaml`

文档结构说明：

- 第 `2` 到 `9` 节主要记录首轮 `retrieval_requirements_smoke_eval` 暴露出的基线问题与原始优化计划
- 第 `10` 节记录 P0 实施后的最新状态、已解决问题、未解决问题以及新增问题
- 第 `12.4` 节记录最新全量 `vector_test` 对照复盘：`refactor_v1` 相比 `vector_rag` 的退化原因与优化方案

## 2. 结论先行

当前 RAG 系统的检索效果不能评价为“优秀”，整体更接近：

- 单跳具体事实问题有一定可用性
- 多跳抽象问题表现一般
- 多跳具体问题存在明显系统性失败

更准确地说，当前系统的主要问题不是单一的“最终答案生成不好”，而是检索链路本身存在多处结构性缺陷，导致：

1. 子问题被编译成弱检索 query
2. 目标合同定位不稳，容易串到错误文档
3. chunk 合并和排序存在实现层面的错误
4. triple rerank 和图扩展存在确定性 bug 或设计缺陷
5. 图节点文本表征过薄，不利于合同类细粒度检索

## 3. 评测结果反映出的核心问题

### 3.1 整体指标不高

根据 `ragas_eval_summary.json`：

- `answer_correctness = 0.454818`
- `faithfulness = 0.419937`
- `context_precision = 0.264198`
- `context_recall = 0.336508`

对于检索系统来说，`context_precision` 和 `context_recall` 才是更直接的检索质量信号。当前 precision 和 recall 都停留在 `0.2 ~ 0.3` 区间，说明系统没有稳定地把 gold evidence 检出来。

### 3.2 一半以上样本完全没有检到参考上下文

逐样本统计显示：

- 总样本数：`9`
- `context_precision = 0` 的样本数：`5`
- `context_recall = 0` 的样本数：`5`
- `precision/recall` 同时为 `0` 的样本数：`5`

这说明当前系统不是“偶尔偏一点”，而是对超过一半样本完全没有命中参考证据。

### 3.3 题型差异明显

按 synthesizer 分组：

- `single_hop_specific_query_synthesizer`
  - `context_precision = 0.666667`
  - `context_recall = 0.666667`
- `multi_hop_abstract_query_synthesizer`
  - `context_precision = 0.125926`
  - `context_recall = 0.342857`
- `multi_hop_specific_query_synthesizer`
  - `context_precision = 0.0`
  - `context_recall = 0.0`

这说明当前系统最严重的短板不在单跳，而在“多跳 + 具体条款/定义/关系”的问题上。

### 3.4 失败模式不止一种

从失败样本可以看出，当前至少存在两类问题：

1. 检错文档
   - 例如 `ragas-cuad-0022`、`ragas-cuad-0042`、`ragas-cuad-0043`
   - 最终 `retrieved_doc_ids` 指向 `WORLDWIDESTRATEGIESINC_11_02_2005-EX-10-RESELLER AGREEMENT#p0`
   - 但参考文档并不是这个合同
2. 命中文档但没命中正确 chunk
   - 例如 `ragas-cuad-0003`
   - 最终命中了正确文档 `GOCALLINC_03_30_2000-EX-10.7-Promotion Agreement#p0`
   - 但 `retrieved_context_ids` 与 `reference_context_ids` 没有重合

因此当前问题不能只归结为“文档级召回差”，还包括 chunk 级定位和排序不稳定。

## 4. 代码层面的根因分析

## 4.1 子问题检索 query 会退化成泛词 query

这是当前最直接、最致命的问题之一。

在运行结果中已经出现了以下子问题：

- `which clause defines party`
- `which clause defines clause`
- `what clause states Section 8`
- `what clause states Section 504`

这类 query 的问题是：

- 信息锚点太弱
- 缺少 party 名、条款主题词、关系词
- 在合同语料上极易匹配到大量无关 boilerplate

对应代码：

- `youtu-graphrag/backend.py:1813-1886`
- `youtu-graphrag/models/retriever/agentic_decomposer.py:171-185`

根因具体包括：

1. `agentic_decomposer.py` 会过滤 `entities` 里的泛词，但 `left_endpoint`、`right_endpoint` 不会做同等强度的泛词过滤。
2. `backend.py` 的 `_compile_requirement_queries()` 在结构化问题下优先生成：
   - `which clause defines {left_endpoint}`
   - `which clause defines {right_endpoint}`
3. 如果 endpoint 被归一化后只剩 `party`、`clause` 这类泛词，最终就会形成极弱 query。

这可以直接解释为什么 `ragas-cuad-0003` 会退化成 `which clause defines party`，以及为什么多跳 specific 问题检索几乎全灭。

## 4.2 目标文档推断过弱，导致错合同污染

当前文档定位逻辑主要依赖 `_infer_target_doc_id_from_question()`：

- `youtu-graphrag/models/retriever/enhanced_kt_retriever.py:474-486`

它的实现方式是：

- 把问题文本归一化
- 检查其中是否直接包含某个 `doc_id` 的归一化字符串

这对真实自然语言问题基本不成立，因为用户问题通常不会包含：

- `OFGBANCORP_03_28_2007-EX-10.23-OUTSOURCING AGREEMENT`
- `GOCALLINC_03_30_2000-EX-10.7-Promotion Agreement`

这种原始文档 ID。

结果是：

- 大多数问题都无法命中 `target_doc_id`
- 后续 chunk retrieval 只能走“无目标文档”的通用逻辑

而这段通用逻辑本身还有明显问题：

- `youtu-graphrag/models/retriever/enhanced_kt_retriever.py:3538-3558`

这里不是全局按相似度取 top chunk，而是：

1. 先按 `doc_id` 分组
2. 再按 `sorted(doc_chunks.keys())` 迭代文档
3. 每个文档取 top-1

这意味着当 `target_doc_id` 缺失时，最终保留下来的 chunk 会受到文档名字典序的影响，而不是只由相关度决定。这正是错文档污染持续出现的重要原因。

## 4.3 多路径 chunk 合并丢失排序信息

结构化检索的双路径合并逻辑位于：

- `youtu-graphrag/models/retriever/enhanced_kt_retriever.py:1394-1433`

当前做法是把来自多条路径的 chunk id 放进 `set`：

```python
all_chunk_ids.update(path1_chunk_ids)
all_chunk_ids.update(path2_chunk_ids)
all_chunk_ids.update(path2_comm_chunk_ids)
all_chunk_ids.update(path3_chunk_ids)
limited_chunk_ids = list(all_chunk_ids)[:self.top_k]
```

这会造成两个问题：

1. 不同路径的得分信息完全丢失
2. 最终 `list(set(...))[:k]` 是无序截断

这类实现非常容易造成：

- 正确 chunk 原本在候选集合里
- 但进入最终上下文时因为无序截断被挤掉

这与 `ragas-cuad-0003` 的现象高度一致：文档对了，但 gold chunk 没进最终 evidence。

## 4.4 triple rerank 存在确定性实现 bug

在 triple 重排函数中：

- `youtu-graphrag/models/retriever/enhanced_kt_retriever.py:2789-2805`

存在一个明确 bug：

```python
for i, (h, r, t) in enumerate(valid_triples):
    ...
    lexical_bonus = 0.22 * self._lexical_overlap_score(triple_text, anchor_terms or [])
```

这里循环内部使用的 `triple_text` 并不是在当前迭代中重新定义的变量，而是前面构造 `triple_texts` 时残留的外层变量。结果会导致：

- 所有 triple 的 lexical bonus 都错误地基于“最后一个 triple 文本”计算
- triple 排序被系统性污染

这会直接伤害：

- 结构化问题的桥接边排序
- 与 anchor term 相关的 clause 关系优先级

这是一个应当优先修复的 P0 级问题。

## 4.5 图扩展只沿出边走，不利于合同多跳检索

邻居扩展逻辑位于：

- `youtu-graphrag/models/retriever/enhanced_kt_retriever.py:1762-1800`
- `youtu-graphrag/models/retriever/faiss_filter.py:398-440`

当前两处实现都主要依赖：

- `self.graph.neighbors(node)`

在 `networkx.MultiDiGraph` 中，这等价于走 successor，也就是只沿出边走。对于合同图谱来说，这会带来明显限制：

- 某些定义关系是从条款节点指向概念节点
- 某些 obligation 或 consequence 关系可能在图中是反向挂接
- 如果只走单向 successor，多跳链路会被截断

因此当前多跳 specific 表现很差，不一定只是 query formulation 的问题，也有图扩展搜索空间被方向性限制的原因。

## 4.6 图节点文本表征过薄

实体节点构建逻辑位于：

- `youtu-graphrag/models/constructor/kt_gen.py:302-334`
- `youtu-graphrag/models/constructor/kt_gen.py:430-455`

当前实体节点的核心属性主要是：

- `name`
- `chunk id`
- `doc_id`
- `schema_type`

而检索时节点文本主要由 `_get_node_text()` 生成：

- `youtu-graphrag/models/retriever/enhanced_kt_retriever.py:1854-1886`

其核心是：

- `name + description`

但大部分实体节点并没有足够丰富的 `description`。这意味着很多节点向量实际上只编码了实体名，而没有编码：

- 该实体在哪个条款中承担什么职责
- 与哪些 section/obligation/trigger 有关
- 该实体相关的局部证据摘要

对于合同问答，尤其是：

- responsibilities
- triggers
- indemnification obligations
- section-to-section relation

这类问题，仅靠“实体名向量”显然不够。

## 4.7 一个本可缓解串文档污染的机制默认被关闭

配置中：

- `youtu-graphrag/config/base_config.yaml:445-451`

当前：

```yaml
doc_consistency:
  enabled: false
```

这意味着系统即便已经在代码中实现了：

- 同文档加分
- 跨文档轻度惩罚

这些逻辑，默认也不会生效。

但后续复盘表明，这个默认值不能简单理解为“过于保守”。在当前检索场景下，存在：

- 单合同 local / 单合同 structural 问题
- 多跳跨条款问题
- 多参考文档或天然跨文档的问题

如果全局开启 `doc_consistency`，它会在后两类问题上压缩检索空间，甚至把首轮错误文档进一步固化。因此，更合理的结论是：

- `doc_consistency` 不应全局默认开启
- 它只能作为“单合同高置信问题”的条件触发 bias
- 对 global 问题和明确跨文档问题应保持关闭

因此，这一机制当前“默认关闭”本身不是 bug；真正的问题是系统缺少更强的目标合同 grounding 手段来替代它。

## 5. 失败样本与根因映射

### 5.1 `ragas-cuad-0003`

问题：

- `What are the responsibilities of PageMaster Corporation during the promotion period?`

现象：

- 最终命中了正确文档
- 但最终 chunk 与参考 chunk 没有重合
- trace 中子问题退化为 `which clause defines party`

对应根因：

- query 编译退化
- chunk 级排序不稳
- 多路径 chunk 合并无序

### 5.2 `ragas-cuad-0022`

问题：

- `Service Level Credit Event` 与 `Core Systems' Availability`

现象：

- 最终检到错误合同
- 子问题退化为 `which clause defines clause`

对应根因：

- structural query 编译严重退化
- target_doc_id 推断失败
- 无目标文档时 chunk retrieval 存在错误排序逻辑

### 5.3 `ragas-cuad-0042`

问题：

- `Section 8` 与 `Section 10.3` 的 indemnification 关系

现象：

- 最终检到错误合同
- 子问题只有 `what clause states Section 8`

对应根因：

- 结构化问题被编译成过窄且缺桥接的 query
- endpoint / bridge 信息丢失
- 错文档污染

### 5.4 `ragas-cuad-0043`

问题：

- `Section 504`、`Privacy Regulations`、`Sensitive Customer Information`、`Section 10.5.2`

现象：

- 一个子问题是 `what clause states Section 504`
- 另一个子问题是 `what clause defines Section 10.5.2`
- 最终仍检到错误合同

对应根因：

- 部分锚点保住了，但概念与桥接关系没有保住
- 文档定位失败
- 结构化扩展和 rerank 不足

## 6. 优化方案

## 6.1 P0：优先修复确定性错误与弱 query 生成

这是最应该优先完成的一组改动，因为它们要么是明确 bug，要么是当前评测失败最直接的诱因。

### P0.1 修复 triple rerank 的 lexical bonus bug

目标：

- 确保每个 triple 都使用自己的文本计算 lexical overlap

建议修改：

- 在 `_rerank_triples_by_relevance()` 的循环中，按 `valid_triples[i]` 或 `triple_texts[i]` 取当前 triple 文本
- 避免复用外层残留变量

预期收益：

- triple 排序稳定性立刻提升
- structural 路由更容易把相关桥接边排到前面

### P0.2 为 query 编译增加“弱 query 闸门”

目标：

- 禁止生成 `which clause defines party`、`which clause defines clause` 一类 query

建议修改：

1. 对 `left_endpoint`、`right_endpoint` 增加与 `entities` 同等级别的泛词过滤。
2. 如果编译结果只包含泛词：
   - 回退到 `anchor + specific entity + specific term`
   - 不允许直接输出只有 `party`、`clause`、`agreement` 的 query
3. 对 structural query 强制保留至少一个具体锚点：
   - party 名
   - section 编号
   - 法律术语
   - 定义项名称

预期收益：

- 多跳 specific 将不再在第一步就退化成无意义检索

### P0.3 修复多路径 chunk 合并的无序截断

目标：

- 保留不同来源 chunk 的分数或排名信号
- 禁止 `set -> list -> slice` 这种无序截断

建议修改：

1. 给 path1/path2/path3 chunk 保留原始分数
2. 做稳定 merge
3. 使用：
   - 加权融合
   - RRF
   - 至少按来源 rank 和 lexical/doc bonus 做统一排序

预期收益：

- 文档已命中但 chunk 丢失的问题会明显减少

### P0.4 修复无目标文档时的 chunk 选择策略

目标：

- 在 `target_doc_id` 缺失时，仍按相关度选 chunk

建议修改：

- 废弃“按 `sorted(doc_id)` 逐文档取 top-1”的逻辑
- 改成：
  1. 全局按 chunk 分数排序
  2. 再加每文档上限，避免单文档垄断
  3. 或使用 MMR / RRF 做多样性控制

预期收益：

- 错合同被字典序放大的问题会消失

## 6.2 P1：加强文档定位与结构化检索能力

### P1.1 改进目标文档推断

当前目标文档推断过于依赖原始 `doc_id` 是否出现在问题中，应改为多信号融合：

建议引入：

1. party 名匹配
2. section 编号匹配
3. clause term 匹配
4. 首轮 top chunk 的文档分布
5. top triple 的文档分布

然后综合出：

- `preferred_doc_ids`
- 而不是只靠字符串包含关系

### P1.2 改为条件触发的 `doc_consistency`

建议不是简单打开全局开关，而是改成“条件触发 + 两阶段使用”：

1. 第一阶段高召回，不做硬限制
2. 第二阶段重排时加入文档一致性偏好

推荐策略：

- 同文档 chunk 加分
- 非主文档 chunk 轻度降权
- 但保留少量 cross-doc chunk 作为补充

适用前提：

- 问题被判定为单合同高置信
- route 为 local 或单合同 structural
- 不是 global 汇总问题
- 不是多参考文档 / 跨文档问题

这样既能抑制串合同污染，又不会把真正需要跨条款或跨文档的证据完全丢掉。

### P1.3 structural query 不再只发一条 broad query

建议对 structural 问题显式拆成三类检索：

1. 左端点定义/条款
2. 右端点定义/条款
3. bridge clause / explicit connection

每类单独检索，再融合排序。

相比现在“把结构化问题编译成一条抽象 query”，这种方式更符合合同检索的实际需求。

## 6.3 P2：改进图表示与检索融合

### P2.1 丰富节点描述文本

建议在构图阶段为 entity / clause 节点补充：

- 局部 chunk 摘要
- evidence sentence
- section 编号
- obligation / trigger / remedy 等关键词

这样 `_get_node_text()` 生成的节点语义才足够支撑合同检索。

### P2.2 引入 sparse + dense 混合检索

合同领域天然存在大量强锚点：

- `Section 8`
- `Section 10.3`
- `Section 504`
- `Service Level Credit Event`
- `Sensitive Customer Information`

仅靠 dense embedding 容易把这些锚点语义化过度，导致命中“相关”但不“精确”的条款。

建议加入：

- BM25 或倒排索引
- dense chunk retrieval
- 图检索

再通过：

- RRF
- cross-encoder rerank
- 规则加分

统一融合。

### P2.3 图扩展改为双向

建议：

- 同时考虑 successors 和 predecessors
- 对结构化问题的多跳搜索做双向 BFS/beam search

这样能显著降低因为图边方向而丢失桥接链路的概率。

## 7. 建议实施顺序

建议按以下顺序推进：

### 第一阶段：先修确定性问题

1. 修复 triple rerank bug
2. 修复无目标文档时的 chunk 选择策略
3. 修复多路径 chunk 合并的无序截断
4. 给 query compilation 增加弱 query 过滤与 fallback

这一阶段的目标是先把明显错误去掉，预计就能改善：

- 错合同污染
- gold chunk 被无序截断挤掉
- structural query 完全失焦

### 第二阶段：再补文档一致性与结构化检索

1. 重做目标文档推断
2. 仅在单合同高置信问题上启用改进后的 `doc_consistency`
3. structural 问题拆成 endpoint + bridge 三段检索

这一阶段的目标是把 multi-hop specific 从“经常 0/0”拉回到“至少能稳定检到相关证据”。

### 第三阶段：做体系级增强

1. 节点文本增强
2. sparse + dense + graph 混合检索
3. 更强的 reranker

这一阶段的目标是追求更稳定、更可扩展的企业合同检索能力。

## 8. 验证方案

优化不能只看最终答案，需要继续以检索指标为主。

建议固定以下回归集：

- `ragas-cuad-0003`
- `ragas-cuad-0022`
- `ragas-cuad-0042`
- `ragas-cuad-0043`
- `ragas-cuad-0041`

建议每轮改动都重点观察：

1. `context_precision`
2. `context_recall`
3. `retrieved_doc_ids` 是否仍串错合同
4. `retrieved_context_ids` 是否开始稳定覆盖 reference chunk

优先验收标准建议如下：

### 短期目标

- `multi_hop_specific_query_synthesizer`
  - `context_precision > 0`
  - `context_recall > 0`
- 零召回样本数从 `5/9` 显著下降

### 中期目标

- 整体 `context_precision` 提升到 `0.4+`
- 整体 `context_recall` 提升到 `0.5+`
- 多跳 specific 不再出现整类 `0/0`

## 9. 最终判断

本次 `retrieval_requirements_smoke_eval` 所反映的问题，不是单点故障，而是一个“从 query 编译到文档定位、再到 chunk 排序和图扩展”的链路性问题。

其中优先级最高的结论是：

1. 当前系统的主要短板是多跳 specific 检索
2. 失败的最前端诱因是弱 query 编译
3. 错文档污染被文档推断缺失和错误 chunk 选择策略进一步放大
4. 检索后端还存在确定性实现 bug，需要先修

因此，最合理的改进路线不是先调答案生成，而是优先修复检索链路本身。

## 10. P0 实施进展与 `retrieval_requirements_smoke_eval_t` 复盘

在完成一轮 P0 修复后，系统重新生成了：

- `outputs/results/ragas/retrieval_requirements_smoke_t.jsonl`
- `outputs/results/ragas/retrieval_requirements_smoke_eval_t/ragas_eval_summary.json`
- `outputs/results/ragas/retrieval_requirements_smoke_eval_t/ragas_eval_per_sample.jsonl`

这一轮结果反映出：系统已经不再停留在“经常完全检不到”的状态，但新的主矛盾变成了“召回上升、精度下降”。

### 10.1 已解决的问题

以下问题已经得到明显缓解或直接修复：

1. `party / clause` 一类极弱 query 大幅减少
   - 例如旧版 `ragas-cuad-0003` 中的 `which clause defines party` 已不再出现
   - 说明 endpoint 泛词过滤与 query fallback 已经起效
2. 多路径 chunk 合并的无序截断已修复
   - 这使得正确 chunk 更容易被保留在候选集合中
3. 无目标文档时按 `doc_id` 字典序取 chunk 的问题已修复
   - 当前不再主要表现为“字母序放大错文档”
4. triple rerank 的 lexical bonus 变量 bug 已修复
5. `_best_subject_terms()` 的实现 bug 已修复
   - 旧逻辑实际会让每组只保留最后一个 term，进一步放大 query 编译失真

从结果上看，这些修复带来的直接变化是：

- 整体 `context_recall` 从 `0.336508` 提升到 `0.528042`
- `precision/recall` 同时为 `0` 的样本数从 `5/9` 降到 `3/9`
- `single_hop_specific_query_synthesizer` 的 `context_recall` 达到 `1.0`
- `multi_hop_specific_query_synthesizer` 不再整类 `0/0`，`context_recall` 从 `0.0` 提升到 `0.222222`

这说明 P0 确实把系统从“完全失焦”推到了“能先把相关证据捞进来”。

### 10.2 尚未解决的问题

当前最主要的未解决问题有三类。

#### 10.2.1 子问题 / query 编译仍然会产出畸形或过弱 query

虽然 `party/clause` 这类最弱 query 已经减少，但新的半退化 query 仍然存在，例如：

- `ragas-cuad-0023`
  - `which clauses mention party (supplier)`
- `ragas-cuad-0041`
  - `which clauses mention Metavante`
- `ragas-cuad-0042`
  - `what clause states section 8 section 10.3`
- `ragas-cuad-0043`
  - `what clause states clause{name or identifier referencing 'Section 504' or 'Privacy Regulations'}`
- `ragas-cuad-0002`
  - `what clause states GOCALLINC_03_30_2000-EX-10.7-Promotion Agreement`

这些 query 的共同问题是：

- 仍然缺少真正的桥接关系约束
- 多跳问题被压扁成“mention 某个词”
- 有些 query 直接带了结构化残片，例如 `clause{...}`
- 有些 query 把合同标题本身当成检索主题，说明 requirement 到 query 的归一化仍不稳定

因此，当前第一主故障环节依然是 `子问题生成 / query 编译`，只是退化形式比旧版更隐蔽了。

#### 10.2.2 最终 chunk 选择仍然会把噪声稳定带入上下文

这轮 `eval_t` 最明显的特征是：

- `context_recall` 显著提高
- `context_precision` 从 `0.264198` 降到 `0.166667`

这说明系统已经更容易“把对的证据捞进来”，但同时也把更多无关 chunk 一起保留了。

对应典型样本：

- `ragas-cuad-0002`
  - `context_recall = 1.0`
  - `context_precision = 0.166667`
- `ragas-cuad-0003`
  - `context_recall = 1.0`
  - `context_precision = 0.0`
- `ragas-cuad-0022`
  - `context_recall = 0.8`
  - `context_precision = 0.333333`
- `ragas-cuad-0042`
  - `context_recall = 0.666667`
  - `context_precision = 0.0`

这类现象说明：

- query 能召回 gold evidence
- 但最终 `per_subquestion_guarantee_then_global_fill` 会把弱子问题的噪声 chunk 也制度性保留下来
- 因此 precision 被显著拉低

所以当前第二主故障环节是 `final chunk selection`，尤其是：

- 每个子问题最少保底 `2` 个 chunk 的策略
- 缺少更强的 chunk 级 rerank 与过滤

#### 10.2.3 目标合同 grounding 仍然偏弱

这轮结果里，错合同污染虽然缓解，但仍然明显存在，例如：

- `ragas-cuad-0021`
- `ragas-cuad-0022`
- `ragas-cuad-0042`
- `ragas-cuad-0043`

根因仍然是：

- `_infer_target_doc_id_from_question()` 依赖问题文本中直接出现归一化 `doc_id`
- 但真正参与检索的是编译后的子问题
- 子问题通常不再保留合同标题

因此，即便总问题里包含 `Contract title`，子问题检索阶段也经常失去文档锚点。

这里需要强调：

- `preferred_doc_ids` 在这轮结果里全部为空
- 这不是 bug，而是因为 `doc_consistency.enabled = false`
- 真正的问题不是“为什么没偏置同文档”，而是“为什么没有更稳的合同 grounding”

### 10.3 新增发现的问题

在这轮实施与复盘中，还发现了几个此前文档里没有单独指出的问题。

#### 10.3.1 `retrieval_trace` 没有完整落盘 `retrieval_queries`

后端实际上已经生成并使用了编译后的多 query：

- `backend.py` 会在 `_requirements_to_sub_questions()` 中为每个子问题构造 `retrieval_queries`
- 在检索阶段逐条执行这些 query

但在最终 `retrieval_trace` 落盘时，`sub_questions` 结构并没有把 `item["retrieval_queries"]` 写进去，因此：

- 结果文件里 `sub_questions[].retrieval_queries` 全部为 `None`
- 这会让后续分析误以为系统没有走 query compilation

这个问题不是检索质量下降的直接根因，但会显著降低问题定位效率。

#### 10.3.2 当前 `answer_correctness` 的提升不能简单视为检索变好

例如：

- `ragas-cuad-0043`
  - `answer_correctness` 从 `0.152567` 提升到 `0.569452`
  - 但 `context_precision = 0.0`
  - `context_recall = 0.0`

这说明：

- 某些样本的答案分数波动可能来自回答风格、判分器容忍度或模型猜测
- 不能把 `answer_correctness` 的上升直接等同于检索链路变好

因此，当前阶段仍应以 `context_precision`、`context_recall` 和 doc/chunk 对齐情况为主。

### 10.4 对当前问题优先级的重新排序

结合 `eval_t`，当前问题优先级应调整为：

1. `子问题生成 / query 编译`
   - 这是完全失败样本和多跳退化样本的最前端诱因
2. `final chunk selection`
   - 这是 recall 上升但 precision 大幅下降的直接原因
3. `目标合同 grounding`
   - 仍然影响错文档污染，但不应简单依赖 `doc_consistency`
4. `多跳图检索能力`
   - 仍然重要，但当前更像是放大器而非首触发点
5. `trace 落盘完整性`
   - 不直接影响指标，但影响调试效率

### 10.5 对实施计划的修正

基于当前结果，后续计划应作如下修正：

1. `doc_consistency`
   - 继续保持默认关闭
   - 不作为下一步主修方向
   - 后续仅考虑做“单合同高置信条件触发”
2. 下一阶段优先级
   - 先修 query 编译中的畸形输出与桥接信息丢失
   - 再改最终 chunk selection，降低弱子问题噪声注入
   - 然后补强目标合同 grounding
3. 验收口径
   - 短期不再追求进一步抬高 recall
   - 而是优先观察 precision 是否从当前 `0.166667` 回升
   - 同时确保 multi-hop specific 的 recall 不回退到 `0`

## 11. 长期跟踪约定

从当前阶段开始，这篇文档不再只是一次性分析报告，而作为 `youtu-graphrag` 检索优化的长期计划与进展记录。

后续维护约定如下：

1. 每完成一轮明确改动，都在本文档中补充记录
   - 记录改动目标
   - 记录涉及模块
   - 记录解决了什么问题
   - 记录仍然遗留什么问题
2. 每次重跑评测后，都更新本文档中的阶段性结论
   - 优先记录 `context_precision`
   - 优先记录 `context_recall`
   - 记录零召回样本数变化
   - 记录典型失败样本是否迁移
3. 如果出现新的失败模式或新的观测盲区，也要补充到本文档
   - 包括新的畸形 query
   - 包括新的串文档模式
   - 包括 trace 缺失或评测解释偏差
4. 优化顺序默认遵循本文档第 `10.4` 与 `10.5` 节
   - 先处理 `query 编译`
   - 再处理 `final chunk selection`
   - 再处理 `目标合同 grounding`
   - 然后再推进更深层的图检索增强

## 12. 持续进展记录

### 12.1 阶段 `P0`

状态：

- 已完成

已完成事项：

- 修复 `party / clause` 一类极弱 query 的首轮退化问题
- 修复多路径 chunk 合并的无序截断
- 修复无目标文档时按 `doc_id` 字典序选 chunk 的问题
- 修复 triple rerank 的 lexical bonus 变量 bug
- 修复 `_best_subject_terms()` 只保留最后一个 term 的实现 bug

结果变化：

- `context_recall: 0.336508 -> 0.528042`
- `context_precision: 0.264198 -> 0.166667`
- `precision/recall` 同时为 `0` 的样本数：`5/9 -> 3/9`

阶段结论：

- `P0` 已经把系统从“经常完全检不到”推进到“能够把一部分 gold evidence 捞进来”
- 当前主矛盾已经从“召回失败”转向“query 编译残缺 + 最终选择噪声偏大”

下一阶段默认目标：

- 优先修复子问题 / query 编译中的畸形输出与桥接信息丢失
- 其次优化 `final chunk selection`
- `doc_consistency` 继续保持默认关闭，不作为当前主修方向

### 12.2 阶段 `P1-1`：query 编译与最终选择策略收敛

状态：

- 已完成代码实现与 `smoke_t2` 行为验证，评测指标待确认

本轮改动目标：

- 继续压制畸形 query
- 避免弱子问题在最终上下文里硬占固定名额
- 补全 trace，便于下一轮精确复盘

已完成事项：

- 在 `backend.py` 中为 requirement query 编译新增 query 清洗逻辑
  - 清洗 `party (supplier)` 这类 generic wrapper
  - 清洗 `clause{name or identifier referencing 'Section 504' ...}` 这类结构化残片
  - 抽取引号内术语和 `Section / Article / Paragraph` 锚点，避免把 `{...}` 原样下发到检索器
- 调整 structural query 编译顺序
  - 优先生成 richer 的 bridge / keyword query
  - 不再默认让单 endpoint 的 `mention` / `define` / `state` query 占据第一位
- 增加 doc-title-like 短语防护
  - 避免把合同标题直接当作 `what clause states ...` 的 clause subject
- 调整 `final chunk selection`
  - 对明显畸形、低质量、`not_found` 的子问题降低保底名额
  - 对没有 query 质量元数据的旧路径保持原有行为，避免破坏基础召回
- 在 `retrieval_trace.sub_questions` 中补充 `retrieval_queries`
  - 解决此前结果文件中该字段恒为 `None` 的观测缺口

新增回归测试：

- query 编译会清洗 structured artifact 与 generic wrapper
- 弱 query / `not_found` 子问题不再在最终选择阶段硬占两个 chunk
- 旧的 per-subquestion guarantee 行为在无额外元数据时保持兼容

当前验证结果：

- `pytest -q tests/test_youtu_retrieval_p0_regressions.py tests/test_youtu_final_chunk_selection.py tests/test_youtu_edge_layer.py`
- `16 passed`
- `outputs/results/ragas/retrieval_requirements_smoke_t2.jsonl`

基于 `smoke_t2` 的行为验证结论：

1. 可以确认有效的部分
   - `retrieval_queries` 已经完整落盘
     - 上一轮 `smoke_t` 中 `9/9` 个样本的 `sub_questions[].retrieval_queries` 都是 `None`
     - 本轮 `smoke_t2` 中 `9/9` 个样本都已经写出了编译后的 query 列表
   - 先前最典型的畸形 query 已明显减少
     - `ragas-cuad-0023` 不再是 `which clauses mention party (supplier)`，而是拆成了 `supplier / insurance / audit` 相关的多条 query
     - `ragas-cuad-0043` 不再保留 `clause{name or identifier ...}` 这类原始结构化残片
     - `ragas-cuad-0042` 也不再只剩 `what clause states section 8 section 10.3` 这一种弱 query，而是出现了包含 `breach impact / liability_excludes_cap_for` 的 richer query
   - 目标合同污染在部分样本上收敛
     - `ragas-cuad-0021` 的最终 `retrieved_doc_ids` 从 `3` 个文档收敛到 `1` 个文档，且保持在参考合同
     - `ragas-cuad-0043` 的最终 `retrieved_doc_ids` 从 `3` 个文档收敛到 `1` 个文档

2. 只能判定为“部分有效”的部分
   - query 清洗已经生效，但并没有彻底变成自然语言检索 query
   - 仍然有大量 graph 内部语法泄漏到 query 中，例如：
     - `party_to -> agreement -> clause -> (grants_right_to, creates obligation)`
     - `clause-[maintains_insurance]->insurance`
     - `governed_by`
     - `payment_triggered_by`
   - 这说明当前清洗主要解决了 `party (supplier)` / `clause{...}` 一类包装问题，但还没有把“图关系路径表达”完全转换成自然语言检索表达

3. 目前还不能证明已经改善 precision 的部分
   - 这一点在 `eval_t2` 中已经得到部分确认，但代价很大
   - 整体指标从 `eval_t -> eval_t2` 变化为：
     - `answer_correctness: 0.462698 -> 0.534929`
     - `faithfulness: 0.403521 -> 0.627085`
     - `context_precision: 0.166667 -> 0.239947`
     - `context_recall: 0.528042 -> 0.247619`
   - 这说明：
     - precision 确实回升了
     - 回答建立在证据上的程度也明显提升了
     - 但 recall 出现了显著回落，系统重新变得“更保守、更容易漏证据”
   - 因此本轮优化不能定义为“整体成功”，更准确地说是：
     - `query hygiene` 和 `噪声控制` 有效
     - 但 `证据覆盖率` 被压得过低

4. 新增发现的问题
   - 关系语义泄漏问题变得更突出
     - 当前 query 编译会把 `maintains_insurance`、`audit_right_over`、`limits_liability_of` 这类图关系名直接拼进检索 query
     - 这些词对图检索可能有帮助，但对 chunk 文本检索往往过于人工，容易扩大噪声
   - query 里仍有明显重复堆叠
     - 例如 `Metavante Customer Metavante Customer ...`
     - 例如 `PageMaster Corporation promotion period PageMaster Corporation promotion period ...`
     - 说明 query 去重目前只在完整字符串级别生效，还没有做到短语级压缩
   - `final_chunk_selection` 虽然仍正常落盘，但仅从 `smoke_t2` 还无法验证“弱子问题降配额”是否真的改善了最终选择质量

本轮结论：

- 当前已经把“明显畸形 query 直接进入检索”和“弱子问题稳定注入噪声”这两个问题继续往下压了一层
- 可以确认本轮改动对 `query 可观测性` 和 `部分 query 畸形问题` 有实质作用
- `eval_t2` 已确认：
  - `context_precision` 从 `0.166667` 回升到 `0.239947`
  - `faithfulness` 从 `0.403521` 提升到 `0.627085`
  - 但 `context_recall` 从 `0.528042` 回落到 `0.247619`
- 当前更准确的判断是：`P1-1` 对 query 编译和最终噪声控制“有用，但过度收缩了召回”
- 这轮优化改善了“检到一堆噪声后胡乱作答”的问题，但没有形成“高 precision 且不丢 recall”的平衡
- `multi_hop_specific_query_synthesizer` 在 `eval_t2` 中重新跌回 `0/0`，说明这类问题仍然没有真正打通

下一步默认目标：

- 重点观察 `0021 / 0022 / 0023 / 0043` 的 `context_precision`、`context_recall`、`retrieved_doc_ids`
- 下一轮 query 编译优化优先级：
  - 把 `maintains_insurance`、`payment_triggered_by`、`governed_by`、`party_to -> agreement -> clause` 这类图关系路径表达改写为自然语言 query
  - 增加短语级去重，避免 `Metavante Customer Metavante Customer` 这类重复堆叠
  - 在不启用全局 `doc_consistency` 的前提下继续补强目标合同 grounding
  - 避免把 final selection / query 收紧到直接压没单跳与 multi-hop specific 的 gold evidence

`eval_t2` 的样本级判断补充如下：

- 明显改善：
  - `ragas-cuad-0021`
    - `context_precision: 0.0 -> 0.1`
    - `context_recall: 0.0 -> 0.4`
  - `ragas-cuad-0022`
    - `context_precision: 0.333333 -> 0.666667`
    - 但 `context_recall: 0.8 -> 0.4`
  - `ragas-cuad-0023`
    - `context_precision: 0.0 -> 0.392857`
    - `context_recall: 0.285714 -> 0.428571`
- 明显退化：
  - `ragas-cuad-0002`
    - `context_recall: 1.0 -> 0.0`
  - `ragas-cuad-0003`
    - `context_recall: 1.0 -> 0.0`
  - `ragas-cuad-0042`
    - `context_recall: 0.666667 -> 0.0`
- 基本无改善：
  - `ragas-cuad-0041`
  - `ragas-cuad-0043`

这进一步说明：

- 当前方向对“部分抽象多跳问题”的 precision 提升是有效的
- 但对子问题编译和最终选择的收紧已经伤害了单跳 specific 与部分 structural 问题的 gold evidence 保留

### 12.3 阶段 `P1-2`：最终回答上下文打包问题定位

状态：

- 已完成代码实施，评测待验证

定位背景：

- 在抽样复盘 `ragas-cuad-0002` 的子问题回答 prompt 与最终回答 prompt 后，发现当前最终回答阶段的上下文组织方式本身也存在明显问题
- 这部分问题与“检索是否命中”不同，它属于“即便前面检到了证据，最终回答阶段也未必以最合理的方式消费这些证据”

当前最终回答 prompt 的真实组成：

1. `Sub-question Findings`
   - 每个子问题的 `sub_answer`
   - 每个子问题的 `reason`
   - 每个子问题的 `support_chunk_ids`
2. `Backing Chunks`
   - 每个子问题前 `2` 个 `support_chunk_ids` 对应的 chunk 文本
   - 合并后最多 `6` 段
3. `Global Triple Pool`
   - 所有子问题累计检到的 triples 去重后取前 `20` 条

对应结论：

- 当前最终回答并不是直接基于“最终筛出的 top chunk”来生成
- 它主要依赖：
  - 子问题答案与理由
  - 少量 backing chunks
  - 一批原始 triples dump

这带来的问题如下。

#### 12.3.1 `Global Triple Pool` 当前更像调试信息，不像高质量回答上下文

当前 `Global Triple Pool` 的形式是原始 triple 文本直接堆叠，例如：

- `(head, relation, tail) [score: ...]`
- `doc_id`
- `schema_type`
- `mention_count`
- `evidence_chunk_ids`
- `[Unknown Node: ...]`

这些信息的问题是：

- 对最终回答生成帮助有限
- 元数据密度过高，语义密度过低
- 容易分散模型注意力
- 很像“检索/图谱调试输出”直接喂给了回答模型

因此，`Global Triple Pool` 当前不是完全没用，但形式明显不对。

#### 12.3.2 structural 问题在最终回答阶段没有保留“结构化骨架”

对 structural 类型问题，最终回答阶段真正应该保留的是：

- `left_endpoint`
- `right_endpoint`
- `bridge_relation`
- top bridge evidence
- 对应 clause / chunk anchor

但当前 prompt 里没有把这些 requirement 级结构字段显式组织出来。结果是：

- 最终模型看不到“当前要确认的左右节点到底是谁”
- 看不到“真正要验证的关系是什么”
- 看不到“哪一条 triple 才是 bridge triple”
- 只能在一堆 triples 里自行猜测重点

这也是为什么当前 structural 回答上下文里虽然有 triples，却没有清晰的“结构化信息”。

#### 12.3.3 子问题回答一旦偏了，最终回答会顺着偏

当前最终回答 prompt 明确要求：

- `Prioritize the sub-question findings and reasons`

这意味着：

- 如果子问题回答已经被 noisy triple 或 noisy chunk 带偏
- 最终回答阶段会把这个偏差当作高优先级输入继续放大

因此，最终回答阶段不是纯纠错器，它对前面子问题结论有明显继承性。

#### 12.3.4 当前最终回答上下文不符合更合理的产品预期

更合理的最终回答上下文应优先包含：

1. 子问题答案
2. 子问题理由
3. 最相关的 backing chunks
4. 必要时补一个“压缩后的结构证据摘要”

而不应该直接包含：

- 原始 triple dump
- 大量图谱内部元数据
- `doc_id / schema_type / mention_count / evidence_chunk_ids` 这类低价值噪声

因此，这部分不是提示词微调问题，而是“上下文打包策略”本身需要重构。

#### 12.3.5 对后续优化计划的修正

后续最终回答阶段的优化应新增一条明确主线：

- 删除或弱化当前 `Global Triple Pool` 的原始 triple dump
- 改成面向 structural 问题的 `Structural Evidence Summary`

建议的新结构如下：

1. `Sub-question Findings`
   - 保留
2. `Backing Chunks`
   - 保留，但后续可从“每个子问题前 2 个”升级为“按最终相关度选出的少量 chunk”
3. `Structural Evidence Summary`
   - 每个 structural 子问题只保留：
     - `left_endpoint`
     - `right_endpoint`
     - `bridge_relation`
     - `1-3` 条自然语言化 bridge evidence
     - 对应 chunk / clause anchor

自然语言化后的 bridge evidence 示例应类似：

- `PageMaster Corporation may use Motorola trademarks to advertise the promotion`
- `Section 8 links Motorola to the promotion through trademark and logo usage rights`

而不是：

- `(PAGEMASTER CORPORATION, grants_license_to, Motorola, Inc.) [score: 0.798]`

#### 12.3.6 下一步默认目标

- 在最终回答 prompt 中移除原始 `Global Triple Pool` dump
- 新增 `Structural Evidence Summary` 生成逻辑
- 让 structural 子问题在最终回答阶段显式带上：
  - `left_endpoint`
  - `right_endpoint`
  - `bridge_relation`
  - top bridge evidence
- 然后重新观察：
  - `faithfulness`
  - `answer_correctness`
  - structural 类型问题的稳定性

#### 12.3.7 本轮实施进展

本轮已在 [backend.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/backend.py) 完成 `P1-2` 的第一版实现，核心改动如下：

- 新增最终回答上下文辅助函数：
  - `_build_structural_evidence_summary()`
  - `_build_supporting_relations()`
  - `_build_final_answer_knowledge_package()`
  - `_build_iterative_reasoning_context()`
- `init_prompt` 不再直接拼接原始 `Global Triple Pool`，而是改成：
  - `Sub-question Findings`
  - `Structural Evidence Summary`
  - `Backing Chunks`
  - 仅在没有结构摘要时才回退到精简后的 `Supporting Relations`
- `structural` 路由下的最终回答迭代推理阶段，也不再直接把原始 triple dump 作为主上下文，而是复用同样的结构摘要 / 关系摘要格式

这轮实现具体解决了两类问题：

- 原始 triple dump 中的大量 `doc_id / schema_type / mention_count / evidence_chunk_ids / [Unknown Node: ...]` 噪声，不再直接进入最终回答 prompt
- structural 问题在最终回答阶段终于显式暴露：
  - `left_endpoint`
  - `right_endpoint`
  - `bridge_relation`
  - `anchor`
  - 来自 `support_spans` 的精简 bridge evidence

当前仍未完成或未验证的点：

- 这轮还没有重跑 `retrieval_requirements_smoke_eval` 系列评测，因此还不能确认：
  - `faithfulness` 是否继续提升
  - `answer_correctness` 是否受益
  - `context_precision / context_recall` 是否会被间接影响
- 当前 `Structural Evidence Summary` 仍然基于已有 `support_spans` 做文本压缩，不是从图路径中显式挑出“最关键 bridge triple”后再重写，所以它已经明显优于原始 triple dump，但还不是最终形态
- 子问题回答 prompt 本身仍然保留原始 triples 作为证据输入，因此“子问题先偏，最终答案继承偏差”的链条只是缓解了一半，还没有完全切断

本轮回归验证：

- 已新增针对最终回答上下文的新测试：
  - 结构摘要会暴露 `left/right/bridge` 字段并清理元数据噪声
  - 最终知识包不再包含 `Global Triple Pool`
- 已通过：

```bash
env XONSH_HISTORY_BACKEND=dummy pytest -q tests/test_youtu_retrieval_p0_regressions.py tests/test_youtu_final_chunk_selection.py tests/test_youtu_edge_layer.py
```

结果：`18 passed in 0.55s`

阶段结论：

- `P1-2` 第一版实现已完成
- 当前已从“问题定位”进入“代码已落地，等待评测验证”阶段
- 下一步默认动作是重跑 `outputs/results/ragas/retrieval_requirements_smoke_eval_*`，优先观察 structural 样本的：
  - 最终回答 prompt 变化
  - `faithfulness`
  - `answer_correctness`

#### 12.3.8 `retrieval_requirements_smoke_eval_t3` 复盘

`P1-2` 落地后，已重跑 [ragas_eval_summary.json](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_eval_t3/ragas_eval_summary.json) 与 [ragas_eval_per_sample.jsonl](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_eval_t3/ragas_eval_per_sample.jsonl)。

相对上一轮 `retrieval_requirements_smoke_eval_t2`，整体指标变化为：

- `answer_correctness`: `0.534929 -> 0.404451`
- `faithfulness`: `0.627085 -> 0.780444`
- `context_precision`: `0.239947 -> 0.203704`
- `context_recall`: `0.247619 -> 0.369841`

这说明 `P1-2` 的作用是明确但不完整的：

- 有效：
  - 最终回答更“贴证据”，`faithfulness` 明显提升
  - 整体 `context_recall` 有所回升，说明这轮没有进一步压缩检索空间
- 无效或副作用明显：
  - `answer_correctness` 明显下降
  - `context_precision` 没有改善，反而略有回落

更准确地说，`P1-2` 解决的是“最终回答 prompt 的结构与噪声形式”，但没有解决“前面子问题阶段已经把证据读偏”的问题。

按 synthesizer 维度看：

- `single_hop_specific_query_synthesizer`
  - `faithfulness = 1.0`
  - `context_precision = 0.333333`
  - `context_recall = 0.333333`
  - 这一类基本稳定，没有明显新收益，也没有明显新损失
- `multi_hop_abstract_query_synthesizer`
  - `context_recall = 0.409524 -> 0.609524`
  - `faithfulness = 0.545937 -> 0.633815`
  - 但 `answer_correctness = 0.67703 -> 0.451352`
  - 说明证据覆盖变宽了，但回答整合质量反而下降
- `multi_hop_specific_query_synthesizer`
  - `context_recall = 0.0 -> 0.166667`
  - `faithfulness = 0.418651 -> 0.707516`
  - 但 `context_precision` 仍为 `0.0`
  - 说明最难的一类依旧没有真正打通，只是回答更保守、更“像是在复述已有证据”

逐样本看，当前 `t3` 暴露出的模式更清楚：

- `0001`
  - 基本稳定，单跳 local 仍然正常
- `0002`
  - `faithfulness = 1.0`，但 `context_precision/context_recall` 仍是 `0/0`
  - 说明回答更像“忠实复述了错误或不完整证据”，不是检索变好了
- `0003`
  - `faithfulness: 0.75 -> 1.0`
  - `answer_correctness: 0.437 -> 0.185`
  - 这是最典型的“更忠实地答错了”
- `0021`
  - `context_recall: 0.4 -> 0.6`
  - 但 `answer_correctness: 0.673 -> 0.477`
  - 说明证据覆盖更广，但回答整合没有把多跳关系解释清楚
- `0022`
  - `context_recall: 0.4 -> 0.8`
  - `faithfulness: 0.462 -> 0.833`
  - 但 `answer_correctness: 0.876 -> 0.515`
  - 说明 evidence packing 更像“守住了证据”，但 final synthesis 没有抓住最关键触发条件
- `0023`
  - `context_precision` 上升，但 `answer_correctness` 和 `faithfulness` 反向波动
  - 说明 structural 相关证据虽然更聚焦，但子问题本身仍可能选错重点
- `0041`
  - `context_recall: 0.0 -> 0.5`
  - `faithfulness: 0.381 -> 0.75`
  - 但 `answer_correctness` 仍下降
  - 说明开始能碰到一部分 gold evidence，但桥接解释仍不成立
- `0042`
  - `faithfulness` 上升，但 `answer_correctness` 下滑且 `recall` 仍是 `0`
  - 依旧是前端召回失败，不是最终回答包装问题
- `0043`
  - `faithfulness` 与 `answer_correctness` 都有一定回升
  - 但 `context_precision/context_recall` 仍是 `0/0`
  - 说明回答更保守，但并未触及真正的 gold context

因此，`t3` 可以支持一个更严格的阶段结论：

- `P1-2` 作为“最终回答上下文清洗”是有价值的
- 它确实把回答从“吃调试信息”推进到了“更忠实于已有证据”
- 但它不能单独提升正确性，因为当前主要噪声源已经不在 final answer prompt，而在更早一层：
  - `sub-question` 回答 prompt 仍然过于冗杂
  - `sub-question` 证据选择仍然会混入低价值 triple 和弱相关 chunk

新增问题确认：

- 当前 `sub-question` 回答 prompt 仍然是：
  - 前 `12` 条原始 triples
  - 前 `6` 段原始 chunks
- 这意味着：
  - `[Unknown Node]`
  - `doc_id / schema_type / mention_count / evidence_chunk_ids / score`
  - `=== Entity Information ===` 一类列表块
  - 弱相关法规或条款片段
  仍然会在子问题阶段直接喂给模型

这也解释了为什么 `t3` 会出现“faithfulness 明显变高，但 correctness 反而下降”的现象：

- final answer 阶段已经更忠实
- 但它忠实继承的是“子问题阶段已经被噪声扰动后的中间结论”

下一步默认目标修正为：

- `P1-3`：重构 `sub-question` 证据打包
- structural 子问题 prompt 应改成：
  - `Sub-question Evidence Summary`
  - `left_endpoint / bridge_relation / right_endpoint / anchor`
  - top `3-5` 条清洗后的 relation lines
  - top `2-4` 段聚焦后的 chunk excerpts
- 明确过滤：
  - `[Unknown Node]`
  - triple 元数据噪声
  - `=== Entity Information ===` 类非证据列表
  - 与当前 requirement 焦点弱相关的 chunk

#### 12.3.9 `t3` 子问题检索归因：到底是“被淹没”还是“没召回到”

进一步对照 [retrieval_requirements_smoke_t3.jsonl](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_t3.jsonl) 的 `retrieval_trace.sub_questions` 与 [ragas_eval_per_sample.jsonl](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_eval_t3/ragas_eval_per_sample.jsonl) 的最终 `retrieved_context_ids / reference_context_ids` 后，可以把当前样本分成三类：

1. `final_hit`
   - `0001`
   - `0021`
   - `0022`
   - `0023`
2. `right_doc_wrong_chunk`
   - `0002`
   - `0003`
   - `0041`
   - `0042`
3. `wrong_doc`
   - `0043`

这意味着：

- 当前主问题并不是“多数样本已经召回了正确 chunk，但后面被 final selection 淹没”
- 更接近：
  - 大多数失败样本已经命中了正确文档
  - 但在子问题检索阶段就没有把 gold chunk 找出来
- 只有少数成功样本同时伴随上下文污染，例如：
  - `0022`
  - `0023`

更具体地说：

- `0002`
  - 正确文档命中
  - 但 gold chunk `9dcb9832-...` 没有进入子问题 `support_chunk_ids`
  - 属于“文档对了，chunk 没召回”
- `0003`
  - 正确文档命中
  - 但 gold chunk `17022fe3-...` 没有进入子问题支持证据
  - 属于“文档对了，chunk 没召回”
- `0041`
  - 两个参考文档都在最终文档集合中
  - 但参考 chunk 仍然没有进入子问题支持证据
  - 属于“多跳桥接失败，不是被后续上下文淹没”
- `0042`
  - 正确文档命中
  - 但两个 reference chunk 都没有进入子问题支持证据
  - 仍然是“right doc, wrong chunk”
- `0043`
  - 连正确文档都没命中
  - 属于真正的 `wrong_doc`

因此，当前阶段应该修正一个很重要的判断：

- `P1-3` 不能被理解成“单纯压缩子问题 prompt”
- 因为对 `0002 / 0003 / 0041 / 0042 / 0043` 这组样本来说，核心问题不是“召回到了但被提示词污染淹没”
- 而是“子问题检索阶段本身就没有把正确 chunk 提到前面”

换句话说：

- 子问题上下文冗杂，确实是问题
- 但它更可能放大已有错误，而不是当前失败样本的唯一第一原因

#### 12.3.10 是否属于“问题拆分 / 意图理解”出了问题

结论：`是，而且是当前 right_doc_wrong_chunk 样本的重要主因之一`，但不是唯一原因。

判断依据主要来自 `t3` 中的子问题文本和编译后的 `retrieval_queries`：

- `0003`
  - 子问题仍然保留大量图路径语法：
    - `party_to -> (agreement) -> [grants_right_to, grants_license_to, limits_liability_of, maintains_insurance]`
    - `survives_for`
  - 这不是自然语言检索表达，而是“图模式直接泄漏到 query”
- `0042`
  - query 仍然包含：
    - `Section 8 from Section 10.3 / remedy / event terminates_on_event`
    - `clause-[section 8]-[is_null_and_void|terminates_on_event|survives_for]->[remedy|event]`
  - 说明 decomposition / query compilation 仍在把结构意图错误地编码成检索字符串
- `0043`
  - 子问题语义已经过载：
    - `governed_by / defines`
    - `confidentiality_applies_to`
    - `with an inferred link from the law to the confidential_info type`
  - 这是典型的“把推理说明写进 query”，不是真正的检索表达
- `0002`
  - 虽然没有明显符号残片，但子问题本身只是在粗糙地拼接实体和关系名：
    - `Motorola PageMaster Corporation party_to ... role`
  - 这种 query 对于“找 Motorola 在 promotion 中的具体角色”仍然缺少 clause-level anchor
- `0041`
  - `Metavante indemnification pays ... Customer expenses`
  - `Neoforma indemnification ... third-party claims obligations`
  - 子问题没有把“报销费用”与“另一合同中的 indemnification 触发条件”准确拆成两个可检索的语义目标

所以，当前“子问题召回效果差”确实有一部分是由 `问题拆分 / 意图理解` 直接造成的，主要表现为：

- 把图路径语法直接泄漏成 query
- 把多个关系或推理说明打包进一个超长子问题
- 没有把问题真正拆成 clause-level、obligation-level、trigger-level 的可检索意图
- 子问题里缺少稳定的 clause / section / event anchor

但同时也不能把问题全部归因给 decomposition，因为还存在另一条独立问题链：

- 即便在子问题已经落到正确文档后
- support chunk 的排序和选择仍可能偏向“语义相近但不是 gold”的 chunk
- 这在 `0002 / 0003 / 0041 / 0042` 上都能看到

因此，当前最准确的根因拆分应是：

1. `子问题拆分 / 意图理解 / query 编译` 仍然有明显问题
2. `子问题 support chunk 排序与选择` 也仍然不够准
3. `子问题 prompt 冗杂` 会进一步放大前两者带来的偏差

#### 12.3.11 后续优化方案修正

基于当前 `t3` 的归因，后续优化顺序需要进一步收敛为 `local-first`，先把单 clause / 单 section / 单 obligation 的检索能力做稳，再考虑是否值得增加 structural 的系统复杂度。

核心原因：

- 当前不少 `structural` 失败样本，本质上并不是“复杂图推理做不到”
- 而是“最基础的 clause / section / trigger / obligation chunk 都没有稳定命中”
- 在这种情况下，如果继续增加 structural 子问题数量或图推理复杂度，只会先放大系统复杂度，不一定先带来收益

因此，`P1-3a` 的方向应修正为：`local-first 的 query 编译与支持证据命中优化`。

主线 A：`P1-3a` local-first 检索意图修复

- 目标：
  - 先提升 `right_doc_wrong_chunk` 样本的 clause/chunk 命中率
  - 让 `structural` 问题也优先退化成可被 local 能力解决的短查询
- 具体动作：
  - 禁止图路径语法、关系枚举、推理说明直接进入 `retrieval_queries`
  - 不优先增加 structural 子问题数量
  - 优先把当前过载的 structural query 编译成更短、更像 local 检索的问题组
  - query 应尽量围绕以下 anchor 组织：
    - `section / clause`
    - `party`
    - `event / trigger`
    - `obligation / remedy / consequence`
  - 删除：
    - 图路径语法
    - 推理型说明语
    - `with an inferred link ...` 之类解释性文字
  - 对很多 structural 问题，先生成类似下面的 local-style 查询，而不是一条过载图 query：
    - `Section 8 breach remedy`
    - `Section 10.3 fail to perform indemnification`
    - `Service Level Credit Event definition`
    - `supplier third-party food safety audits`

主线 B：`P1-3b` 子问题支持证据排序与上下文压缩

- 目标：
  - 在已经命中正确文档的前提下，提高 gold chunk 进入 `support_chunk_ids` 的概率
  - 减少 noisy triple / noisy chunk 对子问题回答的干扰
- 具体动作：
  - 子问题回答 prompt 改成：
    - `Sub-question Evidence Summary`
    - top `3-5` 条清洗后的 relation lines
    - top `2-4` 段聚焦 chunk excerpts
  - 对 support chunk 选择增加 requirement-aware rerank：
    - 对命中 `section / clause / anchor / party / event` 的 chunk 提高分数
    - 对只含泛化实体或外围定义的 chunk 降权
  - 过滤：
    - `[Unknown Node]`
    - triple 元数据
    - `=== Entity Information ===`
    - `community_context`
    - 明显弱相关的法规或定义性外围段落

当前阶段不作为第一优先级的方向：

- 不优先把 structural 问题继续拆成更多子问题
- 只有在 local-first 优化完成后，仍然存在明确的跨 clause / 跨关系桥接失败样本时，才考虑新增更细粒度的 structural decomposition

优先级建议修正为：

- 第一优先级：`P1-3a`
  - 先把 query 编译收敛成短自然语言检索式，优先把 local 检索能力做稳
- 第二优先级：`P1-3b`
  - 再压缩子问题上下文，并提升 support chunk 排序
- 第三优先级：是否需要更复杂的 structural 拆分
  - 仅在 local-first 路线跑完后再判断

验收方式也应相应调整：

- 对 `P1-3a`
  - 重点看子问题 `retrieval_queries` 是否还残留图路径/解释性文本
  - 重点看 query 是否已经收敛成短 local-style 检索语句
  - 重点看 `support_chunk_ids` 是否开始覆盖 gold chunk
- 对 `P1-3b`
  - 重点看子问题 prompt 是否明显变短、变干净
  - 重点看 `faithfulness` 是否继续维持，同时 `answer_correctness` 不再继续下滑

#### 12.3.12 图结构直接清洗后的 `t3` 结果复盘

在当前阶段，曾尝试直接对 [cuad_v3_new.json](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/output/graphs/cuad_v3_new.json) 做一次“主检索图清洗”，包括：

- 删除：
  - `has_attribute`
  - `member_of`
  - `keyword_of`
  - `kw_filter_by`
  - `represented_by`
- 仅保留 `entity -> entity` 三元组
- 删除缺失 `doc_id / chunk_id` 的节点参与的三元组
- 删除明显噪声节点参与的三元组：
  - `entity_type:*`
  - `role:*`
  - `action:*`
  - `unknown / agreement / this agreement / party / clause / event / document / payment / remedy / duration / type / date / location / law`

这次清洗把图从 `20690` 条三元组压到了 `6886` 条。

随后重跑 [ragas_eval_summary.json](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_eval_t3/ragas_eval_summary.json)，相对上一轮 `t2` 得到：

- `answer_correctness`: `0.534929 -> 0.470911`
- `faithfulness`: `0.627085 -> 0.771053`
- `context_precision`: `0.239947 -> 0.296296`
- `context_recall`: `0.247619 -> 0.284656`

这组结果的含义很明确：

- 图清洗确实减少了噪声传播，`context_precision` 上升
- 最终回答更忠实于已有证据，`faithfulness` 明显上升
- 但 `answer_correctness` 反而下降，说明系统没有更接近正确答案，只是“更忠实地使用了当前检到的证据”

这轮最能说明问题的样本有两组。

改善组：

- `0022`
  - `context_precision ≈ 1.0`
  - `context_recall = 0.8`
  - 最终只落在正确合同
- `0023`
  - `context_precision ≈ 1.0`
  - `context_recall = 0.4286`
  - 检索结果明显更干净

这说明图污染确实是问题，尤其对“已经部分命中但噪声过多”的样本，清图是有价值的。

退化组：

- `0021`
  - 最终反而混入 `6` 个文档
  - `context_precision = 0`
  - `context_recall = 0`
- `0041`
  - 仍然没有把关键 gold chunk 检出来
- `0043`
  - 仍然没有解决 wrong-doc / wrong-chunk 问题

这说明图清洗虽然减少了噪声，但也删掉了一部分原本在多跳或弱桥接场景里还能勉强提供召回信号的节点/边。

因此，这轮尝试支持一个更谨慎的结论：

- 图结构确实存在污染，清理是必要的
- 但当前这种“激进裁图”不是主方向
- 因为它能提升：
  - `precision`
  - `faithfulness`
- 却不能解决：
  - `right doc, wrong chunk`
  - `wrong doc`
  - `query 编译不准`

更准确地说，当前系统的主瓶颈已经重新暴露为：

1. `子问题 / query 编译` 仍然不能稳定命中 clause-level intent
2. `support chunk` 排序和选择仍然不够准
3. 图污染只是一层放大器，不是全部根因

所以后续策略需要收敛为：

- 停止继续做大幅裁图
- 图结构清洗后续只做“保守过滤”，不再作为当前主战场
- 重新回到 `local-first` 路线：
  - 优先提升 clause / section / trigger / obligation 的 local 命中率
  - 优先修 `right doc, wrong chunk`

阶段结论修正：

- 当前不再继续把“清图”作为主要优化方向
- 图侧只保留必要的保守过滤思路
- 后续默认主线重新回到：
  - `P1-3a` local-first query 编译
  - `P1-3b` support chunk 选择与子问题上下文压缩

#### 12.3.13 `P1-3a` local-first 检索流程修复已开始实施

本轮没有继续增加 structural decomposition 的复杂度，而是直接回到检索主链路本身，优先解决“编译后的 clean query 被脏原始子问题 query 稀释”和“support chunk 排序不够像 local clause 命中”这两个问题。

已完成的代码改动如下：

- 在 [backend.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/backend.py) 中新增 `retrieval query specificity` 打分与排序逻辑
  - 对 query 中的图路径符号、结构化残片、解释性文本、下划线关系名做降分
  - 对 `Section / Article / clause-like anchor`、短 natural-language local query 做加分
- 子问题检索时，不再无条件把原始 `sq_text` 追加进 `retrieval_queries`
  - 只有当原始子问题文本本身足够像高质量检索 query 时才允许进入候选
  - 这一步是为了避免 structural 样本里那条最脏的解释性 query 再次污染召回
- 新增 `support pair rerank`
  - 根据 `retrieval_queries + retrieval_requirement` 提取 focus terms
  - 对包含 `=== Entity Information ===`、项目符号实体列表等低价值 chunk 做降权
  - 对命中 `Section / anchor / obligation / audit / trigger` 等焦点项的 chunk 做加权
  - 在进入子问题回答与 IRCoT 之前，先把 support chunk 顺序改成更接近 local evidence 的顺序
- 新增 `local lookup field` 显式化规则
  - 对 `location / address / contact / notice / governing law / payment amount / payment due date` 这类 local 字段型需求做启发式识别
  - 将原本的实体词袋 query 改写为 `what is <entity> <field>`、`which clause lists <entity> <field>` 这类显式 local query
  - 例如 `Go Call, Inc. + Cambridge Ontario + location` 不再优先退化成 `go call inc cambridge ontario`，而是优先编译成 `what is Go Call, Inc. address`
- 子问题 prompt 的 triple 证据改成 `compact` 形式
  - 仅清理子问题回答与 IRCoT prompt 中的 raw triple 元数据
  - 去除 `doc_id / schema_type / mention_count / evidence_chunk_ids / [score]`
  - 过滤 `[Unknown Node: ...]`
  - 保持真实检索结果、chunk id、评测落盘不变，避免影响后续 RAGAS 召回指标统计
- IRCoT follow-up query 改成先规范化再检索
  - 不再直接把 `Please retrieve... Ensure the retrieval focuses...` 这类说明文原样送进检索器
  - 会先压缩成 `Section / Article / term` 风格的短 query 列表
  - 例如：
    - `Section 504 Privacy Regulations`
    - `Section 10.5.2 Sensitive Customer Information`
    - `which clause connects Section 504 and Section 10.5.2`
  - 原始 thought 仍保留在 trace 中，但实际检索使用的是规范化后的 query

这一轮的目标不是直接解决全部 structural 失败样本，而是先把检索流程改成：

- 优先信任 clean compiled query
- 少让 noisy raw sub-question 参与召回
- 少让 entity dump / metadata chunk 抢掉 clause evidence 的前排位置

当前状态：

- 代码实现已完成
- 回归测试已通过：`25 passed`
- 尚未基于这轮改动重跑新的 smoke eval，因此是否真正改善：
  - `0002 / 0003 / 0041 / 0042 / 0043` 的 gold chunk 命中率
  - 以及 `0022 / 0023` 的 precision 保持情况
  还需要下一轮评测验证

#### 12.3.14 `retrieval_requirements_smoke_eval_t4` 复盘：IRCoT follow-up 规范化之前的状态

这一轮 [ragas_eval_summary.json](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_eval_t4/ragas_eval_summary.json) 是在 `IRCoT follow-up query normalization` 落地之前得到的结果，因此它可以视为对上一阶段 `P1-3a local-first` 改动效果的直接验证。

相对于 `t3`，`t4` 的整体指标为：

- `answer_correctness`: `0.470911 -> 0.497122`
- `faithfulness`: `0.771053 -> 0.838303`
- `context_precision`: `0.296296 -> 0.305556`
- `context_recall`: `0.284656 -> 0.415344`

这说明上一阶段的改动并不是只改变了 prompt 形式，而是真正带来了可观测的检索收益，尤其体现在：

- 检索覆盖面扩大，`context_recall` 有明显提升
- 回答更建立在已有证据之上，`faithfulness` 持续上升
- 检索噪声没有随 recall 一起明显恶化，`context_precision` 仍小幅提高

从样本与类型分布看，改进主要来自以下几方面。

1. `local lookup field` 显式化带来的单跳字段命中改善

- `what is the address of go call inc in cambridge ontario?`
  - `context_precision ≈ 1.0`
  - `context_recall = 1.0`
- 这说明 `location/address` 类 local 词袋 query 在改写成：
  - `what is <entity> address`
  - `which clause lists <entity> address`
  之后，确实更容易稳定命中 gold chunk

因此，这一类收益主要归因于：

- `local lookup field` 显式化
- `local-first query compilation`

2. `query specificity` 排序与不再无条件追加原始 `sq_text` 带来的 multi-hop recall 提升

`multi_hop_specific_query_synthesizer` 在 `t4` 的变化非常关键：

- `context_recall`: `0.111111 -> 0.388889`
- `answer_correctness`: `0.346388 -> 0.541552`
- `faithfulness`: `0.666305 -> 0.869048`

虽然这一类的 `context_precision` 仍然是 `0.0`，说明 exact chunk 还没完全打准，但它已经从“经常完全检不到”进展到了“至少能打到相关证据区域”。

这类改善最可能来自：

- `retrieval query specificity` 打分
- 不再无条件把 noisy raw sub-question 重新塞回 `retrieval_queries`
- IRCoT 之前的主检索 query 本身已经比早期更像短 clause/section query

因此，`t4` 支持一个重要判断：

- `query 编译` 的确是主瓶颈之一
- 而且此前的 `local-first` query 修正已经开始改善 multi-hop specific 的“区域召回”

3. `support pair rerank` 与子问题上下文清洗带来的 faithfulness 提升

`t4` 的 `faithfulness = 0.838303`，已经显著高于 `t3` 的 `0.771053`。

这说明模型在子问题回答阶段，越来越多地是基于当前真正送进 prompt 的证据作答，而不是被噪声 triples / 弱相关 chunk 带偏。

这部分提升主要应归因于：

- `support pair rerank`
  - 将更像 clause evidence 的 chunk 提到前面
  - 压低 `=== Entity Information ===`、项目符号列表、弱相关 dump
- 子问题 prompt triple 改成 `compact` 形式
  - 去掉 `doc_id / schema_type / mention_count / evidence_chunk_ids / [score]`
  - 过滤 `[Unknown Node: ...]`

需要强调的是，这部分改动：

- 改善的是 `prompt evidence quality`
- 不直接改变检索结果或 RAGAS 的 context id 统计

因此，`faithfulness` 的提升可以合理归因于“提示给模型的证据更干净了”。

4. `t4` 仍然表明 exact chunk precision 还没有解决

尽管 `t4` 总体提升明显，但主要短板仍然清晰：

- `single_hop_specific_query_synthesizer`
  - 只有地址类样本显著成功
  - `Motorola role`、`PageMaster responsibilities` 仍是 `0 / 0`
- `multi_hop_specific_query_synthesizer`
  - `context_recall` 已经起来
  - 但 `context_precision` 仍然是 `0.0`
- `Section 504 / Sensitive Customer Information`
  - `context_precision = 0.0`
  - `context_recall = 0.0`
  - 但 `answer_correctness = 0.678980`
  - 这说明检索仍未真正命中参考证据，只是回答形式看起来较合理

所以，`t4` 最准确的阶段结论是：

- `local-first` 的 query 编译和 support rerank 已经开始起作用
- 系统已经从“经常完全检不到”推进到“更常检到相关区域”
- 但 exact clause / exact chunk 命中仍然不足
- 尤其是 section 型 structural follow-up query 仍然过长、过像说明文

也正因为如此，后续新增的 `IRCoT follow-up query normalization` 是顺着 `t4` 暴露出的下一层问题继续推进，而不是另起一条新路线。

### 12.3.15 P0 `Hybrid Retrieval` 阶段 1 已开始实施

当前已完成第一版最小可行接入，目标是先验证 `dense + sparse + RRF` 是否能改善 first-stage chunk 候选池质量。

本轮代码实现点：

- 在 [enhanced_kt_retriever.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/models/retriever/enhanced_kt_retriever.py) 中新增基于 chunk 文本的 sparse lexical retrieval
- 为 chunk 构建 sparse index cache，并纳入 retriever 生命周期
- 在 `path1.chunk_results` 上引入 `RRF` 融合：
  - dense chunk retrieval
  - sparse chunk retrieval
- 保持现有后续链路兼容：
  - `chunk_ids`
  - `scores`
  - `chunk_contents`
- 同时保留 dense/sparse 来源信息，供后续 trace 与复盘使用

当前判断：

- 这一步只改变 first-stage chunk candidate 生成，不改变后续 RAGAS 的 chunk id 统计口径
- 当前只能确认实现与回归测试通过，还不能确认评测指标收益

已补回归测试并通过：

- `RRF` 融合会把 dense+sparse 同时命中的 chunk 提前
- sparse 为空时，hybrid 会安全回退到 dense-only

### 12.3.16 `retrieval_requirements_smoke_eval_t5` 复盘：Hybrid Retrieval 阶段 1 的真实效果

`t5` 是引入 `Hybrid Retrieval` 阶段 1 之后的第一轮评测结果，核心结论是：

- first-stage chunk candidate 召回确实增强了
- 但新增召回同时带来了新的噪声
- 当前系统已经进入“召回增强有效，但后续 chunk 选择和答案生成还没跟上”的阶段

整体指标相较于 `t4` 的变化：

- `context_precision`: `0.305556 -> 0.316667`
- `context_recall`: `0.415344 -> 0.515344`
- `answer_correctness`: `0.497122 -> 0.461992`
- `faithfulness`: `0.838303 -> 0.668473`

这组变化非常关键，因为它说明：

- `Hybrid Retrieval` 阶段 1 已经证明“first-stage 召回不足”确实可以被改善
- 但它也同步暴露出：当前系统缺的已经不只是 first-stage，而是更强的 second-stage chunk selection

从样本分布看，`t5` 的改善和退化都很典型。

1. `single_hop_specific` 明显受益，说明 hybrid 对 local / clause 级召回有效

这一类在 `t5` 的整体表现：

- `context_precision`: `0.333333 -> 0.533333`
- `context_recall`: `0.333333 -> 1.0`

其中最有代表性的是：

- `ragas-cuad-0002`
  - `context_recall: 0.0 -> 1.0`
- `ragas-cuad-0003`
  - `context_recall: 0.0 -> 1.0`
  - `context_precision: 0.0 -> 0.5`

这说明 hybrid 对以下场景是有效的：

- clause / responsibility / role 相关的局部证据召回
- lexical anchor 与 dense semantic 互补
- dense 过去打不到的 gold chunk，现在更容易进入候选池

因此，这一轮已经可以确认：

- `Hybrid Retrieval` 阶段 1 的方向是对的
- 它不是只增加噪声，而是确实改善了 first-stage recall

2. 但 `answer_correctness` 和 `faithfulness` 回落，说明后续 evidence use 还没跟上

最值得注意的是：

- `ragas-cuad-0002`
  - `context_recall` 已提升到 `1.0`
  - 但 `answer_correctness: 0.400040 -> 0.308674`
- `ragas-cuad-0003`
  - `context_recall` 已提升到 `1.0`
  - 但 `faithfulness: 1.0 -> 0.5`

这说明当前问题已经不再只是“没检到”。

更准确地说，当前系统开始出现下面这种状态：

- first-stage 已经把 gold chunk 拉进候选池
- 但 support chunk 选择、子问题回答和最终聚合，没有稳定把这些新增候选用好

因此，`t5` 强化了一个判断：

- 下一步最值得投入的是 `Stronger Chunk Reranker`
- 因为当前瓶颈已经开始从“召回不够”转移到“召回后怎么选”

3. `multi_hop_abstract` 被额外噪声拖累，说明 hybrid 还需要更强约束

最明显的退化样本是：

- `ragas-cuad-0021`
  - `context_precision: 0.5 -> 0.0`
  - `context_recall: 0.6 -> 0.0`

同时，文档命中范围也从 `t4` 的单文档，退化成 `t5` 的多文档混入。

这说明当前 `Hybrid Retrieval` 阶段 1 还存在一个明确问题：

- sparse 路径对日期、agreement title、termination 这类 lexical signal 的约束还不够
- `RRF` 能提升 recall，但也会把 lexical 邻近的错误合同一起抬进候选池

因此，阶段 1 的正确结论不是“hybrid 已经完成”，而是：

- first-stage recall 已经验证有效
- 但需要更强的 rerank 与 metadata-aware 筛选来消化新增候选

4. `multi_hop_specific` 仍未真正解决，说明 exact bridge chunk 不是只靠 hybrid 就能打通

这一类在 `t5` 的整体表现：

- `context_precision`: 仍然是 `0.0`
- `context_recall`: `0.388889 -> 0.222222`

典型样本：

- `ragas-cuad-0041`
  - `context_recall: 0.5 -> 0.0`
- `ragas-cuad-0043`
  - 仍然 `0 / 0`
  - 且 `answer_correctness` 明显下降

这说明：

- hybrid 更适合补 lexical / section / term recall
- 但对跨 clause、跨 relation 的 bridge evidence，当前仍缺真正的 exact chunk selection 能力

阶段结论应更新为：

- `Hybrid Retrieval` 阶段 1 已成功证明 first-stage recall 可提升
- 但系统还没有获得对应的 second-stage 精准筛选能力
- 所以下一阶段必须优先推进：
  - `Stronger Chunk Reranker`

否则，新增召回会继续同时表现为：

- 一部分真正的 recall 改善
- 一部分新的上下文污染

### 12.3.17 `t5` 对检索路线的进一步修正

基于 `t5` 的结果，以及对当前拆解与检索代码的再次核对，需要把“目标检索链路”的描述进一步收敛。

当前 `retrieval_requirements` 的 schema 是统一的，而不是每个 route 各一套。  
在 [agentic_decomposer.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/models/retriever/agentic_decomposer.py) 中，所有 requirement 最终都会被标准化为：

- `route_type`
- `intent`
- `route_reason`
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

这意味着当前系统不是“先按 route 定义不同的元数据格式”，而是“统一 schema，后续按 route 使用不同字段”。  
这一点本身没有明显问题。

真正的问题出在后半段：

1. `route_type` 虽然参与了 query compilation，但在检索阶段还没有充分拉开主链路差异
2. `local` 仍然会进入 triple-only 支线，而不是彻底 `chunk-first`
3. `structural` 虽然有 bridge query，但检索骨架仍然和 `local` 高度相似
4. `chunk-first` 理应成为主召回路径，但在 `t5` 中并未真正生效，导致很多结果仍然由 triple / graph 支线兜底

所以，当前路线修正应当是：

- 不修改 requirement schema
- 不强调“为不同 route 重新设计元数据格式”
- 而是强调“让统一 schema 在不同 route 下驱动出更清晰的 retrieval 主路径”

具体期望是：

- `local`
  - 以 `chunk-first` 为主
  - 重点依赖 `entities / terms / anchors / intent`
  - triple 只做补充证据
- `structural`
  - 也以 `chunk-first` 为主
  - 但允许 `left_endpoint / right_endpoint / bridge_relation` 深度参与 query planning 与 rerank
  - graph/triple 主要负责 bridge evidence，而不是替代 chunk 召回
- `global`
  - 当前仍主要是 fallback / community 路线
  - 不是接下来的主战场

### 12.3.18 `multi_hop_abstract` 指标最好说明了什么

`t5` 中，`multi_hop_abstract_query_synthesizer` 是当前表现最均衡的一类，见 [ragas_eval_summary.json](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_eval_t5/ragas_eval_summary.json)：

- `answer_correctness = 0.642933`
- `context_precision = 0.666667`
- `context_recall = 0.47619`
- `doc_hit = 1.0`
- `doc_purity = 0.9`
- `support_chunk_hit = 1.0`
- `final_chunk_hit = 1.0`

这说明：

1. 当前系统并不是“不会做多跳”
   - 至少对抽象型多跳问题，系统已经能把正确文档和相关证据区域稳定带进后半段
2. 当前系统已经具备“找到对的地方”的能力
   - `support_chunk_hit = 1.0`
   - `final_chunk_hit = 1.0`
3. 当前真正缺的是“精确找到对的那一段”
   - 与之对比，`multi_hop_specific_query_synthesizer` 仍表现为：
     - `context_precision = 0.0`
     - `context_recall = 0.222222`

因此，`t5` 的启示不是“原方向错了”，而是：

- `chunk-first + graph-assisted` 这条总方向对 `abstract` 已经成立
- 但它对 `specific` 还不够
- 后续必须补上一层面向 exact chunk / exact clause 的精确能力

### 12.3.19 基于 `t5` 的执行顺序再收敛

从 `t5` 往后，执行顺序需要进一步收敛为：

1. 先修 `chunk first-stage` 真正参与检索
   - 当前 `t5` 中：
     - `first_stage_chunk_hit = 0.0`
     - `lightweight_reranked_chunk_hit = 0.0`
   - 说明这轮运行里 chunk 主链路实际上没有参与成功
2. 保留统一 requirement schema，不再动拆解格式
3. 把 `strong chunk reranker` 的目标限定为：
   - 主要服务 `single_hop_specific` 与 `multi_hop_specific`
   - 解决 `right area, wrong chunk`
4. 对 `multi_hop_abstract` 保守优化
   - 这类问题已经说明当前主方向基本成立
   - 后续优化应避免“为了提升 specific 而破坏 abstract”

### 12.3.20 评估指标已补充分层诊断项

当前 `ragas_eval` 流程除了原有的：

- `answer_correctness`
- `faithfulness`
- `context_precision`
- `context_recall`

之外，已经新增一组可直接从当前 trace 和评测产物中计算的分层诊断指标。

新增指标包括：

- `doc_hit`
- `doc_purity`
- `support_chunk_hit`
- `support_chunk_precision`
- `support_chunk_recall`
- `final_selected_chunk_hit`
- `final_selected_chunk_precision`
- `final_selected_chunk_recall`
- `final_chunk_hit`
- `final_chunk_precision`
- `final_chunk_recall`

这些指标已经接入 [run_ragas_eval.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/src/evaluation/run_ragas_eval.py)，会直接出现在：

- `ragas_eval_per_sample.jsonl`
- `ragas_eval_summary.json`
- `ragas_eval_summary.csv`

当前这组指标的作用边界需要明确：

- 它们可以帮助定位：
  - 文档是否命中
  - first-stage chunk 候选池是否已经覆盖 gold chunk
  - lightweight rerank 后 gold chunk 是否仍被保留
  - `support_chunk_ids` 是否覆盖 gold chunk
  - 最终上下文是否覆盖 gold chunk
- 但它们还不能精确区分：
  - 后续 stronger rerank 单独一层的效果

原因是：

- 当前 trace 里已经新增并落盘：
  - `first_stage_chunk_ids`
  - `lightweight_reranked_chunk_ids`
- 但 stronger reranker 这一层还未实现，因此也还没有：
  - `strong_reranked_chunk_ids`

所以当前阶段的最准确判断是：

### 12.3.21 `chunk first-stage` 未生效的根因与修复

在继续推进 `strong chunk reranker` 之前，新增的分层诊断指标暴露出一个更基础的问题：

- `first_stage_chunk_hit = 0.0`
- `lightweight_reranked_chunk_hit = 0.0`

进一步排查 [retrieval_requirements_smoke_t5.jsonl](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_t5.jsonl) 后确认：

- 不是评估脚本没读到字段
- 而是 `retrieval_trace.sub_questions[]` 中：
  - `first_stage_chunk_ids`
  - `lightweight_reranked_chunk_ids`
  本身就是空数组

根因定位到两处实现缺口：

1. `type-filtered` 检索路径会丢失 `chunk_results`
   - 在 [enhanced_kt_retriever.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/models/retriever/enhanced_kt_retriever.py) 中，`_type_filtered_node_relation_retrieval()` 与 `_type_filtered_node_relation_path()` 原先只返回：
     - `top_nodes`
     - `one_hop_triples`
   - 没有把 `_hybrid_chunk_retrieval()` 产生的 `chunk_results` 一起保留下来
   - 结果是只要走到 type-filtered 路径，后续 trace 中的 first-stage / lightweight rerank 相关字段就会天然为空

2. `build_indices()` 没有显式构建 chunk embeddings
   - 这会导致“即使重建索引”，`chunk first-stage` 也可能仍未真正启用

对应修复已经落地：

- `build_indices()` 现在会显式调用 `_precompute_chunk_embeddings()`
- `_type_filtered_node_relation_retrieval()` 现在会保留 `chunk_results`
- `_type_filtered_node_relation_path()` 现在会保留 `chunk_results`
- `_hybrid_type_filtered_retrieval()` 合并多路径 chunk 时，也会把 `path1.chunk_results.chunk_ids` 作为优先来源之一

这次修复的直接意义不是“已经提升了评测指标”，而是：

- 让 `chunk-first` 主链路终于能在 type-filtered / route-aware 场景下真正参与检索
- 让后续新增的：
  - `first_stage_chunk_hit`
  - `lightweight_reranked_chunk_hit`
  终于具备可解释性

因此，后续在评估 `strong chunk reranker` 之前，必须先基于这次修复重新跑问答与评测。

### 12.3.22 重新跑 `t5` 后的结果：`chunk-first` 主链路已接回

在修复 `type-filtered` 路径丢失 `chunk_results`、并让 `build_indices()` 显式构建 chunk embeddings 之后，重新生成并评估了：

- [retrieval_requirements_smoke_t5.jsonl](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_t5.jsonl)
- [ragas_eval_summary.json](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_eval_t5/ragas_eval_summary.json)

这一轮的关键变化非常明确：

- `first_stage_chunk_hit = 0.888889`
- `first_stage_chunk_recall = 0.833333`
- `lightweight_reranked_chunk_hit = 0.888889`
- `lightweight_reranked_chunk_recall = 0.833333`

这说明：

1. 之前 `first_stage/lightweight` 全为 `0` 的现象已经消失
2. `chunk-first` 主链路现在已经真正参与检索
3. 后续再做 `strong chunk reranker`，终于有了可解释的输入候选池

整体检索表现也明显抬升：

- `context_precision = 0.666667`
- `context_recall = 0.706878`
- `doc_hit = 1.0`
- `final_chunk_hit = 0.888889`
- `final_chunk_recall = 0.759259`

其中最关键的阶段判断是：

- `single_hop_specific`
  - `first_stage_chunk_hit = 1.0`
  - `lightweight_reranked_chunk_hit = 1.0`
  - `final_chunk_hit = 1.0`
  - 说明简单 clause / fact 类问题已经基本打通
- `multi_hop_abstract`
  - `first_stage_chunk_hit = 1.0`
  - `lightweight_reranked_chunk_hit = 1.0`
  - `final_chunk_hit = 1.0`
  - 说明当前总路线对 abstract 问题是成立的
- `multi_hop_specific`
  - `first_stage_chunk_hit = 0.666667`
  - `lightweight_reranked_chunk_hit = 0.666667`
  - `final_chunk_hit = 0.666667`
  - 说明 exact chunk / exact clause 仍是当前主要短板

这一轮还有一个非常重要的新结论：

- `first_stage` 与 `lightweight rerank` 的 hit/recall 指标几乎完全相同

这不表示 lightweight rerank 没有执行，而更可能表示：

- 当前 lightweight rerank 主要在做候选重排序
- 但它并没有显著改变 top-k 候选集合的成员

因此，当前 layered diagnostics 已经足够说明：

- `chunk-first` 主链路已经接回
- 当前真正缺的已经不是“有没有 chunk 候选”
- 而是“在已有候选集合里，如何更强地把 gold chunk 排到最前面”

这进一步确认了下一步主线仍然应是：

- `strong chunk reranker`
- 更细的 rank-based 评估（不仅看 hit/recall，也要看排序位置变化）

- `doc / first-stage / lightweight rerank / support / final` 五层已经可以稳定诊断
- stronger reranker 的单独命中情况，需要等该模块落地后再继续补 trace 与指标

### 12.3.23 `strong chunk reranker` 第一版已落地

在确认 `chunk-first` 主链路已经接回之后，当前第一版 `strong chunk reranker` 已经落地到 `support chunk` 选择层，目标不是继续提升 broad recall，而是解决：

- 候选已经召回到位，但 `support top-2` 仍未覆盖最关键 gold chunk
- 同主题跨合同 chunk 混入后，没有足够强的 target-doc 抑制
- section / clause specific 问题里，两个 gold chunk 都在候选池中，但没有同时进入最终支持证据

这版实现的作用域明确限定为：

- 输入：`lightweight rerank` 之后的 chunk/support 候选
- 输出：`strong_reranked_chunk_ids` 与新的 `support_chunk_ids`
- 影响：
  - 子问题回答 prompt 中使用的支持 chunk 顺序
  - `support top-2` 的覆盖质量
- 不影响：
  - first-stage retrieval
  - triple 支线召回
  - final answer prompt 模板本身

当前第一版采用的是 `heuristic strong rerank`，不是 cross-encoder。它显式使用：

- `target_doc_id`
- `anchors`
- `entities / terms`
- `left_endpoint / right_endpoint`
- `bridge_relation`
- `intent`

并在排序时加入：

- target-doc bonus / wrong-doc penalty
- section / article / numeric anchor 命中
- endpoint coverage
- relation / intent alignment
- anchor coverage diversification

也就是说，它的核心用途是：

- 把已经召回到候选池中的正确 chunk 更稳定地推到 `support top-2`
- 特别服务 `0042 / 0041 / 0043` 这类 `specific` 问题

### 12.3.24 `ragas-cuad-0042` 作为 `strong reranker` 的主回归样本

`ragas-cuad-0042` 现在是非常典型的“召回已成功，但证据使用失败”样本。

当前这题已经满足：

- `first_stage_chunk_hit = 1.0`
- `lightweight_reranked_chunk_hit = 1.0`
- `final_chunk_hit = 1.0`
- `support_chunk_hit = 1.0`

这说明：

- Section 8 的 gold chunk 已经被召回
- Section 10.3 的 gold chunk 也已经被召回
- 问题不再是 first-stage retrieval

但这题仍然暴露出两个关键短板：

1. `doc_purity` 仍然偏低，说明跨合同相似 indemnification chunk 仍在混入
2. `support_chunk_ids` 没有稳定把两个最关键的 OFGBANCORP gold chunk 顶到最前

因此，后续评估 `strong chunk reranker` 是否有效时，`0042` 的验收标准应该改成：

- 不是“能不能把 gold chunk 捞进来”
- 而是“能不能把两个 gold chunk 更稳定地排进 support top-2，并压下 wrong-doc indemnification chunks”

### 12.3.25 `t6` 结果复盘：当前主问题已收敛到 `specific query narrowing + top-2 exact ranking`

在接入第一版 `strong chunk reranker` 后，重新评估了：

- [ragas_eval_summary.json](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_eval_t6/ragas_eval_summary.json)
- [ragas_eval_per_sample.jsonl](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_eval_t6/ragas_eval_per_sample.jsonl)

这一轮最重要的结论不是“reranker 已经完成”，而是：

- `first_stage_chunk_hit = 1.0`
- `lightweight_reranked_chunk_hit = 1.0`
- `strong_reranked_chunk_hit = 1.0`

这说明：

1. gold chunk 现在已经能稳定进入候选池
2. 即使经过 strong rerank，也没有在候选层被丢掉
3. 当前主矛盾已经不是“召回不到”

但同时，整体仍然存在两个明显问题：

1. `specific` 类问题的 `context_precision` 仍不稳定  
   说明当前 strong rerank 还更像“保住正确候选”，而不是“稳定把最关键证据排到最前”。

2. `support_chunk_recall` 仍然不够高  
   这说明最终真正喂给子问题回答的 `top-2 support` 还没有稳定覆盖最关键 chunk。

对 `t6` 的正确解读应该是：

- `multi_hop_abstract` 没有被拉坏，反而：
  - `answer_correctness`、`faithfulness`、`context_precision` 都改善
  - 说明第一版 strong rerank 没有破坏 abstract 路线
- 当前真正卡住的是：
  - `single_hop_specific`
  - `multi_hop_specific`

因此，后续优化不应继续泛化成“提升全局 rerank”，而应精确收敛到两件事：

- 收紧 `specific` 子需求编译，让 query 本身更窄
- 把 strong reranker 从“保住 gold”推进到“top-2 exact ranking”

### 12.3.26 下一步实施计划：从“保住 gold”推进到“top-2 exact ranking”

基于 `t6`，后续不再停留在抽象目标，而是执行两条明确、可直接编码的修改。

#### P1. 收紧 `specific` 子需求编译

代码落点：

- [backend.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/backend.py) 中 `_compile_requirement_queries()`

当前问题：

- `specific` 问题仍可能被编译成过宽 query，例如：
  - `section 8 section 10.3`
  - `what clause defines ...`
- 这会让 first-stage 召回到“相关区域”，但同一区域内混入过多近邻 chunk

实施方式：

1. 对 `specific` requirement 强制生成 `narrow query set`，而不是允许裸拼接 query
2. query 只允许来自三类模板：
   - `anchor_queries`
     - 例如：`Section 8`
     - 例如：`Section 10.3 Indemnification Procedures`
   - `term_queries`
     - 例如：`termination fee`
     - 例如：`indemnification procedures`
   - `bridge_queries`
     - 例如：`Section 8 termination fee`
     - 例如：`Section 10.3 notice defense control`
3. 对 `specific` 增加硬约束：
   - 最多输出 `3` 条 query
   - 每条 query 必须命中至少一个 `anchor` 或一个强 `term`
   - 禁止保留 `section 8 section 10.3` 这种裸 anchor 拼接

验收标准：

- `specific` 子问题 trace 中不再出现裸 section 拼接 query
- `retrieval_queries` 明显更窄、更像 exact clause lookup

#### P2. 把 `support top-2` 改成 `anchor-bucket exact ranking`

代码落点：

- [backend.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/backend.py) 中 `_strong_rerank_support_pairs_for_subquestion()`

当前问题：

- strong rerank 已经能保住 gold chunk
- 但还不能保证 `support top-2` 同时覆盖两个关键证据角色

实施方式：

1. 对 `specific + anchors>=2` 的问题，先给每个 chunk 打 `anchor_bucket`
   - 例如命中 `Section 8`
   - 例如命中 `Section 10.3`
2. `support top-2` 不再直接取全局前两名，而是分槽选择：
   - 第 1 条：target doc 内 `anchor A` 分数最高的 chunk
   - 第 2 条：target doc 内 `anchor B` 分数最高的 chunk
   - 若只有一个 anchor 有命中，再用全局最高分补齐
3. 对 wrong-doc chunk 增加更强的补位抑制：
   - 不能让 wrong-doc chunk 在已存在 target-doc anchor 命中的情况下挤掉第二个 anchor 槽位

验收标准：

- `0042 / 0041 / 0043` 中，`support_chunk_ids[:2]` 更稳定覆盖两个关键 anchor 对应 chunk
- `support_chunk_recall` 提升应先于整体 `context_precision`
- 对 `0042` 而言，验收重点是：
  - `support top-2` 是否同时覆盖 Section 8 与 Section 10.3 的 gold chunk

#### 现阶段不做的事

当前暂不继续扩大 strong reranker 的复杂度，不立即上：

- cross-encoder
- ColBERT / late interaction
- 更复杂的 decomposition 分裂

理由是：

- `t6` 已经说明 first-stage 与候选保留问题基本打通
- 当前更值得先修的是 deterministic narrowing 与 top-2 support exact ranking

### 12.3.27 `t7` 结果复盘：系统已进入“support/use 阶段”主导的问题区间

在继续收紧 `specific` query 编译并改进 `support top-2` 选择后，评估了：

- [ragas_eval_summary.json](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_eval_t7/ragas_eval_summary.json)
- [ragas_eval_per_sample.jsonl](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_eval_t7/ragas_eval_per_sample.jsonl)

相较于 `t6`，`t7` 的整体变化是：

- `answer_correctness: 0.448506 -> 0.465858`
- `faithfulness: 0.831755 -> 0.868774`
- `context_precision: 0.503704 -> 0.611111`
- `context_recall: 0.777249 -> 0.795767`
- `support_chunk_hit: 0.666667 -> 0.777778`
- `support_chunk_recall: 0.425926 -> 0.574074`

这说明当前路线仍然是有效的，尤其是：

- `specific query narrowing` 确实让检索候选更窄
- `support top-2` 选择开始更稳定
- `faithfulness` 和 `context_precision` 同时提升，说明 evidence use 的噪声被压下去了一部分

但 `t7` 也非常清楚地表明：

- `first_stage_chunk_hit = 1.0`
- `lightweight_reranked_chunk_hit = 1.0`
- `strong_reranked_chunk_hit = 1.0`

这意味着当前已经不该再把主问题归咎于“召回不到 gold”。

当前真正的主问题已经转移到：

1. `support selection` 仍然不稳定  
   - 典型样本：`0041`
   - gold chunk 已进入 final context，但没有进入 support top-2

2. `final answer synthesis` 仍会对 support 证据过度推断  
   - 典型样本：`0042`
   - support top-2 已经命中两个 gold chunk，但答案归纳仍然过宽

也就是说，`t7` 标志着当前系统进入了一个新的阶段：

- 前半段 retrieval 基本打通
- 后半段 `support/use/generation` 成为主瓶颈

### 12.3.28 `0041 / 0042` 的分层归因

#### `ragas-cuad-0041`

这一题的核心特征是：

- `final_chunk_hit = 1.0`
- `final_chunk_recall = 1.0`
- 但 `support_chunk_hit = 0.0`
- `support_chunk_recall = 0.0`

这说明：

- gold chunk 已经在最终上下文中
- 但 support top-2 没有选到 gold chunk

因此，这题不是 first-stage 问题，而是：

- 子问题编译仍然不够准
- support selection 被错误 query 带偏

具体地说，当前拆出来的子问题仍然过宽：

- `which clause connects Metavante and indemnification`
- `what is Metavante Customer payment amount`
- `Neoforma Metavante Customer indemnification hold harmless liability`

这些 query 没有把原问题里真正关键的：

- `reimbursable expenses`
- `third-party claims`
- `Neoforma indemnification obligations`

稳定保留下来，所以 support chunk selection 在错误的语义重心上做了排序。

#### `ragas-cuad-0042`

这一题的核心特征是：

- `support_chunk_hit = 1.0`
- `support_chunk_recall = 1.0`
- `final_chunk_hit = 1.0`
- `final_chunk_recall = 1.0`

这说明：

- Section 8 的 gold chunk 已进入 support top-2
- Section 10.3 的 gold chunk 也已进入 support top-2

所以 `0042` 已经证明：

- 当前 narrowing + anchor-bucket support selection 是有效的
- 这题已经不再是 support selection 问题

但它仍然存在：

- `answer_correctness` 低
- `context_precision` 低
- `doc_purity = 0.4`

这意味着：

- 最终上下文仍然偏脏
- 更重要的是，final answer 对两段证据进行了过度泛化

因此，`0042` 的主问题已经后移到：

- final answer synthesis 约束不足

### 12.3.29 当前阶段判断：`问题拆解/编译` 与 `final answer 约束` 是新的双核心

基于 `t7`，当前优先级应重新排序为：

1. `specific / structural` 子问题拆解与编译  
   - 因为 `0041` 说明前面仍然会“问错问题”

2. `final answer` 的证据约束  
   - 因为 `0042` 说明即使 support 对了，答案也可能“推得太过”

3. `support selection` 继续细化  
   - 仍重要，但已经不是唯一主矛盾

换句话说：

- 之前的重点是：`retrieval`
- 现在的重点已经变成：`ask the right sub-question + answer only what support proves`

### 12.3.30 后续改进方案：从 retrieval-only 走向 retrieval + evidence-governed synthesis

结合 `t7` 暴露出的现象，后续优化应分成三层。

#### A. 意图识别 / 规划层：从模糊搜索转向确定性指令

目标：

- 不再让 LLM 自由生成过宽子问题
- 而是让 query planning 绑定合同结构和 clause anchor

推荐落地方式：

1. `Contract-aware deterministic schema binding`
   - 在 query planning 阶段引入合同地图（contract map）
   - 强制 `specific` query 至少包含：
     - `contract/doc binding`
     - `clause/section anchor`
     - `keyword/term`

2. 对包含 section 编号、article 编号、defined term 的 query：
   - sparse/BM25 权重应显著高于 dense
   - 防止 `10.3` 被相邻 `10.2` 或泛 indemnification 段落干扰

当前代码可落点：

- [backend.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/backend.py)
  - `_compile_requirement_queries()`
  - requirement-aware query planning 逻辑

#### B. 检索执行层：`slot reservation + bridge chunk`

目标：

- 解决 `support top-2` 虽然有 rerank，但仍可能错过关键 role / anchor 的问题

推荐落地方式：

1. `slot reservation`
   - 对多 anchor 问题，support top-2 必须优先由不同 anchor bucket 占位

2. `bridge chunk boosting`
   - 对同时包含 `anchor A + anchor B` 的 cross-reference / bridge chunk 单独加分
   - 尤其适合法律合同中的交叉引用条款

当前代码可落点：

- [backend.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/backend.py)
  - `_strong_rerank_support_pairs_for_subquestion()`

#### C. 生成前自校准：evidence-governed synthesis

目标：

- 防止 `0042` 这类“support 正确，但答案推得太过”

推荐先落轻量版，而不是一开始就做完整 CoVe：

1. `claim-source consistency check`
   - 最终答案的核心 claim，必须能在同一 `doc_id` 的 support chunk 中找到支撑

2. `cross-doc conflict guard`
   - 如果关键 claim 的支撑 chunk 来自不同文档，优先降级回答为：
     - `insufficient evidence`
     - 或只回答 support 明示的最保守结论

当前代码可落点：

- [backend.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/backend.py)
  - final answer prompt 构建
  - final answer 生成后的 post-check

#### 12.3.30.1 已实施（第一批）

已在 [backend.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/backend.py) 落地第一批改动：

1. `Contract-aware deterministic schema binding`
   - 在 `_compile_requirement_queries()` 中增加 `contract-aware` 软绑定 query。
   - 绑定信息来自用户问题中的 `Contract title` 和 `dated ...` 等合同锚点。
   - 采用“增强而非替代”策略：
     - 新增 `doc-bound` query 变体；
     - 原有 `soft query` 保留，用于 recall 兜底，避免因 anchor 识别失败导致检索归零。

2. `bridge chunk boosting`
   - 在 `_strong_rerank_support_pairs_for_subquestion()` 中对同时命中多个 anchor 且显式包含 cross-reference / bridge 信号的 chunk 加分。
   - 典型信号包括：
     - `under`
     - `in accordance with`
     - `subject to`
     - `as provided in`
     - `upon termination`
     - `failure to`
     - `third-party`

3. `slot reservation` 风险控制
   - 对多 anchor 问题继续保留 bucket 化 support 选择；
   - 同时增加最低相关度阈值：
     - 若某 anchor bucket 的最佳 chunk 分数低于阈值，则不强行占位；
     - 避免把低质量 `B` bucket 噪声硬塞进 support top-2。

4. `lightweight evidence consistency check`
   - final answer 阶段会先按 `preferred_doc_ids` 过滤 backing chunks，优先使用同一合同的 support。
   - 若最终上下文仍出现明显跨文档混入，则追加一次轻量 revision：
     - 保留同 doc backing chunks；
     - 要求模型只保留 backing chunks 明示支持的结论；
     - 不再允许把弱相关跨条款线索泛化成强因果结论。

### 12.3.31 优先级修正

当前最优先的工作顺序应调整为：

1. 继续收紧 `specific / structural` query planning
2. 给 support selection 增加更强的 bridge chunk 识别
3. 给 final answer 增加轻量版 evidence consistency check

当前暂不优先做：

- 更激进的图改造
- 完整 CoVe
- 重型 cross-encoder / late interaction

因为：

- `t7` 已经说明 retrieval 主链路基本可用
- 当前最短板是 planning 和 synthesis 两端

### 12.3.32 retrieval_requirements_smoke_eval_t8 复盘

结合 [ragas_eval_summary.json](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_eval_t8/ragas_eval_summary.json) 和 [ragas_eval_per_sample.jsonl](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_eval_t8/ragas_eval_per_sample.jsonl)，`t8` 的状态可以概括为：

- `coverage` 继续上升；
- `doc purity` 继续改善；
- 但 `precision` 没有同步收紧；
- 并且 `single_hop_specific` 的 `support top-2` 出现系统性退化。

关键指标变化：

- `answer_correctness = 0.528492`
- `faithfulness = 0.782724`
- `context_precision = 0.558642`
- `context_recall = 0.833862`
- `doc_hit = 1.0`
- `doc_purity = 0.832099`
- `first_stage_chunk_hit = 1.0`
- `strong_reranked_chunk_hit = 1.0`
- `support_chunk_hit = 0.666667`
- `support_chunk_recall = 0.537037`

这说明：

1. `retrieval` 主链路依然有效  
   - `first_stage/lightweight/strong reranked hit` 全部为 `1.0`
   - 说明 gold chunk 基本都能进入候选集并被保住

2. `precision` 相关指标仍不理想，但含义需区分  
   - `first_stage_chunk_precision = 0.029023`
   - `strong_reranked_chunk_precision = 0.023657`
   - 这两项仍然低，主要是因为评估对象是整条候选列表，而 gold chunk 通常只有 `1-2` 条，不能直接等价解释为“检索失败”

3. 当前真正的异常点在 `support selection`  
   - `single_hop_specific_query_synthesizer`：
     - `support_chunk_hit = 0.0`
     - `support_chunk_recall = 0.0`
     - 但同时：
       - `final_selected_chunk_hit = 1.0`
       - `final_chunk_hit = 1.0`
       - `context_precision = 0.777778`
       - `context_recall = 1.0`
   - 这说明：
     - gold chunk 最终能进入 final context；
     - 但 `support top-2` 被新的 support 逻辑带偏；
     - 当前 support selector 对 `single-hop exact fact/role lookup` 并不适配

### 12.3.33 t8 暴露出的新增问题

#### A. 子问题编译仍会失真

从 [retrieval_requirements_smoke_t8.jsonl](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_t8.jsonl) 可见，当前子问题编译仍会出现两类失真：

1. `漏掉关键实体 / 关系焦点`
   - 例如 Village Media 那题里，`NFL / National Football League` 没有进入主 structural 子问题的 query 骨架，只被塞进了一个较弱的 local-style query 分支

2. `添加原句中不存在或过度泛化的限定词`
   - 例如 `promotion period`
   - 例如 `term effective from to`
   - 这会使 sparse/BM25 精确匹配偏离真正的 gold clause

结论：

- 当前 `specific/structural` query planning 仍会“问偏问题”
- 这会直接把后续 rerank 的候选空间带宽

#### B. `Doc Binding` 目前是 soft bias，不是 hard filter

代码核实结果：

- retriever 内部确实会根据问题推断 `target_doc_id`，见 [enhanced_kt_retriever.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/models/retriever/enhanced_kt_retriever.py#L2629)
- `target_doc_id` 也确实会传入 chunk retrieval / rerank 路径，见 [enhanced_kt_retriever.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/models/retriever/enhanced_kt_retriever.py#L2635)
- 但当前实现是：
  - `boost / prioritize`
  - `preferred_doc_ids bias`
  - 而不是 strict filtering

对应落点：

- [enhanced_kt_retriever.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/models/retriever/enhanced_kt_retriever.py#L3860)
- [enhanced_kt_retriever.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/models/retriever/enhanced_kt_retriever.py#L3914)
- [enhanced_kt_retriever.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/models/retriever/enhanced_kt_retriever.py#L3970)
- [enhanced_kt_retriever.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/models/retriever/enhanced_kt_retriever.py#L4009)

因此，当前系统在检索层仍然遵循：

- `谁分高谁优先`
- 而不是：
  - `先目标合同内 top-k，再全局补位`

这也是为什么：

- wrong-doc chunk 仍能混入 final context
- `support top-2` 会被高分近邻 chunk 挤掉
- `final_chunk` 往往能把 gold 带回来，但 `support_chunk` 和 `precision` 仍不稳定

#### C. 当前 support 逻辑对 `single_hop_specific` 不适配

`t8` 说明：

- `multi-hop` support 逻辑增强后，整体 coverage 是上去了
- 但 `single-hop exact lookup` 被误伤

典型现象：

1. `ragas-cuad-0001`
   - gold 是 `c480fd67-...`
   - support 选成了 `3ebc9465-...` 和 `96f0cc7d-...`
   - 最终能答对地址，但 support 对 gold 命中为 `0`

2. `ragas-cuad-0002`
   - gold 在 final context 中
   - 但 support top-2 没选中 gold role chunk

3. `ragas-cuad-0003`
   - gold 在 final context 中
   - 但 support top-2 仍被邻近“promotion period”相关 chunk 带偏

结论：

- 当前 support 逻辑过于偏向：
  - anchor diversity
  - bridge semantics
  - broad same-doc coverage
- 但 `single_hop_specific` 真正需要的是：
  - `top-1 exact clause / exact fact`

### 12.3.34 当前阶段判断（基于 t8 修正）

截至 `t8`，当前系统状态应修正为：

1. `retrieval` 主链路不是主瓶颈  
   - gold chunk 通常已经能进入 first-stage / rerank / final context

2. `planning` 重新上升为最高优先级  
   - 因为子问题编译仍会：
     - 漏掉关键实体
     - 添加错误限定词
     - 把问题压成过宽或偏移的 sparse query

3. `support selection` 需要按问题类型分叉  
   - `single_hop_specific` 应走：
     - `exact-single-clause priority`
   - `multi_hop_specific` 才继续走：
     - `anchor buckets + bridge chunk boosting`

4. `doc binding` 还没有真正进入 strict retrieval mode  
   - 当前只是 soft bias
   - 这意味着 cross-doc pollution 仍可能在高分候选阶段出现

### 12.3.35 下一步实施重点（t8 后）

后续优先级应进一步收敛为：

1. `specific / structural` query planning 继续收紧  
   - 保留关键实体和关系词
   - 禁止无依据新增限定词
   - 对 `NFL / third-party / reimbursable / termination fee / notice procedure` 这类高价值词做显式保留

2. support selection 按类型分流  
   - `single_hop_specific`：
     - 单独实现 `exact-single-clause support selection`
     - 不再复用多 anchor diversification 逻辑
   - `multi_hop_specific`：
     - 保留当前 bridge-aware support 逻辑

3. 检索层新增 `strict target-doc mode`
   - 对高置信 doc-bound query：
     - 先在目标合同内取 top-k
     - 再全局补位
   - 目标是把当前 `soft doc binding` 升级成真正可控的检索约束

### 12.3.36 已实施：分层 query planning 兜底

针对“用户问题较模糊时，hard binding 可能导致 recall 归零”的风险，已在 [backend.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/backend.py) 的 `_compile_requirement_queries()` 中补上分层 query planning：

1. `exact queries`
   - 保留 section / clause / hard term / contract-aware 的窄 query

2. `soft queries`
   - 保留自然语言化的 clause / intent 查询

3. `fallback queries`
   - 在 `max_query_variants` 内显式保留更宽的 endpoint / subject 查询
   - 避免 hard term（如 `reimbursable expenses`）垄断 recall

当前策略含义：

- `hard binding` 只用于提升 precision
- 不再作为唯一入口
- 对模糊问题仍保留 broader recall guard

### 12.3.37 retrieval_requirements_smoke_eval_t9 复盘

结合 [ragas_eval_summary.json](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_eval_t9/ragas_eval_summary.json) 和 [retrieval_requirements_smoke_t9.jsonl](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_t9.jsonl)，`t9` 的状态比 `t8` 明显退化，尤其体现在 `NOT_FOUND` 子问题数量和 `multi_hop_specific` 表现上。

整体指标：

- `answer_correctness = 0.41506`
- `faithfulness = 0.784768`
- `context_precision = 0.505688`
- `context_recall = 0.740212`
- `doc_purity = 0.808642`
- `support_chunk_hit = 0.777778`
- `support_chunk_recall = 0.555555`

与 `t8` 相比，主要问题不是主检索链路失效，而是：

1. 子问题 `NOT_FOUND` 数量明显偏多  
   - 共 `19` 个子问题
   - 其中 `9` 个为 `NOT_FOUND`
   - `NOT_FOUND ratio = 0.4737`

2. 典型失败样本：
   - Village Media / NFL 题：`1/3` 子问题 `NOT_FOUND`
   - Supplier insurance / audits 题：`1/3` 子问题 `NOT_FOUND`
   - Metavante / Neoforma reimbursable expenses 题：`3/3` 子问题 `NOT_FOUND`
   - Section 8 / 10.3 indemnification 题：`1/1` 子问题 `NOT_FOUND`
   - Section 504 / Sensitive Customer Information 题：`3/3` 子问题 `NOT_FOUND`

3. 这说明当前新增的 query planning fallback 并没有把 recall 稳住，反而在若干 hard-term / doc-bound 问题上把 query 进一步推向了：
   - 更偏标题化
   - 更偏术语化
   - 更容易触发 `NOT_FOUND`

### 12.3.38 对 12.3.36 的纠偏说明

`12.3.36` 的实施内容与 `12.3.35` 的规划目标存在明显偏差，这一点必须明确记录。

`12.3.35` 的原规划是：

1. 继续收紧 `specific / structural` query planning  
   - 核心是保留关键实体和关系词
   - 核心是禁止无依据新增限定词

2. support selection 按类型分流  
   - `single_hop_specific` 走 `exact-single-clause priority`
   - `multi_hop_specific` 才继续走 bridge-aware support

3. 检索层新增 `strict target-doc mode`
   - 先目标合同内检索，再全局补位

但 `12.3.36` 实际实施的是：

- 在 `_compile_requirement_queries()` 里增加 `exact + soft + fallback` query 组
- 这属于 query planning 的一个局部兜底策略
- 它并没有完成 `12.3.35` 中真正更关键的两项：
  - `single-hop support` 分流
  - `strict target-doc mode`

因此，`12.3.36` 不是 `12.3.35` 的完整落地，最多只能算：

- `12.3.35.1` 的一个试探性子步骤
- 而且从 `t9` 看，这一步的收益并不稳定，甚至带来了额外的 `NOT_FOUND` 问题

### 12.3.39 当前纠偏结论

基于 `t9`，当前应停止把 `fallback query planning` 继续当作主方向推进，原因是：

1. 它没有解决 `12.3.35` 里定义的主要矛盾
   - 没解决 `single-hop support` 分流
   - 没解决 strict `target-doc` 约束

2. 它反而把一部分 query 推向了：
   - 标题式
   - 合同名拼接式
   - 容易 `NOT_FOUND` 的方向

3. 当前更该优先做的仍然是 `12.3.35` 的原计划，而不是继续扩写 query 变体

### 12.3.40 修正后的优先级

`t9` 后应恢复到如下优先级：

1. 实现 `single_hop_specific` 专用 support 选择  
   - 目标：`top-1 exact clause / exact fact`
   - 不再套用多 anchor / bridge diversification 逻辑

2. 实现 `strict target-doc mode`
   - 对高置信 doc-bound query：
     - 先目标合同内 top-k
     - 再全局补位

3. 回头重构 `specific / structural` query planning
   - 不是继续加更多 fallback query
   - 而是回到：
     - 保留关键实体
     - 保留关键关系词
     - 禁止无依据新增限定词

### 12.3.41 已实施：回到 12.3.35 主线

本轮代码实施已明确回到 `12.3.35` 中真正关键的两项，而不再继续扩展 `12.3.36` 的 layered query fallback。

实施位置：

- [backend.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/backend.py)
- [enhanced_kt_retriever.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/models/retriever/enhanced_kt_retriever.py)

#### A. `single_hop_specific` / local exact support 选择分流

在 [backend.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/backend.py) 的 `_strong_rerank_support_pairs_for_subquestion()` 中新增 `single_hop_exact_mode`：

- 触发条件：
  - `route=local`
  - 或 `route=structural` 但：
    - anchor 数量不超过 `1`
    - 不存在 `bridge_relation`
    - intent 更像 `fact_lookup / definition_lookup / role_lookup / address / notice / payment amount`

- 该模式下的行为：
  - 不再执行多 anchor bucket reservation
  - 不再做 bridge-aware diversification
  - 明显提高 `target_doc_match` 与 exact multi-word focus term 的权重
  - 对错文档 chunk 加更强惩罚
  - 直接按 exact-single-clause / exact-fact 分数排序输出

这一步的目标不是提高 `coverage`，而是避免 `single_hop_specific` 被 `multi-hop` support 逻辑误伤。

#### B. 检索层新增 `strict target-doc mode`

在 [enhanced_kt_retriever.py](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/models/retriever/enhanced_kt_retriever.py) 中新增 `_should_use_strict_target_doc_mode()`，并将其接入：

- `retrieve_by_route(...)`
- `_type_based_retrieval(...)`
- `_type_filtered_node_relation_retrieval(...)`
- `_type_filtered_node_relation_path(...)`
- `_hybrid_type_filtered_retrieval(...)`
- `_node_relation_retrieval(...)`
- `_parallel_dual_path_retrieval(...)`
- `_hybrid_chunk_retrieval(...)`
- `_chunk_embedding_retrieval(...)`
- `_sparse_chunk_retrieval(...)`

严格模式仅在以下条件满足时触发：

- 已成功推断出 `target_doc_id`
- 且 `question` / `original_question` 中存在高置信合同绑定信号，例如：
  - `Contract title:`
  - 明确 agreement/document 提示
  - 或归一化 `doc_id` 命中

严格模式行为：

- 若目标合同内已有命中 chunk：
  - `dense` 和 `sparse` first-stage 都只保留目标合同 chunk
- 若目标合同内没有命中 chunk：
  - 自动退回原有 soft behavior
  - 仍允许全局补位，避免 recall 归零

这一步落实了用户前面强调的风险控制：

- 不做无条件 hard filter
- 只在高置信 doc-bound 问题上启 strict mode
- 并保留 target-doc miss 时的 fallback

### 12.3.42 本轮实施后的预期观察点

下一轮评测应重点观察：

1. `single_hop_specific_query_synthesizer`
   - `support_chunk_hit`
   - `support_chunk_recall`
   - 是否从 `0` 恢复

2. cross-doc pollution
   - `doc_purity`
   - local / single-hop 样本的 `support_chunk_ids`
   - final backing chunk 是否仍混入错合同

3. `NOT_FOUND` ratio
   - 本轮没有继续加大 query fallback 复杂度
   - 理论上不应再像 `t9` 那样因为 query 变形导致明显的 `NOT_FOUND` 扩散

### 12.3.43 retrieval_requirements_smoke_eval_t10 复盘

结合 [ragas_eval_summary.json](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_eval_t10-auto/ragas_eval_summary.json) 和 [ragas_eval_per_sample.jsonl](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_eval_t10-auto/ragas_eval_per_sample.jsonl)，`t10-auto` 的状态可以概括为：

- `NOT_FOUND` 子问题显著减少
- gold chunk 在大多数样本上已经能进入：
  - first-stage
  - lightweight rerank
  - strong rerank
  - final context
- 但 precision、support 纯度和最终回答质量并未同步收敛

这说明当前系统已经不再主要卡在“能不能捞到 gold evidence”，而是转向：

1. `query planning` 是否把问题问对
2. `support top-2` 是否把最关键 chunk 顶到最前
3. final answer 是否严格受 support 证据约束

同时，`strict target-doc mode: auto` 在这一轮并未证明自己是稳定增益项，反而暴露出：

- 一部分问题会被推向 `NOT_FOUND`
- 因此它不应继续作为默认主策略

### 12.3.44 三个代表样本揭示的当前问题分层

本轮最有代表性的三个样本是：

- `ragas-cuad-0041`
- `ragas-cuad-0002`
- `ragas-cuad-0042`

它们分别对应当前系统的三类主要残余问题。

#### 12.3.44.1 `ragas-cuad-0041`：子问题编译失真，support 在错误目标上优化

对应产物：

- [retrieval_requirements_smoke_t10-auto.jsonl](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_t10-auto.jsonl)

现象：

- gold chunk `c5c9acb9-...` 与 `4be9c784-...` 已进入：
  - first-stage
  - strong rerank
  - final context
- 但 `support_chunk_hit = 0.0`

原因不是 reranker 没保住 gold，而是 query planning 先把问题问偏了。原问题关心：

- `reimbursable from third parties`
- `third-party claims`

但子问题被压成了：

- `Metavante expenses`
- `payment amount`
- `liability responsibility risk allocation`

因此 strong reranker 实际是在一个已经失真的候选空间里做优化，support top-2 被错误方向占据。

结论：

- `0041` 的主问题在前段
- 优先修复方向是 `specific/multi-hop` 的关系词保留与 query compilation 约束

#### 12.3.44.2 `ragas-cuad-0002`：single-hop exact ranking 不足，gold 被 near-miss 挤出 top-2

对应产物：

- [retrieval_requirements_smoke_t10-auto.jsonl](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_t10-auto.jsonl)

现象：

- gold chunk `9dcb9832-...` 已进入：
  - first-stage
  - lightweight rerank
  - strong rerank
  - final context
- 但 `support_chunk_hit = 0.0`
- `support_chunk_ids` 被同合同、同主题的 near-miss chunk 占据

同时该样本里仍可见：

- `target_doc_id = None`
- `preferred_doc_ids = []`

说明当前 `single_hop_exact_mode` 虽已存在，但 same-doc exactness 仍不够强，gold 还会被更泛化、但语义很近的 chunk 挤出 top-2。

结论：

- `0002` 的主问题在中段
- 优先修复方向是 `single-hop` 的 exact top-1 / top-2 支持证据选择

#### 12.3.44.3 `ragas-cuad-0042`：support 已正确，但 final answer 仍然写偏

对应产物：

- [retrieval_requirements_smoke_t10-auto.jsonl](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_t10-auto.jsonl)

现象：

- `strong_reranked_chunk_ids` 前两名就是 gold：
  - `3ed0bde8-...`
  - `ffaffba5-...`
- `support_chunk_ids` 也是这两个
- 指标上：
  - `support_chunk_precision = 1.0`
  - `support_chunk_recall = 1.0`
  - `final_chunk_recall = 1.0`

但同时：

- `answer_correctness` 仍低
- `faithfulness` 仍低
- `final_chunk_precision` 很差

这说明：

- reranker 对 `0042` 已经足够
- 当前失败不是“top-2 没选到”
- 而是 final context 仍偏宽，且 final answer 没有被 support 证据严格约束

结论：

- `0042` 的主问题在后段
- 优先修复方向是 final answer 的 evidence-constrained synthesis

### 12.3.45 当前阶段判断（基于 `0041 / 0002 / 0042`）

当前系统已经不是单点故障，而是三段式问题：

1. 前段：`问偏了`
   - 代表样本：`0041`
   - 本质：子问题编译仍会丢关键关系词、泛化关系意图

2. 中段：`问题大体对，但 top-2 没排准`
   - 代表样本：`0002`
   - 本质：single-hop rerank 仍会让 near-miss chunk 挤掉 gold

3. 后段：`support 已经对了，但答案还会写偏`
   - 代表样本：`0042`
   - 本质：final answer 对 support 证据的约束不够强

这也解释了为什么当前会同时出现：

- `chunk_recall` 很高
- `chunk_precision` 仍低
- 最终回答质量却仍不理想

系统现在更像是：

- 能把 gold chunk 捞进来
- 但不能稳定把非 gold 压出去
- 即便 support 已正确，答案也还会过度展开或过度推断

### 12.3.46 后续改进方向（t10 后）

基于 `t10`，后续应按如下顺序推进：

1. 优先修 `specific/multi-hop` 的 query planning
   - 强制保留原问题中的关键关系词
   - 例如：
     - `reimbursable`
     - `third-party claims`
     - `exclusive communication`
     - `notice`
   - 禁止被泛化为：
     - `payment amount`
     - `liability responsibility`
     - `agreement connection`

2. 再修 `single-hop` 的 support 选择
   - 目标不再是“保住 gold”
   - 而是“把 gold 顶到 top-1 / top-2”
   - 强化：
     - exact multiword term match
     - same-doc near-miss suppression
     - top-1 exact-first，再决定是否补第 2 条

3. 最后修 final answer 的证据约束
   - `single-hop`：只回答最直接命中的核心事实
   - `multi-hop specific`：只总结 `support_chunk_ids` 明示的关系
   - 若两段证据之间没有显式桥接句，则输出保守表述，不补逻辑

### 12.3.47 已实施：strong reranker diagnostics 落盘

为了不再盲调 reranker，本轮已在 strong rerank 阶段为每个候选 chunk 增加可解释分项打分，并落入 trace。

新增字段：

- `retrieval_trace.sub_questions[].strong_rerank_diagnostics`

每个候选会保留：

- `chunk_id`
- `doc_id`
- `score`
- `matched_anchors`
- `score_breakdown`

其中 `score_breakdown` 拆分为：

- `prior_score`
- `doc_score`
- `term_score`
- `anchor_score`
- `bridge_score`
- `exactness_score`
- `redundancy_penalty`
- `noise_penalty`

### 12.3.48 阶段收敛：已证伪 / 已回滚的路线

截至真实接口复测（重点样本：`ragas-cuad-0041`），以下路线应明确降级，不再作为当前主线继续扩写：

1. `strict target-doc mode: auto`
   - 该策略在离线评测与真实问答中都表现出明显副作用：
     - 一部分样本被推向 `NOT_FOUND`
     - 问题并没有因此稳定收敛到目标合同
   - 当前结论：
     - 只能保留为实验开关
     - 不能作为默认检索策略

2. `support answer fallback`
   - 这条路线试图在 `sub_answer = NOT_FOUND` 时，从 support chunk 中启发式抽取答案
   - 真实接口验证表明：
     - 它会掩盖真正的 query / rerank 问题
     - 容易把“近似相关句”误包装成已回答
   - 当前结论：
     - 已从代码中移除
     - 后续不再以“抽取兜底”方式掩盖子问题失败

3. `layered fallback query planning` 作为主线
   - `exact + soft + fallback` 的分层 query 只能作为 recall 保护思路存在
   - 它没有解决当前最核心的问题：
     - 关键关系短语丢失
     - support top-2 不够纯
   - 当前结论：
     - 不再继续扩张 query 变体数量
     - 当前主线改为“更准”，而不是“更多”

### 12.3.48.1 纠偏说明：被证伪的是 `strict target-doc` 的实现方式，不是 `target-doc alignment` 方向

需要明确区分两件事：

1. 被证伪的是：
   - `strict target-doc mode` 在 `first-stage chunk retrieval` 上做过早硬裁剪
   - 典型实现是：
     - 只要目标合同内存在候选，就直接丢弃其他合同候选
   - 这一实现会导致：
     - recall 下降
     - 一部分样本被推向 `NOT_FOUND`

2. 没有被证伪的是：
   - `target-doc alignment`
   - 即：
     - 目标合同识别
     - same-doc bias
     - final / support 阶段的同合同优先

真实接口最新结果已经证明：

- 当 `target_doc_id` 只作为 `support rerank / final chunk selection` 的同合同 bias 使用时，
- 它是正向的：
  - wrong-doc 污染下降
  - `doc_score` 生效后，同合同 chunk 更稳定排到前面

因此当前结论应修正为：

- 不应在 `first-stage` 使用硬式 `strict target-doc gating`
- 但应继续加强：
  - `target_doc_id` 识别
  - `support rerank` 的 same-doc bias
  - `final chunk selection` 的同合同优先

换言之，`strict target-doc mode` 的失败，不代表“数据隔离/合同对齐”方向错误；它只代表：

- `first-stage hard gating`

这个实现位置和强度是错误的。

### 12.3.49 真实接口校正后的当前唯一主线

在去掉上述错误分支后，当前问题已经收敛成两个可执行主项。

#### 12.3.49.1 Query planning：保留完整关系短语，而不是泛化关系意图

`0041` 的真实接口表现说明，当前最大的前段问题仍然是：

- 子问题没有保住完整关系短语
- 例如只剩：
  - `Metavante reimbursable`
  - `Metavante services Customer reimbursable`
- 而没有保住：
  - `reimbursable from third parties`
  - `third-party claims`

因此当前 query planning 的唯一主线应是：

- 对 `specific / local / structural` 问题保留完整关系短语
- 抑制泛化表达：
  - `payment amount`
  - `agreement connection`
  - `liability responsibility`
- 优先保证：
  - 原问题里的关键法律关系词不丢
  - 原问题中的关键实体不丢

换言之，当前 query planning 的目标不再是“生成更多 query”，而是：

- `少生成`
- `但每条更贴近原问题真正想确认的 clause intent`

#### 12.3.49.2 Support rerank：让 `target_doc_id` 真正进入打分

最新真实接口结果还暴露出一个非常明确的实现缺口：

- `target_doc_id` 经过修复后已经能更稳定识别
- 但 `strong_rerank_diagnostics` 中 `doc_score` 仍长期为 `0.0`

这意味着：

- 目标合同绑定虽然识别出来了
- 但并没有真正转化成 support 排序中的 same-doc 加分

因此当前 rerank 的唯一主线应是：

- 把 `target_doc_id` 显式打进 `_strong_rerank_support_pairs_for_subquestion()` 的排序逻辑
- 让 same-doc bias 真正影响：
  - support top-2
  - final backing chunks

在这件事修通之前，不应再继续扩写新的 retrieval 分支。

### 12.3.50 文档使用约束：后续记录只围绕主线更新

从本节开始，长期文档的使用方式应收敛为：

1. 新增实验时，先判断它属于：
   - 主线改进
   - 还是支线试探

2. 若实验被真实接口或评测明确证伪：
   - 直接写入“已证伪 / 已回滚”
   - 不再在“后续主线”里继续保留

3. 当前允许继续推进的主线只有两条：
   - `完整关系短语 query planning`
   - `target_doc_id -> support rerank doc_score 打通`

4. 在这两条主线没有打通前：
   - 不继续扩张 fallback
   - 不继续扩张 query 变体
   - 不继续引入新的中间兜底层

这样做的目的不是减少实验，而是避免问题空间继续发散，导致方向再次偏离当前真实瓶颈。

意义：

- 之后可以直接定位：
  - gold 没进 top-2 是被 wrong-doc 压了
  - 还是被 same-doc near-miss 压了
  - 还是被 bucket / 冗余 / 噪声带偏了

这一步不直接改变指标，但补齐了 `strong rerank precision` 低时的可观测性。

### 12.3.51 `0041 / 0042` 的系统性复盘：不是两个 bug，而是三段失真

最新真实接口表现说明，`0041` 和 `0042` 不是两个孤立 bug，而是同一套链路在不同阶段的两种失真。

#### 12.3.51.1 当前系统的三段失真

1. 前段失真：`问题表示` 被压成“语义相关 query”，而不是“待验证命题”
   - `0041` 的核心不是检索不到，而是系统没有稳定保住：
     - `reimbursable from third parties`
     - `third-party claims`
   - 这说明当前 decomposition / query planning 仍然更像“关键词改写”，而不是“法律关系建模”

2. 中段失真：`support` 仍然是排序产物，而不是论证结构
   - 当前 support 的主逻辑仍然是：
     - 分数高
     - same-doc 优先
     - 少量多样化
   - 但合同问答真正需要的是：
     - 事实前提 chunk
     - 关系/定义 chunk
     - 桥接或后果 chunk
   - 也就是说，support 需要承担“证据角色”，而不只是“前两名”

3. 后段失真：`final answer` 建立在“相关证据集合”上，而不是“最小充分证据集合”上
   - `0042` 已经证明：
     - support top-2 可以是对的
     - 但 final answer 仍然会过度推断
   - 这说明系统当前默认假设仍是：
     - “只要相关证据给到模型，模型会自己做对归纳”
   - 对法律/合同场景，这个假设不成立

#### 12.3.51.2 `0041` 揭示的真正问题：系统不知道要证明什么关系

`0041` 的主问题不是某个 chunk 没被召回，而是：

- 原问题想确认的是：
  - 是否存在 third-party reimbursement mechanism
  - 以及它与 Neoforma indemnification 的关系
- 但系统会把它压成：
  - `Metavante reimbursable`
  - `payment due date`
  - `indemnification obligations`

这说明系统当前没有把问题表示成：

- `claim`
- `required_relation`
- `required_entities`
- `answerable_if`

而只是表示成一组可检索 query。

因此，`0041` 的本质不是简单 query bug，而是：

- 系统把“复杂法律关系问题”错误降维成了几个语义相关词团

#### 12.3.51.3 `0042` 揭示的真正问题：系统拿到正确证据后仍然会写偏

`0042` 的主问题不是 reranker 失败，而是：

- support top-2 已经正确
- 但 final answer 仍然会把相邻条款写成更宽的逻辑结论

这说明系统当前缺少：

- 显式 bridge evidence 检查
- support 证据边界约束
- “无明示关系时必须保守”的回答纪律

因此，`0042` 的本质不是生成模型太差，而是：

- 系统把 final answer 当成“综合概括”
- 用户真正需要的却是“证据约束下的法律关系判断”

### 12.3.52 当前主线的进一步收敛：从 query/rerank 修补，转向 claim/support/verification

结合 `0041 / 0042` 的真实表现，后续主线需要进一步收敛成三层。

#### 12.3.52.1 Claim-oriented decomposition

后续子需求不应再只表示成“几条检索 query”，而应至少隐式或显式保留：

- `claim`
- `required_relation`
- `required_entities`
- `answerable_if`

对 `0041` 这类问题，内部表示不应是：

- `Metavante reimbursable`

而应接近：

- 是否存在 `reimbursable from third parties` 机制
- 以及它是否与 Neoforma 对 `third-party claims` 的 indemnification 被同合同显式关联

#### 12.3.52.2 Role-based support selection

后续 support 不应继续等价于“排序后取前 2”，而应区分证据角色。

不同题型至少应区分：

- `single-hop fact`
  - 一个主证据即可
- `multi-hop specific`
  - clause A
  - clause B
  - bridge / no-bridge evidence

这意味着 support 选择目标应从：

- “分数最高的前两条”

转向：

- “足以支撑最终结论的最小充分证据组合”

#### 12.3.52.3 Verification-constrained final answering

final answer 不应只做综合生成，而应先检查：

- support 中是否存在显式 bridge
- support 是否来自同一目标合同
- 若只有相关但不构成明示连接的条款，答案是否已经被收缩为保守表述

对 `0042` 这类问题，若 support 中没有明示：

- “违反 A 会导致 B”

则最终答案只能回答：

- 发现了相关条款
- 但未见明确桥接关系

### 12.3.53 执行顺序修正

基于以上系统性复盘，后续优先级应调整为：

1. 继续完成 `12.3.49.1`
   - 即：完整关系短语 query planning
   - 这是 `claim-oriented decomposition` 的最小现实起点

2. 完成 `12.3.49.2`
   - 即：让 `target_doc_id` 真正进入 support rerank 的 `doc_score`
   - 这是 `role-based support selection` 的前置条件之一

3. 在前两项稳定后，再进入：
   - `verification-constrained final answering`

理由：

- `0041` 说明前段仍然会“问错问题”
- `0042` 说明即使 support 已对，后段仍会“答得太过”

因此当前系统不是只缺一个 reranker 或一个 prompt，而是需要逐步把：

- `query planning`
- `support selection`
- `final answering`

统一收敛到“围绕待验证法律关系组织证据”的同一主线。

### 12.3.54 Hierarchical Indexing / Scope-aware retrieval：先判定文档范围，再决定元数据约束强度

`0041` 的最新在线结果进一步说明，`target-doc alignment` 不能被实现成全局统一的 first-stage 约束策略。

问题不在于：

- 是否应该使用合同绑定信号

而在于：

- 不同问题所需的文档范围不同
- 但当前系统没有先判定“这题需要单合同、双合同，还是全局开放检索”

于是同一套 `target_doc` 约束会在不同题型上产生相反效果：

- 对 `0042` 这类单合同问题，same-doc 优先是正向的
- 对 `0041` 这类跨合同桥接问题，强 same-doc 会直接压掉另一半 gold evidence，导致全部 `NOT_FOUND`

#### 12.3.54.1 核心思想

在检索前先做一层 scope decision，而不是让 `strict target-doc mode` 作为全局开关直接作用于 first-stage。

建议的最小 scope 类型：

- `single_doc`
  - 目标是单份合同内回答
  - 可以使用更强的 same-doc bias
- `cross_doc_bridge`
  - 目标是连接两份或少量文档中的证据
  - 不允许强 same-doc 压制另一份文档
- `global_open`
  - 目标是全库开放检索
  - 仅保留弱 soft bias

#### 12.3.54.2 为什么这是对 `strict target-doc` 的上位替代

此前被证伪的是：

- `first-stage hard gating`

而不是：

- `target_doc alignment`

`Hierarchical Indexing / Scope-aware retrieval` 的作用，是把“是否启用强 same-doc 约束”从全局配置，提升为：

- 先判定题型需要的文档范围
- 再决定约束强度

换言之：

- `strict_mode = auto/force`
  不应独立决定检索空间
- 它应受 `scope_type` 约束

#### 12.3.54.3 对当前样本的解释

- `0042`
  - 更接近 `single_doc`
  - 适合更强 same-doc bias

- `0041`
  - 本质是 `cross_doc_bridge`
  - 需要同时保留：
    - `OFGBANCORP` 中的 `Expenses` 定义
    - `Neoforma` 中的 indemnification clause
  - 因此不应被单合同约束压成纯 `Neoforma` 检索

#### 12.3.54.4 最小可实施版本

不需要一开始就做完整“合同地图”，最小版可以先只做 scope decision：

1. `single_doc`
   - 问题中存在高置信 `Contract title`
   - 且 claim / entities / terms 未明显指向第二份合同
   - 执行策略：
     - first-stage 使用 soft target-doc priority
     - support/final 使用 stronger same-doc bias

2. `cross_doc_bridge`
   - 问题中存在高置信 `Contract title`
   - 但 claim / entities / terms 明显指向另一份协议、另一组实体或另一类合同术语
   - 执行策略：
     - primary doc 与 secondary doc 分配独立候选预算
     - 禁止 first-stage same-doc 独占
     - support 阶段按 doc slots 选择，而不是纯总分前 2

3. `global_open`
   - 无高置信合同绑定
   - 或问题本身为广域搜索
   - 执行策略：
     - 保持当前 broad retrieval
     - 仅保留弱 soft bias

#### 12.3.54.5 当前阶段的实施意义

这不是额外加一套新系统，而是为了避免继续在以下二元对立中来回摆动：

- `strict target-doc on`
- `strict target-doc off`

真正要控制的不是“开不开 target-doc”，而是：

- 这道题需要多大的文档范围

因此，后续检索主线应再补一层：

- `claim-oriented decomposition`
- `scope-aware document selection`
- `role-based support selection`
- `verification-constrained final answering`

#### 12.3.54.6 对当前配置项的定位修正

在这一框架下，配置中的：

- `retrieval.target_doc.strict_mode`

仍然保留，但其定位应修正为：

- 实验性 first-stage bias strength 开关

而不是：

- 全局性的文档范围决策器

后续真正决定是否允许强 same-doc 的，应是：

- `scope_type`

而不是单独的 `strict_mode`。

#### 12.3.54.7 实施约束 A：避免 scope 判定被泛词误触发

`scope_type` 不能主要由通用术语触发，否则会把大量普通合同问题误判成 `cross_doc_bridge`。

低权重或应忽略的触发信号包括：

- `甲方`
- `乙方`
- `费用`
- `服务`
- `赔偿`
- `责任`
- `义务`

高权重触发信号应优先限定为：

- `Contract title`
- 专有名词实体（Proper Nouns）
- 明确协议名 / agreement title
- section / article 编号
- 唯一日期
- 与某一合同强绑定的定义项或稀有术语

因此最小 scope 判定应遵循：

1. 先由 `Contract title` 确定 `primary_doc`
2. 只有当 claim / entities / terms 中再出现第二份合同的高置信绑定信号时，才升级为 `cross_doc_bridge`
3. 不允许因为泛词共现而直接触发 `cross_doc_bridge`

#### 12.3.54.8 实施约束 B：`strict_mode` 主要通过 candidate slot allocation 生效，而不是浮点分值微调

法律检索场景中，单纯通过 similarity score 或 rerank score 的浮点加减，往往无法稳定控制候选组成。

因此后续 `strict_mode` 的正确实现重心应从：

- score bias

转向：

- candidate slot allocation

建议的最小配额方案：

- `single_doc`
  - `primary_doc_slots ≈ 70%`
  - `global_fill ≈ 30%`
- `cross_doc_bridge`
  - `primary_doc_slots ≈ 40%`
  - `secondary_doc_slots ≈ 40%`
  - `global_fill ≈ 20%`
- `global_open`
  - 不做 doc-slot 预留

这样做的意义是：

- 对 `single_doc`，可以稳定保住目标合同
- 对 `cross_doc_bridge`，可以避免 secondary doc 被 primary doc 的高分冗余 chunk 完全挤掉
- 对 `global_open`，可以避免任何残留 doc bias 影响全局召回

#### 12.3.54.9 实施约束 C：`scope_type` 必须级联控制所有 doc bias

`scope_type` 不能只控制 first-stage，它必须级联控制至少三层：

1. `first-stage candidate allocation`
2. `support rerank doc_score`
3. `final chunk selection / preferred_doc_ids`

最小级联规则应明确为：

- `single_doc`
  - 允许 primary doc bias
  - 不允许 other docs 与 primary doc 竞争主槽位

- `cross_doc_bridge`
  - 允许 primary / secondary 双文档 bias
  - 禁止任何单一文档独占

- `global_open`
  - 关闭 target-doc bias
  - `strict_mode` 视为无效，不应残留 first-stage 或 support 的 same-doc 权重

也就是说，后续真正决定 doc bias 是否存在、强到什么程度的，不应是 `strict_mode` 本身，而应是：

- `scope_type × strict_mode`

其中：

- `scope_type` 决定是否允许 doc bias
- `strict_mode` 决定 bias / slot allocation 的强度

#### 12.3.54.10 下一步改造计划：从 `strict target-doc` 升级为 `scope-aware prefilter retrieval`

基于当前后端代码排查，可以确认：

- `target_doc_id` 已经透传到了检索器
- 但 dense / sparse chunk retrieval 仍主要是：
  - 先全局召回
  - 再按 `target_doc_id / scope_plan` 做候选收束
  - 最后在 rerank / final selection 阶段加 doc bias

因此，当前真正需要落地的，不是继续强化单一的 `strict target-doc` 开关，而是把这一步升级成一套完整的：

- `scope-aware`
- `probe-verified`
- `fallback-safe`
- `entity-rescue-enabled`

的 prefilter retrieval 方案。

为避免后续讨论继续混淆“hard doc-only retrieval”和“安全的 doc-scoped prefilter”，本仓库已单独建立实施清单：

- [docs/youtu_doc_scoped_prefilter_plan.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/youtu_doc_scoped_prefilter_plan.md)

这份方案文档明确约束了四类风险：

1. `Decomposer Error`
   - 如果 scope 误判，不能直接物理切断外部证据
2. `Entity-Document Mismatch`
   - 必须把“目标合同内结论”和“外部语料补充信息”拆开
3. `Sparse Statistics Distortion`
   - 不能对子集重算 lexical 统计基数
4. `Index Corruption`
   - 子集检索必须基于稳定向量 ID 与 manifest 校验

后续实施顺序也不再是“直接上 hard filter”，而是：

1. 先补索引 manifest / consistency check
2. 再补 scope probe 与 strict gate
3. 再把 dense retrieval 改成真正的 doc-scoped candidate generation
4. 再把 sparse retrieval 改成：
   - `global idf + doc mask`
5. 最后补 `entity rescue lane`

这一步的定位修正为：

- 不是“让 `target_doc_id` 更强”
- 而是“让 doc scope 成为可验证、可回退、可解释的 first-stage retrieval 约束”

因此，`retrieval.target_doc.strict_mode` 后续应只被视为：

- prefilter 强度控制项

而不是：

- 文档范围决策器

真正决定是否启用强 doc-scoped retrieval 的，应是：

- `scope_type`
- `probe outcome`
- `fallback policy`

三者联动，而不是任何单一开关。

### 12.3.55 retrieval_requirements_smoke_eval_t11 复盘

结合 [ragas_eval_summary.json](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_eval_t11-auto/ragas_eval_summary.json) 和 [ragas_eval_per_sample.jsonl](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_eval_t11-auto/ragas_eval_per_sample.jsonl)，`t11-auto` 的状态可以概括为：

- 不是“整体收敛完成”
- 而是“主线问题分化得更清楚”

相对 `t10-auto`：

- 变好：
  - `faithfulness: 0.747540 -> 0.798841`
  - `doc_purity: 0.775309 -> 1.0`
  - `support_chunk_precision: 0.312963 -> 0.361111`
  - `support_chunk_recall: 0.444444 -> 0.555556`
- 变差：
  - `answer_correctness: 0.452296 -> 0.429332`
  - `context_precision: 0.485450 -> 0.322222`
  - `context_recall: 0.870899 -> 0.703175`
  - `final_chunk_hit: 1.0 -> 0.888889`

这说明：

- 当前系统对“已有证据的忠实使用”继续提升
- 但“是否拿到了最该用的那两段 support”仍未稳定
- 因此 correctness 没有随 faithfulness 同步提升

#### 12.3.55.1 `ragas-cuad-0041`：跨文档桥接主线已经明显优化

对应产物：

- [retrieval_requirements_smoke_t10-auto.jsonl](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_t10-auto.jsonl)
- [retrieval_requirements_smoke_t11-auto.jsonl](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_t11-auto.jsonl)

从 `t10` 到 `t11`：

- `answer_correctness: 0.303381 -> 0.587195`
- `support_chunk_hit: 0.0 -> 1.0`

同时在线复测已确认：

- `answer_scope_target_doc_id = Neoforma...`
- `answer_composition_mode = cross_document_bridge`
- `semantic_alignment.alignment_type = conceptual_overlap`

说明：

- `cross_document_bridge`
- `doc scope ownership`
- `semantic alignment`

这一整条链路已经基本打通。

因此，`0041` 不再是当前第一主矛盾。

#### 12.3.55.2 `ragas-cuad-0042`：support 已基本正确，final answer 约束有所改善但未完全收敛

从 `t10` 到 `t11`：

- `support_chunk_hit` 持续为 `1.0`
- `answer_correctness: 0.323720 -> 0.406063`
- `faithfulness: 0.5 -> 0.777778`

这说明：

- `0042` 的问题仍然主要在后段
- 但 final answer 的 evidence-constrained synthesis 已经比 `t10` 更稳定

因此，`0042` 当前属于：

- 需要继续保留为回归样本
- 但不是 `t12` 的第一实施目标

#### 12.3.55.3 `ragas-cuad-0002`：当前第一主矛盾已经收敛到 single-hop exact support 选不准

`t11` 里最关键的信号是：

- `doc_purity = 1.0`
- `support_chunk_hit = 0.0`
- `answer_correctness: 0.471220 -> 0.276903`

这说明：

- 文档范围已经不脏
- `target_doc_id / answer_scope_target_doc_id` 也是正确的
- 当前不是“找错合同”
- 而是“进了对的合同，还是没把最该用的 support chunk 排到 top-2”

进一步看 `t11` 样本：

- gold chunk `9dcb9832-...` 已经进入：
  - first-stage
  - lightweight rerank
  - strong rerank
  - final context
- 但 `support_chunk_ids` 仍被：
  - `1971eaa1-...`
  - `be20b9f1-...`
  这类 near-miss chunk 占据

其根因不是 doc scope，而是：

1. `role` 问题被编译成过于泛化的 query
   - 例如：
     - `Promotion Agreement Motorola PageMaster Corporation`
     - `What clause, if any, states whether Motorola and * are connected?`
   - 这会把问题导向：
     - trademark license
     - indemnity
     - direct connection
   - 而不是：
     - `Description of the Promotion`
     - `shall offer free new Motorola pagers`

2. `single-hop exact rerank` 还缺少对“描述性 role clause”的强偏置
   - 当前 strong rerank 更容易奖励：
     - 实体共现
     - 通用 role 词
     - rights / license / indemnity 相关条款
   - 但对：
     - `Description of the Promotion`
     - `shall offer`
     - `free new Motorola ... pagers`
     - `no activation fee`
   这类真正回答问题的 clause 模式奖励不够

因此，`0002` 说明当前第一优先级已经不是：

- `route/doc_scope` 是否继续加强

而是：

- `single-hop role/fact lookup` 的 query compilation 与 exact support ranking

#### 12.3.55.4 对 `t11` 指标的解释边界

`t11` 中 `context_precision/context_recall/final_chunk_recall` 的下降，不能被简单解读为 `0041` 路线退化。

原因是：

- 当前 `0041` 已经把：
  - `retrieved_contexts`
  - `external_related_contexts`
  拆开
- bridge 题的外部证据不再混入 primary contract 主上下文

因此，对于 `0041` 这类题：

- 实际回答质量变好
- 但只看 primary `retrieved_contexts` 的 `context_*` 指标会低估真实 bridge retrieval 质量

也就是说，`t11` 的总体结论应是：

- `0041` 主线有效
- `0042` 后段改善
- `0002` 成为当前主瓶颈

### 12.3.56 t12 改进方向：聚焦 `0002` 类 single-hop role/fact 问题

基于 `t11`，`t12` 不应再把主力放在：

- `cross_document_bridge`
- `semantic_alignment`
- 或继续强化 `doc_scope`

因为这些方向已经在 `0041` 上得到实质验证。

`t12` 的第一主线应明确收敛为：

- `single-hop role/fact lookup`
- `exact support top-1 / top-2`
- `near-miss suppression`

#### 12.3.56.1 核心目标

目标不是“让回答更保守”，而是：

- 在正确文档内
- 更稳定地把真正回答问题的 clause
- 排到 `support_chunk_ids[:2]`

对 `0002` 而言，验收标准应从“是否提到了 Motorola”收紧为：

- 是否稳定选中 `Description of the Promotion` 相关 chunk
- 是否不再被：
  - trademark license
  - indemnity
  - fulfillment responsibilities
  这类 near-miss chunk 挤掉

#### 12.3.56.2 具体改造方向

1. `role_lookup` 的 query compilation 重构
   - 不再默认扩成：
     - `party_to`
     - `grants_right_to`
     - `grants_license_to`
   - 对 promotion / provider / offer 类问题，优先生成：
     - `description_clause`
     - `offer_clause`
     - `provider_role_clause`
     - `who provides [product/service] in the promotion`

2. 为描述性 role clause 增加强 exactness bias
   - 在 strong rerank 中显式奖励：
     - `Description of the Promotion`
     - `shall offer`
     - `free new`
     - `no activation fee`
     - `provider of [product/service]`
   - 这些模式应在 `single_hop_exact_mode` 下高于：
     - `license`
     - `indemnity`
     - `unauthorized advertising`
     - `rights / obligations` 泛关系条款

3. 对 near-miss clause 增加抑制
   - 当问题是 `role / provider / promotion description` 时，弱化：
     - trademark license
     - IP protection
     - indemnity
     - generic fulfillment
   - 除非问题显式包含：
     - `license`
     - `indemnity`
     - `obligation`
     - `liability`

4. support 选择要更像“答案证据”而不是“高分上下文”
   - `support_chunk_ids[:2]` 的目标应是：
     - 最能直接回答主问题的 clause
   - 而不是：
     - 同主题、同文档、分数高但只能间接说明的 clause

#### 12.3.56.3 `t12` 的回归样本与验收顺序

`t12` 应按以下顺序验收：

1. `ragas-cuad-0002`
   - 主验收样本
   - 重点看：
     - `support_chunk_hit`
     - `support_chunk_recall`
     - `answer_correctness`

2. `ragas-cuad-0041`
   - 回归防护样本
   - 确保：
     - `cross_document_bridge`
     - `semantic_alignment`
     - `external_related` ownership
     不被新改动破坏

3. `ragas-cuad-0042`
   - 后段回归样本
   - 确保：
     - final answer 不因 `0002` 的 query/rerank 改动而重新放宽

#### 12.3.56.4 当前阶段判断

因此，`t12` 的最优先问题应重新表述为：

- 不是“继续优化 route/doc_scope”
- 而是“优化 single-hop role/fact 的 query compilation 与 same-doc exact support 排序”

更直白地说：

- 当前最大问题不是“找错合同”
- 而是“进了对合同，还是拿错证据”

### 12.3.57 retrieval_requirements_smoke_eval_t12 复盘

`t12` 的总体表现已经优于 `t11`，而且是目前三轮里全局指标最好的一轮：

- `answer_correctness: 0.429332 -> 0.467153`
- `faithfulness: 0.798841 -> 0.808957`
- `context_precision: 0.322222 -> 0.522222`
- `context_recall: 0.703175 -> 0.725397`
- `doc_purity: 1.0 -> 1.0`
- `support_chunk_hit: 0.666667 -> 0.777778`
- `support_chunk_precision: 0.361111 -> 0.455556`
- `support_chunk_recall: 0.555556 -> 0.666667`

这说明：

- `t12` 不是退化轮
- `0002` 专项确实对整体 retrieval/support 有正增益
- 但 `t12` 也把新的剩余问题暴露得更清楚了

#### 12.3.57.1 `ragas-cuad-0022`：检索基本正确，但 compare/relate 被编译成 explicit-link 审查题

`0022` 的关键信号是：

- `context_precision ≈ 1.0`
- `faithfulness = 0.95`
- `support_chunk_hit = 1.0`
- `support_chunk_recall = 0.666667`
- `answer_correctness = 0.498295`

这说明：

- 文档没错
- support 也不是主问题
- 答案大部分内容有证据支撑

但它仍然没有答到 benchmark 想要的“比较/对照”口径。

当前系统把这题编成了：

- `What clause, if any, states whether CMS Electronic Funds Delivery Service Level and Core Systems' Availability are connected?`

于是最终答案偏向：

- “没有 direct connection”

而不是：

- CMS 的 credit event threshold 是什么
- Core Systems 的 threshold 是什么
- 两者的 availability / scheduled hours / shared precondition 如何异同

因此，`0022` 当前不是主要 retrieval failure，而是：

- `compare / relate` intent 被编译成了 `explicit clause connection`
- final synthesis 过度防守

#### 12.3.57.2 `ragas-cuad-0043`：当前最值得优先处理的是真正的检索编排失败

`0043` 与 `0022` 不同，它不是“检到了、答保守了”，而是：

- decomposition
- doc scope
- bridge routing

在检索阶段就一起错了。

当前 `t12` 样本里，`0043` 的真实执行链路是：

1. 只拆出了 `1` 个子问题
   - `What clause, if any, states whether Section 504 and Section 10.5.2 are connected?`

2. 该子问题被定成：
   - `answer_composition_mode = target_contract_primary`
   - `scope_plan.scope_type = single_doc`
   - `target_doc_id = OFGBANCORP...`

3. `reasoning_trace.sub_question_answers[0].sub_answer = NOT_FOUND`

4. `external_related_doc_ids = []`

这意味着系统当前把一个双文档 bridge 题错误地执行成了：

- 单子问题
- 单文档 hard scope
- 显式连接条款审查

而不是：

- 左边找 `Section 504 / Privacy Regulations`
- 右边找 `Section 10.5.2`
- 然后做 cross-document bridge synthesis

这也解释了为什么 `0043` 会出现一种“指标看上去不完全崩，但 recall 全是 0”的表象：

- `context_precision ≈ 1.0`
  - 因为检到的 OFG 隐私监管定义对问题前半段确实高度相关
- `faithfulness = 0.764706`
  - 因为答案前半段基本有依据
- 但 `support_chunk_hit / support_chunk_recall / final_chunk_recall = 0`
  - 因为它没有桥到参考答案要求的另一个文档，也没有命中 gold chunk

因此，`0043` 当前反映的不是后段语言问题，而是：

- 上游 decomposition 把桥接题压扁
- doc scope 过早单文档化
- retrieval 根本没被允许去找另一侧证据

#### 12.3.57.3 `t12` 之后的优先级重排

经过 `t12`，当前剩余问题不应再被笼统表述为：

- “继续调 overall retrieval”

而应该拆成两类：

1. `0022` 类
   - compare/relate 被误编译成 explicit-link
   - 主要问题在 synthesis / query interpretation

2. `0043` 类
   - 真实的 retrieval orchestration failure
   - 主要问题在 decomposition + bridge routing + doc scope

在这两类里，下一步更值得优先处理的是 `0043`，因为它还没有进入“检到了但答得保守”的阶段，而是更上游地“没有被正确检索”。

### 12.3.58 t13 改进方向：只聚焦 `ragas-cuad-0043`

`t13` 明确只做一件事：

- 修正 `0043` 这类 `structural definition bridge` 问题

不再同时扩写：

- `0022`
- `0042`
- 或其他 single-hop / answer-style 问题

#### 12.3.58.1 `t13` 的问题定义

`0043` 当前的失败本质是：

- 双锚点（`Section 504`、`Section 10.5.2`）
- 双定义概念（`Privacy Regulations`、`Sensitive Customer Information`）
- 一个业务语境锚点（`Section 10.5.2 / Training and Education Gross Margin`）

但参考答案的真实合成方式并不是：

- 在 `Section 10.5.2` 所在文档里直接找到 `Sensitive Customer Information` 的定义

而是允许一种错位的跨文档合成：

1. `Section 504 -> Privacy Regulations`
2. `Sensitive Customer Information -> OFGBANCORP` 中的命名定义
3. `Section 10.5.2 -> Neoforma` 中的业务运营上下文

再由回答层做：

- `privacy regulatory framework + protected data category + operational/business context`

的跨文档综合。

因此，`t13` 的目标不应再被定义成简单的：

- `structural definition bridge`

而应重定义为：

- `contextual cross-document bridge`
- 更具体地说，是 `clause-context + external-definition bridge`

#### 12.3.58.2 `t13` 只改的内容

`t13` 只应改下面这一条主链路：

1. 识别 `structural + definition_lookup + dual explicit section anchors`
   - 尤其是同时包含：
     - `Section ...`
     - 命名定义概念
     - `relate / connect / significance`

2. 不再默认编译成：
   - `What clause, if any, states whether X and Y are connected?`

3. 不再把右腿理解成：
   - `去找 Section 10.5.2 本身定义了什么`

4. 而是拆成三条受约束的 bridge legs
   - 左侧：
     - `Section 504 / Privacy Regulations`
   - 中间定义腿：
     - `Sensitive Customer Information / definition`
   - 右侧上下文腿：
     - `Section 10.5.2 / Training and Education Gross Margin / operational context`

5. 禁止在 decomposition 阶段直接把整题锁成单文档
   - 但也不能把右腿做成 `global open`
   - 需要允许受约束的多文档 bridge retrieval

6. 最终回答层继续沿用现有：
   - ownership
   - source attribution
   - cross-document bridge synthesis

也就是说，`t13` 的改造范围明确限制为：

- decomposition
- route/scoping
- sub-question compilation
- bridge leg construction

而不是：

- 大范围 rerank
- broad answer prompt 重写
- 其他 unrelated 样本优化

#### 12.3.58.3 `t13` 的验收标准

`t13` 完成后，`0043` 至少应满足：

1. 不再只生成 `1` 个 `whether X and Y are connected` 子问题
2. 不再把右腿压成单一 `Section 10.5.2 define` 查询
3. 至少形成：
   - `Section 504 / Privacy Regulations`
   - `Sensitive Customer Information`
   - `Section 10.5.2 / business context`
   这三种证据目标中的两个以上
4. 不再是 `single_doc OFGBANCORP` 独占
5. `external_related_doc_ids` 或等价 bridge evidence 中应出现 `Neoforma...`
6. `support_chunk_hit / support_chunk_recall` 不再是 `0`

如果这几条还做不到，就说明：

- `0043` 仍然停留在“检索失败”
- 还没有资格进入“答案是否足够好”的讨论阶段

### 12.3.59 当前阶段风险判断：避免继续“拆东墙补西墙”

在 `0043` 这一轮修复推进后，当前系统已经暴露出一个更上层的结构性风险：

- 不是某一个样本单独坏掉
- 而是 `sub-question compiler` 的语义分层仍不稳定
- 新增的 bridge / structural / section-anchor 分支，正在与原有 `compare/relate` 分支发生优先级竞争

这个风险在 `ragas-cuad-0022` 上已经体现出来：

- `t12` 里，`0022` 还表现为多子问题，其中一条 `NOT_FOUND`
- 当前 live 行为里，它已经不再 `NOT_FOUND`
- 但却被压成了单个子问题，并被错误编译成：
  - `What clause, if any, states whether X and Y are connected?`

这说明问题不在 retriever 的底层召回，而在：

- `retrieval_requirement -> sub-questions` 的编译层
- 某个更泛化的 `structural / explicit-link` 分支过早命中
- 截胡了本该进入 `compare / relate / dual-fact synthesis` 的问题

换句话说，当前阶段如果继续沿着“按样本补条件”的方式推进，很容易出现：

- 为了修 `0043`，增强 `bridge / structural / section-anchor`
- 结果 `0022` 这种 compare 题被误送进 explicit-link 模板
- 再去修 `0022`，又可能影响别的 definition/context 题

这就是典型的：

- `拆东墙补西墙`

#### 12.3.59.1 当前应如何理解这个问题

当前的核心矛盾不再只是：

- 某个样本没检到
- 某个样本答案写偏

而是：

- `compiler` 不是一套稳定的意图层级系统
- 而是一组不断追加的高耦合分支

只要继续在现有分支体系里为单题加条件，风险就会持续放大：

- 分支命中顺序越来越难预测
- 样本间回归会越来越频繁
- 每修一题都可能把另一类题的优先级挤掉

#### 12.3.59.2 后续重构方向

下一阶段不应再以“单题补救”为主，而应转向：

- `sub-question compiler` 的证据需求分析与子问题路由重构

这里不应再把原问题先强行归入某个固定题型模板，而应按问题真正需要的证据语义来拆解：

1. 先做 `evidence requirement analysis`
   - 识别原问题需要哪些证据成分，而不是先判它“属于哪一类题”
   - 一题中可以同时存在多个语义需求，例如：
     - `definition_lookup`
     - `context_lookup`
     - `fact_or_role_lookup`
     - `compare_or_relational_analysis`
     - `cross_document_bridge`

2. 再按这些证据需求生成子问题
   - 一个原问题可以生成 `1..N` 个子问题
   - 每个子问题只承载一种主要证据任务
   - 例如：
     - `X 是什么 / X 如何定义`
     - `Section Y 讲什么 / 提供什么业务语境`
     - `A 的触发条件是什么`
     - `B 的 requirement 是什么`

3. 再对子问题分别路由
   - 路由应归属于子问题，而不是整个原问题一次性绑定
   - 每个子问题再独立走：
     - `local`
     - `structural`
     - `global`
     - 或受约束的 bridge / scoped retrieval

4. 最后才做合成
   - `compare/relate` 型问题，不应在子问题阶段就提前问成：
     - `whether A and B are connected`
   - 更合理的做法是：
     - 先把 `A` 和 `B` 的事实分别找全
     - 再在回答层做 compare / relation synthesis

因此，后续应避免再使用“固定拆成几类语义”的思路，而应改成：

- `按证据需求拆出若干语义子问题，并分别归类到相应路由`

这意味着：

- 一题不应被迫只归到一种题型
- `0043` 可以同时需要：
  - `definition_lookup`
  - `context_lookup`
  - `cross_document_bridge`
- `0022` 可以同时需要：
  - 两个事实/定义子问题
  - 一个 compare/relate 合成层
- `0002` 则主要只需要：
  - `fact_or_role_lookup`
  - 不应被抬成 `explicit_link` 或 `structural bridge`

#### 12.3.59.3 当前结论

因此，当前阶段的判断应明确记录为：

- `0043` 的主链路已经基本打通
- 但系统整体还不能算稳定
- 继续按单题打补丁，会持续造成分支互相截胡

后续如果继续推进，应优先做：

- `compiler` 的决策层重构

而不是：

- 再按 `0022 / 0042 / 0043` 单独加条件

### 12.3.60 后续重构设计：从单题补救转向中层架构重构

当前已经可以明确：

- 问题不再只是某一个样本的 retrieval miss
- 也不只是最终回答写偏
- 而是整条中层 orchestration 链路缺少稳定的数据结构与决策边界

因此，后续重构不应继续按样本补丁推进，而应按阶段重构中层架构。

#### 12.3.60.1 重构目标

目标不是替换底层 dense / sparse / graph retrieval 能力，而是重构以下中层能力：

- `question -> evidence requirements`
- `requirements -> sub-question plans`
- `sub-question -> route + scope`
- `retrieval outputs -> evidence aggregation`
- `evidence -> final reasoning`

也就是说，后续改造主要针对：

- compiler
- scope orchestration
- evidence aggregation
- final synthesis

而不是：

- 图构建 schema
- chunk 切分策略
- 底层向量召回器主体

#### 12.3.60.2 分阶段实施方案

##### Phase 1：统一中间表示

先建立稳定的中间表示，而不是继续在松散 dict 上叠字段。

建议至少引入三类对象：

1. `EvidenceRequirement`
   - `requirement_id`
   - `semantic_need`
   - `target_entity`
   - `anchor_terms`
   - `target_scope_hint`
   - `evidence_weight`

2. `SubQuestionPlan`
   - `sub_question_id`
   - `intent`
   - `query_text`
   - `route_type`
   - `scope_plan`
   - `priority`

3. `EvidenceItem`
   - `origin_sub_question_id`
   - `origin_route`
   - `origin_query`
   - `doc_id`
   - `chunk_id`
   - `evidence_role`
   - `score`
   - `scope_label`

没有这层，后续的 budget / gating / reasoning 都仍会建立在不稳定的状态拼装上。

##### Phase 2：证据需求分析与子问题生成

将“先判断题型，再套模板”改成：

- 先做 `evidence requirement analysis`
- 再按 requirement 生成子问题
- 每个子问题独立路由

重点原则：

- 一题可包含多个语义需求
- 一个 requirement 对应一个主要证据任务
- compare 题默认拆成：
  - 两个事实/定义 requirement
  - 一个 compare/relation 合成 requirement
- bridge 题默认拆成多腿 requirement，而不是直接问：
  - `whether X and Y are connected`

##### Phase 3：Dynamic Scope Gating

将作用域从“整题继承”改成“子问题显式声明”。

每个 `SubQuestionPlan` 应显式携带：

- `target_scope`
- `bridge_scope`
- `scope_confidence`

关键约束：

- `single_doc` 只对明确单文档事实题启用
- bridge 子问题允许独立选档
- section anchor 优先于泛语义相似度
- rescue 必须成为标准 attempt，而不是旁路补丁

##### Phase 4：Evidence Matrix + Dynamic Budget

在子问题稳定之后，再引入资源调度与去重。

这里不再把所有 chunk 先揉平成一个大池，而是建立：

- `Evidence Matrix`
- `Cross-Subquestion Dedup`
- `Global Context Budget`

建议控制方式：

- pre-retrieval budget
- post-retrieval cross-dedup
- 子问题优先级驱动的上下文配额

##### Phase 5：Reasoning over Facts

最后才重构最终回答层。

合成层结构应明确拆成：

1. `Fact Presentation`
2. `Semantic Alignment`
3. `Inference`

这样：

- compare 题先陈述 `A` 和 `B`
- bridge 题先陈述各腿事实
- 显式 section anchored finding 必须先被陈述
- 然后才做 relation / compare / bridge synthesis

#### 12.3.60.3 当前检索系统与目标系统的区别

当前系统与后续目标系统之间，关键区别在于：

##### 1. 当前系统：题目级模板主导  
目标系统：证据需求主导

当前系统更像：

- 先把整题判成某种模板
- 再沿该模板编译 `1~2` 个子问题

目标系统则应改为：

- 先抽取这道题到底需要哪些证据
- 再生成若干子问题
- 再对子问题分别路由

##### 2. 当前系统：子问题结果过早揉平  
目标系统：Evidence Matrix 保留 provenance

当前系统的问题是：

- 多 query
- 多 route
- 多 doc
- 多子问题

的结果在中途很容易被揉平，导致：

- provenance 丢失
- ownership 错位
- support 选择混 cluster

目标系统要求：

- 每个 finding 都保留来源
- route/doc/query/sub-question 不应在聚合前丢失

##### 3. 当前系统：scope 继承偏静态  
目标系统：scope gating 偏动态

当前系统仍大量依赖：

- 原问题目标合同
- 父问题 target_doc_id

向子问题继承作用域

目标系统则要求：

- 每个子问题可以有自己的 scope
- bridge leg 可以独立选档
- section anchor / identity sync / scope hint 共同决定最终作用域

##### 4. 当前系统：最终回答依赖 prompt 自由综合  
目标系统：最终回答依赖结构化 fact reasoning

当前系统中，很多回答偏差并不是没检到，而是：

- fact 已有
- provenance 已部分存在
- 但 final synthesis 仍在做自由概括

目标系统则要求：

- 回答层优先消费 Evidence Matrix
- 先陈述 fact
- 再做 semantic alignment
- 最后才推断 relation

##### 5. 当前系统：资源控制是静态的  
目标系统：资源控制是按 requirement 动态分配的

当前系统更接近：

- 固定 top-k
- 固定 final context size
- 再做后置 rerank

目标系统则应转向：

- requirement-aware context budget
- 子问题优先级调度
- cross-dedup
- 延迟加载低优先级 context

#### 12.3.60.4 可行性与风险判断

从全局检索流程看，这套重构方案是可行的，而且有能力系统性解决当前的主问题。

但风险同样明确：

1. 如果不先重构中间表示，而直接叠加 budget / gating / reasoning
   - 系统复杂度会继续失控

2. 如果 ERA 还不稳定，就过早引入复杂调度
   - 下游将建立在错误 requirement 上

3. 如果试图一步落完整套方案
   - 会进入长时间不稳定期

因此，后续推进必须严格遵守：

- `先中间表示，再编排，再预算，再推理`

#### 12.3.60.5 当前结论

当前最合理的判断是：

- 这套方案值得做
- 它有机会系统性解决当前主问题
- 但它必须作为一次中层架构重构来推进
- 不能继续以“在现有链路上逐点嫁接能力”的方式实施

#### 12.3.60.6 `t13-refactor_v1` 复盘：当前两类核心问题

在 `refactor_v1` 完成 `Phase 1~5` 的首轮串联后，`t13-refactor_v1` 的结果表明：

- 新链路已经真正跑通
  - `orchestration_mode = refactor_v1`
  - `final_chunk_selection_strategy = evidence_matrix_budgeted_selection`
  - `reasoning_plan` / `evidence_matrix` / `phase4_budget` 均已进入线上输出与评估结果
- 但整体效果还不能判定为优于 `legacy/t12`
- 当前最需要正视的是两类问题：
  1. `Phase 2` 的 requirement classification 仍不稳定
  2. `Phase 5` 的长结构化回答污染了当前评估口径

##### 12.3.60.6.1 `Phase 2`：requirement classification 仍有错分

`Phase 2` 的目标是“按证据需求拆子问题”，但 `t13-refactor_v1` 已经显示：

- 这条链路虽然已经能稳定拆题
- 但对某些题目的 semantic need / mode 归类仍然会错

最典型的是：

- `ragas-cuad-0041`
  - 在 `t13-refactor_v1` 中被编进了 `reasoning_plan_mode = compare`
  - 但这题本质上仍应是跨文档 bridge，而不是 compare

这说明当前的 `Phase 2` 虽然比旧模板链路更可分析，但它还没有完全解决：

- compare / bridge / fact 之间的 requirement 区分边界
- 某些多腿 bridge 问题在 ERA 阶段被错误压缩成 compare/relation 问题

因此，当前不能把 `Phase 2` 理解成“已稳定完成”。

更准确的判断是：

- `Phase 2` 已完成“从模板到 requirement-aware 编排”的结构迁移
- 但 requirement classification 的判别规则还需要继续收敛

后续若继续优化 `refactor_v1`，`Phase 2` 的优先问题应定义为：

- 修正 requirement classification 的错分问题
- 重点关注：
  - bridge 被误分为 compare
  - compare 被误分为 explicit-link
  - role/fact 被误抬成 structural/connected

##### 12.3.60.6.2 `Phase 5`：当前评估口径被长结构化回答污染

`Phase 5` 引入 `Reasoning over Facts` 后，最终回答不再只是短答案，而是通常包含：

1. `Reasoning over Facts`
2. `Grounded Facts ...`
3. `Semantic Alignment ...`
4. `Inference / Bridge Conclusion / Final Answer`

这对产品层和调试层是有价值的，但当前 `run_ragas_eval.py` 仍然直接拿整段 `response` 做评分。

这会带来两个后果：

- `answer_correctness` 被大段 reasoning 文本稀释
- `faithfulness / context_*` 也可能被格式性和解释性文本拖低

也就是说，当前的低 `answer_correctness` 并不一定都代表“结论错了”，其中一部分可能只是：

- evaluator 在拿整段推理文本当最终答案打分
- 而不是只拿真正的 `Final Answer / Inference and Conclusion`

因此，`t13-refactor_v1` 的当前评估口径与 `legacy/t12` 已不完全等价。

当前更合理的做法应是：

- 保留完整 `answer` 给产品与调试使用
- 但在 compare/eval 层单独抽取一个用于评分的字段，例如：
  - `response_for_eval`
  - 或 `scored_response`

抽取策略应优先取：

- `Final Answer:`
- `Inference and Conclusion:`
- 若无明确标签，再回退到完整 `response`

因此，这一阶段的评估问题不是“Phase 5 不该存在”，而是：

- `Phase 5` 的产品回答格式
- 与当前评估脚本的评分口径

之间还没有正式解耦。

##### 12.3.60.6.3 当前结论

`t13-refactor_v1` 的当前结论应明确写成：

- 新 orchestration 架构已经完成首轮闭环
- 但还不是性能优于旧链路的收敛版本

当前最优先的两项工作是：

1. 修正 `Phase 2` 的 requirement classification 错分  
   尤其是 `0041` 这类 bridge 题被误分成 compare 的问题

2. 将 `Phase 5` 的完整长回答与评估口径解耦  
   让评估只对 `Final Answer / Inference` 段打分，而不是直接评分整段 reasoning 输出

#### 12.3.60.7 `t13-refactor_v1-1` 复盘：`0041` 暴露的是运行时 scope/support 联动回归

结合以下产物：

- [retrieval_requirements_smoke_t13-refactor_v1-1.jsonl](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_t13-refactor_v1-1.jsonl)
- [retrieval_requirements_smoke_t13-refactor_v1-1_youtu_graph_rag_predictions.jsonl](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_t13-refactor_v1-1_youtu_graph_rag_predictions.jsonl)
- [retrieval_requirements_smoke_eval_t13-refactor_v1/ragas_eval_per_sample.jsonl](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/outputs/results/ragas/retrieval_requirements_smoke_eval_t13-refactor_v1/ragas_eval_per_sample.jsonl)

最新 `t13-refactor_v1-1` 对 `ragas-cuad-0041` 的表现说明：

- 这已经不是“旧产物残留”
- 而是当前最新代码路径下仍然存在的真实运行时回归

最关键的现象不是简单的 “Section 15.4 没被看见”，而是：

1. `answer_scope_target_doc_id` 已经是 `Neoforma...#p0`
2. 但 `sq_2 / sq_4 / sq_6` 这几条 Neoforma indemnification 子问题的 `target_doc_id` 仍被解析成了 `GOCALL...#p0`
3. 同一条子问题的 triples 中其实已经存在：
   - `Section 15.4`
   - `indemnification obligations of Neoforma`
4. 但最终 `support_doc_ids` 仍全部落在 `GOCALL...#p0`
5. 然后子问题回答被写成 `NOT_FOUND`
6. 最后 final answer 又把 “selected support 里没拿到 target-contract clause” 误写成了 “Neoforma contract 不包含该条款”

也就是说，`0041` 当前不是：

- 图谱里没有 `Section 15.4`

而是：

- scope 先选错文档
- support 再按错 scope 强化错误文档
- synthesis 最后把 `not retrieved` 升级成了 `not exist`

##### 12.3.60.7.1 根因一：scope 决策在 `Neoforma` 与 `third-party claims` 之间发生平分后按字典序选错

当前 `_resolve_refactor_subquestion_scope()` 的打分方式本质上仍是：

- 对唯一命中的 phrase 给 doc 加权
- 然后按 `(-score, doc_id)` 排序选 `primary_doc_id`

在 `0041` 这题上，运行时真实 `matched_phrases` 表明：

- `GOCALL...#p0`
  - `entity_or_anchor:third-party claims`
- `Neoforma...#p0`
  - `entity_or_anchor:Neoforma`

这两条信号在当前实现里都属于普通唯一命中 phrase，因此：

- 两边分数打平
- 最终由 doc id 字典序决定主文档
- `GOCALL...` 被错误选成 `primary_doc_id`

这说明：

- 现在的问题不再是“没有 target entity”
- 而是“claim-bearing entity 与 generic bridge phrase 的优先级没有被显式建模”

因此，`0041` 当前真正暴露的是：

- scope tie-break 规则仍然过弱
- 对 bridge 问题，`Neoforma` 这类 claim owner 实体不应和 `third-party claims` 这种 generic phrase 同权竞争

##### 12.3.60.7.2 根因二：strong support reranker 会沿着错误 `target_doc_id` 继续压低 Neoforma clause

一旦 `sq_2` 的 `target_doc_id` 被错定成 `GOCALL...#p0`，后续 strong reranker 就会：

- 对 `doc_id == target_doc_id` 的 chunk 给 bonus
- 对其他 doc 的 chunk 给 penalty

因此即使 Neoforma 侧 triples 中已经明确出现：

- `Section 15.4`
- `indemnification by Neoforma`

support top-2 仍会被推向：

- `GOCALL` 的 indemnity / third-party claims chunks

这正是当前 `0041` 最值得警惕的地方：

- triples 层已经看到了正确 target-contract clause
- 但 support 层因为 scope 错误，反而把正确 clause 压下去了

所以 `Reasoning` 维度里看到的：

- “推理步骤 2 错误地判定 Section 15.4 不存在”

其本质并不是：

- graph/triples 缺失

而是：

- `selected support` 缺失被误表述成了 `contract clause` 缺失

##### 12.3.60.7.3 根因三：reasoning plan 只保留 external grounded facts，导致 final answer 被迫坍缩到 `target_contract_primary`

`t13-refactor_v1-1` 中 `0041` 的 `reasoning_plan.facts` 只保留了三条 `Expenses` definition 相关的 external facts。

Neoforma indemnification 这条腿由于子问题回答已经变成 `NOT_FOUND`，在当前 plan 构建中没有被保留下来作为 grounded target finding。

结果就是：

- `answer_composition_mode` 没有进入 `cross_document_bridge`
- `semantic_alignment` 也为空
- final answer prompt 只能按 `target_contract_primary` 的规则写

于是系统最后会自然坍缩成：

- “目标合同对两边都没有直接支持”
- “外部合同只能当 related context”

这解释了为什么当前 `0041` 的最终错误不是孤立生成问题，而是前三层问题传导后的必然结果。

##### 12.3.60.7.4 对当前结论的修正

此前文档中 “`0041` 主线已经明显优化、已不再是第一主矛盾” 的判断，针对的是：

- `t11-auto`
- 以及那一轮 `cross_document_bridge` 走通的在线状态

但 `t13-refactor_v1-1` 的最新结果说明：

- 在新 orchestration 链路里，`0041` 又以新的形式回退了
- 回退点不再是“是否支持 cross-document bridge”
- 而是“scope tie-break + support lane ranking + final wording” 三者的联动回归

因此，对 `t13` 当前阶段更准确的结论应改写为：

- `Phase 2` 的 requirement-aware orchestration 已经搭起来
- 但 `0041` 证明新链路在 bridge 题上仍不稳定
- 当前第一主问题不只是 `compare/bridge` 分类
- 而是 bridge 子问题进入 runtime scope/support 之后，缺少更强的 claim-bearing contract 保护

### 12.3.61 `t14` 重设计：先修 provenance / grounding，再修 scope / support / synthesis

结合 `t13-refactor_v1` 与 `t13-refactor_v1-1` 的复盘，以及 `t14` 一度落地后在 `0041` 上暴露出的严重问题，当前需要明确修正一个判断：

- `0041` 的主问题已经不只是 `runtime scope/support/synthesis` 三段回归
- 更不是单纯的 `wording` 过强
- 而是系统已经出现了：
  - 跨文档事实“偷窃”
  - target-contract attribution fraud
  - chunk grounding rupture

也就是说，问题不再只是：

- `target_doc_id` 选错
- support top-2 没拉回 target clause

而是已经升级成：

- 外部合同中的事实被拿来填补目标合同的空白
- findings 层把 external evidence 贴成 target-contract finding
- final answer 再把这类伪 finding 写成“目标合同明确规定”

因此，原始 `t14-P1/P2/P3` 的设计口径已经不够。

它默认：

- `support_doc_ids`
- `support_chunk_ids`
- findings ownership

这些中间态本身是可信的。

但当前代码路径与现象已经表明，这个前提并不成立。

所以，`t14` 必须改成一轮更底层的“行为可信化”迭代：

- 先封死 provenance 被改写
- 再让 scope/support 建立在真实 provenance 之上
- 最后才收紧 synthesis/wording

换句话说，`t14` 的目标不再只是“让系统更会选文档”，而是：

- 让系统不再有机会对错误来源的事实进行合法化包装

#### 12.3.61.1 为什么原版 `t14` 不够：`0041` 暴露的是四层问题，不只是三层

从当前 `0041` 的失败形态看，原版 `t14` 只覆盖了后半段，却没有覆盖最致命的前提层。

当前至少有四层失真：

1. scope 失真
   - `Neoforma`
   - `third-party claims`
   在 scope 解析时发生同权竞争，导致 `primary_doc_id` 可能被 generic phrase 抢走

2. support 失真
   - target contract 的 clause/anchor 即使已经在 triples 层出现
   - support lane 与 rerank 仍可能把 wrong doc 压成 top support

3. provenance 失真
   - external contract 的 facts/chunks 被重新包装成 target-contract ownership
   - findings 层出现“support_doc_ids 看似属于目标合同，但 backing chunks 实际并不支持”的断裂

4. synthesis 失真
   - 系统把：
     - `not retrieved`
     - `not grounded`
     - `only external support`
   - 升级写成：
     - `target contract says X`
     - 或 `target contract does not contain X`

原版 `t14` 只覆盖了：

- `1 + 2 + 4`

但没有把：

- `3. provenance 失真`

放到第一优先级。

而 `0041` 这次最严重的问题恰恰证明：

- 只要 provenance 还能被改写
- 后面的 scope/rerank/wording 再聪明
- 也仍然可能“自洽地说谎”

所以，重设计后的 `t14` 必须先回答一个更基础的问题：

- 哪些信息是“观测到的证据”
- 哪些信息只是“推断出的作用域”
- 哪些信息可以影响检索策略
- 哪些信息绝不能回写成 support ownership

#### 12.3.61.2 `t14-R1`：先建立不可改写的 provenance ledger

`t14` 的第一步不应再是 scope tie-break，而应是：

- 先把 support provenance 与 inferred target scope 彻底拆开

后续所有判断都必须建立在一条不可改写的 provenance ledger 之上。

至少要明确区分三类字段：

1. `observed_support_chunk_ids`
   - 只记录实际进入 sub-question support 的 chunk

2. `observed_support_doc_ids`
   - 只能由：
     - `support_chunk_ids -> chunk_id_to_doc_id`
   - 单向映射得到
   - 不能由 `target_doc_id`
   - `answer_scope_target_doc_id`
   - `scope_plan.primary_doc_id`
   - 或任何 rescue / rerank 结果反向补写

3. `inferred_target_doc_id`
   - 只表达：
     - 当前系统认为这条子问题“应该以哪个合同为主解释”
   - 它可以影响 retrieval / rerank / answer policy
   - 但不能被当作 support provenance

这一步的核心约束应写死为：

- target scope 是推断
- support ownership 是观测
- 推断不能改写观测

因此，重设计后的 `t14-R1` 必须封死以下行为：

1. 不能因为 `target_doc_id = Neoforma...`，就把 `Neoforma...` 插进 `support_doc_ids`
2. 不能因为 final answer 需要 target contract finding，就把 external finding 迁移成 target finding
3. 不能因为某个 chunk 文本“像是对的”，就让它跨 doc 复用 chunk id / doc ownership

`R1` 的目标不是提升 recall，而是建立一条最基本的可信边界：

- 每个 finding 的 doc ownership 都必须可物理回指到实际 chunk

只有这层成立，后面的 scope/support 优化才有意义。

#### 12.3.61.3 `t14-R2`：scope/support 只能在真实 provenance 之上工作

在 `R1` 之后，原版 `t14-P1/P2` 中关于 `0041` 的洞察仍然保留，但它们的地位要改。

新的定义应是：

- scope 只决定“优先检索谁、优先解释谁”
- support 只决定“把哪些 chunk 送进当前子问题”
- 但这两者都不能改变最终 evidence 的归属

因此，`t14-R2` 应包含两条主线。

第一条：修 scope tie-break，但只把它当成 retrieval preference，而不是 evidence ownership。

应明确保留以下原则：

1. `claim-bearing entity / explicit party`
   - 高于 `generic relation phrase`
2. `section/article anchor`
   - 高于纯语义泛词
3. 打平时禁止按 doc id 字典序直接决策
4. `Neoforma` 这类 claim owner 与 `third-party claims` 这类 generic phrase 不得再同权竞争

但新的表述要明确：

- 即使 scope 推成 `Neoforma`
- 如果 support chunk 仍全部来自别的 doc
- 该子问题也只能被标为：
  - target-scope inferred
  - 但 not target-grounded

第二条：修 support lane / rescue，但 rescue 只能“救回 chunk”，不能“救回 ownership”。

也就是说：

- 如果 triples 已经显示 `Section 15.4`
- support lane 应优先尝试把 Neoforma clause chunk 拉回 support
- 但若最终没拉回来
- 系统只能承认：
  - target clause anchor exists
  - body support unstable / missing

不能把状态升级成：

- support_doc_ids 已支持 Neoforma clause

新的 `R2` 验收标准不应再写成：

- `support_doc_ids[:2]` 至少包含一个 `Neoforma...#p0`

而应写成两层：

1. 最优行为
   - 当 triples 已见 `Section 15.4` 时，support 应尽量拉回至少一个 Neoforma clause chunk

2. 底线行为
   - 即使 support 没拉回 Neoforma chunk
   - 也绝不能伪造 Neoforma support ownership

换句话说，`R2` 的目标是：

- 优先让 support 变对
- 但更优先保证 support 不会被写假

#### 12.3.61.4 `t14-R3`：final synthesis 改成 claim-level grounding，而不是 findings 自由综合

原版 `t14-P3` 把重点放在：

- 不要把 `not retrieved` 写成 `not exist`

这个方向是对的，但还不够。

因为 `0041` 当前真正危险的不只是“说不存在”，而是：

- 把 external fact 写成 target-contract fact

所以，`R3` 需要把 final answer 的输入从：

- “一组已经写好的 findings 文本”

改成：

- “一组带 provenance 的 claim objects”

每条 claim 至少要显式携带：

1. `claim_text`
2. `claim_scope`
   - `target_contract`
   - `external_related`
   - `cross_document_alignment`
   - `insufficient_support`
3. `support_chunk_ids`
4. `support_doc_ids`
5. `grounding_status`
6. `forbidden_as_target_fact`

在这个结构下，final synthesis 才能形成真正的硬护栏：

1. 若某条 claim 的 support docs 不含 target contract
   - 它不得被写成 target-contract factual sentence

2. 若某条 claim 只由 external doc 支持
   - 它只能被写成：
     - external agreement says ...
     - related agreement defines ...
   - 不能写成：
     - the target agreement defines ...

3. 若 target contract 仅有 clause anchor / triple evidence
   - 但未拿到 clause body
   - 只能写：
     - current target-contract support is insufficient to state the clause content
   - 不能写：
     - the agreement does not contain such clause

4. `contract does not contain X`
   - 只能在以下条件同时满足时使用：
     - target chunk 缺失
     - target clause anchor 缺失
     - target triple positive evidence 缺失
     - 且不存在 external evidence 被误用为 target finding 的风险

因此，`R3` 的目标不只是 wording 更保守，而是：

- 让 final answer 的每一句 target-contract factual sentence 都能回指到真实 target chunk

#### 12.3.61.5 `t14-R4`：补上 adapter / eval 链路，防止修完后端仍然继续制造伪对齐

如果 `t14` 只改 backend，而不审视 adapter / eval 链路，仍然存在一个现实风险：

- 后端修了 provenance
- 但 adapter / compare 输出仍把 chunk/doc 对齐做错
- 最终评估产物继续表现成“证据链自洽”，而实际上底层已经断裂

因此，`R4` 必须把下游链路也纳入 `t14` 定义。

至少要覆盖三类风险：

1. `retrieved_chunks` 与 `retrieved_chunk_ids` 的 index zip
   - 只要长度或顺序不稳定，就可能制造 fake chunk grounding

2. 基于文本 hash 的 chunk id fallback
   - 只要不同合同出现相似/重复文本，就可能制造跨文档 chunk attribution 错配

3. compare/eval 默认宽松 doc scope
   - 如果评估入口天然允许 cross-doc context 混入
   - 但输出层又没有把 provenance 显式分层
   - 最终就会继续把“不应混成 target finding 的内容”混进可评分答案

这一步的意义不是为了立即大改评估脚本，而是为了让 `t14` 的文档口径完整一致：

- 不再把 backend fidelity 与 compare fidelity 视为两件互不相干的事

#### 12.3.61.6 `t14` 的最小实施顺序应改写为 `R1 -> R2 -> R3 -> R4`

重设计后的 `t14` 不建议再按原来的：

- `P1 scope`
- `P2 support`
- `P3 wording`

顺序推进。

更合理的顺序应是：

1. `R1 provenance ledger`
   - 先封死 support ownership 被改写

2. `R2 scope/support over provenance`
   - 再修 `0041` 的 scope tie-break 与 support rescue

3. `R3 claim-level synthesis`
   - 再把 final answer 约束到真实 chunk grounding 上

4. `R4 adapter/eval alignment`
   - 最后清理 compare/output 侧的伪对齐与口径漂移

原因很明确：

- 若 `R1` 不先做，`R2` 只是在错误 ownership 上继续优化 rerank
- 若 `R2` 不做，`R3` 会被迫长期依赖保守措辞掩盖 retrieval 错误
- 若 `R4` 不做，系统即使局部修好，也仍可能在评估产物中表现成另一套失真

所以，新的 `t14` 首要原则应是：

- 先防止系统“合法化地说谎”
- 再追求它“更聪明地找对证据”

#### 12.3.61.7 重设计后的 `t14` 验收标准

`t14` 完成后，`0041` 的验收标准应从“选对 doc / 拉回 clause”扩展成“归因可信 + grounding 可核查 + 回答不越权”。

至少应满足：

1. provenance
   - 任意一条 target-contract finding 都必须能回指到：
     - 至少一个实际属于 target contract 的 support chunk
   - 不允许仅凭 `target_doc_id` 或 `answer_scope_target_doc_id` 获得 target ownership

2. external fact attribution
   - OFG/Metavante 合同中的 `Expenses` 定义即使被使用
   - 也必须明确标成：
     - external evidence
     - related agreement fact
   - 不能写成：
     - `The agreement [Neoforma] defines Expenses as ...`

3. grounding
   - findings 中引用的 `support_chunk_ids`
   - 必须能在 backing chunks 中物理对应到同一 doc / 同一内容来源
   - 不允许出现：
     - chunk id 存在
     - 但 backing chunks 中无对应内容
     - 或 chunk 实际属于另一份合同

4. target-clause calibration
   - 若 target contract 只出现 clause anchor / triple evidence
   - 但未稳定拿到 clause body
   - 回答只能写成：
     - support insufficient
     - clause anchor observed but body not grounded
   - 不能写成：
     - contract does not contain ...

5. final answer
   - 每一句 target-contract factual sentence
   - 都必须由 target chunk 直接支撑
   - external evidence 只能出现在：
     - source-attributed related finding
     - 或 cross-document semantic alignment

6. regression coverage
   - 新增回归必须覆盖：
     - generic phrase 与 explicit entity 打平时选错 doc
     - triples 已见 target clause，但 support 未拉回 target chunk
     - support provenance 被 target scope 反向补写
     - findings 引用了不存在或跨 doc 错配的 chunk id
     - external fact 被写成 target-contract fact

#### 12.3.61.8 `t14` 的代表回归样本也应相应调整

新的回归样本不再只是验证“检索是否更聪明”，而是验证“系统是否还会造假归因”。

主样本至少应包括：

- `ragas-cuad-0041`
  - 主验：
    - attribution fraud
    - grounding rupture
    - target-contract claim calibration
- `ragas-cuad-0042`
  - 防止在修 `0041` 时破坏已有的 section-anchor / target-clause 行为
- `ragas-cuad-0043`
  - 防止新的 provenance 约束把本来应合法跨文档对齐的 section/context 题全部压死
- `ragas-cuad-0002`
  - 防止 support/provenance 收紧后伤到单跳精确题

若后续还需要扩展回归集，也应优先扩那些能够验证：

- external evidence 是否被错误升级成 target finding
- backing chunks 与 findings 是否仍保持物理一致

##### 12.3.61.9 当前建议

因此，当前对 `t14` 的实施口径应明确改写为：

- 不是继续沿原版 `P1/P2/P3` 做局部补丁
- 不是继续把问题理解成单纯的 bridge/runtime regression
- 而是把 `t14` 定义成：
  - 一轮围绕 provenance integrity
  - claim-level grounding
  - target-contract attribution discipline
  的可信化收敛迭代

只有这轮收紧完成后，`refactor_v1` 才能真正从：

- “结构上已经跑通”

进入：

- “行为上不再会通过错配证据制造伪结论”

### 12.3.62 `t14` 最新复盘：系统已经开始围绕同一组中层误判“原地转圈”

在 `t14` 迭代落地并经过 `0041 / 0043` 的最新真实接口复测后，需要给当前状态一个更直白的结论：

- 系统不是“又出现了一个新的孤立 bug”
- 而是已经开始围绕同一组不稳定中层判定做原地转圈式迭代

这里的“转圈”不是指没有做出改进，而是指：

- 每轮修复都能压住一类显性症状
- 但由于底层控制顺序没有真正改掉
- 同一条误判链会在另一道题、另一种表现形态中重新出现

这组核心中层判定主要包括：

1. `decomposition / requirement classification`
2. `sub-question scope`
3. `support rerank / evidence ownership`
4. `answer_composition_mode`
5. `final synthesis`

当前的问题不在于某一层单独失效，而在于：

- 前面一层的误判
- 会被后面几层继续放大并合理化

所以系统表现出来的就不是“简单答错”，而是：

- 先把题解释成某种模式
- 再按这种模式重组证据
- 最后产出一段逻辑自洽但事实归属错误的答案

#### 12.3.62.1 为什么说已经开始“原地转圈”

从文档回溯可以看到，当前 `0043` 暴露出的失真，并不是第一次出现，只是这次已经从“检索编排失败”升级成了“事实归属倒置”。

此前已经出现过的前兆至少包括：

1. `12.3.54.7 ~ 12.3.54.10`
   - 已明确警告：
     - `scope_type` 不能主要由通用术语触发
     - 不允许因为泛词共现直接触发 `cross_doc_bridge`
   - 这说明系统早就存在：
     - 普通题 / 单跳题被误桥接化

2. `12.3.57.2 ragas-cuad-0043`
   - 已经指出：
     - `0043` 会被错误执行成单文档 hard scope
     - 主要问题在 `decomposition + bridge routing + doc scope`
   - 这说明 `0043` 这题一直不是稳定的“后段语言问题”
   - 而是上游编排长期不稳

3. `12.3.58`
   - 为了修 `0043`，系统把它重新定义成：
     - `clause-context + external-definition bridge`
   - 这一步在当时是必要的
   - 但同时也把一个新的题型先验植入到了系统里：
     - `0043` 天然应该按 bridge 解释

4. `12.3.60.7`
   - 虽然案例主角是 `0041`
   - 但已经把同构机制说得很清楚：
     - scope 先选错
     - support 再按错 scope 强化错误文档
     - synthesis 最后把错误中间态写成最终结论

也就是说，系统并不是在经历两种完全无关的失败。

更准确地说：

- 早期 `0043` 暴露的是：
  - 题目没有被正确检索 / 编排
- 中期为了修复它，系统引入了更强的 bridge 化解释框架
- 最新 `0043` 则说明：
  - 当这套 bridge 先验没有被主合同事实优先确认机制约束时
  - 它会反过来把本来已经检到的主合同事实洗成 external evidence

所以这不是“新问题替代旧问题”，而是：

- 同一组不稳定中层判断
- 在不同迭代中先后表现成：
  - 检索不到
  - scope 误判
  - support ownership 漂移
  - 主合同事实自我否认

#### 12.3.62.2 `0043` 最新回退形态：不是检不到，而是“主合同事实被误桥接化”

`0043` 这次线上返回说明，系统已经从“检索失败”进一步退化成：

- 明明检到了主合同中的关键事实
- 却在中层解释中把它重新归类成 external_related
- 最后再在 final answer 中自我否认主合同事实

这次表现最关键的两个现象是：

1. 主合同事实的“自我否认”
   - Backing Chunks 中主合同 `OFGBANCORP` 已经明确写出：
     - `Privacy Regulations` shall mean the regulations promulgated under `Section 504`
   - 但最终回答仍写成：
     - 主合同没有明确 naming `Section 504`
     - 只给出了模糊的通用法规定义

2. 证据归属倒置
   - `sq_1` 实际 support 在 `OFGBANCORP`
   - 但因为其 `target_doc_id / inferred_target_doc_id` 被定成了 `Neoforma`
   - 所以 `_classify_sub_question_evidence_scope()` 只能把这条主合同事实判成：
     - `external_related`
   - 随后整题又被升级为：
     - `answer_composition_mode = cross_document_bridge`

于是本来应该是：

- 主合同单跳可回答的事实题

被链式扭曲成：

- 主合同 facts 不充分
- 外部合同提供 semantic alignment
- 再做 bridge conclusion

这就是为什么这次 `0043` 的最终错误不是简单 hallucination，而是：

- 错误 scope
- 错误 ownership
- 错误 composition mode

三者共同作用下产生的“结构化错答”。

#### 12.3.62.3 为什么会开始围绕同一问题“转圈”

当前系统最根本的顺序问题仍然是：

- 它经常先判断“这是什么题”
- 再决定“证据应该属于谁”
- 然后再让 synthesis 去解释这些已被模式化的中间态

而不是：

1. 先确认主合同里有哪些已经 grounded 的事实
2. 再决定 external evidence 是否仍有必要
3. 最后才决定是否需要 bridge / discrepancy / compare 叙述模板

只要顺序还是“题型先验 -> ownership -> synthesis”，系统就会不断在两类失败之间摆动：

1. 修得太保守
   - 本来需要 cross-doc bridge 的题
   - 被压成单文档 hard scope
   - 表现为“检不到”

2. 修得太开放
   - 本来主合同已足够回答的事实题
   - 又被抬成 bridge / external alignment
   - 表现为“主合同事实被外部化”

这就是当前“原地转圈”的本质：

- 不是 retrieval 能力不够
- 不是 prompt 技巧不够
- 而是中层 orchestration 的控制顺序还没有被真正锁死

#### 12.3.62.4 对 `t14` 当前状态的结论修正

因此，`t14` 当前不能再被理解成：

- 一轮已经基本完成、只剩细节收尾的可信化迭代

更准确的判断应是：

- `t14` 已经把 provenance / claim grounding 的方向提出来了
- 也已经修掉了一部分最粗暴的 attribution fraud
- 但它还没有真正改变系统的主控制顺序

所以目前系统仍然会在以下路径上持续复发：

- `sub-question target_doc_id` 先错
- `evidence_scope_label` 被迫跟着错
- `answer_composition_mode` 再把错误升级成 bridge
- `final answer` 最后把这一切语言化

换句话说：

- `t14` 解决了“证据能不能随便被改写”的一部分问题
- 但还没有解决“谁有权先定义这道题该怎么被解释”的问题

这也是为什么接下来不能继续沿 `t14` 的修补式路线加条件，而必须进入下一轮迭代。

### 12.3.63 `t15` 迭代规划：从“回答子问题”转向“构建事实账本”

基于 `t14` 的最新复盘，`t15` 需要明确换一个方向：

- 不再让子问题阶段承担“理解这道题是什么题型”的职责
- 而是先让系统稳定地产出一份不可篡改的事实账本
- 再在最终阶段延迟决定使用哪种叙述模板

这轮的核心逻辑应从：

- `question -> classify type -> answer sub-questions -> synthesize`

改成（**可循环**，以支持子问题检索链路上的 **IRCoT**）：

- `question -> [IRCoT loop] -> late commitment synthesis`
- 其中 **`[IRCoT loop]`** 的每次迭代都遵循文档 **`12.3.63.0`** 的四步（Thought → 物理打标 → 账本 → 终止判定）；仅在终止判定为「足够」时，才将**整份 Inventory** 一次性交给 Synthesis，而不是在循环中途写最终结论。

换句话说，`t15` 的核心原则是：

- 从“回答子问题”
- 转向“构建事实清单”
- 且该清单必须作为**系统显式内存（Inventory）**维护，并与 **IRCoT 的检索迭代**同频更新，而不是依赖模型在上下文里“记住碎片”

#### 12.3.63.0 `t15` 流程修正：保证子问题检索可走 IRCoT 多轮迭代

为保证「检索—Thought—再检索」可闭环迭代，**不得**把「物理打标 / 账本更新」与「是否还需要检索」混在同一次 LLM 调用里随意完成。`t15` 在子问题/多跳检索链路上的主循环应按下面**四步**编排；每一轮 IRCoT 均执行第 1→2→3 步，经第 4 步决定是进入 **Synthesis** 还是开启下一轮检索。

**第一步：LLM 指挥（IRCoT Thought）**

- 调用 LLM：在**仅基于当前已有证据片段**（含已写入 Inventory 的摘要指针或结构化视图，而非要求模型凭纯记忆）的前提下，判断「是否还需要额外信息才能可靠地回答用户问题」。
- 若需要：输出**新的检索词 / 检索子问题 / 检索意图**（实现层可与现有子问题分解对齐）。
- **硬约束：禁止在这一步给出最终结论、禁止输出面向用户的完整答案。** 本步职责仅限于「是否需要继续搜、下一轮搜什么」，不是「这道题是什么题型」也不是 bridge/compare 叙事。

**第二步：物理打标（Physical Tagging）**

- 检索返回的 Chunk 经 **Reranker** 重排后，**不经 LLM 指派归属**。
- 由 **`backend.py`（Python）** 根据 `doc_id` 与当前会话中的主合同标识、救援/外部文档集合做**确定性对比**：
  - 若 Chunk 属于**主合同**：标记为 `Fact_primary`（与既有 `PRIMARY_TARGET` / `source_scope` 设计对齐即可，名称以代码枚举为准）。
  - 若 Chunk 属于**救援文档**：标记为 `Fact_rescue`（与 `EXTERNAL_RESCUE` 等对齐）。
- `source_scope` / primary vs rescue **必须由代码生成**，严禁由本步或后续叙述模型改写。

**第三步：账本更新（Ledger Update）**

- 将本步得到的、已带物理标签的「事实碎片」写入**临时内存对象（Inventory / Fact Ledger）**，作为系统显式状态。
- **不得**把「只把片段留在对话上下文里、靠模型记住」当作账本——避免上下文窗口漂移、以及叙事层对归属的隐性重写。

**第四步：终止判定（Termination Condition）**

- 由模型判定（单独一次结构化调用即可）：「**当前 Inventory 中的事实是否已足以回答用户问题？**」
- **若否**：回到**第一步**，开启下一轮 IRCoT（新检索输入来自第一步；第二步—第三步在新一轮上继续向 Inventory **累积或去重合并**，合并策略在实现阶段用确定性规则 + 可选轻量模型辅助，但归属仍以第二步为准）。
- **若是**：将**整份 Inventory** **一次性**喂给 **Synthesis** 模块（其内再进行 Late Commitment 的叙述模板选择与成文）。

**与下文 `P0–P3` 的关系（避免读文档时打架）**

- **`12.3.63.2`（物理不变量）** 对应上文的**第二步**；**`12.3.63.3`（账本）** 对应**第三步**。
- **`12.3.63.4`（Late Commitment）** 仅发生在**第四步判定为「足够」之后**的 Synthesis 阶段，不得在**第一步**或「尚未终止 IRCoT 循环」时提前选定 bridge/compare 叙事并反噬事实归属。
- **Guardrail（`12.3.63.6` 中的 P3）** 仍以最终成文与 Inventory 的一致性为准，与 IRCoT 是否多轮无关。

#### 12.3.63.1 核心原则：子问题阶段不再尝试“理解题型”，只做事实搬运

`t15` 最关键的设计变化是：

- 不要让模型在子问题阶段尝试决定：
  - 这是 bridge 题
  - 这是 compare 题
  - 这是单跳题
  - 还是 discrepancy 题

因为当前转圈的根源正是：

- 一旦子问题阶段先带入题型先验
- 后面的 scope / ownership / synthesis 就会被它整体带偏

因此，`t15` 中子问题阶段的角色应被重新定义成：

- 不是“回答者”
- 而是“搬运工”（对应 **`12.3.63.0` 的第二、三步**：检索结果经 Rerank 后由代码打标并写入 Inventory）

与之并列、但**职责不同**的是 **`12.3.63.0` 的第一步与第四步**：

- **第一步（IRCoT Thought）**：只负责「还要不要搜、搜什么」，**禁止下最终结论**。
- **第四步（终止判定）**：只负责「Inventory 够不够」，**不够则回到第一步**，够了才进入 Synthesis。

「搬运工」段的职责应只包括：

1. 执行检索（检索请求可由第一步或上层分解产生）
2. 返回与该检索步相关的原子事实候选，并经代码完成 `Fact_primary` / `Fact_rescue` 标注
3. 将碎片写入 Inventory，而不是把归属交给子问题 LLM 自由书写

它不负责：

1. 提前判断整题最终应写成什么模板
2. 提前推断这条事实在最终答案里是否属于 bridge leg
3. 提前把事实综合成自然语言结论（该能力仅应在 Inventory 已就绪后的 Synthesis 中出现）

#### 12.3.63.2 `t15-P0`：先做纯物理属性标注（Physical Invariant）

在 `t15` 中，每条子问题检索完成后，不应直接返回：

- `sub_answer`
- `reason`
- `evidence_scope_label`

而应优先返回：

- 带物理标签的原子事实（Atomic Facts）

其最小结构建议为：

1. `fact_id`
2. `origin_sub_question_id`
3. `text`
4. `chunk_id`
5. `doc_id`
6. `source_scope`
7. `fact_type`
8. `grounding_status`

最关键的是 `source_scope`：

- 它必须由 `backend.py` 根据 `doc_id` 与当前主合同物理对比后生成
- 严禁由 LLM 自由修改

例如：

- `Fact_1`
  - `text = ...`
  - `doc_id = OFG_001`
  - `source_scope = PRIMARY_TARGET`

- `Fact_2`
  - `text = ...`
  - `doc_id = NEO_002`
  - `source_scope = EXTERNAL_RESCUE`

这里的关键不是字段名，而是约束关系：

- `source_scope` 是物理不变量
- 它不是解释结论
- 也不是 LLM 可协商的标签

一旦这层成立，就能直接锁死：

- `sq_1` 的 OFG 证据永远不能在后面被“解释成” Neoforma 事实

#### 12.3.63.3 `t15-P1`：先生成“事实账本”（Fact Ledger / Inventory），再进入叙事层

在进入最终 synthesis 之前，系统应先把所有 atomic facts 聚合成一份非叙事性的账本，而不是立刻写自然语言答案。

建议的中间表示类似：

事实账本（Internal Representation）：

1. `[主合同事实]`
   - `Section 504` 授权了 `Privacy Regulations`
   - 来源：`OFG_001`

2. `[外部参考事实]`
   - `Section 10.5.2` 在 `Neoforma` 协议中定义了培训中心运营上下文
   - 来源：`NEO_002`

3. `[检测结果]`
   - 主合同未发现 `Section 10.5.2`
   - 来源：`System Probe`

这一步最重要的设计意图是：

- 把“事实存在与否”
- 和“这些事实应该如何被组织成答案”

彻底拆开。

也就是说，在 `Fact Ledger` 阶段：

- 系统只陈列事实
- 不做桥接结论
- 不做 compare 总结
- 不做外部语义升格

这样可以直接切断当前系统常见的错误路径：

- 子问题一被桥接化
- support ownership 就跟着被桥接化
- 最终 facts 还没稳定，答案模板已经先成型

#### 12.3.63.4 `t15-P2`：延迟触发“合成模式”（Late Commitment）

在 `t15` 中，只有当模型看到整份事实账本后，它才被允许决定应该用哪种叙述模板。

也就是说：

- `cross_document_bridge`
- `target_contract_primary`
- `discrepancy_report`
- `compare`

这些不应再主要由前段题型先验决定，而应由账本中的事实分布来触发。

理想顺序应是：

1. 模型观察到账本中：
   - 主合同缺失概念 A
   - 外部文档存在概念 A
   - 且两者存在可解释的语义对应

2. 在此基础上，才触发：
   - `cross_document_bridge`
   - 或 `discrepancy_report`

这和当前路径的本质区别是：

- 现在是“因为我是 bridge 题，所以我要把证据洗成 bridge 逻辑”
- `t15` 应改成“因为事实就分布在两份合同里，所以我被迫写成 bridge 形式”

所以 `Late Commitment` 的意义不是为了让模板更多，而是为了避免：

- 模板先验反过来塑造事实归属

#### 12.3.63.5 为什么 `t15` 必须这么改：它直接对应当前“转圈”根因

这条路线并不是新的抽象偏好，而是直接针对当前系统“原地转圈”的根因。

它至少能解决三类当前反复出现的问题：

1. 消除“题型先验”偏置
   - 以前是：
     - 我是 bridge 题
     - 所以我要把所有证据都洗成 bridge 逻辑
   - 现在应改成：
     - 事实已经摆在这里
     - 它们物理上分布在两份合同
     - 所以最终只能写成 bridge 格式

2. 锁死 ownership
   - 通过 `chunk_id / doc_id / source_scope` 的物理绑定
   - 让 `sq_1` 的证据无法再被解释成属于另一份合同
   - 模型可以做语义对齐
   - 但不能篡改物理来源

3. 防止 synthesis 掩盖错误
   - 一旦有了 `Fact Ledger`
   - 就可以加一层轻量级 guardrail：
     - 若最终答案说“主合同有 10.5.2”
     - 但账本里这条事实只存在于外部参考
     - 则系统直接拦截或报警

也就是说，`t15` 的目标不是让 synthesis 更聪明，而是：

- 让 synthesis 没有机会掩盖前面的错误

#### 12.3.63.6 `t15` 的最小实施顺序

这一轮不应再从提示词或局部 rerank 开始，而应先落实 **`12.3.63.0` 的 IRCoT 四步编排**，再按下面分层推进（编号与 **`12.3.63.0`** 对应，避免实现时漏掉「可迭代」）：

0. **IRCoT 控制面（先于一切叙事）**
   - 实现「第一步 Thought」与「第四步终止判定」两个 LLM 调用边界：**Thought 不得产出最终结论**；**终止为否时回到 Thought**，为是时带着**整份 Inventory**进入下游。
   - Inventory 的读取接口应对 Thought/终止可见，但**归属标签仍以第二步的 Python 打标为准**。

1. `P0 Physical Invariant`（对应 **`12.3.63.0` 第二步**）
   - 先让每条 atomic fact 拥有不可改写的 `doc_id + source_scope`（`Fact_primary` / `Fact_rescue`）

2. `P1 Fact Ledger`（对应 **`12.3.63.0` 第三步**）
   - 把子问题输出从「记在上下文里的 `sub_answer`」改造成**进程内显式 `fact inventory`**，供多轮 IRCoT 累积

3. `P2 Late Commitment`（仅在 **终止判定为「足够」之后** 的 Synthesis 中）
   - 最终答案才根据**整份 Inventory 的事实分布**决定叙述模式，而不是在 IRCoT 循环内提前拍板

4. `P3 Guardrail`
   - 在 final answer 生成后，加一层账本一致性校验

原因是：

- 若没有 **`12.3.63.0` 的循环编排**
  - 物理打标与账本仍会退化成「单次检索 + 模型心里记账」，无法稳定支持 IRCoT
- 若没有 `P0`
  - 账本本身就不可信
- 若没有 `P1`
  - 题型先验仍会在子问题阶段污染事实
- 若没有 `P2`
  - bridge / compare / discrepancy 仍会提前夺走控制权
- 若没有 `P3`
  - 最终 synthesis 仍可能在语言层绕过前面约束

#### 12.3.63.7 `t15` 的核心验收标准

`t15` 完成后，至少应满足以下行为约束：

1. **IRCoT 第一步**不得输出最终结论或面向用户的完整答案；**第四步**为「是否足够」的显式判定，不足则回到第一步而非强行合成
2. **Inventory** 为系统显式对象，事实碎片不得仅依赖上下文「记在模型心里」充当账本
3. 子问题/检索链路返回的是 atomic facts（及累积后的账本视图），而不是带综合判断的 `sub_answer` 作为主中间表示
4. 每条 atomic fact 的 `doc_id / source_scope`（`Fact_primary` / `Fact_rescue`）都由 backend 物理生成，LLM 不可修改
5. 最终是否进入 `cross_document_bridge`
   - 必须由 `Fact Ledger` 的事实分布触发
   - 不能由题型先验直接拍板
6. 若主合同中已存在某条定义
   - 该事实不得在后续 stages 被重新归类成 external_related
7. 若账本中某条信息只存在于外部参考
   - 最终答案不得把它写成主合同事实
8. 若最终答案与账本的 `source_scope` 冲突
   - 系统必须能够拦截、降级或报警

#### 12.3.63.8 `t15` 的代表回归样本

`t15` 的主回归样本至少应包括：

1. `ragas-cuad-0043`
   - 验证：
     - 主合同单跳事实不会再被误桥接化
     - `Section 504 / Privacy Regulations` 不会再被外部化

2. `ragas-cuad-0041`
   - 验证：
     - external fact 不会再被贴成 target fact
     - bridge leg 仍可在 late commitment 阶段被正确总结

3. `ragas-cuad-0042`
   - 验证：
     - section-anchor / target clause 类型题不会因账本化而退化

4. `ragas-cuad-0002`
   - 验证：
     - 单跳精确题不会被无意义地升级成 bridge/discrepancy

##### 12.3.63.9 当前建议

因此，下一轮 `t15` 的实施口径应明确收敛为：

- 不再继续围绕 `scope tie-break / support rerank / wording guardrail` 打补丁
- 不再让“题型先验”主导 facts 的 ownership
- 而是把系统的核心执行顺序改成（详见 **`12.3.63.0`**）：
  - `IRCoT Thought（禁结论）`
  - `retrieve + rerank`
  - `physical fact binding（Fact_primary / Fact_rescue）`
  - `fact ledger / inventory（显式内存，非上下文记忆）`
  - `termination -> 不足则回到 Thought，足够则整包进 synthesis`
  - `late commitment synthesis`
  - `guardrail`

只有这样，系统才有机会真正停止当前这种：

- 修一轮 bridge
- 爆一轮 ownership
- 再修一轮 wording
- 最后又回到桥接误判

的原地转圈式迭代。

#### 12.3.63.10 `t15` 的中间对象设计草案（面向 `backend.py`）

为了避免 `t15` 再次停留在概念层，下一步应直接把中间对象定义到足够可实施的粒度。

最小建议至少引入三类对象（与 **`12.3.63.0`** 对齐：`FactLedger` 即进程内 **Inventory** 的规范名；IRCoT 每一轮向其中**累积/去重**，终止判定通过后再整体交给 Synthesis）：

1. `AtomicFact`
   - 作用：
     - 作为子问题检索后的最小事实单元
   - 建议字段：
     - `fact_id`
     - `origin_sub_question_id`
     - `fact_text`
     - `chunk_id`
     - `doc_id`
     - `source_scope`
     - `fact_type`
     - `grounding_status`
     - `anchor_terms`
     - `section_refs`
     - `selection_stage`
   - 关键约束：
     - `doc_id / chunk_id / source_scope` 由 backend 生成
     - LLM 只能消费，不能改写
     - `source_scope` 在实现层可与 **`Fact_primary` / `Fact_rescue`**（主合同 / 救援文档）及既有 `PRIMARY_TARGET` / `EXTERNAL_RESCUE` 等枚举**互映射**，且仅能在 **`12.3.63.0` 第二步** 由 Python 依据 `doc_id` 对比写入

2. `FactLedger`
   - 作用：
     - 汇总整题所有 `AtomicFact`
     - 在叙事之前形成稳定的内部事实账本
   - 建议字段：
     - `primary_target_facts`
     - `external_reference_facts`
     - `system_probes`
     - `missing_expected_facts`
     - `fact_conflicts`
     - `ledger_summary`
   - 关键约束：
     - 账本阶段不做 narrative synthesis
     - 只做：
       - 分类
       - 去重
       - 缺失项探测
       - 物理 ownership 汇总

3. `SynthesisDecision`
   - 作用：
     - 在 **`12.3.63.0` 第四步** 判定「Inventory 已足够」之后、**一次性**输入完整账本时，延迟决定最终叙述模板（**不在** IRCoT 循环内的 Thought/终止步提前决定）
   - 建议字段：
     - `mode`
       - `target_contract_primary`
       - `cross_document_bridge`
       - `discrepancy_report`
       - `compare`
     - `decision_reason`
     - `required_fact_ids`
     - `forbidden_fact_ids`
     - `required_sections`
   - 关键约束：
     - `mode` 必须由账本状态触发
     - 不能由前段题型模板直接指定

如果需要再往下细化，`backend.py` 中当前最适合承接这些对象的层次大致是：

0. **IRCoT 调度层（包裹子问题/检索循环）**
   - 实现 **`12.3.63.0` 第一步 + 第四步** 与检索触发器的衔接；循环内每轮执行检索 → Rerank → 第二步打标 → 第三步写 `FactLedger`

1. 单次检索步完成后
   - 用 `AtomicFact` 替代当前直接产出 `sub_answer + reason` 作为**主**中间表示（兼容字段可保留）

2. `_build_refactor_reasoning_plan()` 之前（或仅在终止后、进入最终合成前）
   - 新增/强化 `FactLedger` 构建层，保证输入为**完整 Inventory** 而非零散对话记忆

3. `_infer_answer_composition_mode()` 之前或取代它（且仅在 **Inventory 已就绪** 之后）
   - 由 `SynthesisDecision` 接管最终叙述模式选择

#### 12.3.63.11 `t15` 对现有字段的迁移与降级策略

`t15` 不适合在当前字段体系上继续叠加能力，否则很容易再次回到 `t14` 的修补路线。

因此，建议明确区分：

1. 保留但降级为兼容字段
2. 新增并逐步替代的账本字段

建议迁移关系如下：

1. `sub_answer`
   - 当前角色：
     - 子问题阶段的自然语言结论
   - `t15` 中的处理：
     - 降级为兼容输出
     - 不再作为主中间表示
   - 替代物：
     - `AtomicFact[]`

2. `reason`
   - 当前角色：
     - 子问题阶段的解释文本
   - `t15` 中的处理：
     - 降级为调试字段
     - 不再主导 final synthesis
   - 替代物：
     - `FactLedger.system_probes`
     - `SynthesisDecision.decision_reason`

3. `evidence_scope_label`
   - 当前角色：
     - 用一条聚合标签概括事实归属
   - `t15` 中的处理：
     - 不再作为事实层主字段
     - 改为账本汇总结果或兼容衍生字段
   - 替代物：
     - `AtomicFact.source_scope`

4. `support_doc_ids`
   - 当前角色：
     - 既像 provenance，又像聚合解释
   - `t15` 中的处理：
     - 只保留为兼容展示字段
   - 替代物：
     - `AtomicFact.doc_id`
     - `FactLedger.primary_target_facts / external_reference_facts`

5. `answer_composition_mode`
   - 当前角色：
     - 较早决定整题最终叙述模式
   - `t15` 中的处理：
     - 改成 late-commitment 的结果字段
     - 不再前置影响 facts ownership
   - 替代物：
     - `SynthesisDecision.mode`

换句话说，`t15` 不是单纯“新增账本字段”，而是要逐步把当前这套：

- `sub_answer`
- `reason`
- `evidence_scope_label`
- `answer_composition_mode`

从“主控制字段”

降成：

- 兼容输出 / 调试输出 / 衍生输出

#### 12.3.63.12 `t15` 的第一批测试 Gate（必须先过再继续扩写）

为了避免 `t15` 再次进入“边写边漂”的状态，建议在正式推进前先锁定第一批必须通过的测试 Gate。

第零组：IRCoT 编排 Gate（对应 **`12.3.63.0`**）

1. **Thought 步**：对固定 fixture 断言输出中不包含「最终答案 / Final Answer」类成文结论（仅允许检索意图、是否继续搜、新检索词等）。
2. **终止步**：在 Inventory 故意不足的场景下必须返回「继续检索」路径并回到 Thought；在 Inventory 已含必要主合同 chunk 的场景下不得无限循环。
3. **Inventory**：同一轮或多轮累积后，事实归属以第二步 Python 打标为准，Thought 文案不得改写已入账 `doc_id/source_scope`。

第一组：物理 ownership Gate

1. 若 `AtomicFact.doc_id = OFG`
   - 后续任何阶段不得把它改写成 `Neoforma`
2. 若某条事实来自 external doc
   - 它不得在账本中进入 `primary_target_facts`
3. 若系统 probe 发现主合同不存在某条 section
   - 该检测结果必须单独入账
   - 不得与 external fact 混写成同一条 claim

第二组：单跳保护 Gate

1. `0043` 中 `Privacy Regulations -> Section 504`
   - 若主合同已有定义
   - 则不得被升级成 external semantic alignment
2. `0002`
   - 单跳 exact fact 题不得被误升级成 bridge / discrepancy

第三组：bridge 合法性 Gate

1. `0041`
   - external fact 可以进入账本
   - 但必须保持 `EXTERNAL_REFERENCE`
   - 不得写成 target fact
2. `0042`
   - section-anchor / target-clause 题不得因账本化丢失 target priority

第四组：final synthesis Guardrail Gate

1. 若 `Final Answer` 中声称：
   - 主合同定义了 X
   - 账本里必须存在：
     - `PRIMARY_TARGET` 的对应 fact
2. 若 `Final Answer` 中声称：
   - 主合同不存在 X
   - 账本里必须不存在：
     - 对应的 `PRIMARY_TARGET` positive fact
3. 若 `Final Answer` 中使用 external definition
   - 必须显式保留 external attribution

这四组 Gate 的意义是：

- 先把最容易再次“转圈”的路径钉死
- 再考虑更复杂的优化项

#### 12.3.63.13 `t15` 的第一批非目标（明确不做）

为防止 `t15` 再次 scope 扩散，建议同步写清楚这一轮不优先做什么。

当前不应优先投入的方向包括：

1. 继续扩展题型 taxonomy
   - 例如再细分更多 bridge / compare 子类

2. 继续增加 rerank 规则分支
   - 尤其是不先引入 `AtomicFact / FactLedger` 就直接补条件

3. 继续优化文风层 prompt
   - 在账本化之前，文风收紧只会继续掩盖中层错位

4. 继续把 `0043` 当成“更聪明的 bridge routing”问题
   - 当前更核心的是：
     - 先确认主合同事实
     - 再决定外部桥接是否必要

因此，`t15` 的范围应刻意收窄为：

- 中间对象
- 账本构建
- late commitment
- guardrail

而不是继续横向扩张更多模板和条件分支。

### 12.3.64 `t15` 新结果阶段复盘：`0043` 已修复，但系统进入“证据在手、结论偏保守”的新区间

在 `doc_id` 对齐链路修复后，重新跑出的 `outputs/results/ragas/retrieval_requirements_smoke_t15` 需要给当前 `t15` 状态一个新的阶段判断。

这次复盘最重要的前提变化是：

- `0043` 不再表现为“主合同事实被误外部化”
- 说明 `t15` 的 `Fact Ledger + Physical Invariant` 路线已经开始在真实接口上发挥作用
- 但与此同时，系统也开始更频繁地把问题收束成：
  - `Fact Ledger` 不足
  - 因此拒绝给出更强结论

这意味着 `t15` 的主故障模式已经发生迁移：

1. 旧阶段主故障：
   - `scope / ownership / composition mode` 串错
   - 最终表现为“结构化错答”
2. 当前阶段主故障：
   - 主合同事实不再容易被改写
   - 但系统仍然不会稳定地把“已经到账本里的事实”组织成足够具体、又不越界的最终答案

换句话说，`t15` 已经不是“还在原地转圈”的状态，而是进入了一个新的、更可控但也更保守的问题区间：

- 以前是：错归属后自信作答
- 现在是：先把归属守住，但经常停在“证据不足，不敢继续成文”

#### 12.3.64.1 汇总指标：`t15` 相比 `t12` 的真实变化

本次新跑 `t15` 汇总行（`method = youtu_graph_rag`）为：

1. `answer_correctness = 0.474506`
2. `faithfulness = 0.587809`
3. `context_precision = 0.522222`
4. `context_recall = 0.740212`
5. `final_chunk_precision = 0.244048`
6. `final_chunk_recall = 0.777778`

对照 `t12`：

1. `answer_correctness`
   - `t12 = 0.467153`
   - `t15 = 0.474506`
   - 小幅上升
2. `faithfulness`
   - `t12 = 0.808957`
   - `t15 = 0.587809`
   - 仍显著低于 `t12`
3. `context_precision`
   - `t12 = 0.522222`
   - `t15 = 0.522222`
   - 基本持平
4. `context_recall`
   - `t12 = 0.725397`
   - `t15 = 0.740212`
   - 小幅上升
5. `final_chunk_precision`
   - `t12 = 0.144444`
   - `t15 = 0.244048`
   - 明显提升
6. `final_chunk_recall`
   - `t12 = 0.759259`
   - `t15 = 0.777778`
   - 小幅提升

这组数据说明：

- `t15` 的主要收益不在“回答语言质量”本身
- 而在：
  - 最终证据选择更干净
  - 主合同证据更稳定保住
  - 最终回答更少把错文档写成对文档

因此，这轮不能简单理解成“回答变强了”或“回答变差了”，而应理解成：

- 系统已经完成了一轮重要的中层去幻觉化、去越权化重构
- 但最终成文能力还没有跟上这套更严格的证据约束

#### 12.3.64.2 `0043` 已经完成本轮最关键的修复验证

`0043` 这次最重要的变化是：

1. `response_for_eval` 已明确写出：
   - `Section 504` 是目标 `Outsourcing Agreement` 中 `Privacy Regulations` 的法源基础
2. `retrieved_doc_ids` 已全部落在：
   - `OFGBANCORP_03_28_2007-EX-10.23-OUTSOURCING AGREEMENT#p0`
3. 样本指标为：
   - `context_precision = 1.0`
   - `context_recall = 0.666667`
   - `doc_hit = 1.0`
   - `doc_purity = 1.0`
   - `faithfulness = 0.75`

这说明此前最危险的错误链条已经被切断：

1. 主合同事实不再被洗成 `unknown_doc`
2. 明明位于 target contract 的定义不再被外部化
3. `doc_id` 对齐修复已经在真实运行结果中生效

因此，`0043` 当前不应再被视为：

- `bridge routing`
- `scope tie-break`
- 或“更聪明的外部对齐”

问题。

当前它已经证明：

- `t15` 的 `Fact Ledger / Physical Invariant` 方向是正确的
- 而且不是停留在设计层，而是已经能在真实接口中阻断 attribution fraud

#### 12.3.64.3 四个代表样本反映出的当前系统状态

当前最值得继续跟踪的四个样本仍是：

1. `ragas-cuad-0002`
2. `ragas-cuad-0041`
3. `ragas-cuad-0042`
4. `ragas-cuad-0043`

它们共同给出的阶段信号如下。

1. `0002`
   - 当前回答：
     - 已能保住 Motorola 相关事实
     - 但仍倾向于给出“Fact Ledger 不足以描述其具体 functional role”的保守结论
   - 指标：
     - `context_recall = 1.0`
     - `doc_hit = 1.0`
     - `faithfulness = 1.0`
   - 说明：
     - 单跳事实已检到
     - 但系统对“已知有限事实可以支持一个有限但明确的回答”仍然不够自信

2. `0041`
   - 当前回答：
     - 不再把外部 `Metavante` 事实写成 Neoforma target fact
     - 而是明确判断账本不足，拒绝构造错误 bridge
   - 指标：
     - `context_precision = 0.0`
     - `context_recall = 0.5`
     - `faithfulness = 0.444444`
   - 说明：
     - 当前系统已经更愿意拒绝错桥接
     - 但仍未形成一套稳定的“外部事实可入账，但不能被误升级，同时还能给出有限可用结论”的中间策略

3. `0042`
   - 当前回答：
     - 已拿到 breach consequence 与 indemnification 相关事实
     - 但仍收束为“无法确定这两条是否按题目要求连接”
   - 指标：
     - `final_chunk_hit = 1.0`
     - `final_chunk_precision = 0.5`
     - `final_chunk_recall = 1.0`
     - `faithfulness = 0.2`
   - 说明：
     - 当前主问题已不再是“完全检不到”
     - 而是：
       - 事实已经到账本
       - 系统却还不会把条款间关系组织成安全、受约束的结论
   - 这也是下一阶段最值得优先处理的样本

4. `0043`
   - 当前回答：
     - 已恢复主合同事实优先
     - 不再发生“主合同定义被外部化”的结构性错答
   - 说明：
     - `t15` 在最危险的 ownership 失真问题上已完成阶段性收敛

因此，这四个样本共同表明：

- 当前系统已经从“错归属后强行写答”
- 进入到“证据在手但结论偏保守”的新区间

#### 12.3.64.4 关于 `response_for_eval` 与 `faithfulness` 下滑的解释修正

这一轮还需要明确修正一个此前容易误判的问题：

- `t15` 当前的 `faithfulness` 下滑
- 并不是因为结构化回答正文污染了 RAGAS 评测

原因是：

1. 当前评测链路已经优先使用：
   - `response_for_eval`
2. 该字段本身已经只保留面向评测的简化回答
   - 不包含完整的 `Grounded Facts / Alignment / Final Answer` 调试脚手架

因此，本轮 `faithfulness` 仍低于 `t12`，更合理的解释是：

1. `t15` 现在更频繁地产生：
   - `insufficient`
   - `cannot be determined`
   - `cannot be synthesized`
   这类保守回答
2. 对 RAGAS 的 judge 来说：
   - 这类“拒绝作答 / 只做有限作答”的回答
   - 往往不会像“直接把检到的句子强行写成结论”那样获得高 `faithfulness`

也就是说：

- `t12` 的高 `faithfulness`
  - 部分建立在“把检到的内容直接成文化”之上
- `t15` 的低一些的 `faithfulness`
  - 则是“先守住归属，再谨慎作答”的副作用

所以，当前 `faithfulness` 的下降不能简单解读成：

- 回答更不忠实了

更准确的说法应是：

- 当前系统从“会用错归属的证据写出貌似很完整的答案”
- 切换成了“在证据链不闭合时更愿意停在有限结论”

这是一种产品行为层面的改进，但在当前自动评测口径下会被部分惩罚。

#### 12.3.64.5 当前阶段判断：`t15` 已完成基础层重构，但尚未完成最终回答质量收敛

综合当前实现与新结果，`t15` 应被定位为：

- 一轮成功的中层基础设施重构
- 而不是一轮已经完成最终回答收敛的迭代

更具体地说：

1. 已基本证明有效的部分
   - `P0 Physical Invariant`
   - `P1 Fact Ledger`
2. 尚未完成真正收敛的部分
   - `P2 Late Commitment`
   - `P3 Guardrail` 之后的安全成文

换句话说，当前系统已经做到：

1. 不再轻易把外部事实写成 target fact
2. 不再轻易把主合同事实外部化
3. 最终证据集比 `t12` 更干净

但它还没有稳定做到：

1. 在账本已足够时输出“有限但明确”的回答
2. 在账本跨条款、跨关系可组合时生成受约束的关系结论
3. 在保持 ownership 正确的同时提升 `answer_correctness` 与 `faithfulness`

因此，`t15` 当前不能被理解成“失败”，而应被理解成：

- 它把系统从一个更危险、也更不可控的问题区间
- 推进到了一个更安全、但需要继续补上成文能力的新阶段

#### 12.3.64.6 下一阶段规划：从 `Fact Ledger` 走向 `ledger-to-answer composition`

基于当前结果，下一阶段不应再回去优先补：

1. `bridge taxonomy`
2. `scope tie-break`
3. 更多 rerank 条件分支
4. 文风层 prompt 收紧

这些方向都更像上一阶段的问题。

当前最应优先投入的，是让系统学会：

1. **在 ownership 已正确时，输出有限但明确的 target-grounded 结论**
   - 典型样本：
     - `0002`
   - 目标不是让模型更敢猜
   - 而是让它学会：
     - 当账本里已有明确 target fact 时
     - 可以输出“有限但成立”的最小结论
     - 同时显式保留未确定部分

2. **把账本从“事实清单”升级为“可组合的 claim / relation ledger”**
   - 典型样本：
     - `0042`
   - 当前问题不是完全没检到
   - 而是：
     - 条款 A 的事实
     - 条款 B 的后果
     - 已经在账本里
     - 却还不会被安全地组织成关系结论

3. **让 `Late Commitment` 不只是“决定是否拒答”，而是“决定能安全输出到哪一层结论”**
   - 也就是说，最终阶段不应只有：
     - 充足 -> 作答
     - 不足 -> 拒答
   - 而应支持第三种中间态：
     - 证据足以支持有限结论
     - 但不足以支持完整 bridge / compare / discrepancy 叙事

4. **把 `Guardrail` 从纯拦截器升级为“结论边界控制器”**
   - 当前 guardrail 更像：
     - 防止错归属
   - 下一阶段应升级为：
     - 规定哪些 claim 可以说到什么粒度
     - 哪些只能保留为“未证明 / 未连接 / 未观察到”

#### 12.3.64.7 下一阶段建议的最小执行顺序

为了避免重新扩散 scope，下一阶段建议收敛为以下顺序：

1. 先处理 `0002`
   - 验证系统是否能基于已有 target fact 输出“有限但明确”的单跳答案
2. 再处理 `0042`
   - 让系统从“事实罗列”升级到“受约束的关系组合”
3. 最后再回看 `0041`
   - 不是为了重新强做 bridge
   - 而是验证：
     - external facts 入账后
     - 系统能否在不越权的前提下给出有限语义定位

而 `0043` 当前应从“主回归样本”降为：

- ownership 已修复后的回归守门样本

它仍需要保留，但不再是下一阶段的第一主矛盾。

#### 12.3.64.8 `t16` 迭代方案：把“事实账本”升级为“可受约束成文的结论层”

基于本轮 `t15` 复盘，`t16` 的目标不应再定义成“继续补检索”或“让模型更敢回答”，而应明确定义成：

- 在 `ownership` 已正确的前提下，把账本中的事实稳定转译成：
  - 有边界的最小结论
  - 可验证的关系结论
  - 明确标注未证实部分的部分回答

也就是说，`t16` 的主任务是补齐：

- `ledger-to-answer composition`
- `bounded claim linking`
- `safe partial answering`

三层能力，而不是回到上一阶段的 `scope tie-break / bridge patch / rerank patch`。

建议将 `t16` 收敛为以下五项实施内容。

1. **新增 Claim Ledger，取代“只有 fact list”的终态**
   - 在现有 `Fact Ledger` 之上增加一层中间对象：
     - `ClaimUnit`
   - 但这里必须明确一条实施约束：
     - `ClaimUnit` 不是新的自由语义层
     - 也不是让模型重新发明一套 SPO 知识图谱
   - 它应被定义为：
     - 从已有 `AtomicFact` 受限提升出来的“可成文结论单元”
     - 而不是脱离 `AtomicFact` 自行生成的新 claim
   - 每个 `ClaimUnit` 至少包含：
     - `claim_id`
     - `claim_type`
     - `subject`
     - `predicate`
     - `object`
     - `doc_scope`
     - `evidence_chunk_ids`
     - `support_level`
     - `open_gaps`
     - `derived_from_fact_ids`
     - `anchor_terms`
     - `section_refs`
   - 其中 `claim_type` 建议先只支持三类：
     - `direct_fact`
     - `bounded_relation`
     - `insufficient_link`
   - 目的不是让模型自由发挥生成更多叙事，而是先把“哪些事实已经可以升格为结论单元”结构化表达出来。
   - 当前阶段应优先限制 `ClaimUnit` 的生成方式：
     - `direct_fact`
       - 只能由单条或同一条款下的多条 `AtomicFact` 直接压缩生成
     - `bounded_relation`
       - 只能由同一 `doc_scope`、且共享明确锚点的 `AtomicFact` 组合生成
     - `insufficient_link`
       - 只能表达“存在两个事实点，但缺少合法连接边”
   - 换句话说：
     - `ClaimUnit` 的作用不是提高抽象度本身
     - 而是把“哪些事实已可安全说成一句结论”显式化、受约束化

2. **把 Late Commitment 改成“三段式结论决策”**
   - `t15` 的实际行为更接近：
     - 能答
     - 不能答
   - `t16` 应改成：
     - `direct_answer`
     - `partial_answer`
     - `withhold_relation`
   - 判定口径建议明确为：
     - 若账本内存在单一 target-grounded 且证据闭合的 `direct_fact`，输出 `direct_answer`
     - 若账本能支持局部事实，但不能支持完整 bridge / compare / discrepancy，输出 `partial_answer`
     - 若条款之间缺少明确连接边，只允许输出 `withhold_relation`，并显式写明缺的连接点
   - 这里需要再补一条实现约束：
     - 上述三种模式不是新的软分类题型
     - 而应是由结构化 gate 触发的结论边界
   - 建议把 gate 写硬为：
     - `direct_answer`
       - 至少存在 1 条 `PRIMARY_TARGET` 的 `direct_fact`
       - 且该 `direct_fact` 已覆盖问题核心问点
       - 且不存在同级冲突 claim
     - `partial_answer`
       - 已存在可确认的 target-grounded 结论单元
       - 但只覆盖问题的一部分
       - 或缺失 bridge / compare / discrepancy 所需的额外连接边
     - `withhold_relation`
       - 仅有事实点
       - 但不存在合法 relation edge
       - 因而禁止输出跨条款、跨文档或更强关系推论
   - 这样可以直接解决 `0002` 这类“明明已有 target fact，却仍整体拒答”的保守过度问题。

3. **引入受约束的 relation composer，只允许从账本中做有限链接**
   - `0042` 暴露出的核心问题不是事实缺失，而是“事实已在账本中，但没有安全的组合器”。
   - 因此 `t16` 应新增一个轻量 relation composer，规则先刻意做窄：
     - 只允许链接同一 `doc_scope` 下的 claim
     - 只允许链接共享条款锚点、共享定义对象或共享触发事件的 claim
     - 若 relation 缺少显式触发依据，只能生成 `insufficient_link`
   - 输出不应直接是最终自然语言，而应先产出：
     - `relation_candidate`
     - `supports`
     - `blocked_by`
     - `allowed_answer_span`
   - 这样 guardrail 就不再只是“拦截错误”，而是依据 `allowed_answer_span` 控制最终成文粒度。

4. **重写 answer composer，优先生成“有限但明确”的 target-grounded 回答**
   - `t16` 的 answer composer 不应再从全文 prompt 自由总结，而应按固定顺序组装：
     - 先写已确认的 target fact
     - 再写已确认的 relation
     - 最后写未确认部分
   - 推荐输出模板收敛为：
     - `Confirmed:` 明确可由 target ledger 支持的结论
     - `Not established:` 当前证据无法证明的连接或比较
     - `Evidence basis:` 对应条款或 chunk 依据
   - 对评测用 `response_for_eval`，则只保留前两层语义：
     - 最小成立结论
     - 未成立部分
   - 但这里也应明确一条与评测口径相关的约束：
     - `response_for_eval` 不应固定套完整模板
     - 而应根据 `answer_decision` 做最小化裁剪
   - 建议规则为：
     - 若 `answer_decision = direct_answer`
       - `response_for_eval` 只保留最小成立结论
       - 不主动附加 `Not established`
     - 若 `answer_decision = partial_answer`
       - `response_for_eval` 保留：
         - 已成立部分
         - 以及与问题强相关的未成立部分
     - 若 `answer_decision = withhold_relation`
       - `response_for_eval` 明确说明：
         - 已知事实是什么
         - 缺失的连接点是什么
   - 这样做的目标是同时压住两类失败：
     - 把未证实关系写成已证实
     - 因为有未证实部分而把已证实事实也一起吞掉

5. **把回归与验收直接绑定到 `0002 / 0042 / 0041 / 0043` 四个样本**
   - `0002`
     - 验收目标：系统必须输出明确的 target-grounded 最小答案，不能再只给“账本不足”的整体拒答
   - `0042`
     - 验收目标：系统必须把已检到的 breach / indemnification 事实组织成“已知什么、未连上什么”的受约束回答
   - `0041`
     - 验收目标：external fact 可以入账，但不能被升级成 target contract 已确认事实
   - `0043`
     - 验收目标：继续作为 ownership 守门样本，确保主合同事实不再被外部化

据此，`t16` 的最小实施顺序建议固定为：

1. 先在 `backend.py` 补 `ClaimUnit / relation_candidate / answer_decision` 三个中间对象
2. 再把 `Late Commitment` 改成 `direct_answer / partial_answer / withhold_relation` 三段式判定
3. 然后只针对 `0002` 接通单跳 `partial_answer`
4. 再针对 `0042` 接通跨条款 `bounded_relation`
5. 最后用 `0041 + 0043` 做 guardrail 回归

`t16` 的完成标准也应明确收窄为：

- 不是让所有样本都更会说
- 而是让系统稳定做到：
  - 有 target fact 时，能给出最小成立结论
  - 有多条账本事实时，能说明哪些已连接、哪些未连接
  - ownership 未闭合时，拒绝升级为更强结论

如果这三点不能稳定成立，说明系统仍停留在“事实账本存在，但结论层缺席”的阶段，`t16` 就不能算完成。

#### 12.3.64.9 `t16` 结果复盘：检索与证据边界继续变强，但回答层贴题性成为新的第一主矛盾

`t16` 跑完之后，需要对这一轮给出一个和 `t15` 不同的阶段判断。

如果说 `t15` 的核心任务是：

- 先把 ownership 守住
- 先证明系统不会再轻易把主合同事实外部化

那么 `t16` 的真实结果说明：

- 这条基础线路没有回退
- 而且 retrieval / evidence-control 还进一步变强了
- 但最终回答层并没有同步收敛

因此，`t16` 不能被简单总结为“变好了”或“变差了”，更准确的说法应是：

- 它在检索纯度、证据覆盖、结论边界控制上继续前进
- 但 answer composer 还不会稳定挑出“最贴题、最值得说的结论”

换句话说，系统当前已经从：

- `ownership` 错误

推进到：

- `relevance-aware composition` 不足

这比前一阶段更局部，也更接近最终回答层的问题。

##### 12.3.64.9.1 汇总指标：`t16` 相比 `t15` 的总体变化

`t16` 汇总行（`method = youtu_graph_rag`）为：

1. `answer_correctness = 0.439377`
2. `faithfulness = 0.638480`
3. `context_precision = 0.702778`
4. `context_recall = 0.756085`
5. `final_chunk_precision = 0.244092`
6. `final_chunk_recall = 0.851852`

对照 `t15`：

1. `answer_correctness`
   - `t15 = 0.474506`
   - `t16 = 0.439377`
   - 下降
2. `faithfulness`
   - `t15 = 0.587809`
   - `t16 = 0.638480`
   - 上升
3. `context_precision`
   - `t15 = 0.522222`
   - `t16 = 0.702778`
   - 明显上升
4. `context_recall`
   - `t15 = 0.740212`
   - `t16 = 0.756085`
   - 小幅上升
5. `final_chunk_precision`
   - `t15 = 0.244048`
   - `t16 = 0.244092`
   - 基本持平
6. `final_chunk_recall`
   - `t15 = 0.777778`
   - `t16 = 0.851852`
   - 明显上升

这组数据非常关键，因为它说明：

1. `t16` 不是整体退化
   - 它在 retrieval 和 evidence 层是明显前进的
2. 拖累整体 end-to-end 指标的主因素
   - 不是检索不到
   - 不是 ownership 回退
   - 而是最终回答层的正确性没有跟上 retrieval 的提升

也就是说：

- `t16` 已经把系统继续往“可控回答”方向推进
- 但还没有把“可控”稳定转成“更贴题、更正确”的最终答案

##### 12.3.64.9.2 `t16` 的阶段性收益：检索纯度和最终证据覆盖继续提升

从汇总看，`t16` 当前最明确的收益至少有三点。

1. `context_precision` 显著提升
   - 从 `0.522222` 升到 `0.702778`
   - 说明进入回答阶段的上下文更干净
   - retrieval 噪声更少

2. `context_recall` 继续小幅提升
   - 从 `0.740212` 升到 `0.756085`
   - 说明系统没有靠“更保守、更少检”换取精度
   - 而是在保持甚至略增覆盖的同时提升了 precision

3. `final_chunk_recall` 明显上升
   - 从 `0.777778` 升到 `0.851852`
   - 说明最终进入回答层的 evidence package 更完整

因此，`t16` 至少已经证明：

- `Claim Ledger / bounded answer / relation composer`
- 并没有把系统重新拖回“检索链路不稳”的旧区间

相反，它进一步提升了：

- 最终证据集的纯度
- 最终证据集的覆盖
- 以及 `faithfulness`

这也是为什么当前不能把 `t16` 当成“失败迭代”。

##### 12.3.64.9.3 关键样本复盘：`0002 / 0042 / 0043` 说明方向正确，`0041` 暴露出新的主问题

这一轮最值得看的四个样本仍然是：

1. `0002`
2. `0041`
3. `0042`
4. `0043`

它们共同揭示了 `t16` 当前真正的问题分层。

1. `0002`
   - `answer_correctness`
     - `0.387982 -> 0.477881`
   - 回答已经从：
     - “账本不足以描述 Motorola role”
   - 变成：
     - “Motorola 的角色被限制在商标/标识授权相关的第三方”
   - 说明：
     - `safe partial answering`
     - 和“最小成立结论”路线在单跳题上是有效的

2. `0043`
   - `answer_correctness`
     - `0.336916 -> 0.384371`
   - `context_precision`
     - 保持 `1.0`
   - 说明：
     - `t15` 修好的 ownership 基础没有回退
     - `t16` 仍然保持主合同事实优先
   - 这意味着：
     - 当前系统已不再主要卡在 `doc_id` 对齐或主合同事实外部化

3. `0042`
   - `faithfulness`
     - `0.2 -> 0.428571`
   - `context_precision`
     - `0.0 -> 1.0`
   - 当前回答已经开始尝试输出：
     - 条款关系的更直接结论
   - 说明：
     - relation composer 方向是对的
     - 系统不再只会把“breach / indemnification”事实并列摆着
   - 但 `answer_correctness` 仍略降，说明：
     - 它还不会稳定区分
       - 哪些关系已经足够成立
       - 哪些关系只适合表达为“有限成立”

4. `0041`
   - `answer_correctness`
     - `0.540031 -> 0.374590`
   - 尽管 `faithfulness` 略升
   - 但整体回答更明显地暴露出：
     - 先说了一段 target-grounded 但并不直接回答问题核心的内容
     - 再补 “What is Not Established”
   - 这说明 `t16` 当前最核心的问题不是：
     - 证据造假
     - ownership 错误
     - 或 relation 完全不会做
   - 而是：
     - **回答层不会正确筛选“哪些已证实事实值得进入主结论”**

因此，四个样本共同说明：

- `t16` 已经开始具备：
  - 最小成立结论
  - 受约束 relation
  - ownership 守门
- 但它还不具备：
  - **对“问题核心所需结论”的主次排序能力**

##### 12.3.64.9.4 当前主矛盾：不是“不会回答”，而是“不会挑最贴题的已证实结论”

`t16` 最值得强调的一点是：

- 当前系统已经不是：
  - 检不到
  - 乱桥接
  - 主合同事实被外部化
  - 或一律只会保守拒答

它现在更像是：

1. 已经能产出：
   - `Confirmed`
   - `Not established`
   - `partial_answer`
2. 但它还不会稳定判断：
   - 哪些 `Confirmed` 该进入主句
   - 哪些虽然真实，但只应作为 supporting note
   - 哪些“已知但不贴题”的信息不该挤占主回答

这就是为什么：

- retrieval 相关指标显著变好
- 但 `answer_correctness` 却没有同步上升

所以，`t16` 的真正问题并不是“结论层设计错了”，而是：

- **结论层的 relevance filtering / answer focus 还没有建立起来**

这比 `t15` 的问题更收敛，也意味着系统已经进入一个更高级、更局部的调优阶段。

##### 12.3.64.9.5 当前阶段判断：`t16` 是成功的“结论层控制”迭代，但不是成功的“结论层表达”迭代

综合数据与样本表现，`t16` 最准确的阶段定位应是：

- 一轮成功的 `answer-boundary control` 迭代
- 但还不是一轮成功的 `answer-focus composition` 迭代

换句话说：

1. 这轮已经做到
   - ownership 没回退
   - retrieval 更干净
   - final evidence package 更完整
   - partial answering 开始生效
   - relation composer 开始有结果

2. 这轮还没做到
   - 把最贴题的结论稳定放进主回答
   - 把次要但真实的结论降到 supporting role
   - 让 `Confirmed / Not established` 真正服务于问题回答，而不是仅仅服务于结构化安全性

因此，`t16` 应被视为：

- 一轮方向正确、且证据层明显前进的迭代
- 但 end-to-end answer correctness 尚未收敛

#### 12.3.65 `t17` 迭代方向：从“结论边界控制”推进到“贴题结论选择”

基于 `t16` 的结果，`t17` 的主矛盾已经可以明确收敛为：

- 不是继续补检索
- 不是继续补 ownership
- 也不是继续扩展 relation taxonomy
- 而是：
  - **让 answer composer 学会从已证实结论里挑出“最贴题、最值得说”的主结论**

换句话说，`t17` 的任务不是再造新一层中间对象，而是：

- 在已有 `ClaimUnit / relation_candidate / answer_decision` 之上
- 增加一层 **relevance-aware composition**

##### 12.3.65.1 `t17` 的核心目标

`t17` 应明确只做下面三件事：

1. **主结论选择（Main Answer Selection）**
   - 从所有 `Confirmed` claim 中
   - 只挑最能直接回答问题核心问点的那一组
   - 其余 claim 降为 supporting note 或完全不进入主回答

2. **问题核心覆盖判定（Question-Core Coverage）**
   - 在成文前显式判断：
     - 当前回答是否覆盖了题目的主谓词
     - 是否只是覆盖了背景事实
   - 若只覆盖背景事实
     - 不应把它们直接上升为主答案

3. **结构化回答压缩（Answer Compression for Evaluation and UX）**
   - `response`
     - 可以保留 `Confirmed / Not established / Evidence basis`
   - `response_for_eval`
     - 应进一步压缩成：
       - 主结论
       - 必要时补一条最相关的未成立说明
   - 避免把正确但边缘的 supporting facts 挤进评测主答案

##### 12.3.65.2 `t17` 的最小新增对象与规则

为了不让 `t17` 再次 scope 膨胀，建议只新增一个很窄的对象：

1. `AnswerFocusUnit`
   - 作用：
     - 不是生成新 claim
     - 而是给现有 `ClaimUnit / relation_candidate` 排序和筛选
   - 建议字段：
     - `focus_id`
     - `question_core_slots`
     - `selected_claim_ids`
     - `supporting_claim_ids`
     - `excluded_claim_ids`
     - `selection_reason`
     - `coverage_status`

这里的关键约束是：

- `AnswerFocusUnit` 不创造新语义
- 只负责判断：
  - 哪些已存在 claim 进入主回答
  - 哪些只作为 supporting note
  - 哪些虽真实但与问题核心不够相关，应被排除

##### 12.3.65.3 `t17` 的回答层规则应如何收窄

建议 `t17` 的 answer composer 明确遵守以下顺序：

1. 先抽取问题核心问点
   - 例如：
     - 角色
     - 影响
     - 条款关系
     - 是否构成 breach
2. 再只从 `Confirmed` claim 中挑能覆盖该问点的内容
3. 若已有 claim 只覆盖背景、未覆盖核心问点
   - 则背景 claim 不能单独占据主回答
4. `Not established`
   - 只能补充当前主问点未闭合的部分
   - 不能替代主结论

这套规则的目标是直接解决 `0041 / 0021 / 0022` 一类问题：

- 不是没有正确事实
- 而是正确事实没有被正确地用于回答题目

##### 12.3.65.4 `t17` 的主回归样本应如何重排

`t17` 的主回归顺序应调整为：

1. `0041`
   - 验证：
     - 不再让“虽然真实但不贴题”的 target-grounded 结论占据主回答
2. `0021`
   - 验证：
     - 高 `context_precision` 下，主回答仍需贴住题目核心
3. `0022`
   - 验证：
     - `Confirmed` 与 `Not established` 的结构不能损害主结论的直接性
4. `0002`
   - 继续守住：
     - 单跳 minimal answer 不回退成整体拒答
5. `0043`
   - 继续守住：
     - ownership 不回退
6. `0042`
   - 继续守住：
     - relation composer 的收益不被 relevance filter 误伤

也就是说：

- `0002 / 0043 / 0042`
  - 从“主推进样本”
  - 逐步转成“正向守门样本”
- `0041 / 0021 / 0022`
  - 则应成为 `t17` 的第一主回归样本

##### 12.3.65.5 `t17` 的完成标准

`t17` 是否完成，不应再看“回答是否更完整”，而应看以下三点是否稳定成立：

1. 当系统已有多个真实 `Confirmed` claim 时
   - 主回答只保留最贴题的一组
2. supporting facts 不再挤占主回答
   - 但仍可在详细输出中保留
3. `response_for_eval`
   - 比 `response` 更短、更聚焦
   - 且不再因结构化安全模板导致 answer correctness 被无谓拉低

如果这三点做不到，说明系统虽然已经有：

- `Fact Ledger`
- `Claim Ledger`
- `relation composer`
- `answer decision`

但仍没有真正完成：

- **面向问题核心的最终答案选择**

##### 12.3.65.6 当前结论

因此，`t17` 不应再被理解成：

- 下一轮继续补 retrieval
- 或再发明一层新的 reasoning taxonomy

更准确的定位应是：

- 在现有证据与结论控制能力之上
- 补齐“主回答该说什么、不该说什么”的最后一层选择能力

只有当这一层收敛后，系统才可能真正把：

- 更高的 `context_precision`
- 更高的 `final_chunk_recall`
- 更强的 ownership 约束

稳定转化成：

- 更高的 `answer_correctness`

否则系统就会继续停留在：

- 检得越来越对
- 但答得不够贴题

的状态中。

##### 12.3.65.7 `t17` 结果复盘：不是“贴题性收敛”，而是一次过早压缩导致的阶段性回退

`t17` 跑完后，需要明确修正对上一节 `12.3.65` 的预期判断。

原本 `t17` 的设计目标是：

- 不再继续补 retrieval
- 不再继续补 ownership
- 而是在已有 `ClaimUnit / answer_decision` 之上
- 增加一层 `relevance-aware composition`

但从真实结果看，`t17` 当前并没有把系统推进到“贴题结论选择已收敛”的阶段。

更准确地说：

- 它像是把“主结论选择”实现成了“更激进的证据与回答压缩”
- 结果既没有把 `answer_correctness` 拉回来
- 还削弱了 `t16` 已经获得的一部分 retrieval / final-pack 收益

因此，`t17` 当前应被理解成：

- 一次方向正确
- 但收敛策略过早、压缩过猛
- 从而导致阶段性回退的迭代

###### 12.3.65.7.1 汇总指标：`t17` 相比 `t16` 的总体变化

`t17` 汇总行（`method = youtu_graph_rag`）为：

1. `answer_correctness = 0.399041`
2. `faithfulness = 0.491358`
3. `context_precision = 0.525926`
4. `context_recall = 0.740212`
5. `final_chunk_precision = 0.190697`
6. `final_chunk_recall = 0.722222`

对照 `t16`：

1. `answer_correctness`
   - `t16 = 0.439377`
   - `t17 = 0.399041`
   - 继续下降
2. `faithfulness`
   - `t16 = 0.638480`
   - `t17 = 0.491358`
   - 明显下降
3. `context_precision`
   - `t16 = 0.702778`
   - `t17 = 0.525926`
   - 明显回退
4. `context_recall`
   - `t16 = 0.756085`
   - `t17 = 0.740212`
   - 小幅回退
5. `final_chunk_precision`
   - `t16 = 0.244092`
   - `t17 = 0.190697`
   - 回退
6. `final_chunk_recall`
   - `t16 = 0.851852`
   - `t17 = 0.722222`
   - 明显回退

这组数据说明：

1. `t17` 不是“回答更聚焦，所以只是 correctness 暂时没涨”
2. 它实际上同时回退了：
   - 回答正确性
   - 忠实度
   - 检索纯度
   - 最终证据覆盖

因此，`t17` 不能被解释成一次“代价可接受的探索”，而应更明确地理解成：

- answer focus 改动和 final evidence retention 发生了过度耦合
- 结果不仅没把主回答变得更贴题
- 反而让 `t16` 已经获得的 retrieval / evidence-control 收益一起被带坏了

###### 12.3.65.7.2 关键样本复盘：`0002` 稳住，`0041` 部分修正，但 `0042 / 0043 / 0022` 共同表明压缩过头

当前最关键的样本信号如下：

1. `0002`
   - `answer_correctness`
     - `t16 = 0.477881`
     - `t17 = 0.476871`
   - 说明：
     - 单跳 minimal answer 路线没有回退
     - 但也没有进一步带来明显收益

2. `0041`
   - `answer_correctness`
     - `t16 = 0.374590`
     - `t17 = 0.426451`
   - 说明：
     - `t17` 相比 `t16` 的确部分缓解了“先讲一大段不贴题 confirmed facts”的问题
     - 但仍未恢复到 `t15 = 0.540031`
   - 也就是说：
     - 它不是成功修复
     - 而只是从“明显答偏”修到“没那么偏”

3. `0042`
   - `answer_correctness`
     - `t16 = 0.387461`
     - `t17 = 0.344589`
   - `final_chunk_recall`
     - `t16 = 1.0`
     - `t17 = 0.5`
   - `final_chunk_precision`
     - `t16 = 0.5`
     - `t17 = 0.166667`
   - 说明：
     - `t16` 好不容易推进出来的 relation composer 收益
     - 在 `t17` 被过度压缩的 final package 误伤了

4. `0043`
   - `answer_correctness`
     - `t16 = 0.384371`
     - `t17 = 0.321971`
   - `context_precision / context_recall`
     - 基本未变
   - 说明：
     - ownership 基础没有回退
     - 但回答层压缩并没有带来更好的表达
     - 反而削弱了已有的解释质量

5. `0022`
   - `answer_correctness`
     - `t16 = 0.537963`
     - `t17 = 0.454096`
   - `faithfulness`
     - `t16 = 0.5`
     - `t17 = 0.333333`
   - 说明：
     - `t17` 试图让回答更直接
     - 但并没有真正解决“主结论 vs supporting fact”的排序问题
     - 只是让结果变得更短，却没有更贴题

这几个样本共同说明：

- `t17` 最大的问题不是方向错
- 而是把“主结论选择”做成了“更强压缩”

也就是说：

1. 它没有建立真正的问题核心覆盖判定
2. 也没有建立真正的 supporting-vs-main 选择机制
3. 它只是更早、更猛地删掉了上下文和成文空间

###### 12.3.65.7.3 当前结论修正：`t17` 暂不能作为下一阶段主线延续

基于本轮真实结果，需要明确修正 `12.3.65` 的执行判断：

1. `relevance-aware composition` 方向本身仍然是对的
2. 但当前 `t17` 的实现方式不能直接作为后续主线延续

因为它已经证明：

- 在“问题核心覆盖判定”没有显式落地前
- 过早压缩 `Confirmed / Not established / response_for_eval`
- 很容易演变成：
  - 既没有更贴题
  - 也没有保住证据覆盖

所以，下一阶段不应继续沿着：

- 更短回答
- 更少 final chunks
- 更激进的 evidence trimming

这条路线往前推。

更合理的结论应是：

- 必须把 `answer focus`
- 和 `evidence retention`

彻底解耦。

只有在两者解耦之后，系统才有可能既保住：

- `t16` 的 retrieval / final pack 优势

又逐步获得：

- 更贴题的主回答

#### 12.3.66 `t18` 计划：解耦 `answer focus` 与 `evidence retention`，先保住 `t16` 的证据优势，再做主回答排序

基于 `t17` 的回退结果，`t18` 的主任务不应再定义成：

- 继续压缩回答
- 继续压缩 final context
- 或继续把 `response_for_eval` 做得更短

`t18` 更合理的定位应是：

- **先恢复并锁住 `t16` 的 evidence retention 行为**
- 再在不改 final evidence package 的前提下
- 只优化主回答层的排序与选择

换句话说，`t18` 的核心原则应是：

- `answer focus` 只能影响：
  - 主回答说什么
  - supporting note 说什么
  - `response_for_eval` 如何裁剪
- 但不能再直接影响：
  - `final_chunk_selection`
  - `final evidence package`
  - relation composer 的候选输入覆盖

##### 12.3.66.1 `t18` 的核心目标

`t18` 应明确只做三件事：

1. **恢复 `t16` 级别的 final evidence retention**
   - 目标不是更少 chunk
   - 而是重新保住：
     - `context_precision`
     - `final_chunk_recall`
   - 至少不再低于 `t16`

2. **把 answer focus 限定为“回答排序层”，而不是“证据裁剪层”**
   - 让 `AnswerFocusUnit` 只做：
     - main answer claim selection
     - supporting claim selection
     - excluded-from-answer selection
   - 不允许它修改：
     - final chunk set
     - claim candidate set
     - relation candidate set

3. **显式补上问题核心覆盖判定（Question-Core Coverage Gate）**
   - 在最终成文前，系统必须先判断：
     - 当前 selected claims 是否真正覆盖题目的主问点
     - 若只覆盖背景事实，则禁止把背景事实直接写成主回答

##### 12.3.66.2 `t18` 的最小中间对象建议

当前阶段不应再新发明复杂对象，只需要在已有 `AnswerFocusUnit` 之上补一个很窄的 gate：

1. `QuestionCoreCoverage`
   - 建议字段：
     - `question_core_slots`
     - `covered_slots`
     - `background_only_claim_ids`
     - `answerable_slots`
     - `missing_slots`
     - `coverage_decision`

关键约束：

- 它不新增 claim
- 不新增 relation
- 只负责判断：
  - 当前主回答是否覆盖问题核心
  - 还是只是拿背景事实在“假回答”

##### 12.3.66.3 `t18` 的执行顺序

为避免再次发生 `t17` 这种“回答压缩带坏 evidence”的问题，`t18` 的最小执行顺序建议固定为：

1. 先回滚或隔离 `t17` 中会影响 final pack 的 answer focus 改动
2. 恢复 `t16` 级别的 final evidence retention 行为
3. 在此基础上新增 `QuestionCoreCoverage`
4. 让 `AnswerFocusUnit` 只负责：
   - main claim selection
   - supporting claim selection
   - `response_for_eval` 裁剪
5. 最后再观察：
   - `answer_correctness` 是否回升
   - 同时 `context_precision / final_chunk_recall` 不再回退

##### 12.3.66.4 `t18` 的主回归样本

`t18` 的样本优先级应调整为：

1. `0042`
   - 验证：
     - relation composer 的收益必须恢复
     - final evidence recall 不得再被压缩误伤
2. `0041`
   - 验证：
     - 主回答不再被不贴题的 confirmed facts 占据
     - 但 external facts 与 target facts 的边界仍然守住
3. `0022`
   - 验证：
     - 主回答必须围绕问题核心 service-level trigger
     - supporting facts 不得再挤占主结论
4. `0021`
   - 验证：
     - 背景事实不能替代 termination impact 的主回答
5. `0002`
   - 继续守住：
     - 单跳 minimal answer 不回退
6. `0043`
   - 继续守住：
     - ownership 与 target-grounded explanation 不回退

##### 12.3.66.5 `t18` 的完成标准

`t18` 是否成功，不应只看 `answer_correctness` 是否回升，还必须同时满足：

1. `context_precision`
   - 不明显低于 `t16`
2. `final_chunk_recall`
   - 不明显低于 `t16`
3. `0042 / 0041 / 0022 / 0021`
   - 主回答比 `t17` 更贴题
   - 但不再通过压缩 final evidence package 来换取“更短”
4. `0002 / 0043`
   - 仍保持：
     - minimal answer
     - ownership 正确

##### 12.3.66.6 当前结论

因此，`t18` 不应再被理解成：

- 下一轮继续做更激进的 focus 压缩

更准确的定位应是：

- 先恢复 `t16` 的 evidence-control 优势
- 再把 answer focus 收窄成真正的“主回答排序层”

只有在这一步完成后，系统才可能真正实现：

1. 检索更准
2. 证据更稳
3. 回答也更贴题

而不是再次落回：

- 为了让答案更短
- 把真正需要的证据和解释一起削掉

的回退路径。

##### 12.3.66.7 `t18` 迭代复盘（结合 `retrieval_requirements_smoke_eval_t18` 与逐样本结果）

本节基于代码侧已落地的 `t18` 改动与一次完整 smoke 复跑产物，对「计划目标是否达成」与「检索维度是否已收敛」作事实核对；结论用于衔接 `12.3.67` 的 cross-encoder rerank 规划。

###### 12.3.66.7.1 实施侧概要（`youtu-graphrag/backend.py`）

本轮 `t18` 在后端已落实的方向包括（与 `12.3.66` 文字计划对齐）：

1. **解耦 answer focus 与证据集**：`_apply_t17_focus_to_answer_decision` 不再用 `selected_claim_ids` 覆盖 `confirmed_claim_ids`，改为写入 `focus_selected_claim_ids`，避免 focus 层裁剪 claim 集进而牵连 final evidence。
2. **`_render_t15_eval_response` / 评测通道**：`eval_response_preview` 重新以 `answer_decision.confirmed_claim_ids` 为主构造；`evaluation_payload.response_for_eval` 在 `t15_experimental` 开启且预览非空时，优先采用 `eval_response_preview`，与「Final Answer:」表面文本解耦。
3. **Synthesis Contract 与 strict rewrite 提示**：将「仅允许 selected 进入主答」改为「focus 优先排序」，减轻过度压缩。
4. **`QuestionCoreCoverage`（`_build_t18_question_core_coverage`）**：输出写入 `t15_experimental.question_core_coverage` / trace，作诊断 gate，不替代检索。

###### 12.3.66.7.2 汇总指标对照（`method = youtu_graph_rag`，`retrieval_requirements_smoke` 9 条）

| 指标 | t16 | t17 | t18 |
|------|-----|-----|-----|
| `answer_correctness` | 0.439377 | 0.399041 | **0.468027** |
| `faithfulness` | 0.638480 | 0.491358 | **0.791358** |
| `context_precision` | **0.702778** | 0.525926 | 0.550000 |
| `context_recall` | **0.756085** | 0.740212 | 0.740212 |
| `final_chunk_precision` | **0.244092** | 0.190697 | 0.240741 |
| `final_chunk_recall` | **0.851852** | 0.722222 | 0.777778 |

可读结论：

1. **相对 `t17`**：`t18` 在 `answer_correctness`、`faithfulness`、检索相关多项上均有回升，`t17` 的「过度压缩」路线得到部分纠正。
2. **相对 `t16`（计划中的「证据优势」锚点）**：`context_precision`、`context_recall`、`final_chunk_recall` **仍未回到 `t16` 水平**；`12.3.66.5` 中「不明显低于 `t16`」的检索向完成标准 **未完全达成**。
3. **`faithfulness` 的大幅上升**与「`response_for_eval` 改为证据贴地的 `eval_response_preview`」强相关：评测器更容易在 `retrieved_contexts` 中找到字面支撑，**不能单独等同于**端到端生成链路在「自然语言 Final Answer」意义上的忠实度质变，需在论文或报告中区分 **eval 信号** 与 **产品回答**。

###### 12.3.66.7.3 逐样本：检索维度**不能**判定为「已没问题」

依据 `outputs/results/ragas/retrieval_requirements_smoke_eval_t18/ragas_eval_per_sample.jsonl`（`youtu_graph_rag`）：

- **`doc_hit` / `doc_purity`**：9/9 为 `1.0`，文档级目标合同与检索文档集合一致，**文档定位不是当前短板**。
- **`context_precision < 0.5`**：**5/9**（`0002`、`0003`、`0023`、`0041`、`0042`）——最终进入评测上下文的 chunk 中，与参考相关的比例仍偏低，**噪声问题仍在**。
- **`context_recall < 0.5`**：**1/9**（`0023`）——参考上下文整体覆盖不足。
- **`final_chunk_recall < 1.0`**：**5/9**（`0021`、`0022`、`0023`、`0041`、`0043`）——最终用于评测的 chunk id 集合未收齐 gold，**chunk 级召回仍不稳定**。

典型样本：

- **`0041`**：`context_precision = 0`、`final_chunk_recall = 0.5`——检索上下文与参考对齐差，且 final 只覆盖一半参考 chunk；与历史文档中「子问题编译 / scope」类问题仍一致，**非仅靠回答层可解**。
- **`0023`**：`context_precision`、`context_recall`、`final_chunk_recall` 均偏弱——多跳抽象场景下问题最突出。
- **`0002`**：`context_recall`、`final_chunk_recall` 高，但 `context_precision` 低——**候选池内有 gold，但无关 chunk 占比高**，符合「需要更强精排 / 截断策略」的形态。

因此：**从检索维度评价，`t18` 不能表述为「已经没问题」**；汇总均值（约 `context_precision ≈ 0.55`、`final_chunk_recall ≈ 0.78`）反映的正是「约半数样本在 chunk 级仍明显拖后腿」。

###### 12.3.66.7.4 与历史轮次（如 `t3`/`t4`）对比时的口径提醒

早期部分评测产物中 **`final_chunk_precision` / `final_chunk_recall` 曾为 `0`**（诊断字段未贯通或口径不同），与 `t5` 之后「chunk id 对齐」下的数字 **不可直接横向比较**。复盘 `t18` 时，应以 **同字段、同 `per_sample` 结构** 为约束，避免用早期 `faithfulness` 等指标单独论证「越迭代越差」。

###### 12.3.66.7.5 运行方差（IRCoT 子问题轮次）

同一 `qid` 在不同次运行中，`retrieval_trace` 内 **子问题数量 / `need_more` 终止点** 可能因 LLM 随机性变化，导致 `final_chunk_ids` 与汇总检索指标波动。**小样本 smoke 上不宜将单次 `t18` 与单次 `t16` 的细微差距全部归因于代码 diff**；重要对比应配合 **固定温度/种子** 或 **多次中位数**，见 `12.3.67.5`。

###### 12.3.66.7.6 小结与下阶段接口

- **`t18` 的价值**：纠正 `t17` 的「focus 牵连证据集」、缓解评测通道与 ledger 脱节；答案与诊断对象更清晰。
- **`t18` 未闭合的主线**：**chunk 级 precision / recall 仍明显弱于 `t16` 锚点，逐样本问题仍在**——下一阶段应把主要工程火力放在 **检索管线中的「候选已存在时的排序与最终选入」**（与 `12.3.67` cross-encoder rerank 对齐），而非继续堆叠结论层规则。

---

#### 12.3.67 Cross-encoder rerank 实施规划（`t19` 候选）

本节承接 `12.3.66.7` 的结论：在 **文档级已稳定**、`doc_hit`/`doc_purity` 无系统性失败的前提下，**chunk 级精度不足、final 选入噪声大** 是当前 smoke 的主矛盾之一；在候选集合往往已包含 gold、但未顶到 final / support top-k 的场景下，**cross-encoder（CE）类逐 query–chunk 重排** 是优先工程选项之一。

本规划为实施蓝图，**不改变**前文对「query 编译失真 / 候选池缺失」类问题的判断：CE **只重排已有候选**，不替代前段召回与 query 质量治理。

##### 12.3.67.1 目标与非目标

**目标**

1. 在 **固定候选规模**（例如 strong rerank 之后 top-K）上，提高 **query–chunk 相关性排序质量**，使 gold chunk 更常进入 **support / final** 截断窗口。
2. 在 `retrieval_requirements_smoke` 上可观测地改善：
   - `context_precision`（减少无关 chunk 进入最终上下文）
   - `final_chunk_recall` / `support_chunk_recall`（在 gold 已进入候选池的样本上）
3. 与 `t16` 或当前基线的 **同口径** `ragas_eval_per_sample.jsonl` 对比，验证 **检索向指标** 相对提升。

**非目标（明确不在本轮 CE 内解决）**

1. 替代 **embedding 召回** 或扩大候选池——若 gold 从未进入候选，CE 无效。
2. 替代 **子问题分解 / query 编译** 修正——见 `0041` 类前段失真。
3. 以 CE 单独拉升 **`answer_correctness`** 作为唯一验收标准（仍受合成与评测字段影响）。

##### 12.3.67.2 接入位置与数据流（建议）

在 `youtu-graphrag` 检索链路中，于 **现有 lightweight / strong rerank 之后**、**写入 `support_chunk_ids` / `final_selected_chunk_ids` / 合并多路 chunk 之前或之后（需读代码定锚点）** 增加可选阶段：

1. 输入：`query`（或子问题文本）、候选 `chunk_id` 列表（已截断至 top-K，例如 32～64）、chunk 文本。
2. CE 打分：对每个 `(query, chunk_text)` 输出标量分数。
3. 重排序：按分数降序排列，再交现有 **budget / per_subquestion_min / answer_scope** 逻辑截取。

**原则**：CE 只对 **小批量候选** 调用，避免全库二次扫描；延迟与成本可控。

##### 12.3.67.3 模型与工程选型（初稿）

| 维度 | 建议 |
|------|------|
| 实现形态 | HuggingFace `CrossEncoder`（如 `cross-encoder/ms-marco-MiniLM-L-6-v2`）或等价 ONNX；与现有 embedding 模型解耦。 |
| 域适配 | 先用通用 MS MARCO 类模型做 smoke；若提升有限，再考虑 **CUAD/合同语料** 上对 CE 做轻量微调（独立实验项）。 |
| 开关 | 配置项如 `enable_cross_encoder_rerank: bool`、`ce_rerank_top_k_in`、`ce_rerank_top_k_out`，默认 `false` 直至 smoke 验收通过。 |
| 遥测 | 在 `retrieval_trace` 或现有 rerank diagnostics 中记录：`ce_model_id`、`ce_scores_summary`、`ce_rerank_latency_ms`。 |

##### 12.3.67.4 验收指标与回归集

1. **主表**：`outputs/results/ragas/retrieval_requirements_smoke_eval_<tag>/ragas_eval_summary.csv` 中 `youtu_graph_rag` 行。
2. **必看列**：`context_precision`、`context_recall`、`final_chunk_precision`、`final_chunk_recall`；并与 **`t16` / `t18` 冻结基线** 对照（同评测协议、同 `response_for_eval` 规则）。
3. **逐样本**：`ragas_eval_per_sample.jsonl` 中重点复查 `12.3.66.7.3` 所列弱样本：`0041`、`0023`、`0002`、`0042`。
4. **守门**：`doc_hit`/`doc_purity` 不得系统性回退；若 CE 仅提升 precision 但伤害 recall，需调整 `top_k_in/out` 或与原分数融合（见 `12.3.67.6`）。

##### 12.3.67.5 可复现性与方差控制

1. 评测跑：**固定** LLM `temperature`、记录 **seed**（若后端支持）；同一批 smoke **至少保留两次运行**或报告均值±方差。
2. 对比 `t18` 与 CE 版本时，**锁版本**：同数据集、同 `client_id`、同 `route_type`（若适用）、同 `max_evidence` 等。

##### 12.3.67.6 风险与缓解

| 风险 | 缓解 |
|------|------|
| 延迟与吞吐 | 仅对 top-K 候选打分；批量 encode；可选异步或缓存热 query。 |
| 域外 CE 乱序合同细粒度条款 | smoke 验证；不行则换 larger CE 或域内微调。 |
| 与原 rerank 分数冲突 | **融合策略**：`final_score = α * ce + (1-α) * strong_rank` 或级联（strong 先剪枝，CE 再精排）。 |
| gold 不在候选池 | 前段仍须靠 query 编译与召回改进；CE 不背锅。 |

##### 12.3.67.7 实施阶段（建议顺序）

1. **P0**：在 `enhanced_kt_retriever.py` / `backend.py` 中定位 strong rerank 输出点，插入 **no-op** 开关与单元测试（候选顺序不变时行为与现网一致）。
2. **P1**：接入 CE，默认关闭；本地单测 1～2 个 query，检查 trace 字段。
3. **P2**：跑全量 `retrieval_requirements_smoke`，产出 `*_eval_t19-ce`（或约定 tag），填 `12.3.67.4` 对比表。
4. **P3**：若检索向指标显著优于 `t18` 且未伤 `doc_*`，默认开启或按环境开启；文档更新本节「落地版本与 commit 指针」。

##### 12.3.67.8 与文档其它章节的关系

- 前文 **§4**（弱 query、doc 推断、merge 丢序、rerank bug）仍成立；CE 解决的是 **「候选已有但排不好」** 的子问题。
- **`0041` 类 scope/support 联动** 若仍失败，需并行跟踪 **子问题 scope 解析**，见 `12.3.60.7` 等节，**不与 CE 互相替代**。

### 12.3.68 `t20` 迭代：子问题拆解增强 + 子问题上下文上限 + **策划层**后续规划（`retrieval_requirements_smoke_eval_t20`）

本节记录 **`t20` 相对 `t18`/`t19` 在「拆解」与「子问题可见上下文」上的增量**，并单独给出 **`t20` 阶段对「策划层」的重构规划**（见 **§12.3.68.6**）。用于 smoke 复跑与论文/实验对照。评测输出目录约定：`outputs/results/ragas/retrieval_requirements_smoke_eval_t20/`（或同协议命名的 `*_t20.jsonl` / `*_t20_v*`）。

#### 12.3.68.1 动机（承接前文分层诊断）

在 `t18` 逐样本归因中已明确两类与 **CE 无关**、但会拉低 `support_chunk_*` / `final_chunk_*` 的机制性问题：

1. **子问题 LLM 可见 chunk 过少**：首轮子回答构造 prompt 时，若只把强重排列表的**前极少个** chunk 拼进 `Evidence chunks`，则 gold 落在第 10～11 位时，模型**从未看到**该 chunk，后续 grounding 再严也无法引用（非「模型不认真」，而是**上下文被截断**）。
2. **多 aspect 题干被压成单子问题**：例如题干同时要求 *insurance / food safety / third-party audits*，分解 LLM 若只产出一条子问题，则检索 query 偏向其中一个 aspect，其它 aspect 对应的 gold chunk 在排序上被压低或根本不在该子问题的 support 集合里。

`t20` 分别从 **配置** 与 **代码兜底** 两侧缓解上述问题。

#### 12.3.68.2 子问题上下文上限：`retrieval.agent.subquestion_ircot_context_limit`

- **配置位置**：`youtu-graphrag/config/base_config.yaml` → `retrieval.agent.subquestion_ircot_context_limit`（`t20` 起由默认 `6` **上调**，仓库中可按实验取 **12～20** 之间的值；当前基线以 **15** 为折中：覆盖「ref 在 strong 第 11 位」类样本，同时控制 token 成本）。
- **已接入代码的路径**（读配置、非硬编码 6）：
  - **Section-anchor rescue** 路径里 rescue prompt 使用的 `rescue_context_limit`；
  - **IRCoT follow-up** 循环里 `loop_ctx` 与二次 `generate_answer` 使用的 `followup_context_limit`。
- **待对齐项（文档诚实记录）**：首轮子回答使用的 `_build_sub_answer_prompt` 内，若仍存在 **`chunk_contents[:6]` 硬编码**，则与上述配置**不一致**；验收 `t20` 时应以代码为准确认是否已改为 `[:subquestion_ircot_context_limit]`。**建议**将主路径与 rescue/IRCoT 使用**同一上限**，避免「配置写了 15～20，首轮子问题仍只吃 6 条」的隐性偏差。

#### 12.3.68.3 子问题拆解增强：多 aspect 显式枚举兜底（`multi_aspect_expansion`）

在 `youtu-graphrag/backend.py` 中，`graphq.decompose()` 产出的 `retrieval_requirements` 经 `_requirements_to_sub_questions` 编译为子问题列表后，追加 **`_expand_sub_questions_for_aspects`**：

| 项 | 说明 |
|----|------|
| **入口** | `_requirements_to_sub_questions` 末尾，对 `compiled` 列表就地扩展 |
| **检测** | `_extract_multi_aspect_terms(question)`：用正则从**原问题**中识别显式枚举（如 `requirements for A, B, and C`、`A and B` 等模式） |
| **触发条件** | 解析出 **≥2 个 aspect**；且对每个 aspect 做**粗粒度覆盖检查**（aspect 中较长 token 是否已出现在已有子问题/检索 query 拼接文本中） |
| **补齐策略** | 对**未被覆盖**的 aspect，追加 `route_type: local` 子问题，文案形如「What clause states the requirements for {aspect}?」，并带 `retrieval_queries` / `retrieval_requirement`（`intent: fact_lookup` 等） |
| **上限** | 单次最多追加 **3** 条子问题，防止拆解爆炸 |
| **trace 语义** | 新增子问题标记 `decomposition_mode: multi_aspect_expansion`，`route_reason` 含 `multi_aspect_expansion: <aspect>` |

**定位**：这是对 **LLM 分解欠覆盖** 的**代码层安全网**，不替代 `decomposition.mode: retrieval_requirements` 与上游 schema；与 `12.3.67` 中 CE 路线 **正交**——CE 解决「候选已有但排序差」，本节解决「拆解粒度不足」与「子问题 prompt 可见 chunk 不足」。

#### 12.3.68.4 验收与对照建议

1. **同协议**重跑 `retrieval_requirements_smoke`，产出 `*_t20` 预测与 `ragas_eval_*_t20`。
2. **优先看列**：`support_chunk_recall`、`support_chunk_precision`、`final_chunk_recall`；并对照 `t18` 中曾出现 **strong_recall≈1 而 support_recall 掉** 的样本（如多 aspect、`ref` 在 strong 列表 8～11 位附近）。
3. **回归注意**：子问题条数增加会带来 **延迟与 LLM 调用次数** 上升；若仅测检索质量，可并行观察 `telemetry.llm_calls` 与单题耗时。

#### 12.3.68.5 与 `decomposition` 段 YAML 的关系

`t20` **不依赖**调整 `decomposition.mode` / `enable_query_compilation` / `max_query_variants`（见 `base_config.yaml` 中 `decomposition:` 段）：这些参数仍分别控制「分解模式」「是否编译 requirement 查询」「**每个子问题**检索 query 变体条数上限」。`t20` 的 multi-aspect 兜底是在 **子问题条数** 层面补洞，与 `max_query_variants` 含义不同，**无需为 `t20` 强行改大 `max_query_variants`**，除非独立实验需要更多检索变体。

#### 12.3.68.6 **`t20` 迭代计划：策划层（Planning Layer）重构规划**

本节回答：**是否必须整链路重构、是否必须新增 `orchestration.mode`**，并给出一条 **范围可控、以拆解为先** 的策划层演进路线。

##### 12.3.68.6.1 问题收敛（与 smoke 归因一致）

当前 smoke 中大量失败**不必**先归因于「检索器弱、聚合弱、七层全坏」。更可操作的**主因**是：

- **子问题集合对原问题的检索需求覆盖不足**：多面 / 多跳抽象题常被压成 **单条泛化子问**，导致后续检索 query 偏一面，gold chunk 不在该子问题的候选与 support 路径上（见 `t18`/`t20` 逐题 trace：`strong_recall` 尚可而 **拆解条数=1**、多 ref 覆盖不足）。

**策划层**在此文档中指：在**进入逐子问题检索循环之前**，显式建立「原问题 → 检索任务集合」的映射，并保证该集合**在结构上能覆盖**题干中的独立检索意图（多 aspect、多条款、桥接两侧等），而不是事后一题题打补丁。

##### 12.3.68.6.2 重构范围：先策划层，而非整站重写

| 纳入 `t20` 策划层范围 | 暂不纳入（除非后续独立立项） |
|------------------------|------------------------------|
| 分解结果的**结构化表示**（证据面 / 子意图 / 与子问题的一对多关系） | 全文「证据集合覆盖」数学最优化、跨模块重写合并器 |
| **覆盖校验**（启发式或轻量 LLM）：是否每个显式 aspect / requirement 都有对应子问题或检索任务 | 新 rerank 族、CE 默认开启 |
| 与现有 `retrieval_requirements` + `_expand_sub_questions_for_aspects` **平滑演进**（见下节） | 为策划层单独新增 `orchestration.mode`（**默认不需要**，见 §12.3.68.6.5） |

##### 12.3.68.6.3 与现有实现的关系（`t20` 已落地的 §12.3.68.2–12.3.68.3）

- **`subquestion_ircot_context_limit` + 主路径 prompt 对齐**：降低「gold 已在 strong 列表较后位置但子答案未看见」的截断问题，与策划层**正交**。
- **`multi_aspect_expansion`**：属于策划层的 **0.x 安全网**——用规则从题干枚举 aspect 并补子问，**不替代**完整策划层，但证明「补子问条数」对多面题有效。

**策划层后续工作**是在此之上：把「aspect / 检索意图」从**仅正则兜底**提升为**与 `graphq.decompose()` / `_requirements_to_sub_questions` 同一数据契约**中的**一等公民**（例如 `compiled` 中显式 `facets[]` 或 `retrieval_slots[]`），并在进入检索前做 **覆盖检查**。

##### 12.3.68.6.4 策划层交付形态（建议分阶段）

1. **契约**：在分解输出中增加可选字段（名称待定），例如每个 slot：`id`、`natural_language`、`linked_sub_question_id` 或 `pending`；便于 trace 与日志验收。
2. **覆盖校验**：分解完成后执行 `validate_coverage(original_question, compiled_slots, sub_questions)`；未覆盖则 **仅扩子问题或补 slot**（可复用 `_expand_sub_questions_for_aspects` 的逻辑，升级为「slot 驱动」而非仅正则）。
3. **可观测**：在 `retrieval_trace`（或 `answer_trace`）中写入 **策划层快照**：`planning_facets` / `coverage_status` / `expansion_reason`，避免 `chunk_stage_trace` 长期为空时无法做策略级复盘。
4. **聚合与预算**：仅在策划层证明能稳定增加「子问—检索需求」覆盖后，再评估 **final backing** 是否需按 slot 预留配额（可作为 `t21+` 议题，**不阻塞**策划层第一期）。

##### 12.3.68.6.5 是否新增 `orchestration.mode`？

**默认结论：第一期不新增。** 优先在现有 `decomposition.mode: retrieval_requirements` 与 `orchestration.mode: t15_experimental`（或当前实验路径）下 **增强分解输出与校验**，避免维护多套编排分叉。

**仅当**出现以下情况时，再考虑新增 mode（例如 `planning_layer_v1`）或独立 feature flag：

- 新逻辑与现网行为差异大、需要 **长期 A/B** 或一键回滚；
- 配置组合爆炸，用单一 `enable_planning_layer: true` 更清晰。

##### 12.3.68.6.6 验收标准（策划层第一期）

| 维度 | 标准 |
|------|------|
| **Smoke** | `retrieval_requirements_smoke` 中 **multi-hop abstract** 子类：`reasoning_steps` 内子问题条数与题干显式多面 **可对照**；`support_chunk_recall` / `final_chunk_recall` 相对仅 `multi_aspect` 正则兜底有 **可报告的提升或稳定不减**（同协议、同 eval）。 |
| **回归** | 单跳、简单题子问题条数 **不无故膨胀**；`telemetry.llm_calls` 与延迟在可接受范围或配置上限内。 |
| **文档** | 本节随实现更新「落地版本 / 关键字段 / 配置开关」。 |

##### 12.3.68.6.7 **具体优化方法（可实施清单）**

以下与 §12.3.68.6.4 的分阶段一一对应，给出**可写进代码/配置**的做法，避免只有方向没有手段。

###### （1）结构化契约：`retrieval_slots` / `planning_facets` 建议形态

在现有 `compiled` 子问题列表之外，增加**与检索一一对应**的策划数组（字段名可二选一，实现时固定一种）：

| 字段 | 类型 | 含义 |
|------|------|------|
| `slot_id` | string | 稳定 id，如 `sq_1_facet_insurance` |
| `source` | enum | `from_decomposer` \| `regex_aspect` \| `expansion` \| `manual_rule` |
| `intent_nl` | string | 该 slot 的自然语言检索意图（**短句**，便于与 `sub_question` 文本对齐） |
| `must_cover_keywords` | string[] | 可选；从题干切出的 **非停用词** token，用于覆盖校验 |
| `linked_sub_question_index` | int \| null | 指向 `compiled` 数组下标；`null` 表示**尚未绑定**子问题 |
| `status` | enum | `pending` \| `covered` \| `expanded` |

**生成方式（按优先级组合，而非单选）**：

1. **解析 `graphq.decompose()` 的 `retrieval_requirements`**：将每条 requirement 映射为 0～1 个 slot（`source=from_decomposer`），避免与现有 schema 打架。
2. **保留并升级 `_extract_multi_aspect_terms` + `_expand_sub_questions_for_aspects`**：命中时写入 `source=regex_aspect`，子问题追加后回填 `linked_sub_question_index`。
3. **可选的一轮轻量补全**：仅当 slot 数仍为 0 且题干长度/连接词满足启发式时，调用 **小型 JSON-only prompt**（见下「分解补全 prompt」），产出 2～5 个 `intent_nl`（`source=from_decomposer` 的补充），**禁止**直接生成最终答案。

###### （2）覆盖校验 `validate_coverage`：具体算法（可无额外 LLM）

对**每个** slot，判定「是否已有子问题承载该检索意图」，建议**默认纯规则**，控制成本：

1. **拼接文本**：`S = 所有 sub_question 的文本 + 各子问题 retrieval_queries 拼接 + 原问题前 200 字`（小写、去多余空白）。
2. **关键词命中**：对 `must_cover_keywords` 中长度 ≥3 的 token，若任一在 `S` 中出现，记该 keyword **已覆盖**。
3. **intent 重叠（无向量时）**：将 `intent_nl` 拆成词组，计算与 `S` 的 **最长公共子串长度 / intent 长度** ≥ 某阈值（如 0.35）则 slot 记 **covered**。
4. **可选加强（仍不生成答案）**：仅对仍 `pending` 的 slot，用 **embedding 相似度**（已有 shared encoder）比较 `intent_nl` 与每条子问文本，若 `max_sim < τ`（如 0.45），则判 **未覆盖**。

输出结构：`coverage_report: { slot_id, status, reason }[]`，供 trace 与单测断言。

###### （3）未覆盖时的**补齐策略**（与一题一补丁的区别）

| 步骤 | 做法 |
|------|------|
| A | 对未覆盖 slot，用**固定模板**生成一条 `local` 子问题，例如：`According to the target agreement, what does it state regarding: {intent_nl}?`（语言与现有 pipeline 一致） |
| B | 为该子问题生成 **1～2 条 `retrieval_queries`**：优先复用 `intent_nl` + 题干中的协议名/当事人名（若有） |
| C | **去重**：新子问与已有子问 embedding 余弦相似度 > 0.92 则合并 slot 链接而不新增子问 |
| D | **全局上限**：与现有 `multi_aspect` 一致，单次追问最多追加 **N** 条（如 3～5），由配置 `retrieval.agent.planning_max_expansion_subquestions` 控制 |

**禁止**：针对某一 gold chunk id 写 if 分支；允许：针对 **slot 类型**（aspect / clause_bridge / definition）使用不同模板（仍属规则族，非逐题补丁）。

###### （4）分解侧：可选的 **JSON-only 补全 prompt**（仅填 slot，不答题）

当规则抽不出 aspect、且 requirement 条数 < 2 时，可插入一次调用，**输入**为 `original_question` + `target_doc_hint`（若有），**输出**严格 JSON：

```json
{
  "retrieval_slots": [
    { "intent_nl": "...", "must_cover_keywords": ["..."] }
  ]
}
```

**系统约束示例（写入 prompt）**：「只输出独立检索子意图，每条一行意图；不得输出答案句；条数 2～5；若问题明显单跳则只输出 1 条。」

该步骤与现有 `graphq.decompose()` **串联或二选一**由配置 `retrieval.decomposition.enable_planning_slot_fill` 控制，默认 **false**，便于灰度。

###### （5）代码挂载点（与仓库现状对齐）

| 挂载点 | 动作 |
|--------|------|
| `youtu-graphrag/backend.py` | 在 `_requirements_to_sub_questions` 返回前、`_expand_sub_questions_for_aspects` 之后，调用 `build_planning_slots(...)` → `validate_coverage(...)` → 必要时 `expand_subquestions_for_slots(...)` |
| `graphq` / decomposer | 若希望 slot **主要来自** LLM：在 `retrieval_requirements` schema 中增加可选字段，由 `decompose()` 一并解析（减少二次 LLM） |
| `retrieval_trace` | 写入 `planning_layer: { slots: [...], coverage_report: [...], expansion_applied: bool }`，与 §12.3.68.6.4 一致 |

###### （6）配置项建议（`base_config.yaml` 草案）

| 键（建议路径） | 含义 | 建议默认 |
|----------------|------|----------|
| `retrieval.agent.planning_layer_enabled` | 是否启用策划层校验与扩容 | `false` → 验证后 `true` |
| `retrieval.agent.planning_max_expansion_subquestions` | 单次最多追加子问题数 | `5` |
| `retrieval.agent.planning_coverage_token_min_len` | 参与覆盖的 keyword 最小长度 | `3` |
| `retrieval.agent.planning_intent_overlap_threshold` | 意图与拼接文本重叠阈值 | `0.35` |
| `retrieval.decomposition.enable_planning_slot_fill` | 是否启用「补全 slot」的额外 JSON-only LLM | `false` |

###### （7）测试与回归（与实现绑定）

- **单元测试**：构造「原问题含 A/B/C、分解只产出 1 条子问」的 fixture，断言 `validate_coverage` 输出 `pending`，扩容后 `compiled` 长度增加且每个 slot `linked_sub_question_index` 非空。
- **黄金 smoke**：现有 `retrieval_requirements_smoke` 中 multi-hop abstract 题号固定时，对比 `planning_layer_enabled` 开/关的 `support_chunk_recall` 与子问题条数。

---

**小结**：`t20` 在文档中同时承载 **已落地的上下文与 multi-aspect 兜底**，以及 **下一阶段的策划层规划**——**主矛盾是「拆解对原问题检索需求的覆盖」**；§12.3.68.6.7 补充了 **契约字段、覆盖校验算法、扩容模板、可选 LLM 补全、挂载点与配置键**，便于直接落地；**不必**先整链路重构，**不必**默认新增 orchestration mode。

### 12.4 全量 `vector_test` 对照复盘：`refactor_v1` 相比 `vector_rag` 的退化原因与优化方案

本节基于最新全量结果目录：

- `outputs/results/ragas/vector_test/ragas_eval_summary.json`
- `outputs/results/ragas/vector_test/ragas_eval_summary.csv`
- `outputs/results/ragas/vector_test/ragas_eval_per_sample.jsonl`

该目录共包含 `120` 条评测记录，其中 `vector_rag` 与 `youtu_graph_rag/refactor_v1` 各 `60` 条，题目一一配对。需要先明确一个结论：**本轮结果不能简单解释为“图增强方法整体比普通向量检索差”。更准确的判断是：`refactor_v1` 前半段召回能力已有明显效果，但后半段 support/final evidence selection 过度收缩，导致 gold evidence 在最终上下文中流失；同时最终答案的评测版本过长、结构化痕迹过重，被 RAGAS 的 answer correctness 和 faithfulness 进一步惩罚。**

#### 12.4.1 全量指标对照

按 `ragas_eval_summary.csv` 的 method 级汇总：

| 指标 | `vector_rag` | `youtu_graph_rag/refactor_v1` | 变化 | 解释 |
|------|--------------|-------------------------------|------|------|
| `answer_correctness` | `0.579996` | `0.512733` | `-0.067263` | 最终答案质量低于 baseline |
| `answer_relevancy` | `0.591844` | `0.714076` | `+0.122232` | 回答更贴近问题表述 |
| `faithfulness` | `0.832790` | `0.612020` | `-0.220770` | 最终回答中存在更多未被最终上下文严格支持的内容 |
| `semantic_similarity` | `0.741626` | `0.797439` | `+0.055813` | 语义接近度提升 |
| `context_precision` | `0.373091` | `0.464491` | `+0.091400` | 最终上下文更“精” |
| `context_recall` | `0.708690` | `0.527698` | `-0.180992` | gold evidence 覆盖不足 |
| `context_mrr` | `0.554623` | `0.633333` | `+0.078710` | 命中时排序位置更靠前 |
| `doc_hit` | `1.000000` | `0.966667` | `-0.033333` | 文档级定位基本稳定，但有少量回退 |
| `final_chunk_precision` | `0.162500` | `0.266336` | `+0.103836` | 最终 chunk 噪声更少 |
| `final_chunk_recall` | `0.700000` | `0.536111` | `-0.163889` | 最终 chunk 漏召回明显 |

配对逐题差值进一步说明：

- `answer_correctness`：`youtu_graph_rag` 赢 `22` 题、输 `38` 题，平均 `-0.0673`
- `faithfulness`：赢 `11` 题、平 `8` 题、输 `41` 题，平均 `-0.2208`
- `context_precision`：赢 `25` 题、平 `22` 题、输 `13` 题，平均 `+0.0914`
- `context_recall`：赢 `6` 题、平 `33` 题、输 `21` 题，平均 `-0.1810`
- `final_chunk_precision`：赢 `39` 题、平 `4` 题、输 `17` 题，平均 `+0.1038`
- `final_chunk_recall`：赢 `9` 题、平 `30` 题、输 `21` 题，平均 `-0.1639`

因此，这轮退化的核心不是“完全检不到”，而是 **precision/relevancy 提升以 recall/faithfulness 为代价**。

#### 12.4.2 按题型定位：`multi_hop_abstract` 是最大失血点

按 synthesizer 分组的差值如下：

| 题型 | `answer_correctness` 差值 | `faithfulness` 差值 | `context_recall` 差值 | `final_chunk_recall` 差值 | 结论 |
|------|---------------------------|---------------------|-----------------------|---------------------------|------|
| `multi_hop_abstract_query_synthesizer` | `-0.1338` | `-0.1706` | `-0.3205` | `-0.3667` | 退化最严重，主要是最终 evidence 覆盖不足 |
| `multi_hop_specific_query_synthesizer` | `+0.0332` | `-0.2153` | `-0.0725` | `+0.0250` | 答案正确性略升，但 faithfulness 明显下降 |
| `single_hop_specific_query_synthesizer` | `-0.1012` | `-0.2764` | `-0.1500` | `-0.1500` | 简单事实题被过度包装与证据选择误伤 |

这说明 `refactor_v1` 的主要短板并不只在复杂多跳题。单跳题也会因为答案过度展开、支持证据选择过窄或 `NOT_FOUND` 误杀而输给简单 `vector_rag`。

#### 12.4.3 阶段指标揭示：gold evidence 在后半段被筛掉

`youtu_graph_rag/refactor_v1` 的诊断指标显示：

| 阶段 | hit | precision | recall | mrr |
|------|-----|-----------|--------|-----|
| `first_stage_chunk_*` | `0.983333` | `0.022831` | `0.911111` | `0.556532` |
| `lightweight_reranked_chunk_*` | `0.983333` | `0.023079` | `0.902778` | `0.558213` |
| `strong_reranked_chunk_*` | `0.933333` | `0.038689` | `0.875000` | `0.657608` |
| `support_chunk_*` | `0.800000` | `0.263757` | `0.611111` | `0.640436` |
| `final_selected_chunk_*` | `0.783333` | `0.135975` | `0.613889` | `0.636296` |
| `final_chunk_*` | `0.733333` | `0.266336` | `0.536111` | `0.633333` |

关键观察：

1. `first_stage` 与 `strong_reranked` 阶段的 recall 已经较高，说明初检索和重排经常能把 gold evidence 找进候选集。
2. 从 `strong_reranked_chunk_recall = 0.875000` 到 `support_chunk_recall = 0.611111`，再到 `final_chunk_recall = 0.536111`，存在明显后段流失。
3. `final_chunk_precision` 高于 baseline，但 `final_chunk_recall` 低于 baseline，说明当前 final selection 更像“过度压缩的精排器”，不是“面向答案充分性的证据选择器”。

逐样本统计中，`60` 个 `youtu_graph_rag` 样本里：

- `first_stage_chunk_hit = 1` 但 `final_chunk_hit = 0` 的样本有 `15` 个
- `strong_reranked_chunk_hit = 1` 但 `final_chunk_hit = 0` 的样本有 `12` 个
- `support_chunk_hit = 1` 但 `final_chunk_hit = 0` 的样本有 `4` 个

这直接证明：**当前最大问题不是候选生成失败，而是 support/final 选择没有 recall guard。**

#### 12.4.4 典型失败样本

| qid | 现象 | 诊断 |
|-----|------|------|
| `ragas-cuad-0008` | `first_stage`、`lightweight`、`strong` 均命中 gold，`support` 与 `final` 变为 `0`，最终回答 `NOT_FOUND` | 典型的后段 evidence selection 误删；不是初检索失败 |
| `ragas-cuad-0038` | `first_stage/strong` recall 为 `1.0`，`support/final` recall 为 `0.0`，最终答案对 affidavit 责任关系产生错误推断 | gold chunk 已出现但未进入最终上下文，导致生成器在错误证据上推断 |
| `ragas-cuad-0039` | baseline 命中 Conversion Week/project manager 相关 chunks；`refactor_v1` 最终回答称合同没有相关规定 | strong 阶段仍有命中，但 final miss 后触发否定式回答 |
| `ragas-cuad-0046` | Section 8 / Service Level Credits 相关问题中，`strong` 仍有部分 recall，最终 `NOT_FOUND` | 条款号、remedy、indemnification 这类多 aspect 查询被 final selection 压窄 |
| `ragas-cuad-0001` | 事实答案正确，但 `answer_correctness` 从 baseline `0.920778` 降至 `0.326` | 答案过度包装，包含 `Supporting Details`、source 解释和额外声明；短 reference 场景下被 RAGAS 惩罚 |

这些样本说明失败模式至少有三类：

1. **后段筛除 gold evidence**：候选里有，final 没有。
2. **误触发 `NOT_FOUND`**：早期阶段有强证据，但最终证据不足时直接否定。
3. **答案评测面过度展开**：事实正确，但 `response_for_eval` 太长、带 markdown/source/sub-question 结构，降低 answer correctness。

#### 12.4.5 根因归纳

##### 12.4.5.1 Final evidence selection 缺少 recall guard

当前 `evidence_matrix_budgeted_selection` 倾向于提升 precision，但没有硬性保证：

- 每个 sub-question 至少保留 top strong evidence
- 每个 detected aspect 至少保留一条候选
- 早期阶段命中过 reference-like chunk 时，final selection 不能把全部命中候选删除
- `NOT_FOUND` 判断必须参考 first/strong/support 多阶段证据，而不能只看 final prompt 中剩下的片段

结果是：系统在前半段“找到了”，但在后半段“忘了”。

##### 12.4.5.2 `response_for_eval` 没有短答案化

`vector_rag` 的平均答案长度约 `66.8` 词，`youtu_graph_rag` 的平均答案长度约 `155.4` 词。`youtu_graph_rag` 的 `60` 条结果中，有 `55` 条包含 markdown、source 说明或 sub-question 结构痕迹。

这会影响 RAGAS：

- reference 很短时，额外解释会被视为偏离答案
- source attribution、supporting details、reasoning statement 对 correctness 没有帮助，反而引入可被判为冗余或不一致的内容
- `final_answer` 与 `final_answer_for_eval` 没有充分区分，导致面向用户的解释格式污染评测答案

##### 12.4.5.3 `NOT_FOUND` 策略过于依赖最终上下文

`ragas-cuad-0008`、`0039`、`0046` 都表明：即使 first/strong 阶段已经命中，最终上下文缺失时仍可能输出 `NOT_FOUND`。这类错误比普通漏答更伤，因为它会同时拉低：

- `answer_correctness`
- `answer_relevancy`
- `context_recall`
- `faithfulness`

正确策略应该是：`NOT_FOUND` 只能在多阶段证据均不足时触发；如果早期存在强候选，应进入 rescue/fallback answer，而不是直接否定。

##### 12.4.5.4 多跳抽象题的 planning/slot 覆盖仍不足

`multi_hop_abstract_query_synthesizer` 的 `context_recall` 与 `final_chunk_recall` 退化最大，说明多 aspect、多条款、多阶段问题仍然容易被压成过少的 evidence slots。即使 §12.3.68 已规划 planning layer，本轮全量结果说明：该方向仍是必要主线，但还要和 final selection recall guard 联动，否则 slot 即使生成了，最终也可能被 budget 机制筛掉。

#### 12.4.6 机制复盘：当前方案中不合理的“叠机制”问题

重新看代码路径后，上一版优化方案中“增加 recall guard / slot quota / final miss rescue / NOT_FOUND 门控”的方向虽然能解释现象，但仍然偏向继续叠补丁。更根本的问题是：当前 `refactor_v1` 后半段已经有太多层选择、压缩、改写与评测面分叉，继续加 guard 只会让链路更难归因。

本节改为减法优先：先删掉或合并互相打架的机制，再观察指标是否回到合理区间。

##### 12.4.6.1 不合理点一：同一批证据被多次压缩

当前后半段至少存在这些连续压缩点：

1. `support_chunk_budget` 先把每个子问题 top chunks 截成 `2/3/4` 个“支持证据”。
2. `_build_refactor_evidence_matrix()` 再按 `max_evidence_per_subquestion` 与 `cross_subquestion_dedup_threshold` 做二次裁剪。
3. `_select_refactor_context_from_evidence_matrix()` 再按 `backing_context_budget` / `global_context_budget` 做三次选择。
4. `final_eval_chunk_ids` 初步由 final chunk 与 external chunk 合并得到。
5. `support_eval_set` 又把 `final_eval_chunk_ids` 收缩到 `evidence_role == support` 的 chunk。

这意味着 gold chunk 只要在任意一层被判断为 `context` 而不是 `support`，或者被 dedup / budget 排除，就不会进入 RAGAS 的 `retrieved_contexts`。这解释了为什么 `first_stage` / `strong` 召回很高，但 `final_chunk_recall` 明显下降。

**减法判断**：这里不应再加一个 `recall_guard`。更合理的是减少选择器层数，只保留一个最终证据选择器。

##### 12.4.6.2 不合理点二：最终回答上下文与评测上下文不一致

代码中先构造 `backing_chunk_ids/backing_chunk_contents` 给 final LLM 生成答案，后面又单独构造 `final_eval_chunk_ids/final_eval_chunk_contents` 给评测。`refactor_v1 + phase4` 下，评测上下文还会被 `support_eval_set` 再过滤一次。

结果是：

- final answer 可能基于 backing context 中的证据生成；
- RAGAS faithfulness 却只看到更窄的 final eval context；
- 因此答案即使来自系统内部证据，也会被判为不忠实。

**减法判断**：不要维护两套上下文。用于生成答案的 chunk id 应与 `evaluation_payload.retrieved_context_ids` 完全一致。若产品回答需要更多背景，那应作为展示层信息，不应参与当前 RAGAS 对照。

##### 12.4.6.3 不合理点三：`NOT_FOUND` 同时承担“答案状态”和“检索控制信号”

当前子问题 LLM 可以输出 `NOT_FOUND`；grounding validator 也可能把答案改写成 `NOT_FOUND`；后续 rescue、reasoning plan、最终生成又会把 `NOT_FOUND` 当作证据不足信号。

这会形成错误反馈环：

1. top-2 support 没覆盖答案；
2. validator 将子答案置为 `NOT_FOUND`；
3. provenance 虽然可能保留 `support_pairs`，但 reasoning/answer 层已经认为该子问题缺失；
4. final answer 倾向输出“未找到”；
5. 评测中表现为 `0008/0039/0046` 这类“前面找到了，最后说没有”。

**减法判断**：`NOT_FOUND` 不应作为检索后半段的硬控制信号。子问题层可以记录“未形成答案”，但不能因此丢弃候选证据，也不能直接推动最终答案全局否定。

##### 12.4.6.4 不合理点四：planning layer 默认开启，但覆盖校验本身有偏差

当前 `base_config.yaml` 中：

```yaml
retrieval:
  agent:
    planning_layer_enabled: true
```

但 `_apply_planning_layer()` 的覆盖文本 `_aggregate_text()` 会把 `original_question[:500]` 拼进去。slot 的关键词本来就来自原问题或原问题中的 aspect，因此覆盖检查很容易因为“原问题里出现了这些词”而判定 covered，而不是因为已有 sub-question 真正覆盖了该检索意图。

这会导致两个问题：

- planning layer 看起来启用了，但可能没有真正补足子问题；
- 如果它确实补问题，又会引入更多子问题、更多 support budget 分配和更多 final selection 竞争。

**减法判断**：在修正覆盖校验前，planning layer 不应默认开启。先关闭它，回到可解释的子问题集合，再决定是否需要重新启用。

##### 12.4.6.5 不合理点五：先强制生成结构化长答案，再试图从中抽短答案

final prompt 明确要求输出：

- `Grounded Facts from the Target Contract`
- `Explicit Semantic Alignment from External Evidence`
- `Inference and Bridge Conclusion`
- `Final Answer:`

随后 `_split_answer_and_reasoning_surfaces()` 再尝试抽出 `Final Answer` 作为 `response_for_eval`。这属于先制造复杂 surface，再做后处理清洗。

在短事实题中，这会直接伤害 `answer_correctness`。例如 `ragas-cuad-0001` 只需要地址，`refactor_v1` 却输出 supporting details、source 说明和额外声明。

**减法判断**：评测路径不应生成结构化长答案。应该直接生成短答案；需要解释时再另走产品展示 surface。

##### 12.4.6.6 不合理点六：跨子问题语义 dedup 对合同 chunk 风险较高

`_build_refactor_evidence_matrix()` 用 `_semantic_text_overlap_ratio()` 和 `cross_subquestion_dedup_threshold` 做跨子问题去重。合同里相邻条款、重复定义、通知地址、义务条款经常共享大量词汇，但法律意义不同。

语义 overlap 去重在开放文本中可能有用，但在合同 QA 中容易误删：

- 同一 section 的前半/后半 chunk；
- 同一概念在定义条款和义务条款中的两次出现；
- 多跳题中 bridge 两侧看起来相似但角色不同的证据。

**减法判断**：先取消语义 dedup，只做 chunk id 级精确去重。等 recall 恢复后，再评估是否需要轻量去重。

#### 12.4.7 减法优先优化方案

##### 12.4.7.1 P0-S1：统一 final prompt context 与 eval context

目标：消除“答案基于 A，上下文评测用 B”的错位。

做法：

1. 生成最终答案时使用的 chunk ids，就是 `evaluation_payload.retrieved_context_ids`。
2. 删除或禁用 `support_eval_set` 对 `final_eval_chunk_ids` 的二次过滤。
3. trace 中只保留一个 `final_context_chunk_ids`，不要同时维护 backing/final/eval 三套主上下文。

预期收益：

- `faithfulness` 会更真实地反映生成是否忠实；
- 如果答案确实来自 backing context，不会因为 eval context 被缩窄而被误扣；
- 逐题归因更简单。

##### 12.4.7.2 P0-S2：只保留一个最终证据选择器

目标：减少后半段多选择器互相覆盖。

优先方案：**临时关闭 `phase4_enabled`，回退到已有的 `_select_chunk_ids_with_subquestion_guarantee()` 路径。**

理由：

- 这个旧路径已经具备 per-subquestion guarantee；
- `phase4` 当前绕开了该 guarantee；
- 不需要新增 recall guard，只要先别绕开已有保证。

建议配置实验：

```yaml
retrieval:
  orchestration:
    phase4_enabled: false
```

如果必须保留 evidence matrix，则把它降级为 trace/diagnostic，不参与 final chunk selection。等指标稳定后，再决定是否重构成唯一选择器。

##### 12.4.7.3 P0-S3：取消 `support`/`context` 对评测证据的硬切分

目标：避免 gold chunk 因不是 top support 而从 eval context 消失。

做法：

1. `support_chunk_budget` 只用于子答案生成 prompt 的前排证据，不再决定最终评测上下文。
2. final context 从每个子问题的 `retrieved_chunks_all` / strong-reranked 列表中按顺序取，而不是只取 `candidate_support_spans`。
3. `evidence_role` 可以保留为 trace 字段，但不能作为 `retrieved_contexts` 的过滤条件。

这是比“strong-hit recall guard”更简单的做法：不额外救回 strong hit，而是从一开始就不要把 strong list 硬切成 support-only。

##### 12.4.7.4 P0-S4：降低 `NOT_FOUND` 的控制权

目标：让 `NOT_FOUND` 从“检索链路控制信号”退回为“答案表达状态”。

做法：

1. grounding validator 不再把子答案硬改成 `NOT_FOUND`；改为写入 `grounding_status = insufficient_support`。
2. 即使子答案为 `NOT_FOUND`，该子问题的 strong chunks 仍可进入 final context。
3. 最终回答是否说“未找到”，只由最终上下文是否为空/是否无相关候选决定，而不是由某个子答案字符串决定。
4. 多子问题场景中，只允许局部缺失，不允许一个子问题 `NOT_FOUND` 使整个问题变成 `NOT_FOUND`。

这不是新增门控，而是删除 `NOT_FOUND` 对检索和上下文选择的副作用。

##### 12.4.7.5 P0-S5：评测路径直接生成短答案

目标：不要先生成长结构化答案，再靠 `_split_answer_and_reasoning_surfaces()` 抽取。

做法：

1. RAGAS/eval 模式下，final prompt 直接要求：
   - answer only the question;
   - no markdown headings;
   - no source explanation;
   - no reasoning sections;
   - one sentence for fact lookup, concise bullets only for list questions。
2. `response_for_eval` 直接使用该短答案。
3. 如果产品需要可解释答案，再单独生成 `response`，但不要让它污染 `response_for_eval`。

这比继续增强 extractor 更稳，因为 extractor 永远是在修补 prompt 诱导出的格式噪声。

##### 12.4.7.6 P1-S1：默认关闭 planning layer，修正后再启用

目标：避免一个覆盖校验有偏差的扩容机制继续增加复杂度。

建议先改配置：

```yaml
retrieval:
  agent:
    planning_layer_enabled: false
```

后续如果要重启，至少先做一个减法修正：`validate_coverage` 的 coverage blob 不应包含 `original_question`，只能检查已有 sub-question 和 retrieval queries 是否覆盖 slot。

在这之前，不建议再做 slot quota、planning budget 联动等加法。

##### 12.4.7.7 P1-S2：取消跨子问题语义 dedup

目标：避免合同证据被“看起来相似”误删。

做法：

- `cross_subquestion_dedup_threshold` 暂时不参与 evidence matrix；
- 只按 `chunk_id` 精确去重；
- 如果 token budget 紧张，宁可提高 per-subquestion top chunk 的截断规则可解释性，也不要用文本 overlap 删除法律片段。

#### 12.4.8 减法实验顺序

建议下一轮只做少量 ablation，避免同时改太多无法归因。

| 实验 | 改动 | 主要验证问题 |
|------|------|--------------|
| A | `phase4_enabled=false`，回退单一 final selector | `final_chunk_recall` 是否恢复，`first_stage_hit && final_miss` 是否下降 |
| B | 在 A 基础上，删除 `support_eval_set` 二次过滤，统一 prompt/eval context | `faithfulness` 是否明显恢复 |
| C | 在 B 基础上，eval prompt 直接生成短答案 | `answer_correctness` 是否恢复，尤其是短事实题 |
| D | 在 C 基础上，`planning_layer_enabled=false` | 多跳抽象题是否更可解释；子问题数量和 final recall 是否稳定 |
| E | 在 D 基础上，取消语义 dedup，仅保留 chunk id 去重 | multi-hop / multi-aspect recall 是否继续改善 |

优先级判断：

1. **先关掉或绕开 phase4 evidence matrix selector**，因为它是绕过 per-subquestion guarantee 的主要变化。
2. **再统一 final/eval context**，因为这是 faithfulness 评测错位的直接来源。
3. **再改短答案 prompt**，因为这是 answer correctness 的表面噪声来源。
4. **最后处理 planning layer 和 dedup**，因为它们影响更广，需要在主链路简化后再看真实贡献。

#### 12.4.9 当前减法结论

本轮不建议继续优先实现“recall guard + slot quota + final rescue + NOT_FOUND 门控”这一组加法方案。更合理的路线是：

- 删掉重复 final/eval context 分叉；
- 删掉 `support_eval_set` 的 support-only 评测过滤；
- 暂停 `phase4` evidence matrix 作为最终选择器；
- 降低 `NOT_FOUND` 对检索链路的副作用；
- 评测路径直接生成短答案，而不是生成长答案后抽取；
- 默认关闭当前 coverage 校验有偏差的 planning layer；
- 合同 QA 中先取消语义 dedup，只做 chunk id 去重。

一句话：**先把系统从“多层筛选 + 多层改写 + 多套上下文”减回“一套候选、一套选择、一套评测上下文、一个短答案面”，再讨论是否需要重新加机制。**

