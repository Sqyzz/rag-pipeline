# CUAD 采样图社区退化检查清单

## 1. 目的

当 QA 先经过采样，再根据采样后的 QA 反推一份新的 docs 子集来构图时，图结构会天然变小。

这不会必然导致 community 构建失败，但可能导致：

- 图过稀
- 社区过碎
- 社区 summary 失去抽象意义
- GraphRAG 的社区优势无法体现

本清单用于判断：

- 采样后的 docs 是否仍适合构建 community
- 当前 community 是否还能作为 GraphRAG / youtu-GraphRAG 的有效高层结构


## 2. 先明确目标

采样 docs 后，构图的目标不是“尽可能还原全量图”，而是同时满足：

1. QA 与图库文档对齐
2. 图仍保留足够的结构信号
3. community 仍有高层聚合意义

如果只能满足第 1 点，而第 2、3 点明显退化，那么这套采样图更适合做局部检索测试，不适合强调 GraphRAG 的 community 能力。


## 3. 必看基础指标

构图完成后，先记录：

- `num_nodes`
- `num_edges`
- `num_communities`
- `num_levels`
- `summarized_communities`

这些字段通常可在构图结果或 build metrics 里拿到。

初步判断规则：

- `num_nodes` 和 `num_edges` 极小：图基本已经没有可聚类价值
- `num_communities` 很多但 `summarized_communities` 很少：说明社区能切出来，但大多不够大或不够稳定，不足以形成有效 summary
- `num_levels` 退化到非常浅：层级结构可能已经不明显


## 4. 社区是否“能跑”与“是否有用”要分开判断

### 4.1 能跑

只要图里还有基本的节点和边，Leiden 等社区算法通常都能跑出结果。

### 4.2 有用

真正重要的是社区是否还有高层语义价值。

以下现象说明“能跑但没用”：

- 大部分社区只含很少节点
- 社区内部支持 chunk 很少
- community summary 只是重复局部事实
- 不同社区之间缺少明显主题差异


## 5. 重点检查项

### 5.1 社区大小分布

要看社区是否几乎全是小团块。

危险信号：

- 大量社区只覆盖 1 到 3 个 chunk
- 大量社区只对应单一边簇
- 几乎没有中等规模以上社区

这说明图的高阶结构已经丢失。

### 5.2 连通性

要看图是不是被切成很多小连通分量。

危险信号：

- 最大连通块很小
- 大多数节点落在很多碎片化连通分量中

这通常意味着采样过度，主题之间失去连接。

### 5.3 community summary 质量

随机抽查若干 community summary，判断它是否真的在做“主题聚合”。

健康表现：

- 能概括一组相关义务、风险或条款主题
- 不是只重复单个事实
- 不同 summary 之间有明显主题区分

危险信号：

- summary 很短且空泛
- summary 只是在重述一个 chunk
- 多个社区 summary 非常相似，缺少区分度

### 5.4 对 global / cross 任务的支持度

如果采样图退化，最先受影响的通常不是 `local_factual`，而是：

- `cross_clause`
- `global_summary`

危险信号：

- local 题还能答，但 cross/global 明显崩掉
- evidence recall 在 chunk 上还能勉强命中，但 community 命中几乎没有意义


## 6. 如何理解 matched/unmatched 采样比例

你当前做法里：

- matched docs：与 QA 对应的文档
- unmatched docs：不直接对应 QA 的其他文档

这里的 unmatched 不是噪声，它们对 community 很重要，因为它们提供：

- 额外边
- 主题桥接
- 结构冗余
- 更稳定的社区边界

因此：

- `matched_doc_ratio` 太高，图容易退化为“只服务 QA 的局部子图”
- `matched_doc_ratio` 适中，通常更有利于保留社区结构

经验上更值得优先尝试的区间是：

- `0.2`
- `0.3`
- `0.4`

不建议一开始就用：

- `0.8`
- `1.0`

除非你的目标只是严格文档对齐，而不是评估 community 能力。


## 7. 建议的实验对照方式

为了判断采样比例是否把 community 破坏掉，建议至少做 3 组对照：

1. `matched_doc_ratio = 1.0`
2. `matched_doc_ratio = 0.3`
3. `matched_doc_ratio = 0.2`

每组都记录：

- `num_nodes`
- `num_edges`
- `num_communities`
- `summarized_communities`
- `global_summary` 表现
- `cross_clause` 表现

如果出现以下情况：

- ratio 提高后 local 表现变化不大
- 但 cross/global 明显下降
- 同时 community 指标明显变碎

就可以基本判断：

- 采样已经破坏了社区层结构


## 8. 判断结论模板

### 8.1 可以继续使用采样图做 community 评测

满足多数条件时可接受：

- 图规模仍然足够
- 社区不是极端碎片化
- 有一定数量的高质量 summarized communities
- global/cross 任务仍能从 community 中受益

### 8.2 不建议继续把它当作 GraphRAG 社区评测集

满足多数条件时应谨慎：

- 图结构明显过小或过稀
- summarized communities 很少
- community summary 质量很差
- global/cross 的退化远大于 local

这种情况下，这套数据更适合做：

- 文档对齐检索实验
- 局部问答实验

而不适合强调 community-level reasoning。


## 9. 最终建议

不要把“community 能跑出来”当作“community 仍然适合评测”的证据。

对于采样后的 CUAD 图，更合理的判断顺序是：

1. QA 是否与文档对齐
2. 图是否仍有足够结构规模
3. community 是否仍然有聚合语义
4. global/cross 是否仍然受益于社区结构

只有这四步都成立，采样后的 docs 才适合作为 GraphRAG / youtu-GraphRAG 的社区评测底座。

