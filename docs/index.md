# Docs 索引

本文档用于快速说明 `docs/` 目录下各文档的定位，方便按主题查阅，而不是逐个猜文件名。

## 1. 总览

当前 `docs/` 目录的内容大致可以分成四类：

1. 统一对比评测流程
2. `CUAD` 数据集上的 QA / 图谱构建方案
3. `Ragas + CUAD` 补充评测支线
4. `youtu-GraphRAG` 的问题分析与检索优化实施文档

如果你是第一次看这个目录，建议优先从以下几份开始：

- [comparison_pipeline.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/comparison_pipeline.md)
  - 主对比流程总纲，定义统一评测口径。
- [comparison_pipeline_youtu_lightrag.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/comparison_pipeline_youtu_lightrag.md)
  - `youtu / LightRAG / graph_rag` 专项对比流程。
- [ragas_cuad_runbook.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/ragas_cuad_runbook.md)
  - `Ragas` 补充评测的可执行操作手册。
- [youtu_retrieval_requirements_smoke_analysis_and_optimization.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/youtu_retrieval_requirements_smoke_analysis_and_optimization.md)
  - `youtu-GraphRAG` 当前检索问题与优化路线的总分析文档。

## 2. 统一对比评测流程

### [comparison_pipeline.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/comparison_pipeline.md)

项目当前的统一对比流程总纲。主要定义：

- 比较哪些方法：`VectorRAG / KG-RAG / GraphRAG / youtu-GraphRAG`
- 评测模式：`reject / open`
- 评测口径：`Top10 / Top20`、`Tokens / Time`
- 目标输出：统一表格与 `Pareto` 图

适合在你想确认“主评测流程应该怎么跑、怎么记指标”时先看这篇。

### [comparison_pipeline_youtu_lightrag.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/comparison_pipeline_youtu_lightrag.md)

这是一个独立于主流程的专项对比文档，聚焦：

- `graph_rag`
- `youtu_graph_rag`
- `lightrag`

它把评测拆成两个阶段：

- `build`：构图成本
- `retrieve`：问答检索质量

适合在只关心 `youtu` 与 `LightRAG` 一类方法专项比较时使用。

### [pipeline_target_refactor_plan.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/pipeline_target_refactor_plan.md)

这是“对比 Pipeline 要改造成什么样”的实施计划文档，主要回答：

- 当前评测链路已经有什么
- 还缺哪些关键能力
- 应该按什么顺序补齐

它更像改造计划，而不是最终运行手册。

## 3. CUAD 数据集上的 QA / 图谱方案

### [cuad_two_stage_qa_graph_pipeline_plan.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/cuad_two_stage_qa_graph_pipeline_plan.md)

这篇是 `CUAD` 上“两阶段 QA / 图谱流水线”的最终实施方案。核心点是：

- 不直接相信“先构图再一次性生成三类 QA”的现成路线
- 推荐先做 `local / cross`
- 再约束 docs 子集
- 然后构图、建 community
- 最后生成 `global`

适合在你要确定 `CUAD` 主线数据构造策略时阅读。

### [cuad_sampled_graph_community_checklist.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/cuad_sampled_graph_community_checklist.md)

这是一个检查清单，用来判断：

- 采样后的 docs 子集是否还适合构图
- 构出来的 community 是否还有意义
- 当前图是否已经退化到不适合强调 `GraphRAG` 优势

适合在做 sampled graph、缩小语料、或者怀疑 community 质量退化时参考。

## 4. Ragas + CUAD 补充评测支线

这组文档描述一条独立于主评测流程之外的补充 benchmark，用 `ragas` 在 `CUAD` 上做自动生成测试题和统一评测。

### [ragas_cuad_supplement_pipeline.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/ragas_cuad_supplement_pipeline.md)

偏方案设计。主要说明：

- 为什么要加这条补充评测线
- 这条线的边界是什么
- 为什么它不能替代主流程

适合先建立全局理解。

### [ragas_cuad_supplement_implementation.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/ragas_cuad_supplement_implementation.md)

偏落地实施。主要说明：

- 当前仓库是否已经具备实施基础
- 还缺哪些脚本或闭环
- 应该按什么顺序把这条支线补齐

适合在真正准备开发或补实现时阅读。

### [ragas_cuad_runbook.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/ragas_cuad_runbook.md)

偏执行手册。主要说明：

- 入口脚本有哪些
- 运行顺序是什么
- 命名和产物路径怎么约束
- 怎么先做 smoke，再扩到正式评测

如果你的目标是“把这条支线跑起来”，优先看这篇。

## 5. youtu-GraphRAG 问题分析与优化

这组文档主要围绕 `youtu-GraphRAG` 在 `CUAD` 或 smoke 场景下的表现问题、根因分析和改造方案。

### 5.1 总分析与根因判断

#### [youtu_retrieval_requirements_smoke_analysis_and_optimization.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/youtu_retrieval_requirements_smoke_analysis_and_optimization.md)

这篇是总分析文档，覆盖：

- 当前 smoke 检索评测结果
- 代码排查范围
- 系统检索链路的主要结构性问题
- 分阶段优化方向

如果只看一篇 `youtu` 检索文档，通常先看它。

#### [youtu_cuad_retrieval_root_cause_analysis.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/youtu_cuad_retrieval_root_cause_analysis.md)

更聚焦根因辨析，重点讨论：

- 问题是否主要来自 final chunk 被污染
- 还是更早发生在子问题检索阶段

适合在你想判断“到底是 rerank 问题，还是 first-stage retrieval / query formulation 问题”时看。

#### [youtu_cuad_doc_consistency_bias_plan.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/youtu_cuad_doc_consistency_bias_plan.md)

聚焦 `doc-consistency bias` 方向，主要记录：

- 当前 smoke 结果里不同题型的失败模式
- 为什么不能用单一解释覆盖所有问题
- 后续如何在检索中引入文档一致性偏置

适合在处理“跨合同噪声、多文档混淆、命中文档不稳定”问题时参考。

### 5.2 检索链路实施计划

#### [youtu_retrieval_p0_hybrid_retrieval_and_reranker_plan.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/youtu_retrieval_p0_hybrid_retrieval_and_reranker_plan.md)

这是一个索引页，说明原来的 `P0` 合并文档已经拆分成两份：

- `Hybrid Retrieval`
- `Chunk Reranker`

如果你是从旧引用跳进来的，可以从这里继续跳转。

#### [youtu_retrieval_p0_hybrid_retrieval_plan.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/youtu_retrieval_p0_hybrid_retrieval_plan.md)

聚焦 first-stage retrieval 改造，主要内容是：

- 从 dense-only 升级为 `dense + sparse + RRF`
- 提升 `Section / Article / defined term` 这类精确 lexical anchor 的召回

适合处理“候选池一开始就没把 gold chunk 拉回来”的问题。

#### [youtu_retrieval_p0_chunk_reranker_plan.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/youtu_retrieval_p0_chunk_reranker_plan.md)

聚焦 second-stage rerank 改造，主要内容是：

- 在候选召回之上做更强的 chunk 级重排
- 解决“相关区域找到了，但正确 chunk 排不上来”的问题

适合处理 `right area, wrong chunk` 这类错误。

#### [youtu_doc_scoped_prefilter_plan.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/youtu_doc_scoped_prefilter_plan.md)

聚焦 retrieval scope 控制，核心是把当前的：

- `late filtering`

升级为：

- `scope-aware prefilter retrieval`

也就是尽量在 first-stage candidate generation 时就限制到目标文档或合理 scope，避免正确证据在最早阶段被挤掉。

### 5.3 图结构改造

#### [youtu_cuad_layered_graph_implementation.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/youtu_cuad_layered_graph_implementation.md)

聚焦 `CUAD` 场景下的图分层改造，主要目标是：

- 降低噪声边对结构检索的污染
- 提高多跳检索的结构可控性
- 避免继续只靠检索白名单硬补问题

适合在你准备调整 `youtu` 构图方式，而不是只修补检索策略时阅读。

## 6. 一个简单的阅读顺序建议

如果你的目标不同，可以按下面顺序进入：

- 想看主评测流程：
  - [comparison_pipeline.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/comparison_pipeline.md)
  - [pipeline_target_refactor_plan.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/pipeline_target_refactor_plan.md)

- 想看 `CUAD` 主线数据方案：
  - [cuad_two_stage_qa_graph_pipeline_plan.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/cuad_two_stage_qa_graph_pipeline_plan.md)
  - [cuad_sampled_graph_community_checklist.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/cuad_sampled_graph_community_checklist.md)

- 想跑 `Ragas` 补充评测：
  - [ragas_cuad_supplement_pipeline.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/ragas_cuad_supplement_pipeline.md)
  - [ragas_cuad_supplement_implementation.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/ragas_cuad_supplement_implementation.md)
  - [ragas_cuad_runbook.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/ragas_cuad_runbook.md)

- 想优化 `youtu-GraphRAG`：
  - [youtu_retrieval_requirements_smoke_analysis_and_optimization.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/youtu_retrieval_requirements_smoke_analysis_and_optimization.md)
  - [youtu_cuad_retrieval_root_cause_analysis.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/youtu_cuad_retrieval_root_cause_analysis.md)
  - 再按方向进入：
    - first-stage 召回： [youtu_retrieval_p0_hybrid_retrieval_plan.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/youtu_retrieval_p0_hybrid_retrieval_plan.md)
    - second-stage rerank： [youtu_retrieval_p0_chunk_reranker_plan.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/youtu_retrieval_p0_chunk_reranker_plan.md)
    - scope 控制： [youtu_doc_scoped_prefilter_plan.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/youtu_doc_scoped_prefilter_plan.md)
    - 图结构改造： [youtu_cuad_layered_graph_implementation.md](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/docs/youtu_cuad_layered_graph_implementation.md)
