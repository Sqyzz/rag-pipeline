# Ragas 补充评测 Pipeline 方案（CUAD + GraphRAG / LightRAG / youtu）

## 1. 文档目标

本文档定义一条独立于主对比流程的补充评测 Pipeline，用于在 `CUAD` 数据集上基于 `ragas` 对以下三种方法进行统一对比：

- `graph_rag`
- `lightrag`
- `youtu_graph_rag`

该 Pipeline 的定位是：

1. 不替代当前主评测流程
2. 不要求复刻 `local_factual / cross_clause / global_summary`
3. 作为一条独立补充 benchmark，提供自动生成测试题 + 统一 RAG 质量评测

---

## 2. 结论摘要

结论如下：

1. `ragas` 适合作为当前项目的补充评测框架
2. `ragas` 可从 CUAD 文档自动生成测试集
3. `ragas` 可统一评估 `graph_rag / lightrag / youtu_graph_rag`
4. `youtu` 不是阻塞项，项目内现有适配层已经能导出：
   - `answer`
   - `evidence_chunks[].text`
   - `evidence_chunks[].chunk_id`
   - `evidence_chunks[].doc_id`
5. 该 Pipeline 应作为“独立补充线”存在，不应宣称与主流程三类能力题完全等价

正式执行前，必须先完成小样本冒烟流程：

- `docs/ragas_cuad_smoke_test_pipeline.md`

---

## 3. Pipeline 的目标边界

### 3.1 本 Pipeline 要解决的问题

本 Pipeline 主要解决以下问题：

1. 自动从 `CUAD` 文档构造一批可复用的测试题
2. 让三种方法在同一批题上统一回答
3. 统一收集各方法的检索上下文与上下文 ID
4. 使用 `ragas` 输出一组独立、可解释、可比较的补充指标

### 3.2 本 Pipeline 不解决的问题

本 Pipeline 不负责复刻当前主流程的以下能力：

1. `Top10 / Top20`
2. `reject / open`
3. 主流程的 `TopK Accuracy`
4. 主流程中严格定义的：
   - `local_factual`
   - `cross_clause`
   - `global_summary`
5. 主流程中基于 `supporting_chunks / query_doc_key / title` 的定制化诊断口径

---

## 4. Ragas 在本项目中的定位

`ragas` 在本项目中的定位应为：

1. 自动测试集生成器
2. 通用 RAG 评测器
3. 主流程之外的补充 benchmark

不应把它当作：

1. 当前主评测协议的替代品
2. 当前三类能力题生成逻辑的直接替代品

---

## 5. Ragas 默认支持的题型

`ragas` 当前默认内置 3 类 query synthesizer：

1. `single_hop_specific`
2. `multi_hop_specific`
3. `multi_hop_abstract`

建议在本文档中将其解释为：

1. `single_hop_specific`
   - 偏局部事实型问题
   - 依赖单个 chunk 或单个文档片段
2. `multi_hop_specific`
   - 偏跨片段、跨概念关联型问题
   - 需要多个片段共同支持
3. `multi_hop_abstract`
   - 偏抽象综合型问题
   - 更接近文档级或主题级综合理解

说明：

1. 这是“近似对应”，不是与主流程能力标签完全等价
2. 补充 Pipeline 中应保留 `ragas` 原始 synthesizer 名称，避免与主流程能力题混淆

---

## 6. 建议的整体架构

推荐实现为四阶段：

1. `CUAD 文档准备`
2. `Ragas 测试集生成`
3. `三方法统一推理与结果归一化`
4. `Ragas 统一评测与汇总`

统一数据流如下：

`CUAD raw -> docs -> ragas testset -> graph_rag/lightrag/youtu answers -> EvaluationDataset -> ragas metrics -> summary`

第一次联调时，不应直接运行全量流程。

建议顺序：

1. 先完成 `docs/ragas_cuad_smoke_test_pipeline.md`
2. smoke test 通过后，再进入本文档的正式补充评测流程

---

## 7. 阶段一：CUAD 文档准备

### 7.1 输入

建议支持以下任一输入：

- `data/raw/cuad/CUADv1.json`
- `data/raw/cuad/test.json`

### 7.2 输出

新增文档资产，例如：

- `data/processed/cuad_ragas_docs.jsonl`

每条文档建议包含：

- `doc_id`
- `title`
- `page_content`
- `source_dataset=cuad`
- 其他必要 metadata

### 7.3 文档组织策略

建议区分“两套粒度”，不要混用：

1. 文档资产层：可以保留“每份合同一个 document”
2. `ragas` 造题层：建议优先使用现有 chunk 资产做 `pre-chunked` 生成

推荐第一版直接基于 chunk 级输入生成测试集，而不是只把整份合同作为一个长 document 交给 `ragas`。

原因：

1. 当前三条 RAG 路线的可对齐检索证据本质上都是 chunk 级
2. `ragas` 已支持 `generate_with_chunks`，可直接复用已有 chunk 切分
3. 这样 `reference_context_ids` 可以稳定使用 `chunk_id`
4. 可避免 `reference_context_ids` 用 `doc_id`、而 `retrieved_context_ids` 用 `chunk_id` 的口径错位
5. 文本级 `ContextPrecision / ContextRecall` 也会更贴近真实检索输入

### 7.4 先做 sampled corpus

正式补充评测前，建议先从 `docs.jsonl` 派生：

- `data/processed/cuad_docs_sampled.jsonl`

并基于 sampled corpus 完成一次完整联调。

原因：

1. 全量 `CUAD` 构图和造题成本较高
2. 三条 RAG 路线的第一次统一联调更适合在小规模上完成
3. `ragas` 测试集生成本身也会消耗额外 token

---

## 8. 阶段二：Ragas 测试集生成

### 8.1 目标

从 `CUAD` 的 sampled chunk 资产自动生成一批补充测试题。

### 8.2 建议生成入口

建议新增脚本：

- `src/experiments/run_ragas_cuad_generate.py`

职责：

1. 读取 sampled corpus 对应的 chunk 资产，优先使用现有 `chunks.jsonl`
2. 转为带 metadata 的 LangChain `Document`
3. 调用 `ragas.testset.synthesizers.generate.TestsetGenerator.generate_with_chunks`
4. 输出标准化测试集文件

第一轮建议直接输入 sampled chunks，而不是全量 docs。

### 8.3 建议输出文件

- `data/queries/ragas_cuad_testset.jsonl`
- `data/queries/ragas_cuad_testset_summary.json`

### 8.4 每条测试样本建议保存的字段

建议保留：

- `qid`
- `question`
- `reference`
- `reference_contexts`
- `reference_context_ids`
- `reference_doc_ids`
- `synthesizer_name`
- `persona_name`
- `query_style`
- `query_length`

其中建议约定：

1. `reference_context_ids` 固定保存 `chunk_id`
2. `reference_doc_ids` 单独保存 `doc_id`
3. 不要把 `doc_id` 塞进 `reference_context_ids`

### 8.5 query_distribution 建议

第一版建议平均采样三类：

1. `single_hop_specific`: `1/3`
2. `multi_hop_specific`: `1/3`
3. `multi_hop_abstract`: `1/3`

若后续发现某类题对 CUAD 不稳定，再调整为：

1. `single_hop_specific`: `0.4`
2. `multi_hop_specific`: `0.4`
3. `multi_hop_abstract`: `0.2`

### 8.6 对生成质量的要求

建议在生成阶段额外输出：

- 各 synthesizer 的题量
- 空 reference 比例
- reference_contexts 长度分布
- reference_context_ids 缺失率
- reference_doc_ids 缺失率

若 `reference_context_ids` 缺失率过高，则后续 `IDBased` 指标价值会下降。

---

## 9. 阶段三：三方法统一推理与结果归一化

### 9.1 目标

让 `graph_rag / lightrag / youtu_graph_rag` 在同一批 `ragas` 测试题上回答，并导出统一格式。

### 9.2 建议新增脚本

- `src/experiments/run_ragas_cuad_compare.py`

职责：

1. 读取 `ragas_cuad_testset.jsonl`
2. 对每题调用三种方法
3. 统一保存：
   - `response`
   - `retrieved_contexts`
   - `retrieved_context_ids`
   - `retrieved_doc_ids`
   - `telemetry`
4. 按方法输出原始结果

### 9.3 建议输出文件

- `outputs/results/ragas/ragas_cuad_graph_rag_predictions.jsonl`
- `outputs/results/ragas/ragas_cuad_lightrag_predictions.jsonl`
- `outputs/results/ragas/ragas_cuad_youtu_predictions.jsonl`
- `outputs/results/ragas/ragas_cuad_compare_merged.jsonl`

---

## 10. 三种方法的结果归一化规则

### 10.1 graph_rag

建议从现有 `answer_with_graphrag` 或统一 compare 产物中提取：

- `answer -> response`
- `evidence_chunks[].text -> retrieved_contexts`
- `evidence_chunks[].chunk_id -> retrieved_context_ids`
- `evidence_chunks[].doc_id -> retrieved_doc_ids`

说明：

1. `graph_rag` 当前的 `evidence` 是 community summary 级证据，不是 chunk 级检索证据
2. 真正可用于 `ragas` 检索评测映射的是 `evidence_chunks`

### 10.2 lightrag

建议从现有 `answer_with_lightrag` 或 compare 结果中提取：

- `answer -> response`
- `evidence_chunks[].text -> retrieved_contexts`
- `evidence_chunks[].chunk_id -> retrieved_context_ids`
- `evidence_chunks[].doc_id -> retrieved_doc_ids`

说明：

1. 当前仓库里的 `lightrag_adapter` 已经输出标准化 `evidence_chunks`
2. 第一版不应再从松散的 `contexts/evidence text` 做二次猜测映射

### 10.3 youtu_graph_rag

`youtu` 必须通过现有适配层接入，不直接消费后端原始响应。

建议固定使用：

- `src/baselines/youtu_graph_rag_adapter.py::answer_with_youtu_graphrag`

因为该适配层已经做了：

1. `retrieved_chunks + retrieved_chunk_ids` 对齐
2. `chunk_id` fallback 修复
3. `doc_id` 对齐补全
4. `evidence_chunks` 标准化

对 `ragas` 的映射建议如下：

1. `payload["answer"] -> response`
2. `payload["evidence_chunks"][].text -> retrieved_contexts`
3. `payload["evidence_chunks"][].chunk_id -> retrieved_context_ids`
4. `payload["evidence_chunks"][].doc_id -> retrieved_doc_ids`

### 10.4 统一归一化后的单条样本结构

建议统一成：

```json
{
  "qid": "ragas-cuad-0001",
  "method": "youtu_graph_rag",
  "question": "...",
  "response": "...",
  "reference": "...",
  "retrieved_contexts": ["...", "..."],
  "retrieved_context_ids": ["chunk_1", "chunk_2"],
  "retrieved_doc_ids": ["doc_1", "doc_1"],
  "reference_contexts": ["...", "..."],
  "reference_context_ids": ["chunk_a", "chunk_b"],
  "reference_doc_ids": ["doc_1", "doc_2"],
  "synthesizer_name": "multi_hop_abstract_query_synthesizer",
  "telemetry": {}
}
```

ID 口径必须固定：

1. `retrieved_context_ids` / `reference_context_ids` 都使用同一套 `chunk_id`
2. `retrieved_doc_ids` / `reference_doc_ids` 单独保存 `doc_id`
3. 若测试集仍只能稳定产出 `doc_id` 而不是 `chunk_id`，则第一版不要启用 `IDBased` 指标

---

## 11. 阶段四：Ragas 统一评测

### 11.1 目标

把三种方法的归一化结果转换成 `ragas` 的 `EvaluationDataset`，统一跑补充指标。

### 11.2 建议新增脚本

- `src/evaluation/run_ragas_eval.py`

职责：

1. 读取归一化后的 merged 文件
2. 按方法分组构建 `EvaluationDataset`
3. 跑指定 `ragas` 指标
4. 输出逐题结果与汇总结果

### 11.3 推荐第一版指标

建议第一版优先启用以下 4 个核心指标：

1. `AnswerCorrectness`
2. `Faithfulness`
3. `ContextPrecision`
4. `ContextRecall`

在满足以下前提后，再启用：

5. `IDBasedContextPrecision`
6. `IDBasedContextRecall`

原因：

1. 前 4 个指标已经能覆盖：
   - 回答正确性
   - 回答是否忠于上下文
   - 检索排序质量
   - 检索覆盖率
2. `IDBased` 指标只有在 `reference_context_ids` 与 `retrieved_context_ids` 同为 chunk 级且缺失率可接受时才有解释性
3. 这组指标最容易解释

建议启用 `IDBased` 的前置验收条件：

1. `reference_context_ids` 大多数样本非空
2. `reference_context_ids` 明确为 `chunk_id`
3. 三种方法输出的 `retrieved_context_ids` 与测试集使用同一套 chunk ID 空间

### 11.4 暂不建议第一版启用的指标

第一版不建议优先上：

1. 复杂自定义 rubric
2. 噪声敏感度
3. 大量 agent 指标
4. 与当前任务无关的传统生成指标

理由是会增加实现复杂度，但不会明显提升当前补充 Pipeline 的核心价值。

---

## 12. 建议输出产物

### 12.1 逐题结果

- `outputs/results/ragas/ragas_cuad_eval_per_sample.jsonl`

每条至少包含：

- `qid`
- `method`
- `synthesizer_name`
- 各项 `ragas` 指标

### 12.2 汇总结果

- `outputs/results/ragas/ragas_cuad_eval_summary.json`
- `outputs/results/ragas/ragas_cuad_eval_summary.csv`
- `outputs/results/ragas/ragas_cuad_eval_summary.md`

### 12.3 推荐汇总维度

建议至少输出两个层次：

1. 按 `method`
2. 按 `method + synthesizer_name`

这样可以同时回答两个问题：

1. 三种方法总体谁更强
2. 在 `single_hop / multi_hop_specific / multi_hop_abstract` 各自谁更强

---

## 13. youtu 接入的专项要求

### 13.1 必须使用现有适配层

`youtu` 的查询结果不可直接裸接 `ragas`。

必须通过：

- `src/baselines/youtu_graph_rag_adapter.py`

原因：

1. 后端返回字段存在多种变体
2. `chunk_id` 可能缺失，需要 fallback
3. `doc_id` 需要通过本地 store map 进行对齐补全
4. 当前仓库已经具备这层兼容逻辑，重复实现没有必要

### 13.2 youtu 在本 Pipeline 的验收标准

要认为 `youtu` 已完整接入，必须满足：

1. 每题都有 `response`
2. 大部分样本都有非空 `retrieved_contexts`
3. `retrieved_context_ids` 缺失率可统计
4. `retrieved_doc_ids` 缺失率可统计
5. `ragas` 文本级指标可稳定运行

附加优选标准：

1. `IDBasedContextPrecision`
2. `IDBasedContextRecall`

都可以在大多数样本上正常计算。

---

## 14. MVP 实施范围

建议第一版严格控制范围，只做最小可用实现。

### 14.1 第一版必须完成

1. CUAD chunk 资产转 `ragas` `pre-chunked` 输入
2. 基于 sampled corpus 完成一次 smoke test
3. 生成一份 `ragas` 测试集
4. 三种方法统一回答并导出归一化结果
5. 跑 4 个核心指标
6. 输出按方法和按题型的 summary

满足 ID 对齐前提后，再补跑 2 个 `IDBased` 指标。

### 14.2 第一版不必完成

1. 自定义 synthesizer
2. 与主流程三类题强绑定映射
3. reject/open 双模式
4. Top10/Top20 双档
5. 大规模参数扫描

---

## 15. 推荐新增文件清单

建议新增以下文件：

1. `src/experiments/run_ragas_cuad_generate.py`
2. `src/experiments/run_ragas_cuad_compare.py`
3. `src/evaluation/run_ragas_eval.py`
4. `docs/ragas_cuad_supplement_pipeline.md`
5. `docs/ragas_cuad_smoke_test_pipeline.md`

可选新增：

1. `src/evaluation/export_ragas_report.py`
2. `src/utils/ragas_converters.py`

---

## 16. 推荐命令形态

### 16.1 生成测试集

```bash
python src/experiments/run_ragas_cuad_generate.py \
  --chunks-file data/processed/qa_aligned_chunks.jsonl \
  --out-testset-file data/queries/ragas_cuad_testset.jsonl \
  --testset-size 120
```

### 16.2 跑三方法统一回答

```bash
python src/experiments/run_ragas_cuad_compare.py \
  --testset-file data/queries/ragas_cuad_testset.jsonl \
  --chunks-file data/processed/qa_aligned_chunks.jsonl \
  --graph-file outputs/graph/qa_aligned_graph_v3_docscoped.json \
  --communities-file outputs/graph/qa_aligned_communities_v3_docscoped_pruned.json \
  --lightrag-working-dir outputs/lightrag/qa_aligned_chunks \
  --youtu-base-url http://127.0.0.1:8000 \
  --youtu-dataset cuad_v3 \
  --out-file outputs/results/ragas/ragas_cuad_compare_merged.jsonl
```

### 16.3 跑 ragas 评测

```bash
python src/evaluation/run_ragas_eval.py \
  --pred-file outputs/results/ragas/ragas_cuad_compare_merged.jsonl \
  --out-dir outputs/results/ragas
```

---

## 17. 风险与应对

### 17.1 风险一：Ragas 生成题与 CUAD 业务分布不完全匹配

影响：

1. 题目会更通用
2. 与主流程能力标签不可直接横向替换

应对：

1. 明确标注为补充评测
2. 报告中使用 `synthesizer_name`，不混称为主流程三类题

### 17.2 风险二：youtu 的 chunk_id 对齐不稳定

影响：

1. `IDBased` 指标覆盖率下降

应对：

1. 强制通过现有 adapter 对齐
2. 报告 `retrieved_context_ids` 缺失率
3. 即使 ID 指标不全，文本级指标仍可保留

### 17.3 风险三：multi_hop_abstract 在合同文档上生成质量不稳定

影响：

1. 问题可能偏泛
2. reference 可能较弱

应对：

1. 第一版保留但控制比例
2. 单独输出该类题的结果，避免吞没其它类型

---

## 18. 最终建议

推荐正式采用以下策略：

1. 保留当前主评测流程不变
2. 新增一条 `ragas` 独立补充 Pipeline
3. 在 CUAD 上统一对比：
   - `graph_rag`
   - `lightrag`
   - `youtu_graph_rag`
4. 第一版重点输出：
   - `AnswerCorrectness`
   - `Faithfulness`
   - `ContextPrecision`
   - `ContextRecall`
   - `IDBasedContextPrecision`
   - `IDBasedContextRecall`
5. 报告维度至少包含：
   - `method`
   - `method + synthesizer_name`

这是当前约束下最稳妥、最可落地、且最容易向论文中解释的一条补充方案。
