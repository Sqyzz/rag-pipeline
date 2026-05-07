# Ragas CUAD 补充评测实施文档

## 1. 文档定位

本文档不是方案讨论稿，而是基于当前仓库真实状态整理的一份实施文档，用于落地一条独立的 `ragas + CUAD + graph_rag / lightrag / youtu_graph_rag` 补充评测线。

目标是回答两个问题：

1. 当前仓库是否具备实施基础
2. 还缺哪些内容，应该按什么顺序补齐

---

## 2. 当前判断

结论如下：

1. 该补充评测线可以实施
2. 当前仓库已经具备三方法统一归一化的核心基础
3. 当前仓库还不具备“按文档直接执行”的条件
4. 主要缺口不在方法适配层，而在 `ragas` 执行闭环与新增脚本

换句话说：

1. 方向可行
2. 设计成立
3. 需要补实现，不能直接运行

---

## 3. 已具备的实施基础

### 3.1 数据准备入口已存在

当前仓库已存在：

- `src/ingestion/load_docs.py`
- `src/ingestion/sample_dedup.py`

可用于：

1. 从 `CUADv1.json` 生成 `docs.jsonl`
2. 从 `docs.jsonl` 生成 sampled corpus

说明：

当前 `load_docs.py` 对 `CUAD` 的输出粒度是 paragraph 级，而不是“每份合同一个 document”。

即当前默认产物更接近：

- `doc_id = {title}#p{idx}`

因此后续实施时应以现有真实口径为准，不要在实现文档中再假设“现成就是合同级 doc”。

### 3.2 三条方法的统一证据出口已存在

当前三种方法都已能输出 `evidence_chunks`，这是补充评测能落地的核心前提。

现状如下：

1. `graph_rag`
   - 已输出 `answer`
   - 已输出 `evidence_chunks`
2. `lightrag`
   - 已输出 `answer`
   - 已输出 `evidence_chunks`
3. `youtu_graph_rag`
   - 已通过 adapter 输出 `answer`
   - 已通过 adapter 输出 `evidence_chunks`
   - 已包含 `chunk_id/doc_id` 对齐修复与统计

这意味着补充评测不需要重写三条方法本身，只需要做统一封装与结果转换。

### 3.3 对比调度主干已存在

当前仓库已存在：

- `src/experiments/run_compare.py`

这说明：

1. 三方法联合调度不是空白
2. `graph_rag / lightrag / youtu_graph_rag` 已有统一调用面
3. 新增 `ragas` compare 脚本时，应尽量复用现有 compare 逻辑，而不是从零重写

### 3.4 本地 vendored ragas 代码已具备关键 API

仓库中已包含本地 `ragas` 源码目录：

- `ragas/`

且已确认包含以下能力：

1. `TestsetGenerator.generate_with_chunks`
2. `EvaluationDataset`
3. `AnswerCorrectness`
4. `Faithfulness`
5. `ContextPrecision`
6. `ContextRecall`
7. `IDBasedContextPrecision`
8. `IDBasedContextRecall`

因此从代码能力上看，本文档依赖的 `ragas` API 是存在的。

---

## 4. 当前缺口

### 4.1 缺少 3 个核心执行脚本

当前仓库中尚不存在以下文件：

1. `src/experiments/run_ragas_cuad_generate.py`
2. `src/experiments/run_ragas_cuad_compare.py`
3. `src/evaluation/run_ragas_eval.py`

这三个脚本是补充评测线的最小闭环：

1. 生成测试集
2. 统一推理
3. 统一评测

在它们实现之前，补充 pipeline 只能算“方案已定义”，不能算“流程已具备”。

### 4.2 当前 Python 环境缺少 ragas 运行所需依赖

当前环境检查结果显示：

1. `ragas` 可导入
2. `langchain_core` 不可导入
3. `datasets` 不可导入

而当前仓库的 `requirements.txt` 中也未显式声明：

1. `langchain-core`
2. `datasets`

这会直接影响：

1. `generate_with_chunks`
2. `EvaluationDataset` 的完整执行
3. 后续基于 `ragas` 的标准化评测脚本

因此补充 pipeline 的第一步不是写业务逻辑，而是先补环境依赖闭环。

### 4.3 文档口径与当前仓库实现存在偏差

当前需要统一以下口径：

1. `CUAD docs` 的默认粒度
2. `doc_id` 与 `chunk_id` 的职责边界
3. 第一版是否强制启用 `IDBased` 指标

现阶段建议统一为：

1. `doc_id` 单独表示文档或 paragraph 级来源
2. `reference_context_ids/retrieved_context_ids` 均以 `chunk_id` 为准
3. 第一版先以文本级指标为主
4. `IDBased` 作为第二阶段开关，而不是冒烟阶段硬门槛

---

## 5. 第一版实施原则

第一版必须遵守以下原则：

1. 不替代现有主评测流程
2. 不复刻主流程的 `local_factual / cross_clause / global_summary`
3. 不一开始追求全量实验
4. 不重新发明 compare 主干
5. 不把 `IDBased` 指标作为首轮联调阻塞项

第一版的唯一目标是：

跑通一条独立、可解释、可复用的补充 benchmark 闭环。

---

## 6. 推荐实施顺序

建议严格按以下顺序推进。

### 6.1 第一步：补依赖闭环

先补齐 `ragas` 所需依赖，并固定安装方式。

至少需要确认：

1. `langchain-core`
2. `datasets`
3. `ragas` 的导入路径与运行方式

建议结果：

1. 项目依赖文件可重建环境
2. 不依赖“本机碰巧装过某些包”

### 6.2 第二步：明确输入口径

先固定第一版的数据口径：

1. `docs` 采用当前仓库生成的 paragraph 级 `doc_id`
2. `ragas` 造题采用 chunk 级输入
3. `reference_context_ids` 固定写 `chunk_id`
4. `reference_doc_ids` 单独写 `doc_id`

这样可以避免：

1. 测试集用 `doc_id`
2. 检索结果用 `chunk_id`
3. 导致 `IDBased` 指标解释失真

### 6.3 第三步：先落地 generate 脚本

新增：

- `src/experiments/run_ragas_cuad_generate.py`

职责应收敛为：

1. 读取 sampled chunk 资产
2. 转为 `ragas` 所需输入格式
3. 生成测试集
4. 输出 `jsonl + summary`

第一版生成脚本必须输出：

1. `qid`
2. `question`
3. `reference`
4. `reference_contexts`
5. `reference_context_ids`
6. `reference_doc_ids`
7. `synthesizer_name`
8. `persona_name`
9. `query_style`
10. `query_length`

同时输出最小质量统计：

1. 各 synthesizer 题量
2. 空 `reference` 比例
3. 空 `reference_contexts` 比例
4. `reference_context_ids` 缺失率

### 6.4 第四步：再落地 compare 脚本

新增：

- `src/experiments/run_ragas_cuad_compare.py`

该脚本不应重写三条方法逻辑，而应尽量复用：

1. 现有 adapter
2. 现有 compare 中的参数组织方式
3. 现有对 `doc scope`、`telemetry`、`answer mode` 的约束

职责应收敛为：

1. 读取 `ragas` 测试集
2. 调用三种方法
3. 把三种输出统一归一化
4. 生成 merged predictions

统一结构建议固定为：

```json
{
  "qid": "ragas-cuad-0001",
  "method": "graph_rag",
  "question": "...",
  "response": "...",
  "reference": "...",
  "retrieved_contexts": ["..."],
  "retrieved_context_ids": ["chunk_1"],
  "retrieved_doc_ids": ["doc_1"],
  "reference_contexts": ["..."],
  "reference_context_ids": ["chunk_a"],
  "reference_doc_ids": ["doc_1"],
  "synthesizer_name": "single_hop_specific",
  "telemetry": {}
}
```

### 6.5 第五步：最后落地 eval 脚本

新增：

- `src/evaluation/run_ragas_eval.py`

职责：

1. 读取 merged predictions
2. 转为 `ragas` `EvaluationDataset`
3. 按方法跑评测
4. 输出 per-sample 与 summary

第一版推荐仅默认启用以下 4 个指标：

1. `AnswerCorrectness`
2. `Faithfulness`
3. `ContextPrecision`
4. `ContextRecall`

以下 2 个指标放为可选开关：

1. `IDBasedContextPrecision`
2. `IDBasedContextRecall`

启用前提：

1. `reference_context_ids` 大多数样本非空
2. `retrieved_context_ids` 大多数样本非空
3. 二者确认使用同一套 `chunk_id` 空间

### 6.6 第六步：最后再扩到全量

只有在 sampled corpus 上满足以下条件后，才进入全量：

1. 测试集生成稳定
2. 三种方法都能稳定返回 `response`
3. 文本级指标可稳定计算
4. 汇总产物格式稳定

---

## 7. 与现有主流程的关系

补充评测线与主流程关系必须写清楚：

1. 它是独立补充 benchmark
2. 它不是主流程替代品
3. 它不对齐主流程的 TopK 口径
4. 它不直接等价于主流程三类能力题

因此论文或报告中推荐表述为：

1. 主流程结果用于核心结论
2. `ragas` 结果用于补充验证通用 RAG 质量

不建议表述为：

1. `ragas` 直接替代主流程评测
2. `ragas` 三类 synthesizer 等同于主流程三类能力题

---

## 8. 第一版最小交付范围

第一版只要求交付以下内容：

1. sampled corpus 上的 `ragas` 测试集生成脚本
2. sampled corpus 上的三方法统一 compare 脚本
3. 基于 merged predictions 的 `ragas` 评测脚本
4. 按 `method` 的 summary
5. 按 `method + synthesizer_name` 的 summary

第一版不要求：

1. 自定义 synthesizer
2. 全量自动批跑
3. 和主流程的强映射分析
4. reject/open 双模式联动
5. 多轮参数扫描

---

## 9. 实施验收标准

### 9.1 冒烟阶段验收

满足以下条件即可通过冒烟：

1. `ragas` 测试集成功生成
2. 三种方法均可在同一题集上回答
3. merged predictions 成功产出
4. 四个文本级指标能在大多数样本上算完
5. summary 文件可稳定导出

### 9.2 第二阶段验收

满足以下条件后，再开启 `IDBased`：

1. `reference_context_ids` 缺失率可接受
2. `retrieved_context_ids` 缺失率可接受
3. 三方法 `chunk_id` 空间一致
4. `IDBased` 指标在大多数样本可计算

---

## 10. 推荐新增文件清单

第一版建议新增：

1. `src/experiments/run_ragas_cuad_generate.py`
2. `src/experiments/run_ragas_cuad_compare.py`
3. `src/evaluation/run_ragas_eval.py`
4. `docs/ragas_cuad_supplement_implementation.md`

可选新增：

1. `src/utils/ragas_converters.py`
2. `src/evaluation/export_ragas_report.py`

---

## 11. 推荐命令形态

以下命令形态仅在上述脚本补齐后成立。

### 11.1 生成测试集

```bash
python src/experiments/run_ragas_cuad_generate.py \
  --chunks-file data/processed/qa_aligned_chunks_sampled.jsonl \
  --out-testset-file data/queries/ragas_cuad_smoke_testset.jsonl \
  --out-summary-file data/queries/ragas_cuad_smoke_testset_summary.json \
  --testset-size 20 \
  --random-seed 42
```

### 11.2 跑三方法统一回答

```bash
python src/experiments/run_ragas_cuad_compare.py \
  --testset-file data/queries/ragas_cuad_smoke_testset.jsonl \
  --chunks-file data/processed/qa_aligned_chunks_sampled.jsonl \
  --graph-file outputs/graph/qa_aligned_graph_sampled.json \
  --communities-file outputs/graph/qa_aligned_communities_sampled.json \
  --lightrag-working-dir outputs/lightrag/cuad_sampled \
  --youtu-base-url http://127.0.0.1:8000 \
  --youtu-dataset cuad_sampled \
  --out-file outputs/results/ragas/ragas_cuad_smoke_compare_merged.jsonl
```

### 11.3 跑 ragas 评测

```bash
python src/evaluation/run_ragas_eval.py \
  --pred-file outputs/results/ragas/ragas_cuad_smoke_compare_merged.jsonl \
  --out-dir outputs/results/ragas/smoke
```

---

## 12. 最终建议

建议正式采用以下实施策略：

1. 保留现有主评测流程不变
2. 单独增加一条 `ragas` 补充评测线
3. 第一版先做 sampled corpus 冒烟闭环
4. 先交付文本级四指标
5. `IDBased` 指标作为第二阶段增强项
6. 实现时优先复用现有 adapter 与 compare 主干

这是当前仓库条件下最稳妥、最容易真正落地的一条实施路径。
