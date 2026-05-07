# 对比 Pipeline（重构版）

## 1. 文档目标

本文档定义当前项目的统一对比流程，用于稳定复现以下目标：

1. 多方法统一对比：`VectorRAG / KG-RAG / GraphRAG / youtu-GraphRAG`
   - 可选扩展：`LightRAG`
2. 双回答策略：`reject / open`
3. 双 Top-K 评估：`Top10 / Top20`
4. 成本统计：`Tokens / Time`
5. 最终产物：
   - 表格：`Method | Dataset | Mode | Top20 | Top10 | Tokens | Time`
   - Pareto 图：`Tokens vs Accuracy`

如需只比较 `youtu-GraphRAG` 与 `LightRAG`，请使用独立流程文档：

- `docs/comparison_pipeline_youtu_lightrag.md`

---

## 2. 统一评估口径（必须遵守）

### 2.1 回答策略（Mode）

- `reject`：证据不足返回 `NOT_FOUND`
- `open`：证据不足允许外部知识补全（提示词要求前缀 `OUTSIDE_EVIDENCE:`）

### 2.2 TopK Accuracy 定义

统一使用：

`TopK_Accuracy = mean( I[answer_correct = 1 AND support_at_k = 1] )`

其中：

- `support_at_k`：`evidence_recall_primary > 0` 记为 1，否则 0
- 当前默认主召回口径：`evidence_recall_chunks`
- `answer_correct`：优先 `answer_semantic_yesno`（启用 Judge 时），否则 `answer_exact_relaxed`

明细列：

- `topk_support`
- `topk_correct`
- `topk_accuracy`

### 2.3 global_summary 主分口径

`run_eval.py` 的 `global_summary` 复合主分为：

- `composite(summary_semantic_similarity, evidence_recall_primary)`

### 2.4 多模式/多 Top-K 去重口径

去重键包含：

- `qid + regime + method + mode + top_k`

### 2.5 CUAD 检索作用域（新增）

- `run_compare.py` / `run_compare_topk_modes.py` 新增 `--cuad-doc-scope true|false`
- 默认 `false`：不做 CUAD 单文档过滤（全库检索）
- 若设为 `true`，`strict_doc_scope` 才会生效（控制文档过滤失败时是否回退全库）

---

## 3. 依赖与前置条件

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 配置环境变量（或 `.env`）：

- `LLM_API_KEY`（或 `DASHSCOPE_API_KEY`）
- `EMB_API_KEY`（或 `DASHSCOPE_API_KEY`）

3. 关键配置文件：

- `config.yaml`
- `config_budget.yaml`

---

## 4. run_compare 前的数据准备（必需）

`run_compare.py` 依赖以下输入资产，缺一不可：

- `queries_file`（问题集）
- `chunks_file`
- `idx_file`（FAISS）
- `store_file`（chunk store）
- `triples_file`
- `graph_file`
- `communities_file`
- `gold_file`（用于 run_eval）

下面给出推荐的标准准备流程。

### 4.0 步骤-产物速查

1. `load_docs.py`：
   - 产物：`data/processed/*_docs.jsonl`
2. `sample_dedup.py`（可选）：
   - 产物：`data/processed/*_docs_sampled.jsonl`
   - 可选产物：`outputs/results/*_sample_stats.json`（`--stats-output` 指定）
3. `run_baselines.py --mode build_only`：
   - 产物：`data/processed/*_chunks.jsonl`
   - 产物：`outputs/indexes/faiss*.idx`
   - 产物：`outputs/indexes/chunk_store*.json`
4. `extract_triples.py / build_graph.py / build_communities.py`：
   - 产物：`outputs/graph/*_triples*.jsonl`
   - 产物：`outputs/graph/*_graph*.json`
   - 产物：`outputs/graph/*_communities*.json`
5. QA 构建：
   - 产物：`data/queries/*_queries*.jsonl`
   - 产物：`data/queries/*_gold*.jsonl`
6. compare + eval：
   - 产物：`outputs/results/compare_answers_*.jsonl`
   - 产物：`outputs/results/compare_metrics_*.json`
   - 产物：`outputs/results/eval_*.csv`
   - 产物：`outputs/results/eval_*_summary.json`
   - 产物：`outputs/results/topk_mode_summary.{csv,md,json}`
7. report 导出：
   - 产物：`outputs/results/final_report.{csv,md}`
   - 产物：`outputs/results/pareto_frontier.png`

### 4.1 Raw -> docs.jsonl

#### Enron

```bash
python src/ingestion/load_docs.py \
  --dataset enron \
  --raw-dir data/raw/enron \
  --out-file data/processed/enron_docs.jsonl
```

本步产物：

- `data/processed/enron_docs.jsonl`

#### CUAD

```bash
python src/ingestion/load_docs.py \
  --dataset cuad \
  --raw-file data/raw/cuad/test.json \
  --out-file data/processed/cuad_docs.jsonl \
  --split-name test
```

本步产物：

- `data/processed/cuad_docs.jsonl`

#### CUAD（可选：按 QA 文档集合对齐 docs）

当评测问题集来自 `data/queries/cuad_capability_queries.jsonl`
这类“只覆盖部分 CUAD 文档”的 QA 子集时，推荐先从
`data/raw/cuad/CUADv1.json` 反筛一份与 QA 文档集合对齐的 `docs.jsonl`，
再继续自行切 chunk 和构图。这样可以避免“QA 所属文档不在图库中”导致的检索失配。

下面的命令会：

- 读取 `CUADv1.json` 全量文档
- 从 QA 文件中提取 `meta.query_doc_key`（缺失时回退 `meta.title`）
- 保留全部与 QA 对应的文档
- 再按 `--matched-doc-ratio` 补采不对应文档
- 输出与 `load_cuad()` 相同格式的 `docs.jsonl`

示例：输出文档集中约 `10%` 为 QA 对应文档，`90%` 为其他文档：

```bash
python src/ingestion/build_cuad_aligned_docs.py \
  --cuad-file data/raw/cuad/CUADv1.json \
  --qa-file data/queries/cuad_converted_queries_generated.jsonl \
  --out-docs-file data/processed/cuad_docs_capability_aligned.jsonl \
  --matched-doc-ratio 0.3 \
  --random-seed 42 \
  --split-name capability_aligned
```

本步产物：

- `data/processed/cuad_docs_capability_aligned.jsonl`

说明：

- `--matched-doc-ratio` 表示输出文档集中“与 QA 对应文档”的目标占比。
- 当前实现会保留全部 matched 文档，再按比例补采 unmatched 文档；若 unmatched 不足，则按可用数量补齐，并在脚本输出统计中给出实际比例。
- 后续若使用该产物继续构图，请将后面的 chunk 构建输入切换为这份新 docs 文件。

### 4.2（可选）去重与抽样

```bash
python src/ingestion/sample_dedup.py \
  --input data/processed/enron_docs.jsonl \
  --output data/processed/enron_docs_sampled.jsonl \
  --sample-size 2000 \
  --stats-output outputs/results/enron_sample_stats.json
```

本步产物：

- `data/processed/enron_docs_sampled.jsonl`
- `outputs/results/enron_sample_stats.json`（若传 `--stats-output`）

### 4.3 docs -> chunks + index/store

推荐使用 `run_baselines.py --mode build_only` 快速构建：

#### CUAD

```bash
python src/experiments/run_baselines.py \
  --dataset cuad \
  --mode build_only \
  --cuad-raw-file data/processed/qa_aligned_docs.jsonl \
  --cuad-split-name test
```

本步产物（CUAD）：

- `data/processed/cuad_chunks.jsonl`
- `outputs/indexes/faiss_cuad.idx`
- `outputs/indexes/chunk_store_cuad.json`

#### Enron

```bash
python src/experiments/run_baselines.py \
  --dataset enron \
  --mode build_only
```

本步产物（Enron）：

- `data/processed/chunks_sampled.jsonl`（或 `chunks.jsonl`）
- `outputs/indexes/faiss_sampled.idx`（或 `faiss.idx`）
- `outputs/indexes/chunk_store_sampled.json`（或 `chunk_store.json`）

### 4.4 chunks -> triples -> graph -> communities

```bash
python src/graph_build/extract_triples.py \
  --chunks-file data/processed/cuad_chunks.jsonl \
  --out-file outputs/graph/cuad_triples_test.jsonl \
  --mode batched \
  --batch-size 5 \
  --concurrency 10 \
  --schema-file config_triple_schema_cuad_core_v2.json \
  --schema-apply-mode strict \
  --progress-every 50

python src/graph_build/build_graph.py \
  --triples-file outputs/graph/cuad_triples_test.jsonl \
  --out-file outputs/graph/cuad_graph_test.json \
  --edge-merge-mode global \
  --node-merge-mode normalized \
  --node-scope-with-type true

python src/graph_build/build_communities.py \
  --graph-file outputs/graph/cuad_graph_test.json \
  --out-file outputs/graph/cuad_communities_test.json \
  --resolutions 0.6,1.0,1.6 \
  --min-parent-overlap 0.8 \
  --summary-level-max 1 \
  --summary-min-size 15 \
  --summary-top-per-level 200 \
  --summary-min-per-level 10
```

本步产物：

- `outputs/graph/cuad_triples_test.jsonl`
- `outputs/graph/cuad_graph_test.json`
- `outputs/graph/cuad_communities_test.json`

### 4.5 构建 queries/gold

#### 方案 A：CUAD 能力导向（推荐做方法对比）

```bash
python src/evaluation/build_cuad_capability_qa.py \
  --graph-file outputs/graph/cuad_graph_test.json \
  --communities-file outputs/graph/cuad_communities_test.json \
  --queries-out data/queries/cuad_capability_queries.jsonl \
  --gold-out data/queries/cuad_capability_gold.jsonl \
  --per-type 20 \
  --random-seed 42 \
  --question-style llm \
  --llm-question-model deepseek-chat \
  --llm-question-temperature 0 \
  --llm-cross-generate-qa \
  --llm-cross-answer-max-tokens 200 \
  --llm-question-max-tokens 200 \
  --progress-every 1
```

本步产物：

- `data/queries/cuad_capability_queries.jsonl`
- `data/queries/cuad_capability_gold.jsonl`

#### 方案 B：CUAD 模板转换为三类 QA（按总量采样）

当你希望直接从 `data/raw/cuad/train_separate_questions.json` 生成三类题
（`local_factual / cross_clause / global_summary`）时，使用：

```bash
python src/evaluation/build_cuad_question_converter.py \
  --cuad-train-file data/raw/cuad/train_separate_questions.json \
  --mode llm \
  --llm-types global_synthesis \
  --total-per-type 50 \
  --answer-mode weak_reference \
  --type-name-scheme capability \
  --progress-every-qa 20 \
  --out-queries-file data/queries/cuad_converted_queries_generated.jsonl \
  --out-gold-file data/queries/cuad_converted_gold_generated.jsonl
```

说明：

- `--total-per-type`：全局每类总题数（随机采样文档生成），不是“每文档数量”。
- `--llm-types global_synthesis`：仅 global 走 LLM；local/cross 使用规则模板。
- `--answer-mode weak_reference`：输出弱参考答案；如需占位答案可改为 `placeholder`。
- `--type-name-scheme capability`：输出类型名对齐现有评测：`local_factual/cross_clause/global_summary`。
- `--progress-every-qa`：按已生成 QA 数量打印进度日志（`[qa_progress] ...`）。

本步产物（方案 B）：

- `data/queries/cuad_converted_queries_generated.jsonl`
- `data/queries/cuad_converted_gold_generated.jsonl`

说明：

- `--question-style llm` 启用 LLM 生成自然问句；生成失败会自动回退模板问句。
- `--llm-cross-generate-qa` 启用 `cross_clause` 的 LLM 问题+答案联合生成；失败会回退规则答案。
- 如需纯模板问句，改为 `--question-style template`（默认值）。

可选：使用 LLM 基于 `supporting_chunks` 重写 `global_summary` 的 gold 答案

```bash
python src/evaluation/rewrite_global_summary_gold.py \
  --gold-file data/queries/cuad_capability_gold.jsonl \
  --chunks-file data/processed/cuad_chunks.jsonl \
  --out-file data/queries/cuad_capability_gold_docsummary.jsonl \
  --max-chunks 8 \
  --max-chunk-chars 2200 \
  --max-completion-tokens 320 \
  --progress-every 1
```

本步产物：

- `data/queries/cuad_capability_gold_docsummary.jsonl`

说明：

- 该步骤仅重写 `type=global_summary` 的答案，其它类型保持不变。
- 执行评测时，将 `--gold-file` 切换为 `data/queries/cuad_capability_gold_docsummary.jsonl`。

#### 方案 C：CUAD 原生问法

```bash
python src/ingestion/build_cuad_qa.py \
  --raw-file data/raw/cuad/test.json \
  --queries-out data/queries/cuad_queries_test.jsonl \
  --gold-out data/queries/cuad_gold_test.jsonl \
  --split-name test \
  --store-file outputs/indexes/chunk_store_cuad.json
```

本步产物：

- `data/queries/cuad_queries_test.jsonl`
- `data/queries/cuad_gold_test.jsonl`

#### 方案 C：Enron 对齐 QA

```bash
python src/evaluation/qa_builder.py \
  --graph-file outputs/graph/graph.json \
  --communities-file outputs/graph/communities.json \
  --chunk-store-file outputs/indexes/chunk_store_sampled.json \
  --out-gold data/queries/gold_qa_aligned.jsonl \
  --out-queries data/queries/queries_aligned.jsonl \
  --out-gold-answer data/queries/gold_aligned.jsonl \
  --n_local 20 \
  --n_cross 20 \
  --n_global 20 \
  --n_trace 20
```

本步产物：

- `data/queries/queries_aligned.jsonl`
- `data/queries/gold_qa_aligned.jsonl`
- `data/queries/gold_aligned.jsonl`

---

## 5. 最小复现（已有资产时）

如果第 4 章资产已齐全，可直接执行：

### 5.1 一键跑 Top10/Top20 × reject/open

```bash
python src/experiments/run_compare_topk_modes.py \
  --queries-file data/queries/cuad_capability_queries.jsonl \
  --gold-file data/queries/cuad_capability_gold_docsummary.jsonl \
  --chunks-file data/processed/cuad_chunks.jsonl \
  --idx-file outputs/indexes/faiss_cuad.idx \
  --store-file outputs/indexes/chunk_store_cuad.json \
  --triples-file outputs/graph/cuad_triples_test.jsonl \
  --graph-file outputs/graph/cuad_graph_test.json \
  --communities-file outputs/graph/cuad_communities_test.json \
  --topk-list 10 \
  --answer-modes reject \
  --regime best_effort \
  --cuad-doc-scope false \
  --strict-doc-scope false \
  --judge-mode llm_yesno \
  --judge-model deepseek-chat \
  --method-mode all \
  --include-lightrag \
  --include-youtu \
  --youtu-client-id web_client \
  --youtu-dataset cuad \
  --youtu-corpus-source-file data/processed/cuad_chunks.jsonl \
  --youtu-sync-mode none \
  --youtu-force-rebuild false \
  --youtu-require-fingerprint-match false \
  --youtu-graph-state-file outputs/graph/youtu_graph_state_cuad.json \
  --max-queries 10
```

本步产物（每个 `topk × mode` 组合各一套）：

- `outputs/results/compare_answers_top{K}_{mode}.jsonl`
- `outputs/results/compare_metrics_top{K}_{mode}.json`
- `outputs/results/eval_top{K}_{mode}.csv`
- `outputs/results/eval_top{K}_{mode}_summary.json`

本步汇总产物：

- `outputs/results/topk_mode_summary.csv`
- `outputs/results/topk_mode_summary.md`
- `outputs/results/topk_mode_summary.json`

### 5.2 导出最终报告 + Pareto

```bash
python src/evaluation/export_report.py \
  --in-csv outputs/results/topk_mode_summary.csv \
  --out-csv outputs/results/final_report.csv \
  --out-md outputs/results/final_report.md \
  --pareto-out outputs/results/pareto_frontier.png \
  --accuracy-col top10_accuracy
```

本步产物：

- `outputs/results/final_report.csv`
- `outputs/results/final_report.md`
- `outputs/results/pareto_frontier.png`

---

## 6. 分步执行（调试用）

### 6.1 单次 compare（单模式）

```bash
python src/experiments/run_compare.py \
  --queries-file data/queries/cuad_capability_queries.jsonl \
  --chunks-file data/processed/cuad_chunks.jsonl \
  --idx-file outputs/indexes/faiss_cuad.idx \
  --store-file outputs/indexes/chunk_store_cuad.json \
  --triples-file outputs/graph/cuad_triples_test.jsonl \
  --graph-file outputs/graph/cuad_graph_test.json \
  --communities-file outputs/graph/cuad_communities_test.json \
  --top-k 20 \
  --answer-mode reject \
  --cuad-doc-scope false \
  --strict-doc-scope false \
  --include-lightrag \
  --include-youtu \
  --youtu-client-id web_client \
  --youtu-route-type local \
  --youtu-corpus-source-file data/processed/cuad_chunks.jsonl \
  --youtu-sync-mode shared_dir \
  --youtu-force-rebuild true \
  --youtu-require-fingerprint-match false \
  --out-file outputs/results/compare_answers_top10_reject.jsonl \
  --metrics-file outputs/results/compare_metrics_top10_reject.json \
  --max-queries 10
```

本步产物：

- `outputs/results/compare_answers_top10_reject.jsonl`
- `outputs/results/compare_metrics_top10_reject.json`

### 6.2 单次 compare（双模式）

```bash
python src/experiments/run_compare.py \
  --queries-file data/queries/cuad_capability_queries.jsonl \
  --chunks-file data/processed/cuad_chunks.jsonl \
  --idx-file outputs/indexes/faiss_cuad.idx \
  --store-file outputs/indexes/chunk_store_cuad.json \
  --triples-file outputs/graph/cuad_triples_test.jsonl \
  --graph-file outputs/graph/cuad_graph_test.json \
  --communities-file outputs/graph/cuad_communities_test.json \
  --top-k 10 \
  --answer-modes reject,open \
  --cuad-doc-scope false \
  --strict-doc-scope false \
  --include-lightrag \
  --include-youtu \
  --youtu-client-id web_client \
  --youtu-route-type local \
  --youtu-corpus-source-file data/processed/cuad_chunks.jsonl \
  --youtu-sync-mode none \
  --youtu-force-rebuild false \
  --youtu-require-fingerprint-match false \
  --out-file outputs/results/compare_answers_top10_both_modes.jsonl \
  --metrics-file outputs/results/compare_metrics_top10_both_modes.json
```

本步产物：

- `outputs/results/compare_answers_top10_both_modes.jsonl`
- `outputs/results/compare_metrics_top10_both_modes.json`

### 6.3 run_eval

```bash
python src/evaluation/run_eval.py \
  --pred-file outputs/results/compare_answers_top10_reject.jsonl \
  --gold-file data/queries/cuad_capability_gold.jsonl \
  --out-csv outputs/results/eval_top10_reject.csv \
  --out-summary outputs/results/eval_top10_reject_summary.json \
  --judge-mode llm_yesno \
  --judge-model deepseek-chat
```

本步产物：

- `outputs/results/eval_top10_reject.csv`
- `outputs/results/eval_top10_reject_summary.json`

可选：

- `--method-mode all|exclude_youtu|only_youtu`
- `--disable-semantic-similarity`
- `--global-summary-primary-mode semantic|composite`

---

## 7. 输出文件说明

### 7.1 compare

- `compare_answers*.jsonl`：逐 query（含 `mode`、`top_k`）
- `compare_metrics*.json`：聚合（含 `answer_mode` / `answer_modes`）

### 7.2 eval

- `eval_*.csv`：明细
- `eval_*_summary.json`：汇总

关键列：

- 匹配：`answer_exact_relaxed`, `answer_semantic_yesno`
- 支持：`evidence_recall_primary`, `evidence_recall_chunks`, `evidence_recall_docs`, `topk_support`, `topk_correct`, `topk_accuracy`
- 成本：`total_tokens`, `latency_ms_total`
- 维度：`regime`, `method`, `mode`, `top_k`, `type`

### 7.3 topk 汇总

- `outputs/results/topk_mode_summary.csv`
- `outputs/results/topk_mode_summary.md`
- `outputs/results/topk_mode_summary.json`

核心列：

- `method, dataset, mode, top10_accuracy, top20_accuracy, tokens, time`

### 7.4 最终报告

- `outputs/results/final_report.csv`
- `outputs/results/final_report.md`
- `outputs/results/pareto_frontier.png`

---

## 8. 配置治理建议

`config.yaml` 建议维护 `evaluation` 段：

```yaml
evaluation:
  answer_modes: ["reject", "open"]
  topk_list: [10, 20]
  include_youtu: true
  method_mode: all
  judge:
    enabled: false
    model: qwen-flash
```

建议每次实验显式记录：

- `answer_mode(s)`
- `top_k / topk_list`
- `judge_mode / judge_model`
- `include_youtu / method_mode`

---

## 9. youtu 对齐说明

开启 youtu 分支后，方法 payload 额外包含：

- `answer_mode_requested`
- `answer_mode`（effective）
- `answer_mode_backend_supported`

用于区分“请求模式”和“后端实际支持状态”。

当前 `ask-question` 为异步两段式：

1. `POST /api/ask-question` 提交任务，返回 `task_id` 与 `status=queued|running`。
2. `GET /api/ask-question/{task_id}` 轮询任务，`status=completed` 时答案在 `data` 字段。

推荐同时显式传入以下参数：

- `--youtu-client-id web_client`（对应 `POST /api/ask-question?client_id=...`）
- `--youtu-route-type <backend定义值>`（如 `local|structural|global`，以及后端支持的别名）

为避免误触发后端重构图，建议默认使用：

- `--youtu-corpus-source-file data/processed/cuad_chunks.jsonl`
- `--youtu-sync-mode none`
- `--youtu-force-rebuild false`
- `--youtu-require-fingerprint-match false`

---

## 10. 常见问题

1. 结果被覆盖：确认使用包含 `mode/top_k` 去重的新版评估流程。
2. TopK Accuracy 偏低：先看 `evidence_recall_primary`（默认 chunks 口径），再看 `answer_correct` 来源。
3. global_summary 分数异常：确认使用 `composite(summary_semantic_similarity, evidence_recall_primary)`。
4. token 统计为 0：通常是 API 未返回完整 `usage`。
5. matplotlib 缓存告警：可设置 `export MPLCONFIGDIR=/tmp/mplcache`。

---

## 11. 论文产出推荐顺序

1. 固定资产（chunks/index/graph/communities）
2. 跑 `run_compare_topk_modes.py`（Top10/Top20 × reject/open）
3. 跑 `export_report.py` 导出最终表与 Pareto
4. 在论文中明确：
   - `TopK_Accuracy` 公式
   - `answer_correct` 选择规则
   - `support_at_k` 定义
