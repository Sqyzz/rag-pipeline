# VectorRAG / KG-RAG / GraphRAG 对比流程说明

## 1. 目标

本项目统一比较三种方法在同一查询集上的表现：

- VectorRAG
- KG-RAG
- GraphRAG

比较维度包括：答案质量、证据可追溯性、token 消耗、调用次数、延迟。

## 2. 从零到产出对比结果（端到端流程）

下面按“**从零开始**（无 processed/index/graph 产物）”的顺序梳理。你也可以跳过已完成的步骤。

### 2.1 环境与配置

1) 安装依赖：

```bash
pip install -r requirements.txt
```

2) 配置密钥（推荐写到项目根目录 `.env` 或导出环境变量）：

- LLM：`LLM_API_KEY`（或 `DASHSCOPE_API_KEY`）
- Embedding：`EMB_API_KEY`（或 `DASHSCOPE_API_KEY`）

3) 关键配置：

- `config.yaml`（`llm.*`, `embedding.*`, `retrieval.*`, `comparison.*`）
- `config_budget.yaml`（`budget_matched` 的统一预算与 method-aware 预算）

### 2.2 准备原始数据（Raw）

把 Enron 数据放到 `data/raw/enron/`，支持两种形式：

- `data/raw/enron/emails.csv`（优先）
- 或者目录下大量 `.txt`

### 2.3 生成 docs（Raw → docs.jsonl）

```bash
python src/ingestion/load_docs.py
```

产物：`data/processed/enron_docs.jsonl`

（可选）去重 + 抽样（更利于快速实验）：

```bash
python src/ingestion/sample_dedup.py --sample-size 2000
```

产物：`data/processed/enron_docs_sampled.jsonl`

### 2.4 分块 + 建向量索引（docs → chunks → FAISS）

推荐直接用 `run_baselines.py` 的 build-only 模式（它会用 chunk_size=1600, overlap=200，并构建 FAISS 索引）：

```bash
python src/experiments/run_baselines.py --mode build_only
```

根据你是否存在 `enron_docs_sampled.jsonl`，它会走两条路径之一：

- **sampled 路径（默认优先）**：
  - `data/processed/chunks_sampled.jsonl`
  - `outputs/indexes/faiss_sampled.idx`
  - `outputs/indexes/chunk_store_sampled.json`
- **full 路径**（无 sampled docs 时）：
  - `data/processed/chunks.jsonl`
  - `outputs/indexes/faiss.idx`
  - `outputs/indexes/chunk_store.json`

> 注意：对比实验的 `--chunks-file/--idx-file/--store-file` 需要保持同一套（sampled 或 full），否则证据映射会错位。

### 2.5 构建共享图资产（chunks → triples → graph → communities）

对比入口会在图资产缺失时自动构建（见 2.6）。你也可以手动分步执行：

```bash
# 1) LLM 三元组抽取（推荐批量模式降低调用次数）
python src/graph_build/extract_triples.py \
  --chunks-file data/processed/chunks_sampled.jsonl \
  --out-file outputs/graph/triples.jsonl \
  --mode batched \
  --batch-size 4 \
  --concurrency 4 \
  --schema-file config_triple_schema.json \
  --min-chars 80 \
  --progress-every 50

# 2) triples → graph.json（带 edge_id/mentions/weight）
python src/graph_build/build_graph.py \
  --triples-file outputs/graph/triples.jsonl \
  --out-file outputs/graph/graph.json

# 3) Leiden 社区划分（含层级）+ 社区摘要（GraphRAG 用）
python src/graph_build/build_communities.py \
  --graph-file outputs/graph/graph.json \
  --out-file outputs/graph/communities.json \
  --resolutions 0.6,1.0,1.6 \
  --summary-level-max 0 \
  --summary-min-size 20 \
  --summary-top-per-level 200
```

这三步现在都带实时进度日志：

- `extract_triples.py`：按 chunk 周期输出（可用 `--progress-every` 调整频率），支持 `--mode batched` + `--concurrency` 降低耗时，支持 `--schema-file` 约束实体类型与关系集合
- `build_graph.py`：输出输入规模、去重边数、节点数、完成状态
- `build_communities.py`：输出 Leiden 各层进度、摘要策略与摘要完成计数（支持只预摘要高层/大社区）

其中 `build_communities.py` 输出：

- `algorithm=leiden`
- `is_hierarchical=true`
- `levels`（每层社区列表，coarse→fine）
- `communities`（每个社区包含 `level/parent_id/children_ids/summary`）

### 2.6 构建 Gold QA 与 Query 集（QA Builder）

先确保图资产和 chunk store 已准备完成（见 2.4/2.5），再执行：

```bash
python src/evaluation/qa_builder.py \
  --graph-file outputs/graph/graph.json \
  --communities-file outputs/graph/communities.json \
  --chunk-store-file outputs/indexes/chunk_store_sampled.json \
  --out-gold data/queries/gold_qa.jsonl \
  --out-queries data/queries/queries.jsonl \
  --out-gold-answer data/queries/gold.jsonl \
  --n_local 20 \
  --n_cross 15 \
  --n_global 15 \
  --n_trace 10 \
  --qa-community-level 0
```

产物：

- `data/queries/gold_qa.jsonl`：完整 gold QA（answer + supporting_chunks/edges/communities）
- `data/queries/queries.jsonl`：仅 `qid/type/query`，可直接喂 `run_compare.py`
- `data/queries/gold.jsonl`：仅 `qid/answer`，可用于 `src/evaluation/run_eval.py` 的 answer-only 评估模式

QA Builder 不调用 LLM，采用规则化构造（边、路径、社区摘要）并做证据对齐校验。

### 2.7 运行三方法对比（同一 queries）

```bash
python src/experiments/run_compare.py \
  --queries-file data/queries/gold_qa.jsonl \
  --chunks-file data/processed/chunks_sampled.jsonl \
  --idx-file outputs/indexes/faiss_sampled.idx \
  --store-file outputs/indexes/chunk_store_sampled.json \
  --triples-file outputs/graph/triples.jsonl \
  --graph-file outputs/graph/graph.json \
  --communities-file outputs/graph/communities.json \
  --regimes both \
  --budget-config-file config_budget.yaml \
  --out-file outputs/results/compare_answers.jsonl \
  --metrics-file outputs/results/compare_metrics.json
```

说明：

- `run_compare.py` 当前预算相关 CLI 只保留 `--budget-config-file`
- `budget_matched` 预算与方法约束从 `config_budget.yaml` 读取
- 运行中会打印状态日志（query/regime/method 进度）

产物：

- `outputs/results/compare_answers.jsonl`：每个 query 在 `best_effort` 与 `budget_matched` 两种 regime 下的三方法答案、证据、预算检查与明细 telemetry
- `outputs/results/compare_metrics.json`：分 regime 聚合指标（调用次数、token、延迟 p50/p95）、按 `type` 分层指标、图与索引构建成本

### 2.8 （可选）快速跑通建议

LLM 三元组抽取是最耗时/最贵的一步。建议先用更小的 chunks 集（例如仓库已有 `data/processed/chunks_sampled_1000.jsonl`）跑通流程：

- 先用该 chunks 建索引（需保证 index 与 chunks 对齐）
- 对比脚本的 `--chunks-file/--idx-file/--store-file` 全部切换到同一套“小样本”文件

### 2.9 评估对比结果（run_eval）

推荐使用完整 gold（含证据）评估：

```bash
python src/evaluation/run_eval.py \
  --pred-file outputs/results/compare_answers.jsonl \
  --gold-file data/queries/gold_qa.jsonl \
  --out-csv outputs/results/eval_compare.csv \
  --out-summary outputs/results/eval_compare_summary.json
```

产物：

- `outputs/results/eval_compare.csv`：逐条 `qid × regime × method` 明细（答案相似度 + 证据匹配 + budget 状态）
- `outputs/results/eval_compare_summary.json`：按 `regime × method` 与 `type` 聚合统计

若仅做答案文本评估（不含证据指标），可改用：

```bash
python src/evaluation/run_eval.py \
  --pred-file outputs/results/compare_answers.jsonl \
  --gold-file data/queries/gold.jsonl \
  --out-csv outputs/results/eval_compare_answer_only.csv \
  --out-summary outputs/results/eval_compare_answer_only_summary.json
```

### 2.10 图结构指标统计（Graph Structure Metrics）

运行：

```bash
python src/evaluation/graph_structure_metrics.py \
  --graph-file outputs/graph/graph.json \
  --communities-file outputs/graph/communities.json \
  --out-json outputs/results/graph_structure_metrics.json \
  --out-dir outputs/results/graph_plots
```

必含指标：

- 平均度（`average_degree`）
- 最大连通分量比例（`largest_connected_component_ratio`）
- 聚类系数（`clustering_coefficient_global` / `clustering_coefficient_avg_local`）
- 边权重分布统计（`weight_stats`）
- 度为 1 的节点比例（`degree_one_ratio`）

图表输出：

- 社区规模分布直方图：`outputs/results/graph_plots/community_size_distribution.png`
- weight 分布直方图：`outputs/results/graph_plots/edge_weight_distribution.png`

说明：

- `run_compare.py` 在写 `compare_metrics.json` 时会自动尝试计算该指标，并写入 `graph_structure_metrics` 字段（失败不阻塞主流程）。

## 3. 统一数据流（概念视图）

1) `data/processed/chunks*.jsonl` 生成后，VectorRAG 构建向量索引（FAISS）。
2) 同一份 `chunks` 使用 LLM 进行三元组抽取，产出共享图资产：
   - `outputs/graph/triples.jsonl`
   - `outputs/graph/graph.json`
   - `outputs/graph/communities.json`
3) 对 `data/queries/queries.jsonl` 的每条 query，在两种 regime 下同时运行：
   - `best_effort`：各方法完整能力上限
   - `budget_matched`：统一 query-time 预算（token/calls）约束
4) 每个 regime 下运行：
   - VectorRAG（FAISS 检索）
   - KG-RAG（entity linking + 1~3 hop traversal + chunk 原文回填）
   - GraphRAG（社区摘要 map-reduce，query-time 不遍历 triples）
5) 统一输出到：
   - `outputs/results/compare_answers.jsonl`
   - `outputs/results/compare_metrics.json`

## 4. 关键实现模块

- 统一对比入口：`src/experiments/run_compare.py`
- VectorRAG 证据化检索：`src/baselines/vector_rag.py` (`retrieve_with_evidence`)
- KG-RAG：`src/baselines/kg_rag.py`
- GraphRAG：`src/baselines/graph_rag.py`
- 共享图构建：
  - `src/graph_build/extract_triples.py`（LLM 抽取）
  - `src/graph_build/build_graph.py`
  - `src/graph_build/build_communities.py`（Leiden + hierarchy）
- 统计工具：
  - `src/utils/telemetry.py`
  - `src/utils/llm_wrapper.py`（usage/latency 回传）
  - `src/utils/embedder.py`（embedding usage/latency 回传）
  - `src/utils/tokenizer.py`（统一 token 计数）
  - `src/utils/budget.py`（BudgetManager）
- QA 构建：
  - `src/evaluation/qa_builder.py`

## 5. 输出格式（核心字段）

`compare_answers.jsonl` 每条记录包含：

- `qid`, `type`, `query`
- `regimes.best_effort.vector_rag / kg_rag / graph_rag`
- `regimes.budget_matched.vector_rag / kg_rag / graph_rag`
- `budget_matched` 下每方法附带 `budget_check`（含 manager/error）
- GraphRAG 若触发预算收缩，会包含 `budget_adaptation`

`compare_metrics.json` 包含：

- 全部 query 数量
- 分 regime 的三种方法聚合指标：
  - `llm_calls`
  - `embedding_calls`
  - `prompt_tokens`, `completion_tokens`, `total_tokens`
  - `llm_latency_ms`, `embedding_latency_ms`
  - `latency_ms.p50 / p95`
- `aggregate_metrics_by_type`：按 query `type` 分层汇总
- `indexing_metrics`：`vector_index / triple_extract / graph_build / community_build`

## 6. 注意事项 / 常见坑

- 首次运行会触发 LLM 三元组抽取和社区摘要，耗时较长。
- 如果 API 返回不含 `usage`，token 统计会记为 0（请在论文中注明供应商返回限制）。
- GraphRAG 当前严格采用“社区摘要 map-reduce”策略：query-time 不直接拼接 triples/edges。
- KG-RAG 当前采用实体链接 + 多跳遍历，并使用统一 embedding 对 traversal 证据进行语义重排。
- `budget_matched` 下 GraphRAG 如首次超预算，会触发一次 adaptive shrink 重跑，因此日志里会看到该方法再次出现 embedding 进度条。
- `run_compare.py` **不会自动构建 FAISS 索引**：请先执行 `run_baselines.py --mode build_only`，或手动用 `vector_rag.build_index(...)` 生成 `idx/store`。
- 如果 `qa_builder.py` 输出中 `graph_chunk_mention_coverage` 很低（例如接近 0），通常是 `chunk_store` 与 `graph/triples` 不是同一套 sampled/full 资产，需重建或切换到一致文件。
