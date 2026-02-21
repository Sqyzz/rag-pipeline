# VectorRAG / KG-RAG / GraphRAG 对比流程说明（当前实现）

## 1. 目标与当前事实

本项目对比三种方法：

- VectorRAG
- KG-RAG
- GraphRAG（已切换为 `youtu-graphrag` 后端）

比较维度：答案质量、证据可追溯性、token 消耗、调用次数、延迟。

当前代码中的关键事实：

1. `kg_rag` 仍基于主仓本地图资产 `outputs/graph/graph.json` 检索。
2. `graph_rag` 通过 HTTP 调用 `youtu-graphrag`（`/api/construct-graph` + `/api/v1/datasets/{dataset}/search`）。
3. `graph_rag` 最终会把 youtu 检索结果映射回主仓 `edge_id/community_id/chunk_id` 空间用于评测。
4. 因此现在是“同源 chunks，双图并存”，不是“KG/Graph 单图共用”。

---

## 2. 从零到可对比（端到端）

### 2.1 安装与配置

1) 安装主仓依赖：

```bash
pip install -r requirements.txt
```

2) 配置密钥（`.env` 或环境变量）：

- LLM：`LLM_API_KEY`（或 `DASHSCOPE_API_KEY`）
- Embedding：`EMB_API_KEY`（或 `DASHSCOPE_API_KEY`）

3) 关键配置文件：

- `config.yaml`
- `config_budget.yaml`

4) youtu 相关配置（`config.yaml`）：

```yaml
graph:
  youtu:
    base_url: "http://127.0.0.1:8080"
    dataset: "enterprise"
    timeout_sec: 120
    construct_poll_sec: 2
    construct_timeout_sec: 1800
    require_id_alignment: true
    chunks_file: "data/processed/chunks_sampled.jsonl"
    graph_dir: "youtu-graphrag/output/graphs"
```

说明：

- `run_compare.py` 会把 CLI 的 `--chunks-file` 注入环境变量 `GRAPH_RAG_CHUNKS_FILE`，优先于 `graph.youtu.chunks_file`。

### 2.2 启动 youtu 服务

在 `youtu-graphrag` 目录启动后端（默认监听 `8080`）：

```bash
python backend.py
```

健康检查：

```bash
curl http://127.0.0.1:8080/api/status
```

### 2.3 数据准备（Raw → docs）

把 Enron 原始数据放入 `data/raw/enron/`（`emails.csv` 或 `.txt`）。

```bash
python src/ingestion/load_docs.py
```

产物：`data/processed/enron_docs.jsonl`

可选抽样：

```bash
python src/ingestion/sample_dedup.py --sample-size 2000
```

产物：`data/processed/enron_docs_sampled.jsonl`

### 2.4 分块 + 向量索引（VectorRAG资产）

```bash
python src/experiments/run_baselines.py --mode build_only
```

常用 sampled 产物：

- `data/processed/chunks_sampled.jsonl`
- `outputs/indexes/faiss_sampled.idx`
- `outputs/indexes/chunk_store_sampled.json`

### 2.5 本地图资产（KG-RAG + 评测映射资产）

`run_compare.py` 会在缺失时自动构建，也可手动执行：

```bash
python src/graph_build/extract_triples.py \
  --chunks-file data/processed/chunks_sampled.jsonl \
  --out-file outputs/graph/triples.jsonl \
  --mode batched \
  --batch-size 4 \
  --concurrency 4 \
  --schema-file config_triple_schema.json \
  --min-chars 80

python src/graph_build/build_graph.py \
  --triples-file outputs/graph/triples.jsonl \
  --out-file outputs/graph/graph.json

python src/graph_build/build_communities.py \
  --graph-file outputs/graph/graph.json \
  --out-file outputs/graph/communities.json
```

这些资产在当前实现中用于：

1. KG-RAG 检索链路（直接使用）
2. GraphRAG 结果映射（`triple -> edge_id`、`youtu community -> community_id`）
3. `run_eval` 证据指标统计

### 2.6 QA/Query 构建

```bash
python src/evaluation/qa_builder.py \
  --graph-file outputs/graph/graph.json \
  --communities-file outputs/graph/communities.json \
  --chunk-store-file outputs/indexes/chunk_store_sampled.json \
  --out-gold data/queries/gold_qa.jsonl \
  --out-queries data/queries/queries.jsonl \
  --out-gold-answer data/queries/gold.jsonl
```

### 2.7 运行对比

```bash
python src/experiments/run_compare.py \
  --queries-file data/queries/queries.jsonl \
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

运行时实际行为：

1. 先 `ensure_graph_assets(...)`（本地图资产，用于 KG + 映射）。
2. `vector_rag` / `kg_rag` 按原链路执行。
3. 第一次调用 `graph_rag` 时，适配层会：
   - 把 `chunks_file` 上传为 youtu 数据集；
   - 调 youtu 构图；
   - 对每个 query 调 youtu `search`；
   - 映射回主仓证据 ID 口径输出。

注意：当前实现中，`graph_rag` 会把 `chunks_file` 指纹持久化到 `outputs/cache/youtu_sync_state.json`。
当“指纹未变化 + 远端 dataset 状态为 `ready`”时，跨进程多次运行 `run_compare` 会复用已构建的 youtu 数据集，不再重复上传/构图。

### 2.8 评估

```bash
python src/evaluation/run_eval.py \
  --pred-file outputs/results/compare_answers.jsonl \
  --gold-file data/queries/gold_qa.jsonl \
  --out-csv outputs/results/eval_compare.csv \
  --out-summary outputs/results/eval_compare_summary.json
```

---

## 3. 统一数据流（当前代码）

1. `chunks*.jsonl` -> Vector 索引（FAISS）。
2. 同一份 `chunks` -> 本地图资产（`triples/graph/communities`）。
3. 同一份 `chunks` -> youtu 数据集上传与 youtu 构图（GraphRAG后端图）。
4. 查询阶段：
   - VectorRAG：查 FAISS；
   - KG-RAG：查主仓 `graph.json`；
   - GraphRAG：查 youtu `search`，再映射回主仓 ID。
5. 输出：
   - `outputs/results/compare_answers.jsonl`
   - `outputs/results/compare_metrics.json`

---

## 4. 关键模块

- 对比入口：`src/experiments/run_compare.py`
- VectorRAG：`src/baselines/vector_rag.py`
- KG-RAG：`src/baselines/kg_rag.py`
- GraphRAG 适配层：`src/baselines/graph_rag.py`
- 本地图构建：
  - `src/graph_build/extract_triples.py`
  - `src/graph_build/build_graph.py`
  - `src/graph_build/build_communities.py`
- youtu 后端（被 GraphRAG 调用）：
  - `youtu-graphrag/backend.py`
  - `youtu-graphrag/models/constructor/kt_gen.py`
  - `youtu-graphrag/models/retriever/enhanced_kt_retriever.py`
- 评估：`src/evaluation/run_eval.py`

---

## 5. 输出格式（核心字段）

`compare_answers.jsonl`：

- `qid`, `type`, `query`
- `regimes.best_effort.vector_rag / kg_rag / graph_rag`
- `regimes.budget_matched.vector_rag / kg_rag / graph_rag`
- `budget_matched` 下每方法 `budget_check`

GraphRAG 输出兼容字段：

- `answer`
- `communities`
- `community_summaries`
- `map_partial_answers`
- `evidence`
- `evidence_chunks`
- `telemetry`
- `subgraph_edges`

`compare_metrics.json`：

- 分 regime / method 的 `llm_calls`、token、延迟等聚合
- `aggregate_metrics_by_type`
- `indexing_metrics`
- `graph_structure_metrics`（若成功计算）

---

## 6. 注意事项（按当前实现）

1. GraphRAG 依赖 youtu 服务在线；`base_url` 端口不对会直接失败。
2. 目前 KG 与 Graph 不是单图共用。
3. `require_id_alignment=true` 时，`chunk_id` 不能对齐会直接报错，不进入评测。
4. `run_compare.py` 不会自动构建向量索引；需先执行 `run_baselines.py --mode build_only`。
5. 若证据召回异常，优先检查 `chunks/index/store/graph/communities` 是否同一套 sampled/full 资产。
