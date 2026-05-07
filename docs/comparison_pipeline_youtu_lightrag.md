## 1. 目标

本流程独立于 `docs/comparison_pipeline.md`，支持比较：

- `graph_rag`
- `youtu_graph_rag`
- `lightrag`

并分开评测两类指标：

1. 构图指标（Build）：`token / time / llm_calls`
2. 端到端问答质量（Retrieve）：答案质量 + 文档/Chunk 证据命中 + `token / time`

## 2. 入口脚本

- `src/experiments/run_compare_youtu_lightrag.py`

支持阶段：

- `--stage build`：只跑构图指标
- `--stage retrieve`：只跑检索质量
- `--stage all`：两阶段都跑

## 3. 核心策略

- 构图阶段默认强制重建：`--build-force-rebuild true`
- 检索阶段默认复用缓存：`--retrieve-reuse-cache true`
- 参与本次运行的系统：`--include-rag graph_rag,lightrag,youtu`
- 默认评测矩阵：`Top10+Top20 × reject/open`
- `Top-k Accuracy` 定义：`I[answer_semantic_yesno = 1 AND support_at_k = 1]`
- `run_eval.py` 会同时输出：
  - `answer_similarity`
  - `answer_token_f1`
  - `answer_semantic_yesno`
  - `topk_accuracy`
  - `chunk_hit_rate_at_k / chunk_recall_at_k`
  - `doc_hit_rate_at_k / doc_recall_at_k`
- 文档级 gold 口径：
  - 优先 `supporting_chunks[].doc_id`
  - 若为空，回退到 `meta.query_doc_key`
  - 再回退到 `meta.title`
- 通用 chunk 合并参数：`--merge-chunks`（默认 `1`，设为 `5` 表示每 5 个 chunk 合并后再入库；对 youtu 不生效）

## 4. 构图评测（Build）

### 4.1 运行示例

```bash
python src/experiments/run_compare_youtu_lightrag.py \
  --stage build \
  --chunks-file data/processed/qa_aligned_chunks.jsonl \
  --include-rag youtu,graph_rag,lightrag \
  --build-parallel true \
  --build-force-rebuild true \
  --triples-file outputs/graph/qa_aligned_triples_v3.jsonl \
  --graph-file outputs/graph/qa_aligned_graph_v3_docscoped.json \
  --communities-file outputs/graph/qa_aligned_communities_v3_docscoped_pruned.json \
  --youtu-base-url http://127.0.0.1:8000 \
  --youtu-dataset cuad_v3 \
  --youtu-schema-file config_triple_schema_cuad_core_v3.json \
  --youtu-sync-mode shared_dir \
  --youtu-construct-timeout-sec 7200 \
  --build-metrics-file outputs/results/rags/youtu_lightrag_build_metrics.json \
  --merge-chunks 1
```

### 4.2 输出文件

- `outputs/results/rags/youtu_lightrag_build_metrics.json`

重点字段：

- `build_benchmark.lightrag.total_tokens`
- `build_benchmark.lightrag.latency_ms_total`
- `build_benchmark.youtu_graph_rag.total_tokens`
- `build_benchmark.youtu_graph_rag.latency_ms_total`

说明：`youtu` 的构图 token 来自后端 `construct-graph` 任务返回的 `build_stats.usage`（真实 usage）。
说明：构图结果支持增量更新。比如下一次只跑 `--include-rag youtu`，会只更新 youtu 条目并保留已存在的 graph_rag/lightrag 条目。
说明：若包含 `graph_rag`，build 阶段需要可写的 `triples_file/graph_file/communities_file` 路径。

## 5. 检索评测（Retrieve）

说明：当前流程会同时评估答案质量和检索证据命中；不是只看 `Top-k Accuracy`。
说明：因此 `retrieve` 阶段会生成 `eval.csv / eval_summary.json / paper_metrics` 等文件。
说明：但若 `--include-rag` 包含 `graph_rag`，则 `retrieve` 阶段仍需要 `graph_file/communities_file`。

### 5.0 可选：先生成三类评测问题（本地/结构/全局）

若不使用既有 `cuad_capability_queries.jsonl`，可先从
`data/raw/cuad/train_separate_questions.json` 生成平衡三类问题：

```bash
python src/evaluation/build_cuad_question_converter.py \
  --cuad-train-file data/raw/cuad/train_separate_questions.json \
  --mode llm \
  --llm-types global_synthesis \
  --total-per-type 20 \
  --answer-mode weak_reference \
  --type-name-scheme capability \
  --progress-every-qa 1 \
  --out-queries-file data/queries/cuad_converted_queries_generated.jsonl \
  --out-gold-file data/queries/cuad_converted_gold_generated.jsonl
```

关键点：

- `--total-per-type`：每类总题数（全局采样，不按文档倍增）。
- `--llm-types global_synthesis`：仅 global 使用 LLM；local/cross 用规则模板。
- `--answer-mode weak_reference`：输出弱参考答案；如需占位答案可改 `placeholder`。
- `--type-name-scheme capability`：类型名输出为 `local_factual/cross_clause/global_summary`。
- `--progress-every-qa`：按 QA 条数输出进度日志。

### 5.1 运行示例

```bash
python src/experiments/run_compare_youtu_lightrag.py \
  --stage retrieve \
  --queries-file data/queries/queries_final.jsonl \
  --gold-file data/queries/gold_final.jsonl \
  --lightrag-working-dir outputs/lightrag/qa_aligned_chunks \
  --chunks-file data/processed/qa_aligned_chunks.jsonl \
  --graph-file outputs/graph/qa_aligned_graph_v3_docscoped.json \
  --communities-file outputs/graph/qa_aligned_communities_v3_docscoped_pruned.json \
  --topk-list 20 \
  --answer-modes reject \
  --regime best_effort \
  --retrieve-reuse-cache true \
  --judge-mode llm_yesno \
  --judge-model qwen-flash \
  --include-rag lightrag \
  --youtu-base-url http://127.0.0.1:8000 \
  --youtu-dataset cuad_v3 \
  --youtu-client-id web_client \
  --youtu-force-rebuild false
```

若使用上面的转换器产物，则替换为：

- `--queries-file data/queries/cuad_converted_queries_generated.jsonl`
- `--gold-file data/queries/cuad_converted_gold_generated.jsonl`

### 5.2 输出文件

每个 `topk × mode`：

- `outputs/results/rags/youtu_lightrag_compare_answers_top{K}_{mode}.jsonl`
- `outputs/results/rags/youtu_lightrag_compare_metrics_top{K}_{mode}.json`
- `outputs/results/rags/youtu_lightrag_eval_top{K}_{mode}.csv`
- `outputs/results/rags/youtu_lightrag_eval_top{K}_{mode}_summary.json`
- `outputs/results/rags/youtu_lightrag_paper_metrics_top{K}_{mode}.json`
- `outputs/results/rags/youtu_lightrag_paper_metrics_top{K}_{mode}.csv`

汇总：

- `outputs/results/rags/youtu_lightrag_topk_mode_summary.csv`
- `outputs/results/rags/youtu_lightrag_topk_mode_summary.md`
- `outputs/results/rags/youtu_lightrag_topk_mode_summary.json`
- `outputs/results/rags/youtu_lightrag_pipeline_summary.json`

### 5.3 独立重刷 `run_eval`

当 `compare_answers` 已存在、但你修改了评测逻辑或手动调整了结果文件时，可单独重刷 `eval.csv / eval_summary.json`：

```bash
python src/evaluation/run_eval.py \
  --pred-file outputs/results/test/youtu_lightrag_compare_answers_top20_reject.jsonl \
  --gold-file data/queries/gold_final.jsonl \
  --out-csv outputs/results/rags/youtu_lightrag_eval_top20_reject.csv \
  --out-summary outputs/results/rags/youtu_lightrag_eval_top20_reject_summary.json \
  --method-mode all \
  --judge-mode llm_yesno \
  --judge-model qwen-flash
```

说明：

- `--pred-file` 指向 `retrieve` 阶段生成的 `compare_answers`
- `--gold-file` 指向对应的 gold 集
- `--method-mode all` 表示同时评估 `graph_rag / youtu_graph_rag / lightrag`
- 若只想看 youtu，可改为 `--method-mode only_youtu`
- 文档级召回现在会对 `local/cross` 题自动回退到 `meta.query_doc_key`

## 6. 参数建议

- 快速冒烟：`--max-queries 10`
- 只评严格回答：`--answer-modes reject`
- 只评一个 TopK：`--topk-list 10`
- 构图冷启动对比：`--stage build --build-force-rebuild true`
- 检索低成本复跑：`--stage retrieve --retrieve-reuse-cache true`
- 可指定输出文件：
  - `--build-metrics-file`
  - `--retrieve-summary-csv`
  - `--retrieve-summary-md`
  - `--retrieve-summary-json`
  - `--pipeline-summary-file`
