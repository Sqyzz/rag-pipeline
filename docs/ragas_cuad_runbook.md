# Ragas CUAD 支线操作文档

本文档是可执行 runbook，用于在当前仓库里跑通一条独立的 `ragas + CUAD + vector_rag / graph_rag / lightrag / youtu_graph_rag` 补充评测支线。

目标分两层：

1. 先在小样本上完成一次冒烟测试
2. 冒烟通过后，再扩大到正式补充评测

该支线是补充 benchmark，不替代主评测流程。

---

## 1. 产物与入口

本支线对应的新增入口如下：

- `src/experiments/run_ragas_cuad_generate.py`
- `src/experiments/run_ragas_cuad_compare.py`
- `src/evaluation/run_ragas_eval.py`

运行顺序固定为：

1. 准备 sampled corpus
2. 生成 ragas testset
3. 多方法统一回答
4. ragas 统一评测

命名约束：

1. 这条支线产生的 `docs / chunks / graph / communities / indexes / working_dir / results` 必须使用 `ragas` 后缀或放在独立 `ragas` 目录下
2. 不要复用主流程已经存在的文件名
3. 不要把支线资产写回主流程默认路径

---

## 2. 前置条件

### 2.1 依赖安装

至少需要安装：

- `datasets`
- `langchain-core`
- `langchain-openai`

如果你直接用仓库依赖文件，执行：

```bash
pip install -r requirements.txt
```

### 2.2 LLM / Embedding 配置

这条支线不单独维护 ragas 专属配置，直接复用当前仓库的：

- `config.yaml -> llm.*`
- `config.yaml -> embedding.*`

同时要求环境变量里有 key：

```bash
export LLM_API_KEY=...
export EMB_API_KEY=...
```

如果你本来就是走 DashScope，也可以只配：

```bash
export DASHSCOPE_API_KEY=...
```

### 2.3 数据与运行资产

你至少要有以下资产中的一部分：

- `data/raw/cuad/CUADv1.json`
- `data/processed/cuad_docs_ragas.jsonl`
- `data/processed/cuad_docs_sampled_ragas.jsonl`
- `data/processed/cuad_chunks_sampled_ragas.jsonl`
- `outputs/graph/cuad_graph_sampled_ragas.json`
- `outputs/graph/cuad_communities_sampled_ragas.json`
- `outputs/indexes/faiss_cuad_sampled_ragas.idx`
- `outputs/indexes/chunk_store_cuad_sampled_ragas.json`
- `outputs/lightrag/cuad_sampled_ragas`

如果你要跑 `youtu_graph_rag`，还需要：

- 可访问的 youtu 后端
- sampled corpus 对应的数据集已上传并可搜索
- 建议使用独立数据集名，例如 `cuad_sampled_ragas`

### 2.4 youtu 问题拆解模式开关

当前 `youtu` 已支持两套可切换的问题拆解流程，配置位置在：

- [`youtu-graphrag/config/base_config.yaml`](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/config/base_config.yaml)

配置项：

```yaml
retrieval:
  decomposition:
    mode: natural_language_subquestions
    enable_query_compilation: true
    max_query_variants: 4
```

`mode` 支持：

- `natural_language_subquestions`
  - 旧流程
  - LLM 直接输出自然语言子问题
- `retrieval_requirements`
  - 新流程
  - LLM 输出结构化检索需求
  - 后端再把它编译成多条 retrieval queries
- `disabled`
  - 可选诊断模式
  - 不做拆题，直接用原问题

`retrieval_requirements` 模式下，后端会尽量生成并写出：

- `retrieval_requirement`
- `retrieval_queries`

这些字段会出现在 compare 结果的：

- `reasoning_trace.sub_question_answers[*].retrieval_requirement`
- `reasoning_trace.sub_question_answers[*].retrieval_queries`

用于检查：

1. 是否仍然走了 `decomposition_fallback`
2. 是否真的产出了：
   - `intent`
   - `entities`
   - `terms`
   - `anchors`
   - `left_endpoint`
   - `right_endpoint`
3. 每个子需求是否真的被编译成了多条 query

建议的使用方式：

1. 默认先保留：
   - `edge_layer.enabled: true`
   - `agent.enable_ircot: true`
2. 将：
   - `retrieval.doc_consistency.enabled: false`
   以减少额外变量
3. 分别跑：
   - `natural_language_subquestions`
   - `retrieval_requirements`
4. 先比较 `compare jsonl` 中的 trace，再看 ragas 分数

如果使用新流程，切换配置后必须：

1. 重启 youtu backend
2. 再重新运行 `run_ragas_cuad_compare.py`

否则 compare 结果仍可能来自旧后端进程。

---

## 3. 小样本冒烟流程

建议第一次只跑 `20` 个测试样本。

### 3.1 从 CUAD raw 生成 docs

如果你还没有 `docs.jsonl`，先执行：

```bash
python src/ingestion/load_docs.py \
  --dataset cuad \
  --raw-file data/raw/cuad/CUADv1.json \
  --out-file data/processed/cuad_docs_ragas.jsonl \
  --split-name full
```

### 3.2 抽样文档

建议先抽 `20` 份文档：

```bash
python src/ingestion/sample_dedup.py \
  --input data/processed/cuad_docs_ragas.jsonl \
  --output data/processed/cuad_docs_sampled_ragas.jsonl \
  --sample-size 5 \
  --sampling-mode random \
  --seed 42 \
  --stats-output outputs/results/cuad_sampled_stats_ragas.json
```

### 3.3 基于 sampled docs 构造 chunk / vector / graph / lightrag / youtu 资产

这一步要求各条路线都只使用 sampled corpus，并且所有资产名都带 `ragas` 后缀。

建议产出以下命名：

- `data/processed/cuad_chunks_sampled_ragas.jsonl`
- `outputs/indexes/faiss_cuad_sampled_ragas.idx`
- `outputs/indexes/chunk_store_cuad_sampled_ragas.json`
- `outputs/graph/cuad_triples_sampled_ragas.jsonl`
- `outputs/graph/cuad_graph_sampled_ragas.json`
- `outputs/graph/cuad_communities_sampled_ragas.json`
- `outputs/lightrag/cuad_sampled_ragas`
- `outputs/graph/youtu_graph_state_sampled_ragas.json`
- `outputs/youtu_sync/ragas_sampled`
- `youtu dataset = cuad_sampled_ragas`

这里建议明确分成两步执行。

#### 3.3.1 先切出 ragas 专用 chunks

使用现有切 chunk 脚本：

```bash
python src/ingestion/chunking.py \
  --in-file data/processed/cuad_docs_sampled_ragas.jsonl \
  --out-file data/processed/cuad_chunks_sampled_ragas.jsonl \
  --chunk-size 2000 \
  --overlap 120
```

说明：

1. 这一步只负责 `docs -> chunks`
2. 当前 `CUAD docs` 在仓库里本身是 paragraph 级，所以很多 paragraph 最终可能只产生一个 chunk，这是正常现象
3. 这一步产出的 `cuad_chunks_sampled_ragas.jsonl` 作为后续 `ragas generate`、`vector_rag`、`graph_rag`、`lightrag`、`youtu` 的统一输入

#### 3.3.2 构造 VectorRAG 资产

普通 `vector_rag` 使用同一份 sampled chunks 构建 FAISS dense index，作为 `youtu_graph_rag` 的纯向量检索对照组。

```bash
PYTHONPATH=src python -c "from pathlib import Path; import json; from baselines.vector_rag import build_index; metrics = build_index('data/processed/cuad_chunks_sampled_ragas.jsonl', 'outputs/indexes/faiss_cuad_sampled_ragas.idx', 'outputs/indexes/chunk_store_cuad_sampled_ragas.json'); Path('outputs/results/ragas').mkdir(parents=True, exist_ok=True); Path('outputs/results/ragas/vector_index_sampled_metrics_ragas.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')"
```

输出：

- `outputs/indexes/faiss_cuad_sampled_ragas.idx`
- `outputs/indexes/chunk_store_cuad_sampled_ragas.json`
- `outputs/results/ragas/vector_index_sampled_metrics_ragas.json`

这一步只构建 dense vector index，不做图扩展、rerank 或问题拆解。

#### 3.3.3 再统一构造 graph / lightrag / youtu 资产

使用：

```bash
python src/experiments/run_compare_youtu_lightrag.py \
  --stage build \
  --chunks-file data/processed/cuad_chunks_sampled_ragas.jsonl \
  --triples-file outputs/graph/cuad_triples_sampled_ragas.jsonl \
  --graph-file outputs/graph/cuad_graph_sampled_ragas.json \
  --communities-file outputs/graph/cuad_communities_sampled_ragas.json \
  --lightrag-working-dir outputs/lightrag/cuad_sampled_ragas \
  --include-rag graph_rag,lightrag,youtu \
  --youtu-base-url http://127.0.0.1:8000 \
  --youtu-dataset cuad_sampled_ragas \
  --youtu-sync-mode shared_dir \
  --youtu-graph-state-file outputs/graph/youtu_graph_state_sampled_ragas.json \
  --youtu-shared-corpus-dir outputs/youtu_sync/ragas_sampled \
  --build-force-rebuild true \
  --build-parallel true \
  --build-metrics-file outputs/results/ragas/build_sampled_metrics_ragas.json
```

如果当前不想让 youtu 阻塞，可以先只构造 `graph_rag + lightrag`：

```bash
python src/experiments/run_compare_youtu_lightrag.py \
  --stage build \
  --chunks-file data/processed/cuad_chunks_sampled_ragas.jsonl \
  --triples-file outputs/graph/cuad_triples_sampled_ragas.jsonl \
  --graph-file outputs/graph/cuad_graph_sampled_ragas.json \
  --communities-file outputs/graph/cuad_communities_sampled_ragas.json \
  --lightrag-working-dir outputs/lightrag/cuad_sampled_ragas \
  --include-rag graph_rag,lightrag \
  --build-force-rebuild true \
  --build-parallel true \
  --results-dir outputs/results/ragas/build_sampled \
  --build-metrics-file outputs/results/ragas/build_sampled_metrics.json
```

这一步的作用边界要明确：

1. `run_compare_youtu_lightrag.py --stage build` 会负责 `graph / communities / lightrag / youtu` 相关资产
2. 它不负责从 `docs` 生成 `chunks`
3. 因此先切 chunk，再 build，是这条支线更稳定的操作顺序

#### 3.3.4 可选：对比 `graph_rag` 和 `youtu` 的图结构

这一小步是可选诊断，不是 `ragas` 支线的硬前置。

结论先说清楚：

1. `src/evaluation/graph_structure_metrics.py` 可以计算单个图的结构指标
2. 但它一次只能看一侧，不会直接生成 `graph_rag vs youtu` 的对比结果
3. 如果你要直接做双边对比，更适合使用 `src/experiments/run_graph_structure_compare.py`

也就是说：

- `graph_structure_metrics.py` 适合单独查看某一侧图结构
- `run_graph_structure_compare.py` 适合直接对比 `graph_rag` 和 `youtu`

前提条件：

1. `graph_rag` 侧已有本地文件，例如：
   - `outputs/graph/qa_aligned_graph_sampled_ragas.json`
   - `outputs/graph/qa_aligned_communities_sampled_ragas.json`
2. `youtu` 侧也必须有本地 `graph` 和 `communities` 文件

注意：

当前 `run_compare_youtu_lightrag.py --stage build` 不会自动把 youtu 后端图资产导出到本地。
所以如果你要做这一步，必须先把 youtu 的图和社区文件落地到本地路径，再比较。

如果两侧本地文件都已具备，直接运行：

```bash
python src/experiments/run_graph_structure_compare.py \
  --left-name graph_rag \
  --left-graph-file outputs/graph/cuad_graph_sampled_ragas.json \
  --left-communities-file outputs/graph/cuad_communities_sampled_ragas.json \
  --right-name youtu_graph_rag \
  --right-graph-file youtu-graphrag/output/graphs/cuad_sampled_ragas_new.json \
  --right-communities-file ''\
  --community-level auto \
  --community-mode consistent \
  --out-json outputs/results/ragas/graph_structure_compare_sampled_ragas.json
```

如果你只想单独查看某一侧的结构指标，可以分别运行 `graph_structure_metrics.py`：

```bash
python src/evaluation/graph_structure_metrics.py \
  --graph-file outputs/graph/qa_aligned_graph_sampled_ragas.json \
  --communities-file outputs/graph/qa_aligned_communities_sampled_ragas.json \
  --out-json outputs/results/ragas/graph_rag_structure_metrics_sampled_ragas.json \
  --out-dir outputs/results/ragas/graph_rag_structure_plots_sampled_ragas

python src/evaluation/graph_structure_metrics.py \
  --graph-file outputs/graph/youtu_graph_sampled_ragas.json \
  --communities-file outputs/graph/youtu_communities_sampled_ragas.json \
  --out-json outputs/results/ragas/youtu_structure_metrics_sampled_ragas.json \
  --out-dir outputs/results/ragas/youtu_structure_plots_sampled_ragas
```

建议把这一步的定位理解为：

1. 检查两侧图规模是否差异过大
2. 检查社区划分是否过碎或过粗
3. 给后续 `ragas` 结果解释提供结构侧背景

不要把它当成 `ragas` 指标本身的一部分。

### 3.4 生成 ragas testset

执行：

```bash
python src/experiments/run_ragas_cuad_generate.py \
  --chunks-file data/processed/cuad_chunks_sampled_ragas.jsonl \
  --out-testset-file data/queries/ragas_cuad_smoke_testset.jsonl \
  --out-summary-file data/queries/ragas_cuad_smoke_testset_summary.json \
  --testset-size 6 \
  --random-seed 42 \
  --max-chunks 100
```

输出：

- `data/queries/ragas_cuad_smoke_testset.jsonl`
- `data/queries/ragas_cuad_smoke_testset_summary.json`

这一步建议检查：

1. 三类 `synthesizer_name` 是否都有样本
2. `reference` 是否大量为空
3. `reference_contexts` 是否大量为空
4. `reference_context_ids` 缺失率是否过高

### 3.5 跑多方法统一 compare

执行：

```bash
python src/experiments/run_ragas_cuad_compare.py \
  --testset-file data/queries/ragas_cuad_smoke_testset.jsonl \
  --chunks-file data/processed/cuad_chunks_sampled_ragas.jsonl \
  --graph-file outputs/graph/cuad_graph_sampled_ragas.json \
  --communities-file outputs/graph/cuad_communities_sampled_ragas.json \
  --lightrag-working-dir outputs/lightrag/cuad_sampled_ragas \
  --youtu-base-url http://127.0.0.1:8000 \
  --youtu-dataset cuad_sampled_ragas \
  --vector-idx-file outputs/indexes/faiss_cuad_sampled_ragas.idx \
  --vector-store-file outputs/indexes/chunk_store_cuad_sampled_ragas.json \
  --methods graph_rag,lightrag,youtu_graph_rag,vector_rag \
  --out-file outputs/results/ragas/ragas_cuad_smoke_compare_merged.jsonl \
  --max-questions-per-type 3
```

输出：

- `outputs/results/ragas/ragas_cuad_smoke_compare_merged.jsonl`
- `outputs/results/ragas/ragas_cuad_smoke_compare_merged_vector_rag_predictions.jsonl`
- `outputs/results/ragas/ragas_cuad_smoke_compare_merged_graph_rag_predictions.jsonl`
- `outputs/results/ragas/ragas_cuad_smoke_compare_merged_lightrag_predictions.jsonl`
- `outputs/results/ragas/ragas_cuad_smoke_compare_merged_youtu_graph_rag_predictions.jsonl`

如果当前不想接 youtu，可以只跑两种方法：

```bash
python src/experiments/run_ragas_cuad_compare.py \
  --testset-file data/queries/ragas_cuad_smoke_testset.jsonl \
  --chunks-file data/processed/cuad_chunks_sampled_ragas.jsonl \
  --graph-file outputs/graph/cuad_graph_sampled_ragas.json \
  --communities-file outputs/graph/cuad_communities_sampled_ragas.json \
  --lightrag-working-dir outputs/lightrag/cuad_sampled_ragas \
  --vector-idx-file outputs/indexes/faiss_cuad_sampled_ragas.idx \
  --vector-store-file outputs/indexes/chunk_store_cuad_sampled_ragas.json \
  --methods vector_rag,graph_rag,lightrag \
  --out-file outputs/results/ragas/ragas_cuad_smoke_compare_merged.jsonl \
  --max-questions-per-type 3
```

这一步建议检查：

1. 每题是否都有 `response`
2. `retrieved_contexts` 是否大面积为空
3. `retrieved_context_ids` 是否大面积为空
4. 不同方法的题量是否一致

### 3.6 跑 ragas 评测

先跑文本级四指标：

```bash
python src/evaluation/run_ragas_eval.py \
  --pred-file outputs/results/ragas/ragas_cuad_smoke_compare_merged.jsonl \
  --out-dir outputs/results/ragas/smoke
```

输出：

- `outputs/results/ragas/smoke/ragas_eval_per_sample.jsonl`
- `outputs/results/ragas/smoke/ragas_eval_summary.json`
- `outputs/results/ragas/smoke/ragas_eval_summary.csv`

如果你已经确认：

1. `reference_context_ids` 大多数非空
2. `retrieved_context_ids` 大多数非空
3. 二者都使用同一套 `chunk_id`

再启用 IDBased 指标：

```bash
python src/evaluation/run_ragas_eval.py \
  --pred-file outputs/results/ragas/ragas_cuad_smoke_compare_merged.jsonl \
  --out-dir outputs/results/ragas/smoke_id \
  --enable-id-based
```

---

## 4. 冒烟通过标准

满足以下条件，就可以认为冒烟通过：

1. `ragas` 测试集成功生成
2. `vector_rag`、`graph_rag` 和 `lightrag` 至少能稳定完成回答
3. 如果启用 `youtu`，它也能产出 `response` 和 `retrieved_contexts`
4. 文本级四指标能在大多数样本上算完
5. summary 文件能稳定导出

不建议在冒烟阶段追求：

- 全量数据
- 参数扫描
- 与主流程三能力题强行一一对应

---

## 5. 正式补充评测流程

冒烟通过后，把 sampled corpus 换成正式资产即可。

### 5.1 生成正式 testset

```bash
python src/experiments/run_ragas_cuad_generate.py \
  --chunks-file data/processed/qa_aligned_chunks.jsonl \
  --out-testset-file data/queries/ragas_cuad_testset.jsonl \
  --out-summary-file data/queries/ragas_cuad_testset_summary.json \
  --testset-size 60 \
  --random-seed 42
```

### 5.2 构建正式 VectorRAG 资产

正式 compare 如果包含 `vector_rag`，必须先用本次 compare 的 `--chunks-file` 构建对应的 FAISS index 和 chunk store。

下面命令只适用于正式 compare 的 `--chunks-file` 确认为 `data/processed/qa_aligned_chunks.jsonl` 的情况：

```bash
PYTHONPATH=src python -c "from pathlib import Path; import json; from baselines.vector_rag import build_index; metrics = build_index('data/processed/qa_aligned_chunks.jsonl', 'outputs/indexes/faiss_cuad_ragas.idx', 'outputs/indexes/chunk_store_cuad_ragas.json'); Path('outputs/results/ragas').mkdir(parents=True, exist_ok=True); Path('outputs/results/ragas/vector_index_metrics_ragas.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')"
```

输出：

- `outputs/indexes/faiss_cuad_ragas.idx`
- `outputs/indexes/chunk_store_cuad_ragas.json`
- `outputs/results/ragas/vector_index_metrics_ragas.json`

如果你仍在跑小样本冒烟，不要使用这两个正式文件名，继续使用：

- `outputs/indexes/faiss_cuad_sampled_ragas.idx`
- `outputs/indexes/chunk_store_cuad_sampled_ragas.json`

注意：

1. `graph_rag` 使用的是 `--graph-file` 和 `--communities-file`，不是直接读取 `--chunks-file`
2. `lightrag` 会读取 `--chunks-file`，但也会复用 `--lightrag-working-dir` 中已有资产；复用前应确认该 working dir 是从同一 chunks 文件构建的
3. `youtu_graph_rag` 的实际检索语料来自 `--youtu-dataset` 对应的后端数据集，`--chunks-file` 主要用于本地 chunk/doc id 映射；必须确认后端数据集也是从同一语料上传的
4. 因此 VectorRAG 的 index 必须和当前 compare 的 `--chunks-file` 同源；如果你换了 `--chunks-file`，就要换一组 `--vector-idx-file / --vector-store-file`

### 5.3 跑正式 compare

```bash
python src/experiments/run_ragas_cuad_compare.py \
  --testset-file data/queries/ragas_cuad_testset.jsonl \
  --chunks-file data/processed/qa_aligned_chunks.jsonl \
  --graph-file outputs/graph/qa_aligned_graph_v3_docscoped.json \
  --communities-file outputs/graph/qa_aligned_communities_v3_docscoped_pruned.json \
  --lightrag-working-dir outputs/lightrag/qa_aligned_chunks \
  --youtu-base-url http://127.0.0.1:8000 \
  --youtu-dataset cuad_v3 \
  --vector-idx-file outputs/indexes/faiss_cuad_ragas.idx \
  --vector-store-file outputs/indexes/chunk_store_cuad_ragas.json \
  --out-file outputs/results/ragas/retrieval_requirements_vector.jsonl \
  --methods youtu_graph_rag,vector_rag \
  --max-workers 8 \
  --max-questions-per-type 3
```

### 5.4 跑正式 eval

```bash
python src/evaluation/run_ragas_eval.py \
  --pred-file outputs/results/ragas/retrieval_requirements_vector.jsonl \
  --out-dir outputs/results/ragas/vector_test \
  --timeout-sec 300 \
  --max-workers 8
```

---

## 6. 输出解释

### 6.1 testset 文件

`run_ragas_cuad_generate.py` 输出的每条样本包含：

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

其中：

- `reference_context_ids` 固定使用 `chunk_id`
- `reference_doc_ids` 单独保存 `doc_id`

### 6.2 merged compare 文件

`run_ragas_cuad_compare.py` 输出的每条样本包含：

- `qid`
- `method`
- `question`
- `response`
- `reference`
- `retrieved_contexts`
- `retrieved_context_ids`
- `retrieved_doc_ids`
- `reference_contexts`
- `reference_context_ids`
- `reference_doc_ids`
- `synthesizer_name`
- `telemetry`

### 6.3 eval summary

`run_ragas_eval.py` 目前会输出两类 summary：

1. `per_method`
2. `per_method_synthesizer`

第一版默认指标为：

- `answer_correctness`
- `faithfulness`
- `context_precision`
- `context_recall`

启用 `--enable-id-based` 后再增加：

- `id_based_context_precision`
- `id_based_context_recall`

---

## 7. 常见问题

### 7.1 报 `ImportError: langchain_core` 或 `datasets`

说明 ragas 运行依赖没装齐。执行：

```bash
pip install -r requirements.txt
```

### 7.2 生成 testset 时直接失败

优先检查：

1. `LLM_API_KEY` / `EMB_API_KEY` 是否存在
2. `config.yaml` 里的 `base_url` / `model` 是否可用
3. `chunks-file` 是否真的有非空文本

### 7.3 compare 阶段 youtu 失败

先不要卡在 youtu 上。可以先只跑：

```bash
--methods graph_rag,lightrag
```

等文本级闭环跑通后，再回头联调 youtu。

### 7.4 IDBased 指标几乎没值

通常说明以下任一问题：

1. `reference_context_ids` 缺失率高
2. `retrieved_context_ids` 缺失率高
3. 两边 `chunk_id` 口径不一致

这时先不要开启 `--enable-id-based`。

---

## 8. 建议执行策略

推荐按下面顺序推进：

1. 先跑 sampled corpus 的 `20` 题冒烟
2. 再扩大到 `50` 题
3. 确认文本级四指标稳定后，再考虑 `IDBased`
4. 最后再上正式规模

如果你只是第一次联调，最稳妥的最小命令集合是：

```bash
python src/ingestion/chunking.py \
  --in-file data/processed/cuad_docs_sampled_ragas.jsonl \
  --out-file data/processed/cuad_chunks_sampled_ragas.jsonl \
  --chunk-size 1000 \
  --overlap 120

python src/experiments/run_compare_youtu_lightrag.py \
  --stage build \
  --chunks-file data/processed/cuad_chunks_sampled_ragas.jsonl \
  --triples-file outputs/graph/cuad_triples_sampled_ragas.jsonl \
  --graph-file outputs/graph/cuad_graph_sampled_ragas.json \
  --communities-file outputs/graph/cuad_communities_sampled_ragas.json \
  --lightrag-working-dir outputs/lightrag/cuad_sampled_ragas \
  --include-rag graph_rag,lightrag \
  --build-force-rebuild true \
  --build-parallel false \
  --results-dir outputs/results/ragas/build_sampled \
  --build-metrics-file outputs/results/ragas/build_sampled_metrics.json

python src/experiments/run_ragas_cuad_generate.py \
  --chunks-file data/processed/cuad_chunks_sampled_ragas.jsonl \
  --out-testset-file data/queries/ragas_cuad_smoke_testset.jsonl \
  --out-summary-file data/queries/ragas_cuad_smoke_testset_summary.json \
  --testset-size 20 \
  --random-seed 42

python src/experiments/run_ragas_cuad_compare.py \
  --testset-file data/queries/ragas_cuad_smoke_testset.jsonl \
  --chunks-file data/processed/cuad_chunks_sampled_ragas.jsonl \
  --graph-file outputs/graph/cuad_graph_sampled_ragas.json \
  --communities-file outputs/graph/cuad_communities_sampled_ragas.json \
  --lightrag-working-dir outputs/lightrag/cuad_sampled_ragas \
  --methods graph_rag,lightrag \
  --out-file outputs/results/ragas/ragas_cuad_smoke_compare_merged.jsonl

python src/evaluation/run_ragas_eval.py \
  --pred-file outputs/results/ragas/ragas_cuad_smoke_compare_merged.jsonl \
  --out-dir outputs/results/ragas/smoke
```
