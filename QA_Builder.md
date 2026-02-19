# QA Builder（对齐当前 pipeline 的版本）

本文件的目的：把“论文/实验需要的 gold QA（含证据）”**对齐到仓库当前已实现的三方法 pipeline**，确保你构建出来的 QA 数据可以：

- 直接导出 `data/queries/queries.jsonl` 喂给 `src/experiments/run_compare.py`
- 用统一的证据字段，对齐 VectorRAG / KG-RAG / GraphRAG 的输出，方便做 evidence 命中率与可追溯性分析

------

## 0. 与当前仓库实现对齐的关键事实（已核对）

### 0.1 真实图资产与 schema（来自 `src/graph_build/*`）

- `outputs/graph/triples.jsonl` 每行（由 `src/graph_build/extract_triples.py` 生成）：

```json
{
  "chunk_id": "...",
  "doc_id": "...",
  "subject": "...",
  "relation": "...",
  "object": "...",
  "evidence": "short quote from source text"
}
```

- `outputs/graph/graph.json`（由 `src/graph_build/build_graph.py` 生成）：
  - `edges[]` 是去重后的三元组边（按 `(subject, relation, object)` 聚合），包含：
    - `edge_id`, `source`, `relation`, `target`, `weight`
    - `mentions[]`: `{chunk_id, doc_id, evidence}`
  - `adjacency` 仅用于快速遍历（注意是从 `source` 出发的邻接表）

- `outputs/graph/communities.json`（由 `src/graph_build/build_communities.py` 生成）：
  - 文件内已包含每个社区的 `summary`（LLM 生成）
  - **不是** `node → community_id` 的扁平映射；它是“层级 communities 列表”
  - 每个社区包含：`community_id, level, nodes[], edges[], summary, parent_id, children_ids`
  - `levels[]` 给出不同 `level` 下可用的 `community_ids`

因此：旧版方案中提到的 `community_reports.jsonl` 在当前仓库**不存在**，应改为直接从 `communities.json` 读取 `summary`。

### 0.2 当前对比入口对 query 的要求（来自 `src/experiments/run_compare.py`）

`data/queries/queries.jsonl` 的每行只需要：

```json
{"qid": "...", "type": "...", "query": "..."}
```

其中 `type` 只是分层统计用的标签（字符串即可）。仓库现有 query 集的类型（见 `data/queries/queries.jsonl`）包括：

- `local_factual`（局部事实/可定位证据）
- `cross_doc_reasoning`（跨文档/跨线索推理）
- `global_summary`（全局总结/GraphRAG 风格）
- `evidence_tracing`（强调引用证据的回答）

------

## 1. QA Builder 的总目标（落地到当前仓库）

构建一个 `gold_qa.jsonl`（建议路径：`data/queries/gold_qa.jsonl`），每条记录包含：

1) **可直接用于对比脚本的 query 三元组**：`qid/type/query`  
2) **gold answer**（用于答案相似度/人工评测）：`answer`（可选 `key_points`）  
3) **gold evidence**（用于证据命中率/可追溯性分析）：`supporting_chunks / supporting_edges / supporting_communities`

推荐统一输出 schema：

```json
{
  "qid": "q001",
  "type": "local_factual | cross_doc_reasoning | global_summary | evidence_tracing",
  "query": "...",
  "answer": "...",
  "key_points": ["..."],
  "supporting_chunks": [{"chunk_id": "...", "doc_id": "..."}],
  "supporting_edges": [{"edge_id": "e000001", "source": "...", "relation": "...", "target": "..."}],
  "supporting_communities": [{"community_id": "l0_c0001", "level": 0}]
}
```

说明：

- `supporting_edges` 更贴合 KG-RAG 的 `subgraph_edges` 输出（它使用 `edge_id/source/relation/target/weight`）
- `supporting_chunks` 更贴合 VectorRAG / KG-RAG 的 `evidence` 输出（都有 `chunk_id/doc_id/text`）
- `supporting_communities` 更贴合 GraphRAG 的 `communities` 与 `evidence`（`community_id/level/summary`）

------

## 2. 输入依赖（与 pipeline 产物一致）

QA Builder 不再依赖 “不存在的文件”，只依赖当前 pipeline 已能生成的产物：

- **chunks（证据原文）**：
  - `outputs/indexes/chunk_store_sampled.json`（或 full 版本 `chunk_store.json`）
  - 或者 `data/processed/chunks_sampled.jsonl`（但建议优先用 store，方便按 `chunk_id` 索引）
- **图资产**：
  - `outputs/graph/triples.jsonl`
  - `outputs/graph/graph.json`
  - `outputs/graph/communities.json`

可选依赖（用于和现有对比脚本预算/设置保持一致）：

- `config.yaml`（`retrieval.top_k`，`comparison.*`）
- `config_budget.yaml`（budget matched regime 的约束）

------

## 3. 预处理索引（QA Builder 内部需要做的“对齐索引”）

### Step 1：chunk_id → chunk_meta（原文、doc_id）

从 `outputs/indexes/chunk_store*.json` 建立：

- `chunk_map[chunk_id] = {chunk_id, doc_id, text, ...}`

这一步用于把三元组/边的 `mentions.chunk_id` 映射到可读证据（以及导出 gold 的 `supporting_chunks`）。

### Step 2：三元组/边的证据对齐（edge_id ↔ triple ↔ mentions）

从 `outputs/graph/graph.json` 建立：

- `edge_by_id[edge_id] = edge`
- `edge_key[(source, relation, target)] = edge_id`
- `mentions_by_edge[edge_id] = edge.mentions[]`

备注：虽然 `triples.jsonl` 已有 `chunk_id`，但 `graph.json` 更适合“聚合后边 + 多 mention”的证据归一。

### Step 3：community_id → community（summary/nodes/edges/level）

从 `outputs/graph/communities.json` 建立：

- `community_by_id[community_id] = community`

并准备一个“可选的 node → community 映射”（按 level）：

- 选择一个 `qa_community_level`（建议与 GraphRAG 默认 `query_level` 保持一致；见 `config.yaml` 的 `comparison.*.graph.query_level`；并与 `src/baselines/graph_rag.py` 语义一致：`<0` 表示选最细粒度 level）
- `node2community[node_id] = community_id`（同一 level 下如果出现冲突，用最大 size / 最大 overlap 的社区优先）

这一步用于：给 `supporting_edges`/`supporting_chunks` 打上“所属社区”标签，形成 `supporting_communities`。

------

## 4. 任务设计（对齐当前 query type 命名）

本仓库现有 query type 并非 `single_hop/multi_hop/global_sensemaking`，因此这里改为使用现有 type 命名，但保持三类能力覆盖。

### 4.1 `local_factual`（≈ single-hop fact，偏 VectorRAG）

**构造来源**：从 `graph.json` 的高权重边中采样（避免 pronoun / 低质量实体）。  
**问题模板**（可规则化、低成本）：

- 给定边 `A --[r]--> B`：
  - Q: “What does **A** {r}?”（答案 B）
  - Q: “What is {r} by **A**?”（答案 B）
  - Q: “Which entity is {r} to **A**?”（答案 B）

**gold evidence**：

- `supporting_edges`: 该边本身（含 `edge_id/source/relation/target`）
- `supporting_chunks`: 从该边 `mentions[]` 里取 1–2 条（优先 weight 高的边、文本可读的 chunk）
- `supporting_communities`: 用 `node2community[A]`/`node2community[B]`（或 edge_id 所在社区）推导

### 4.2 `cross_doc_reasoning`（≈ multi-hop reasoning，偏 KG-RAG）

**构造来源**：在 `graph.json` 上采样 2-hop 或 3-hop 路径（KG-RAG 代码默认支持 `max_hops=1~2`，见 `config.yaml`）。  
**路径形式**：

- 2-hop：`A --[r1]--> B --[r2]--> C`

**问题模板**（规则化 baseline + 可选 LLM 改写）：

- Q: “How is **A** connected to **C**?”（答案：B 或“通过 B 连接”）
- Q: “What intermediate entity links **A** and **C**?”（答案：B）
- Q: “Through which factor does **A** relate to **C**?”（答案：B）

**gold evidence**：

- `supporting_edges`: 路径上的 2 条边
- `supporting_chunks`: 从两条边的 `mentions[]` 各取 1 条（合计 2–4 chunks）
- `supporting_communities`: 对 A/B/C 做 `node2community` 映射后去重

### 4.3 `global_summary`（GraphRAG 风格的“社区级总结”）

**构造来源**：从 `communities.json` 选定某个 `level` 下的社区，直接利用其 `summary`。  
GraphRAG 当前实现是“社区摘要检索 + map-reduce 汇总”（见 `src/baselines/graph_rag.py`），因此 gold 的核心证据也应以“社区 summary”为主。

**问题模板**：

- Q: “Summarize the major themes and risks in this knowledge base.”（全局）
- 或针对单社区：
  - Q: “What are the main issues and risks discussed in this context?”（社区级）

**gold answer** 的两种可选策略：

- **策略 A（低成本、可复现）**：`answer = community.summary`（并可从 summary 规则抽取 3–6 条 `key_points`）
- **策略 B（更自然，但更贵）**：用 LLM 基于 `community.summary` 生成更精炼的 answer + key_points（注意记录 token/calls，论文中说明）

**gold evidence**：

- `supporting_communities`: 目标社区（1–3 个）
- `supporting_edges`: 可选（从社区 `edges[]` 选 top weight 的若干条边作为结构化补充）
- `supporting_chunks`: 可选（从这些边的 mentions 反推 chunk_id）

### 4.4 `evidence_tracing`（强调“引用与可追溯性”）

这个类型用于强迫模型在回答时给出证据线索（doc_id/chunk 引用）。  
构造方式可以复用 `local_factual` 或 `cross_doc_reasoning` 的样本，但要求：

- query 明确要求 “citing document IDs and quoted snippets”
- gold 中 `supporting_chunks` 至少 3 条，且每条都能在 chunk_store 找到原文

------

## 5. 数据集规模建议（与论文工作量匹配）

建议总量 40–80 条（足够毕设/论文实验）：

| type                | count（建议） |
|---------------------|--------------|
| local_factual        | 15–25        |
| cross_doc_reasoning  | 10–20        |
| global_summary       | 10–20        |
| evidence_tracing     | 5–15         |

------

## 6. 与现有 pipeline 的对接方式（强对齐）

### 6.1 从 `gold_qa.jsonl` 导出 `data/queries/queries.jsonl`

因为 `run_compare.py` 只需要 `qid/type/query`，所以导出规则是：

- 对每条 gold QA：写入 `{"qid": qid, "type": type, "query": query}`

### 6.2 运行对比（保持 chunks/index 与 gold evidence 对齐）

务必确保：gold QA 构建时使用的 `chunk_store*.json` 与 `run_compare.py` 使用的 `--store-file` 是同一套 sampled/full（否则 chunk_id 会错位）。

（命令示例见 `docs/comparison_pipeline.md` 的 2.6。）

------

## 7. Repo 建议新增模块位置（贴合现有目录结构）

当前仓库已有 `src/evaluation/`，建议把 QA Builder 放这里：

```text
src/evaluation/qa_builder.py
data/queries/gold_qa.jsonl
data/queries/queries.jsonl          # 从 gold 导出，仅含 qid/type/query
data/queries/gold.jsonl             # 可选：仅含 qid/answer，给 src/evaluation/run_eval.py 用
```

------

## 8. 执行命令（约定接口）

下面是建议的 CLI 形状（用于后续实现 `src/evaluation/qa_builder.py`）：

```bash
python src/evaluation/qa_builder.py \
  --graph-file outputs/graph/graph.json \
  --communities-file outputs/graph/communities.json \
  --chunk-store-file outputs/indexes/chunk_store_sampled.json \
  --out-gold data/queries/gold_qa.jsonl \
  --out-queries data/queries/queries.jsonl \
  --n_local 20 \
  --n_cross 15 \
  --n_global 15 \
  --n_trace 10 \
  --qa-community-level 0
```

实现时的最低验收标准：

- 生成的 `data/queries/queries.jsonl` 能被 `python src/experiments/run_compare.py ...` 直接读取运行
- `gold_qa.jsonl` 中的 `supporting_chunks[].chunk_id` 都能在 `chunk_store_file` 中找到
- `supporting_edges[].edge_id` 都能在 `graph.json` 的 `edges[]` 中找到
- `supporting_communities[].community_id` 都能在 `communities.json` 中找到，且 level 与 `qa-community-level` 一致