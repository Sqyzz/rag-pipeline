# youtu-GraphRAG 独立测试 Pipeline 实施文档

## 1. 目标

构建一条仅测试 `youtu-graphrag` 的独立 pipeline，并保证其输出指标与 `src/experiments/run_compare.py` 的 `graph_rag` 指标口径完全一致，可直接接入现有评估链路。

## 2. 可行性分析

结论：**可行，且改动风险可控**。原因如下：

1. 现有对比指标逻辑已高度模块化，可直接复用：
   - 预算与预算检查：`src/utils/budget.py`、`run_compare.py::_check_budget`
   - telemetry 聚合：`run_compare.py::_merge_telemetry`
   - 延迟统计：`run_compare.py::_latency_stats`
2. 评估侧对 compare 结果的读取只依赖固定 schema，不依赖方法内部实现：
   - `src/evaluation/run_eval.py` 读取 `qid/type/query/regimes/*/<method>`（代码中会遍历 `("vector_rag","kg_rag","graph_rag")`；若独立 pipeline 只产出 `graph_rag`，需在下游聚合时按 `method="graph_rag"` 过滤，或在输出中为其它 method 写入空壳以避免“空答案方法”被误统计）
   - 证据评估读取 `evidence/subgraph_edges/communities/evidence_chunks`
3. 图资产与图结构指标可直接沿用现有流程：
   - **优先**复用 youtu 已构建的图结构（导出为本项目 `graph.json/communities.json`），再计算 `graph_structure_metrics`
   - 若 youtu 暂不支持导出图资产，则降级为使用本地 `run_compare.py::ensure_graph_assets` 生成评估所需文件（见 4.3.1 路径 B）
   - `evaluation/graph_structure_metrics.py::compute_graph_structure_metrics`

## 3. 关键对齐约束（必须满足）

### 3.1 输出文件与顶层结构对齐

1. 结果文件（jsonl）保持 compare 布局：
   - `qid`
   - `type`
   - `query`
   - `regimes.best_effort.graph_rag`
   - `regimes.budget_matched.graph_rag`
2. 指标文件（json）保持 compare 摘要布局：
   - `aggregate_metrics`
   - `aggregate_metrics_by_type`
   - `graph_assets_metrics`
   - `indexing_metrics`
   - `graph_structure_metrics`
   - `budget`
   - `regimes`

### 3.2 graph_rag payload 对齐

每个 regime 的 `graph_rag` 至少包含：

1. `answer`
2. `communities`
3. `community_summaries`
4. `query_level`
5. `use_hierarchy`
6. `use_community_summaries`
7. `shuffle_communities`
8. `use_map_reduce`
9. `map_keypoints_limit`
10. `map_partial_answers`
11. `subgraph_edges`
12. `evidence`
13. `evidence_chunks`
14. `telemetry`
15. `budget_check`（仅 `budget_matched`）

> 说明：当前基线实现（`src/baselines/graph_rag.py::answer_with_graphrag`）还会输出 `generate_summary_on_demand`、`embedding_cache`、`on_demand_summaries_generated` 等字段；它们不是评估必需字段，但若目标是“与 compare 的 graph_rag 输出尽可能一致”，建议一并保留。

### 3.3 telemetry 口径对齐

`telemetry`（对齐 `src/utils/telemetry.py::Telemetry.to_dict()`）必须输出以下键：

1. `llm_calls`
2. `embedding_calls`
3. `llm_latency_ms`
4. `embedding_latency_ms`
5. `prompt_tokens`
6. `completion_tokens`
7. `total_tokens`
8. `extra`（**允许为 dict**；`run_compare.py::_merge_telemetry` / `_check_budget` 不依赖其数值化，但可用于记录缺失字段、估算方式等）

> 注意：`run_compare.py` 的预算检查与聚合基于上述数值键（`extra` 除外）；缺失会导致统计漂移或预算判断失真。

### 3.4 图结构复用约束（新增硬性要求）

1. 当测试数据未发生变化时，**必须复用** youtu 已构建图结构，不允许每次测试都重新构图。
2. 仅在以下场景触发重构图：
   - 输入语料变化（新增、删除、内容修改）
   - 构图参数变化（例如分块策略、schema、社区配置）
   - 显式指定强制重构（`--force-rebuild`）
3. 每次运行必须在 metrics 中记录本次行为：
   - `graph_reuse.used_cached_graph`（`true/false`）
   - `graph_reuse.reason`（`fingerprint_match` / `fingerprint_changed` / `forced_rebuild`）
   - `graph_reuse.fingerprint`

## 4. 方案设计

## 4.1 新增独立入口

新增：`src/experiments/run_youtu_graphrag_test.py`

职责：

1. 复用 query 加载、regime 解析、预算配置读取逻辑。
2. 确保图资产可用并复用 compare 的指标落盘逻辑（图资产优先走 youtu 导出；无法导出时再走本地降级，见 4.3.1）。
3. 对每个 query 仅执行 youtu-GraphRAG，并写入 `regimes.<rg>.graph_rag`。
4. 复用 compare 的聚合逻辑，生成完全同口径 summary。

## 4.2 新增 youtu 适配层

新增：`src/baselines/youtu_graph_rag_adapter.py`

职责：

1. 封装 youtu 的请求、响应解析、异常处理。
2. 将 youtu 返回映射为本项目 `graph_rag` 标准 payload。
3. 构造标准 `telemetry`。

### 4.2.1 youtu 服务交互协议（实现前置假设）

为避免将 youtu 作为 python 包直接引入（以及潜在 `utils` 命名冲突），建议按 `HTTP 服务调用 + 适配层` 方式接入。默认假设 youtu 提供如下接口（见 `docs/youtu_graphrag_integration_plan.md`）：

1. 构图提交：
   - `POST /api/construct-graph`
2. 构图状态查询（轮询直到完成）：
   - `GET /api/construct-graph/{task_id}`
3. 检索问答：
   - `POST /api/v1/datasets/{dataset_name}/search`

建议新增通用 client（若后续需要复用）：

- `src/adapters/youtu_client.py`
  - `health_check()`
  - `construct_graph(dataset_name, ...) -> task_id`
  - `poll_construct(task_id, timeout_sec, poll_sec) -> final_status`
  - `search(dataset_name, payload) -> {data, meta}`

建议新增配置项（可放 `config.yaml` 或以 CLI 覆盖）：

```yaml
youtu:
  base_url: "http://127.0.0.1:8000"
  dataset: "enterprise"
  timeout_sec: 120
  construct_poll_sec: 2
  construct_timeout_sec: 1800
```

### 4.2.2 youtu -> 本项目 GraphRAG 字段映射（必须执行）

独立测试 pipeline 的输出要与 compare 的 `graph_rag` payload shape 对齐；建议按以下映射（与 `docs/youtu_graphrag_integration_plan.md` 一致）：

| 本项目字段 | youtu 来源（示例路径） | 处理规则 |
|---|---|---|
| `answer` | `data.answer` | `str` 透传，缺失则 `""` |
| `telemetry.prompt_tokens` | `meta.usage.prompt_tokens` | `int`，缺失为 0 |
| `telemetry.completion_tokens` | `meta.usage.completion_tokens` | `int`，缺失为 0 |
| `telemetry.total_tokens` | `meta.usage.total_tokens` | `int`，缺失则用 `prompt+completion` 或 0 |
| `telemetry.llm_calls` | `len(meta.llm_calls)` 或 `meta.llm_call_count` | `int`，缺失为 0 |
| `telemetry.llm_latency_ms` | `sum(meta.llm_calls[].latency_ms)` | `int`，缺失为 0 |
| `telemetry.embedding_calls` | `len(meta.embedding_calls)` 或 `meta.embedding_call_count` | `int`，缺失为 0 |
| `telemetry.embedding_latency_ms` | `sum(meta.embedding_calls[].latency_ms)` | `int`，缺失为 0 |
| `evidence` | 由 youtu evidence 派生 | **必须存在**：优先填社区证据（含 `community_id`），否则直接复用 `evidence_chunks` 的前 `max_evidence` 条并确保包含 `chunk_id`（用于 `run_eval.py` 证据评估） |
| `evidence_chunks` | `data.retrieved_chunks` | 映射为 `[{chunk_id,text,...}]`；无 ID 时 `chunk_id=""`，但键必须存在 |
| `subgraph_edges` | `data.retrieved_triples` / `data.retrieved_edges` | 映射为 `[{edge_id,source,relation,target,text,...}]`；无 `edge_id` 时 `edge_id=""` |
| `communities` | youtu 返回的社区/子问题字段（若有） | 若无稳定社区 ID，返回 `[]`（但键必须存在） |
| `community_summaries` | youtu 返回社区摘要（若有） | 无则 `[]` |
| `map_partial_answers` | youtu 返回 map 阶段中间产物（若有） | 无则 `[]` |

关于 `telemetry.extra`：

- 必须为 dict（允许为空 `{}`），建议写入：
  - `missing_fields: [...]`
  - `usage_complete: true/false`（若 usage 不完整，`budget_matched` 下预算对齐应进入降级策略，见第 6 节）

建议接口：

```python
def answer_with_youtu_graphrag(
    query: str,
    graph_file: str,
    communities_file: str,
    top_communities: int = 3,
    max_evidence: int = 12,
    query_level: int = 0,
    use_hierarchy: bool = True,
    use_community_summaries: bool = True,
    shuffle_communities: bool = True,
    use_map_reduce: bool = True,
    max_summary_chars: int = 1800,
    map_keypoints_limit: int = 5,
    max_completion_tokens: int | None = None,
) -> dict:
    ...
```

签名故意与 `answer_with_graphrag(...)` 对齐，便于共享 regime 参数。

## 4.3 图复用与增量构建控制（新增）

新增：`src/adapters/youtu_graph_state.py`

职责：

1. 计算测试数据指纹（fingerprint），建议包含：
   - `chunks_file` 内容哈希（优先全量哈希；若必须采样，需同时记录文件大小与 mtime，并在 metrics 里注明 `fingerprint_method="sampled"`）
   - 构图关键参数哈希（schema、community、dataset 名称）
2. 维护本地状态文件：
   - 建议路径：`outputs/graph/youtu_graph_state.json`
   - 记录：`dataset`、`fingerprint`、`graph_task_id`、`built_at`、`build_params`
3. 在测试启动阶段执行“是否重构图”判定：
   - 指纹一致：跳过构图，直接复用
   - 指纹不一致：触发 youtu 构图并更新状态文件
   - `--force-rebuild`：无条件重构

### 4.3.1 远端构图与本地评估资产的对齐策略（关键补强）

现有评估链路中有两类“需要图文件”的地方：

1. `compute_graph_structure_metrics(graph_file, communities_file)`：需要本地 `graph.json/communities.json`，且 schema 与本仓库 `graph_build/*` 产物一致。
2. `run_eval.py::_build_community_chunk_map(graph_file, communities_file)`：若提供这两个文件，会用 `community.edges -> graph.edges[].mentions[].chunk_id` 回填 chunk 覆盖率；缺文件则退化为空映射（不影响主评估流程，但会让“社区命中带来的 chunk 回填”失效）。

因此独立测试 pipeline 推荐两种实现路径（按优先级）：

- **路径 A（推荐，严格复用 youtu 图结构）**：yotu 构图完成后，从 youtu **导出/下载** `graph.json` 与 `communities.json` 到本项目 `--graph-file/--communities-file`（要求 youtu 提供 artifacts 下载接口，或在 `GET construct-graph/{task_id}` 的结果中返回下载地址）。  
  - 好处：图结构指标、community->chunk 映射与 youtu 实际图一致；满足“必须复用 youtu 图结构”的硬约束。
- **路径 B（降级，仅保证评估不崩）**：若 youtu 暂不支持导出图资产，则：
  - `run_eval.py` 侧可不依赖 `graph_file/communities_file`（为空映射）；
  - `graph_structure_metrics` 在 summary 中写入 `{"error": "...not available..."}`（保持字段存在，但明确不可用）；
  - 同时在 metrics 的 `graph_reuse` 中记录 `reason="fingerprint_match|fingerprint_changed|forced_rebuild"` 与 `used_cached_graph`，确保“远端图复用”仍然生效。
  - 注意：该路径不满足“结构指标严格等同于 youtu 图”的需求，只适合作为短期过渡。

### 4.3.2 chunks 数据集同步（保证 fingerprint 真正对应 youtu 数据）

独立测试 pipeline 的 `chunks_file` 变化必须能触发 youtu 侧重构图，否则会出现“本地 fingerprint 变了但 youtu 仍复用旧数据集”的错配。

建议新增：`src/adapters/youtu_dataset_sync.py`

职责（两种实现任选其一，按 youtu 实际能力落地）：

1. **API 上传模式**：将 `chunks_file` 转换为 youtu 约定格式并上传/更新（例如 `POST /api/v1/datasets/{dataset}/upload` 之类；具体以 youtu 实际接口为准）。  
2. **共享目录模式**：将 `chunks_file` 转为 youtu 可读取的 `corpus.json`（或其它固定文件名），写入 youtu 服务挂载目录，并在 `construct-graph` 时只传 `dataset_name + corpus_path`。

无论哪种模式，都必须在 `youtu_graph_state.json` 的 `build_params` 中记录：

- `dataset`
- `chunks_source`（文件路径或对象存储 key）
- `chunks_fingerprint`
- `sync_mode`（`api_upload` / `shared_dir`）

建议 CLI 增加：

1. `--reuse-graph`（默认 `true`）
2. `--force-rebuild`（默认 `false`，优先级高于 `--reuse-graph`）
3. `--graph-state-file`（默认 `outputs/graph/youtu_graph_state.json`）
4. `--youtu-base-url` / `--youtu-dataset`（可选；也可只用 config）
5. `--export-youtu-artifacts`（默认 `true`，若 youtu 支持导出）

## 4.4 budget_matched 自适应策略对齐

复刻 `run_compare.py` 现有逻辑：

1. 首轮执行后若 `_check_budget(...).within_budget=False`，触发重跑。
2. 收缩参数：
   - `top_communities=1`
   - `use_map_reduce=False`
   - `shuffle_communities=False`
   - `max_summary_chars=min(current, 500)`（实现中回退默认值为 1200：`min(int(graph_kwargs.get("max_summary_chars", 1200)), 500)`）
   - `map_keypoints_limit=min(current, 3)`
3. 在 payload 加 `budget_adaptation` 字段，记录原参数与收缩参数。

## 4.5 指标聚合完全复用

在 `run_youtu_graphrag_test.py` 中直接复制并复用以下函数实现，避免口径偏差（其中图资产相关函数需按 4.3.1 选择实现路径）：

1. `_merge_telemetry`
2. `_latency_stats`
3. `_check_budget`
4. `_load_indexing_metrics`
5. `ensure_graph_assets`（仅路径 B：yotu 不支持导出 artifacts 时的降级）

并新增（路径 A 推荐）：

6. `ensure_youtu_graph_assets(...)`：负责
   - 调用 `construct-graph` / 轮询完成
   - （可选）同步 `chunks_file` 到 youtu 数据集（见 4.3.2）
   - 导出/下载 artifacts 到 `--graph-file/--communities-file`
   - 产出 `graph_assets_metrics.youtu_*`（例如 task_id、耗时、是否复用缓存等）

## 5. 实施步骤

### Phase 1：MVP（跑通并产出同 schema）

1. 新增 `run_youtu_graphrag_test.py`。
2. 新增 `youtu_graph_rag_adapter.py`，完成基础映射。
3. 输出 `compare_answers` 同布局（仅填 `graph_rag`）。
4. 输出 `compare_metrics` 同布局（建议保留 `vector_rag/kg_rag/graph_rag` 三个 method 的键；其中 `graph_rag` 为真实汇总，其余可写入全 0 的 telemetry 以保证 shape 一致；并按 compare 口径补充 `latency_ms` 统计字段）。
5. 增加图复用状态记录（至少含 `used_cached_graph/reason/fingerprint`）。

验收：

1. `python src/experiments/run_youtu_graphrag_test.py --regimes both` 可跑完。
2. `src/evaluation/run_eval.py` 可直接读取结果并完成评估。
3. 同一份输入运行两次，第二次默认不触发构图。

### Phase 2：预算与自适应对齐

1. 接入 `BudgetManager`。
2. 输出 `budget_check.manager/error`。
3. 实现超预算自适应重跑及 `budget_adaptation`。

验收：

1. `budget_matched` 下可看到 `within_budget`、`error`、`manager.used`。
2. token/calls 统计与 compare 口径一致。

### Phase 3：一致性校验与回归

1. 新增 `src/evaluation/validate_youtu_alignment.py`：
   - schema 校验（字段路径、类型）
   - 聚合口径校验（给定 telemetry 输入，验证输出一致）
   - 图复用校验（相同 fingerprint 下不触发构图）
2. 新增 `tests/test_youtu_pipeline_alignment.py`。

验收：

1. CI/本地可自动验证对齐，不依赖人工 spot check。

## 6. 风险与应对

1. youtu 返回缺失 token/latency 字段。
   - 应对：适配层提供默认值 0，并在 `telemetry.extra.missing_fields` 记录缺失项；同时建议增加 `telemetry.extra.usage_complete=false`，在 `budget_matched` 下将 “usage 不完整” 视为 **无法严格对齐预算口径**（可选择：直接启用 shrink、或将 `budget_check.within_budget` 标记为 `false` 并记录原因）。
2. youtu 证据缺少稳定 `chunk_id/edge_id/community_id`。
   - 应对：先保证文本证据可用；ID 为空字符串但保留键，避免 schema 断裂。
3. youtu 接口抖动导致单 query 失败。
   - 应对：适配层增加有限重试；失败时 payload 返回空答案+错误信息，流程不中断。
4. 误判“数据未变化”导致复用了旧图。
   - 应对：指纹同时覆盖输入内容与关键构图参数；提供 `--force-rebuild` 人工兜底。

## 7. 交付清单

1. `src/experiments/run_youtu_graphrag_test.py`
2. `src/baselines/youtu_graph_rag_adapter.py`
3. `src/evaluation/validate_youtu_alignment.py`
4. `tests/test_youtu_pipeline_alignment.py`
5. 本文档：`docs/youtu_graphrag_independent_test_pipeline_implementation.md`

## 8. 建议运行命令

```bash
python src/experiments/run_youtu_graphrag_test.py \
  --queries-file data/queries/gold_qa.jsonl \
  --chunks-file data/processed/chunks_sampled.jsonl \
  --triples-file outputs/graph/triples.jsonl \
  --graph-file outputs/graph/graph.json \
  --communities-file outputs/graph/communities.json \
  --youtu-base-url http://127.0.0.1:8000 \
  --youtu-dataset enterprise \
  --reuse-graph true \
  --graph-state-file outputs/graph/youtu_graph_state.json \
  --export-youtu-artifacts true \
  --regimes both \
  --budget-config-file config_budget.yaml \
  --out-file outputs/results/youtu_compare_answers.jsonl \
  --metrics-file outputs/results/youtu_compare_metrics.json
```
