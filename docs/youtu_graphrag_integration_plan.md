# Youtu-GraphRAG 直接替换实施方案（仅保留对比 Pipeline 兼容）

## 1. 目标与边界

目标：直接用 `youtu-graphrag` 替换当前 `GraphRAG` 实现，不做双后端切换，不做回退方案，只保证现有对比 pipeline 可继续运行。

强约束：
1. `src/baselines/graph_rag.py` 仍保留同名入口 `answer_with_graphrag(...)`，但内部逻辑改为 youtu 调用。
2. `src/experiments/run_compare.py`、`src/experiments/run_ablations.py`、`src/evaluation/run_eval.py` 不改调用方式（参数/字段兼容）。
3. 对外结果结构维持现有评测依赖字段。

不在范围：
1. 保留旧 GraphRAG 代码路径。
2. 灰度开关、A/B 切换、回滚脚本。
3. 重构 VectorRAG 与 KG-RAG 逻辑。

---

## 2. 兼容目标（以现有 compare pipeline 为准）

必须兼容的入口：
1. `python src/experiments/run_compare.py ...`
2. `python src/experiments/run_ablations.py ...`
3. `python src/evaluation/run_eval.py ...`

必须兼容的 GraphRAG 输出键：
1. `answer`
2. `communities`
3. `community_summaries`
4. `map_partial_answers`
5. `evidence`
6. `evidence_chunks`
7. `telemetry`
8. `subgraph_edges`

其中 `run_eval.py` 的证据评估依赖 `chunk_id/community_id/edge_id`，这些字段缺失会导致证据指标失真。

---

## 3. 总体设计

采用“`HTTP 服务调用 + 结果适配层`”直接替换：

1. 本项目不直接 import youtu 内部模块，避免 `utils` 命名冲突。
2. 由本项目调用 youtu 后端接口：
   - `POST /api/construct-graph`
   - `GET /api/construct-graph/{task_id}`
   - `POST /api/v1/datasets/{dataset_name}/search`
3. 在 `src/baselines/graph_rag.py` 内完成输出字段适配，保持现有签名与返回协议。

为什么采用该方案：
1. 替换点单一（仅 `graph_rag.py` 主入口）。
2. 对 compare pipeline 调用方零改动。
3. token/延迟信息可直接从 youtu `meta` 落到当前 `telemetry`。

---

## 4. 实施步骤

## Phase 1：硬替换接入（MVP）

任务：
1. 新增 `src/adapters/youtu_client.py`
   - 封装 health/check、construct-graph 提交与轮询、search 调用。
2. 改造 `src/baselines/graph_rag.py`
   - 保持函数签名不变。
   - 全量改为调用 `youtu_client.search(...)`。
3. 新增 `src/adapters/youtu_schema_adapter.py`
   - 把 youtu `data/meta` 映射为现有 GraphRAG 返回字段。

验收：
1. `run_compare.py` 不改 CLI 可跑通。
2. `run_ablations.py` 不改代码可跑通。

## Phase 2：资产与预算兼容

任务：
1. 新增 `src/adapters/youtu_dataset_sync.py`
   - 将 `chunks_file` 同步为 youtu 可识别语料（`corpus.json` 或上传接口）。
2. 在 `run_compare.py` 进入 query 循环前，新增 youtu 构图确保步骤：
   - 若 youtu 图不存在，则调用 `construct-graph` 并轮询完成。
3. 预算口径对齐（`meta.usage + llm_calls -> telemetry`）：
   - `prompt_tokens/completion_tokens/total_tokens`
   - `llm_calls`
   - `llm_latency_ms`（由 `llm_calls[].latency_ms` 汇总）

验收：
1. `budget_matched` 下 `budget_check` 正常产出。
2. `compare_metrics.json` 的 `graph_rag` 聚合统计可用。

## Phase 3：证据字段对齐

任务：
1. youtu `retrieved_chunks` 映射为 `evidence_chunks`（尽量回填 `chunk_id`）。
2. youtu `retrieved_triples` 映射为 `subgraph_edges`（无 `edge_id` 时置空并保留文本）。
3. 社区字段兼容：
   - `communities` 无稳定 ID 时可为空数组。
   - `community_summaries`/`map_partial_answers` 保底给空数组，不允许缺键。

验收：
1. `run_eval.py` 在含证据 gold 上可跑完。
2. 输出 JSON schema 稳定，无缺字段报错。

---

## 5. 字段映射规范（必须执行）

| 当前字段 | youtu 来源 | 处理规则 |
|---|---|---|
| `answer` | `data.answer` | 原样透传 |
| `telemetry.prompt_tokens` | `meta.usage.prompt_tokens` | `int` |
| `telemetry.completion_tokens` | `meta.usage.completion_tokens` | `int` |
| `telemetry.total_tokens` | `meta.usage.total_tokens` | `int` |
| `telemetry.llm_calls` | `len(meta.llm_calls)` | 计数 |
| `telemetry.llm_latency_ms` | `sum(meta.llm_calls[].latency_ms)` | 汇总 |
| `evidence_chunks` | `data.retrieved_chunks` | 转 `{chunk_id,text}`；无 ID 时置 `chunk_id=""` |
| `subgraph_edges` | `data.retrieved_triples` | 转 `{edge_id, text}`；`edge_id=""` 可接受 |
| `communities` | `data.sub_questions`/可选社区数据 | 无稳定社区 ID 时返回 `[]` |

---

## 6. 配置与运行约定

建议新增配置（无后端切换语义）：

```yaml
graph:
  youtu:
    base_url: "http://127.0.0.1:8000"
    dataset: "enterprise"
    timeout_sec: 120
    construct_poll_sec: 2
    construct_timeout_sec: 1800
```

运行约定：
1. 对比任务前需确保 youtu 服务已启动。
2. `run_compare.py` 仅保留一个 GraphRAG 路径（即 youtu）。
3. 若 youtu 服务不可用，直接失败并报错，不做本地 GraphRAG 兜底。

---

## 7. 验收清单

1. `run_compare --regimes both` 可完整结束。
2. `compare_answers.jsonl` 中 `graph_rag` 始终包含第 2 节约定的全部键。
3. `run_eval.py` 在 `gold_qa.jsonl` 模式可产出 evidence 指标。
4. `compare_metrics.json` 中 `graph_rag` 具有有效 token/call/latency 统计。

---

## 8. 风险与处理

1. 证据 `chunk_id` 无法稳定映射。
   处理：先保证 `evidence_chunks.text` 可评估；并记录 `unmapped_chunk_count`。
2. youtu 构图耗时较长导致 compare 启动慢。
   处理：把构图前置到 pipeline 初始化阶段，只构建一次。
3. youtu 接口字段变更导致适配失效。
   处理：在适配层增加 schema 校验，缺关键字段即失败并提示具体路径。
