# Youtu-GraphRAG 直接替换实施方案（仅保留 compare pipeline 兼容）

## 1. 目标与决策

目标：直接用 `youtu-graphrag` 替换当前 `GraphRAG`，不做双后端切换、不做回退路径；唯一目标是保证当前 compare pipeline 指标输出口径可用。

固定决策：
1. `answer_with_graphrag(...)` 保留原函数签名，内部改为 youtu 调用。
2. `run_compare.py`、`run_ablations.py`、`run_eval.py` 调用方式不变。
3. 只接受“字段兼容 + 指标口径可对齐”的替换方案。

---

## 2. 可行性结论（先回答能不能做）

结论：`ID` 级证据字段可补齐，方案可行。

依据：
1. youtu 检索内部已经产出 `chunk_ids`，但 API 最终只回传了 `retrieved_chunks` 文本，可在 backend 层直接补出 ID 字段。
2. youtu 图文件 `output/graphs/*_new.json` 可直接查看社区节点与摘要（`community` 节点 `properties.description`），独立后端不影响构建可观测性。
3. youtu 默认构图会重新生成随机短 `chunk_id`，与当前 pipeline 的 UUID `chunk_id` 不一致；这不是“不可做”，而是“必须显式对齐”的工程约束。

硬门槛：
1. 如果不能把 youtu 查询结果中的 `chunk_id` 对齐到当前 gold 体系，就不能宣称 evidence 指标口径等价。

---

## 3. 兼容目标（以现有评测脚本为准）

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

证据评估要求：
1. `run_eval.py` 依赖 `chunk_id/community_id/edge_id` 做集合匹配。
2. 缺 ID 或 ID 体系不一致，会直接拉低 evidence recall/f1。

---

## 4. 总体架构

采用“`独立 youtu 服务 + 主仓适配层`”：

1. 主仓不直接 import youtu 内部模块，规避命名冲突与依赖污染。
2. 通过 HTTP 调用 youtu：
   - `POST /api/construct-graph`
   - `GET /api/construct-graph/{task_id}`
   - `POST /api/v1/datasets/{dataset_name}/search`
3. 在主仓适配层统一输出当前 GraphRAG 协议。

说明：
1. 独立后端只隔离运行时，不隔离可观测资产。
2. youtu 内部图与社区摘要仍可直接查阅（`youtu-graphrag/output/graphs/*_new.json`）。

---

## 5. ID 级证据字段对齐方案（核心）

## 5.1 `chunk_id` 对齐（最高优先级）

目标：
1. youtu 返回的 `chunk_id` 与主仓 gold 使用同一 ID 空间。

方案：
1. 主仓将 `chunks_file` 同步为 youtu 数据集时，显式携带 `chunk_id` 与 `doc_id`。
2. youtu 构图阶段优先使用外部 `chunk_id`（若提供），不再无条件生成随机短 ID。
3. youtu `search` 响应新增 `retrieved_chunk_ids`（与 `retrieved_chunks` 一一对应）。
4. 主仓适配时输出：
   - `evidence_chunks=[{chunk_id, doc_id, text}]`

验收门槛：
1. `unmapped_chunk_count == 0`（默认要求）。
2. 任意 query 的 `retrieved_chunk_ids` 长度与 `retrieved_chunks` 对齐。

## 5.2 `edge_id` 对齐

目标：
1. 输出 `subgraph_edges[].edge_id`，可参与 run_eval 的 edge 指标。

方案：
1. 优先从当前 `outputs/graph/graph.json` 建立 `(source, relation, target) -> edge_id` 映射。
2. 将 youtu 检索三元组归一化后映射到 `edge_id`。
3. 未命中时保留文本并记录 `unmapped_edge_count`，但不静默丢失。

## 5.3 `community_id` 对齐

目标：
1. 输出 `communities` 与 `community_summaries`，并尽量落到当前社区 ID 空间。

方案：
1. 从 youtu 图中提取 `community` 节点 `name/description`。
2. 基于 chunk 覆盖或实体重叠映射到当前 `communities.json` 的 `community_id`。
3. 无稳定映射时记录 `unmapped_community_count` 并显式告警。

---

## 6. 输出映射规范

| 当前字段 | youtu 来源 | 对齐规则 |
|---|---|---|
| `answer` | `data.answer` | 原样透传 |
| `telemetry.prompt_tokens` | `meta.usage.prompt_tokens` | `int` |
| `telemetry.completion_tokens` | `meta.usage.completion_tokens` | `int` |
| `telemetry.total_tokens` | `meta.usage.total_tokens` | `int` |
| `telemetry.llm_calls` | `len(meta.llm_calls)` | 计数 |
| `telemetry.llm_latency_ms` | `sum(meta.llm_calls[].latency_ms)` | 汇总 |
| `evidence_chunks[].chunk_id` | `data.retrieved_chunk_ids` | 必须可用并对齐 gold ID 空间 |
| `evidence_chunks[].text` | `data.retrieved_chunks` | 与 ID 一一对应 |
| `subgraph_edges[].edge_id` | 三元组映射 | 由 `(s,r,t)` 对齐主仓 `graph.json` |
| `communities[]` | 社区映射结果 | 输出当前社区 ID |
| `community_summaries[]` | youtu 社区摘要 | 来自 `community.description` 或检索上下文 |

---

## 7. 分阶段计划（文档方案，不含实现）

## Phase A：接口补齐（ID 透出）

1. youtu `search` 增加结构化返回：
   - `retrieved_chunk_ids`
   - `retrieved_triples_struct`（结构化 `(s,r,t,score)`）
2. 保留现有 `retrieved_chunks/retrieved_triples` 文本字段，避免上层一次性断裂。

## Phase B：ID 空间统一

1. 主仓向 youtu 同步语料时传入外部 `chunk_id/doc_id`。
2. youtu 构图保留外部 `chunk_id`。
3. 校验 `chunk_id` 对齐率。

## Phase C：Edge/Community 映射

1. 建立 `triple -> edge_id` 映射器。
2. 建立 `youtu community -> current community_id` 映射器。
3. 输出 `unmapped_*` 诊断字段。

## Phase D：Pipeline 验收

1. `run_compare --regimes both` 全链路跑通。
2. `run_eval` evidence 模式可跑，且不出现系统性 ID 丢失。
3. `compare_metrics.json` 的 GraphRAG token/calls/latency 可用。

---

## 8. 配置约定

```yaml
graph:
  youtu:
    base_url: "http://127.0.0.1:8000"
    dataset: "enterprise"
    timeout_sec: 120
    construct_poll_sec: 2
    construct_timeout_sec: 1800
    require_id_alignment: true
```

语义：
1. `require_id_alignment=true` 时，发现 `chunk_id` 未对齐直接失败，不进入评测。

---

## 9. Go/No-Go 判定

Go 条件（全部满足）：
1. `chunk_id` 可稳定对齐现有 gold ID 空间。
2. `run_eval` evidence 指标可正常统计。
3. `graph_rag` 输出字段完整且 schema 稳定。

No-Go 条件（任一命中）：
1. youtu 无法输出或保留可对齐 `chunk_id`。
2. 大量 `unmapped_chunk_count` 导致 evidence 指标失真。
3. 关键 telemetry 字段缺失无法完成预算统计。

---

## 10. 风险与处理

1. 风险：外部 `chunk_id` 无法被 youtu 构图链路保留。
   处理：将“保留外部 `chunk_id`”设为实现前置项；不满足即 No-Go。
2. 风险：三元组文本格式波动导致 `edge_id` 映射不稳。
   处理：优先依赖结构化三元组返回，不依赖纯文本解析。
3. 风险：社区映射多对一或不稳定。
   处理：映射策略固定化并输出命中率，低于阈值时禁用社区指标对比声明。
