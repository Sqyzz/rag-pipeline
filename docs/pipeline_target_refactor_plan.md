# 对比 Pipeline 目标改造实施文档

## 1. 背景与目标

基于 `对比pipeline目标.md`，本次改造目标是让项目评估流程满足以下要求：

1. 多方法统一对比（VectorRAG / KG-RAG / GraphRAG / youtu-GraphRAG）
2. 双模式评估（Open / Reject）
3. Top-10 / Top-20 Accuracy
4. Token / Time 统计
5. Pareto Frontier（Token vs Accuracy）
6. 标准报告表输出（`Method | Dataset | Mode | Top20 | Top10 | Tokens | Time`）


## 2. 当前状态与缺口

### 2.1 已具备能力

1. `run_compare.py` 已支持三方法统一运行，并可通过 `--include-youtu` 接入 youtu 方法。
2. 已有两种 regime（`best_effort` / `budget_matched`）与 token/latency 聚合。
3. `run_eval.py` 已支持 compare 布局评估并输出 csv/json 汇总。

### 2.2 关键缺口

1. 缺少 `Open / Reject` 回答策略模式（当前是预算模式，不是回答策略模式）。
2. 缺少一次执行内的 `Top-10 / Top-20` 双结果汇总。
3. 缺少严格定义的 LLM Judge（Yes/No 语义等价判定）。
4. 缺少 Pareto 自动产出。
5. 缺少目标表头的最终报告导出。


## 3. 改造总览（实施顺序）

1. 新增回答策略模式（Open / Reject）
2. 增加 Top-K 双档执行与汇总
3. 增加 LLM Judge 评估通道
4. 增加 Pareto 与最终报告表导出
5. 将 youtu 方法纳入统一统计与报告
6. 补测试与文档


## 4. 具体改动清单

### 4.1 回答策略模式（Open / Reject）

#### 改动目标

让三种 baseline 以及 youtu 方法在同一评估协议下支持两种可切换回答行为：

1. `reject`：证据不足时拒答（保持当前 `NOT_FOUND` 逻辑）
2. `open`：证据不足时可基于常识补全回答

#### 代码改动

1. 修改 `src/baselines/vector_rag.py`
2. 修改 `src/baselines/kg_rag.py`
3. 修改 `src/baselines/graph_rag.py`
4. 修改 `src/baselines/youtu_graph_rag_adapter.py`（后端支持时透传 answer_mode）

#### 改动内容

1. 在各 `answer_*` 入口新增参数：`answer_mode: str = "reject"`。
2. 统一 prompt 模板：
   - `reject` 模板包含“证据不足返回固定拒答词”。
   - `open` 模板包含“证据不足可基于自身知识回答”。
3. 保持 `query_type=global_summary` 与普通 QA 分支都支持两种 mode。
4. 在返回 payload 中增加 `answer_mode` 字段，便于评估分组。

#### 验收标准

1. 三种 baseline 均可通过参数切换回答策略。
2. youtu 方法在后端支持时可切换；不支持时在结果中显式标记。
3. 同一 query 在 `reject/open` 下输出可区分。
4. 不影响原默认行为（默认仍可与当前结果对齐）。


### 4.2 run_compare 接入 mode 维度

#### 改动目标

让 compare 输出能同时保留评估模式维度（Open / Reject）。

#### 代码改动

1. 修改 `src/experiments/run_compare.py`

#### 改动内容

1. 新增 CLI 参数：
   - `--answer-mode`（`reject|open`，默认 `reject`）
   - 可选：`--answer-modes`（`reject,open`）用于一次跑双模式
2. 将 mode 透传到 `_run_vector/_run_kg/_run_graph/_run_youtu_branch`。
3. 输出结构中增加：
   - 行级 `mode`
   - summary 元信息 `answer_mode` 或 `answer_modes`
4. 兼容旧格式（单 mode 运行时不破坏现有字段读取）。

#### 验收标准

1. `compare_answers.jsonl` 可按 mode 区分结果。
2. `compare_metrics.json` 包含 mode 元信息。
3. `aggregate_metrics` 与 `aggregate_metrics_by_type` 在 `--include-youtu` 时包含 youtu 方法。


### 4.3 Top-10 / Top-20 双档评估

#### 改动目标

实现目标文档要求的 Top-10、Top-20 对比口径。

#### Top-k Accuracy 定义（最终口径）

`TopK_Accuracy = mean( I[answer_correct = 1 AND support_at_k = 1] )`

其中：

1. `support_at_k`：
   - 定义为“top-k 检索证据支持命中”。
   - 实现口径：`evidence_recall_docs > 0` 记为 1，否则为 0。
2. `answer_correct`：
   - 优先使用 `answer_semantic_yesno`（LLM Judge，0/1）。
   - 若未启用 Judge，则回退到 `answer_exact_relaxed`（0/1）。
3. 行级可解释字段（建议落表）：
   - `topk_support`（0/1）
   - `topk_correct`（0/1）
   - `topk_accuracy`（0/1，等于前两者乘积）

#### 代码改动（推荐新增脚本）

1. 新增 `src/experiments/run_compare_topk_modes.py`

#### 改动内容

1. 脚本顺序执行以下组合：
   - `top_k=10`
   - `top_k=20`
2. 每个 top_k 可按 `answer_mode` 执行（建议 `reject/open` 两次或一次双模式）。
3. 自动调用 `run_eval.py` 产出每组评估结果。
4. 汇总为统一宽表（含 youtu 方法）：
   - `method, dataset, mode, top10_accuracy, top20_accuracy, tokens, time`

#### 验收标准

1. 单次脚本可产出 Top10/Top20 对照结果。
2. 结果可直接用于论文表格。
3. `top10_accuracy/top20_accuracy` 的计算完全符合“support_at_k AND answer_correct”定义。


### 4.4 LLM Judge 语义判分通道

#### 改动目标

引入 Yes/No 语义等价判分，满足目标文档的 Judge 口径。

#### 代码改动

1. 新增 `src/evaluation/judges.py`
2. 修改 `src/evaluation/run_eval.py`

#### 改动内容

1. `judges.py` 新增函数：
   - `semantic_equivalent_yes_no(pred, gold, model=...) -> 0/1`
2. prompt 采用文档定义：
   - “Are the following two answers semantically equivalent?... Respond only Yes or No.”
3. `run_eval.py` 新增参数：
   - `--judge-mode`（`off|llm_yesno`）
   - `--judge-model`（默认 `qwen-flash`）
4. 每行增加：
   - `answer_semantic_yesno`（0/1）
5. summary 增加：
   - 按 `regime/method/mode/top_k`（含 youtu）聚合 yesno 准确率
6. 默认关闭 judge，避免影响旧实验可复现性。

#### 验收标准

1. judge 可独立开关。
2. judge 模型与生成模型可配置分离。
3. 输出存在稳定 0/1 指标列。


### 4.5 Pareto Frontier 与最终报告导出

#### 改动目标

产出目标文档要求的成本-性能图与标准表。

#### 代码改动

1. 新增 `src/evaluation/export_report.py`

#### 改动内容

1. 读取 Top10/Top20 汇总与 metrics 文件。
2. 输出：
   - `outputs/results/final_report.csv`
   - `outputs/results/final_report.md`
3. 表结构固定：
   - `Method | Dataset | Mode | Top20 | Top10 | Tokens | Time`
4. 绘图：
   - `outputs/results/pareto_frontier.png`
   - x 轴 `Tokens`
   - y 轴 `Accuracy`（默认 Top20，可参数切换）

#### 验收标准

1. 报告表字段完整且可复现。
2. Pareto 图可按 method/mode 区分点位。


### 4.6 配置与参数治理

#### 改动目标

保证“统一模型设置”的可追踪与可复现实验。

#### 代码改动

1. 修改 `config.yaml`
2. 修改 `docs/comparison_pipeline.md`

#### 改动内容

1. 增加评估配置段：
   - `evaluation.answer_modes`
   - `evaluation.topk_list`
   - `evaluation.judge.model`
   - `evaluation.judge.enabled`
   - `evaluation.include_youtu`
   - `evaluation.method_mode`（`all|exclude_youtu|only_youtu`）
2. 文档补充推荐命令与最小复现实验命令。

#### 验收标准

1. 所有关键参数可在 config/CLI 中显式记录。
2. 复现实验无需改源码。


### 4.7 测试补充

#### 改动目标

保证新能力不会破坏现有流程。

#### 代码改动

1. 新增 `tests/test_answer_mode_prompts.py`
2. 新增 `tests/test_eval_judge_yesno.py`
3. 修改 `tests/test_run_eval_alignment.py`（必要时）
4. 新增 `tests/test_compare_with_youtu_mode_propagation.py`

#### 测试点

1. `reject/open` prompt 分支选择正确。
2. compare 输出包含 mode 元信息。
3. eval 在 `judge_mode=llm_yesno` 下可写出新列。
4. Top10/Top20 汇总逻辑正确。
5. `--include-youtu` 时汇总报表包含 youtu 方法。


## 5. 交付物清单

1. 新脚本：`run_compare_topk_modes.py`
2. 新模块：`evaluation/judges.py`
3. 新模块：`evaluation/export_report.py`
4. 更新：三个 baseline 的 answer_mode 支持
5. 更新：`run_compare.py` mode 透传与输出
6. 更新：`run_eval.py` judge 支持与 youtu 聚合兼容
7. 文档：本实施文档 + comparison_pipeline 更新
8. 测试：新增与更新测试文件


## 6. 推荐执行命令（改造后）

```bash
# 1) 跑对比（Top10 + reject）
python src/experiments/run_compare.py \
  --queries-file data/queries/cuad_capability_queries.jsonl \
  --top-k 10 \
  --include-youtu \
  --answer-mode reject \
  --out-file outputs/results/compare_answers_top10_reject.jsonl \
  --metrics-file outputs/results/compare_metrics_top10_reject.json

# 2) 跑评估（启用 LLM Judge）
python src/evaluation/run_eval.py \
  --pred-file outputs/results/compare_answers_top10_reject.jsonl \
  --gold-file data/queries/cuad_capability_gold.jsonl \
  --out-csv outputs/results/eval_top10_reject.csv \
  --out-summary outputs/results/eval_top10_reject_summary.json \
  --judge-mode llm_yesno \
  --judge-model qwen-flash
```


## 7. 风险与注意事项

1. LLM Judge 成本较高，建议支持缓存与增量评估。
2. Open 模式会引入外部知识，需在论文中单独标注“非纯检索约束”。
3. Top10/Top20 若各自独立运行，需固定随机性与数据切片保证可比性。
4. 旧结果兼容：新增字段不应破坏既有解析脚本。

### 7.1 影响最终结果的口径问题（需优先修复）

1. `global_summary` 主分数的复合项必须使用“文档级召回（docs recall）”而非社区召回占位值，否则主分数会系统性偏低。
2. 评估去重键必须包含 `mode/top_k` 维度，否则多模式或多 top_k 合并评估时会被覆盖。
3. 报告中的 “Accuracy” 必须统一引用上述 `TopK_Accuracy` 定义，不得混用 `answer_similarity` 或其他分数替代。


## 8. 里程碑建议（2-3 天）

1. Day 1：实现 answer_mode + run_compare 透传 + 基础测试
2. Day 2：实现 run_eval judge + TopK 汇总脚本
3. Day 3：报告导出 + Pareto + 文档收口
