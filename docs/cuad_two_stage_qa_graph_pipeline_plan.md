# CUAD 两阶段 QA / 图谱流水线最终实施计划

## 1. 文档目的

本文档用于基于**当前项目代码现状 + 已有实验结论**，重新定义 CUAD 数据集上的最终实施路线。

这里先明确一个前提：

- 当前仓库里确实已经有一条“直接基于最终 `graph + communities` 生成三类 QA”的现成实现；
- 但你已经实际跑过这条整体对齐路线，结果显示 `global` 类型 QA 的表现**异常差**，且不是个别方法差，而是**几乎所有方法的正确率都接近 0**；
- 因此，是否采用某条路线，不能只看“代码里是否已经实现”，而必须看它生成的评测集是否**可评估、可区分、可解释**。

基于这一点，本文档给出的最终结论是：

1. 当前仓库中现成的 `build_cuad_capability_qa.py` 路线，**可以保留为参考实现/对照线**；
2. 但它**不能直接被认定为最终可信主线**，尤其不能直接作为 `global_summary` 的最终标准生成方案；
3. 最终推荐主线应恢复为你最初提出的**两阶段对齐方案**：
   - 先生成不依赖社区的 `local/cross`
   - 用这批 QA 对齐并约束最终 docs 子集
   - 再基于该 docs 子集构图、建社区
   - 最后再从**最终 community** 生成 `global`
   - 必要时再将 `global` 增量合并回 `local/cross`

一句话版：

**之所以保留两阶段主线，不是为了再造一条支线，而是因为现成“整体对齐路线”已经在实验上暴露出 `global` 几乎全 0 的系统性问题。**


## 2. 当前实现现状与实验事实

当前项目中和 CUAD QA 构造最相关的脚本有四个：

### 2.1 `src/evaluation/build_cuad_capability_qa.py`

作用：

- 基于 `graph.json + communities.json`
- 直接生成：
  - `local_factual`
  - `cross_clause`
  - `global_summary`

特点：

- 现成可跑
- 与当前 `run_eval.py` 的 chunk-based evidence 口径兼容
- `global_summary` 会带：
  - `supporting_communities`
  - `supporting_chunks`

问题：

- 从“代码可运行”角度它是完整的；
- 但从“实验效果”角度，它已经暴露出 `global` 几乎全方法接近 0 的问题；
- 因此它更适合被视为：
  - **当前实现参考线**
  - **对照实验线**
  - 而不是默认最终主线

### 2.2 `src/evaluation/build_cuad_question_converter.py`

作用：

- 从 `train_separate_questions.json` 抽取/转换 CUAD 原始 QA
- 可输出 `local/cross/global` 类型问题

价值：

- `local/cross` 不依赖社区，因此适合做“前置 QA 集”
- 可用于先确定文档覆盖范围
- 可作为两阶段流程的第一阶段输入

限制：

- 它直接生成的 `global` 并不依赖最终采样图上的最终 communities
- 因此不能直接作为最终 GraphRAG `global` 评测集

### 2.3 `src/ingestion/build_cuad_aligned_docs.py`

作用：

- 从 QA 文件中抽取 `meta.query_doc_key` / `meta.title`
- 从 `CUADv1.json` 中保留全部 matched docs
- 再按 `--matched-doc-ratio` 补充 unmatched docs

价值：

- 可把文档集约束到和前置 QA 集一致的覆盖范围
- 是两阶段路线里的关键桥接环节

### 2.4 `src/evaluation/rewrite_global_summary_gold.py`

作用：

- 对 `global_summary` 的 gold answer 做基于 `supporting_chunks` 的重写

价值：

- 可以改善 `global_summary` 的 gold 文本质量
- 但它不能从根本上修复“问题定义、证据定义、评分定义”三者不一致的问题


## 3. 为什么必须保留两阶段主线

这部分是本文档最关键的结论。

如果当前“整体对齐路线”已经跑过，并且观察到：

- `global_summary` 对几乎所有方法都接近 0；
- 这种低分不是某一个方法特有，而是普遍现象；

那就说明问题大概率不在单个方法，而在**评测集构造本身**。

### 3.1 这不是“方法弱”，而是“任务定义错位”

当所有方法在同一类题上都接近全灭时，最常见的原因不是：

- 所有方法都同样差

而是：

- 题目定义的抽象层级过高
- gold 答案形式与可检索证据不匹配
- evidence 口径与问题口径不一致
- 社区本身不稳定，导致所谓 `global` 题没有真实、稳定的语义锚点

也就是说，当前 `global` 的问题不是“难”，而是**失配**。

### 3.2 当前 `global_summary` 很可能同时存在三层失配

#### 失配一：问题来源过于抽象

当前 `global_summary` 是从 community summary 反推出来的。这意味着问题本身是高层主题概括型问题。

但 CUAD 合同图中的 community 未必真的稳定承载“高层主题”。如果 community 本身语义就弱，那么 `global` 题会天然漂浮。

#### 失配二：gold 仍主要依赖 chunk 证据

当前评测主口径仍主要落在：

- `supporting_chunks`
- `evidence_recall_chunks`

这意味着：

- 问题是“社区级总结题”
- 但主证据仍然是“chunk 级局部证据”

这会导致评测对象被迫同时完成：

- 社区主题理解
- 证据聚合
- 抽象总结生成

但 gold 却未必严格可由这些 chunk 稳定推出。

#### 失配三：答案文本和证据文本不在同一层级

若 gold answer 更像 community summary 或经过重写后的抽象总结，而不是与支持 chunk 高度对齐的标准答案，那么：

- 命中相关 chunk 不代表能答对
- 召回正确证据不代表语义相似度高
- 最终会出现“方法看起来都不行”的假象

### 3.3 两阶段路线的意义是修复这个失配

两阶段路线不是为了增加复杂度，而是为了把 `global` 的定义重新锚定到**最终真实要评估的图结构**上：

1. `local/cross` 先从原始 QA 中生成
   - 因为它们本身不依赖 community
   - 可以先作为评测集骨架和文档覆盖约束

2. 再根据这些 QA 去反筛 docs
   - 保证最终图不会偏离真正要评估的文档集合

3. 再基于这份 docs 子集构图并建社区
   - 让后续 `global` 依赖的 community 来自最终真实图库

4. 最后再从最终 community 生成 `global`
   - 避免“global 提前生成、却和最终图不一致”的问题
   - 也为后续单独控制 `global` 的问题形式和 gold 形式提供空间

因此，两阶段路线不是支线，而是**为了解决 `global` 评测失真而必须保留的主线**。


## 4. 最终推荐的主线流程

### Phase 1：先生成 `local/cross`

输入：

- `data/raw/cuad/train_separate_questions.json`

建议脚本：

- `src/evaluation/build_cuad_question_converter.py`

目标：

- 只生成不依赖社区的两类题：
  - `local_factual`
  - `cross_clause`

要求：

- 这一阶段不把 `global` 作为最终产物使用；
- 即便脚本当前具备 `global` 生成能力，最终主流程里也应**显式关闭或忽略**这部分输出；
- 本阶段生成的 QA 主要用途是：
  - 形成前置评测骨架
  - 确定文档覆盖范围

期望输出：

- `queries_local_cross.jsonl`
- `gold_local_cross.jsonl`

### Phase 2：按 `local/cross` 的文档集合对齐 docs

输入：

- `data/raw/cuad/CUADv1.json`
- `queries_local_cross.jsonl`

建议脚本：

- `src/ingestion/build_cuad_aligned_docs.py`

目标：

- 保留与 `local/cross` QA 对应的全部 matched docs
- 再按比例补采 unmatched docs

为什么必须做这一步：

- 防止最终图脱离真正被评测的文档集合
- 在保留 QA 对齐性的同时，尽量保留图的结构完整性

期望输出：

- `qa_aligned_docs.jsonl`

建议保留统计：

- matched docs 数量
- unmatched docs 数量
- 实际 matched ratio
- 最终 paragraph docs 数量

### Phase 3：docs -> chunks -> triples -> graph -> communities

输入：

- `qa_aligned_docs.jsonl`

目标：

- 在最终评测用的文档子集上构建完整图谱与社区

典型输出：

- `chunks.jsonl`
- `triples.jsonl`
- `graph.json`
- `communities.json`

说明：

- 当前实现下仍然不建议跳过 chunk 层；
- 原因不是 chunk 必须很细，而是必须保留稳定的 evidence unit；
- `supporting_chunks`、`evidence_recall_chunks`、community 到原文的映射都依赖 chunk。

### Phase 4：基于最终 communities 单独生成 `global`

输入：

- `graph.json`
- `communities.json`
- `chunks.jsonl`

目标：

- 只生成真正依赖**最终社区结构**的 `global_summary`

这一阶段的核心要求：

- `global` 的来源必须是最终 community，而不是原始 QA 转换阶段；
- `global` 的问题设计应尽量避免不可回答的“纯抽象大总结题”；
- `global` 的 gold answer 必须尽量与 `supporting_chunks` 可支撑的内容同层级；
- 每条 `global` 必须至少带：
  - `supporting_communities`
  - `supporting_chunks`

实现建议：

- 可以新增专用脚本，例如：`src/evaluation/build_cuad_global_from_communities.py`
- 也可以在现有 `build_cuad_capability_qa.py` 基础上拆出单独的 `global` 生成逻辑
- 但无论采用哪种实现，**它在流程上的角色都应独立于 Phase 1**

期望输出：

- `queries_global.jsonl`
- `gold_global.jsonl`

### Phase 5：合并三类 QA

输入：

- `queries_local_cross.jsonl`
- `gold_local_cross.jsonl`
- `queries_global.jsonl`
- `gold_global.jsonl`

目标：

- 合并成最终评测集

要求：

- `qid` 不冲突
- 类型集合固定为：
  - `local_factual`
  - `cross_clause`
  - `global_summary`
- `global` 的 evidence 字段结构与最终 eval 脚本兼容

期望输出：

- `queries.jsonl`
- `gold.jsonl`

说明：

- 这一步虽然在当前仓库里没有作为主线脚本现成存在，但从实验逻辑上是必要的；
- 它不是“人为拼接两个不一致的数据源”，而是把：
  - 不依赖社区的前置题
  - 依赖最终社区的后置题
  合并成一套自洽评测集。


## 5. 实施清单

本节把上面的主线流程拆成真正可执行的实施清单。顺序上分为：

- 第一部分：文档与脚本改造
- 第二部分：数据构建与产物验证
- 第三部分：最终交付物

### 5.1 脚本改造清单

#### Task 1：限制 `build_cuad_question_converter.py` 只产出 `local/cross`

目标：

- 支持显式指定输出类型
- 在两阶段主线里可稳定关闭 `global`

实施项：

- 增加或确认一个类型过滤参数，例如：
  - `--types local_retrieval,structural_reasoning`
- 保证输出类型名可以映射到：
  - `local_factual`
  - `cross_clause`
- 保证关闭 `global` 时：
  - 不生成 `global_summary`
  - 不遗留空行或错位 `qid`

完成标准：

- 单独运行 converter 时，可只输出 `local/cross`
- `queries_local_cross.jsonl` 与 `gold_local_cross.jsonl` 行数一致

#### Task 2：确认 `build_cuad_aligned_docs.py` 满足两阶段需要

目标：

- 让 docs 对齐逻辑足够稳定并可追踪

实施项：

- 确认脚本优先从 `meta.query_doc_key` 取文档 key
- 缺失时回退 `meta.title`
- 保留现有：
  - matched 全保留
  - unmatched 按比例补采
- 在输出 summary 中保留：
  - `num_selected_docs`
  - `num_selected_matched_docs`
  - `num_selected_unmatched_docs`
  - `actual_matched_doc_ratio`

完成标准：

- 输入 `queries_local_cross.jsonl` 后，能稳定得到 `qa_aligned_docs.jsonl`
- summary 能直接用于记录实验配置

#### Task 3：新增社区版 `global` 生成脚本

目标：

- 把最终 `global` 从现有一体化生成器里独立出来

建议脚本：

- `src/evaluation/build_cuad_global_from_communities.py`

最小职责：

- 读取：
  - `graph.json`
  - `communities.json`
  - `chunks.jsonl` 或可等价恢复 chunk 文本的输入
- 产出：
  - `queries_global.jsonl`
  - `gold_global.jsonl`
- 每条 `global_summary` 至少包含：
  - `qid`
  - `type`
  - `query`
  - `answer`
  - `meta`
  - `supporting_communities`
  - `supporting_chunks`

必须加入的控制项：

- community 过滤规则
- supporting chunk 质量过滤规则
- doc consistency / evidence consistency 规则
- gold answer 长度与抽象度控制

完成标准：

- 能单独运行生成 `global_summary`
- 输出格式与 `run_eval.py` 兼容
- 样本不再是明显不可回答的纯抽象题

#### Task 4：新增 QA 合并脚本

目标：

- 将前置 `local/cross` 与后置 `global` 合并为最终评测集

建议脚本：

- `src/evaluation/merge_qa_sets.py`

最小职责：

- 合并 queries 文件
- 合并 gold 文件
- 检查 `qid` 冲突
- 检查 queries / gold 的 `qid` 一一对应
- 输出最终：
  - `queries.jsonl`
  - `gold.jsonl`

完成标准：

- 合并后类型分布正确
- 不出现重复 `qid`
- 不出现 query/gold 缺口

#### Task 5：补充必要测试

目标：

- 保证两阶段流程的关键环节可回归

优先测试项：

- converter 仅输出 `local/cross`
- aligned docs 对齐结果与 QA 文档 key 一致
- global 生成脚本输出包含 `supporting_communities + supporting_chunks`
- merge 脚本对 `qid` 冲突有保护

完成标准：

- 至少覆盖新增脚本与新增分支逻辑
- 不要求一次把全流程端到端全测齐，但关键环节必须可测

### 5.2 数据构建清单

#### Task 6：构建前置 `local/cross` QA

产物：

- `queries_local_cross.jsonl`
- `gold_local_cross.jsonl`

检查项：

- 类型只包含：
  - `local_factual`
  - `cross_clause`
- query / gold 行数一致
- `meta.query_doc_key` 尽可能完整

#### Task 7：基于前置 QA 对齐 docs

产物：

- `qa_aligned_docs.jsonl`
- 对齐统计 summary

检查项：

- 所有 QA 对应文档都在 docs 子集中
- 文档总量不会小到让图完全退化

#### Task 8：在对齐 docs 上重建图谱与社区

产物：

- `chunks.jsonl`
- `triples.jsonl`
- `graph.json`
- `communities.json`

检查项：

- 图不是空图
- community 数量不为 0
- 社区摘要不是明显噪声

#### Task 9：生成社区版 `global`

产物：

- `queries_global.jsonl`
- `gold_global.jsonl`

检查项：

- 每条 global 都有 `supporting_communities`
- 每条 global 都有足够支撑的 `supporting_chunks`
- gold answer 与 supporting chunk 处于同层级

#### Task 10：合并最终评测集

产物：

- `queries.jsonl`
- `gold.jsonl`

检查项：

- 三类题都存在
- `qid` 无冲突
- queries/gold 完整对齐

### 5.3 评估与验收清单

#### Task 11：跑一轮 compare / eval

目标：

- 用新数据集验证 `global` 是否仍系统性塌陷

核心观察项：

- `local_factual` 是否正常
- `cross_clause` 是否正常
- `global_summary` 是否仍几乎全方法接近 0

#### Task 12：如果 `global` 仍塌陷，继续回调生成逻辑

优先调整项：

- community 过滤阈值
- supporting chunk 选取规则
- gold answer 抽象度
- 问题模板

停止条件：

- `global_summary` 不再出现明显“全方法近 0”
- 或至少不同方法之间开始出现可解释的分层

### 5.4 最终交付清单

最终交付物应包含：

- 两阶段主线脚本改造结果
- 最终评测集产物
- 一版可复现的操作指南

其中最后一项必须是：

- **新版流程操作指南**


## 6. 当前现成实现的正确定位

### 6.1 `build_cuad_capability_qa.py`

最终定位：

- 参考实现
- 对照线
- 可复用其局部逻辑

不应直接视为：

- 最终唯一主线
- 尤其不应直接把其 `global_summary` 产物视为最终可信的 `global` 标准集

原因很简单：

- 你已经有实验结果表明，这条线在 `global` 上产生了系统性塌陷

### 6.2 `build_cuad_question_converter.py`

最终定位：

- 两阶段主线的第一阶段工具
- 用于生成前置 `local/cross`
- 不再承担最终 `global` 生成职责

### 6.3 `build_cuad_aligned_docs.py`

最终定位：

- 两阶段主线中的 docs 对齐桥梁
- 必须保留

### 6.4 `rewrite_global_summary_gold.py`

最终定位：

- 可选后处理工具
- 仅用于改善已有 `global` 答案文本
- 不能替代“重新定义 `global` 的生成逻辑”


## 7. `global` 几乎全 0 说明了什么

这部分应写进最终计划，而不是只作为实验备注。

如果观察到：

- 多种方法在 `global_summary` 上都接近 0
- 而 `local/cross` 并没有同样程度的整体崩塌

那么更合理的解释是：

### 7.1 不是单个方法失败，而是 `global` QA 构造存在系统性问题

这意味着当前 `global` 更像是在测：

- 社区摘要重写能力
- 高层抽象能力
- 对 gold phrasing 的偶然贴合度

而不是在测：

- 基于图与证据的真实全局推理能力

### 7.2 当前 `global` 很可能缺乏稳定可回答性

如果一个 `global` 问题：

- 无法被稳定地从 `supporting_chunks` 支撑
- 或不同方法即使命中相近证据也答不出接近 gold 的文本

那它就不是一个好的主评测题。

### 7.3 最终实施计划必须把“防止 global 塌陷”写成硬约束

因此，后续 `global` 生成方案必须加入至少以下验收：

- 问题是否由最终 community 稳定支撑
- `supporting_chunks` 是否足够支撑 gold
- gold 是否与证据文本处于同一语义层级
- 是否出现“全方法接近 0”的系统性崩塌

如果仍然出现这类崩塌，就说明 `global` 的生成逻辑还不能作为最终版。


## 8. 评测口径如何与新主线对齐

### 8.1 `local_factual`

保持当前主口径即可：

- 以 `supporting_chunks` 为主 evidence
- 辅以 `supporting_edges`

### 8.2 `cross_clause`

保持当前主口径即可：

- 以多 chunk 证据组合为主
- gold 需要可由对应 edge / chunk 组合支撑

### 8.3 `global_summary`

这里不能只说“沿用当前实现”，而必须明确：

- 问题来源：最终 communities
- 主证据：仍可保持 chunk 口径，以兼容现有 `run_eval.py`
- 但 gold 答案必须回到 chunk 可支撑、可验证、可比较的层级
- `supporting_communities` 作为辅助诊断指标保留

也就是说，最终目标不是放弃当前 eval 框架，而是：

- 在保持 eval 主框架兼容的前提下
- 把 `global` QA 本身重新定义得更可回答、更稳定


## 9. 新版流程操作指南

本节是实施的最后一步交付物，用于把两阶段方案落成一套可复现操作流程。

### 9.1 前置准备

确保以下输入文件可用：

- `data/raw/cuad/train_separate_questions.json`
- `data/raw/cuad/CUADv1.json`

确保以下目录存在或可自动创建：

- `data/queries`
- `data/processed`
- `outputs/graph`

### 9.2 Step 1：生成前置 `local/cross`

目标：

- 只生成不依赖 community 的前置 QA

建议命令形态：

```bash
python src/evaluation/build_cuad_question_converter.py \
  --cuad-train-file data/raw/cuad/train_separate_questions.json \
  --type-name-scheme capability \
  --types local_factual,cross_clause \
  --out-queries-file data/queries/queries_local_cross.jsonl \
  --out-gold-file data/queries/gold_local_cross.jsonl \
  --total-per-type 2
```

本步产物：

- `data/queries/queries_local_cross.jsonl`
- `data/queries/gold_local_cross.jsonl`

本步检查：

- 类型只包含 `local_factual` 和 `cross_clause`
- query / gold 行数一致

### 9.3 Step 2：按前置 QA 对齐 docs

建议命令：

```bash
python src/ingestion/build_cuad_aligned_docs.py \
  --cuad-file data/raw/cuad/CUADv1.json \
  --qa-file data/queries/queries_local_cross.jsonl \
  --out-docs-file data/processed/qa_aligned_docs.jsonl \
  --matched-doc-ratio 0.5 \
  --random-seed 42 \
  --split-name qa_aligned
```

本步产物：

- `data/processed/qa_aligned_docs.jsonl`

本步检查：

- summary 中 `actual_matched_doc_ratio` 合理
- QA 文档在 docs 子集中全部可找到

### 9.4 Step 3：在对齐 docs 上构建 chunks

建议命令形态：

```bash
python src/ingestion/chunking.py \
  --in-file data/processed/qa_aligned_docs.jsonl \
  --out-file data/processed/qa_aligned_chunks.jsonl \
  --chunk-size 1000 \
  --overlap 120
```

本步产物目标：

- `data/processed/qa_aligned_chunks.jsonl`

本步检查：

- chunk 数量非 0
- `chunk_id` 可稳定回溯到原文

### 9.5 Step 4：构建 triples / graph / communities

建议命令形态：

```bash
python src/graph_build/extract_triples.py \
  --chunks-file data/processed/qa_aligned.jsonl \
  --out-file outputs/graph/qa_aligned_triples.jsonl \
  --progress-every 10 \
  --mode per_chunk \
  --concurrency 10 \
  --schema-file config_triple_schema_cuad_core_v2.json \
  --batch-size 1 \
  --schema-apply-mode strict

python src/graph_build/build_graph.py \
  --triples-file outputs/graph/qa_aligned_triples.jsonl \
  --out-file outputs/graph/qa_aligned_graph.json

python src/graph_build/build_communities.py \
  --graph-file outputs/graph/qa_aligned_graph.json \
  --out-file outputs/graph/qa_aligned_communities.json
```

本步产物：

- `outputs/graph/qa_aligned_triples.jsonl`
- `outputs/graph/qa_aligned_graph.json`
- `outputs/graph/qa_aligned_communities.json`

本步检查：

- 图不是空图
- community 数量大于 0
- community summary 不是明显噪声

### 9.6 Step 5：生成最终社区版 `global`

建议命令形态：

```bash
python src/evaluation/build_cuad_global_from_communities.py \
  --graph-file outputs/graph/qa_aligned_graph_v3_docscoped.json \
  --communities-file outputs/graph/qa_aligned_communities_v3_docscoped_pruned.json \
  --chunks-file data/processed/qa_aligned_chunks.jsonl \
  --queries-out data/queries/queries_global.jsonl \
  --gold-out data/queries/gold_global.jsonl \
  --question-style llm
```

本步产物：

- `data/queries/queries_global.jsonl`
- `data/queries/gold_global.jsonl`

本步检查：

- 每条 global 都有 `supporting_communities`
- 每条 global 都有 `supporting_chunks`
- gold answer 不再是脱离证据的纯抽象总结

### 9.7 Step 6：合并最终 QA 集

建议命令形态：

```bash
python src/evaluation/merge_qa_sets.py \
  --query-files data/queries/queries_local_cross.jsonl,data/queries/queries_global.jsonl \
  --gold-files data/queries/gold_local_cross.jsonl,data/queries/gold_global.jsonl \
  --out-queries-file data/queries/queries_final.jsonl \
  --out-gold-file data/queries/gold_final.jsonl
```

本步产物：

- `data/queries/queries_final.jsonl`
- `data/queries/gold_final.jsonl`

本步检查：

- 类型集合完整
- `qid` 无冲突
- queries/gold 一一对应

### 9.8 Step 7：进入 compare / eval

将最终输入切换为：

- `data/queries/queries_final.jsonl`
- `data/queries/gold_final.jsonl`

核心检查：

- `local_factual` 正常可评估
- `cross_clause` 正常可评估
- `global_summary` 不再出现全方法接近 0 的系统性塌陷

### 9.9 建议记录的实验日志字段

每次跑新版本流程，建议至少记录：

- converter 参数
- matched doc ratio
- docs / chunks / graph / communities 规模
- global 过滤规则
- 最终三类 QA 数量
- 各类型主分与 evidence recall

### 9.10 新版流程的最终判断标准

只有同时满足以下条件，才认为新版流程可用：

- 三类 QA 都能稳定生成
- `global_summary` 具备可回答性
- `global_summary` 不再系统性塌陷
- 不同方法在 `global_summary` 上开始出现可解释差异


## 10. 最终验收标准

这套两阶段主线最终落地后，应满足：

1. `local/cross` 来自不依赖社区的前置生成逻辑。
2. 最终 docs 集与 `local/cross` 的文档集合严格对齐。
3. `global` 来自最终采样图上的最终 communities，而不是前置 converter 阶段提前生成。
4. `global` 的 gold 至少包含：
   - `supporting_communities`
   - `supporting_chunks`
5. `global` 的 gold answer 与 supporting evidence 处于同一可验证语义层级。
6. 最终三类 QA 合并后，可直接进入 compare / eval。
7. 采样后的图仍需通过 `docs/cuad_sampled_graph_community_checklist.md` 的社区退化检查。
8. 最关键的一条：`global_summary` 不再出现“几乎所有方法正确率都接近 0”的系统性塌陷。


## 11. 一句话结论

最终实施计划应当以**两阶段对齐方案**为主线；保留它不是为了额外造一条不稳定支线，而是因为现成整体路线已经在实验上证明：当前 `global` 的定义方式会把评测推向系统性失真。
