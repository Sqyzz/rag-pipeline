# Youtu CUAD Retrieval Root Cause Analysis

本文档记录当前 `youtu-graphrag` 在 `CUAD` smoke 结果中的根因判断，重点回答两个问题：

1. 当前问题是否主要来自 `chunk` 过多导致的后续污染？
2. 如果不是，问题更可能出在子问题拆分，还是检索引擎本身？

相关结果文件：

- `outputs/results/ragas/edge_layer_smoke_eval_t_ircot/ragas_eval_summary.csv`
- `outputs/results/ragas/edge_layer_smoke_eval_t_ircot_docconsistency/ragas_eval_summary.csv`
- `outputs/results/ragas/edge_layer_youtu_smoke_t_ircot.jsonl`
- `outputs/results/ragas/edge_layer_youtu_smoke_t_ircot_docconsistency.jsonl`

## 1. 结论先行

当前主问题不是“明明已经检到正确证据，只是在最终 10 个 chunk 里被污染挤掉”。

更接近事实的判断是：

- 很多题在**子问题级检索阶段就没有把正确证据稳定检回来**
- `chunk` 偏多确实会放大问题
- 但 `chunk` 偏多更像**放大器**，不是根因

因此，当前问题的主叙事应从：

- “final chunk selection 不够好”

转为：

- “子问题 query formulation 与子问题级 rerank 还不足以稳定命中 gold clause / gold contract”

## 2. 为什么说不是 final-stage 污染主导

如果问题主要是后续污染，应该能看到这种模式：

1. 子问题检索结果里已经有明显正确证据
2. 子问题答案本身基本正确
3. 最终 10 个 chunk 没保住，才导致最终答偏

但当前样本里更常见的是：

1. 子问题检索结果本身就偏
2. 子问题答案已经偏或 `NOT_FOUND`
3. 最终回答只是忠实复述前面的偏差

因此：

- 最终聚合不是当前主故障点
- final-stage `doc-consistency bias` 也不该作为主修复方向

## 3. 样本级判断

### 3.1 `ragas-cuad-0002`

问题：

- `What is Motorola's role in the PageMaster Corporation promotion for paging services?`

gold：

- Motorola 是促销中免费 pager 的提供方

当前现象：

- 子问题检索里确实出现了 `Motorola`、`PageMaster`
- 但高位 chunk 更偏：
  - trademark/license
  - indemnity
  - agreement boilerplate

判断：

- **local 子问题 query 没有把“promotion role / device provider”收紧到足够具体**
- 检索命中了“相关语义”，但没命中 gold clause

结论：

- 这不是 final chunk selection 的问题
- 是 **local 检索排序能力不足**

### 3.2 `ragas-cuad-0003`

问题：

- `What are the responsibilities of PageMaster Corporation during the promotion period?`

gold：

- 建立并维护 toll-free 电话等职责

当前现象：

- 检索里确实命中了 promotion period 的正确日期片段
- 但同时混入多条其他职责/条款
- 回答最终抓到的是“相关职责”，不是 gold 的主职责条款

判断：

- **local / structural 子问题都拿到了部分相关内容**
- 但没把 gold 所需 clause 稳定排到最前

结论：

- 问题仍然主要在 **子问题级检索排序**
- 不是最终 context 太大才坏掉

### 3.3 `ragas-cuad-0041`

问题：

- Metavante 第三方费用定义
- 与 Neoforma indemnification 的关系

当前现象：

- local 子问题拿到很多 `Metavante` 相关 chunk
- 但并没有稳定命中“第三方费用定义”那条核心 clause
- structural 子问题也没拿到足够强的 indemnification bridge

判断：

- **不是最终丢桥，而是 structural 子问题一开始就没把桥检回来**
- 同时 local 那一跳也没有精确落到 gold clause

结论：

- 属于 **structural 检索能力不足**
- 也伴随 **子问题 formulation 过宽**

### 3.4 `ragas-cuad-0042`

问题：

- gold 实际需要 Metavante 合同中的 `Section 8 + 10.3`

当前现象：

- 子问题级检索一开始拿到的是 TouchStar Reseller Agreement 的 `Section 8`
- 不是先检到正确合同，再在后面被挤掉

判断：

- 这里的根因是 **主文档/主合同从子问题检索开始就错了**

结论：

- 这不是 `chunk` 太大造成的污染
- 是 **检索引擎在该子问题上直接检错合同**

### 3.5 `ragas-cuad-0043`

问题：

- `Section 504`
- `Privacy Regulations`
- `Sensitive Customer Information`
- `Section 10.5.2`

当前现象：

- `sq_1` 能抓到 `Privacy Regulations`
- `sq_2` 只部分抓到 `Sensitive Customer Information`
- `sq_3 structural` 几乎没有桥接证据

判断：

- 拆题并非完全错误
- 但第三个 structural 子问题仍然偏抽象
- 即使 IRCoT follow-up，也没有把 bridge clause 真正检回来

结论：

- 这里是 **拆题与检索能力双重不足**
- 但主瓶颈仍然偏向 **structural 检索能力**

## 4. 对 doc-consistency bias 的重新判断

这次实验表明：

- `doc-consistency bias` 的目标没有问题
- 但之前的实现方式有问题

问题点在于：

- 之前通过 `sub_question_answers[*].support_chunk_ids / retrieved_chunk_ids_all`
  反推 `preferred_doc_ids`
- 如果子问题本身已经检错文档
- 后续 bias 会把这个错误进一步固化

因此当前应区分：

- **目标层面**
  - 压制错文档污染，这个方向是对的
- **实现层面**
  - 不能用已污染子问题结果去反推主文档

## 5. 根因排序

当前更合理的根因优先级如下：

1. **子问题级检索排序能力不足**
   - 特别是 `local`
   - 经常能拿到相关 chunk，但不是 gold clause

2. **部分多跳题主文档判断直接错误**
   - 典型如 `0042`
   - 从第一轮检索开始就检错合同

3. **structural 子问题 query formulation 不够可检索**
   - 典型如 `0041`、`0043`
   - 表达了“关系问题”，但没有具体到可稳定命中 bridge clause

4. **chunk 规模偏大**
   - 会放大前述错误
   - 但不是根因

## 6. 当前不建议的方向

### 6.1 不建议继续优先优化 final chunk selection

原因：

- 当前失败多数早于 final stage
- final stage 只能缓解，不能纠正前面没检到的问题

### 6.2 不建议继续强化 final-stage doc bias

原因：

- 若主文档判断已经错了
- final-stage bias 会把错误稳定化

## 7. 建议的后续优先级

### 7.1 对问题拆解做结构性重构

当前的“问题拆解”默认产物是自然语言子问题：

- `What is X?`
- `How does X relate to Y?`

这条路线的问题已经比较明确：

- 生成出来的子问题看起来合理
- 但它们仍然更像人类推理问题
- 不像检索器真正需要的中间表示

因此，后续不应只把当前 decomposition prompt 继续往 clause-oriented 上微调，
而应把“问题拆解”重构成**一条可独立切换的检索需求生成流程**。

换句话说：

- 旧流程：输出自然语言子问题
- 新流程：输出按 route 分组的检索需求（retrieval intent / anchors / keywords / endpoints / bridge relation）

这两条流程应该并存一段时间，并可在配置中切换，
方便做 A/B 验证，而不是一次性把旧流程直接删掉。

建议在 [`base_config.yaml`](/Users/sqy/Desktop/sensei/毕业论文/imp/enterprise-graphrag/youtu-graphrag/config/base_config.yaml) 中增加类似配置：

```yaml
retrieval:
  decomposition:
    mode: natural_language_subquestions
```

后续可切到：

```yaml
retrieval:
  decomposition:
    mode: retrieval_requirements
```

可选地，再允许：

```yaml
retrieval:
  decomposition:
    mode: disabled
```

用于直接比较：

- 不拆题
- 旧自然语言拆题
- 新检索需求拆题

### 7.2 旧流程：自然语言子问题拆解

这里的“改子问题生成模板”，不是把 LLM 拆题改成规则拆题。

当前系统仍然应该由 LLM 自行决定是否拆题、拆几题、走哪条 route。

但当前 decomposition prompt 更偏：

- route-aware
- reasoning-oriented

还不够：

- retrieval-oriented
- clause-oriented
- anchor-oriented

也就是说，当前 prompt 在教模型“如何按 local / structural / global 拆题”，
但没有充分教模型“如何把子问题改写成最容易命中合同条款的检索 query”。

下一步真正要改的是：

- decomposition prompt 中对子问题文本的约束
- 让模型生成更适合检索器的 query，而不是更像自然语言推理问题的 query

#### 7.2.1 当前 prompt 的不足

当前 decomposition prompt 主要要求：

- 生成 `1-3` 个 retrieval-oriented sub-questions
- 每个子问题带 `route_type`
- `local` 适合单事实
- `structural` 适合关系/多跳
- `global` 适合总结

这些要求足以让模型“会拆”。

但不足以让模型拆成：

- 条款导向的问题
- 带强检索锚点的问题
- 容易命中 gold clause 的问题

因此当前常见失败是：

- 子问题逻辑上没错
- 但表达得太宽、太抽象、太像人类理解问题
- 不像检索 query

#### 7.2.2 改写目标

改写后的子问题应满足：

1. **优先指向 clause / section / definition span**
   - 优先问“哪一条款说明了什么”
   - 而不是直接问开放式语义关系

2. **保留原问题中的强锚点**
   - section 编号
   - 合同实体名
   - 明确术语
   - 义务/触发/限制等关键词

3. **让每个子问题尽量对应一个可检索证据点**
   - local：一个定义、日期、职责、限制
   - structural：桥的两端 + 桥接关系

4. **对子问题做“答案表述风格 -> 条款表述风格”的改写**
   - 不直接问抽象答案
   - 改问证据在合同中如何写

#### 7.2.3 改写规则

##### A. local 子问题

当前容易生成：

- `What is Motorola's role in the promotion?`
- `What are the responsibilities of PageMaster during the promotion period?`

这类问题的风险是：

- 语义上正确
- 但检索时容易命中“相关角色”“相关职责”
- 不一定命中 gold clause

更好的写法应优先是：

- `Which clause states Motorola's role in providing devices or services in the promotion?`
- `Which clause states PageMaster Corporation's responsibilities during the promotion period?`
- `What exact clause defines the promotion period start and end dates?`
- `What clause defines the term 'Privacy Regulations'?`

也就是说：

- 少问 `what is X`
- 多问 `which clause states / defines X`

##### B. structural 子问题

当前容易生成：

- `How are A and B related?`
- `How does X relate to Y?`

这类写法对推理友好，但对检索不友好。

更好的写法应分成三类：

1. 先找桥的左端
   - `Which clause defines reimbursable third-party expenses?`
   - `Which clause defines Sensitive Customer Information?`

2. 再找桥的右端
   - `Which clause defines Neoforma's indemnification obligations for third-party claims?`
   - `Which clause defines Privacy Regulations under Section 504?`

3. 最后再问桥本身
   - `Is there any clause explicitly linking reimbursable third-party expenses to indemnification for third-party claims?`
   - `Which clause shows how Privacy Regulations govern or restrict Sensitive Customer Information?`
   - `Does the agreement explicitly connect Section 8 breach consequences to Section 10.3 indemnification procedures?`

关键点是：

- structural 不应直接问一个大而泛的关系问题
- 应优先问“定义端点 + 明确桥接 clause”

##### C. global 子问题

global 当前不是主要故障点，但也应避免写得太宽。

应优先写成：

- `What sections summarize the overall responsibilities regarding ... ?`
- `Across the agreement, which clauses describe the main risks or obligations related to ... ?`

而不是：

- `Summarize everything about ...`

#### 7.2.4 反例与正例

##### `ragas-cuad-0002`

反例：

- `What role does Motorola have in the PageMaster Corporation promotion agreement?`

问题：

- 太容易命中商标许可、第三方、indemnity 等相关条款

正例：

- `Which clause states Motorola's role in providing the pager offered in the promotion?`
- `Does the agreement state that Motorola provides the free pager, and where is that stated?`

##### `ragas-cuad-0003`

反例：

- `What are the responsibilities of PageMaster during the promotion period?`

问题：

- 会把“所有相关职责”都拉进来

正例：

- `Which clause states PageMaster Corporation's responsibilities during the promotion period?`
- `What exact clause states that PageMaster must establish and maintain the toll-free number?`
- `What clause defines the promotion period dates?`

##### `ragas-cuad-0041`

反例：

- `How does the agreement link third-party expenses to indemnification obligations?`

问题：

- 太像最终推理问题
- 检索时难以直接命中桥接 clause

正例：

- `Which clause defines reimbursable third-party expenses incurred by Metavante?`
- `Which clause defines Neoforma's indemnification obligations for third-party claims?`
- `Is there any clause explicitly connecting third-party expenses to third-party indemnification obligations?`

##### `ragas-cuad-0042`

反例：

- `What is the relationship between failure under Section 10.3 and breach of Section 8?`

问题：

- 只保留了 section 号，但没有说清“想找哪种法律后果/程序性连接”

正例：

- `What clause states the consequences or remedies for a breach of Section 8?`
- `What clause states the notice, defense, or procedure obligations under Section 10.3?`
- `Does the agreement explicitly connect Section 8 breach consequences with Section 10.3 indemnification procedures?`

##### `ragas-cuad-0043`

反例：

- `How does Privacy Regulations relate to Sensitive Customer Information?`

问题：

- 抽象
- 语义对，但检索弱

正例：

- `What clause defines Privacy Regulations under Section 504?`
- `What clause defines Sensitive Customer Information under Section 10.5.2?`
- `Which clause states how Privacy Regulations govern, restrict, or protect Sensitive Customer Information?`

#### 7.2.5 应加入 decomposition prompt 的约束

下一版 decomposition prompt 应明确告诉 LLM：

1. local 子问题优先写成：
   - `Which clause states ...`
   - `What clause defines ...`
   - `What exact section/date/limit/obligation states ...`

2. structural 子问题优先拆成：
   - 左端定义
   - 右端定义
   - 桥接 clause

3. 必须保留原问题中的强锚点：
   - section 编号
   - 合同实体名
   - 明确术语

4. 避免生成过宽的抽象问题：
   - `how are X and Y related`
   - `what is the significance of X`
   - `summarize the relationship`

5. 若问题是 factoid / clause lookup：
   - 优先只返回一个 clause-oriented local 子问题
   - 不要为了“看起来更完整”额外生成 structural 问题

#### 7.2.6 这一步的边界

仅仅把子问题从：

- `What is X?`
- `How does X relate to Y?`

改成：

- `Which clause states X?`
- `Which clause explicitly connects X and Y?`

并不天然保证检索变好。

这类改写的价值主要在于：

- 避免子问题过宽、过抽象
- 让 route 分配更稳定
- 让 local / structural 更像 clause lookup

但它不能单独解决更根本的问题：

**当前数据库不是 QA 对，而是从原始文档构建出来的图和 chunk。**

因此用户问题和文档证据之间天然存在表达鸿沟：

- 用户问的是“答案语义”
- 文档写的是“条款表达”
- 图和 chunk 存的是“文档中的局部表述”

所以，只改 decomposition prompt，最多只是把问题拆得更细、更稳，
还不足以把问题真正转换成适合检索器命中文档证据的形式。

#### 7.2.7 为什么这一步仍然值得做

因为当前大量失败样本不是：

- 完全没理解原问题

而是：

- 拆出来的子问题逻辑正确
- 但检索表达不够“像合同条款查询”

如果不先解决这层，后面的：

- rerank
- doc bias
- final selection

都只能在错误候选上做后处理。

但从当前实验现象看，真正更核心的一层不是旧式自然语言 decomposition rewrite，
而是新的 retrieval-oriented decomposition。

### 7.3 新流程：retrieval-oriented decomposition

这里建议的新方向，不是继续让模型输出“更好看的子问题”，
而是让模型**直接按检索模式输出子需求**。

也就是说，模型不再把核心产物定义为：

- `sub-question`

而是定义为：

- `route_type`
- `intent`
- `entities`
- `terms`
- `anchors`
- `query_keywords`
- `left_endpoint`
- `right_endpoint`
- `bridge_relation`
- `target_patterns`

这样做的核心原因是：

- 检索器真正需要的是检索信号
- 不是一段自然语言问句本身

#### 7.3.1 新流程的目标

让 LLM 负责：

- 理解原问题
- 识别 route
- 提取实体、术语、section anchors
- 识别 structural 问题的左右端点与桥接关系

让程序负责：

- 把这些结构化检索需求编译成多种 retrieval queries
- 再进行并行检索、融合与 rerank

这样就把“理解问题”和“生成检索 query”两件事拆开了。

#### 7.3.2 建议的新输出 schema

对于每个 route，建议输出的不是单一子问题，而是一条 `retrieval_requirement`：

```json
{
  "route_type": "local|structural|global",
  "intent": "definition_lookup|fact_lookup|bridge_lookup|summary_lookup",
  "route_reason": "...",
  "entities": [],
  "terms": [],
  "anchors": [],
  "query_keywords": [],
  "target_patterns": [],
  "left_endpoint": "",
  "right_endpoint": "",
  "bridge_relation": "",
  "scope": ""
}
```

其中：

- `entities`
  - 实体、party、产品、合同对象
- `terms`
  - 法律术语、定义术语、责任术语
- `anchors`
  - `Section 8`、`Section 10.3`、日期、标题等强锚点
- `query_keywords`
  - 供 sparse / BM25 / hybrid 检索直接使用
- `target_patterns`
  - 用于生成 clause-oriented query 模板
- `left_endpoint/right_endpoint/bridge_relation`
  - 仅对 structural 有意义

#### 7.3.3 三条 route 分别应该产出什么

##### A. local

local 不应该只是输出：

- `sub-question: What is X?`

而应该输出：

- `intent: definition_lookup / fact_lookup`
- `entities`
- `terms`
- `anchors`
- `query_keywords`
- `target_patterns`

例如：

```json
{
  "route_type": "local",
  "intent": "fact_lookup",
  "route_reason": "single clause factual lookup",
  "entities": ["Motorola", "PageMaster"],
  "terms": ["pager", "promotion"],
  "anchors": [],
  "query_keywords": ["Motorola", "PageMaster", "free pager", "promotion"],
  "target_patterns": ["which clause states", "provides", "promotion"]
}
```

##### B. structural

structural 不应该只输出：

- `How does X relate to Y?`

而应该输出：

- 左端
- 右端
- 桥接关系
- 强锚点
- 检索关键词

例如：

```json
{
  "route_type": "structural",
  "intent": "bridge_lookup",
  "route_reason": "requires explicit connection between two clauses",
  "left_endpoint": "Section 8 breach consequences",
  "right_endpoint": "Section 10.3 indemnification procedures",
  "bridge_relation": "explicit legal connection",
  "anchors": ["Section 8", "Section 10.3"],
  "query_keywords": ["breach", "consequences", "indemnification", "notice", "defense", "procedures"],
  "target_patterns": ["explicitly connect", "which clause states", "remedies", "procedures"]
}
```

##### C. global

global 应该输出：

- 汇总主题
- 范围约束
- 关键词簇

例如：

```json
{
  "route_type": "global",
  "intent": "summary_lookup",
  "route_reason": "requires agreement-wide aggregation",
  "themes": ["insurance", "food safety", "audit rights"],
  "scope": "agreement-wide",
  "query_keywords": ["insurance", "coverage", "food safety", "audit", "supplier obligations"],
  "target_patterns": ["which sections describe", "across the agreement", "main obligations"]
}
```

#### 7.3.4 retrieval rewrite 在新流程中的位置

在新流程里，retrieval rewrite 不再是“给自然语言子问题换个句式”，
而是把 `retrieval_requirement` 编译成多种 query。

也就是说，新流程更完整的形态应该是：

`original question -> retrieval-oriented decomposition -> query compilation -> retrieval -> rerank`

其中：

- decomposition 输出结构化检索需求
- query compilation 根据这些需求生成多种 query

#### 7.3.5 query compilation 应生成什么

对于每条 `retrieval_requirement`，程序应自动生成：

1. 原始语义版 query
2. 条款/定义版 query
3. 锚点强化版 query
4. structural 时额外生成桥接版 query

这样就不再依赖 LLM 一步写出“完美子问题”，
而是让 LLM 给出信号，程序负责系统性扩展成检索 query 集合。

#### 7.3.6 为什么要做成独立可切换流程

因为这已经不是对旧 decomposition prompt 的小修小补，
而是一次真正的问题拆解重构。

如果不把它做成独立流程，而是直接塞回旧 prompt：

- 难以做 A/B
- 难以判断收益来自哪里
- 难以回退

因此更合理的工程方式是：

- 保留旧自然语言拆题流程
- 新增 retrieval-oriented decomposition 流程
- 通过配置切换

推荐配置示意：

```yaml
retrieval:
  decomposition:
    mode: retrieval_requirements
```

并允许：

```yaml
retrieval:
  decomposition:
    enable_query_compilation: true
```

#### 7.3.7 推荐的实施顺序

1. 保留旧自然语言拆题逻辑不动
2. 新增 `retrieval_requirements` 输出 schema
3. 新增 query compilation 层
4. 在配置中切换新旧流程
5. 用 smoke test 比较：
   - `disabled`
   - `natural_language_subquestions`
   - `retrieval_requirements`

### 7.4 给 structural follow-up 做专门 rerank

当前 IRCoT 已经能生成 follow-up query，但还不够。

下一步应该：

- 对 structural 子问题的 follow-up retrieval
- 用：
  - 原始总问题
  - 子问题
  - follow-up query
  联合做更强 rerank

### 7.5 暂时回退 final-stage doc bias

保守策略：

- 如果保留 doc bias，只保留 chunk rerank 层的轻量 bonus
- 不要再让 final chunk selection 根据已污染子问题结果推主文档

## 8. 最终判断

一句话总结：

**当前不是“检到了，后面污染了”的问题。**

更接近的是：

**很多题从子问题检索开始就没检对，后面的 chunk 污染只是把这个失败放大。**

因此后续优化主线应当是：

1. 用可切换的 decomposition 流程做 A/B
2. 新 retrieval-oriented decomposition 负责输出检索需求
3. query compilation 负责生成真正的检索 query 集合
4. 子问题级 rerank
5. structural follow-up 检索增强

而不是继续把精力集中在最终 10 个 chunk 的后处理上。
