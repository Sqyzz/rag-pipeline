# Youtu CUAD 分层图实施文档

## 1. 文档定位

本文档用于落地 `youtu-graphrag` 在 `CUAD` 场景下的图分层改造。

目标不是重写整套检索系统，而是在当前架构上做最低成本、可验证的结构治理，解决以下问题：

1. 多跳问题中 `structural` 路被噪声边污染
2. 最终证据选择难以稳定保留每一跳真正需要的 chunk
3. `global` 路在合同语义图上的收益偏弱，但辅助边仍会污染主检索图

本文档的核心结论是：

1. 当前问题主因在构图，不在评测脚本
2. 不建议继续靠检索白名单硬补
3. 第一阶段应采用“单图分层”而不是“完全双图双系统”

---

## 2. 当前问题确认

### 2.1 当前 `CUAD v3` 图的真实结构

文件：

- `youtu-graphrag/output/graphs/cuad_v3_new.json`

当前图并不是传统的聚合对象 `{nodes, edges, communities}`，而是边列表。实际统计结果：

1. 边数：`20690`
2. 唯一节点数：约 `10644`
3. 节点类型中高频类型：
   - `attribute`: `4062`
   - `entity`: `3194`
   - `event`: `639`
   - `clause`: `328`
   - `party`: `274`
4. 高频关系：
   - `has_attribute`: `7562`
   - `confidentiality_applies_to`: `2068`
   - `member_of`: `1211`
   - `requires_notice_before`: `502`
   - `terminates_on_event`: `459`
   - `governed_by`: `390`
   - `pays`: `321`
   - `payment_triggered_by`: `311`
   - `payment_due_on`: `300`
5. 同 chunk 边：`2200`
6. 跨 chunk 边：`8355`

判断：

1. 图具备多跳价值，因为跨 chunk 边很多
2. 图同时存在明显属性膨胀与结构噪声
3. `structural` 检索不是“没有图可走”，而是“图里可走的坏边太多”

### 2.2 当前 schema 不是唯一问题

文件：

- `youtu-graphrag/schemas/cuad_v3.json`

当前 schema 中真正定义的合同关系本身是合理的，例如：

1. `requires_notice_before`
2. `terminates_on_event`
3. `governed_by`
4. `payment_due_on`
5. `payment_triggered_by`
6. `limits_liability_of`
7. `confidentiality_applies_to`

因此当前问题不能简单归因为“schema 设计完全错误”。

### 2.3 构图阶段的真实问题

文件：

- `youtu-graphrag/models/constructor/kt_gen.py`
- `youtu-graphrag/utils/tree_comm.py`

当前噪声来源主要有三类：

1. `has_attribute`
   - 在 `_process_attributes()` 中被系统直接注入
   - 不属于合同问答主语义边
2. `represented_by`
3. `kw_filter_by`
   - 在 `tree_comm.py` 中由社区/关键词构建额外注入
   - 主要服务索引与摘要，不适合进入主推理路径

虽然 `base_config.yaml` 中：

1. `enforce_schema_relations: true`
2. `enforce_schema_entity_types: true`

但这些开关只约束 LLM 直接抽取出的实体/关系。

后处理与社区增强生成的辅助边仍会进入最终图，因此当前问题是：

**schema 只约束了抽取层，没有约束最终检索图的边层级。**

---

## 3. 目标方案

### 3.1 核心原则

第一阶段不做两套完全独立的 GraphRAG 系统，而做：

**单图分层**

即：

1. 保留一张图
2. 为边增加 `edge_layer`
3. 检索时按 route 读取不同 layer

推荐层级：

1. `semantic`
2. `auxiliary`

### 3.2 两层的职责

#### `semantic`

用于真正的问答推理、多跳路径搜索、最终证据选择。

应保留：

1. `party_to`
2. `named_as`
3. `effective_on`
4. `signed_on`
5. `has_initial_term`
6. `expires_on`
7. `renews_for`
8. `requires_notice_before`
9. `governed_by`
10. `dispute_resolved_in`
11. `notice_to`
12. `grants_right_to`
13. `grants_license_to`
14. `assignable_by`
15. `assignment_requires_consent_of`
16. `terminates_on_event`
17. `survives_for`
18. `is_null_and_void`
19. `termination_fee_for`
20. `pays`
21. `payment_amount`
22. `payment_due_on`
23. `payment_triggered_by`
24. `is_nonrefundable`
25. `audit_right_over`
26. `limits_liability_of`
27. `liability_excludes_cap_for`
28. `maintains_insurance`
29. `confidentiality_applies_to`
30. `exclusive_to`

#### `auxiliary`

用于召回增强、community/keyword 辅助，不直接作为多跳主路径。

应迁移到该层的边：

1. `has_attribute`
2. `represented_by`
3. `kw_filter_by`
4. `member_of`

说明：

1. `has_attribute` 对 fact recall 有帮助，但对 multi-hop 主推理噪声极高
2. `represented_by / kw_filter_by` 本质是社区辅助结构，不应进入主语义路径
3. `member_of` 默认也建议按辅助层处理，避免社区结构主导 structural 路

---

## 4. 对构图流程的改造

### 4.1 边层级标记

文件：

- `youtu-graphrag/models/constructor/kt_gen.py`
- `youtu-graphrag/utils/tree_comm.py`

新增规则：

1. 所有 schema 中的合同关系，默认 `edge_layer = "semantic"`
2. `_process_attributes()` 生成的 `has_attribute`，写为 `edge_layer = "auxiliary"`
3. `tree_comm.py` 中生成的 `represented_by / kw_filter_by / member_of`，写为 `edge_layer = "auxiliary"`

推荐数据结构：

```json
{
  "relation": "requires_notice_before",
  "edge_layer": "semantic"
}
```

```json
{
  "relation": "has_attribute",
  "edge_layer": "auxiliary"
}
```

### 4.2 属性节点保留策略

第一阶段不强制删除属性节点，但要改变其用途。

原则：

1. 属性节点仍可存在
2. 但 `has_attribute` 不进入 structural 主路径
3. local 路可以在候选召回阶段参考属性节点

这样可以在不推翻现有构图流程的前提下，先降低属性噪声。

### 4.3 社区构建策略

当前社区构建若直接基于全图，会把辅助边也纳入主题结构。

第一阶段建议：

1. community 继续保留
2. 但若重建 community，优先基于 `semantic` 子图

如果短期不重建 community，也至少保证：

1. `structural` 不走社区辅助边
2. `global` 即使使用 community，也应在最终落地到 `semantic` 对应 chunk

---

## 5. 对检索流程的改造

### 5.1 `local`

目标：

1. 保持高召回
2. 不牺牲明确事实问题表现

改法：

1. 候选召回可同时参考 `semantic + auxiliary`
2. 但最终 triple/chunk 排序优先 `semantic`
3. 属性边仅作召回提示，不直接提高路径权重

### 5.2 `structural`

目标：

1. 降低噪声路径
2. 提升多跳问题 `context_precision/context_recall`

改法：

1. 默认仅在 `semantic` 边上做主路径搜索
2. 辅助边只允许用于起点候选扩展，不参与主路径得分
3. triple rerank 时优先保留：
   - 跨 clause
   - 跨 entity
   - 跨 chunk 的语义边

不建议：

1. 继续在 `structural` 路里让 `has_attribute` 成为常规可走边
2. 继续让 `represented_by / kw_filter_by` 进入主路径

### 5.3 `global`

目标：

1. 保留摘要能力
2. 避免社区结构直接污染最终证据

改法：

1. `global` 仍可用 `community` 和 keyword 辅助结构
2. 但最终输出给回答层和 `ragas` 的 chunk，必须回落到 `semantic` 子图对应证据

---

## 6. 最低成本落地顺序

### 阶段 1：单图分层，不改外部接口

目标：

1. 不改 compare / ragas 输出协议
2. 只改图内部边层级与 route 读取策略

实施项：

1. 为边增加 `edge_layer`
2. 在 `kt_gen.py` 中把 `has_attribute` 标为 `auxiliary`
3. 在 `tree_comm.py` 中把 `represented_by / kw_filter_by / member_of` 标为 `auxiliary`
4. 在 retriever 中新增按 layer 过滤能力
5. `structural` 默认只读 `semantic`

收益：

1. 工程风险最小
2. 不影响现有接口和评测管线
3. 最可能直接改善多跳题表现

### 阶段 2：community 语义化

目标：

1. 提升 `global` 路稳定性
2. 降低社区结构对合同语义的偏移

实施项：

1. 基于 `semantic` 子图重建社区
2. 将 community 输出与 `semantic` chunk 对齐

### 阶段 3：属性收缩

目标：

1. 进一步压缩图噪声
2. 减少属性型 hub 节点

实施项：

1. 将低价值属性改为节点属性而非图节点
2. 只保留对检索有直接价值的少数属性边

---

## 7. 成本评估

### 7.1 构图成本

会增加，但可控。

增加项：

1. 边层级判断与写入
2. 可能的 `semantic` 子图 community 重建

判断：

1. 属于离线成本
2. 不会接近“双图双系统”的复杂度

### 7.2 检索成本

预计持平或下降。

原因：

1. structural 路候选边更少
2. 无效路径更少
3. rerank 噪声下降
4. 最终 10 个 chunk 的浪费减少

### 7.3 工程复杂度

主要新增成本在认知与调试复杂度，而不是纯算力成本。

需要维护的新概念：

1. `edge_layer`
2. route 与 layer 的映射
3. community 是否基于 `semantic` 子图

---

## 8. 验证标准

改造后重点观察 `youtu` 在 `ragas` 上的变化：

1. `single_hop_specific` 不应明显退化
2. `multi_hop_specific.context_precision` 应提升
3. `multi_hop_specific.context_recall` 应提升
4. `multi_hop_abstract.answer_correctness` 应提升
5. 低检索分 + 低正确率样本数应下降

建议重点抽查：

1. `retrieval_trace.sub_questions[*].retrieved_chunk_ids`
2. `retrieval_trace.sub_questions[*].final_selected_chunk_ids`
3. 多跳样本是否每一跳都有 chunk 保留进最终上下文

---

## 9. 当前建议

当前最值得实施的不是：

1. 继续扩检索白名单
2. 继续只调最终回答 prompt

而是：

1. 先做单图分层
2. 先让 `structural` 从辅助边中解耦
3. 再重跑 `CUAD + ragas`

这是当前 `youtu` 在 `full` 结果上继续提升的最高优先级路径。
