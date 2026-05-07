# 1. 目标

复现论文中的完整评估 pipeline：

1. 多方法统一对比
2. 双模式（Open / Reject）评估
3. Top-k Accuracy
4. Token/Time 统计

------

# 2. 实验环境规范

## 2.1 统一模型设置

必须保证所有方法使用：config.yaml或者相同配置

------

# 3. 数据准备

## 3.1 标准数据集

cuad

enron

------

## 3.2 统一数据格式

统一成 JSONL：

```
{
  "id": "...",
  "question": "...",
  "answer": "...",
  "context_docs": [...]
}
```

------

# 4. Baseline 复现规范

## 4.1 Baseline 列表

必须包含：

### 1️⃣ Vector RAG

- 向量检索
- 无图

### 2️⃣ GraphRAG

### 3️⃣ KG-RAG

------

## 4.2 统一推理流程

所有方法 inference 流程：

```
Query
  ↓
Retrieve top-k
  ↓
Generate answer
  ↓
Evaluate
```

禁止修改 prompt 结构。

------

# 5. 双模式评估协议

## 5.1 Reject Mode Prompt

```
If the extracted knowledge is not enough to answer, reject to answer.
```

若模型输出：

- “I don't know”
- “Insufficient evidence”
- “Cannot determine”

视为 REJECT。

------

## 5.2 Open Mode Prompt

```
If knowledge insufficient, answer based on your own knowledge.
```

------

# 6. 指标实现

# 6.1 Top-k Accuracy

定义：

若正确答案在 top-k 检索结果支持下生成正确 → 计为 1

实现：

```
def compute_accuracy(pred, gold):
    return semantic_match(pred, gold)
```

------

## 6.2 语义匹配（LLM Judge）

调用 qwen-flash进行判断：

Prompt：

```
Are the following two answers semantically equivalent?
Answer A:
Answer B:
Respond only Yes or No.
```

返回 Yes → 1
 否则 → 0

⚠ 注意：

- Judge 模型必须与生成模型分离
- 推荐使用独立模型避免 bias

------

## 6.3 Top-20 / Top-10

分别设置：

```
top_k = 20
top_k = 10
```

------

# 7. 构建成本统计

## 7.1 Token 消耗

记录：

- Extraction tokens
- Community detection tokens
- Generation tokens

实现：

```
total_tokens += response.usage.total_tokens
```

------

## 7.2 时间统计

```
start = time.time()
...
elapsed = time.time() - start
```

分别统计：

- Graph construction time
- Community detection time
- QA inference time

------

# 8. Pareto Frontier 分析

记录：

| 方法 | Token | Accuracy |

绘图：

```
plt.scatter(token_cost, accuracy)
```

目标：

展示性能 vs 成本 trade-off。

------

# 9. 输出报告格式

最终输出：

## 9.1 表格

```
Method | Dataset | Mode | Top20 | Top10 | Tokens | Time
```

------

## 9.2 可视化

- Token vs Accuracy
- Open vs Reject 对比图



------

# 10. 复现流程总结

完整流程：

```
1. 构建图
2. 统计构建token/time
3. 双模式推理
4. Top-k accuracy
5. Judge判定
6. 统计总token
```