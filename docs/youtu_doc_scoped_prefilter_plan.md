# Youtu Doc-Scoped Prefilter Implementation Plan

本文档用于记录 `youtu-graphrag` 检索链路中，从 `late filtering` 升级到 `scope-aware prefilter retrieval` 的具体实施方案。

适用范围：

- `youtu-graphrag/backend.py`
- `youtu-graphrag/models/retriever/enhanced_kt_retriever.py`
- `youtu-graphrag/config/base_config.yaml`
- 相关 retrieval regression tests

相关背景文档：

- [docs/youtu_retrieval_requirements_smoke_analysis_and_optimization.md](./youtu_retrieval_requirements_smoke_analysis_and_optimization.md)
- [docs/youtu_cuad_doc_consistency_bias_plan.md](./youtu_cuad_doc_consistency_bias_plan.md)

## 1. 问题定义

当前后端虽然已经把 `target_doc_id`、`scope_plan`、`strict_target_doc_mode` 透传到检索器，但 dense chunk retrieval 仍然是：

1. 先对全量 chunk FAISS 索引做检索
2. 再按 `target_doc_id` / `scope_plan` 做候选优先级收束
3. 最后在 rerank 与 final selection 阶段再加 doc bias

这意味着当前系统仍属于：

- global search
- late filtering
- post-hoc doc bias

而不是：

- scope-aware candidate generation
- metadata-aware prefilter
- verified fallback retrieval

在多合同、多附件、多协议场景下，这会导致：

- 目标文档 chunk 可能在 first-stage 就被挤出候选池
- reranker 无法“复活”被过滤掉的真实证据
- 系统给出逻辑完整但事实缺失的 `NOT_FOUND`

## 2. 目标与非目标

### 2.1 目标

本阶段目标不是简单实现“硬单文档检索”，而是实现：

1. `scope-aware retrieval`
   - 根据 `scope_type` 决定检索范围和约束强度
2. `verified prefilter`
   - 只有在 scope 经过验证后，才启用强 doc-scoped search
3. `graceful fallback`
   - strict lane 失败时，不直接返回 `NOT_FOUND`
4. `entity rescue`
   - 保留“目标合同外的相关实体信息”作为辅助视角
5. `index safety`
   - 防止映射错位导致子集检索返回无关 chunk

### 2.2 非目标

本阶段不做：

- 全量向量库替换
- 多季度规模的索引系统重构
- 彻底移除全局索引

## 3. 设计原则

### 3.1 把 scope 当作 prior，不是 axiom

`target_doc_id` 不应成为一次性“生死判决”。更合理的做法是：

- 先估计文档范围
- 再验证该范围是否可靠
- 只有可靠时才启用硬约束

### 3.2 把答案归属和实体探索拆开

对于法律或企业文档问答：

- `strict evidence`
  - 决定“目标合同里是否明确写了”
- `related corpus evidence`
  - 决定“语料库中还有哪些相关信息”

这两者不能混成一个证据池。

### 3.3 sparse 检索保留全局统计，局部应用 mask

子集检索不能改写 BM25 类特征的统计基数。正确做法是：

- 全局预计算 `idf`
- 查询时只对 `allowed_doc_ids` 内 chunk 累分分数

### 3.4 子集检索必须基于稳定向量 ID

不能依赖“当前 row index 恰好与缓存映射对齐”这种隐式约定。任何 doc-scoped retrieval 都必须：

- 绑定稳定 vector id
- 有 manifest 与 consistency check
- 支持 cache mismatch 时的显式回退

## 4. 需要规避的四类风险

### 4.1 守门人失效风险（Decomposer Error）

问题：

- 如果 decomposer 或 scope planner 把本应跨文档的问题误判为 `single_doc`
- 再直接套用 hard filter
- 外部证据会被物理性切断

规避策略：

1. 增加 `probe stage`
   - 先做小规模全局探测
   - 不直接进入 strict lane
2. strict lane 启用条件必须同时满足：
   - 显式 doc anchor 存在
   - scope 置信度高
   - probe 中目标文档候选占优
3. strict lane 失败后进入 widened lane
   - `single_doc -> primary + secondary -> global/open`
4. `NOT_FOUND` 只能在所有 lane 都失败后给出

### 4.2 实体-文档解耦风险（Entity-Document Mismatch）

问题：

- 用户问的是“某实体在目标合同中的角色”
- 但该实体定义或角色信息只在其他合同中存在

如果直接 hard filter：

- 系统会返回“目标合同未提及”
- 虽然法理上可能正确
- 但业务辅助价值可能不足

规避策略：

1. strict lane 只回答：
   - 目标合同里是否有直接证据
2. 增加 `entity rescue lane`
   - 全局检索实体定义、角色、别名、相关合同
3. 最终答案分栏输出：
   - `In target contract`
   - `Related corpus context`
4. 禁止 external evidence 与 strict evidence 混用为同一 claim 的直接支撑

### 4.3 稀疏检索统计偏置（Sparse Statistics Distortion）

问题：

- 如果对过滤后的 doc 子集重新计算 `df/idf`
- 小样本合同会放大常见词
- 词面通道容易产生主题漂移

规避策略：

1. `idf` 固定使用全局语料统计
2. prefilter 只通过 `doc mask` 控制可累分 chunk
3. 当 `allowed_doc_ids` 非常小：
   - 降低 sparse lane 在 RRF 中的权重
   - 或要求 lexical lane 只保留 hard anchor 词
4. 对小合同场景引入最小词面阈值与 stopword 扩展

### 4.4 索引映射一致性风险（Index Corruption）

问题：

- 手工维护 `doc_id -> chunk_indices`
- 一旦数据更新导致 row id 漂移
- 子集检索可能命中完全错误的 chunk

规避策略：

1. 使用稳定 vector id，而不是依赖 row index
2. 构建索引时写 manifest：
   - dataset fingerprint
   - chunk count
   - vector_id -> chunk_id hash
   - doc histogram
3. 服务启动时做 consistency check
4. mismatch 时自动退回全局检索，不允许静默继续
5. 使用原子切换，不原地热改索引与映射

## 5. 目标架构

建议把检索链路拆成四条 lane：

1. `probe lane`
   - 小规模全局探测
   - 用于验证 scope 是否可信
2. `strict lane`
   - 对 `primary_doc_id` 做真正 prefilter retrieval
3. `assist lane`
   - 对 `primary_doc_id + secondary_doc_ids` 做受控检索
   - 用于 bridge / multi-doc 支撑
4. `entity rescue lane`
   - 对全局实体相关证据做探索式补充

最终回答策略：

- `strict evidence` 优先
- `assist evidence` 作为桥接补充
- `entity rescue evidence` 单独标注为外部上下文
- 所有 lane 失败后才返回 `NOT_FOUND`

## 6. 分阶段实施清单

### 6.1 Phase 0：索引元数据与安全底座

目标：

- 先把索引状态管理做安全

改动：

1. 在 `KTRetriever` 中新增：
   - `chunk_embeddings_matrix`
   - `vector_id_to_chunk_id`
   - `chunk_id_to_vector_id`
   - `doc_id_to_vector_ids`
   - `index_manifest`
2. chunk embedding cache 落盘时同步写 manifest
3. 启动加载时新增一致性校验
4. manifest 不一致时：
   - 日志告警
   - 禁用 scoped prefilter
   - 回退全局检索

验收：

- manifest mismatch 不会返回错误 chunk
- scoped mode 能被显式降级

### 6.2 Phase 1：scope 验证与 strict gate

目标：

- 防止误判 scope 直接触发 hard filter

改动：

1. 新增 `probe retrieval`
   - dense global top-20~40
   - sparse global top-50~100
2. 计算 probe 指标：
   - target doc hit ratio
   - target doc candidate count
   - anchor coverage
   - secondary doc evidence ratio
3. 只有满足 gate 条件时，才进入 strict lane
4. 引入新的运行态决策：
   - `scope_decision = strict | prefer | open`

验收：

- single-doc 误判不会直接把题打成 `NOT_FOUND`

### 6.3 Phase 2：dense chunk retrieval 真正 prefilter

目标：

- 把 `target_doc_id` 下沉到 candidate generation，而不是只下沉到 post selection

改动：

1. 新增 `_resolve_allowed_doc_ids(...)`
   - 根据 `scope_plan` 计算可搜索 doc 集
2. 改造 `_chunk_embedding_retrieval()`
   - `global_open`：保留全局 FAISS
   - `single_doc`：只在 `primary_doc_id` 子集上搜索
   - `cross_doc_bridge`：按 `primary` / `secondary` 分开搜索后再按 slot 合并
3. 第一版采用：
   - 子集精确向量搜索
   - 不先上 per-doc FAISS
4. 继续保留全局 fallback

验收：

- 目标文档 chunk 不再因为全局噪声被 first-stage 挤出

### 6.4 Phase 3：sparse retrieval 改为 global-idf + doc-mask

目标：

- 避免子集重算词频导致 lexical 偏置

改动：

1. 保留现有全局 `idf`
2. `_sparse_chunk_retrieval()` 中只对 `allowed_doc_ids` 内 chunk 累分
3. 小子集时降低 lexical lane 权重
4. 在 RRF trace 中落盘：
   - sparse active doc count
   - sparse filtered candidate count

验收：

- 小合同不会因为常见词而被 lexical lane 带偏

### 6.5 Phase 4：entity rescue lane

目标：

- 防止“极度冷漠的准确”

改动：

1. 基于 `_normalized_entity_doc_ids` 扩展实体救援查询
2. rescue lane 只返回：
   - entity definitions
   - aliases
   - role mentions
   - source doc ids
3. 输出格式显式区分：
   - strict support
   - external related support

验收：

- 目标合同未出现实体时，系统仍能给出可解释的外部补充上下文

### 6.6 Phase 5：配置与 rollout

目标：

- 让新能力能灰度验证，而不是一次性替换

建议新增配置：

```yaml
retrieval:
  target_doc:
    strict_mode: auto
    prefilter_enabled: false
    probe_enabled: true
    probe_dense_top_k: 30
    probe_sparse_top_k: 80
    min_target_doc_hits: 2
    min_target_doc_ratio: 0.25
    allow_entity_rescue: true
    prefilter_fallback_mode: prefer
```

说明：

- `strict_mode`
  - 控制强度
- `prefilter_enabled`
  - 控制是否启用真 prefilter
- `probe_enabled`
  - 控制 strict gate 是否先做验证
- `prefilter_fallback_mode`
  - `strict | prefer | open`

## 7. 代码改动清单

### 7.1 retriever

主要文件：

- `youtu-graphrag/models/retriever/enhanced_kt_retriever.py`

重点函数：

- `_precompute_chunk_embeddings()`
- `_load_chunk_embedding_cache()`
- `_chunk_embedding_retrieval()`
- `_sparse_chunk_retrieval()`
- `_hybrid_chunk_retrieval()`
- `_infer_scope_plan()`
- `_should_use_strict_target_doc_mode()`

新增 helper：

- `_resolve_allowed_doc_ids()`
- `_run_probe_retrieval()`
- `_decide_scope_retrieval_mode()`
- `_dense_subset_search()`
- `_merge_doc_scoped_results()`
- `_run_entity_rescue_retrieval()`
- `_load_or_build_index_manifest()`
- `_check_index_manifest_consistency()`

### 7.2 backend

主要文件：

- `youtu-graphrag/backend.py`

改动点：

- 子问题检索前接入 scope probe 结果
- 在 retrieval trace 中区分：
  - `scope_plan`
  - `scope_decision`
  - `probe_trace`
  - `entity_rescue_trace`
- 最终回答 prompt 中显式区分：
  - 目标合同内证据
  - 相关语料外部证据

### 7.3 config

主要文件：

- `youtu-graphrag/config/base_config.yaml`
- `youtu-graphrag/config/config_loader.py`

改动点：

- 扩展 `retrieval.target_doc` 配置项
- 做 schema 校验

## 8. 测试清单

### 8.1 检索正确性

- `single_doc`：
  - 目标文档存在证据时，必须稳定召回目标 chunk
- `cross_doc_bridge`：
  - primary / secondary docs 都能进入受控配额
- `global_open`：
  - 行为保持与当前全局检索兼容

### 8.2 风险回归

- decomposer 错判 `single_doc`
  - strict lane 失败后能够 widened fallback
- 实体仅存在于外部合同
  - strict lane 返回未提及
  - rescue lane 返回相关实体上下文
- 小子集 sparse retrieval
  - 不因为子集重算统计而放大常见词
- manifest mismatch
  - 自动禁用 scoped prefilter

### 8.3 性能

- `global_open`：
  - 延迟不明显退化
- `single_doc`：
  - 子集搜索通常快于全局搜索
- `cross_doc_bridge`：
  - 延迟增加可控

## 9. 发布顺序

推荐顺序：

1. `Phase 0`
2. `Phase 1`
3. `Phase 2`
4. `Phase 3`
5. `Phase 4`
6. `Phase 5`

原因：

- 先修安全与 gate
- 再修 dense prefilter
- 再修 sparse 偏置
- 最后补业务体验层的 entity rescue

## 10. 本阶段的工程结论

这一步不应被定义成：

- “把 target_doc_id 下沉到检索器”

因为这件事参数层已经做过了。

真正要做的是：

- 把 `target_doc_id / scope_plan` 从 post-hoc bias 升级成 verified candidate generation constraint
- 同时保留可解释的 fallback 与 entity rescue

也就是说，本阶段不是：

- naive hard filter

而是：

- scope-aware, probe-verified, fallback-safe prefilter retrieval

## 11. 外部资料

以下官方资料支持本方案中的关键设计点：

- Faiss `SearchParameters` / `IDSelector`
  - https://faiss.ai/cpp_api/struct/structfaiss_1_1SearchParameters.html
- Faiss `IndexIDMap / IndexIDMap2`
  - https://faiss.ai/cpp_api/file/IndexIDMap_8h.html
- Milvus filtered search
  - https://milvus.io/docs/filtered-search.md
- Qdrant payload / filtering
  - https://qdrant.tech/documentation/concepts/payload/
  - https://qdrant.tech/documentation/concepts/filtering/
- Elasticsearch `dfs_query_then_fetch`
  - https://artifacts.elastic.co/javadoc/org/elasticsearch/elasticsearch/8.18.4/org.elasticsearch.server/org/elasticsearch/action/search/SearchType.html
