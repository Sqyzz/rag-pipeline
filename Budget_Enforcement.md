- # Budget Enforcement 实施设计文档

  ------

  # 一、设计目标

  实现一个：

  - 统一
  - 可配置
  - 方法感知（method-aware）
  - 不破坏语义
  - 不粗暴截断

  的预算控制系统。

  核心思想：

  > 预算不只是 token，而是“推理资源总量”。

  ------

  # 二、预算类型（必须分层）

  预算分为三类：

  ------

  ## 1️⃣ 通用预算（所有方法共享）

  控制 query-time LLM 消耗：

  ```
  budget:
    evidence_token_limit: 2500
    max_completion_tokens: 800
    max_llm_calls: 2
    max_total_tokens: 3800
  ```

  ------

  ## 2️⃣ 方法粒度预算（method-aware）

  控制不同方法的结构规模：

  ```
  budget:
    vector:
      max_chunks: 8
  
    kg:
      max_hops: 2
      max_nodes: 30
  
    graph:
      max_communities: 1
      summary_level: 2
      map_keypoints_limit: 5
  ```

  ------

  ## 3️⃣ Indexing 预算（可选）

  如果你要控制 indexing-time：

  ```
  budget:
    indexing:
      triple_max_tokens: 20000
      community_summary_max_tokens: 40000
  ```

  ------

  # 三、完整 budget.yaml 示例

  ```
  regime: budget_matched
  
  budget:
    evidence_token_limit: 2500
    max_completion_tokens: 800
    max_llm_calls: 2
    max_total_tokens: 3800
  
    vector:
      max_chunks: 8
  
    kg:
      max_hops: 2
      max_nodes: 30
  
    graph:
      max_communities: 1
      summary_level: 2
      map_keypoints_limit: 5
  ```

  ------

  # 四、BudgetManager 设计升级

  ------

  ## 1️⃣ 初始化时读取 YAML

  ```
  import yaml
  
  with open("config_budget.yaml") as f:
      cfg = yaml.safe_load(f)
  
  budget_cfg = cfg["budget"]
  ```

  ------

  ## 2️⃣ BudgetManager 结构

  ```
  class BudgetManager:
      def __init__(self, tokenizer, cfg):
          self.tokenizer = tokenizer
          self.evidence_limit = cfg["evidence_token_limit"]
          self.max_completion = cfg["max_completion_tokens"]
          self.max_calls = cfg["max_llm_calls"]
          self.max_total = cfg["max_total_tokens"]
  
          self.calls_used = 0
          self.total_tokens_used = 0
  
      def count(self, text):
          return len(self.tokenizer.encode(text))
  
      def can_add(self, current, new):
          return self.count(current + new) <= self.evidence_limit
  
      def register_call(self):
          if self.calls_used >= self.max_calls:
              raise RuntimeError("LLM call budget exceeded")
          self.calls_used += 1
  
      def register_tokens(self, prompt_tokens, completion_tokens):
          self.total_tokens_used += prompt_tokens + completion_tokens
          if self.total_tokens_used > self.max_total:
              raise RuntimeError("Total token budget exceeded")
  ```

  ------

  # 五、VectorRAG 更新

  ------

  ## 使用 YAML 中的 max_chunks

  ```
  max_chunks = cfg["budget"]["vector"]["max_chunks"]
  retrieved = retrieve(query)[:max_chunks]
  ```

  再做 dynamic packing。

  ------

  # 六、KG-RAG 更新

  ------

  ## 从 YAML 读取参数

  ```
  kg_cfg = cfg["budget"]["kg"]
  MAX_HOPS = kg_cfg["max_hops"]
  MAX_NODES = kg_cfg["max_nodes"]
  ```

  Traversal 逻辑中加入：

  ```
  if hop > MAX_HOPS:
      break
  
  if len(nodes_collected) > MAX_NODES:
      break
  ```

  ------

  # 七、GraphRAG 更新（关键）

  ------

  ## 读取 YAML

  ```
  graph_cfg = cfg["budget"]["graph"]
  MAX_COMMUNITIES = graph_cfg["max_communities"]
  SUMMARY_LEVEL = graph_cfg["summary_level"]
  MAP_KEYPOINT_LIMIT = graph_cfg["map_keypoints_limit"]
  ```

  ------

  ## 控制社区数量

  ```
  communities = rank_communities(query)[:MAX_COMMUNITIES]
  ```

  ------

  ## 控制 summary 层级

  ```
  summary = community.summary[level=SUMMARY_LEVEL]
  ```

  ------

  ## Map Prompt 模板动态生成

  ```
  prompt = f"""
  Extract at most {MAP_KEYPOINT_LIMIT} key bullet points relevant to the query.
  Keep each bullet under 20 words.
  Do not provide full explanation.
  
  Community Summary:
  {summary}
  """
  ```

  ------

  ## Reduce 阶段

  ```
  budget_manager.register_call()
  
  final_answer = llm.generate(
      prompt=reduce_prompt,
      max_tokens=budget_manager.max_completion
  )
  ```

  ------

  # 八、Tokenizer 统一实现（更新后版本）

  在 `src/utils/tokenizer.py`

  ```
  import tiktoken
  from transformers import AutoTokenizer
  
  
  class TokenizerProvider:
      def __init__(self, backend, model_name):
          self.backend = backend
          self.model_name = model_name
  
          if backend == "api":
              self.tokenizer = tiktoken.encoding_for_model(model_name)
  
          elif backend == "local":
              try:
                  self.tokenizer = AutoTokenizer.from_pretrained(model_name)
              except:
                  self.tokenizer = tiktoken.get_encoding("cl100k_base")
  
      def encode(self, text):
          return self.tokenizer.encode(text)
  
      def count(self, text):
          return len(self.encode(text))
  ```

  ------

  # 九、日志记录（必须升级）

  每次 LLM 调用后：

  ```
  usage = response.usage
  
  budget_manager.register_tokens(
      usage.prompt_tokens,
      usage.completion_tokens
  )
  ```

  输出：

  ```
  {
    "method": "graph_rag",
    "regime": "budget_matched",
    "llm_calls": 2,
    "prompt_tokens": 2400,
    "completion_tokens": 700
  }
  ```