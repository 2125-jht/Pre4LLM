# TAAC2025 Baseline 方法全流程详解

> 本文面向推荐领域初学者，从数据输入到最后输出，逐层拆解 baseline 的完整 pipeline。

---

## 一、整体架构概览

这个 baseline 是一个**序列推荐模型**，核心思想是：

> 把用户的**历史行为序列**（看过哪些商品）输入 Transformer，学习用户的兴趣表示，然后用这个表示去**检索相似的商品**。

整体流程分为三大阶段：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  阶段1: 数据准备                                                            │
│  原始数据(parquet) → 用户行为序列 → 特征填充 → 训练样本(正/负样本对)        │
├─────────────────────────────────────────────────────────────────────────────┤
│  阶段2: 模型训练                                                            │
│  序列 + 特征 → Embedding → Transformer Encoder → 用户兴趣向量               │
│  → 与正负样本做内积 → BCE Loss → 优化                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  阶段3: 推理输出                                                            │
│  测试用户序列 → 用户兴趣向量(query)                                         │
│  + 候选库商品向量(candidate) → ANN近似检索 → Top-K 推荐结果                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、阶段1：数据输入与处理（dataset.py）

### 2.1 原始数据结构

数据存放在数据目录下，包含：

| 文件/目录 | 内容 |
|-----------|------|
| `seq/` | 用户行为序列（parquet格式），每个用户有一个 `item_id` 列表 + `action_type`（行为类型）+ `timestamp` |
| `item_feat/` | 商品特征（parquet格式），如类目、标签等 sparse 特征 |
| `user_feat/` | 用户特征（parquet格式），如 demographics 等 |
| `mm_emb/` | 多模态特征（图片/文本的embedding），如 `81`~`86` |
| `indexer.pkl` | ID映射字典，将原始 creative_id/user_id 映射为连续的 re-id |

### 2.2 序列数据加载（`load_seq_as_list`）

```python
# 从 parquet 读取用户序列
full_events, user_indices = load_seq_as_list(data_dir / "seq")
```

- `full_events`: 所有用户行为展平后的大数组，每个元素是 `(item_id, action_type, timestamp)`
- `user_indices`: `{user_id: (start_idx, length)}`，记录每个用户在 `full_events` 中的起止位置

**为什么要这样存？** 节省内存，避免每个用户存一个 Python list。

### 2.3 特征加载

```python
self.item_feat_dict = load_feat_dict_from_parquet_folder(item_feat_src, "item_id", item_sparse_ids)
self.user_feat_dict = load_feat_dict_from_parquet_folder(user_feat_src, "user_id", user_sparse_ids + user_array_ids)
self.mm_emb_dict    = load_mm_emb_v2(data_dir, mm_emb_ids)  # 多模态embedding
```

特征分为几类：

| 特征类型 | 说明 | 例子 |
|----------|------|------|
| `sparse` | 离散单值特征，做 Embedding lookup | `100`(类目), `103`(性别) |
| `array` | 离散多值特征，先 Embedding 再求和 | `106`(兴趣标签列表) |
| `continual` | 连续数值特征，直接拼接入向量 | （本赛题暂无） |
| `emb` | 预训练多模态向量，过线性层降维 | `81`(32维), `82`(1024维) |

### 2.4 训练样本构造（`__getitem__`）

这是最关键的部分。对**每个用户**，代码做以下事情：

#### Step 1: 加载用户完整序列

```python
user_sequence = self.new_load_user_data(uid)
# 返回: [(uid, item_id, user_feat, item_feat, action_type, timestamp), ...]
```

#### Step 2: 构建交错序列

把用户特征和商品特征交错排列：

```
序列: [user_feat, item_1_feat, item_2_feat, item_3_feat, ..., user_feat]
类型: [   2    ,      1      ,      1      ,      1      , ...,    2    ]
```

- `token_type=1` 表示 item
- `token_type=2` 表示 user

**为什么要把 user_feat 插入序列？** 让 Transformer 能同时看到用户画像和商品历史。

#### Step 3: 生成训练样本（Next Item Prediction）

从左到右遍历序列，每个位置预测**下一个 item**：

```
序列: [u, i1, i2, i3, u, i4, i5]
      ↓  ↓  ↓  ↓  ↓  ↓
正样本: [i1, i2, i3, u, i4, i5]   (下一个真实访问的item)
负样本: [?,  ?,  ?,  ?,  ?,  ?]   (随机采样用户没见过的item)
```

注意：
- 只有 `next_token_type == 1`（下一个是 item）的位置才计算 loss
- 负样本通过 `_random_neq` 随机采样，确保不在用户历史里

#### Step 4: Padding 填充

所有序列统一到 `maxlen+1`（默认101）长度，**左边补0**：

```python
seq = np.zeros([self.maxlen + 1], dtype=np.int32)   # 序列ID
pos = np.zeros([self.maxlen + 1], dtype=np.int32)   # 正样本ID
neg = np.zeros([maxlen + 1], dtype=np.int32)        # 负样本ID
token_type = np.zeros([maxlen + 1], dtype=np.int32) # 当前类型(1=item, 2=user)
next_token_type = ...  # 下一个token类型
seq_feat = ...         # 每个位置的特征字典
```

#### Step 5: 缺失特征填充（`fill_missing_feat`）

如果某个商品/用户缺少某些特征，用默认值填充：
- sparse 特征默认 `0`
- array 特征默认 `[0]`
- emb 特征默认 `np.zeros(dim)`，但如果多模态表里有就用真实的

### 2.5 测试集构造（`MyTestDataset.__getitem__`）

与训练集的区别：
- **没有 pos/neg**，只需要用户历史序列
- **保留原始 user_id**（字符串如 `user_xxxxxx`），方便输出结果对照
- **处理冷启动**：测试集里没见过的新 item，特征 value 可能是字符串，转为 `0`

---

## 三、阶段2：模型结构（model.py）

模型叫 `BaselineModel`，核心是 **Transformer + 特征融合**。

### 3.1 Embedding 层

#### 基础 Embedding Table

```python
self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)  # 商品ID → 向量
self.user_emb = nn.Embedding(user_num + 1, hidden_units, padding_idx=0)  # 用户ID → 向量
self.pos_emb  = nn.Embedding(2 * maxlen + 1, hidden_units, padding_idx=0) # 位置编码
```

#### 稀疏特征 Embedding

```python
self.sparse_emb[k] = nn.Embedding(feat_num + 1, hidden_units, padding_idx=0)
```

每个 sparse/array 特征都有独立的 Embedding Table。

#### 多模态特征变换

```python
self.emb_transform[k] = nn.Linear(emb_dim, hidden_units)
```

把预训练的 32/1024/3584 维多模态向量，映射到 `hidden_units`（默认32）维。

### 3.2 特征融合（`feat2emb`）

这是模型最核心的操作，把各种特征拼成统一的向量：

```
输入: seq_id + feature_dict_list + token_type_mask

Step 1: 基础ID Embedding
  - 如果 include_user=True:
      user_mask处 → user_emb, item_mask处 → item_emb
  - 否则只取 item_emb

Step 2: 加特征Embedding
  item_sparse特征  → sparse_emb[k](value)           → 拼到item侧
  item_array特征   → sparse_emb[k](value).sum(2)    → 拼到item侧（多值求和）
  user_sparse特征  → sparse_emb[k](value)           → 拼到user侧
  user_array特征   → sparse_emb[k](value).sum(2)    → 拼到user侧
  item_emb(多模态) → emb_transform[k](value)        → 拼到item侧

Step 3: DNN融合
  item侧: concat所有 → Linear + ReLU → hidden_units维
  user侧: concat所有 → Linear + ReLU → hidden_units维

Step 4: 相加
  seqs_emb = item_emb + user_emb
```

### 3.3 Transformer 编码器（`log2feats`）

```python
def log2feats(self, log_seqs, mask, seq_feature):
    # 1. 特征 → Embedding
    seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True)
    
    # 2. 缩放 + 位置编码
    seqs *= sqrt(hidden_units)
    seqs += pos_emb(position)
    seqs = dropout(seqs)
    
    # 3. 构造因果掩码（Causal Mask）
    # 下三角矩阵 & padding mask
    attention_mask = tril_mask & pad_mask
    
    # 4. N层 Transformer Block
    for each block:
        if norm_first:
            x = LayerNorm(seqs)
            mha = MultiHeadAttention(x, x, x, mask)
            seqs = seqs + mha  # 残差
            seqs = seqs + FFN(LayerNorm(seqs))
        else:
            mha = MultiHeadAttention(seqs, seqs, seqs, mask)
            seqs = LayerNorm(seqs + mha)
            seqs = LayerNorm(seqs + FFN(seqs))
    
    return LayerNorm(seqs)  # [batch, maxlen, hidden_units]
```

**注意掩码设计**：
- `tril_mask`: 因果掩码，只看当前位置及之前（不能偷看未来）
- `pad_mask`: padding位置为0，不参与attention

### 3.4 训练前向传播（`forward`）

```python
def forward(self, user_item, pos_seqs, neg_seqs, mask, next_mask, seq_feature, pos_feature, neg_feature):
    # 1. 用户序列 → Transformer → 兴趣向量
    log_feats = self.log2feats(user_item, mask, seq_feature)  # [B, maxlen, H]
    
    # 2. 正负样本 → 特征融合（不含user侧）
    pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)  # [B, maxlen, H]
    neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False)  # [B, maxlen, H]
    
    # 3. 内积计算相似度
    pos_logits = (log_feats * pos_embs).sum(dim=-1)  # [B, maxlen]
    neg_logits = (log_feats * neg_embs).sum(dim=-1)  # [B, maxlen]
    
    # 4. 只保留"下一个是item"的位置
    pos_logits = pos_logits * (next_mask == 1)
    neg_logits = neg_logits * (next_mask == 1)
    
    return pos_logits, neg_logits
```

### 3.5 损失函数（main.py 中）

```python
bce_criterion = nn.BCEWithLogitsLoss(reduction='mean')

pos_labels = torch.ones_like(pos_logits)
neg_labels = torch.zeros_like(neg_logits)

loss = bce_criterion(pos_logits[indices], pos_labels[indices])  # 正样本loss
loss += bce_criterion(neg_logits[indices], neg_labels[indices]) # 负样本loss
loss += args.l2_emb * torch.norm(item_emb)  # L2正则
```

这是一个**二分类问题**：把正样本的 logit 推向1，负样本推向0。

---

## 四、阶段3：推理与输出（infer.py + eval.py）

### 4.1 整体推理流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│ 测试用户序列 │ ──→ │ 模型predict │ ──→ │ 用户query向量    │
└─────────────┘     └─────────────┘     └─────────────────┘
                                              ↓
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│ 候选商品特征 │ ──→ │ save_item_emb│ ──→ │ 商品embedding库 │
└─────────────┘     └─────────────┘     └─────────────────┘
                                              ↓
                                    ┌─────────────────┐
                                    │ FAISS ANN检索   │
                                    │ (query · candidate) │
                                    └─────────────────┘
                                              ↓
                                    ┌─────────────────┐
                                    │ Top-10 推荐结果  │
                                    └─────────────────┘
```

### 4.2 用户向量生成（`model.predict`）

```python
def predict(self, log_seqs, seq_feature, mask):
    log_feats = self.log2feats(log_seqs, mask, seq_feature)
    final_feat = log_feats[:, -1, :]  # 取序列最后一个位置
    return final_feat
```

**关键**：取序列**最后一个有效位置**的输出作为用户兴趣表示。

### 4.3 候选库构建（`get_candidate_emb_parquet`）

从 `candidate/` 目录读取所有待推荐的商品：
1. 读取每个商品的特征
2. 冷启动处理（没见过的特征 value → 0）
3. 多模态特征查表补充
4. 调用 `model.save_item_emb` 生成 embedding
5. 保存为二进制文件 `embedding.fbin` + `id.u64bin`

### 4.4 ANN 近似检索（FAISS）

```bash
faiss_demo \
  --dataset_vector_file_path=embedding.fbin \   # 候选库向量
  --dataset_id_file_path=id.u64bin \            # 候选库ID
  --query_vector_file_path=query.fbin \         # 用户query向量
  --result_id_file_path=id100.u64bin \          # 输出结果
  --query_ann_top_k=10                          # 取Top-10
```

FAISS 使用 **HNSW（图索引）** 做近似最近邻搜索，在大量候选中快速找到最相似的Top-K。

### 4.5 结果输出（`eval.py`）

```python
top10s, user_list = infer()

result = {
    'time': 耗时,
    'top10s': [[creative_id1, creative_id2, ...], ...],  # 每个用户的Top-10推荐
    'user': ['user_xxxx', ...]
}

with open("result.json", 'w') as f:
    json.dump(result, f)
```

最终输出 `result.json`，格式为：

```json
{
  "user": ["user_001", "user_002", ...],
  "top10s": [
    ["cid_123", "cid_456", ...],
    ["cid_789", "cid_abc", ...]
  ],
  "time": 123.45
}
```

---

## 五、全流程总结图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         数据输入层 (dataset.py)                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │ seq/     │  │item_feat/│  │user_feat/│  │ mm_emb/  │                │
│  │用户行为序列│  │商品稀疏特征│  │用户稀疏特征│  │多模态向量 │                │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘                │
│       └──────────────┴──────────────┴──────────────┘                    │
│                      ↓                                                  │
│              ┌───────────────┐                                          │
│              │  __getitem__  │  构造: seq, pos, neg, features           │
│              │  填充+负采样   │                                          │
│              └───────┬───────┘                                          │
└──────────────────────┼──────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         模型层 (model.py)                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐ │
│  │ feat2emb    │───→│ log2feats   │───→│ Transformer Encoder         │ │
│  │ 特征融合     │    │ +位置编码    │    │ (Multi-Head Attention + FFN)│ │
│  └─────────────┘    └─────────────┘    └─────────────────────────────┘ │
│                                                  ↓                      │
│                              ┌─────────────────────────────────┐       │
│                              │ 用户兴趣向量 [B, maxlen, H]      │       │
│                              └───────────────┬─────────────────┘       │
│                                              ↓                          │
│                              ┌─────────────────────────────────┐       │
│                              │ 与 pos/neg emb 做内积 → logits   │       │
│                              └─────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         输出层 (infer.py / eval.py)                      │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ 训练时: pos_logits vs neg_logits → BCE Loss → 优化参数              │ │
│  │ 推理时: 取最后一层 hidden state → 用户query向量                      │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                              ↓                                          │
│                    ┌─────────────────┐                                  │
│                    │ FAISS ANN 检索   │  query · candidate_embedding     │
│                    │ Top-10 近似最近邻 │                                  │
│                    └─────────────────┘                                  │
│                              ↓                                          │
│                    ┌─────────────────┐                                  │
│                    │  result.json    │                                  │
│                    │  {user, top10s} │                                  │
│                    └─────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 六、关键超参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `maxlen` | 101 | 序列最大长度 |
| `hidden_units` | 32 | 模型隐藏维度 |
| `num_blocks` | 1 | Transformer层数 |
| `num_heads` | 1 | 注意力头数 |
| `dropout_rate` | 0.2 | Dropout比例 |
| `batch_size` | 2048 | 训练批次大小 |
| `lr` | 0.001 | 学习率 |
| `mm_emb_id` | `['81']` | 使用的多模态特征 |
| `l2_emb` | 0.0 | Embedding L2正则系数 |
| `norm_first` | False | 是否先LayerNorm再Attention |

---

## 七、Baseline 的核心特点速记

| 特点 | 说明 |
|------|------|
| **论文基础** | SASRec (Self-Attentive Sequential Recommendation, KDD 2018) |
| **骨干网络** | Transformer Encoder + Causal Mask |
| **特征融合** | 手工设计：sparse/array/emb 分别处理 + `userdnn`/`itemdnn` 拼接 |
| **序列构造** | User/Item 交错序列（`token_type` 区分） |
| **训练目标** | BCE Loss（Pointwise 二分类，每步独立） |
| **推理方式** | 取最后一个 hidden state → FAISS ANN 检索 Top-K |
| **Item 表示** | 连续 re-id（`nn.Embedding` 直接 lookup） |
| **Action 使用** | **未使用**（数据中有记录，但模型未接收） |
| **阶段数** | 单阶段端到端 |
| **附加结构** | 无 GNN / LLM / MoE / Semantic ID |
