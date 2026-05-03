# 核心公式伪代码

---

## 1. 交叉熵

### 公式

$$\mathcal{L} = -\frac{1}{N}\sum_{i}\log(p_{y_i})$$

### 伪代码

```python
import torch.nn.functional as F

log_probs = F.log_softmax(logits, dim=-1)          # logits: [B, C]
loss = -log_probs[range(B), labels].mean()          # labels: [B]
```

### 1.2 Softmax（数值稳定版）

### 公式

$$p_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

> 减去最大值是为了防止指数爆炸，不影响结果。

### 伪代码

```python
max_logits = logits.max(dim=-1, keepdim=True).values    # [B, 1]
exp_logits = torch.exp(logits - max_logits)
probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)   # [B, C]
```

---

## 2. PPO Loss（完整版）

### 2.1 Value Loss

### 公式

$$\mathcal{L}^V = \frac{1}{2} \mathbb{E}\left[(V_\theta(s_t) - R_t)^2\right]$$

### 伪代码

```python
value_loss = 0.5 * F.mse_loss(values, returns)      # values: [B, T], returns: [B, T]
```

### 2.2 Policy Loss（Clipped Surrogate）

### 公式

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$$
$$\mathcal{L}^{CLIP} = -\hat{\mathbb{E}}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

### 伪代码

```python
ratio = torch.exp(log_probs - old_log_probs)        # [B, T]
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
policy_loss = -torch.min(surr1, surr2).mean()
```

### 2.3 Entropy Bonus

### 公式

$$\mathcal{H} = -\mathbb{E}_{a \sim \pi}\left[\log \pi(a|s)\right]$$

### 伪代码

```python
probs = F.softmax(logits, dim=-1)
entropy = -(probs * log_probs).sum(dim=-1).mean()
```

### 2.4 PPO 总 Loss

### 公式

$$\mathcal{L}^{PPO} = \mathcal{L}^{CLIP} + c_1 \mathcal{L}^V - c_2 \mathcal{H}$$

### 伪代码

```python
total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
```

### 2.5 PPO-RLHF 变体（加 KL）

### 公式

$$\mathcal{L}^{RLHF} = \mathcal{L}^{CLIP} + c_1 \mathcal{L}^V + c_{KL} D_{KL}(\pi_\theta  \pi_{ref})$$

### 伪代码

```python
with torch.no_grad():
    ref_log_probs = ref_model(states).log_softmax(dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)

kl = (log_probs - ref_log_probs).mean()             # [B, T]
total_loss = policy_loss + 0.5 * value_loss + kl_coef * kl
```

---

## 3. GRPO Loss

### 公式

$$A_i = \frac{r_i - \text{mean}(r_j)}{\text{std}(r_j)}$$
$$\mathcal{L} = -\min(r_i A_i,\ \text{clip}(r_i) A_i) + \beta D_{KL}$$

### 伪代码

```python
A = (rewards - rewards.mean()) / (rewards.std() + 1e-8)    # 组内归一化
ratio = torch.exp(log_probs - old_log_probs)
surr1, surr2 = ratio * A, torch.clamp(ratio, 1-eps, 1+eps) * A
kl = log_probs - ref_log_probs
loss = -(torch.min(surr1, surr2) - beta * kl).mean()
```

---

## 4. DPO Loss

### 公式

$$\mathcal{L} = -\log\sigma\left(\beta\log\frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)}\right)$$

### 伪代码

```python
ratio_c = (policy_log_probs_chosen - ref_log_probs_chosen).sum(dim=-1)
ratio_r = (policy_log_probs_rejected - ref_log_probs_rejected).sum(dim=-1)
loss = -F.logsigmoid(beta * (ratio_c - ratio_r)).mean()
```

---

## 5. InfoNCE

### 公式

$$\mathcal{L} = -\log \frac{\exp(q \cdot k^+ / \tau)}{\exp(q \cdot k^+ / \tau) + \sum_{i=1}^{K} \exp(q \cdot k_i^- / \tau)}$$

> 本质：把正例从 K+1 个候选里"挑出来"，等价于交叉熵形式的对比损失。

### 参数说明


| 符号      | 含义                    |
| ------- | --------------------- |
| $q$     | query 的 embedding     |
| $k^+$   | 正例（与 q 配对）的 embedding |
| $k_i^-$ | 第 i 个负例的 embedding    |
| $\tau$  | 温度系数，越小越严格            |
| $K$     | 负例数量                  |


### 伪代码

```python
import torch.nn.functional as F

# q: [B, D], pos: [B, D], neg: [K, D]
pos_sim = (q * pos).sum(dim=-1) / tau           # [B]
neg_sim = (q @ neg.T) / tau                      # [B, K]

logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=-1)    # [B, K+1]
labels = torch.zeros(B, dtype=torch.long)         # 正例在第 0 位
loss = F.cross_entropy(logits, labels)
```

---

| 算法      | 核心公式记忆点                                                          |
| ------- | ---------------------------------------------------------------- |
| Softmax | 先减 max 再 exp，防数值溢出                                               |
| PPO     | `min(ratio*A, clip(ratio)*A)` + `MSE(value, return)` + `entropy` |
| GRPO    | 组内 `(r - mean)/std` 当 A，不用 Critic                                |
| DPO     | `logsigmoid(beta * (chosen_ratio - rejected_ratio))`             |
| InfoNCE | `cross_entropy( [pos_sim, neg_sims], label=0 )`                  |


