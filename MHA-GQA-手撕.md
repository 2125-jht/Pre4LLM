# MHA & GQA 手撕伪代码

## 1. MHA (Multi-Head Attention)

```python
def MHA(x, W_q, W_k, W_v, W_o, n_heads):
    # x: [batch, seq, dim]
    batch, seq, dim = x.shape
    head_dim = dim // n_heads

    # 1) Linear投影
    Q = x @ W_q    # [b, s, dim]
    K = x @ W_k
    V = x @ W_v

    # 2) 分头
    Q = Q.view(batch, seq, n_heads, head_dim).transpose(1, 2)  # [b, h, s, d]
    K = K.view(batch, seq, n_heads, head_dim).transpose(1, 2)
    V = V.view(batch, seq, n_heads, head_dim).transpose(1, 2)

    # 3) Scaled Dot-Product Attention
    scores = Q @ K.transpose(-2, -1) / sqrt(head_dim)  # [b, h, s, s]
    attn   = softmax(scores, dim=-1)
    out    = attn @ V                                  # [b, h, s, d]

    # 4) 拼接 + 输出投影
    out = out.transpose(1, 2).contiguous().view(batch, seq, dim)
    return out @ W_o
```

---

## 2. GQA (Grouped Query Attention)

```python
def GQA(x, W_q, W_k, W_v, W_o, n_heads, n_kv_heads):
    # n_heads:    Q 的头数 (如 8)
    # n_kv_heads: K/V 的头数 (如 2)
    batch, seq, dim = x.shape
    head_dim = dim // n_heads

    # 1) Linear投影 (K/V 的输出维度是 n_kv_heads * head_dim)
    Q = x @ W_q    # [b, s, dim]
    K = x @ W_k    # [b, s, n_kv_heads * head_dim]
    V = x @ W_v

    # 2) 分头：Q 分 n_heads，K/V 分 n_kv_heads
    Q = Q.view(batch, seq, n_heads,    head_dim).transpose(1, 2)  # [b, n_h,  s, d]
    K = K.view(batch, seq, n_kv_heads, head_dim).transpose(1, 2)  # [b, n_kvh, s, d]
    V = V.view(batch, seq, n_kv_heads, head_dim).transpose(1, 2)

    # 3) 把 K/V 复制(repeat)到与 Q 相同的头数，以便做矩阵乘法
    n_repeat = n_heads // n_kv_heads
    K = K.repeat_interleave(n_repeat, dim=1)   # [b, n_h, s, d]
    V = V.repeat_interleave(n_repeat, dim=1)

    # 4) Attention
    scores = Q @ K.transpose(-2, -1) / sqrt(head_dim)
    attn   = softmax(scores, dim=-1)
    out    = attn @ V

    # 5) 拼接 + 输出投影
    out = out.transpose(1, 2).contiguous().view(batch, seq, dim)
    return out @ W_o
```

---



---

## 5. 交叉熵损失 (Cross-Entropy Loss)

**公式**：

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{N}\sum_{i=1}^{N} \log p_{\theta}(y_i \mid x_i)
$$

```python
def ce_loss(logits, targets):
    log_p = log_softmax(logits, dim=-1)
    nll   = -gather(log_p, targets)
    return mean(nll)
```

---

## 6. PPO Loss

**公式**：

$$
\mathcal{L}^{\text{CLIP}} = -\mathbb{E}_t\left[ \min\left( r_t A_t, \; \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) A_t \right) \right]
$$

$$
\mathcal{L}^{\text{VF}} = \frac{1}{2} \mathbb{E}_t\left[ \max\left( (V_{\theta} - R_t)^2, \; (V_{\theta}^{\text{clip}} - R_t)^2 \right) \right]
$$

```python
def ppo_loss(π_old, π_new, A, V_old, V_new, R):
    # Policy clip
    ratio  = exp(π_new - π_old)
    surr1  = ratio * A
    surr2  = clip(ratio, 1-ε, 1+ε) * A
    L_clip = -mean(min(surr1, surr2))

    # Value clip
    v_clip = V_old + clip(V_new - V_old, -c, c)
    L_vf   = 0.5 * mean(max((V_new - R)², (v_clip - R)²))

    return L_clip + L_vf
```

---

## 7. GRPO Loss

**公式**：

$$
A_i = \frac{R_i - \text{mean}(R)}{\text{std}(R) + \epsilon}
$$

$$
\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[ \min\left( r A, \; \text{clip}(r, 1-\varepsilon, 1+\varepsilon) A \right) \right] + \beta \, \mathbb{E}\left[ \log \pi_{\text{ref}} - \log \pi_{\theta} \right]
$$

```python
def grpo_loss(π_θ, π_ref, rewards, old_logp, ε, β):
    # 组内标准化 advantage（无 Critic）
    A = (rewards - mean(rewards, axis=group)) / (std(rewards, axis=group) + 1e-8)

    # Policy clip
    ratio    = exp(log(π_θ) - old_logp)
    surr1    = ratio * A
    surr2    = clip(ratio, 1-ε, 1+ε) * A
    L_policy = -mean(min(surr1, surr2))

    # KL penalty
    KL = mean(log(π_ref) - log(π_θ))

    return L_policy + β * KL
```

**核心**：一组采样 → 组内标准化得 Advantage → 无需 Critic。

---

## 8. DPO Loss

**公式**：

$$
r_{\theta}(y \mid x) = \log \frac{\pi_{\theta}(y \mid x)}{\pi_{\text{ref}}(y \mid x)}
$$

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\left( \beta \left( r_{\theta}(y_w \mid x) - r_{\theta}(y_l \mid x) \right) \right)
$$

```python
def dpo_loss(π_θ, π_ref, y_w, y_l, β):
    r_θ   = log π_θ(y_w)   - log π_θ(y_l)
    r_ref = log π_ref(y_w) - log π_ref(y_l)

    loss = -log_sigmoid(β * (r_θ - r_ref))
    return mean(loss)
```

**核心**：直接优化偏好对，无需显式 Reward Model，用 `β` 控制偏离 ref 的程度。

