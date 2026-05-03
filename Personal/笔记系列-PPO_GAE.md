# GAE（Generalized Advantage Estimation）详解

> GAE 是 PPO 中最常用的优势估计方法，在"偏差-方差权衡"之间找到了一个灵活可调的平衡点。

---

## 一、为什么需要 GAE？

### 1.1 优势函数的本质

优势函数衡量的是：**某个动作比"在同样状态下的平均水平"好多少**。

$$A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)$$

- $Q^\pi(s,a)$：在状态 $s$ 执行动作 $a$ 后的期望累计回报
- $V^\pi(s)$：在状态 $s$ 按照策略 $\pi$ 行动的期望累计回报

**优势为正**：这个动作比平均水平好，应该鼓励  
**优势为负**：这个动作比平均水平差，应该抑制  

### 1.2 问题的来源

$Q$ 和 $V$ 都是未知的，需要估计。这就引出了两种极端方法，各有利弊。

---

## 二、两种极端方法

### 2.1 方法一：MC 回报（Monte-Carlo，无偏但高方差）

用实际观测到的累计回报 $G_t$ 代替 $Q$，减去价值估计 $V(s_t)$：

$$\hat{A}_t^{MC} = G_t - V(s_t)$$

其中 $G_t = \sum_{l=0}^{\infty} \gamma^l r_{t+l}$ 是从 $t$ 时刻开始的实际折扣回报。

**优点**：无偏——只要样本够多，期望就是真实的优势  
**缺点**：方差极大——一条轨迹的回报可能因为随机性而大幅波动

**直觉类比**：

> 你评估一步棋的好坏，方法是**下完整盘棋**看最终输赢。如果赢了，这步棋就是"好"的；输了就是"坏"的。但一盘棋的输赢受到后面几十步的影响，这步棋本身可能并不差，只是后面走错了。用最终结果来评判单步棋，噪声太大。

---

### 2.2 方法二：TD 残差（Temporal-Difference，低方差但有偏）

只用单步信息：

$$\hat{A}_t^{TD} = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**优点**：方差低——只用了一步的奖励信息，噪声小  
**缺点**：有偏——$V(s_{t+1})$ 本身是不完美的估计，误差会累积  

**直觉类比**：

> 你评估一步棋，只看**下一步**局面的变化。这很稳定（不需要下完一整盘），但太短视了——一步好棋可能暂时让局面看起来变差（比如牺牲一个子换先手），但后续才能体现价值。只看下一步会误判。

---

### 2.3 两者的对比

| 维度 | MC 回报 | TD 残差 |
|------|---------|---------|
| 公式 | $G_t - V(s_t)$ | $r_t + \gamma V(s_{t+1}) - V(s_t)$ |
| 偏差 | 无偏 | 有偏（依赖 $V$ 的估计质量） |
| 方差 | 极高 | 低 |
| 需要的样本 | 大量轨迹才能收敛 | 少量样本即可 |
| 对 Critic 准确度的依赖 | 低（$V$ 只作基线） | 高（$V$ 的误差直接影响 $\hat{A}$） |

---

## 三、GAE 的核心公式

GAE 是 MC 和 TD 之间的**加权插值**：

$$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V$$

### 3.1 展开写

$$
\begin{aligned}
\hat{A}_t &= \delta_t \\
&\quad + (\gamma\lambda) \delta_{t+1} \\
&\quad + (\gamma\lambda)^2 \delta_{t+2} \\
&\quad + (\gamma\lambda)^3 \delta_{t+3} \\
&\quad + \cdots
\end{aligned}
$$

其中单步 TD 残差：

$$\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$$

### 3.2 两个超参的含义

| 超参 | 名称 | 作用 | 典型值 |
|------|------|------|--------|
| $\gamma$ | 折扣因子 | 控制远期奖励的重要性。$\gamma=0.99$ 表示 100 步后的奖励衰减为 $0.99^{100} \approx 0.37$ | 0.99 ~ 0.999 |
| $\lambda$ | GAE 衰减系数 | **GAE 的核心参数**，控制 MC 和 TD 的混合比例。$\lambda=0$ 纯 TD，$\lambda=1$ 纯 MC | 0.9 ~ 0.97 |

### 3.3 $\lambda$ 的直观理解

$\lambda$ 决定了你"回头看多远"来评估当前动作：

| $\lambda$ | 行为 | 偏差 | 方差 |
|-----------|------|------|------|
| **0** | 只看下一步（$\delta_t$） | 高 | 低 |
| **0.5** | 看接下来几步，权重指数衰减 | 中等 | 中等 |
| **0.95** | 看很长一段，但远处权重小 | 低 | 较高 |
| **1** | 看到轨迹终点（完整 MC） | 无 | 极高 |

**类比**：

> 下棋时评估一步棋：
> - $\lambda=0$：只看对手下一步的回应
> - $\lambda=0.5$：看接下来 3-5 步的走势
> - $\lambda=0.95$：推演到终局，但远处的局面打折扣
> - $\lambda=1$：必须下完这盘棋，看最终输赢

---

## 四、GAE 的等价形式

GAE 还有另一种理解方式——**指数加权的回报与价值的差**：

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \left( G_{t:l} - V(s_t) \right)$$

其中 $G_{t:l}$ 是从 $t$ 开始的 $l$ 步截断回报：

$$G_{t:l} = \sum_{m=0}^{l} \gamma^m r_{t+m} + \gamma^{l+1} V(s_{t+l+1})$$

这说明 GAE 实际上是**多个不同长度截断回报的加权平均**，每个截断回报都减去了同一个 $V(s_t)$。这是一种更紧凑的理解，但实际计算中通常用 TD 残差的递推形式。

---

## 五、GAE 的高效计算：反向递推

GAE 不需要对每个位置展开无穷级数，可以用**反向递推**在 $O(T)$ 时间内算完整个序列：

$$
\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}
$$

### 5.1 递推算法

```
从序列最后一个 token 往前遍历：
  Â_T = δ_T                       # 最后一个位置没有后续
  for t = T-1, T-2, ..., 0:
    Â_t = δ_t + γλ * Â_{t+1}
```

### 5.2 伪代码

```python
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    rewards: 每个位置的奖励 [r_0, r_1, ..., r_T]
    values:  Critic预测的价值 [V(s_0), V(s_1), ..., V(s_T), V(s_{T+1})]
             注意V(s_{T+1})是序列结束后的值，通常设为0
    """
    T = len(rewards)
    deltas = []
    
    # Step 1: 计算每个位置的 TD 残差 δ_t
    for t in range(T):
        # δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * values[t+1] - values[t]
        deltas.append(delta)
    
    # Step 2: 反向递推计算 GAE
    advantages = [0] * T
    advantages[-1] = deltas[-1]  # 最后一个位置
    
    for t in range(T-2, -1, -1):
        # Â_t = δ_t + γλ * Â_{t+1}
        advantages[t] = deltas[t] + gamma * lam * advantages[t+1]
    
    return advantages
```

---

## 六、大模型场景下的 GAE

### 6.1 特殊挑战：非逐 Token 奖励

在大模型 RLHF 中，Reward Model 通常只在**序列末尾**给一次分数，而不是每个 token 都有奖励：

$$r_t = 0 \quad \text{for } t = 0, 1, ..., T-1$$
$$r_T = R(x, y)$$

这意味着中间所有位置的 TD 残差变成：

$$\delta_t = \gamma V(s_{t+1}) - V(s_t) \quad (t < T)$$

只有最后一步有奖励信号：

$$\delta_T = R(x,y) - V(s_T)$$

### 6.2 递推过程的具体表现

假设序列长度 $T=5$，$\gamma=0.99$，$\lambda=0.95$：

```
位置:     0          1          2          3          4
奖励:     0          0          0          0          R
δ:        γV1-V0     γV2-V1     γV3-V2     γV4-V3     R-V4

反向递推:
Â_4 = δ_4 = R - V4
Â_3 = δ_3 + γλ * Â_4 = (γV4-V3) + γλ(R-V4)
Â_2 = δ_2 + γλ * Â_3 = (γV3-V2) + γλ * [(γV4-V3) + γλ(R-V4)]
...
```

**关键观察**：
- 最后一步的优势直接反映 RM 打分
- 前面的位置通过 $V(s)$ 的差异来间接传递奖励信号
- 如果 Critic 学得好（$V$ 准确），前面位置也能得到合理的优势估计
- 如果 Critic 学得差，误差会通过递推传播

### 6.3 实际简化处理

实践中，大模型 RLHF 常做**进一步简化**：

**Option A：直接用整句回报**

$$\hat{A} = R(x,y) - V(s_0)$$

只有一个全局优势值，所有 token 用同一个 $\hat{A}$。最简单但粗糙。

**Option B：用折扣回报减价值**

对每个位置 $t$：

$$G_t = \gamma^{T-t} \cdot R(x,y)$$
$$\hat{A}_t = G_t - V(s_t)$$

越靠后的 token 背的奖励越大，体现它们对最终结果的影响更大。

**Option C：标准 GAE（推荐）**

按照上述反向递推计算。效果最好但对 Critic 质量有要求。

### 6.4 各方法的对比

| 方法 | 公式 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|----------|
| 全局优势 | $R - V(s_0)$ | 极简，无 Critic 也行 | 完全无法区分不同位置贡献 | 快速实验、短序列 |
| 折扣回报 | $\gamma^{T-t}R - V(s_t)$ | 简单，体现位置差异 | 没有利用 Critic 的局部预测 | 中等长度序列 |
| 标准 GAE | $\sum(\gamma\lambda)^l \delta_{t+l}$ | 最精细，偏差-方差平衡好 | 依赖 Critic 质量 | 推荐的标准做法 |

---

## 七、GAE 与 Critic 的共生关系

### 7.1 Critic 的目标

Critic 训练目标是让 $V(s_t)$ 尽量接近真实的累计回报 $G_t$：

$$L^{VF} = \frac{1}{2} \mathbb{E}_t \left[ (V_\phi(s_t) - G_t)^2 \right]$$

### 7.2 为什么 Critic 的准确度决定 GAE 的质量

回顾 TD 残差：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

如果 $V$ 是完美的（即 $V(s) = \mathbb{E}[G|s]$），那么：

$$\mathbb{E}[\delta_t] = \mathbb{E}[r_t + \gamma V(s_{t+1}) - V(s_t)] = 0$$

这就是 Bellman 方程。此时 $\delta_t$ 的期望为零，它捕捉的**恰好是超出期望的部分**——也就是优势。

**Critic 越准** → $\delta_t$ 越接近真实的瞬时优势 → GAE 估计越可靠  
**Critic 越差** → $\delta_t$ 含有系统性偏差 → GAE 估计偏离真实优势

### 7.3 实际训练中的配合

```
每次 PPO 迭代:
  1. 用 π_old 采样一批回答
  2. Critic 预测每个位置的价值 V(s_t)
  3. 用 GAE 计算每个位置的优势 Â_t
  4. 更新 Critic: 让 V(s_t) 更接近实际回报 G_t
  5. 更新 Actor: 用 PPO-Clip + Â_t 指导策略更新
  6. 同步 π_old ← π_θ
```

**重要**：Critic 的损失和 Actor 的损失是**独立的**——Critic 学的是"预言"，Actor 学的是"行动"。但两者共享同一批数据，通常也共享骨干网络。

---

## 八、GAE 的偏差-方差分析

### 8.1 数学推导

GAE 的期望优势可以写成：

$$\mathbb{E}\left[\hat{A}_t^{GAE}\right] = \sum_{l=0}^{\infty} (\gamma\lambda)^l \mathbb{E}[\delta_{t+l}]$$

如果 Critic 有偏差 $\epsilon_V$（即 $V \neq V^\pi$），那么每个 $\delta_t$ 都含有 $O(\epsilon_V)$ 的误差。Gover 等人的论文证明了：

**GAE 的偏差量级**：$O\left(\frac{1-\lambda}{1-\gamma\lambda} \epsilon_V\right)$

- $\lambda=0$：偏差 $O(\epsilon_V)$ —— 完全依赖 Critic 质量
- $\lambda \to 1$：偏差 $\to 0$ —— 趋近无偏的 MC
- $\gamma\lambda < 1$：偏差被几何级数压缩

### 8.2 实际调参建议

| 场景 | 推荐 $\gamma$ | 推荐 $\lambda$ | 理由 |
|------|--------------|----------------|------|
| Critic 很准（训练充分） | 0.99 | 0.9 ~ 0.95 | 可以承受稍低的 $\lambda$，减少方差 |
| Critic 不准（训练初期） | 0.99 | 0.97 ~ 0.99 | 提高 $\lambda$ 减少偏差 |
| 序列很长（>512 tokens） | 0.999 | 0.95 | 长序列需要更大的 $\gamma$ 保留远程信号 |
| 序列很短（<128 tokens） | 0.99 | 0.9 | 短序列信号传递快，不需要太大 $\lambda$ |
| 奖励稀疏（RM 只在末尾打分） | 0.99 | 0.95 ~ 0.97 | 依赖 Critic 传递信号，需要较大 $\lambda$ |

---

## 九、GAE 的变体

### 9.1 GAE 与归一化

实践中通常对 GAE 结果做**按 batch 归一化**：

$$\tilde{A}_t = \frac{\hat{A}_t - \text{mean}(\hat{A})}{\text{std}(\hat{A}) + 10^{-8}}$$

这确保 PPO 的更新步长在不同 batch 之间保持一致，不受具体奖励尺度影响。

### 9.2 截断 GAE（Truncated GAE）

对长序列（如 2048 tokens），无限求和不现实。实际实现中：

$$\hat{A}_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}$$

即自然截断到序列末尾。这正是反向递推算法的默认行为。

### 9.3 N-step 优势估计（GAE 的简化版）

如果不用 GAE，可以固定看 $n$ 步：

$$\hat{A}_t^{(n)} = \sum_{l=0}^{n-1} \gamma^l r_{t+l} + \gamma^n V(s_{t+n}) - V(s_t)$$

这是 GAE 的一个特例——只保留前 $n$ 项且 $\lambda=1$。但 GAE 通过 $\lambda$ 提供了更平滑的 $n$ 步控制，通常效果更好。

---

## 十、总结速查

### 核心公式

| 公式 | 含义 |
|------|------|
| $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ | TD 残差：单步预测误差 |
| $\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$ | GAE：指数加权的 TD 残差和 |
| $\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}$ | 反向递推形式（实际计算用） |

### 关键直觉

> GAE 就是**多个不同"预测视野"的 TD 误差的加权平均**。
> 
> - 最近的预测（$\delta_t$）权重最高，最可信
> - 越远的预测权重按 $\gamma\lambda$ 指数衰减
> - $\lambda$ 控制"信多远的预测"

### 一句话记忆

> **GAE = 往回看的望远镜，$\lambda$ 就是焦距。焦距短（$\lambda$ 小）只看近处，稳定但短视；焦距长（$\lambda$ 大）看得远，全面但可能模糊。**
