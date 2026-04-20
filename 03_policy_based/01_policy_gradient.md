# 1. 策略梯度（REINFORCE）

## 1.1 为什么要直接学策略？

价值方法的局限：
1. 连续动作做 max 困难
2. 学到的是确定性的 greedy 策略，缺乏天然探索
3. 在多模态最优策略下不灵活

策略方法直接参数化 $\pi_\theta(a|s)$，然后**梯度上升最大化期望回报**。

## 1.2 目标函数

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \gamma^t r_t\right] = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

我们想求 $\nabla_\theta J(\theta)$ 然后梯度上升。

## 1.3 策略梯度定理（最核心公式）

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]}$$

**推导**（用 log-derivative trick）：

$$
\begin{aligned}
\nabla_\theta J &= \nabla_\theta \int p_\theta(\tau) R(\tau) d\tau \\
                &= \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) R(\tau) d\tau \\
                &= \mathbb{E}_\tau[\nabla_\theta \log p_\theta(\tau) R(\tau)]
\end{aligned}
$$

而：

$$\log p_\theta(\tau) = \log \rho(s_0) + \sum_t [\log \pi_\theta(a_t|s_t) + \log P(s_{t+1}|s_t,a_t)]$$

环境部分与 $\theta$ 无关，求导消去 → 只剩 $\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)$。

## 1.4 直觉理解

$$\nabla_\theta J \approx \frac{1}{N} \sum_{i=1}^N \sum_t \nabla_\theta \log \pi_\theta(a_t^{(i)}|s_t^{(i)}) \cdot G_t^{(i)}$$

- $G_t > 0$（这条轨迹回报好） → 增大 $\pi(a_t|s_t)$
- $G_t < 0$（回报差） → 降低 $\pi(a_t|s_t)$

**就像监督学习**：把 (s, a) 当 (x, y)，权重为 $G_t$ 的加权交叉熵。

## 1.5 REINFORCE 算法

```
for episode = 1, M:
    用 π_θ 采一条完整轨迹 τ
    for t = 0, T:
        G_t = sum_{k=t}^T γ^(k-t) r_k
    ∇θ J ≈ Σ_t ∇θ log π_θ(a_t|s_t) · G_t
    θ ← θ + α ∇θ J
```

## 1.6 PyTorch 最小实现（10 行核心）

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical

policy = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, n_actions))
optim = torch.optim.Adam(policy.parameters(), lr=1e-2)

for ep in range(2000):
    log_probs, rewards = [], []
    obs, _ = env.reset()
    while True:
        logits = policy(torch.tensor(obs, dtype=torch.float32))
        dist = Categorical(logits=logits)
        a = dist.sample()
        log_probs.append(dist.log_prob(a))
        obs, r, term, trunc, _ = env.step(a.item())
        rewards.append(r)
        if term or trunc: break
    
    G, returns = 0, []
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)   
    
    loss = -torch.stack([lp * R for lp, R in zip(log_probs, returns)]).sum()
    optim.zero_grad(); loss.backward(); optim.step()
```

## 1.7 致命问题：方差太大

$G_t$ 是单条轨迹的随机变量，方差极大 → 梯度估计噪声大 → 训练慢甚至不收敛。

**解法**：
1. **Baseline**：减一个不依赖 $a$ 的基线 $b(s)$，不改变期望但减小方差
   $$\nabla_\theta J = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t))\right]$$
   最优 baseline 就是 $V^\pi(s_t)$ → **引出 Actor-Critic**！
2. **Reward to go**：只对未来 reward 求和，不算过去
3. **Advantage Function**：$A_t = G_t - V(s_t)$
4. **GAE**：广义优势估计

## 进一步阅读

- Williams 1992, "Simple Statistical Gradient-Following Algorithms"
- [OpenAI Spinning Up: Intro to Policy Gradient](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)

## 思考题

- 推导策略梯度定理
- 为什么需要 baseline？最优 baseline 是什么？
- on-policy 是什么意思？为什么 REINFORCE 是 on-policy？
