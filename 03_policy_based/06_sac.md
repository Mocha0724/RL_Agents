# 6. SAC (Soft Actor-Critic)：最大熵 RL

> 当前连续控制 SOTA，工业机器人、四足、自动驾驶仿真中的常用算法。

## 6.1 最大熵 RL 框架

传统 RL 目标：$\max_\pi \mathbb{E}[\sum r_t]$
最大熵 RL：$\max_\pi \mathbb{E}\Big[\sum (r_t + \alpha H[\pi(\cdot|s_t)])\Big]$

**直觉**：在保证回报的前提下，让策略尽可能随机。
**好处**：
1. 探索更充分
2. 多模态策略（不同最优解都能学到）
3. 鲁棒性更强

## 6.2 软价值函数

$$V^\pi(s) = \mathbb{E}_{a \sim \pi}[Q^\pi(s,a) - \alpha \log \pi(a|s)]$$

$$Q^\pi(s,a) = r + \gamma \mathbb{E}_{s'}[V^\pi(s')]$$

## 6.3 SAC 的 3 个关键设计

1. **双 Q 网络**：缓解高估
   $$y = r + \gamma \big(\min_{i=1,2} Q_{\phi_i^-}(s', a') - \alpha \log \pi(a'|s')\big),\ a' \sim \pi(\cdot|s')$$
2. **Squashed Gaussian Policy**：策略输出 mean+std，再用 tanh 压到 [-1, 1]
   ```python
   normal = Normal(mean, std)
   z = normal.rsample()                # 重参数化
   action = torch.tanh(z)
   log_prob = normal.log_prob(z) - torch.log(1 - action**2 + 1e-6)
   ```
3. **自动温度调节**：$\alpha$ 用拉格朗日乘子学习
   $$L(\alpha) = \mathbb{E}\big[-\alpha \log \pi(a|s) - \alpha \bar{H}\big]$$
   $\bar{H}$ 是目标熵（如 $-\dim(\mathcal{A})$）

## 6.4 完整算法

```
初始化 Q_φ1, Q_φ2, Q_φ1⁻, Q_φ2⁻, π_θ, α, replay buffer D
for step = 1, ...:
    a ~ π_θ(·|s)
    s', r, done = env.step(a)
    D.push(s, a, r, s', done)
    
    采样 batch (s, a, r, s', done):
    # 1. Critic 更新
    a' ~ π_θ(·|s')
    y = r + γ * (1-done) * (min Q⁻(s', a') - α log π(a'|s'))
    update Q_φ1, Q_φ2 by minimizing (Q - y)²
    
    # 2. Actor 更新（重参数化）
    a_new ~ π_θ(·|s)   # 可微采样
    L_π = α log π(a_new|s) - min Q(s, a_new)
    update θ
    
    # 3. 温度更新
    L_α = -α (log π(a_new|s) + H_target)
    update α
    
    # 4. 软更新 target
    Q_φi⁻ ← τ Q_φi + (1-τ) Q_φi⁻
```

## 6.5 关键超参

| | 推荐 |
|--|-----|
| lr | 3e-4 (all networks) |
| γ | 0.99 |
| τ (target soft update) | 0.005 |
| batch size | 256 |
| buffer size | 1M |
| target entropy | -dim(A) |
| α 初值 | 0.2 |

## 6.6 与定位/导航场景

- 自动驾驶仿真中的转向/加减速控制（连续动作）
- IMU/GPS 融合中卡尔曼滤波器**Q/R 矩阵参数自适应**（连续）

## 进一步阅读

- Haarnoja et al. 2018, ["Soft Actor-Critic"](https://arxiv.org/abs/1801.01290)
- 推荐阅读 [CleanRL SAC 单文件实现](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py)

## 面试常考点

- SAC 为什么能在连续控制上超过 PPO？
- 解释 reparameterization trick
- 自动温度调节是怎么做的？
