# 3. Double DQN：解决高估偏差

## 3.1 问题：DQN 为什么高估？

DQN 的 target：

$$y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')$$

由于 **max 操作 + 估计噪声**，会系统性地把价值估高。

**直觉**：假设真实 Q 全为 0，但估计 Q 有 ±0.1 的噪声。`max([-0.1, 0.05, 0.1])` = 0.1，恒为正 → 估计偏高。

## 3.2 Double DQN 的修正

把「**选动作**」和「**评估动作**」用两个网络分开做：

$$y = r + \gamma\, Q_{\theta^-}\Big(s',\ \underbrace{\arg\max_{a'} Q_\theta(s', a')}_{\text{用 online net 选}}\Big)$$

**改一行代码**：

```python
with torch.no_grad():
    a_max = q_net(s2).argmax(1, keepdim=True)
    q_next = target_net(s2).gather(1, a_max).squeeze(1)
    target = r + gamma * (1 - d) * q_next
```

## 3.3 效果

在 Atari 上 Double DQN 比 DQN 稳定很多，部分游戏分数显著提升。

## 进一步阅读

- van Hasselt et al. 2015, ["Deep Reinforcement Learning with Double Q-learning"](https://arxiv.org/abs/1509.06461)

## 面试常考点

- 为什么 max 会导致高估？
- 怎么用一行代码修复？
