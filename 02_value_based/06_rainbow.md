# 6. Rainbow：DQN 改进的集大成者

DeepMind 2017 把 6 大 DQN 改进合体，得到 Rainbow。

## 6.1 6 大组件

| 改进 | 解决问题 |
|-----|---------|
| **Double DQN** | 高估偏差 |
| **Dueling Network** | V/A 分离表示 |
| **Prioritized Replay** | 样本利用率 |
| **Multi-step Bootstrapping** | TD vs MC 折中（n-step return） |
| **Distributional RL (C51)** | 估计回报分布而非期望 |
| **Noisy Net** | 参数空间探索（替代 ε-greedy） |

## 6.2 n-step Return

$$y_t^{(n)} = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n \max_{a'} Q_{\theta^-}(s_{t+n}, a')$$

n=1 是经典 DQN；n→∞ 趋近 MC。常用 n=3。

## 6.3 Distributional RL（核心思想）

不再学 $\mathbb{E}[G]$，而是学回报的**分布** $Z(s,a)$。这能保留风险信息，对探索更友好。

$$Z(s, a) \stackrel{D}{=} R + \gamma Z(s', \arg\max_{a'} \mathbb{E}[Z(s', a')])$$

C51 用 51 个原子离散化分布，QR-DQN 用分位数回归。

## 6.4 综合效果

Rainbow 在 57 款 Atari 游戏的 median human-normalized score 远超单项改进。

## 6.5 工业实践建议

- 不一定全部用：Double + Dueling + n-step 是性价比最高的组合
- C51/QR-DQN 实现较复杂，调参敏感
- Noisy Net 在简单环境效果不明显

## 进一步阅读

- Hessel et al. 2018, ["Rainbow: Combining Improvements in Deep RL"](https://arxiv.org/abs/1710.02298)
- C51: Bellemare et al. 2017, ["A Distributional Perspective on RL"](https://arxiv.org/abs/1707.06887)
