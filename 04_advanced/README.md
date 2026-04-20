# 04 进阶主题

> **本章定位**：拓宽视野的章节，覆盖经典 model-free 之外的几大方向。**Offline RL 和 Imitation Learning** 在工业界有大量历史日志数据的场景中尤其有用，建议优先阅读。

## 文件列表

| 文件 | 主题 |
|-----|-----|
| [01_model_based_rl.md](./01_model_based_rl.md) | 模型基础 RL / 世界模型 / Dreamer / MuZero |
| [02_offline_rl.md](./02_offline_rl.md) | 离线 RL：CQL / IQL / Decision Transformer |
| [03_imitation_learning.md](./03_imitation_learning.md) | 模仿学习 / BC / DAgger / GAIL / IRL |
| [04_marl.md](./04_marl.md) | 多智能体 RL：QMIX / MAPPO |
| [05_meta_hrl.md](./05_meta_hrl.md) | 元学习 / 分层 RL |
| [06_exploration.md](./06_exploration.md) | 探索策略：ICM / RND / NoisyNet |

## 推荐阅读顺序

```mermaid
flowchart LR
    O[02 Offline RL] --> I[03 Imitation Learning]
    I --> M[01 Model-based]
    M --> E[06 Exploration]
    E --> A[04 MARL]
    A --> H[05 Meta/HRL]
```

## 为什么 Offline RL + IL 在工业界很重要？

很多真实业务场景具有以下特点：
- **历史日志海量**（如 GPS 轨迹、推荐系统日志、用户操作日志）
- **在线试错代价高**（不能让线上系统"摆烂"试错）
- **真值/标注数据稀有**

→ 这正是 **Offline RL** 的标准场景：从已有数据离线学习策略，部署时不再交互。
→ 也是 **Behavior Cloning + DAgger** 的天然舞台：从专家轨迹学策略。
