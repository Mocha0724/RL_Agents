# 经典论文索引（按算法）

> 详细的定位/导航相关论文清单见 [`../06_localization_rl/papers.md`](../06_localization_rl/papers.md)

## 经典 RL 算法

| 简称 | 论文 | 年份 |
|-----|-----|-----|
| Q-learning | Watkins, *Learning from Delayed Rewards* | 1989 |
| REINFORCE | Williams, *Simple Statistical Gradient-Following Algorithms* | 1992 |
| TD-Gammon | Tesauro, *Temporal Difference Learning and TD-Gammon* | 1995 |

## 深度 RL

| 简称 | 论文 | 链接 |
|-----|-----|-----|
| DQN | Mnih et al., *Playing Atari with DRL* | [arxiv](https://arxiv.org/abs/1312.5602) |
| DQN-Nature | Mnih et al., *Human-level control through DRL* | [Nature](https://www.nature.com/articles/nature14236) |
| Double DQN | van Hasselt et al. | [arxiv](https://arxiv.org/abs/1509.06461) |
| Dueling DQN | Wang et al. | [arxiv](https://arxiv.org/abs/1511.06581) |
| PER | Schaul et al. | [arxiv](https://arxiv.org/abs/1511.05952) |
| Rainbow | Hessel et al. | [arxiv](https://arxiv.org/abs/1710.02298) |
| TRPO | Schulman et al. | [arxiv](https://arxiv.org/abs/1502.05477) |
| GAE | Schulman et al. | [arxiv](https://arxiv.org/abs/1506.02438) |
| A3C | Mnih et al. | [arxiv](https://arxiv.org/abs/1602.01783) |
| PPO | Schulman et al. | [arxiv](https://arxiv.org/abs/1707.06347) |
| DDPG | Lillicrap et al. | [arxiv](https://arxiv.org/abs/1509.02971) |
| TD3 | Fujimoto et al. | [arxiv](https://arxiv.org/abs/1802.09477) |
| SAC | Haarnoja et al. | [arxiv](https://arxiv.org/abs/1801.01290) |
| C51 | Bellemare et al. | [arxiv](https://arxiv.org/abs/1707.06887) |
| QR-DQN | Dabney et al. | [arxiv](https://arxiv.org/abs/1710.10044) |

## Model-Based / 世界模型

| | |
|--|--|
| AlphaGo | Silver et al., Nature 2016 |
| AlphaZero | Silver et al., Science 2018 |
| MuZero | Schrittwieser et al. 2019 |
| Dreamer V1/V2/V3 | Hafner et al. 2020/2021/2023 |
| IRIS | Micheli et al. 2023 |

## Offline RL

| | |
|--|--|
| BCQ | Fujimoto et al. 2019 |
| CQL | Kumar et al. 2020 |
| IQL | Kostrikov et al. 2021 |
| Decision Transformer | Chen et al. 2021 |
| 综述 | Levine et al. 2020 |

## 模仿学习 / IRL

| | |
|--|--|
| DAgger | Ross et al. 2011 |
| GAIL | Ho & Ermon 2016 |
| AIRL | Fu et al. 2017 |
| MaxEnt IRL | Ziebart et al. 2008 |

## RLHF / LLM Alignment

| | |
|--|--|
| PPO RLHF | Ouyang et al. 2022 (InstructGPT) |
| DPO | Rafailov et al. 2023 |
| GRPO | DeepSeek 2024 |
| Constitutional AI | Bai et al., Anthropic 2022 |

## 探索

| | |
|--|--|
| ICM | Pathak et al. 2017 |
| RND | Burda et al. 2018 |
| NoisyNet | Fortunato et al. 2017 |

---

## 论文阅读方法（送给你）

每篇论文 30 分钟即可：
1. **题目 + abstract** (3 min)
2. **Intro 最后一段（贡献列表）+ figure 1** (5 min)
3. **方法的核心公式** (10 min)
4. **实验 main table + 1 个关键 ablation** (5 min)
5. **写一句话笔记**：解决了什么问题、用了什么方法、相比之前好在哪 (5 min)

每周读 3-5 篇，1 年读 200 篇，你就是这个领域的"半个专家"。
