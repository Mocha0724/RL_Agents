# 定位/导航 × RL 论文阅读清单

> 按子方向分类。⭐ 表示推荐重点精读；🔥 表示 2024-2025 最新工作。

---

## 一、综述与基础

| 论文 | 年份 | 推荐 |
|-----|-----|-----|
| Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed) | 2018 | ⭐ 圣经 |
| *A Survey of Reinforcement Learning Informed by Natural Language* | 2019 | - |
| Levine et al., *Offline Reinforcement Learning: Tutorial, Review, and Perspectives* | 2020 | ⭐ |

---

## 二、GNSS / GPS 定位修正

| 论文 | 期刊/会议 | 关键贡献 |
|-----|---------|---------|
| ⭐🔥 *Improving GNSS Positioning Correction Using Deep RL with Adaptive Reward Augmentation* | NAVIGATION 2024 | 自适应 reward shaping + DRL 修正 |
| ⭐ *Increasing GPS Localization Accuracy with Reinforcement Learning* | IEEE TITS 2020 | DRL 替代经验权重 |
| 🔥 *Fusing Vehicle Trajectories and GNSS Measurements... Actor-Critic* | ION GNSS+ 2023 | A2C 融合 |
| *Deep Learning Based GNSS Multipath Detection* | GPS Solutions 2021 | DL 检测多径（不是 RL，但相关） |
| *DeepNav: Deep Reinforcement Learning for Robust GNSS Navigation* | IEEE Sensors 2022 | DRL 导航策略 |

**预备阅读建议**：
1. 先读 Misra & Enge《GPS: Signals, Measurements, and Performance》前 5 章
2. 再读 NAVIGATION 2024 那篇

---

## 三、地图匹配 (Map Matching)

| 论文 | 年份 | 关键贡献 |
|-----|-----|---------|
| Newson & Krumm, *Hidden Markov Map Matching* | ACM SIGSPATIAL 2009 | HMM MM 经典 |
| ⭐🔥 *RLOMM: An Efficient and Robust Online Map Matching with RL* | arxiv 2502.06825, 2025 | OMDP 建模 + GNN + 对比学习 |
| *DeepMM: Deep Learning Based Map Matching with Data Augmentation* | IEEE TKDE 2022 | seq2seq |
| *L2MM: Learning to Map Matching with Deep Models* | KDD 2023 | 学习式 MM |

---

## 四、传感器融合 / Kalman 自适应

| 论文 | 年份 | 关键贡献 |
|-----|-----|---------|
| *Adaptive Kalman Filter using Deep Reinforcement Learning* | IEEE Sensors 2021 | DRL 调 Q/R |
| *KalmanNet: Neural Network Aided Kalman Filtering* | IEEE TSP 2022 | 端到端可微 KF |
| *RL-based Tightly Coupled INS/GNSS* | Sensors 2023 | 紧耦合 INS/GNSS 的 RL 增强 |
| *Learning Adaptive EKF Noise Covariance Online* | ICRA 2024 | 在线学习 Q/R |

---

## 五、轨迹预测与规划 (含自动驾驶)

| 论文 | 年份 | 关键贡献 |
|-----|-----|---------|
| ⭐🔥 *Reinforced Imitative Trajectory Planning for Urban Automated Driving* | arxiv 2410.15607, 2024 | RL + IL 联合 + Bayesian Reward |
| ⭐🔥 *WorldRFT: Latent World Model Planning with RFT* | arxiv 2512.19133, 2025 | 世界模型 + RFT |
| *Reinforcement-Learning-Based Trajectory Learning in Frenet Frame* | MDPI 2024 | Frenet 坐标系下 RL |
| *Trajectory Prediction for Autonomous Driving: Progress, Limitations, Future Directions* | arxiv 2503.03262, 2025 | 综述 |
| *Iterative Reward Prediction for Robust Driving Policies* | 2024 | reward 预测 + 不确定性 |

---

## 六、路径规划与 ETA

| 论文 | 年份 | 关键贡献 |
|-----|-----|---------|
| *DeepETA (Uber)* | KDD 2022 | Transformer ETA |
| *DeepRoute: Path Recommendation* | KDD 2020 | 路径推荐 |
| *DiDi WDR for ETA* | KDD 2018 | Wide-Deep-Recurrent |
| *Reinforcement Learning Based Routing in Stochastic Networks* | IEEE TITS 2021 | RL 网络路由 |

---

## 七、相关基础（DRL 算法本身）

| 论文 | 简称 |
|-----|-----|
| Mnih et al. 2013/2015 | DQN |
| van Hasselt et al. 2015 | Double DQN |
| Wang et al. 2016 | Dueling DQN |
| Schaul et al. 2016 | PER |
| Hessel et al. 2018 | Rainbow |
| Schulman et al. 2015 | TRPO |
| Schulman et al. 2017 | PPO |
| Lillicrap et al. 2015 | DDPG |
| Fujimoto et al. 2018 | TD3 |
| Haarnoja et al. 2018 | SAC |
| Kostrikov et al. 2021 | IQL |
| Rafailov et al. 2023 | DPO |

---

## 八、推荐入门读完顺序（4 周）

| 周 | 必读 (1 篇/周) | 选读 (2 篇/周) |
|---|--------------|---------------|
| 1 | NAVIGATION 2024 (GNSS) | Sutton & Barto 选章; PPO 论文 |
| 2 | RLOMM | HMM Map Matching; DeepMM |
| 3 | Reinforced Imitative Planning | Adaptive KF; KalmanNet |
| 4 | Levine Offline RL Tutorial | IQL; Decision Transformer |

每读一篇，写 1 页中文笔记，回答 4 个问题：
1. 解决了什么问题？
2. 创新点是什么？
3. MDP 怎么建模的？
4. 我能在哪个业务场景用？
