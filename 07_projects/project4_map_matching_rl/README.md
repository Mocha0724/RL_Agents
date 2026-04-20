# Project 4: 基于 RL 的地图匹配（RLOMM 简版）★

**难度**：⭐⭐⭐⭐ · **预计耗时**：1 周

## 题目

复现一个 **RLOMM 风格的地图匹配 demo**：
- 用合成路网（5-20 条路段）
- 模拟带噪声的 GPS 轨迹
- 用 DQN 学习"当前 GPS 点应匹配到哪条路段"
- 与 baseline（最近邻 / HMM）对比准确率

## 学习目标

- ✅ 自定义 gym Env（含路网拓扑）
- ✅ 把"路段选择"建模成离散动作 RL
- ✅ 设计 reward（匹配正确 + 时序一致）
- ✅ 与传统 baseline 对比

## 文件清单

- `make_road_network.py` —— 生成合成路网（节点、边、几何）
- `simulate_gps.py` —— 沿真实路径采样 + 加高斯噪声生成 GPS 轨迹
- `env.py` —— `MapMatchingEnv` 自定义 Gym 环境
- `train.py` —— DQN 训练
- `baselines.py` —— 最近邻 / HMM-Viterbi 实现
- `eval.py` —— 准确率对比

## MDP 建模

| 元素 | 设计 |
|------|-----|
| State | (当前 GPS 点的 x,y, 速度, 历史 3 帧 GPS) + 周围 K 个候选路段的相对位置/方向 |
| Action | 离散：选择 K 个候选路段中的哪一个（K=5） |
| Reward | +1 匹配正确，-1 错误。再加 -0.5 如果与上一次选的路段不连通（鼓励连续性） |

## 起手代码框架

我们提供了 **`env.py` 的最小可运行版本**，剩下的留给你完成。

跑通后写一段对比表：

| 方法 | 准确率 | 路口附近准确率 |
|-----|-------|--------------|
| Nearest Neighbor | 70% | 50% |
| HMM (Viterbi) | 85% | 65% |
| **DQN (yours)** | ? | ? |

## 提示

- 路网可以用 `networkx` 表示；几何可以用 `shapely`
- 候选路段查询用空间索引：`shapely.STRtree` 或 `rtree`
- 训练数据：固定 N 条真实路径，沿它们采样 GPS 加噪
- 测试数据：另外 M 条路径

## 进阶

- 用真实 OSM 数据的某个小区域（如 0.5 km × 0.5 km）
- 用 GNN 编码路网拓扑
- 实现 RLOMM 的 future-oriented reward
- 写一篇博客发出来
