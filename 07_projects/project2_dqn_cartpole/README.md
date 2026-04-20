# Project 2: DQN on CartPole

**难度**：⭐⭐ · **预计耗时**：2 天

## 题目

从零实现 DQN，在 CartPole-v1 上达到 ≥ 195 平均回报。然后扩展实现 Double DQN / Dueling DQN，对比效果。

## 学习目标

- ✅ 实现 Replay Buffer
- ✅ 实现 Target Network
- ✅ 调通 DQN 超参
- ✅ 体会 Double DQN 的稳定性提升
- ✅ 学会用 TensorBoard 看实验

## 任务清单

- [ ] 实现 vanilla DQN（参考 `../../02_value_based/02_dqn_cartpole.ipynb`）
- [ ] 实现 Double DQN（target 改一行）
- [ ] 实现 Dueling DQN（架构改 V/A 分支）
- [ ] 三种方法各跑 5 个 seed，统计均值/方差
- [ ] 用 TensorBoard 记录 loss / Q 均值 / epsilon
- [ ] **挑战**：加入 PER（优先经验回放）

## 评估指标

| 算法 | 收敛 step（达到 195）| 最终性能均值 | 方差 |
|-----|-------------------|------------|-----|
| DQN | ? | ? | ? |
| Double DQN | ? | ? | ? |
| Dueling DQN | ? | ? | ? |

填表后写到 `report.md`。

## 进阶

跑通后试试：
- LunarLander-v2（动作 4 个，更复杂）
- 用 SB3 的 DQN 跑同样环境，看你手写的有没有更好（一般差不多甚至略好，因为代码简单调起来快）
