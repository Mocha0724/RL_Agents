# Project 1: GridWorld Q-Learning

**难度**：⭐ · **预计耗时**：1 天

## 题目

在一个 6×9 的 GridWorld 中实现表格 Q-learning，让 agent 从起点走到终点，避开障碍。

```
. . . . . . . . G        S = start
. . . # # # # . .        # = wall  
. . . . . . . . .        G = goal
. # . . . . . . .
. # . . . . . . .
S . . . . . . . .
```

## 学习目标

- ✅ 实现 ε-greedy
- ✅ 理解 on-policy / off-policy
- ✅ 对比 SARSA vs Q-learning
- ✅ 学会画学习曲线

## 任务清单

- [ ] 实现 `GridWorld` 类（reset / step）
- [ ] 实现 Q-learning（表格）
- [ ] 实现 SARSA
- [ ] 训练 1000 episode，画学习曲线
- [ ] 可视化最优策略
- [ ] **挑战**：加入"风"——某些列每步会被吹一格，看策略变化

## 参考

`../../01_foundations/06_temporal_difference.ipynb` 提供了 Cliff Walking 的完整实现，可以直接迁移。
