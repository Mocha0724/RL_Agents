# Project 5: 基于 RL 的 GNSS 定位修正

**难度**：⭐⭐⭐⭐⭐ · **预计耗时**：2 周

## 题目

复现 NAVIGATION 2024 那篇论文的简化版：
- 模拟带噪声的 GNSS 轨迹（带可控的多径/NLOS 模型）
- 用 PPO 学一个修正策略，输入 (原始 GNSS + 上下文)，输出修正量 (Δx, Δy)
- 与 EKF baseline 对比 RMSE

## 学习目标

- ✅ 自定义连续动作 RL 环境
- ✅ 用 PPO 解连续控制
- ✅ 设计 reward shaping
- ✅ 与传统 EKF 对比，**学会写实验报告**

## 文件清单

- `simulate.py` —— 生成合成 GNSS 数据（含真值）
- `env.py` —— `GNSSCorrectionEnv`
- `ekf_baseline.py` —— 经典 EKF
- `train_ppo.py` —— 用 SB3 PPO 训练
- `eval.py` —— RMSE 对比

## MDP 建模

| 元素 | 设计 |
|-----|------|
| State | 原始 GNSS (lat, lon) + 速度 + DOP + 历史 5 帧位置 + 当前可见卫星数 + 当前路网约束方向 |
| Action | 二维修正量 $\Delta = (\Delta x, \Delta y) \in [-15, 15]$ m |
| Reward | $-\|p_{raw} + \Delta - p_{truth}\|_2$ + shaping |
| Algorithm | PPO（连续动作） |

## 起手代码

我们提供了 `simulate.py`、`env.py`、`ekf_baseline.py` 的最小可运行版本。
你需要：
1. 跑通 `train_ppo.py`（SB3 PPO 几行代码）
2. 写 `eval.py`，对比三种方法：raw GPS / EKF / RL
3. 画 CDF 曲线对比误差分布
4. 调研 reward shaping，至少试 3 种

## 评估指标

| 方法 | 平均误差 (m) | 95% 误差 (m) | 最大误差 (m) |
|-----|-----------|-------------|-----------|
| Raw GPS | ? | ? | ? |
| EKF | ? | ? | ? |
| RL (yours) | ? | ? | ? |

## 进阶玩法

- 用真实数据：Google Smartphone Decimeter Challenge ([Kaggle 数据](https://www.kaggle.com/competitions/google-smartphone-decimeter-challenge))
- 实现 Adaptive Reward Augmentation（NAVIGATION 2024 创新）
- 加 IMU 数据，做 Tightly Coupled
- 把 RL 模型蒸馏成端侧小模型 (< 1MB)
- 整理为一篇技术博客或会议 poster
