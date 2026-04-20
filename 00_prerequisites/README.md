# 00 预备知识

## 你需要会什么

| 主题 | 最低要求 | 推荐资源 |
|-----|---------|---------|
| 线性代数 | 矩阵乘法、特征值、范数 | 3Blue1Brown 线代本质 |
| 概率统计 | 期望、方差、贝叶斯、马尔可夫链 | 陈希孺《概率论与数理统计》 |
| 最优化 | 梯度下降、KKT、对偶 | Boyd《Convex Optimization》前 5 章 |
| 机器学习 | 线性回归、神经网络、反向传播 | 吴恩达 ML / 周志华西瓜书 |
| Python/PyTorch | tensor、autograd、nn.Module | 官方 60-min Blitz |
| Gymnasium | env.reset / step API | 见 [env_setup.md](./env_setup.md) |

## 文件列表

- [`math_basics.md`](./math_basics.md) — RL 中真正用到的数学（不是大全，是高频用的）
- [`ml_basics.md`](./ml_basics.md) — RL 视角下的神经网络复习
- [`env_setup.md`](./env_setup.md) — Gymnasium / SB3 / 项目环境搭建
- [`pytorch_quickstart.ipynb`](./pytorch_quickstart.ipynb) — 5 分钟 PyTorch 速通

## 自检清单

- [ ] 能手推矩阵求导：$\frac{\partial}{\partial \mathbf{w}} \mathbf{w}^T A \mathbf{w}$
- [ ] 知道为什么策略梯度里要用 log
- [ ] 能用 PyTorch 写一个 3 层 MLP 并训练它拟合 sin(x)
- [ ] 跑通 `gymnasium.make("CartPole-v1")` 的 random policy
- [ ] 装好 conda 环境，`import torch; torch.cuda.is_available()` 能正常返回
