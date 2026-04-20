# 07 实战项目

> **理念**：每个项目对应前面章节学过的知识，**从最简单到逐步贴近业务**。

## 项目列表

| 项目 | 难度 | 对应章节 | 核心算法 | 预计耗时 |
|-----|-----|--------|---------|---------|
| [project1_gridworld_qlearning](./project1_gridworld_qlearning/) | ⭐ | 01 | 表格 Q-learning | 1 天 |
| [project2_dqn_cartpole](./project2_dqn_cartpole/) | ⭐⭐ | 02 | DQN | 2 天 |
| [project3_ppo_lunarlander](./project3_ppo_lunarlander/) | ⭐⭐⭐ | 03 | PPO | 3 天 |
| [project4_map_matching_rl](./project4_map_matching_rl/) ★ | ⭐⭐⭐⭐ | 06 | DQN + 自定义 Env | 1 周 |
| [project5_gnss_correction_rl](./project5_gnss_correction_rl/) ★ | ⭐⭐⭐⭐⭐ | 06 | PPO + Real GNSS | 2 周 |

## 推荐做法

1. **不要直接 copy 代码**：先看 README 理解题目，自己写一遍，卡住了再看参考实现
2. **写实验日志**：每次跑都记下超参/曲线/结论
3. **写复盘**：项目完成后写 1-2 页博客（中文/英文都行），可以发自己的 GitHub
4. **可视化优先**：能画图就画图，方便你回看也方便面试时讲

## 如何展示给面试官

```
GitHub 仓库 README：
- 一句话说项目做了什么
- 一张 GIF/图说明效果
- 一段说明你解决的核心难点
- 链接到 blog post
```

> 互联网算法岗最爱的简历项目长这样：「**复现 RLOMM 简版，在自建路网上比 HMM 提升 X%**」

## 通用工程规范

每个项目目录约定包含：
- `README.md`：题目 + 思路 + 结果
- `train.py` 或 `train.ipynb`：训练代码
- `eval.py`：评估代码
- `requirements.txt` 或继承根目录
- `results/`：曲线图、模型 ckpt
- `report.md`：1-2 页复盘
