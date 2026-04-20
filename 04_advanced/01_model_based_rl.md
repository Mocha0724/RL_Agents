# 1. Model-based RL & 世界模型

## 1.1 思想

Model-free RL 直接从交互学策略，样本效率低（CartPole 都要几万步）。
Model-based RL 先学**环境模型** $\hat{P}(s'|s,a), \hat{R}(s,a)$，再在模型里"做梦"训练策略，**样本效率提升 10-100 倍**。

```mermaid
flowchart LR
    Real[真实环境] -- s,a,r,s' --> Buf[Buffer]
    Buf --> Model[学习环境模型 P̂, R̂]
    Model --> Imag[在模型里 imagine 大量轨迹]
    Imag --> Policy[更新策略 π]
    Policy --> Real
```

## 1.2 三类典型方法

### A. Dyna 风格（早期）
- 学一个简单模型，与真实环境交替更新策略
- 代表：Dyna-Q

### B. 基于规划（Planning）
- 不学策略，每次决策时用模型搜索
- 代表：**MuZero**（国际象棋/围棋/Atari 通杀，AlphaZero 的 model-based 版）

### C. 基于潜空间世界模型
- 学一个 latent state space 的动力学
- 在 latent 中 rollout + Dreamer 风格的 actor-critic
- 代表：**PlaNet, Dreamer V1/V2/V3, IRIS**

## 1.3 Dreamer V3 的核心架构

```mermaid
graph LR
    obs[观测 o_t] --> Enc[编码器]
    Enc --> z[隐状态 z_t]
    z --> RSSM[RSSM 动力学<br/>z_t+1 ~ p z_t+1 | z_t, a_t]
    RSSM --> z2[z_t+1]
    z2 --> Dec[解码器: 预测 o, r, done]
    z --> Actor[Actor]
    Actor --> a[a_t]
    z --> Critic[Critic]
```

DreamerV3 在 150+ 任务上用同一套超参 work，成为 model-based 的里程碑。

## 1.4 与传统模型方法的关系

很多经典工程方法本质上也属于 model-based，例如：

- **卡尔曼滤波 / EKF**：线性（或线性化）高斯系统下的最优滤波器，定位与导航中的核心算法之一
- **MPC（模型预测控制）**：依赖已知动力学方程做 receding horizon 规划
- **AlphaZero / MuZero**：把"模型 + 搜索"思路推到极致

以定位为例，里面就有大量天然的"世界模型"：车辆运动学（IMU 双积分）、GPS 观测模型，KF 实际上就是一个线性高斯世界模型。

→ Model-based RL 的一个常见落地方式不是**取代**这些传统模型，而是与它们结合：让网络去学**残差**、**自适应噪声参数**或**未知部分的动力学**，在物理先验之上继续优化。

## 1.5 优缺点

| 优 | 缺 |
|---|---|
| 样本效率高 | 模型偏差累积（model bias） |
| 可做 long-horizon planning | 模型架构复杂 |
| 可做 transfer learning | 训练不稳定 |
| 同时获得世界理解 | 计算成本高 |

## 进一步阅读

- Dreamer V3: Hafner et al. 2023, ["Mastering Diverse Domains through World Models"](https://arxiv.org/abs/2301.04104)
- MuZero: Schrittwieser et al. 2019
