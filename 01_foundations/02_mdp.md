# 2. 马尔可夫决策过程（MDP）

> MDP 是 RL 的「数学语言」。所有问题的第一步都是：**怎么把它建模成 MDP？**

## 2.1 形式化定义

一个 MDP 是一个五元组：

$$
\mathcal{M} = \langle \mathcal{S},\ \mathcal{A},\ \mathcal{P},\ \mathcal{R},\ \gamma \rangle
$$

| 元素 | 含义 | 例子（地图匹配） |
|------|-----|----------------|
| $\mathcal{S}$ | 状态空间 | 当前轨迹 + 候选路段集合 |
| $\mathcal{A}$ | 动作空间 | {选路段 1, 2, 3, …} |
| $\mathcal{P}(s'\|s,a)$ | 转移概率 | 选了路段后下一步的轨迹分布 |
| $\mathcal{R}(s,a)$ | 奖励函数 | 是否匹配正确（+1/-1） |
| $\gamma$ | 折扣因子 | 0.99（长程考虑） |

---

## 2.2 马尔可夫性质

$$
P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} \mid s_t, a_t)
$$

**人话**：未来只依赖当前状态，不依赖历史。

> ⚠️ **现实陷阱**：很多问题不天然满足马尔可夫性。比如车辆轨迹中，"上一秒的速度"是有用的历史信息。
>
> **解决办法**：
> 1. 把历史塞进状态（如最近 N 帧 GPS） → 仍是 MDP，状态扩维
> 2. 用 RNN/Transformer 编码历史 → POMDP
> 3. 接受 MDP 假设的局限，用工程经验弥补

---

## 2.3 策略 Policy

策略是 **状态到动作的映射**：

- **确定性策略**：$\pi(s) = a$
- **随机性策略**：$\pi(a|s) = P(a|s)$

为什么需要随机策略？
1. 探索（exploration）
2. 部分可观测环境下，随机策略可能严格优于确定性策略
3. 多智能体博弈中，混合策略可能是 Nash 均衡

---

## 2.4 回报与价值

**回报（Return）**：从 t 时刻起的折扣累积奖励

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^\infty \gamma^k r_{t+k+1}
$$

**γ 的作用**：
- $\gamma = 0$：只看眼前，短视
- $\gamma \to 1$：看长远，但收敛慢、方差大
- 实战常用 0.95~0.99

**状态价值函数**：在策略 $\pi$ 下，从状态 $s$ 出发的期望回报

$$
V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]
$$

**动作价值函数**：进一步指定第一步动作

$$
Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]
$$

**两者关系**：

$$
V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s,a)
$$

$$
Q^\pi(s,a) = \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma V^\pi(s')\right]
$$

---

## 2.5 优势函数 Advantage

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

衡量「在状态 s，选动作 a 比平均水平好多少」。

**为什么重要？** PPO/A2C 等所有 actor-critic 算法都用 advantage 替代原始 reward 来减小方差。

---

## 2.6 一个具象例子：3×3 GridWorld

```
+---+---+---+
| S |   |   |
+---+---+---+
|   | # |   |        S = 起点
+---+---+---+        # = 障碍
|   |   | G |        G = 终点 (+1)
+---+---+---+
```

- $\mathcal{S}$：9 个格子（除去 #）
- $\mathcal{A}$：{上, 下, 左, 右}
- $\mathcal{P}$：朝目标方向走 90% 成功，10% 滑到旁边
- $\mathcal{R}$：到 G +1，每步 -0.04
- $\gamma$：0.9

> 💡 这就是 `04_dynamic_programming.ipynb` 要解的环境。

---

## 2.7 把定位问题建模成 MDP（思考练习）

**任务**：在 GNSS 城市峡谷场景下，输出修正后的位置。

| 元素 | 一种建模 |
|------|---------|
| State | 当前 GNSS（lat,lon,alt）+ 历史 5 帧 + 卫星可见数 + DOP + IMU |
| Action | 修正向量 $\Delta = (\Delta x, \Delta y) \in [-5m, 5m]^2$（连续） |
| Reward | $-\|\hat{p} - p_{\text{truth}}\|_2$（如果有真值）<br>或 $-\|\hat{p} - p_{\text{map\_road}}\|_2$（如果只有道路约束） |
| Discount | 0.95 |

> ⚠️ 实际 paper 里 reward 设计是核心难点，第 6 章会详细分析 *Adaptive Reward Augmentation* 这篇 NAVIGATION 2024 的论文。

---

## 进一步阅读

- Sutton & Barto Ch.3
- Puterman《Markov Decision Processes》—— 学术圣经

## 面试常考点

- 写出 MDP 五元组，并解释 $\gamma$ 的作用
- 什么是 POMDP？怎么处理？
- 如何把一个实际业务问题建模成 MDP？（必考开放题，准备 1-2 个自己的例子）
