# 5. 元学习（Meta-RL）与 分层强化学习（HRL）

## 5.1 Meta-RL：学如何学

**目标**：训练一个 agent，使它能在**新任务**上**快速适应**（few-shot）。

```mermaid
graph LR
    Tasks[任务分布 p_T] --> Sample[采样任务 T_i]
    Sample --> Adapt[内循环：用少量数据适应]
    Adapt --> Eval[评估]
    Eval --> Outer[外循环：更新 meta-参数]
    Outer --> Sample
```

## 5.2 主流方法

### MAML（Model-Agnostic Meta-Learning）
- 学一个初始参数 $\theta$，使它在每个新任务上 1 步梯度后就能很好
- 二阶梯度，计算贵

### RL² / PEARL（Recurrent Meta-RL）
- 用 RNN/Transformer 编码"任务 context"
- 把适应过程内嵌到 forward pass

## 5.3 HRL：分层动作

把决策拆成 **高层（选 option/skill）+ 低层（执行 option）**。

```
高层每 K 步选一次 option
低层执行选定的 option
```

代表：Options Framework, FeUdal Networks, HIRO

**应用**：
- 长 horizon 任务（如导航 + 取物）
- 路径规划：高层选下一个路口，低层控制方向盘

## 5.4 适用场景

Meta-RL 与 HRL 都属于"上层方法论"，更适合下面这类问题：

- **任务分布**：训练时见过一类任务，测试时希望模型在分布内的新任务上快速适应（meta-RL）
- **长 horizon 决策**：单步策略很难直接学到长期目标，需要把任务拆成子任务（HRL）
- **可复用 skill**：同一组低层 skill 在多个高层任务里反复使用

举例：一个机器人导航任务里，"高层选下一个路口、低层控制方向盘"就是经典的 HRL 设定；"在多个城市/场景间迁移定位修正策略"则可以套 meta-RL 的思路。

## 进一步阅读

- MAML: Finn et al. 2017
- 张志华《强化学习》Ch.10
