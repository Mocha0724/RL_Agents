# Project 3: PPO on LunarLander

**难度**：⭐⭐⭐ · **预计耗时**：3 天

## 题目

实现 PPO，在 `LunarLander-v2`（离散动作，目标分数 ≥ 200）和 `LunarLanderContinuous-v2`（连续动作）上跑通。

## 学习目标

- ✅ 实现 GAE
- ✅ 实现 PPO-Clip
- ✅ 处理离散 + 连续动作
- ✅ 学会调 PPO 关键超参（clip ε, entropy coef, n_epochs）

## 任务清单

- [ ] 离散版 PPO（参考 `../../03_policy_based/04_ppo_cartpole.ipynb`）
- [ ] 连续版 PPO：策略输出 (mean, std)，用 Normal 分布
- [ ] 实现 GAE
- [ ] 加 advantage normalization
- [ ] 5 个 seed 训练
- [ ] **挑战**：实现并行多 env 采样（VectorEnv）

## 关键代码片段（连续动作）

```python
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                                    nn.Linear(64, 64), nn.Tanh())
        self.mean = nn.Linear(64, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))    
        self.critic = nn.Linear(64, 1)
    
    def forward(self, x):
        h = self.shared(x)
        mean = self.mean(h)
        std = self.log_std.exp().expand_as(mean)
        v = self.critic(h).squeeze(-1)
        return mean, std, v
    
    def act(self, x):
        mean, std, v = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        a = dist.sample()
        return a, dist.log_prob(a).sum(-1), v
```

## 评估

| 任务 | 目标分数 | 你的分数 |
|-----|--------|---------|
| LunarLander-v2 | 200 | ? |
| LunarLanderContinuous-v2 | 200 | ? |

## 进阶

- 跑 BipedalWalker-v3（更难连续控制）
- 跑 Atari Pong（图像 + CNN backbone）
- 与 SAC 对比连续控制效果
