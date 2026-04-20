# 环境搭建

## 1. Python 环境

```bash
conda create -n rl python=3.10 -y
conda activate rl

cd RL_Agents
pip install -r requirements.txt

python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

> Mac M 系列芯片可以用 `mps`：`torch.backends.mps.is_available()`

## 2. Gymnasium 上手

Gymnasium 是 OpenAI Gym 的官方继承者（OpenAI 已停止维护 Gym），API 几乎一样。

```python
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset(seed=42)

for _ in range(200):
    action = env.action_space.sample()       
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

**核心 API（必须烂熟）**：

| API | 作用 |
|-----|------|
| `env.reset() -> (obs, info)` | 重置环境，返回初始状态 |
| `env.step(action) -> (obs, reward, terminated, truncated, info)` | 执行动作 |
| `env.observation_space` | 状态空间（Box / Discrete） |
| `env.action_space` | 动作空间 |

> ⚠️ 注意：从 Gym v0.26 开始，`done` 拆成了 `terminated`（自然结束，如赢/输）和 `truncated`（超时截断）。在 Q-learning 里，bootstrap 时只在 `terminated=True` 时才不加 next state value。

## 3. 主流环境清单

| 环境 | 类型 | 用途 |
|-----|-----|------|
| `CartPole-v1` | 离散 | DQN 入门 |
| `LunarLander-v2` | 离散 | DQN/PPO 进阶 |
| `Pendulum-v1` | 连续 | DDPG/SAC 入门 |
| `MountainCar-v0` | 离散 | 探索算法 |
| `Acrobot-v1` | 离散 | 经典控制 |
| `BipedalWalker-v3` | 连续 | 较难连续控制 |
| `ALE/Pong-v5` | 离散+图像 | DQN 复现 |

## 4. Stable-Baselines3 上手

SB3 = 工业级 RL 算法库，5 行代码就能训练。

```python
from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs")
model.learn(total_timesteps=50_000)
model.save("ppo_cartpole")
```

可视化：`tensorboard --logdir ./logs`

## 5. 自定义 Gym 环境模板（后面定位项目要用）

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MyMapMatchEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(8, dtype=np.float32)
        return self.state, {}

    def step(self, action):
        reward = -1.0                     
        terminated = False
        truncated = False
        info = {}
        return self.state, reward, terminated, truncated, info

    def render(self):
        pass
```

## 6. 排错 FAQ

- **`Box2D` 装不上**：`pip install swig && pip install gymnasium[box2d]`
- **Mac 渲染黑屏**：`render_mode="rgb_array"` + matplotlib 保存图片
- **训练巨慢**：先看是否 GPU 利用率上来了，CartPole 这种小环境其实 CPU 比 GPU 快
- **wandb 不想用**：注释掉相关代码，用 tensorboard 本地查看

## 7. 推荐 IDE & 工作流

- VSCode + Python + Jupyter 插件
- 或者 JupyterLab（本仓库 ipynb 多）
- Git LFS（如果存大模型 ckpt）
