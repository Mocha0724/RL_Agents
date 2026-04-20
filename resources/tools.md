# 工具与生态

## 核心 Python 库

| 库 | 用途 | 安装 |
|---|------|------|
| **gymnasium** | RL 环境标准接口（Gym 继任者） | `pip install gymnasium[all]` |
| **stable-baselines3** | 工业级 RL 算法实现 | `pip install stable-baselines3` |
| **sb3-contrib** | SB3 扩展（QR-DQN、TQC 等） | `pip install sb3-contrib` |
| **CleanRL** | 单文件实现，最适合学习 | `git clone` |
| **rllib (Ray)** | 分布式 RL | `pip install ray[rllib]` |
| **TorchRL** | PyTorch 官方 RL 库 | `pip install torchrl` |
| **PettingZoo** | 多智能体 RL 环境 | `pip install pettingzoo` |

## 实验跟踪

| 工具 | 备注 |
|-----|------|
| **TensorBoard** | 默认选择，本地 |
| **Weights & Biases (wandb)** | 云端，免费额度够用 |
| **MLflow** | 企业内部部署 |

## 仿真环境

| 名字 | 用途 |
|-----|------|
| **MuJoCo** | 经典机器人物理仿真 |
| **PyBullet** | 开源替代 MuJoCo |
| **Isaac Gym (NVIDIA)** | GPU 加速仿真 |
| **CARLA** | 自动驾驶 |
| **SUMO** | 交通仿真 |
| **AirSim** | 无人机/汽车 |

## 定位/导航专用

| 工具 | 用途 |
|-----|------|
| **RTKLIB** | 开源 GNSS 处理 |
| **gnss-sdr** | 软件无线电 GNSS |
| **GTSAM** | 因子图优化（C++ + Python 绑定） |
| **Ceres Solver** | 非线性最小二乘（Google） |
| **OSMnx** | OSM 路网下载与分析 |
| **shapely** | 几何运算 |
| **pyproj** | 坐标系转换 |
| **rtree** | 空间索引 |
| **networkx** | 图算法 |

## 命令行 / 工程

| 工具 | 用途 |
|-----|------|
| **VSCode** | 主力 IDE |
| **JupyterLab** | Notebook 开发 |
| **conda / mamba** | 环境管理 |
| **uv** | 超快 pip 替代品 |
| **pytest** | 单元测试 |
| **ruff / black** | 代码格式化 |
| **pre-commit** | Git hook |

## 学习/查阅辅助

| | |
|--|--|
| **arxiv-sanity** | 论文订阅 |
| **PapersWithCode** | 论文 + 代码 + 排行榜 |
| **Connected Papers** | 论文关系图 |
| **Notion / Obsidian** | 笔记系统 |
| **Anki** | 记公式 / 算法卡片 |

## 推荐 GitHub Star List（必关注）

- `vwxyzjn/cleanrl` — 单文件 RL 实现
- `DLR-RM/stable-baselines3` — SB3
- `Farama-Foundation/Gymnasium` — Gym 继任者
- `pytorch/rl` — TorchRL
- `huggingface/deep-rl-class` — HF Deep RL 课程仓库
- `boyu-ai/Hands-on-RL` — 张伟楠《动手学强化学习》代码
- `datawhalechina/easy-rl` — EasyRL
