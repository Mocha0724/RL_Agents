# 05 LLM Agent 时代的 RL

> **本章定位**：2023 之后，"Agent" 这个词的含义从「RL agent」逐渐扩展到「LLM agent」。本章帮你理解这两个世界的交集与差异。

## 文件列表

| 文件 | 主题 |
|-----|-----|
| [01_agent_concept.md](./01_agent_concept.md) | Agent 概念演进：从 RL agent 到 LLM agent |
| [02_react_pattern.md](./02_react_pattern.md) | ReAct / Plan-and-Execute / 推理-行动循环 |
| [03_rlhf.md](./03_rlhf.md) | RLHF / DPO / GRPO：用 RL 对齐 LLM |
| [04_frameworks.md](./04_frameworks.md) | LangChain / LangGraph / AutoGen / CrewAI |
| [05_tool_use.md](./05_tool_use.md) | Function Calling / MCP / Computer Use |

## 两个 Agent 世界

```mermaid
graph TB
    subgraph "经典 RL Agent"
        E1[Environment] -- s,r --> A1[Policy π θ]
        A1 -- a --> E1
    end
    subgraph "LLM Agent"
        E2[Tools / APIs / 用户] -- observation --> A2[LLM]
        A2 -- thought + action --> E2
    end
```

**核心相似**：都是 perception → decision → action 闭环
**核心差异**：

| | 经典 RL Agent | LLM Agent |
|--|-------------|-----------|
| 决策器 | 神经网络 π_θ（小） | LLM（巨大） |
| 训练方式 | 大量交互试错 | 预训练 + 指令微调 + RLHF |
| 状态/动作空间 | 数值向量/离散/连续 | 自然语言 |
| 推理能力 | 弱（end-to-end） | 强（CoT、Plan） |
| 部署成本 | 低 | 高（每步 API 调用） |
| 数据效率 | 极低 | 极高（zero-shot） |

## LLM Agent 与经典 RL 的边界

LLM Agent 适合的场景：
- 用户意图理解、自然语言交互
- 需要常识/知识的开放性任务
- 多步推理 + 工具调用
- 数据标注、信息抽取等

经典 RL/控制更擅长的场景：
- 高频、低延迟、确定性强的控制（机器人、SLAM、定位）
- 状态空间是数值向量、动作是连续控制量
- 需要严格可验证、可重复的策略

**两者并非互斥**：LLM Agent 常用 RLHF/DPO 来对齐，本质又用回了 RL；而经典 RL 系统也开始用 LLM 做 high-level planner。
