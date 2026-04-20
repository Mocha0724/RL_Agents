"""DQN 训练脚本：在 MapMatchingEnv 上训练并保存模型。

使用方法:
    python train.py --total-steps 50000

训练完成后会在 results/ 下保存：
    dqn_mm.pt          # 模型参数
    train_curve.png    # 学习曲线
"""
from __future__ import annotations

import argparse
import os
import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env import MapMatchingEnv


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(np.array(s2), dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


def train(args):
    os.makedirs("results", exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = MapMatchingEnv(seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q = QNet(obs_dim, n_actions).to(device)
    target = QNet(obs_dim, n_actions).to(device)
    target.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=1e-3)
    buf = ReplayBuffer()

    eps_start, eps_end, eps_decay = 1.0, 0.05, args.total_steps // 2

    obs, _ = env.reset()
    ep_reward = 0
    returns = []
    for step in range(1, args.total_steps + 1):
        eps = max(eps_end, eps_start - step * (eps_start - eps_end) / eps_decay)
        if random.random() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = int(q(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).argmax(-1).item())
        obs2, r, term, trunc, _ = env.step(a)
        done = term or trunc
        buf.push(obs, a, r, obs2, float(term))
        obs = obs2
        ep_reward += r

        if done:
            returns.append(ep_reward)
            ep_reward = 0
            obs, _ = env.reset()

        if len(buf) >= 1000:
            s, ab, rb, s2, db = buf.sample(64)
            s, ab, rb, s2, db = s.to(device), ab.to(device), rb.to(device), s2.to(device), db.to(device)
            q_pred = q(s).gather(1, ab.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q_next = target(s2).max(-1).values
                y = rb + 0.99 * (1 - db) * q_next
            loss = ((q_pred - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 5.0)
            opt.step()

        if step % 500 == 0:
            target.load_state_dict(q.state_dict())

        if step % 2000 == 0 and returns:
            print(f"step={step:6d}  eps={eps:.3f}  recent_return={np.mean(returns[-20:]):.2f}")

    torch.save(q.state_dict(), "results/dqn_mm.pt")
    plt.figure(figsize=(10, 4))
    plt.plot(returns, alpha=0.3)
    if len(returns) > 20:
        smooth = np.convolve(returns, np.ones(20) / 20, mode="valid")
        plt.plot(np.arange(19, len(returns)), smooth, linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN on MapMatching Env")
    plt.grid(True)
    plt.savefig("results/train_curve.png", dpi=120)
    print("Saved results/dqn_mm.pt and results/train_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=0)
    train(parser.parse_args())
