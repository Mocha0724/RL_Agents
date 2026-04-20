"""GNSS 仿真器：生成"真值轨迹 + 带噪 GNSS 观测"。

本仿真简化为 2D 平面，包含三类噪声：
    1. 基础高斯噪声（开阔地带，σ ≈ 1m）
    2. 多径噪声（在指定区域 σ 显著放大）
    3. NLOS 偏置（在指定区域观测有方向性偏差）
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrajectorySample:
    truth: np.ndarray   # shape (T, 2)
    gnss: np.ndarray    # shape (T, 2)
    speed: np.ndarray   # shape (T,)  m/s
    dop: np.ndarray     # shape (T,)  (HDOP)
    nsat: np.ndarray    # shape (T,)  visible satellites


URBAN_CANYON_REGIONS = [
    ((50.0, 100.0), (-20.0, 20.0)),     
    ((180.0, 240.0), (50.0, 90.0)),
]


def in_canyon(x: float, y: float) -> bool:
    for (xlo, xhi), (ylo, yhi) in URBAN_CANYON_REGIONS:
        if xlo <= x <= xhi and ylo <= y <= yhi:
            return True
    return False


def make_truth(rng: np.random.Generator, total_steps: int = 300, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """生成一条平滑随机走的真值轨迹和速度序列。"""
    pts = np.zeros((total_steps, 2))
    speeds = np.zeros(total_steps)
    heading = rng.uniform(-np.pi, np.pi)
    speed = rng.uniform(8, 15)            
    pos = np.array([0.0, 0.0])
    for t in range(total_steps):
        heading += rng.normal(0, 0.05)    
        speed += rng.normal(0, 0.3)
        speed = float(np.clip(speed, 5.0, 20.0))
        pos = pos + speed * dt * np.array([np.cos(heading), np.sin(heading)])
        pts[t] = pos
        speeds[t] = speed
    return pts, speeds


def add_gnss_noise(rng: np.random.Generator, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = truth.shape[0]
    gnss = truth.copy()
    dop = np.zeros(T)
    nsat = np.zeros(T, dtype=int)
    for t in range(T):
        x, y = truth[t]
        canyon = in_canyon(x, y)
        if canyon:
            sigma = rng.uniform(8.0, 15.0)
            bias = rng.normal(0, 5.0, size=2)
            dop[t] = rng.uniform(3.0, 6.0)
            nsat[t] = rng.integers(4, 7)
        else:
            sigma = 1.5
            bias = np.zeros(2)
            dop[t] = rng.uniform(0.8, 1.5)
            nsat[t] = rng.integers(8, 14)
        gnss[t] = truth[t] + rng.normal(0, sigma, size=2) + bias
    return gnss, dop, nsat


def sample_trajectory(seed: int = 0, total_steps: int = 300) -> TrajectorySample:
    rng = np.random.default_rng(seed)
    truth, speed = make_truth(rng, total_steps)
    gnss, dop, nsat = add_gnss_noise(rng, truth)
    return TrajectorySample(truth=truth, gnss=gnss, speed=speed, dop=dop, nsat=nsat)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sample = sample_trajectory(seed=42)
    plt.figure(figsize=(10, 6))
    plt.plot(sample.truth[:, 0], sample.truth[:, 1], "g-", label="Truth")
    plt.scatter(sample.gnss[:, 0], sample.gnss[:, 1], c="r", s=8, alpha=0.6, label="GNSS (noisy)")
    for (xlo, xhi), (ylo, yhi) in URBAN_CANYON_REGIONS:
        plt.gca().add_patch(plt.Rectangle((xlo, ylo), xhi - xlo, yhi - ylo, alpha=0.2, color="gray", label="Canyon"))
    plt.legend()
    plt.gca().set_aspect("equal")
    plt.title("Simulated trajectory (truth vs noisy GNSS)")
    plt.savefig("sample_trajectory.png", dpi=120)
    print("Saved sample_trajectory.png")
