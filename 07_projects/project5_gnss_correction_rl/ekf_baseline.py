"""二维 CV 模型的 EKF 基线，用于和 RL 方案对比。

State : x = [px, py, vx, vy]
Motion: x_{k+1} = F x_k + w,  F 为 CV 模型
Obs   : z = [px, py] = H x + v
"""
from __future__ import annotations

import numpy as np

from simulate import sample_trajectory


def run_ekf(gnss: np.ndarray, dt: float = 1.0, q_scale: float = 0.5, r_scale: float = 4.0) -> np.ndarray:
    n = gnss.shape[0]
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    Q = np.eye(4) * q_scale
    R = np.eye(2) * r_scale

    x = np.array([gnss[0, 0], gnss[0, 1], 0.0, 0.0])
    P = np.eye(4) * 10.0
    out = np.zeros((n, 2))
    for k in range(n):
        x = F @ x
        P = F @ P @ F.T + Q
        z = gnss[k]
        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(4) - K @ H) @ P
        out[k] = x[:2]
    return out


if __name__ == "__main__":
    sample = sample_trajectory(seed=0)
    fused = run_ekf(sample.gnss)
    raw_err = np.linalg.norm(sample.gnss - sample.truth, axis=1)
    ekf_err = np.linalg.norm(fused - sample.truth, axis=1)
    print(f"Raw GNSS  RMSE = {np.sqrt((raw_err ** 2).mean()):.2f} m")
    print(f"EKF       RMSE = {np.sqrt((ekf_err ** 2).mean()):.2f} m")
