"""评估三种方法的 RMSE/CDF：raw GNSS / EKF / RL。

用法:
    python eval.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from ekf_baseline import run_ekf
from env import GNSSCorrectionEnv
from simulate import sample_trajectory


def collect_errors(n_episodes: int = 50, seed_offset: int = 100):
    raw_errs, ekf_errs, rl_errs = [], [], []
    model = PPO.load("results/ppo_gnss") if os.path.exists("results/ppo_gnss.zip") else None
    if model is None:
        print("[warn] results/ppo_gnss.zip not found — skipping RL evaluation")

    for ep in range(n_episodes):
        sample = sample_trajectory(seed=seed_offset + ep)
        raw_errs.extend(np.linalg.norm(sample.gnss - sample.truth, axis=1))

        ekf_out = run_ekf(sample.gnss)
        ekf_errs.extend(np.linalg.norm(ekf_out - sample.truth, axis=1))

        if model is not None:
            env = GNSSCorrectionEnv(seed=seed_offset + ep)
            obs, _ = env.reset()
            ep_errs = []
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, info = env.step(action)
                ep_errs.append(info["err"])
            rl_errs.extend(ep_errs)

    return np.asarray(raw_errs), np.asarray(ekf_errs), np.asarray(rl_errs)


def report(errs: np.ndarray, name: str):
    if errs.size == 0:
        return
    print(f"  {name:8s}  mean={errs.mean():6.2f}  rmse={np.sqrt((errs**2).mean()):6.2f}  p95={np.percentile(errs,95):6.2f}  max={errs.max():6.2f}")


def main():
    raw, ekf, rl = collect_errors()
    print("\n=== GNSS Correction Errors (m) ===")
    report(raw, "Raw")
    report(ekf, "EKF")
    report(rl, "RL")

    plt.figure(figsize=(8, 5))
    for arr, label in [(raw, "Raw"), (ekf, "EKF"), (rl, "RL")]:
        if arr.size > 0:
            sorted_arr = np.sort(arr)
            cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
            plt.plot(sorted_arr, cdf, label=label, linewidth=2)
    plt.xlabel("Position error (m)")
    plt.ylabel("CDF")
    plt.title("GNSS correction error CDF")
    plt.legend()
    plt.grid(True)
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/cdf.png", dpi=120)
    print("\nSaved results/cdf.png")


if __name__ == "__main__":
    main()
