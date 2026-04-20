"""用 stable-baselines3 PPO 训练 GNSS 修正策略。

用法:
    python train_ppo.py --total-steps 200000

训练完成后会保存 results/ppo_gnss.zip 和 tensorboard 日志。
"""
from __future__ import annotations

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env import GNSSCorrectionEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    def make_env():
        return GNSSCorrectionEnv(seed=args.seed)

    env = make_vec_env(make_env, n_envs=args.n_envs)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="results/tb",
        seed=args.seed,
    )
    model.learn(total_timesteps=args.total_steps)
    model.save("results/ppo_gnss")
    print("Saved results/ppo_gnss.zip")


if __name__ == "__main__":
    main()
