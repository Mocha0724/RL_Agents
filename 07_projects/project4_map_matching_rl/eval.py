"""评估脚本：DQN vs 最近邻 vs HMM 三种方法的匹配准确率。

用法:
    python eval.py
"""
from __future__ import annotations

import numpy as np
import torch

from baselines import evaluate, hmm_viterbi_match, nearest_neighbor_match
from env import MapMatchingEnv
from train import QNet


def dqn_predictor(model: QNet, device):
    def _pred(env: MapMatchingEnv):
        preds = []
        env._t = 0
        env._last_action_seg = None
        for t in range(len(env._gps_points)):
            obs = env._get_observation()
            with torch.no_grad():
                a = int(model(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).argmax(-1).item())
            cand_ids = env._candidates(env._gps_points[t])
            preds.append(cand_ids[a])
            env._t += 1
            env._last_action_seg = cand_ids[a]
        return preds
    return _pred


def main():
    env = MapMatchingEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nn_acc = evaluate(env, nearest_neighbor_match, num_episodes=200)
    hmm_acc = evaluate(env, hmm_viterbi_match, num_episodes=200)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    model = QNet(obs_dim, n_actions).to(device)
    model.load_state_dict(torch.load("results/dqn_mm.pt", map_location=device))
    model.eval()
    dqn_acc = evaluate(env, dqn_predictor(model, device), num_episodes=200)

    print("\n=== Map Matching Accuracy ===")
    print(f"  Nearest Neighbor: {nn_acc:.3f}")
    print(f"  HMM Viterbi:      {hmm_acc:.3f}")
    print(f"  DQN (yours):      {dqn_acc:.3f}")


if __name__ == "__main__":
    main()
