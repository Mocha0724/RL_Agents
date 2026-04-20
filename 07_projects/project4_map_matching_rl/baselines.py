"""传统地图匹配 baselines：最近邻 + HMM-Viterbi。

把它们和你训练的 DQN 放在同一接口下评估。
"""
from __future__ import annotations

import math
import numpy as np

from env import MapMatchingEnv, RoadSegment


def nearest_neighbor_match(env: MapMatchingEnv) -> list[int]:
    """每个 GPS 点选距离最近的路段。"""
    matches = []
    for pt in env._gps_points:
        scored = sorted(env.network, key=lambda s: s.project(pt)[0])
        matches.append(scored[0].seg_id)
    return matches


def hmm_viterbi_match(env: MapMatchingEnv, sigma: float = 1.0) -> list[int]:
    """简化版 HMM 地图匹配。
    发射概率 ∝ exp(-d²/(2σ²))；转移概率：连通=1.0，不连通=0.1。"""
    pts = env._gps_points
    segs = env.network
    N = len(pts)
    M = len(segs)

    log_emit = np.zeros((N, M))
    for i, pt in enumerate(pts):
        for j, s in enumerate(segs):
            d, _ = s.project(pt)
            log_emit[i, j] = -0.5 * (d / sigma) ** 2

    log_trans = np.full((M, M), math.log(0.1))
    for i, sa in enumerate(segs):
        for j, sb in enumerate(segs):
            for p in (sa.start, sa.end):
                for q in (sb.start, sb.end):
                    if math.isclose(p[0], q[0], abs_tol=1e-6) and math.isclose(p[1], q[1], abs_tol=1e-6):
                        log_trans[i, j] = math.log(1.0)
                        break

    dp = np.full((N, M), -np.inf)
    bp = np.zeros((N, M), dtype=int)
    dp[0] = log_emit[0]
    for t in range(1, N):
        for j in range(M):
            scores = dp[t - 1] + log_trans[:, j]
            bp[t, j] = int(np.argmax(scores))
            dp[t, j] = log_emit[t, j] + scores[bp[t, j]]

    matches = [int(np.argmax(dp[-1]))]
    for t in range(N - 1, 0, -1):
        matches.append(int(bp[t, matches[-1]]))
    return list(reversed(matches))


def evaluate(env: MapMatchingEnv, predictor, num_episodes: int = 100) -> float:
    correct = 0
    total = 0
    for ep in range(num_episodes):
        env.reset(seed=ep)
        preds = predictor(env)
        for t, pred in enumerate(preds):
            truth = env._truth_seg_at(t)
            correct += int(pred == truth)
            total += 1
    return correct / total


if __name__ == "__main__":
    env = MapMatchingEnv()
    print(f"Nearest Neighbor accuracy: {evaluate(env, nearest_neighbor_match):.3f}")
    print(f"HMM Viterbi accuracy:     {evaluate(env, hmm_viterbi_match):.3f}")
