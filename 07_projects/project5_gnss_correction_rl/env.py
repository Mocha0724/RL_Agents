"""GNSSCorrectionEnv：连续动作 RL 环境，用于学习 GNSS 修正策略。

每步：
    obs   = (raw_gnss xy, speed, dop, nsat, 历史 5 帧 gnss xy)
    action = (Δx, Δy)  ∈ [-15, 15] m
    reward = -‖raw + Δ - truth‖₂  (+ 可选 shaping)
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulate import sample_trajectory, TrajectorySample


HISTORY_LEN = 5


class GNSSCorrectionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 300, seed: int | None = None, shaping: bool = True):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.shaping = shaping
        self._seed_base = seed if seed is not None else 0
        self._episode_idx = 0

        feat_dim = 2 + 1 + 1 + 1 + 2 * HISTORY_LEN
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(feat_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-15.0, high=15.0, shape=(2,), dtype=np.float32)

        self._sample: TrajectorySample | None = None
        self._t = 0
        self._history: list[np.ndarray] = []
        self._last_err: float | None = None

    def _new_sample(self):
        self._sample = sample_trajectory(seed=self._seed_base + self._episode_idx, total_steps=self.max_episode_steps)
        self._episode_idx += 1

    def _get_obs(self) -> np.ndarray:
        s = self._sample
        gnss = s.gnss[self._t]
        feats = [gnss[0], gnss[1], s.speed[self._t], s.dop[self._t], float(s.nsat[self._t])]
        for i in range(HISTORY_LEN):
            idx = max(0, self._t - 1 - i)
            feats.extend(s.gnss[idx].tolist())
        return np.asarray(feats, dtype=np.float32)

    def reset(self, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed_base = seed
            self._episode_idx = 0
        self._new_sample()
        self._t = 0
        self._history.clear()
        self._last_err = None
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -15.0, 15.0)
        s = self._sample
        gnss_t = s.gnss[self._t]
        truth_t = s.truth[self._t]
        corrected = gnss_t + action
        err = float(np.linalg.norm(corrected - truth_t))

        reward = -err
        if self.shaping and self._last_err is not None:
            reward += 0.5 * (self._last_err - err)
        self._last_err = err

        self._t += 1
        terminated = self._t >= self.max_episode_steps
        info = {"err": err, "raw_err": float(np.linalg.norm(gnss_t - truth_t))}
        obs = self._get_obs() if not terminated else np.zeros_like(self.observation_space.sample())
        return obs, reward, terminated, False, info


if __name__ == "__main__":
    env = GNSSCorrectionEnv(seed=0)
    obs, _ = env.reset()
    total = 0.0
    while True:
        a = np.zeros(2, dtype=np.float32)
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term:
            break
    print(f"Zero-correction (raw GNSS) total reward = {total:.2f}")
