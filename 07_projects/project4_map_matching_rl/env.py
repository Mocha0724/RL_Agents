"""MapMatchingEnv — 一个合成路网上的地图匹配 RL 环境。

为 Project 4 提供起手脚手架。学员需要补全 `step` 中的 reward 设计、
`_get_observation` 中的特征工程，以及训练 DQN 的 `train.py`。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class RoadSegment:
    seg_id: int
    start: tuple[float, float]
    end: tuple[float, float]

    @property
    def length(self) -> float:
        dx, dy = self.end[0] - self.start[0], self.end[1] - self.start[1]
        return math.hypot(dx, dy)

    def project(self, point: tuple[float, float]) -> tuple[float, tuple[float, float]]:
        """把点投影到线段，返回（距离，投影点）。"""
        x0, y0 = self.start
        x1, y1 = self.end
        px, py = point
        dx, dy = x1 - x0, y1 - y0
        if dx == 0 and dy == 0:
            return math.hypot(px - x0, py - y0), self.start
        t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy)))
        proj = (x0 + t * dx, y0 + t * dy)
        return math.hypot(px - proj[0], py - proj[1]), proj


def make_demo_network() -> list[RoadSegment]:
    """一个简单的 H 型路网，方便观察分歧匹配."""
    return [
        RoadSegment(0, (0.0, 0.0), (10.0, 0.0)),    
        RoadSegment(1, (10.0, 0.0), (20.0, 0.0)),   
        RoadSegment(2, (10.0, 0.0), (10.0, 10.0)),  
        RoadSegment(3, (10.0, 10.0), (0.0, 10.0)),  
        RoadSegment(4, (10.0, 10.0), (20.0, 10.0)), 
    ]


class MapMatchingEnv(gym.Env):
    """A toy online map matching environment.

    每个 episode：
        1. 随机选一条 ground-truth 路径序列（按路段连接）
        2. 沿路径采样 GPS 点 + 高斯噪声
        3. agent 在每个 GPS 点选一个候选路段
        4. reward = 是否选对 + 是否与上一次选的连通
    """

    K_CANDIDATES = 5
    metadata = {"render_modes": []}

    def __init__(self, gps_noise_std: float = 0.5, max_steps: int = 30, seed: int | None = None):
        super().__init__()
        self.network = make_demo_network()
        self.gps_noise_std = gps_noise_std
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        feat_per_cand = 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2 + 2 + feat_per_cand * self.K_CANDIDATES,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.K_CANDIDATES)

        self._truth_path: list[int] = []
        self._gps_points: list[tuple[float, float]] = []
        self._t = 0
        self._last_action_seg: int | None = None

    def _sample_truth_path(self) -> list[int]:
        candidates = [
            [0, 1],            
            [0, 2, 4],         
            [0, 2, 3],         
            [3, 2, 4],         
        ]
        return list(self.rng.choice(candidates, size=1)[0])

    def _generate_gps_along_path(self, path_segs: list[int]) -> list[tuple[float, float]]:
        pts = []
        for seg_id in path_segs:
            seg = self.network[seg_id]
            for t in np.linspace(0, 1, 6, endpoint=False):
                x = seg.start[0] + t * (seg.end[0] - seg.start[0])
                y = seg.start[1] + t * (seg.end[1] - seg.start[1])
                x += self.rng.normal(0, self.gps_noise_std)
                y += self.rng.normal(0, self.gps_noise_std)
                pts.append((x, y))
        return pts[: self.max_steps]

    def _truth_seg_at(self, t: int) -> int:
        idx = min(t // 6, len(self._truth_path) - 1)
        return self._truth_path[idx]

    def _candidates(self, point: tuple[float, float]) -> list[int]:
        scored = sorted(self.network, key=lambda s: s.project(point)[0])
        return [s.seg_id for s in scored[: self.K_CANDIDATES]]

    def _get_observation(self) -> np.ndarray:
        if self._t >= len(self._gps_points):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        cur = self._gps_points[self._t]
        prev = self._gps_points[self._t - 1] if self._t > 0 else cur
        feats = [cur[0], cur[1], cur[0] - prev[0], cur[1] - prev[1]]
        for seg_id in self._candidates(cur):
            seg = self.network[seg_id]
            dist, proj = seg.project(cur)
            heading = math.atan2(seg.end[1] - seg.start[1], seg.end[0] - seg.start[0])
            feats.extend([dist, proj[0] - cur[0], proj[1] - cur[1], heading])
        while len(feats) < self.observation_space.shape[0]:
            feats.append(0.0)
        return np.array(feats[: self.observation_space.shape[0]], dtype=np.float32)

    def reset(self, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._truth_path = self._sample_truth_path()
        self._gps_points = self._generate_gps_along_path(self._truth_path)
        self._t = 0
        self._last_action_seg = None
        return self._get_observation(), {}

    def step(self, action: int):
        cur = self._gps_points[self._t]
        cand_ids = self._candidates(cur)
        chosen_seg = cand_ids[int(action)]
        truth_seg = self._truth_seg_at(self._t)

        reward = 1.0 if chosen_seg == truth_seg else -1.0
        if self._last_action_seg is not None and chosen_seg != self._last_action_seg:
            connected = self._segments_connected(self._last_action_seg, chosen_seg)
            if not connected:
                reward -= 0.5

        self._last_action_seg = chosen_seg
        self._t += 1
        done = self._t >= len(self._gps_points)
        return self._get_observation(), reward, done, False, {"truth_seg": truth_seg, "chosen_seg": chosen_seg}

    def _segments_connected(self, a: int, b: int) -> bool:
        sa, sb = self.network[a], self.network[b]
        for p in (sa.start, sa.end):
            for q in (sb.start, sb.end):
                if math.isclose(p[0], q[0], abs_tol=1e-6) and math.isclose(p[1], q[1], abs_tol=1e-6):
                    return True
        return False


if __name__ == "__main__":
    env = MapMatchingEnv(seed=0)
    obs, _ = env.reset()
    total_reward = 0.0
    while True:
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        total_reward += r
        if term or trunc:
            break
    print(f"random policy total reward = {total_reward:.2f}")
