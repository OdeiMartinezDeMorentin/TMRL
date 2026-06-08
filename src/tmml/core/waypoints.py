import json
import math
from typing import List, Optional, Tuple

import numpy as np


class WaypointManager:
    def __init__(self, waypoints_path: str, look_aheads: Optional[List[int]] = None):
        self.look_aheads = look_aheads or [1, 3, 7]
        self.waypoints = self.load_waypoints(waypoints_path)
        self._current_idx = 0

    @staticmethod
    def load_waypoints(path: str) -> np.ndarray:
        with open(path, "r") as f:
            data = json.load(f)
        wps = np.array(data["waypoints"], dtype=np.float64)
        return wps

    @staticmethod
    def save_waypoints(path: str, waypoints: np.ndarray, track_name: str = "", spacing: float = 2.5):
        data = {
            "track_name": track_name,
            "waypoints": waypoints.tolist(),
            "spacing": spacing,
            "source": "demo_smoothed",
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def find_nearest(self, position: List[float], start_idx: Optional[int] = None) -> int:
        pos = np.array(position[:3])
        idx = start_idx if start_idx is not None else self._current_idx
        search_start = max(0, idx - 5)
        search_end = min(len(self.waypoints), idx + 50)
        if search_start >= search_end:
            search_end = len(self.waypoints)
            search_start = max(0, search_end - 50)

        subset = self.waypoints[search_start:search_end]
        diffs = subset - pos
        dists = np.sum(diffs ** 2, axis=1)
        nearest_in_subset = int(np.argmin(dists))
        return search_start + nearest_in_subset

    def get_progress(self, position: List[float]) -> Tuple[int, float]:
        new_idx = self.find_nearest(position)
        progress = max(new_idx - self._current_idx, 0)
        self._current_idx = max(new_idx, self._current_idx)

        pos = np.array(position[:3])
        if new_idx < len(self.waypoints) - 1:
            next_wp = self.waypoints[new_idx + 1]
            dist_to_next = float(np.linalg.norm(next_wp - pos))
        else:
            dist_to_next = 0.0

        return progress, dist_to_next

    def get_lookahead_angles(self, position: List[float], yaw: float, current_idx: Optional[int] = None) -> List[float]:
        idx = current_idx if current_idx is not None else self._current_idx
        pos = np.array(position[:3])
        angles = []

        for la in self.look_aheads:
            target_idx = min(idx + la, len(self.waypoints) - 1)
            target = self.waypoints[target_idx]
            dx = target[0] - pos[0]
            dz = target[2] - pos[2]
            angle_to_target = math.atan2(dx, dz)
            relative_angle = angle_to_target - yaw
            relative_angle = math.atan2(math.sin(relative_angle), math.cos(relative_angle))
            normalized = relative_angle / math.pi
            angles.append(normalized)

        return angles

    def get_lookahead_distances(self, position: List[float], current_idx: Optional[int] = None, max_dist: float = 50.0) -> List[float]:
        idx = current_idx if current_idx is not None else self._current_idx
        pos = np.array(position[:3])
        distances = []

        for la in self.look_aheads:
            target_idx = min(idx + la, len(self.waypoints) - 1)
            target = self.waypoints[target_idx]
            dist = float(np.linalg.norm(target - pos))
            distances.append(min(dist / max_dist, 1.0))

        return distances

    @property
    def total_count(self) -> int:
        return len(self.waypoints)

    def progress_fraction(self, current_idx: Optional[int] = None) -> float:
        idx = current_idx if current_idx is not None else self._current_idx
        return idx / max(len(self.waypoints) - 1, 1)

    def reset(self):
        self._current_idx = 0

    @staticmethod
    def smooth_waypoints(raw_positions: np.ndarray, window: int = 50, iterations: int = 3) -> np.ndarray:
        smoothed = raw_positions.copy()
        for _ in range(iterations):
            result = smoothed.copy()
            for i in range(len(result)):
                start = max(0, i - window // 2)
                end = min(len(result), i + window // 2 + 1)
                result[i] = smoothed[start:end].mean(axis=0)
            smoothed = result
        return smoothed

    @staticmethod
    def respace_waypoints(smoothed: np.ndarray, spacing: float = 2.5) -> np.ndarray:
        diffs = np.diff(smoothed, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
        cumulative = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative[-1]
        num_points = max(int(total_length / spacing), 2)
        target_distances = np.linspace(0, total_length, num_points)

        result = np.zeros((num_points, smoothed.shape[1]))
        for i, td in enumerate(target_distances):
            idx = np.searchsorted(cumulative, td, side="right") - 1
            idx = max(0, min(idx, len(smoothed) - 2))
            seg_len = segment_lengths[idx]
            if seg_len > 0:
                t = (td - cumulative[idx]) / seg_len
            else:
                t = 0.0
            result[i] = smoothed[idx] * (1 - t) + smoothed[idx + 1] * t

        return result