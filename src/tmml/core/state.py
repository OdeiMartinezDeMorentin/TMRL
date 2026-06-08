import math
from typing import List, Optional

import gymnasium
import numpy as np

from tmml.core.waypoints import WaypointManager


class StateExtractor:
    SPEED_MAX = 300.0
    RPM_MAX = 8000.0
    ANGULAR_MAX = 10.0
    GEAR_MAX = 5.0
    STEER_MAX = 65536.0
    DIST_MAX = 50.0

    def __init__(self, waypoint_manager: WaypointManager):
        self.wpm = waypoint_manager

    @property
    def observation_size(self) -> int:
        base = 3 + 1 + 3 + 3 + 4 + 1 + 1 + 1 + 1 + 1
        lookahead = len(self.wpm.look_aheads)
        return base + lookahead + 1 + lookahead

    @property
    def observation_space(self) -> gymnasium.spaces.Box:
        size = self.observation_size
        return gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(size,), dtype=np.float32
        )

    def extract(self, car_state, waypoint_idx: Optional[int] = None) -> np.ndarray:
        vel = [v / self.SPEED_MAX for v in car_state.velocity[:3]]
        speed = [car_state.display_speed / self.SPEED_MAX]
        ypr = [v / math.pi for v in car_state.yaw_pitch_roll[:3]]
        angular = [v / self.ANGULAR_MAX for v in car_state.angular_speed[:3]]
        wheels = [float(c) for c in car_state.wheel_ground_contact[:4]]
        sliding = [float(car_state.is_sliding)]
        rpm = [car_state.engine_rpm / self.RPM_MAX]
        gear = [car_state.engine_gear / self.GEAR_MAX]
        steer = [car_state.input_steer / self.STEER_MAX]
        gas = [car_state.input_gas / self.STEER_MAX]

        pos = car_state.position[:3]
        yaw = car_state.yaw_pitch_roll[0]
        idx = waypoint_idx if waypoint_idx is not None else self.wpm._current_idx
        lookahead_angles = self.wpm.get_lookahead_angles(pos, yaw, idx)
        progress = [self.wpm.progress_fraction(idx)]
        lookahead_dists = self.wpm.get_lookahead_distances(pos, idx)

        obs = (
            vel
            + speed
            + ypr
            + angular
            + wheels
            + sliding
            + rpm
            + gear
            + steer
            + gas
            + lookahead_angles
            + progress
            + lookahead_dists
        )

        return np.array(obs, dtype=np.float32)