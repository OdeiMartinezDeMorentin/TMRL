from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class RewardConfig:
    progress_weight: float = 1.0
    speed_weight: float = 0.1
    step_penalty: float = 0.001
    completion_bonus: float = 100.0
    no_progress_timeout: int = 200


class RewardCalculator:
    def __init__(self, config: RewardConfig, waypoint_manager):
        self.config = config
        self.wpm = waypoint_manager
        self._last_waypoint_idx = 0
        self._no_progress_steps = 0

    def compute(self, car_state, prev_idx: int) -> Tuple[float, dict]:
        position = car_state.position[:3]
        current_idx = self.wpm.find_nearest(position, start_idx=prev_idx)
        current_idx = max(current_idx, prev_idx)
        progress = current_idx - prev_idx
        speed_reward = car_state.display_speed / 300.0 * self.config.speed_weight

        reward = (
            progress * self.config.progress_weight
            + speed_reward
            - self.config.step_penalty
        )

        info = {
            "waypoint_idx": current_idx,
            "progress": progress,
            "speed": car_state.display_speed,
        }

        return reward, info, current_idx

    def should_terminate(self, car_state, current_idx: int, total_steps: int, max_steps: int) -> Tuple[bool, Optional[str]]:
        if not any(car_state.wheel_ground_contact):
            return True, "fallen"

        if car_state.race_finished:
            return True, "finished"

        if current_idx == self._last_waypoint_idx:
            self._no_progress_steps += 1
        else:
            self._no_progress_steps = 0

        self._last_waypoint_idx = current_idx

        if self._no_progress_steps >= self.config.no_progress_timeout:
            return True, "no_progress"

        if total_steps >= max_steps:
            return True, "timeout"

        return False, None

    def reset(self):
        self._last_waypoint_idx = 0
        self._no_progress_steps = 0