import os
import csv
from datetime import datetime
from typing import Optional


class TrainingLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_log_path = os.path.join(log_dir, f"episodes_{timestamp}.csv")
        self.eval_log_path = os.path.join(log_dir, f"eval_{timestamp}.csv")

        self._episode_fields = [
            "episode", "reward", "steps", "epsilon",
            "term_reason", "waypoint_idx", "speed_avg", "progress_pct",
        ]
        self._eval_fields = [
            "episode", "avg_reward", "avg_steps", "completion_rate",
        ]

        self._init_csv(self.episode_log_path, self._episode_fields)
        self._init_csv(self.eval_log_path, self._eval_fields)

        self._recent_rewards = []
        self._recent_window = 100

    def _init_csv(self, path: str, fields: list):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

    def log_episode(
        self,
        episode: int,
        reward: float,
        steps: int,
        epsilon: float,
        term_reason: str,
        waypoint_idx: int,
        speed_avg: float,
        progress_pct: float,
    ):
        row = {
            "episode": episode,
            "reward": f"{reward:.4f}",
            "steps": steps,
            "epsilon": f"{epsilon:.4f}",
            "term_reason": term_reason,
            "waypoint_idx": waypoint_idx,
            "speed_avg": f"{speed_avg:.2f}",
            "progress_pct": f"{progress_pct:.4f}",
        }

        with open(self.episode_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._episode_fields)
            writer.writerow(row)

        self._recent_rewards.append(reward)
        if len(self._recent_rewards) > self._recent_window:
            self._recent_rewards.pop(0)

        print(
            f"Episode {episode:5d} | "
            f"Reward: {reward:8.2f} | "
            f"Steps: {steps:4d} | "
            f"Eps: {epsilon:.4f} | "
            f"Reason: {term_reason:10s} | "
            f"WP: {waypoint_idx:4d} | "
            f"Progress: {progress_pct:.2%}"
        )

    def log_eval(self, episode: int, avg_reward: float, avg_steps: float, completion_rate: float):
        row = {
            "episode": episode,
            "avg_reward": f"{avg_reward:.4f}",
            "avg_steps": f"{avg_steps:.1f}",
            "completion_rate": f"{completion_rate:.4f}",
        }

        with open(self.eval_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._eval_fields)
            writer.writerow(row)

        print(
            f"\n=== EVAL @ Episode {episode} === "
            f"Avg Reward: {avg_reward:.2f} | "
            f"Avg Steps: {avg_steps:.1f} | "
            f"Completion: {completion_rate:.2%}\n"
        )

    def get_summary(self) -> dict:
        if not self._recent_rewards:
            return {"avg_reward": 0.0, "num_episodes": 0}
        return {
            "avg_reward": sum(self._recent_rewards) / len(self._recent_rewards),
            "num_episodes": len(self._recent_rewards),
        }

    def close(self):
        pass