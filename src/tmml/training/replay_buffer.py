import numpy as np
from typing import Tuple
import random


class ReplayBuffer:
    def __init__(self, capacity: int, obs_size: int):
        self.capacity = capacity
        self.obs_size = obs_size
        self.pos = 0
        self.size = 0
        self._obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)

    def push(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        self._obs[self.pos] = obs
        self._actions[self.pos] = action
        self._rewards[self.pos] = reward
        self._next_obs[self.pos] = next_obs
        self._dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = random.sample(range(self.size), min(batch_size, self.size))
        return (
            self._obs[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_obs[indices],
            self._dones[indices],
        )

    def __len__(self) -> int:
        return self.size