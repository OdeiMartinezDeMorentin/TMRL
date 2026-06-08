import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tmml.models.networks import QNetwork
from tmml.training.replay_buffer import ReplayBuffer
from tmml.config.loader import DQNConfig


class DQNAgent:
    def __init__(self, obs_size: int, n_actions: int, config: Optional[DQNConfig] = None):
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.config = config or DQNConfig()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(
            obs_size=obs_size,
            n_actions=n_actions,
            hidden_layers=self.config.hidden_layers,
            activation=self.config.activation,
        ).to(self.device)

        self.target_net = QNetwork(
            obs_size=obs_size,
            n_actions=n_actions,
            hidden_layers=self.config.hidden_layers,
            activation=self.config.activation,
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        self.loss_fn = nn.HuberLoss()

        self.replay_buffer = ReplayBuffer(
            capacity=self.config.replay_buffer_size,
            obs_size=obs_size,
        )

        self.epsilon = self.config.epsilon_start
        self._step_count = 0

    def select_action(self, obs: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            q_values = self.policy_net(obs_t)
            return int(q_values.argmax(dim=1).item())

    def store_transition(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        self.replay_buffer.push(obs, action, reward, next_obs, done)

    def train_step(self, batch_size: int) -> Optional[float]:
        if len(self.replay_buffer) < batch_size:
            return None

        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(batch_size)

        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.policy_net(obs_t).gather(1, actions_t)

        with torch.no_grad():
            next_q_max = self.target_net(next_obs_t).max(dim=1, keepdim=True)[0]
            target_q = rewards_t + self.config.gamma * next_q_max * (1 - dones_t)

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        self._step_count += 1
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.config.epsilon_end:
            decay_rate = (self.config.epsilon_start - self.config.epsilon_end) / self.config.epsilon_decay
            self.epsilon = max(self.config.epsilon_end, self.epsilon - decay_rate)

    def save(self, path: str):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self._step_count,
            "config": {
                "obs_size": self.obs_size,
                "n_actions": self.n_actions,
                "hidden_layers": self.config.hidden_layers,
                "activation": self.config.activation,
            },
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self._step_count = checkpoint["step_count"]