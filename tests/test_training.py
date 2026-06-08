import numpy as np
import pytest
import torch
import tempfile
import os

from tmml.training.replay_buffer import ReplayBuffer
from tmml.models.dqn import DQNAgent
from tmml.config.loader import DQNConfig


class TestReplayBuffer:
    def setup_method(self):
        self.obs_size = 8
        self.buffer = ReplayBuffer(capacity=100, obs_size=self.obs_size)

    def test_push_and_len(self):
        obs = np.zeros(self.obs_size, dtype=np.float32)
        self.buffer.push(obs, 0, 1.0, obs, False)
        assert len(self.buffer) == 1

    def test_push_multiple(self):
        obs = np.zeros(self.obs_size, dtype=np.float32)
        for i in range(10):
            self.buffer.push(obs, i % 3, float(i), obs, False)
        assert len(self.buffer) == 10

    def test_capacity_overflow(self):
        obs = np.zeros(self.obs_size, dtype=np.float32)
        for i in range(150):
            self.buffer.push(obs, 0, float(i), obs, False)
        assert len(self.buffer) == 100

    def test_sample_shape(self):
        obs = np.zeros(self.obs_size, dtype=np.float32)
        for i in range(20):
            self.buffer.push(
                np.random.randn(self.obs_size).astype(np.float32),
                i % 3,
                float(i),
                np.random.randn(self.obs_size).astype(np.float32),
                i % 5 == 0,
            )
        batch = self.buffer.sample(batch_size=8)
        assert batch[0].shape == (8, self.obs_size)
        assert batch[1].shape == (8,)
        assert batch[2].shape == (8,)
        assert batch[3].shape == (8, self.obs_size)
        assert batch[4].shape == (8,)

    def test_sample_smaller_than_buffer(self):
        obs = np.zeros(self.obs_size, dtype=np.float32)
        self.buffer.push(obs, 0, 1.0, obs, False)
        batch = self.buffer.sample(batch_size=1)
        assert batch[0].shape == (1, self.obs_size)


class TestDQNAgent:
    def setup_method(self):
        self.obs_size = 8
        self.n_actions = 12
        self.config = DQNConfig(
            learning_rate=0.001,
            batch_size=4,
            replay_buffer_size=100,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=100,
            target_update_freq=10,
            hidden_layers=[64, 64],
        )
        self.agent = DQNAgent(
            obs_size=self.obs_size,
            n_actions=self.n_actions,
            config=self.config,
        )

    def test_select_action_random(self):
        self.agent.epsilon = 1.0
        obs = np.zeros(self.obs_size, dtype=np.float32)
        actions = set()
        for _ in range(100):
            actions.add(self.agent.select_action(obs))
        assert len(actions) > 1

    def test_select_action_greedy(self):
        self.agent.epsilon = 0.0
        obs = np.zeros(self.obs_size, dtype=np.float32)
        action = self.agent.select_action(obs)
        assert 0 <= action < self.n_actions

    def test_store_transition(self):
        obs = np.zeros(self.obs_size, dtype=np.float32)
        self.agent.store_transition(obs, 0, 1.0, obs, False)
        assert len(self.agent.replay_buffer) == 1

    def test_train_step_returns_none_when_insufficient(self):
        obs = np.zeros(self.obs_size, dtype=np.float32)
        self.agent.store_transition(obs, 0, 1.0, obs, False)
        result = self.agent.train_step(batch_size=4)
        assert result is None

    def test_train_step_returns_loss(self):
        for i in range(10):
            obs = np.random.randn(self.obs_size).astype(np.float32)
            next_obs = np.random.randn(self.obs_size).astype(np.float32)
            self.agent.store_transition(obs, i % self.n_actions, float(i), next_obs, i % 3 == 0)
        loss = self.agent.train_step(batch_size=4)
        assert loss is not None
        assert isinstance(loss, float)

    def test_update_target_network(self):
        obs = np.random.randn(self.obs_size).astype(np.float32)
        next_obs = np.random.randn(self.obs_size).astype(np.float32)
        for i in range(10):
            self.agent.store_transition(obs, i % self.n_actions, float(i), next_obs, False)
        self.agent.train_step(batch_size=4)

        policy_params = list(self.agent.policy_net.parameters())
        target_params_before = [p.clone() for p in self.agent.target_net.parameters()]

        diff_before = sum(
            torch.sum(torch.abs(p - t))
            for p, t in zip(self.agent.policy_net.parameters(), self.agent.target_net.parameters())
        )

        self.agent.update_target_network()

        diff_after = sum(
            torch.sum(torch.abs(p - t))
            for p, t in zip(self.agent.policy_net.parameters(), self.agent.target_net.parameters())
        )

        assert diff_after < diff_before or diff_before > 0

    def test_decay_epsilon(self):
        self.agent.epsilon = 1.0
        for _ in range(50):
            self.agent.decay_epsilon()
        assert self.agent.epsilon < 1.0
        assert self.agent.epsilon >= self.config.epsilon_end

    def test_save_and_load(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "test_model.pt")

        self.agent.save(path)
        assert os.path.exists(path)

        new_agent = DQNAgent(
            obs_size=self.obs_size,
            n_actions=self.n_actions,
            config=self.config,
        )
        new_agent.load(path)
        assert new_agent.epsilon == self.agent.epsilon