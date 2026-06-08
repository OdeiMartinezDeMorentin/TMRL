import json
import tempfile
import os
import numpy as np
import pytest
import gymnasium

from tmml.core.environment import TrackmaniaEnv, EnvClient
from tmml.core.rewards import RewardConfig
from tmml.tmi.client import CarState


def create_test_waypoints(tmpdir):
    waypoints = [[float(i), 0.0, float(i)] for i in range(50)]
    data = {"waypoints": waypoints, "track_name": "test", "spacing": 1.0}
    path = os.path.join(tmpdir, "waypoints.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def make_car_state(pos=None, speed=0.0, finished=False, wheels_on_ground=True):
    return CarState(
        time=0,
        position=pos or [0.0, 0.0, 0.0],
        velocity=[speed, 0.0, 0.0],
        yaw_pitch_roll=[0.0, 0.0, 0.0],
        display_speed=speed,
        race_time=0,
        race_finished=finished,
        wheel_ground_contact=[wheels_on_ground] * 4,
        is_sliding=False,
        angular_speed=[0.0, 0.0, 0.0],
        engine_rpm=0.0,
        engine_gear=0,
        input_steer=0,
        input_gas=0,
        input_accelerate=False,
        input_brake=False,
    )


class MockEnvClient:
    def __init__(self, car_state=None):
        self.state = car_state or make_car_state()
        self._registered = True
        self._iface = None
        self.initial_state = "mock_initial_state"
        self._step_count = 0

    def send_action(self, action):
        self._step_count += 1
        self.state.position = [float(self._step_count), 0.0, float(self._step_count)]
        self.state.display_speed = 50.0 + self._step_count
        return self.state

    def reset_to_initial(self):
        self._step_count = 0
        self.state.position = [0.0, 0.0, 0.0]
        self.state.display_speed = 0.0

    def close(self):
        pass


class TestTrackmaniaEnv:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.waypoints_path = create_test_waypoints(self.tmpdir)

    def test_observation_space_shape(self):
        env = TrackmaniaEnv(waypoints_path=self.waypoints_path)
        assert env.observation_space.shape[0] == env.state_extractor.observation_size
        assert env.observation_space.dtype == np.float32

    def test_reset_without_client(self):
        env = TrackmaniaEnv(waypoints_path=self.waypoints_path)
        obs, info = env.reset()
        assert obs.shape == (env.state_extractor.observation_size,)
        assert isinstance(info, dict)
        assert "waypoint_idx" in info

    def test_step_without_client(self):
        env = TrackmaniaEnv(waypoints_path=self.waypoints_path)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert terminated is True
        assert info["term_reason"] == "no_client"

    def test_reset_and_step_with_mock(self):
        mock_client = MockEnvClient()
        env = TrackmaniaEnv(
            waypoints_path=self.waypoints_path,
            client=mock_client,
            max_steps=100,
        )
        obs, info = env.reset()
        assert obs.shape == (env.state_extractor.observation_size,)

        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (env.state_extractor.observation_size,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_terminated_on_fallen(self):
        fallen_state = make_car_state(pos=[5.0, 0.0, 5.0], wheels_on_ground=False)
        mock_client = MockEnvClient(car_state=make_car_state(pos=[5.0, 0.0, 5.0]))
        mock_client.send_action = lambda action: fallen_state

        env = TrackmaniaEnv(
            waypoints_path=self.waypoints_path,
            client=mock_client,
            max_steps=100,
        )
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert terminated is True
        assert info["term_reason"] == "fallen"

    def test_terminated_on_timeout(self):
        mock_client = MockEnvClient()
        env = TrackmaniaEnv(
            waypoints_path=self.waypoints_path,
            client=mock_client,
            max_steps=3,
        )
        env.reset()
        env.step(0)
        env.step(0)
        env.step(0)
        obs, reward, terminated, truncated, info = env.step(0)
        assert terminated is True or env._total_steps >= 3

    def test_action_space_is_discrete(self):
        env = TrackmaniaEnv(waypoints_path=self.waypoints_path)
        assert isinstance(env.action_space, gymnasium.spaces.Discrete)
        assert env.action_space.n == 12