import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tmml.core.rewards import RewardCalculator, RewardConfig
from tmml.core.waypoints import WaypointManager
from tmml.tmi.client import CarState
import numpy as np


def create_test_waypoints(tmpdir):
    waypoints = np.array([[float(i * 2.5), 0.0, float(i * 2.5)] for i in range(100)])
    path = os.path.join(tmpdir, "waypoints.json")
    WaypointManager.save_waypoints(path, waypoints, track_name="test")
    return path


def make_car_state(pos=None, speed=0, finished=False, wheels_on_ground=True):
    state = CarState()
    if pos is not None:
        state.position = pos
    state.display_speed = speed
    state.race_finished = finished
    if wheels_on_ground:
        state.wheel_ground_contact = [True, True, True, True]
    else:
        state.wheel_ground_contact = [False, False, False, False]
    return state


def test_progress_reward():
    tmpdir = tempfile.mkdtemp()
    path = create_test_waypoints(tmpdir)
    wpm = WaypointManager(path)
    config = RewardConfig()
    calc = RewardCalculator(config, wpm)

    state = make_car_state(pos=[5.0, 0.0, 5.0], speed=100)
    reward, info, idx = calc.compute(state, prev_idx=0)
    assert reward > 0
    assert info["progress"] >= 0


def test_step_penalty():
    tmpdir = tempfile.mkdtemp()
    path = create_test_waypoints(tmpdir)
    wpm = WaypointManager(path)
    config = RewardConfig(step_penalty=0.01)
    calc = RewardCalculator(config, wpm)

    state = make_car_state(pos=[0.0, 0.0, 0.0], speed=0)
    reward, info, idx = calc.compute(state, prev_idx=0)
    assert reward < 0


def test_speed_bonus():
    tmpdir = tempfile.mkdtemp()
    path = create_test_waypoints(tmpdir)
    wpm = WaypointManager(path)
    config = RewardConfig(speed_weight=0.1)

    calc_fast = RewardCalculator(config, wpm)
    state_fast = make_car_state(pos=[2.5, 0.0, 2.5], speed=200)
    reward_fast, _, _ = calc_fast.compute(state_fast, prev_idx=0)

    calc_slow = RewardCalculator(config, wpm)
    state_slow = make_car_state(pos=[2.5, 0.0, 2.5], speed=50)
    reward_slow, _, _ = calc_slow.compute(state_slow, prev_idx=0)

    assert reward_fast > reward_slow


def test_should_terminate_fallen():
    tmpdir = tempfile.mkdtemp()
    path = create_test_waypoints(tmpdir)
    wpm = WaypointManager(path)
    config = RewardConfig()
    calc = RewardCalculator(config, wpm)
    calc.reset()

    state = make_car_state(wheels_on_ground=False)
    terminated, reason = calc.should_terminate(state, current_idx=5, total_steps=100, max_steps=3000)
    assert terminated == True
    assert reason == "fallen"


def test_should_terminate_finished():
    tmpdir = tempfile.mkdtemp()
    path = create_test_waypoints(tmpdir)
    wpm = WaypointManager(path)
    config = RewardConfig()
    calc = RewardCalculator(config, wpm)
    calc.reset()

    state = make_car_state(finished=True)
    terminated, reason = calc.should_terminate(state, current_idx=50, total_steps=100, max_steps=3000)
    assert terminated == True
    assert reason == "finished"


def test_should_terminate_no_progress():
    tmpdir = tempfile.mkdtemp()
    path = create_test_waypoints(tmpdir)
    wpm = WaypointManager(path)
    config = RewardConfig(no_progress_timeout=5)
    calc = RewardCalculator(config, wpm)
    calc.reset()

    state = make_car_state(pos=[0.0, 0.0, 0.0], speed=0)
    for i in range(5):
        terminated, reason = calc.should_terminate(state, current_idx=0, total_steps=i, max_steps=3000)
        if i < 4:
            assert not terminated
        else:
            assert terminated
            assert reason == "no_progress"


def test_should_terminate_timeout():
    tmpdir = tempfile.mkdtemp()
    path = create_test_waypoints(tmpdir)
    wpm = WaypointManager(path)
    config = RewardConfig()
    calc = RewardCalculator(config, wpm)
    calc.reset()

    state = make_car_state(pos=[0.0, 0.0, 0.0], speed=10)
    calc._last_waypoint_idx = 0
    terminated, reason = calc.should_terminate(state, current_idx=1, total_steps=3000, max_steps=3000)
    assert terminated == True
    assert reason == "timeout"


def test_reset():
    tmpdir = tempfile.mkdtemp()
    path = create_test_waypoints(tmpdir)
    wpm = WaypointManager(path)
    config = RewardConfig()
    calc = RewardCalculator(config, wpm)

    calc._last_waypoint_idx = 50
    calc._no_progress_steps = 100
    calc.reset()
    assert calc._last_waypoint_idx == 0
    assert calc._no_progress_steps == 0


if __name__ == "__main__":
    test_progress_reward()
    test_step_penalty()
    test_speed_bonus()
    test_should_terminate_fallen()
    test_should_terminate_finished()
    test_should_terminate_no_progress()
    test_should_terminate_timeout()
    test_reset()
    print("All reward tests passed!")