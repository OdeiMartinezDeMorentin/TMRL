import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tmml.core.actions import (
    ACTIONS, NUM_ACTIONS, STEER_VALUES,
    Action, action_to_tminterface, tminterface_to_action,
    get_action, get_action_index,
)


def test_num_actions():
    assert NUM_ACTIONS == 12, f"Expected 12 actions, got {NUM_ACTIONS}"


def test_steer_values():
    assert STEER_VALUES == [-1, 0, 1]


def test_action_composition():
    accel_count = sum(1 for a in ACTIONS if a.accelerate)
    brake_count = sum(1 for a in ACTIONS if a.brake)
    assert accel_count == 6
    assert brake_count == 6


def test_action_to_tminterface():
    action = Action(accelerate=True, brake=False, steer=-1)
    inputs = action_to_tminterface(action)
    assert inputs["accelerate"] == True
    assert inputs["brake"] == False
    assert inputs["steer"] == -65536

    action2 = Action(accelerate=False, brake=True, steer=0)
    inputs2 = action_to_tminterface(action2)
    assert inputs2["steer"] == 0

    action3 = Action(accelerate=True, brake=True, steer=1)
    inputs3 = action_to_tminterface(action3)
    assert inputs3["steer"] == 65536


def test_tminterface_to_action():
    action = tminterface_to_action(True, False, -65536)
    assert action.accelerate == True
    assert action.brake == False
    assert action.steer == -1

    action2 = tminterface_to_action(False, True, 0)
    assert action2.steer == 0

    action3 = tminterface_to_action(True, True, 65536)
    assert action3.steer == 1


def test_round_trip():
    for action in ACTIONS:
        inputs = action_to_tminterface(action)
        restored = tminterface_to_action(inputs["accelerate"], inputs["brake"], inputs["steer"])
        assert restored.accelerate == action.accelerate
        assert restored.brake == action.brake
        assert restored.steer == action.steer


def test_get_action():
    for i in range(NUM_ACTIONS):
        action = get_action(i)
        assert action == ACTIONS[i]


def test_get_action_index():
    for i, action in enumerate(ACTIONS):
        assert get_action_index(action) == i


def test_neutral_action():
    neutral = Action(accelerate=False, brake=False, steer=0)
    idx = get_action_index(neutral)
    assert idx >= 0
    assert ACTIONS[idx] == neutral


if __name__ == "__main__":
    test_num_actions()
    test_steer_values()
    test_action_composition()
    test_action_to_tminterface()
    test_tminterface_to_action()
    test_round_trip()
    test_get_action()
    test_get_action_index()
    test_neutral_action()
    print("All action tests passed!")