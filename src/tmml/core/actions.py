from dataclasses import dataclass
from typing import List
import gymnasium
import numpy as np


@dataclass
class Action:
    accelerate: bool
    brake: bool
    steer: int


STEER_VALUES = [-1, 0, 1]

ACTIONS: List[Action] = []
for accel in [False, True]:
    for brake in [False, True]:
        for steer in STEER_VALUES:
            ACTIONS.append(Action(accelerate=accel, brake=brake, steer=steer))

NUM_ACTIONS = len(ACTIONS)

ACTION_SPACE = gymnasium.spaces.Discrete(NUM_ACTIONS)


def action_to_tminterface(action: Action) -> dict:
    return {
        "accelerate": action.accelerate,
        "brake": action.brake,
        "steer": int(action.steer * 65536),
    }


def tminterface_to_action(accelerate: bool, brake: bool, steer: int) -> Action:
    steer_value = round(steer / 65536)
    steer_value = max(-1, min(1, steer_value))
    if steer_value not in STEER_VALUES:
        closest = min(STEER_VALUES, key=lambda s: abs(s - steer_value))
        steer_value = closest
    return Action(accelerate=accelerate, brake=brake, steer=steer_value)


def get_action(index: int) -> Action:
    return ACTIONS[index]


def get_action_index(action: Action) -> int:
    for i, a in enumerate(ACTIONS):
        if a.accelerate == action.accelerate and a.brake == action.brake and a.steer == action.steer:
            return i
    return -1