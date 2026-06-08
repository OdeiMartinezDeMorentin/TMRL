import yaml
from dataclasses import dataclass, field
from typing import List, Optional

from tmml.core.rewards import RewardConfig


@dataclass
class DQNConfig:
    learning_rate: float = 0.001
    batch_size: int = 64
    replay_buffer_size: int = 100000
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 10000
    target_update_freq: int = 1000
    hidden_layers: List[int] = field(default_factory=lambda: [128, 128])
    activation: str = "relu"


@dataclass
class TrainingConfig:
    max_episodes: int = 10000
    max_steps_per_episode: int = 2000
    failure_countdown: int = 200
    eval_frequency: int = 100
    eval_episodes: int = 5
    save_frequency: int = 500
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"


@dataclass
class EnvironmentConfig:
    waypoints_path: str = "config/tracks/default/waypoints.json"
    max_steps: int = 3000
    look_aheads: List[int] = field(default_factory=lambda: [1, 3, 7])
    reward_config: RewardConfig = field(default_factory=RewardConfig)


@dataclass
class TMMLConfig:
    dqn: DQNConfig = field(default_factory=DQNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)


def load_config(path: str) -> TMMLConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    reward_config = RewardConfig(
        progress_weight=raw.get("rewards", {}).get("progress_weight", 1.0),
        speed_weight=raw.get("rewards", {}).get("speed_weight", 0.1),
        step_penalty=raw.get("rewards", {}).get("step_penalty", 0.001),
        completion_bonus=raw.get("rewards", {}).get("completion_bonus", 100.0),
        no_progress_timeout=raw.get("rewards", {}).get("no_progress_timeout", 200),
    )

    env_raw = raw.get("environment", {})
    env_config = EnvironmentConfig(
        waypoints_path=env_raw.get("waypoints_path", "config/tracks/default/waypoints.json"),
        max_steps=env_raw.get("max_steps", 3000),
        look_aheads=raw.get("state", {}).get("look_aheads", [1, 3, 7]),
        reward_config=reward_config,
    )

    dqn_raw = raw.get("dqn", {})
    dqn_config = DQNConfig(
        learning_rate=dqn_raw.get("learning_rate", 0.001),
        batch_size=dqn_raw.get("batch_size", 64),
        replay_buffer_size=dqn_raw.get("replay_buffer_size", 100000),
        gamma=dqn_raw.get("gamma", 0.99),
        epsilon_start=dqn_raw.get("epsilon_start", 1.0),
        epsilon_end=dqn_raw.get("epsilon_end", 0.01),
        epsilon_decay=dqn_raw.get("epsilon_decay", 10000),
        target_update_freq=dqn_raw.get("target_update_freq", 1000),
        hidden_layers=dqn_raw.get("hidden_layers", [128, 128]),
        activation=dqn_raw.get("activation", "relu"),
    )

    train_raw = raw.get("training", {})
    training_config = TrainingConfig(
        max_episodes=train_raw.get("max_episodes", 10000),
        max_steps_per_episode=train_raw.get("max_steps_per_episode", 2000),
        failure_countdown=train_raw.get("failure_countdown", 200),
        eval_frequency=train_raw.get("eval_frequency", 100),
        eval_episodes=train_raw.get("eval_episodes", 5),
        save_frequency=train_raw.get("save_frequency", 500),
        log_dir=train_raw.get("log_dir", "logs"),
        checkpoint_dir=train_raw.get("checkpoint_dir", "checkpoints"),
    )

    return TMMLConfig(dqn=dqn_config, training=training_config, environment=env_config)