import os
from typing import Optional

from tmml.core.environment import TrackmaniaEnv, EnvClient
from tmml.models.dqn import DQNAgent
from tmml.training.logger import TrainingLogger
from tmml.config.loader import TMMLConfig


class Trainer:
    def __init__(
        self,
        env: TrackmaniaEnv,
        agent: DQNAgent,
        logger: TrainingLogger,
        config: TMMLConfig,
    ):
        self.env = env
        self.agent = agent
        self.logger = logger
        self.config = config

        self._total_steps = 0

    def train(self):
        print(f"Starting training for {self.config.training.max_episodes} episodes...")
        print(f"Device: {self.agent.device}")
        print(f"Observation size: {self.agent.obs_size}, Actions: {self.agent.n_actions}")
        print(f"Epsilon: {self.agent.epsilon:.4f}")

        for episode in range(1, self.config.training.max_episodes + 1):
            obs, info = self.env.reset()
            episode_reward = 0.0
            episode_steps = 0
            term_reason = "running"
            waypoint_idx = 0
            speed_sum = 0.0

            done = False
            while not done:
                action_idx = self.agent.select_action(obs)
                next_obs, reward, terminated, truncated, step_info = self.env.step(action_idx)

                self.agent.store_transition(obs, action_idx, reward, next_obs, terminated)

                loss = self.agent.train_step(self.config.dqn.batch_size)

                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                self._total_steps += 1

                waypoint_idx = step_info.get("waypoint_idx", waypoint_idx)
                speed_sum += step_info.get("speed", 0.0)

                if terminated or truncated:
                    done = True
                    term_reason = step_info.get("term_reason", "unknown")

            self.agent.decay_epsilon()

            if episode % self.config.dqn.target_update_freq == 0:
                self.agent.update_target_network()

            total_waypoints = self.env.wpm.total_count
            progress_pct = waypoint_idx / max(total_waypoints - 1, 1)
            speed_avg = speed_sum / max(episode_steps, 1)

            self.logger.log_episode(
                episode=episode,
                reward=episode_reward,
                steps=episode_steps,
                epsilon=self.agent.epsilon,
                term_reason=term_reason,
                waypoint_idx=waypoint_idx,
                speed_avg=speed_avg,
                progress_pct=progress_pct,
            )

            if episode % self.config.training.eval_frequency == 0:
                self.evaluate()

            if episode % self.config.training.save_frequency == 0:
                self.save_checkpoint(episode)

        self.logger.close()
        print("Training complete.")

    def evaluate(self):
        print(f"\n--- Evaluation ({self.config.training.eval_episodes} episodes) ---")
        total_rewards = []
        total_steps = []
        completions = 0

        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0

        for _ in range(self.config.training.eval_episodes):
            obs, info = self.env.reset()
            episode_reward = 0.0
            episode_steps = 0
            done = False

            while not done:
                action_idx = self.agent.select_action(obs)
                next_obs, reward, terminated, truncated, step_info = self.env.step(action_idx)
                obs = next_obs
                episode_reward += reward
                episode_steps += 1

                if terminated or truncated:
                    done = True
                    if step_info.get("term_reason") == "finished":
                        completions += 1

            total_rewards.append(episode_reward)
            total_steps.append(episode_steps)

        self.agent.epsilon = original_epsilon

        avg_reward = sum(total_rewards) / len(total_rewards)
        avg_steps = sum(total_steps) / len(total_steps)
        completion_rate = completions / self.config.training.eval_episodes

        self.logger.log_eval(
            episode=self._total_steps,
            avg_reward=avg_reward,
            avg_steps=avg_steps,
            completion_rate=completion_rate,
        )

    def save_checkpoint(self, episode: int):
        os.makedirs(self.config.training.checkpoint_dir, exist_ok=True)
        path = os.path.join(
            self.config.training.checkpoint_dir,
            f"dqn_episode_{episode}.pt",
        )
        self.agent.save(path)
        print(f"Checkpoint saved: {path}")