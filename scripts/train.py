import argparse
import sys
import threading
import time

from tminterface.client import run_client

from tmml.core.environment import TrackmaniaEnv, EnvClient
from tmml.core.actions import NUM_ACTIONS
from tmml.models.dqn import DQNAgent
from tmml.training.logger import TrainingLogger
from tmml.training.trainer import Trainer
from tmml.config.loader import load_config


def main():
    parser = argparse.ArgumentParser(description="Train TMML agent")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config YAML")
    parser.add_argument("--episodes", type=int, default=None, help="Override max episodes")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--epsilon-start", type=float, default=None, help="Override epsilon start")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.episodes:
        config.training.max_episodes = args.episodes
    if args.lr:
        config.dqn.learning_rate = args.lr
    if args.epsilon_start:
        config.dqn.epsilon_start = args.epsilon_start

    print("Connecting to TMInterface...")
    client = EnvClient(verbose=True)
    server_name = "TMInterface0"

    training_result = {"error": None}

    def run_training():
        try:
            print(f"Waiting for connection to {server_name}...")
            timeout = 30
            start_time = time.time()
            while not client._registered and (time.time() - start_time) < timeout:
                time.sleep(0.1)

            if not client._registered:
                training_result["error"] = f"Failed to connect to {server_name} after {timeout}s."
                return

            print("Connected! Waiting for simulation to start...")

            sim_timeout = 60
            start_time = time.time()
            while client.initial_state is None and (time.time() - start_time) < sim_timeout:
                time.sleep(0.1)

            if client.initial_state is None:
                training_result["error"] = f"No simulation state received after {sim_timeout}s."
                return

            print("Simulation ready. Creating environment...")

            env = TrackmaniaEnv(
                waypoints_path=config.environment.waypoints_path,
                client=client,
                reward_config=config.environment.reward_config,
                max_steps=config.environment.max_steps,
                look_aheads=config.environment.look_aheads,
            )

            obs_size = env.observation_space.shape[0]
            print(f"Observation size: {obs_size}")

            agent = DQNAgent(
                obs_size=obs_size,
                n_actions=NUM_ACTIONS,
                config=config.dqn,
            )

            logger = TrainingLogger(log_dir=config.training.log_dir)

            trainer = Trainer(
                env=env,
                agent=agent,
                logger=logger,
                config=config,
            )

            try:
                trainer.train()
            except KeyboardInterrupt:
                print("\nTraining interrupted by user.")
                trainer.save_checkpoint(episode=trainer._total_steps)
            finally:
                env.close()
                logger.close()
                print("Cleanup done.")

        except Exception as e:
            training_result["error"] = str(e)

    training_thread = threading.Thread(target=run_training, daemon=True)
    training_thread.start()

    run_client(client, server_name)

    training_thread.join(timeout=5)
    if training_result["error"]:
        print(f"Error: {training_result['error']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()