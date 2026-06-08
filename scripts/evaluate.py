import argparse
import sys
import threading
import time

from tminterface.client import run_client

from tmml.core.environment import TrackmaniaEnv, EnvClient
from tmml.core.actions import NUM_ACTIONS
from tmml.models.dqn import DQNAgent
from tmml.config.loader import load_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate TMML agent")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model checkpoint")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config YAML")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    args = parser.parse_args()

    config = load_config(args.config)

    print("Connecting to TMInterface...")
    client = EnvClient(verbose=True)
    server_name = "TMInterface0"

    eval_result = {"error": None}

    def run_evaluation():
        try:
            print(f"Waiting for connection to {server_name}...")
            timeout = 30
            start_time = time.time()
            while not client._registered and (time.time() - start_time) < timeout:
                time.sleep(0.1)

            if not client._registered:
                eval_result["error"] = f"Failed to connect to {server_name} after {timeout}s."
                return

            print("Connected! Waiting for simulation...")
            while client.initial_state is None and (time.time() - start_time) < 60:
                time.sleep(0.1)

            if client.initial_state is None:
                eval_result["error"] = "No simulation state received."
                return

            env = TrackmaniaEnv(
                waypoints_path=config.environment.waypoints_path,
                client=client,
                reward_config=config.environment.reward_config,
                max_steps=config.environment.max_steps,
                look_aheads=config.environment.look_aheads,
            )

            obs_size = env.observation_space.shape[0]

            agent = DQNAgent(
                obs_size=obs_size,
                n_actions=NUM_ACTIONS,
                config=config.dqn,
            )
            agent.load(args.model)
            agent.epsilon = 0.0

            print(f"\nEvaluating for {args.episodes} episodes (epsilon=0)...\n")

            total_rewards = []
            total_steps = []
            completions = 0

            for ep in range(1, args.episodes + 1):
                obs, info = env.reset()
                episode_reward = 0.0
                episode_steps = 0
                done = False

                while not done:
                    action_idx = agent.select_action(obs)
                    next_obs, reward, terminated, truncated, step_info = env.step(action_idx)
                    obs = next_obs
                    episode_reward += reward
                    episode_steps += 1

                    if terminated or truncated:
                        done = True
                        if step_info.get("term_reason") == "finished":
                            completions += 1

                total_rewards.append(episode_reward)
                total_steps.append(episode_steps)
                print(f"Episode {ep:3d} | Reward: {episode_reward:8.2f} | Steps: {episode_steps:4d} | Reason: {step_info.get('term_reason', 'unknown')}")

            avg_reward = sum(total_rewards) / len(total_rewards)
            avg_steps = sum(total_steps) / len(total_steps)
            completion_rate = completions / args.episodes

            print(f"\n=== RESULTS ===")
            print(f"Avg Reward: {avg_reward:.2f}")
            print(f"Avg Steps:  {avg_steps:.1f}")
            print(f"Completions: {completions}/{args.episodes} ({completion_rate:.0%})")

            env.close()

        except Exception as e:
            eval_result["error"] = str(e)

    eval_thread = threading.Thread(target=run_evaluation, daemon=True)
    eval_thread.start()

    run_client(client, server_name)

    eval_thread.join(timeout=5)
    if eval_result["error"]:
        print(f"Error: {eval_result['error']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()