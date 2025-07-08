import os
import time
import csv
import argparse
import numpy as np
import torch
import random
from mbt_gym.gym.TradingEnvironment import TradingEnvironment
from mbt_gym.agents.DDPGMarketMakerAgent import DDPGMarketMakerAgent
from mbt_gym.agents.PQCDDPGMarketMakerAgent import PQCDDPGMarketMakerAgent
from mbt_gym.agents.VQCDDPGMarketMakerAgent import VQCDDPGMarketMakerAgent
from mbt_gym.gym.index_names import INVENTORY_INDEX

def get_last_episode(metrics_file):
    # Returns last recorded episode from metrics CSV
    if not os.path.isfile(metrics_file):
        return 0
    with open(metrics_file, "r") as file:
        lines = file.readlines()
        if len(lines) < 2:
            return 0
        return int(lines[-1].split(",")[0])

def main(agent_type, episodes, batch_size, save_path, metrics_path):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    try:
        if not tf.config.list_physical_devices("GPU"):
            print("No usable GPU found. Forcing CPU fallback.")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    except Exception as e:
        print(f"TensorFlow GPU check failed: {e}\nFalling back to CPU-only mode.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)

    env = TradingEnvironment()
    seed = 6
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    action_size = int(np.prod(env.action_space.shape))
    state_size = env.observation_space.shape[0]

    # Initialise agent
    if agent_type == "ddpg":
        agent = DDPGMarketMakerAgent(state_size, action_size, env.action_space.high[0])
    elif agent_type == "pqc":
        agent = PQCDDPGMarketMakerAgent(state_size, action_size, env.action_space.high[0])
    elif agent_type == "vqc":
        agent = VQCDDPGMarketMakerAgent(state_size, action_size, env.action_space.high[0])
    else:
        raise ValueError("Invalid agent type! Choose from 'ddpg', 'pqc', 'vqc'.")

    # Load existing model if available
    try:
        if os.path.isfile(save_path + "_actor.h5"):
            agent.load(save_path)
            print(f"Loaded model from {save_path} - Resuming training")
    except Exception as e:
        print(f"Failed to load model weights from {save_path}: {e}")
        print("Continuing with randomly initialized model.")

    last_episode = get_last_episode(metrics_path)
    print(f"Resuming from Episode {last_episode + 1}")

    best_reward = float('-inf')
    no_improvement_count = 0
    patience = 5000

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    file_exists = os.path.isfile(metrics_path)

    with open(metrics_path, "a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Episode", "Total Reward", "Steps", "Inventory Size", "Epsilon", "Time (s)", "Actor Loss", "Critic Loss"])

        try:
            for episode in range(last_episode + 1, last_episode + 1 + episodes):
                state = env.reset()
                total_reward = 0
                start_time = time.time()
                done = False
                step = 0

                while not done:
                    step += 1
                    action = agent.act(state)
                    next_state, reward, done, _ = env.step(action)

                    # Inventory penalty shaping
                    inventory_size = next_state[0, 1]
                    if env.normalise_observation_space_:
                        inventory_size = (
                            (inventory_size + 1)
                            * env._gradient_obs_norm[INVENTORY_INDEX]
                            + env._intercept_obs_norm[INVENTORY_INDEX]
                        )
                    reward -= 0.01 * abs(inventory_size)

                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward

                    if step % 4 == 0:
                        agent.train()

                agent.train()
                episode_time = time.time() - start_time

                reward_scalar = float(total_reward.item() if hasattr(total_reward, "item") else total_reward)
                inventory_scalar = float(inventory_size.item() if hasattr(inventory_size, "item") else inventory_size)
                actor_loss = getattr(agent, 'last_actor_loss', 'N/A')
                critic_loss = getattr(agent, 'last_critic_loss', 'N/A')

                if reward_scalar > best_reward:
                    best_reward = reward_scalar
                    no_improvement_count = 0
                    agent.save(save_path + "_best")
                    print(f"New BEST model saved at Episode {episode} with reward: {reward_scalar:.2f}")
                else:
                    no_improvement_count += 1

                agent.update_noise(episode, last_episode + episodes)

                writer.writerow([
                    episode, reward_scalar, step, inventory_scalar,
                    getattr(agent, 'epsilon', 'N/A'), episode_time,
                    actor_loss, critic_loss
                ])
                file.flush()

                # Moving average reporting
                if episode - last_episode >= 100:
                    recent_rewards = [float(row[1]) for row in list(csv.reader(open(metrics_path)))[-100:]]
                    reward_ma = np.mean(recent_rewards)
                    print(f"Episode {episode}/{last_episode + episodes} - Reward: {reward_scalar:.2f}, MA(100): {reward_ma:.2f}, Steps: {step}, Inventory: {inventory_scalar:.2f}, Time: {episode_time:.2f}s")
                else:
                    print(f"Episode {episode}/{last_episode + episodes} - Reward: {reward_scalar:.2f}, Steps: {step}, Inventory: {inventory_scalar:.2f}, Time: {episode_time:.2f}s")

                if no_improvement_count >= patience:
                    print(f"Early stopping at episode {episode} (no improvement for {patience} episodes)")
                    break

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")

        finally:
            agent.save(save_path + "_final")
            print(f"Final model saved to {save_path}_final")
            print(f"Metrics written to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a market making agent.")
    parser.add_argument("--agent", type=str, required=True, choices=["ddpg", "pqc", "vqc"], help="Agent type")
    parser.add_argument("--episodes", type=int, required=True, help="Number of episodes")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--save_path", type=str, required=True, help="Model save path prefix (no extension)")
    parser.add_argument("--metrics_path", type=str, required=True, help="CSV metrics file path")
    args = parser.parse_args()

    main(
        agent_type=args.agent,
        episodes=args.episodes,
        batch_size=args.batch_size,
        save_path=args.save_path,
        metrics_path=args.metrics_path
    )
