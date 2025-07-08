"""
evaluate_ddpg.py

Evaluates a trained DDPG market making agent using saved actor weights.
Outputs episode rewards and final inventory levels.
"""

import os
import numpy as np
import argparse
import random
import tensorflow as tf

# Suppress TensorFlow logs except fatal errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Reproducibility settings
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from mbt_gym.gym.TradingEnvironment import TradingEnvironment
from mbt_gym.agents.DDPGMarketMakerAgent import DDPGMarketMakerAgent
from mbt_gym.gym.index_names import INVENTORY_INDEX

def evaluate(model_path_actor, episodes=5):
    """
    Runs evaluation episodes using the specified actor weights.
    """
    env = TradingEnvironment(seed=SEED)

    action_size = int(np.prod(env.action_space.shape))
    state_size = env.observation_space.shape[0]

    # Load agent with pre-trained weights
    agent = DDPGMarketMakerAgent(
        state_size=state_size,
        action_size=action_size,
        action_bounds=env.action_space.high[0],
        seed=SEED 
    )
    agent.actor.load_weights(model_path_actor)
    print(f"Loaded actor weights from {model_path_actor}")

    all_rewards = []
    all_inventory = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            action = agent.act(state, use_noise=False)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1

        # Extract final inventory (denormalise if needed)
        inventory_size = next_state[0, 1]
        if env.normalise_observation_space_:
            inventory_size = (
                (inventory_size + 1)
                * env._gradient_obs_norm[INVENTORY_INDEX]
                + env._intercept_obs_norm[INVENTORY_INDEX]
            )

        reward_scalar = float(total_reward.item() if hasattr(total_reward, "item") else total_reward)
        inventory_scalar = float(inventory_size.item() if hasattr(inventory_size, "item") else inventory_size)

        all_rewards.append(reward_scalar)
        all_inventory.append(inventory_scalar)

        print(f"Episode {episode}: Total Reward = {reward_scalar:.2f}, Final Inventory = {inventory_scalar:.2f}, Steps = {steps}")

    print("\n Evaluation Summary:")
    print(f"Average Reward: {np.mean(all_rewards):.2f}")
    print(f"Average Final Inventory: {np.mean(all_inventory):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained DDPG agent.")
    parser.add_argument("--model_path_actor", type=str, required=True, help="Path to saved actor weights (e.g. *_actor.h5)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    args = parser.parse_args()

    evaluate(model_path_actor=args.model_path_actor, episodes=args.episodes)
s