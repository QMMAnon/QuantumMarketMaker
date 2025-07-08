"""
evaluate_vqc_ddpg.py

Evaluates a trained VQC-DDPG market making agent using saved actor weights.
Outputs episode rewards and final inventory levels.
"""

import os
import numpy as np
import argparse
import random
import tensorflow as tf

# Reproducibility and minimal logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from mbt_gym.gym.TradingEnvironment import TradingEnvironment
from mbt_gym.agents.VQCDDPGMarketMakerAgent import VQCDDPGMarketMakerAgent
from mbt_gym.gym.index_names import INVENTORY_INDEX

def evaluate(model_path_actor, episodes=5):
    """
    Runs evaluation episodes using the specified actor weights.
    """
    env = TradingEnvironment(seed=SEED)
    action_size = int(np.prod(env.action_space.shape))
    state_size = env.observation_space.shape[0]

    # Load agent with pre-trained weights
    agent = VQCDDPGMarketMakerAgent(
        state_dim=state_size,
        action_dim=action_size,
        action_bound=env.action_space.high[0],
    )

    dummy_input = tf.zeros((1, state_size))
    agent.actor(dummy_input)
    agent.actor.load_weights(model_path_actor)
    print(f"Loaded actor weights from {model_path_actor}")

    all_rewards, all_inventory = [], []

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward, done, steps = 0, False, 0

        while not done:
            action = agent.act(state, use_noise=False)
            if len(action.shape) == 1:
                action = action.reshape((1, -1))
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1

        inventory_size = next_state[0, 1]
        if env.normalise_observation_space_:
            inventory_size = (
                (inventory_size + 1)
                * env._gradient_obs_norm[INVENTORY_INDEX]
                + env._intercept_obs_norm[INVENTORY_INDEX]
            )

        all_rewards.append(float(total_reward))
        all_inventory.append(float(inventory_size))

        print(f"Episode {episode}: Total Reward = {float(total_reward):.2f}, Final Inventory = {float(inventory_size):.2f}, Steps = {steps}")

    print("\n VQC-DDPG Evaluation Summary:")
    print(f"Average Reward: {np.mean(all_rewards):.2f}")
    print(f"Average Final Inventory: {np.mean(all_inventory):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained VQC-DDPG agent.")
    parser.add_argument("--model_path_actor", type=str, required=True, help="Path to saved actor weights")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    args = parser.parse_args()

    evaluate(model_path_actor=args.model_path_actor, episodes=args.episodes)
