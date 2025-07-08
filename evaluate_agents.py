"""
main_test.py

Evaluates DDPG, PQC, and VQC agents in the MBT-Gym trading environment
using pre-trained actor weights. Outputs episode rewards and inventory.
"""

import os
import random
import numpy as np
import tensorflow as tf
import argparse

# Environment setup for reproducibility
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs except errors
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from mbt_gym.gym.TradingEnvironment import TradingEnvironment
from mbt_gym.gym.index_names import INVENTORY_INDEX
from mbt_gym.agents.DDPGMarketMakerAgent import DDPGMarketMakerAgent
from mbt_gym.agents.PQCMarketMakerAgent import PQCMarketMakerAgent
from mbt_gym.agents.VQCDDPGMarketMakerAgent import VQCDDPGMarketMakerAgent

def evaluate(agent_class, model_path_actor, episodes=5, label="Agent"):
    """
    Runs evaluation episodes for a given agent class and pre-trained model.
    """
    print(f"\n Evaluating {label}...")
    env = TradingEnvironment(seed=SEED)
    action_size = int(np.prod(env.action_space.shape))
    state_size = env.observation_space.shape[0]

    # Instantiate agent and load weights
    agent = agent_class(state_size, action_size, env.action_space.high[0], seed=SEED)
    agent.actor.load_weights(model_path_actor)
    print(f" Loaded {label} weights from {model_path_actor}")

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

        # Extract final inventory size (denormalised if required)
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

        print(f"{label} - Episode {episode}: Total Reward = {reward_scalar:.2f}, Final Inventory = {inventory_scalar:.2f}, Steps = {steps}")

    print(f"\n {label} Evaluation Summary:")
    print(f"Average Reward: {np.mean(all_rewards):.2f}")
    print(f"Average Final Inventory: {np.mean(all_inventory):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all agents.")
    parser.add_argument("--ddpg_actor", type=str, required=True, help="DDPG actor weights file")
    parser.add_argument("--pqc_actor", type=str, required=True, help="PQC actor weights file")
    parser.add_argument("--vqc_actor", type=str, required=True, help="VQC actor weights file")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    args = parser.parse_args()

    evaluate(DDPGMarketMakerAgent, args.ddpg_actor, args.episodes, label="DDPG")
    evaluate(PQCMarketMakerAgent, args.pqc_actor, args.episodes, label="PQC")
    evaluate(VQCDDPGMarketMakerAgent, args.vqc_actor, args.episodes, label="VQC")
