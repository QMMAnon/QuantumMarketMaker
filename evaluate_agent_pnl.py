"""
Evaluation script for DDPG, PQC, and VQC Market Maker agents.
Exports policy outputs and prints evaluation statistics.
"""

import tensorflow as tf
import numpy as np

from mbt_gym.gym.TradingEnvironment import TradingEnvironment
from mbt_gym.agents.DDPGMarketMakerAgent import DDPGMarketMakerAgent
from mbt_gym.agents.PQCDDPGMarketMakerAgent import PQCDDPGMarketMakerAgent
from mbt_gym.agents.VQCDDPGMarketMakerAgent import VQCDDPGMarketMakerAgent

def evaluate_and_export(agent_class, weights_path, export_path, state_size, action_size, action_bounds, agent_name, env_config=None, **kwargs):
    """
    Loads agent, evaluates on environment, exports sample policy outputs.
    """
    if env_config is None:
        env_config = {}
    env = TradingEnvironment(seed=12)

    # Instantiate agent
    agent = agent_class(state_size, action_size, action_bounds, **kwargs)
    agent.actor(tf.zeros((1, state_size)))  # Build model

    # Load weights
    try:
        agent.actor.load_weights(weights_path)
    except Exception as e:
        print(f"Failed to load weights for {agent_name}: {e}")
        return

    # Save loaded actor weights
    actor_weights_path = f"{export_path}_actor.weights.h5"
    agent.actor.save_weights(actor_weights_path)
    print(f"Saved actor weights to: {actor_weights_path}")

    # Export sample policy outputs
    sample_states = np.random.uniform(low=-1, high=1, size=(1000, state_size)).astype(np.float32)
    actions = agent.actor(sample_states).numpy()
    np.savez(f"{export_path}_policy_outputs.npz", states=sample_states, actions=actions)
    print(f"Saved 1000 sample policy outputs to: {export_path}_policy_outputs.npz")

    # Evaluate agent on episodes
    num_episodes = 5
    max_steps = env_config.get("max_episode_steps", 1000)
    rewards, pnls, exec_ratios = [], [], []

    for _ in range(num_episodes):
        state = env.reset()
        total_reward, total_pnl, executed, total_orders = 0, 0, 0, 0
        done = False

        for _ in range(max_steps):
            action = agent.act(state, use_noise=False)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            total_pnl += info.get("pnl", 0)
            executed += info.get("executed", 0)
            total_orders += 1
            state = next_state
            if done:
                break

        rewards.append(total_reward)
        pnls.append(total_pnl)
        exec_ratios.append(executed / max(total_orders, 1))

    # Print evaluation statistics
    print(f"\n===== {agent_name.upper()} EVALUATION =====")
    print(f"Avg Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Avg PnL:    {np.mean(pnls):.2f} ± {np.std(pnls):.2f}")
    print(f"Exec Ratio: {np.mean(exec_ratios) * 100:.2f}%\n")

state_size = 4
action_size = 2
action_bounds = 1.0

env_config = {}

# Evaluate DDPG agent
evaluate_and_export(
    DDPGMarketMakerAgent,
    "results/ddpg_model19_best_actor.weights.h5",
    "ddpg_policy",
    state_size, action_size, action_bounds,
    "DDPG",
    env_config=env_config
)

# Evaluate PQC agent
evaluate_and_export(
    PQCDDPGMarketMakerAgent,
    "results/pqc41_best_actor.weights.h5",
    "pqc_policy",
    state_size, action_size, action_bounds,
    "PQC",
    env_config=env_config,
    n_qubits=4
)

# Evaluate VQC agent
evaluate_and_export(
    VQCDDPGMarketMakerAgent,
    "results/vqc76_best_actor.weights.h5", 
    "vqc_policy",
    state_size, action_size, action_bounds,
    "VQC",
    env_config=env_config,
)
