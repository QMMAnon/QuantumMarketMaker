import time
import argparse
import numpy as np
import os
import torch
from mbt_gym.gym.TradingEnvironment import TradingEnvironment
from mbt_gym.agents.DQNMarketMakerAgent import DQNMarketMakerAgent
from mbt_gym.agents.PQCMarketMakerAgent import PQCMarketMakerAgent
from mbt_gym.agents.VQCMarketMakerAgent import VQCMarketMakerAgent

def test(agent_type, episodes, model_path):
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Testing on:", device)

    env = TradingEnvironment()
    print(f"env.action_space: {env.action_space}")
    print(f"env.observation_space.shape: {env.observation_space.shape}")

    state_size = env.observation_space.shape[0]
    action_size = int(np.prod(env.action_space.shape))

    # Initialise agent based on type
    if agent_type == "dqn":
        agent = DQNMarketMakerAgent(state_size, action_size)
    elif agent_type == "pqc":
        agent = PQCMarketMakerAgent(state_size, action_size)
    elif agent_type == "vqc":
        agent = VQCMarketMakerAgent(state_size, action_size)
    else:
        raise ValueError("Only 'dqn' agent is supported in this test script.")

    # Load trained model
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    agent.load(model_path)
    print(f"Loaded model from {model_path}")

    total_rewards = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0
        start_time = time.time()

        while not done:
            step += 1
            action = agent.act(state, explore=False)
            action = np.asarray(action)

            # Reshape action if needed
            if action.shape == (4,):
                action = action.reshape(2, 2)[0].reshape(1, 2)
            elif action.shape == (2, 2):
                action = action[0].reshape(1, 2)
            elif action.shape != (1, 2):
                raise ValueError(f"Unsupported action shape: {action.shape}")

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

        episode_time = time.time() - start_time
        total_rewards.append(float(total_reward))
        print(f"Episode {episode}: Total Reward = {float(total_reward):.2f}, Steps = {step}, Time = {episode_time:.2f}s")

    avg_reward = np.mean(total_rewards)
    print("\nTESTING COMPLETE")
    print(f"Average Reward over {episodes} episodes: {avg_reward:.2f}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained market maker agent.")
    parser.add_argument("--agent", type=str, required=True, choices=["ddpg", "pqc", "vqc"], help="Type of agent to test")
    parser.add_argument("--episodes", type=int, required=True, help="Number of test episodes")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")

    args = parser.parse_args()
    test(agent_type=args.agent, episodes=args.episodes, model_path=args.model_path)
