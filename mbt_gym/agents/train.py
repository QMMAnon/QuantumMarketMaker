def train_agent(agent, env, episodes=1000):
    for episode in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.update_q_values(obs, action, reward, next_obs)
            total_reward += reward
            obs = next_obs

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")
