import gym
from gym import spaces
import numpy as np

class CustomMarketMakingEnv(gym.Env):
    def __init__(self):
        super(CustomMarketMakingEnv, self).__init__()
        
        # Define the observation space and action space
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)  # Fixed to 4 features
        self.action_space = spaces.Discrete(3)
        
        # Initialize the state and episode length
        self.state = np.random.random(4)
        self.episode_length = 100
        self.current_step = 0

    def step(self, action):
        # Simulate market dynamics
        self.state = np.random.random(4)  # Generate new observation
        reward = np.random.random() - 0.5  # Random reward for demonstration
        self.current_step += 1
        
        # Check if episode should end
        done = self.current_step >= self.episode_length

        # Provide additional info if needed
        info = {"step": self.current_step}

        return self.state, reward, done, info

    def reset(self):
        self.state = np.random.random(4)
        self.current_step = 0
        return self.state

    def render(self, mode="human"):
        print(f"Current State: {self.state}")

    def close(self):
        pass
