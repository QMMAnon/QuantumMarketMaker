import numpy as np

class RandomMarketMakerAgent:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def act(self, state):
        # Return a single (1, 2) action with values uniformly sampled from [-1, 1]
        return np.random.uniform(low=-1.0, high=1.0, size=(1, self.action_dim))

    def load(self, path):
        # Random agent doesn't need to load any model
        pass
