from gym.envs.registration import register
from .market_maker_env import CustomMarketMakingEnv

# Correct registration with standard format
register(
    id="MarketMaking-v0",  # Removed the namespace prefix for simplicity
    entry_point="mbt_gym.market_maker_env:CustomMarketMakingEnv",
)
