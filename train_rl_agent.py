import pandas as pd
from stable_baselines3 import PPO
from env.trading_env import tradingEnv

df = pd.read_csv("data/spy_features_regime.csv")

env = tradingEnv(df)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate = 3e-4,
    batch_size = 256,
    gamma=0.99,
    gae_lambda=0.95
)

model.learn(total_timesteps = 200_000)

model.save("models/rappo_trading_agent")

print("Training complete")