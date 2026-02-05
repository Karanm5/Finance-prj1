import pandas as pd
from stable_baselines3 import PPO
from env.trading_env import tradingEnv
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/spy_features_regime.csv")

# Initialize environment
env = tradingEnv(df)

# Load trained model
model = PPO.load("model/rappo_trading_agent")

# Reset environment
obs, info = env.reset()
done = False

# Track portfolio over time
portfolio_history = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    portfolio_history.append(info)

# Convert to DataFrame for easy analysis
portfolio_df = pd.DataFrame(portfolio_history)
portfolio_df.to_csv("backtesting/portfolio_history.csv", index=False)

# Plot Net Worth
plt.plot(portfolio_df["net_worth"])
plt.title("Portfolio Net Worth Over Time")
plt.xlabel("Step")
plt.ylabel("Net Worth")
plt.show()

# Print key metrics
final_net_worth = portfolio_df["net_worth"].iloc[-1]
total_profit = final_net_worth - env.initial_cash
max_drawdown = portfolio_df["Drawdown"].max()

print(f"Final Net Worth: {final_net_worth}")
print(f"Total Profit: {total_profit}")
print(f"Max Drawdown: {max_drawdown}")
