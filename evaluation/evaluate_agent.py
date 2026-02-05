# evaluation/evaluate_agent.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.trading_env import tradingEnv

# ===============================
# CONFIGURATION
# ===============================
MODEL_PATH = "model/rappo_trading_agent.zip"
DATA_PATH = "data/spy_features_regime.csv"
OUTPUT_FOLDER = "evaluation/results"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ===============================
# LOAD DATA & ENV
# ===============================
df = pd.read_csv(DATA_PATH)
env = tradingEnv(df=df, initial_cash=10000)

# ===============================
# LOAD MODEL
# ===============================
model = PPO.load(MODEL_PATH)

# ===============================
# RUN BACKTEST
# ===============================
obs, info = env.reset()
done = False

net_worths = []
cash_history = []
shares_history = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    net_worths.append(info["net_worth"])
    cash_history.append(info["cash"])
    shares_history.append(info["shares_held"])

# Convert to numpy arrays
net_worths = np.array(net_worths)
cash_history = np.array(cash_history)
shares_history = np.array(shares_history)

# ===============================
# METRICS
# ===============================
final_net_worth = net_worths[-1]
total_profit = final_net_worth - env.initial_cash
max_drawdown = np.max(np.maximum.accumulate(net_worths) - net_worths) / np.max(np.maximum.accumulate(net_worths))
daily_returns = pd.Series(net_worths).pct_change().fillna(0)
sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)  # annualized
volatility = daily_returns.std() * np.sqrt(252)

metrics = {
    "Final Net Worth": final_net_worth,
    "Total Profit": total_profit,
    "Max Drawdown": max_drawdown,
    "Sharpe Ratio": sharpe_ratio,
    "Volatility": volatility
}

print(metrics)

# Save metrics to CSV
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(os.path.join(OUTPUT_FOLDER, "evaluation_metrics.csv"), index=False)

# ===============================
# PLOTS
# ===============================
plt.figure(figsize=(12,6))
plt.plot(net_worths, label="Net Worth")
plt.plot(cash_history, label="Cash")
plt.plot(shares_history * df['Close'].values[:len(shares_history)], label="Shares Value")
plt.xlabel("Step")
plt.ylabel("Value ($)")
plt.title("Portfolio Value Over Time")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "portfolio_value.png"))
plt.close()

# Drawdown plot
running_max = np.maximum.accumulate(net_worths)
drawdowns = (running_max - net_worths) / running_max

plt.figure(figsize=(12,4))
plt.plot(drawdowns, label="Drawdown", color="red")
plt.xlabel("Step")
plt.ylabel("Drawdown")
plt.title("Drawdown Over Time")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "drawdown.png"))
plt.close()

print(f"Plots and metrics saved to '{OUTPUT_FOLDER}' folder.")
