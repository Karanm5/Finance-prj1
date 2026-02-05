import pandas as pd
import matplotlib.pyplot as plt

# Load backtest results CSV
df = pd.read_csv("backtesting/portfolio_history.csv")  # replace with your path

plt.figure(figsize=(12,6))
plt.plot(df["net_worth"], label="Portfolio Net Worth", color="blue")
plt.title("Portfolio Net Worth Over Time")
plt.xlabel("Time Step")
plt.ylabel("Net Worth ($)")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("backtesting/net_worth_plot.png")
#CASH VS SHARE VALUE
plt.figure(figsize=(12,6))
plt.plot(df["cash"], label="Cash", color="green")
plt.plot(df["net_worth"] - df["cash"], label="Shares Value", color="orange")
plt.title("Cash vs Shares Value")
plt.xlabel("Time Step")
plt.ylabel("Value ($)")
plt.legend()
plt.show()
plt.savefig("backtesting/cashvssharevalue.png")


