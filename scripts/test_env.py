import sys
import os


# add project root to python path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from env.trading_env import tradingEnv

env = tradingEnv()
obs = env.reset()


for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(info)

    if done:
        break
    