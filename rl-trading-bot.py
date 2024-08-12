import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
import gym_anytrading

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C

AMZN_1 = pd.read_csv('data/AMZN.csv')
AMZN_1 = pd.DataFrame(AMZN_1)
AMZN_1 = AMZN_1.drop('Adj Close', axis=1)

AMZN_1['Date'] = pd.to_datetime(AMZN_1['Date'])
AMZN_1.set_index('Date', inplace=True)

env_maker = lambda: gym.make('stocks-v0', df=AMZN_1, frame_bound=(10, 252), window_size=10)
env = DummyVecEnv([env_maker])

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

env = gym.make('stocks-v0', df=AMZN_1, frame_bound=(10, 252), window_size=10)
obs, _ = env.reset()

while True:
    action, _states = model.predict(obs)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated 

    if done:
        print("info:", info)
        break

plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.savefig('AMZN_A2C.png')