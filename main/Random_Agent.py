from Env_Minesweeper import MinesweeperEnv
from Settings import ROWS, COLS
import numpy as np
import random

env = MinesweeperEnv()

NUM_EPISODES = 500      
MAX_STEPS = 400    
random_rewards = []     

for episode in range(NUM_EPISODES):
    obs = env.reset()
    done = False
    steps = 0
    total_reward = 0.0

    while not done and steps < MAX_STEPS:
        action = random.randint(0, ROWS * COLS - 1)
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        obs = next_obs
        steps += 1

    if episode % 10 == 0:
        print(f"Episode {episode}: steps={steps}, total_reward={total_reward:.2f}")
        random_rewards.append(total_reward)


import json
with open("random_rewards.json", "w") as f:
    json.dump(random_rewards, f)