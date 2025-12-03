from Env_Minesweeper import MinesweeperEnv
import numpy as np
import random

env = MinesweeperEnv()

NUM_EPISODES = 200      # smaller while testing
MAX_STEPS = 400         # safety cap per episode

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