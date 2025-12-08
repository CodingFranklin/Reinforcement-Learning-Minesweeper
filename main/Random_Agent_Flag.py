from Env_Minesweeper import MinesweeperEnv
from Settings import ROWS, COLS
import random
import json

NUM_EPISODES = 3000
MAX_STEPS = 100

NUM_ACTIONS = ROWS * COLS * 2

env = MinesweeperEnv()

random_rewards = []

for episode in range(NUM_EPISODES):
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    for step in range(MAX_STEPS):
        action = random.randrange(NUM_ACTIONS)
        next_obs, reward, done, _ = env.step(action)

        total_reward += reward
        obs = next_obs
        steps += 1

        if done:
            break

    if episode % 20 == 0:
        print(f"Random Ep {episode:4d} | steps={steps:3d} | total_reward={total_reward:7.2f}")
        random_rewards.append(total_reward)

with open("../random_rewards_flag.json", "w") as f:
    json.dump(random_rewards, f)