import json
import matplotlib.pyplot as plt
import numpy as np

# ===== Load DQN data =====
with open("dqn_rewards.json", "r") as f:
    dqn_data = json.load(f)

dqn_episodes = dqn_data["episodes"]
dqn_rewards = dqn_data["rewards"]

# ===== Load Random Agent data =====
with open("random_rewards.json", "r") as f:
    random_rewards = json.load(f)

random_episodes = list(range(len(random_rewards)))

# ===== Optional smoothing (moving average) =====
def smooth(x, w=50):
    return np.convolve(x, np.ones(w)/w, mode='valid')

dqn_smooth = smooth(dqn_rewards, 50)
rand_smooth = smooth(random_rewards, 50)

# ===== Plot =====
plt.figure(figsize=(12,6))
plt.plot(dqn_episodes[:len(dqn_smooth)], dqn_smooth, label="DQN Agent", linewidth=2)
plt.plot(random_episodes[:len(rand_smooth)], rand_smooth, label="Random Agent", linewidth=2)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Random Agent vs. DQN Agent — Training Rewards")
plt.legend()
plt.grid(True)
plt.show()