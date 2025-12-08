import json
import matplotlib.pyplot as plt
import numpy as np

# ===== Load DQN data =====
with open("../dqn_rewards.json", "r") as f:
    dqn_data = json.load(f)

dqn_episodes = dqn_data["episodes"]
dqn_rewards = dqn_data["rewards"]

# ===== Load DQN data with flag operations =====
with open("../dqn_rewards_flag.json", "r") as f:
    dqn_flag_data = json.load(f)

dqn_flag_episodes = dqn_flag_data["episodes"]
dqn_flag_rewards = dqn_flag_data["rewards"]

# ===== Load Random Agent data =====
with open("../random_rewards.json", "r") as f:
    random_rewards = json.load(f)

random_episodes = [i * 20 for i in range(len(random_rewards))]

# ===== Load Random Agent data with flag =====
with open("../random_rewards_flag.json", "r") as f:
    random_flag_rewards = json.load(f)


random_flag_episodes = [i * 20 for i in range(len(random_flag_rewards))]

# ===== Optional smoothing =====
def smooth(x, w=10):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w)/w, mode='valid')

dqn_smooth = smooth(dqn_rewards, 10)
rand_smooth = smooth(random_rewards, 3)
dqn_flag_smooth = smooth(dqn_flag_rewards, 10)
rand_flag_smooth = smooth(random_flag_rewards, 3)
# ===== Plot =====
plt.figure(figsize=(12,6))
plt.plot(dqn_episodes[:len(dqn_smooth)], dqn_smooth, label="DQN Agent", linewidth=2)
plt.plot(random_episodes[:len(rand_smooth)], rand_smooth, label="Random Agent", linewidth=2)
#plt.plot(dqn_flag_episodes[:len(dqn_flag_smooth)], dqn_flag_smooth, label="DQN Agent (Flag)", linewidth=2)
#plt.plot(random_flag_episodes[:len(rand_flag_smooth)], rand_flag_smooth, label="Random Agent (Flag)", linewidth=2)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Random Agent vs. DQN Agent — Training Rewards")
plt.legend()
plt.grid(True)
plt.show()