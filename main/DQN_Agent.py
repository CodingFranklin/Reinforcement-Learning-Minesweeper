import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from Env_Minesweeper import MinesweeperEnv
from Settings import ROWS, COLS

# ===== Q-Network =====
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ROWS * COLS, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, ROWS * COLS)   # one Q-value per tile
        )

    def forward(self, x):
        return self.net(x)

# ===== Hyperparameters =====
EPISODES = 500
MAX_STEPS = 400
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
REPLAY_SIZE = 50000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.999


dqn_rewards = []
episodes_list = []    


# ===== Setup =====
env = MinesweeperEnv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
replay = deque(maxlen=REPLAY_SIZE)

eps = EPS_START

def select_action(state):
    global eps
    if random.random() < eps:
        return random.randint(0, ROWS * COLS - 1)
    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = policy_net(state_t)
    return int(torch.argmax(q_values))

def replay_train_step():
    if len(replay) < BATCH_SIZE:
        return

    batch = random.sample(replay, BATCH_SIZE)

    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    actions = torch.tensor(np.array(actions), dtype=torch.int64, device=device)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.bool, device=device)

    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0]
        target = rewards + GAMMA * next_q * (~dones)

    loss = nn.MSELoss()(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ===== Training Loop =====
for episode in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0

    for step in range(MAX_STEPS):
        action = select_action(state)
        next_state, reward, done, info = env.step(action)

        replay.append((state, action, reward, next_state, done))
        replay_train_step()

        state = next_state
        total_reward += reward

        if done:
            break

    eps = max(EPS_END, eps * EPS_DECAY)

    # update target network every 20 episodes
    if episode % 20 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if episode % 20 == 0:
        print(f"Ep {episode} | Reward: {total_reward:.2f} | eps={eps:.3f}")
        dqn_rewards.append(total_reward)
        episodes_list.append(episode)

print("Training finished.")


import json
with open("dqn_rewards.json", "w") as f:
    json.dump({"episodes": episodes_list, "rewards": dqn_rewards}, f)