import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from Env_Minesweeper import MinesweeperEnv
from Settings import ROWS, COLS
import json

# ===== Q-Network =====
NUM_ACTIONS = ROWS * COLS * 2
GAMMA = 0.99
BATCH_SIZE = 64
LR = 1e-3

NUM_EPISODES = 3000
MAX_STEPS = 100

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_EPISODES = 1000

TARGET_UPDATE = 50

REPLAY_CAPACITY = 50000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            nn.Linear(256, NUM_ACTIONS),
        )

    def forward(self, x):
        # x: (batch, ROWS, COLS)
        return self.net(x)


# ===== 工具：epsilon-greedy 选动作 =====
def linear_eps(episode: int) -> float:
    if episode >= EPS_DECAY_EPISODES:
        return EPS_END
    ratio = episode / EPS_DECAY_EPISODES
    return EPS_START + (EPS_END - EPS_START) * ratio


# replay buffer: (state, action, reward, next_state, done)
replay = deque(maxlen=REPLAY_CAPACITY)

env = MinesweeperEnv()
policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
loss_fn = nn.MSELoss()


def select_action(state, eps: float) -> int:
    """
    state: (ROWS, COLS) numpy array
    """
    if random.random() < eps:
        return random.randrange(NUM_ACTIONS)

    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = policy_net(state_t)  # (1, NUM_ACTIONS)
    action = int(torch.argmax(q_values).item())
    return action


def replay_train_step():
    if len(replay) < BATCH_SIZE:
        return

    batch = random.sample(replay, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32, device=device)          # (B, R, C)
    actions = torch.tensor(np.array(actions), dtype=torch.int64, device=device)          # (B,)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=device)        # (B,)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)# (B, R, C)
    dones = torch.tensor(np.array(dones), dtype=torch.bool, device=device)               # (B,)

    #  Q(s,a)
    q_all = policy_net(states)                     # (B, NUM_ACTIONS)
    q_values = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

    #  target = r + gamma * max_a' Q_target(s', a')
    with torch.no_grad():
        next_q_all = target_net(next_states)       # (B, NUM_ACTIONS)
        next_q_max, _ = next_q_all.max(dim=1)      # (B,)
        target = rewards + GAMMA * next_q_max * (~dones)

    loss = loss_fn(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


episodes_list = []
dqn_rewards = []

for episode in range(NUM_EPISODES):
    state = env.reset()
    done = False
    total_reward = 0.0

    eps = linear_eps(episode)

    for step in range(MAX_STEPS):
        action = select_action(state, eps)
        next_state, reward, done, _ = env.step(action)

        replay.append((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward

        replay_train_step()

        if done:
            break

    if (episode + 1) % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if episode % 20 == 0:
        print(f"Ep {episode:4d} | Reward: {total_reward:7.2f} | eps={eps:.3f}")
        dqn_rewards.append(total_reward)
        episodes_list.append(episode)

print("Training finished.")

with open("../dqn_rewards_flag.json", "w") as f:
    json.dump({"episodes": episodes_list, "rewards": dqn_rewards}, f)