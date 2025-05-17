import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
import os
from datetime import datetime

# 超参数
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
LEARNING_RATE = 0.00025
NUM_EPISODES = 1000
RENDER = False
STACK_SIZE = 4
RESIZE_SIZE = (84, 84)

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 日志记录
log_dir = f"runs/dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir)

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 环境创建
env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
env = gym.wrappers.AtariPreprocessing(env, frame_skip=1, screen_size=84, grayscale_obs=True, scale_obs=False)
env = gym.wrappers.FrameStack(env, STACK_SIZE)
env.seed(42)
n_actions = env.action_space.n

# 图像预处理（不需要自己处理通道）
transform = T.Compose([
    T.ToTensor()
])


# DQN 网络
class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(STACK_SIZE, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.fc = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, outputs)

    def forward(self, x):
        x = x.float() / 255.0  # 归一化
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc(x))
        return self.head(x)


# 状态处理
def preprocess_state(state):
    state = np.array(state)  # shape: (4, 84, 84)
    state = torch.from_numpy(state).unsqueeze(0).to(device)  # shape: (1, 4, 84, 84)
    return state


# 经验回放
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states)
        actions = torch.tensor(actions, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.cat(next_states)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


# 网络初始化
dummy_obs = preprocess_state(env.reset()[0])
_, _, screen_height, screen_width = dummy_obs.shape

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(MEMORY_SIZE)


# 选择动作
def select_action(state, eps_threshold):
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


# 优化函数
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

    current_q = policy_net(states).gather(1, actions.unsqueeze(1))
    next_q = target_net(next_states).max(1)[0].detach()
    expected_q = rewards + (GAMMA * next_q * (1 - dones))

    loss = nn.MSELoss()(current_q.squeeze(), expected_q)

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss.item()


# 主训练循环
eps_threshold = EPS_START
episode_rewards = []

for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    state = preprocess_state(obs)
    total_reward = 0
    done = False

    while not done:
        if RENDER:
            env.render()

        action = select_action(state, eps_threshold)
        obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        next_state = preprocess_state(obs)

        memory.push(state, action.item(), reward, next_state, done)
        state = next_state
        total_reward += reward

        loss = optimize_model()
        if loss is not None:
            writer.add_scalar('Loss/train', loss, episode)

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    eps_threshold = max(EPS_END, eps_threshold * EPS_DECAY)

    episode_rewards.append(total_reward)
    writer.add_scalar('Reward/train', total_reward, episode)

    print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {eps_threshold:.2f}")

    if episode % 100 == 0:
        torch.save(policy_net.state_dict(), f"{log_dir}/dqn_{episode}.pth")

# 保存最终模型
torch.save(policy_net.state_dict(), f"{log_dir}/dqn_final.pth")
writer.close()
env.close()
