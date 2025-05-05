import gym
import torch
import torch.nn as nn
import numpy as np
import cv2
from collections import deque
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

FEATURE_SIZE = 84
STACK_SIZE = 4
SKIP_FRAMES = 4

class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc_value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        value = self.fc_value(conv_out)
        advantage = self.fc_advantage(conv_out)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

def preprocess(obs):
    obs = obs[31:217, 0:248]
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_AREA)
    obs = cv2.Canny(obs, 100, 200)
    obs = obs.astype(np.float32) / 255.0
    return obs

class Agent(object):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(len(COMPLEX_MOVEMENT))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DuelingDQN((STACK_SIZE, FEATURE_SIZE, FEATURE_SIZE), self.action_space.n).to(self.device)
        self.model.load_state_dict(torch.load("model.pth", map_location=self.device))
        self.model.eval()

        self.frame_stack = deque(maxlen=STACK_SIZE)
        for _ in range(STACK_SIZE):
            self.frame_stack.append(np.zeros((FEATURE_SIZE, FEATURE_SIZE), dtype=np.float32))

        self.skip_count = 0
        self.last_action = 0
        self.epsilon = 0.1  # 10% 機率執行隨機行為

    def act(self, observation):
        processed = preprocess(observation)
        self.frame_stack.append(processed)

        if self.skip_count > 0:
            self.skip_count -= 1
            return self.last_action

        if np.random.rand() < self.epsilon:
            action = self.action_space.sample()
        else:
            state = np.array(self.frame_stack)
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state)
                action = torch.argmax(q_values[0]).item()

        self.last_action = action
        self.skip_count = SKIP_FRAMES - 1
        return action
