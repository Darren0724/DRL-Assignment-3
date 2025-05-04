import torch
import torch.nn as nn
import numpy as np

# Dueling DQN 網路結構，與 train.py 一致
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
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# 作業要求的 Agent 類
class Agent:
    def __init__(self):
        self.device = 'cpu'  # 作業要求在 CPU 上運行
        self.model = DuelingDQN(input_shape=(4, 84, 84), n_actions=12).to(self.device)
        self.model.load_state_dict(torch.load("model.pth", map_location=self.device))
        self.model.eval()  # 設置為評估模式

    def act(self, observation):
        observation = torch.tensor(np.array(observation), device=self.device).float().unsqueeze(0)  # 添加批次維度
        with torch.no_grad():
            q_values = self.model(observation)
        return torch.argmax(q_values[0]).item()