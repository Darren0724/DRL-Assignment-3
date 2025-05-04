import torch
import torch.nn as nn
import numpy as np
import cv2
from collections import deque

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

class Agent:
    def __init__(self):
        self.device = 'cpu'
        self.model = DuelingDQN(input_shape=(4, 84, 84), n_actions=12).to(self.device)
        self.model.load_state_dict(torch.load("model.pth", map_location=self.device, weights_only=True))
        self.model.eval()
        self.frame_stack = deque(maxlen=4)
        self.feature_size = 84
        print("Agent initialized and model loaded.")

    def _preprocess_observation(self, obs):
        # Ensure input is uint8 for OpenCV compatibility
        if obs.dtype == np.float32 or obs.dtype == np.float64:
            obs = (obs * 255).clip(0, 255).astype(np.uint8)
        obs = obs[31:217, 0:248]  # Crop
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
        obs = cv2.resize(obs, (self.feature_size, self.feature_size), interpolation=cv2.INTER_AREA)  # Resize
        obs = obs.astype(np.uint8)  # Ensure uint8 for Canny
        obs = cv2.Canny(obs, 100, 200)  # Apply Canny edge detection
        obs = obs.astype(np.float32) / 255.0  # Normalize
        return obs

    def act(self, observation):
        # Process raw observation (240, 256, 3) to (84, 84)
        processed_obs = self._preprocess_observation(observation)
        # Append to frame stack
        self.frame_stack.append(processed_obs)
        # Repeat first frame if stack is not full
        while len(self.frame_stack) < 4:
            self.frame_stack.append(processed_obs)
        # Stack frames to (4, 84, 84)
        stacked_frames = np.stack(self.frame_stack, axis=0)
        # Convert to tensor with batch dimension
        observation = torch.tensor(stacked_frames, device=self.device).float().unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(observation)
        return torch.argmax(q_values[0]).item()

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        print(f"Model loaded from {model_path}")