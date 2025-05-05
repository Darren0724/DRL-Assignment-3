import gym
import numpy as np
import cv2
import torch
from collections import deque
import random
import torch.nn as nn
import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
FEATURE_SIZE = 84

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
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

class CustomObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(CustomObservationWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(FEATURE_SIZE, FEATURE_SIZE), dtype=np.float32)

    def observation(self, obs):
        obs = obs[31:217, 0:248]  
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  
        obs = cv2.resize(obs, (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_AREA)  
        obs = cv2.Canny(obs, 100, 200)  
        obs = obs.astype(np.float32) / 255.0  
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        max_frame = np.max(self._obs_buffer, axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        obs = self.env.reset()
        self._obs_buffer[0] = obs
        self._obs_buffer[1] = obs
        return obs


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

STEP = 92

class TestAgent:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.device = 'cpu'  
        self.model = DuelingDQN(state_space, action_space).to(self.device)
        self.model.load_state_dict(torch.load("model.pth", map_location=self.device))
        self.model.eval()

    def act(self, state):
        state = torch.tensor(np.array(state), device=self.device).float().unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values[0]).item()


# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self, dqn=None):
        self.action_space = gym.spaces.Discrete(12)
        
        self.obs_stack = deque(maxlen=4)
        self.steps = 0
        self._obs_buffer = np.zeros((2, 84, 84), dtype=np.uint8)
        self.last_action = None
        self.current_action = None
        self.last_obs = None
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        env = MaxAndSkipEnv(env, skip=4)
        env = CustomObservationWrapper(env)
        env = FrameStack(env, num_stack=4)
        self.env = env 
        self.state = self.env.reset()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dqn is not None:
            self.dqn = dqn
        else:
            self.dqn = DQN((4, 84, 84), self.action_space.n)
            self.dqn.load_state_dict(torch.load("model.pth", weights_only=False, map_location=self.device))
            self.dqn.eval()
            self.dqn.to(self.device)
        
    def get_action(self, state):
        if self.steps == 0:
            self.state, reward, done, info = self.env.step(6) 
        if self.steps <= STEP:
            self.current_action = 6
            return 6
        state = torch.tensor(np.array(state), device=self.device).float().unsqueeze(0)
        with torch.no_grad():
            q_values = self.dqn(state)
        return torch.argmax(q_values[0]).item()
        
        
    def act(self, obs):
        if self.steps % 4 == 0:
            action = self.get_action(self.state)
            self.state, reward, done, info = self.env.step(action) 
            #self.env.render()
            #print(self.state.shape)
            self.current_action = action
        self.steps += 1
        return self.current_action