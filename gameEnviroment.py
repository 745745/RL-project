import time
from collections import deque
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import readData
from gameInput import KeyboardInput
from stable_baselines3 import PPO,DQN
import torch
import torch.nn as nn
from torchvision import models

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor



class FeatureNetwork(BaseFeaturesExtractor):
    def __init__(self,observation_space: spaces.Dict, features_dim: int = 4097):
        super().__init__(observation_space, features_dim)
        self.output_size=8
        self.vgg16_model = models.vgg16(pretrained=True)
        self.vgg16_model.classifier = self.vgg16_model.classifier[:-1]
        self.vgg16_model.requires_grad_(False)
        self.relu=nn.ReLU()
        self.nn=nn.Conv1d(in_channels=5,out_channels=1,kernel_size=1)

    def forward(self,observation):
        img=observation['img']
        img=img.squeeze(dim=0)
        if(img.dim()==5):
            flash = torch.argmax(observation['flash'], dim=2,keepdim=True)
            shape=img.shape
            img=img.view((shape[0]*shape[4],shape[3],shape[2],shape[1]))
            feature=self.vgg16_model(img)
            feature = feature.view((shape[0], 5, -1))
            feature=self.nn(feature)

            feature=torch.concatenate((feature,flash),dim=2)
        else:
            flash = torch.argmax(observation['flash'], keepdim=True)
            img=img.permute(3,2,0,1)
            feature=self.vgg16_model(img)
            feature=self.relu(feature)
            feature=self.nn(feature)
            feature=torch.concatenate((feature,flash),dim=1)
        return feature


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        self.dataReader=readData.gameData()

        self.action_space = spaces.Discrete(8)
        self.action_key=['UP','LEFT','DOWN','RIGHT','up+d','left+d','down+d','right+d']

        self.observation_space = spaces.Dict({'img':spaces.Box(low=0, high=255,
                                            shape=(224, 224, 3,5), dtype=np.uint8),
                                              'flash':spaces.Discrete(2)})

        self.frames=deque(maxlen=5)
        for i in range(5):
            self.frames.append(np.zeros((224, 224,3,1)))

    def step(self, action):
        if action==3:
            KeyboardInput(self.action_key[action],timegap=0)
        KeyboardInput(self.action_key[action])
        observation=self.dataReader.grabScreen(mode="VGG")
        self.frames.append(observation)
        observation=np.array(self.frames)
        observation=observation.squeeze(axis=4)
        observation=np.transpose(observation,(1,2,3,0))
        if self.dataReader.getAllData()[2] != 0.0:
            flash=0
        else:
            flash=1
        observation={"img":observation,"flash":flash}
        reward=self.reward(action)
        terminated = True if self.dataReader.lastData[1] == 0.0 else False
        truncated =False
        info={}
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.dataReader.lastData = [0.0, 1500.0, 0.0]
        for i in range(5):
            self.frames.append(np.zeros((224, 224,3,1)))
        while True:
            time.sleep(0.5)
            KeyboardInput("space")
            if self.dataReader.getAllData()[1] ==1500.0:
                return {'img':self.dataReader.grabScreen(),'flash':1}, {}


    def render(self):
        pass

    def close(self):
        pass

    def reward(self,action):
        allData = self.dataReader.getAllData()
        position=self.dataReader.readPosition()
        KeyboardInput('space',timegap=0)

        # penalty = health loss
        health_penalty =  (allData[1] -self.dataReader.lastData[1])/100

        # reward = time incremental
        time_reward= allData[0] - self.dataReader.lastData[0]


        # If flash is usable while AI get health loss, penalty should be larger.
        if self.dataReader.lastData[2] == 0.0:
            health_penalty *= 1.1

        # encourage agent stay in center area
        edge_reward=0
        if position[0]<=930 and position[0]>=150 and position[1]<=736 and position[1]>=108:
            edge_reward=2

        # if agent stay in edge area, it should not head edge area further.
        edge_penalty=0
        if (position[0] >= 1040 and (action%4==3)) or (position[0] <= 100 and action%4==1) or (position[1] >= 780 and action%4==2) or (position[1] <= 50 and action%4==0):
            edge_penalty=-2


        # Should not use flash when flash is unusable
        flash_penalty=0
        if action >= 4 and self.dataReader.lastData[2] != 0.0:
            flash_penalty = -0.2

        # game is reseting, should not compute reward
        if allData[1] == 0.0 and self.dataReader.lastData[1] == 1500.0:
            return 0

        self.dataReader.lastData = allData
        total=health_penalty+flash_penalty+edge_penalty+time_reward+edge_reward
        print(total)
        return total





env = CustomEnv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_kwargs = dict(
    features_extractor_class=FeatureNetwork
)

model = PPO("MultiInputPolicy", env, verbose=1,policy_kwargs=policy_kwargs,learning_rate=0.005 ,device=device)
for i in range(100):
    model.learn(total_timesteps=5000)
    model.save("dodge")
