import time
import gymnasium as gym
import numpy as np
import optuna
from gymnasium import spaces
import readData
from gameInput import KeyboardInput
from stable_baselines3 import PPO,DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv,VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import torch
from ImageProcessing import ImageProcessing
import os



class SaveModel(BaseCallback):
    def __init__(self,path,verbose=0):
        super(SaveModel,self).__init__(verbose=0)
        self.save_path=path

    def _on_rollout_end(self) -> None:
        pass

    def _on_step(self) -> bool:
        if self.n_calls % 10000==0:
            self.model.save(self.save_path)
            print(self.n_calls," steps finished")
        return True



class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        self.dataReader=readData.gameData()

        self.action_space=spaces.Discrete(5)
        self.action_key=['up','left','down','right','t']

        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(224, 224, 1), dtype=np.uint8)

        self.t=time.perf_counter()
        self.KeyboardInput=KeyboardInput()
        self.last_obs=np.zeros((224,224,1))

    # except keyboard input time, on average take 0.04s to execute another step
    # create a keyboard input thread to avoid long time blocking(x)-> observation and reward should be done after action.
    # dynamic moving distance -> introducing extra output to indicate the time gap of keyboard press and release
    def test_time(self):
        ti=time.perf_counter()-self.t
        self.t=time.perf_counter()
        print(ti)

    def step(self, action):
        #self.test_time()
        self.KeyboardInput.space()
        self.KeyboardInput.input(self.action_key[action])
        observation=self.dataReader.grabScreen()
        observation=ImageProcessing(observation)

        reward=self.reward(action)
        terminated = True if self.dataReader.lastData[1] == 0.0 else False
        truncated =False
        info={}
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.dataReader.lastData = [0.0, 1500.0, 0.0]
        while True:
            time.sleep(0.5)
            self.KeyboardInput.input("space")
            if self.dataReader.getAllData()[1] ==1500.0:
                return np.zeros(((224, 224,1))), {}


    def render(self):
        pass

    def close(self):
        pass

    def reward(self,action):
        allData = self.dataReader.getAllData()
        position=self.dataReader.readPosition()


        # penalty = health loss
        health_penalty =  (allData[1] -self.dataReader.lastData[1])/100

        # reward = time incremental
        time_reward= allData[0] - self.dataReader.lastData[0]


        # If flash is usable while AI get health loss, penalty should be larger.
        #if self.dataReader.lastData[2] == 0.0:
        #    health_penalty *= 1.1

        # encourage agent stay in center area
        edge_reward=0
        if position[0]<=930 and position[0]>=150 and position[1]<=736 and position[1]>=108:
            edge_reward=4

        # if agent stay in edge area, it should not go edge area further.
        edge_penalty=0
        if (position[0] >= 1040 and (action%4==3)) or (position[0] <= 100 and action%4==1) or (position[1] >= 780 and action%4==2) or (position[1] <= 50 and action%4==0):
            edge_penalty=-3

        # game is reseting, should not compute reward
        if allData[1] == 0.0 and self.dataReader.lastData[1] == 1500.0:
              return 0

        self.dataReader.lastData = allData
        total=health_penalty+time_reward+edge_penalty
        return total



def optimize_ppo(trial):
    return {
        'n_steps':trial.suggest_int('n_steps',2048,8192),
    'gamma':trial.suggest_loguniform('gamma',0.8,0.9999),
    'learning_rate':trial.suggest_loguniform('learning_rate',1e-5,1e-4),
    'clip_range':trial.suggest_uniform('clip_range',0.1,0.4),
    'gae_lambda':trial.suggest_uniform('gae_lambda',0.8,0.99),
    }

def optimize_agent(trial):
    model_params=optimize_ppo(trial)

    env = CustomEnv()
    env=Monitor(env,"./log")
    env=DummyVecEnv([lambda : env])
    env=VecFrameStack(env,5,channels_order='last')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PPO("CnnPolicy", env, verbose=0,tensorboard_log="./log",device=device,**model_params)
    path="./models/"+str(trial.number)+'.zip'
    callback = SaveModel(path)
    model.learn(total_timesteps=100000,callback=callback)

    mean_reward, _=evaluate_policy(model,env,n_eval_episodes=100)
    return mean_reward


#env = DummyVecEnv([lambda: env])

study=optuna.create_study(direction='maximize')
study.optimize(optimize_agent,n_trials=50,n_jobs=1)
print(study.best_params)

#env=VecFrameStack(env,n_stack=3)
