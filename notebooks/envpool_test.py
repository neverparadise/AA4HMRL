import envpool
# import gymnasium as gym
import gym

import argparse
import os
import random
import time
from collections import deque
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

num_envs = 96
batch_size = 48

envs = envpool.make(
    "HalfCheetah-v4",
    env_type="gym",
    num_envs=num_envs,
    batch_size=batch_size,
    # num_threads=16,
    seed=42,
)
envs.batch_size = batch_size
class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        print(f"env type: {type(env)}")
        
        self.batch_size = getattr(env, "batch_size", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        print(f"kwargs: {kwargs}")
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.batch_size, dtype=np.float32)
        self.episode_lengths = np.zeros(self.batch_size, dtype=np.int32)
        self.lives = np.zeros(self.batch_size, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.batch_size, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.batch_size, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )
        
envs = RecordEpisodeStatistics(envs)
envs.async_reset()

total_rewards = 0
for i in range(10):
    obs, rew, done, info = envs.recv()
    env_id = info["env_id"]
    print(done)
    print(info)
    # action = np.random.randn(batch_size, 6)
    action = np.array([envs.action_space.sample() for i in range(batch_size)])
    # print(action.shape)
    envs.send(action, env_id)
    total_rewards += rew
    # next_obs, reward, done, info = envs.step(action)
print(total_rewards)