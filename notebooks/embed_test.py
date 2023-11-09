from pathlib import Path
import shutil
import time
from typing import Dict
import random
import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf, open_dict
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from tqdm import tqdm
import wandb
import pprint
# import gymnasium as gym
import gym
import numpy as np
import torch
from torch.linalg import matrix_norm
from tensordict.tensordict import TensorDict
from modules.ppo_agent import PPOAgent
from modules.utils import set_seed, make_envpool_env
from itertools import combinations
import logging
import os
import seaborn as sns
import matplotlib.pyplot as plt
from gym.spaces import Box as GymBox
from gym.spaces import Discrete as GymDiscrete
from gymnasium.spaces import Box as GymnasiumBox
from gymnasium.spaces import Discrete as GymnasiumDiscrete


from omegaconf import OmegaConf
cfg_path = "/home/kukjin/Projects/HMRL_final/configs/ddppo_trainer.yaml"
nn_cfg_path = "/home/kukjin/Projects/HMRL_final/configs/nn/ppo_nn.yaml"
ppo_cfg_path = "/home/kukjin/Projects/HMRL_final/configs/ppo/ppo.yaml"
cfg = OmegaConf.load(cfg_path)
nn_cfg = OmegaConf.load(nn_cfg_path)
ppo_cfg = OmegaConf.load(ppo_cfg_path)
cfg.nn = nn_cfg
cfg.ppo = ppo_cfg

pretraining_env_ids = cfg.experiment.pretraining_env_ids
finetuning_env_ids = cfg.experiment.finetuning_env_ids

all_pretraining_env_list = []
for j, env_id in enumerate(pretraining_env_ids):
    train_envs = make_envpool_env(j, env_id, cfg)
    all_pretraining_env_list.append(train_envs)

agent = PPOAgent(cfg, 
                 pretraining_env_ids,
                 all_pretraining_env_list,
                 mode='pretraining')


all_finetuning_env_list = []
for j, env_id in enumerate(finetuning_env_ids):
    train_envs = make_envpool_env(j, env_id, cfg)
    all_finetuning_env_list.append(train_envs)

agent.set_finetuning_envs(finetuning_env_ids,
                          all_finetuning_env_list)