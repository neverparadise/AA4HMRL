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
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.linalg import matrix_norm
from tensordict.tensordict import TensorDict
from modules.iql_agent import IQLAgent
from modules.buffers.replay_buffer import ReplayBuffer
import d4rl
from modules.utils import set_seed
from itertools import combinations
import logging
import os
import seaborn as sns
import matplotlib.pyplot as plt
from gym.spaces import Box as GymBox
from gym.spaces import Discrete as GymDiscrete
from gymnasium.spaces import Box as GymnasiumBox
from gymnasium.spaces import Discrete as GymnasiumDiscrete
import warnings
from typing import TypeVar, List, Tuple, Dict, Union
GymSpace = TypeVar('GymSpace', GymBox, GymDiscrete, GymnasiumBox, GymnasiumDiscrete)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env

@torch.no_grad()
def eval_actor(
    env: gym.Env, 
    agent: nn.Module, 
    device: str, 
    n_episodes: int, 
    seed: int,
    is_distributed: bool,
    rank: int = 0,
) -> np.ndarray:
    env.seed(seed)
    agent.eval()
    episode_rewards = []
    for e in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        timestep=1
        while not done:
            rollout_start_time = time.time()
            if is_distributed:
                action = agent.module.act(env, state, device)
            else:
                action = agent.act(env, state, device)
            state, reward, done, _ = env.step(action)
            rollout_end_time = time.time() - rollout_start_time
            if timestep % 100 == 0:
                print(f"Rank: {rank}. Env: {env.env_id}. Eval episode: {e}. Timestep: {timestep}. Rollout step time:{int(rollout_end_time)}")
            episode_reward += reward
            timestep += 1
        episode_rewards.append(episode_reward)
        print(f"Rank: {rank}. Env: {env.env_id}. Eval episode: {e}: {episode_reward}")

    agent.train()
    return np.asarray(episode_rewards)

def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)

def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


class DDIQLTrainer:
    def __init__(self, cfg: OmegaConf) -> None:
        self.cfg = cfg
        exp_cfg = cfg.experiment
        
        # * directory of experiment log and files
        self.output_dir = str(Path(cfg.paths.dir))
        self.ckpt_dir = Path(cfg.paths.checkpoints)
        self.ckpt_dir.mkdir(exist_ok=True, parents=False)
        shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "modules"), dst=self.output_dir+"/modules", dirs_exist_ok=True)
        
        if cfg.experiment.resume:
            self.load_checkpoint()
        
        # * record some information on config
        OmegaConf.set_struct(cfg, True)
        batch_size = cfg.iql.batch_size
        
        # * logger
        file_handler = logging.FileHandler(Path(cfg.paths.dir) / 'log.txt')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # * create ranked env ids
        self.env_ids = list(exp_cfg.env_ids)
        
        # * divide environments for distributed training
        self.is_distributed = cfg.distributed.world_size > 1 and cfg.distributed.multiprocessing_distributed
        if self.is_distributed:
            self.ranked_env_ids = []
            divided_ids = []
            for i, env_id in enumerate(self.env_ids):
                divided_ids.append(env_id)
                if len(divided_ids) == int(len(self.env_ids) // cfg.distributed.world_size):
                    self.ranked_env_ids.append(divided_ids)
                    divided_ids = []
            self.ranked_env_ids.append(divided_ids)
            print(self.ranked_env_ids)
            
        # * set seed, deivce, torch cudnn deterministic
        if exp_cfg.seed is not None:
            seed = exp_cfg.seed
            self.eval_seed = cfg.evaluation.eval_seed
            set_seed(seed)
        
        self.use_compile = cfg.nn.actor_critic.use_compile
        if cfg.nn.actor_critic.encoder_net_1d == 's4' or \
            cfg.nn.actor_critic.decoder_net == 's4':
                self.use_compile = False # S4 does not support torch compile
        
    def run(self, wandb_logger):
        # * tensorboard
        rank = 0
        cfg = self.cfg
        exp_cfg = cfg.experiment
        self.writer = SummaryWriter(str(Path(cfg.paths.log)))
        device = exp_cfg.device
        # * set environments, dataset_dict
        train_different_env_list = []
        env_statistics = dict() # key: env_id, value: (mean, std)
        replay_buffer_dict = dict()
        for j, env_id in enumerate(self.env_ids):
            train_env = gym.make(env_id)
            train_env.env_id = env_id

            dataset = d4rl.qlearning_dataset(train_env)
            print(f"{j+1}/{len(self.env_ids)} environment {env_id} is loaded...")
            if self.cfg.iql.norm_reward:
                modify_reward(dataset, env_id)
            if self.cfg.iql.norm_obs:
                state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
                env_statistics[env_id] = state_mean, state_std
            else:
                state_mean = 0, 
                state_std = 1
                env_statistics[env_id] = state_mean, state_std
            dataset["observations"] = normalize_states(
                                            dataset["observations"], state_mean, state_std
                                            )
            dataset["next_observations"] = normalize_states(
                                            dataset["next_observations"], state_mean, state_std
                                        )
            train_env = wrap_env(train_env, state_mean=state_mean, state_std=state_std)
            train_different_env_list.append(train_env)
            # TODO: image, discrete action 고려하기
            obs_dim = np.prod(train_env.observation_space.shape)
            act_dim = np.prod(train_env.action_space.shape)
            replay_buffer = ReplayBuffer(
                obs_dim, 
                act_dim,
                self.cfg.iql.buffer_size,
                device,
            )
            replay_buffer.load_d4rl_dataset(dataset)
            set_seed(self.cfg.experiment.seed)
            replay_buffer_dict[env_id] = replay_buffer
        
        # * set agent
        agent = IQLAgent(cfg, self.env_ids, train_different_env_list).to(device)
        total_num_params = sum([np.prod(p.size()) for p in agent.parameters()])
        agent = torch.compile(agent, 
                              mode="reduce-overhead",
                              disable=not self.use_compile)
        print(f"train total_num_params: {total_num_params}")
        with open(self.output_dir+'/total_num_params.txt', mode='w') as f:
            f.write(str(total_num_params))
        
        print("---------------------------------------")
        print(f"Training IQL, Env: {self.env_ids}, Seed: {cfg.experiment.seed}")
        print("---------------------------------------")

        env_evaluations = {env_id: [] for env_id in self.env_ids}
        for t in range(int(cfg.experiment.total_timesteps)):
            total_env_loss = 0.0
            for env in train_different_env_list:
                if t % 100 == 0:
                    print(f"Time steps: {t + 1}, env: {env.env_id}")
                env_id = env.env_id
                batch = replay_buffer_dict[env_id].sample(cfg.iql.batch_size, device)
                batch = [b.to(device) for b in batch]
                per_env_sum_loss, log_dict = agent.compute_loss(env, batch)
                total_env_loss += per_env_sum_loss
                log_dict['losses/per_env_sum_loss'] = per_env_sum_loss
                wandb_logger.log(log_dict, step=t)
            log_dict['losses/total_env_loss'] = total_env_loss
            
            # ! Update agent
            agent.optim_zero_grad()
            total_env_loss.backward()
            agent.optim_step()
            
            # ! Update target network
            agent.update_target_net()
                
            if (t + 1) % cfg.evaluation.eval_freq == 0:
                for env in train_different_env_list:
                    eval_scores = eval_actor(
                        env,
                        agent,
                        device=device,
                        n_episodes=cfg.evaluation.n_episodes,
                        seed=exp_cfg.seed,
                        is_distributed=self.is_distributed,
                        rank=rank,
                    )
                    eval_score = eval_scores.mean()
                    normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
                    env_evaluations[env_id].append(normalized_eval_score)
                    print("---------------------------------------")
                    print(
                        f"{env_id}: Evaluation over {cfg.evaluation.n_episodes} episodes: "
                        f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
                    )
                    if exp_cfg.save_ckpt:
                        torch.save(
                            agent.get_state_dict(t),
                            os.path.join(self.ckpt_dir, f"checkpoint_{t}.pt"),
                        )
                    print("---------------------------------------")
                    wandb_logger.log(
                            {f"{env_id}/d4rl_normalized_score": normalized_eval_score}, step=t
                        )
                
        for envs in train_different_env_list:
            env.close()
        self.finish()
    
    def setup(self, rank, cfg):
        os.environ['MASTER_ADDR'] = 'localhost'
        # port = random.randint(10000, 13000)
        os.environ['MASTER_PORT'] = str(cfg.distributed.port) # '12358'
        # os.environ['MASTER_PORT'] = str(port)
        world_size = cfg.distributed.world_size
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.distributed.barrier()  
    
    def distributed_run(self, rank, wandb_logger=None, *args,):
        cfg = self.cfg
        exp_cfg = cfg.experiment
        self.setup(rank, cfg)
        local_gpu_id = int(cfg.distributed.device_ids[rank])
        print(f"rank: {rank}. local_gpu_id: {local_gpu_id}")
        device = torch.device(f"cuda:{local_gpu_id}" \
                     if torch.cuda.is_available() and exp_cfg.cuda else "cpu")
        torch.cuda.set_device(local_gpu_id)
        torch.backends.cudnn.deterministic = exp_cfg.torch_deterministic
        print(f"device: {device}")
        
        sub_env_ids = self.ranked_env_ids[rank]
        print(f"rank: {rank}. sub_env_ids: {sub_env_ids}")
        
        
        # # * set agent
        all_different_env_list = []
        for j, env_id in enumerate(self.env_ids):
            train_env = gym.make(env_id)
            all_different_env_list.append(train_env)
        agent = IQLAgent(cfg, self.env_ids, all_different_env_list).to(device)
        agent = torch.compile(agent, 
                              mode="reduce-overhead",
                              disable=not self.use_compile)
        total_num_params = sum([np.prod(p.size()) for p in agent.parameters()])
        del all_different_env_list
        print(f"train total_num_params: {total_num_params}")
        with open(self.output_dir+'/total_num_params.txt', mode='w') as f:
            f.write(str(total_num_params))
        agent = DDP(agent, device_ids=[device], output_device=device)
        print(f"rank: {rank}. agent is initialized")

        train_different_env_list = []
        env_statistics = dict() # key: env_id, value: (mean, std)
        replay_buffer_dict = dict()

        # * set vector environments
        for j, env_id in enumerate(sub_env_ids):
            train_env = gym.make(env_id)
            train_env.env_id = env_id
            dataset = d4rl.qlearning_dataset(train_env)
            print(f"{j+1}/{len(sub_env_ids)} environment {env_id} is loaded...")
            if self.cfg.iql.norm_reward:
                modify_reward(dataset, env_id)
            if self.cfg.iql.norm_obs:
                state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
                env_statistics[env_id] = state_mean, state_std
            else:
                state_mean = 0, 
                state_std = 1
                env_statistics[env_id] = state_mean, state_std
            dataset["observations"] = normalize_states(
                                            dataset["observations"], state_mean, state_std
                                            )
            dataset["next_observations"] = normalize_states(
                                            dataset["next_observations"], state_mean, state_std
                                        )
            train_env = wrap_env(train_env, state_mean=state_mean, state_std=state_std)
            train_different_env_list.append(train_env)
            # TODO: image, discrete action 고려하기
            obs_dim = np.prod(train_env.observation_space.shape)
            act_dim = np.prod(train_env.action_space.shape)
            replay_buffer = ReplayBuffer(
                obs_dim, 
                act_dim,
                self.cfg.iql.buffer_size,
                device,
            )
            replay_buffer.load_d4rl_dataset(dataset)
            set_seed(self.cfg.experiment.seed)
            replay_buffer_dict[env_id] = replay_buffer
                                    
        print(f"rank: {rank}. replay buffers are initialized")
        print("---------------------------------------")
        print(f"Training IQL, Env: {self.env_ids}, Seed: {cfg.experiment.seed}")
        print("---------------------------------------")
        env_evaluations = {env_id: [] for env_id in self.env_ids}
        for t in range(int(cfg.experiment.total_timesteps)):
            dist.barrier()
            total_env_loss = 0.0
            for env in train_different_env_list:
                if t % 100 == 0:
                    print(f"Rank: {rank}. Time steps: {t + 1}, env: {env.env_id}")
                env_id = env.env_id
                batch = replay_buffer_dict[env_id].sample(cfg.iql.batch_size, device)
                batch = [b.to(device) for b in batch]
                per_env_sum_loss, log_dict = agent.module.compute_loss(env, batch)
                total_env_loss += per_env_sum_loss
                log_dict['losses/per_env_sum_loss'] = per_env_sum_loss
                wandb_logger.log(log_dict, step=t)
            log_dict['losses/total_env_loss'] = total_env_loss
            
            # ! Update agent
            agent.module.optim_zero_grad()
            total_env_loss.backward()
            agent.module.optim_step()
            
            # ! Update target network
            agent.module.update_target_net()

            # * evaluate agent
            if (t + 1) % cfg.evaluation.eval_freq == 0:
                for env in train_different_env_list:
                    eval_scores = eval_actor(
                        env,
                        agent,
                        device=device,
                        n_episodes=cfg.evaluation.n_episodes,
                        seed=exp_cfg.seed,
                        is_distributed=self.is_distributed,
                        rank=rank,
                    )
                    eval_score = eval_scores.mean()
                    normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
                    env_evaluations[env_id].append(normalized_eval_score)
                    print("---------------------------------------")
                    print(
                        f"Rank: {rank}. {env_id}: Evaluation over {cfg.evaluation.n_episodes} episodes: "
                        f"Rank: {rank}. {eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
                    )
                    if exp_cfg.save_ckpt and rank==0:
                        torch.save(
                            agent.module.get_state_dict(t),
                            os.path.join(self.ckpt_dir, f"checkpoint.pt"),
                        )
                    print("---------------------------------------")
                    wandb_logger.log(
                            {f"{env_id}/d4rl_normalized_score": normalized_eval_score}, step=t
                        )

        for envs in train_different_env_list:
            env.close()
        if rank == 0:
            self.finish()
    
    def save_checkpoint(self, agent, update_idx: int) -> None:
        save_dir = (Path(self.ckpt_dir) / Path(str(update_idx)))
        save_dir.mkdir(exist_ok=True, parents=False)
        torch.save(update_idx, save_dir / 'epoch.pt')
        if self.is_distributed:
            torch.save({
                "agent": agent.module.state_dict(),
                }, save_dir / 'agent.pt')
        else:
                torch.save({
                "agent": agent.state_dict(),
                }, save_dir / 'agent.pt')
        # torch.save({
        #         "agent_optimizer": self.agent_optimizer.state_dict(),
        #     }, save_dir / 'optimizer.pt')
        logger.info(f"ckpt is successfully saved to {save_dir}") 
    
    def load_checkpoint(self) -> None:
        exp_cfg = self.cfg.experiment
        ckpt_str = exp_cfg.resume_dir + '/checkpoints/' + str(exp_cfg.resume_update_idx)
        ckpt_dir = Path(ckpt_str)
        self.start_update_idx = torch.load(ckpt_dir / 'epoch.pt') + 1
        agent_state_dict = torch.load(ckpt_dir / 'agent.pt', map_location=self.device)
        self.agent.load_state_dict(agent_state_dict['agent'])
        # ckpt_opt = torch.load(ckpt_dir / 'optimizer.pt', map_location=self.device)
        # self.agent_optimizer.load_state_dict(ckpt_opt['agent_optimizer'])
        # logger.info(f'Successfully loaded models, optimizers from {ckpt_dir}.')
        
    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: batch[k].to(self.device) for k in batch}
                                                        
    def finish(self) -> None:
        wandb.finish()
        