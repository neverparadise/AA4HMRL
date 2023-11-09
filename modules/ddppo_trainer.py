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
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
from gym.core import Env as GymEnv
from gymnasium.core import Env as GymnasiumEnv
import warnings
from typing import TypeVar, List, Tuple, Dict

GymSpace = TypeVar('GymSpace', GymBox, GymDiscrete, GymnasiumBox, GymnasiumDiscrete)
Env = TypeVar('Env', GymEnv, GymnasiumEnv)

warnings.filterwarnings("ignore", category=DeprecationWarning) 
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

    
class DDPPOTrainer:
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
        batch_size = int(exp_cfg.num_envs * exp_cfg.num_rollout_steps)
        minibatch_size = int(batch_size // cfg.ppo.num_minibatches)
        with open_dict(cfg):
            cfg.ppo.batch_size = batch_size
            cfg.ppo.minibatch_size = minibatch_size
            self.batch_size = int(exp_cfg.num_envs * exp_cfg.num_rollout_steps) #64 * 512
            self.minibatch_size = int(self.batch_size // cfg.ppo.num_minibatches) 
        print(f"train batch_size: {self.batch_size}")
        print(f"train minibatch_size: {self.minibatch_size}")
        
        # * logger
        file_handler = logging.FileHandler(Path(cfg.paths.dir) / 'log.txt')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # * load environment list
        self.pretraining_env_ids = list(exp_cfg.pretraining_env_ids)
        self.finetuning_env_ids = list(exp_cfg.finetuning_env_ids)
        self.from_scratch_env_ids= list(exp_cfg.finetuning_env_ids)
                
        # * create ranked env ids
        # * divide environments for distributed training
        self.is_distributed = cfg.distributed.world_size > 1 and cfg.distributed.multiprocessing_distributed
        if self.is_distributed:
            self.pre_ranked_env_ids = [] # ! List[List[Env]]
            divided_ids = []
            for i, env_id in enumerate(self.pretraining_env_ids):
                divided_ids.append(env_id)
                if len(divided_ids) == int(len(self.pretraining_env_ids) // cfg.distributed.world_size):
                    self.pre_ranked_env_ids.append(divided_ids)
                    divided_ids = []
            self.pre_ranked_env_ids.append(divided_ids)
            
            self.fine_ranked_env_ids = []
            self.from_scratch_env_ids = []
            divided_ids = []
            for i, env_id in enumerate(self.finetuning_env_ids):
                divided_ids.append(env_id)
                if len(divided_ids) == int(len(self.finetuning_env_ids) // cfg.distributed.world_size):
                    self.fine_ranked_env_ids.append(divided_ids)
                    self.from_scratch_env_ids.append(divided_ids)
                    divided_ids = []
            self.fine_ranked_env_ids.append(divided_ids)
            self.from_scratch_env_ids.append(divided_ids)
            
        # * set seed, deivce, torch cudnn deterministic
        if exp_cfg.seed is not None:
            seed = exp_cfg.seed
            self.eval_seed = cfg.evaluation.eval_seed
            set_seed(seed)
        
        self.use_compile = cfg.nn.actor_critic.use_compile
        if cfg.nn.actor_critic.encoder_net_1d == 's4' or \
            cfg.nn.actor_critic.decoder_net == 's4':
                self.use_compile = False # S4 does not support torch compile
                
        self.single_env_learning = cfg.experiment.single_env_learning
        
    def make_env_storages(self,
                          env_list: List,
                            env_ids: List,
                            device,
                          ):
        exp_cfg = self.cfg.experiment
        envs_storages = TensorDict({}, batch_size=[exp_cfg.num_rollout_steps, exp_cfg.num_envs])
        exp_cfg = self.cfg.experiment
        for i, env in enumerate(env_list):
            env_id = env_ids[i]
            obs = torch.zeros((exp_cfg.num_rollout_steps, exp_cfg.num_envs) \
                + env.single_observation_space.shape).to(device)
            actions = torch.zeros((exp_cfg.num_rollout_steps, exp_cfg.num_envs) \
                + env.single_action_space.shape).to(device)
            logprobs = torch.zeros((exp_cfg.num_rollout_steps, exp_cfg.num_envs)).to(device)
            rewards = torch.zeros((exp_cfg.num_rollout_steps, exp_cfg.num_envs)).to(device)
            dones = torch.zeros((exp_cfg.num_rollout_steps, exp_cfg.num_envs)).to(device)
            values = torch.zeros((exp_cfg.num_rollout_steps, exp_cfg.num_envs)).to(device)
            storage = TensorDict({
                        "obs": obs,
                        "actions": actions,
                        "logprobs": logprobs,
                        "rewards": rewards,
                        "dones": dones,
                        "values": values
                        }, batch_size=[exp_cfg.num_rollout_steps, exp_cfg.num_envs])
            envs_storages[env_id] = storage
        return envs_storages
    
    def train(self, 
              global_step,
              rank: int, 
              wandb_logger, 
              env_ids: List[str],
              env_list: List[Env],
              agent, 
              device, 
              mode='pretraining'):
        cfg = self.cfg
        exp_cfg = cfg.experiment
        start_time = time.time()
        total_num_updates = exp_cfg.total_timesteps // self.batch_size
        self.total_num_updates = total_num_updates
        
         # * set PPO storage for each environment
        envs_storages = self.make_env_storages(
                                env_list,
                                env_ids,
                                device,
                                )
        print(f"rank: {rank}. storage is initialized")
        print(f"rank: {rank}. keys: {envs_storages.keys()}")
        
        # * init env
        next_obs_dict = TensorDict({}, batch_size=exp_cfg.num_envs)
        next_done_dict = TensorDict({}, batch_size=exp_cfg.num_envs)
        for i, env in enumerate(env_list):
            env_id = env_ids[i]
            if exp_cfg.env_type == "gymnasium":
                next_obs, infos = env.reset()
            elif exp_cfg.env_type == "gym":
                next_obs = env.reset()
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.zeros(exp_cfg.num_envs).to(device)
            next_obs_dict[env_id] = next_obs
            next_done_dict[env_id] = next_done
        
        start_update_idx = 1 
        if self.cfg.experiment.save_ckpt:
            interval_size = int(total_num_updates/exp_cfg.num_checkpoints)
            intervals = np.arange(interval_size, total_num_updates+interval_size, interval_size)
            logger.info(f"total_num_updates: {total_num_updates}")
            logger.info(f"ckpt intervals: {intervals}")
            if self.is_distributed:
                agent.module.save_checkpoint(self.ckpt_dir, start_update_idx, mode=mode)
            else:
                agent.save_checkpoint(self.ckpt_dir, start_update_idx, mode=mode)
                
        if self.cfg.experiment.resume:
            start_update_idx = self.start_update_idx
        
        print(f"################# start {mode} ################# ")
        for update_idx in range(start_update_idx, total_num_updates + 1):
            if cfg.ppo.anneal_lr:
                frac = 1.0 - (update_idx - 1.0) / total_num_updates
                actor_lr_now = frac * cfg.ppo.actor_lr
                critic_lr_now = frac * cfg.ppo.critic_lr
                if self.is_distributed:
                    agent.module.actor_optimizer.param_groups[0]["lr"] = actor_lr_now
                    agent.module.critic_optimizer.param_groups[0]["lr"] = critic_lr_now
                else:
                    agent.actor_optimizer.param_groups[0]["lr"] = actor_lr_now
                    agent.critic_optimizer.param_groups[0]["lr"] = critic_lr_now

            envs_returns = dict()
            envs_lengths = dict()
            
            # ! rollout
            rollout = torch.compile(self.rollout, 
                              mode="reduce-overhead",
                              disable=not self.use_compile)
            rollout(rank,
                         agent, 
                        env_list, 
                        env_ids,
                        envs_storages,
                        next_obs_dict, 
                        next_done_dict, 
                        envs_returns, 
                        envs_lengths,
                        device,
                        update_idx,
                        global_step,
                        mode,)
            global_step += 1 * exp_cfg.num_envs * exp_cfg.num_rollout_steps
            wandb_logger.log(data=envs_returns, step=global_step)
            # wandb_logger.log(data=envs_lengths, step=global_step)
            
            # ! bootstrap value if not done
            calculate_advantages = torch.compile(self.calculate_advantages, 
                              mode="reduce-overhead",
                              disable=not self.use_compile)
            calculate_advantages(
                                    cfg,
                                    env_ids,
                                    env_list,
                                    agent,
                                    next_obs_dict,
                                    next_done_dict,
                                    envs_storages,
                                    device,
                                    )
                    
            # ! flatten the batch
            
            reshaped_storages = envs_storages.reshape(-1)
            
            # ! Optimizing the policy and value network
            # train_agent = torch.compile(self.train_agent, 
                            #   mode="reduce-overhead",
                            #   disable=not self.use_compile)
            loss_tuple = self.train_agent(reshaped_storages, agent, env_list, env_ids)
            total_envs_loss, total_value_loss, total_policy_loss, total_entropy_loss = loss_tuple
            if self.is_distributed:
                wandb_logger.log({
                f'{mode}/actor_lr': agent.module.actor_optimizer.param_groups[0]["lr"],
                f'{mode}/critic_lr': agent.module.critic_optimizer.param_groups[0]["lr"],
                f'{mode}/total_value_loss': total_value_loss,
                f'{mode}/total_policy_loss': total_policy_loss,
                f'{mode}/total_entropy_loss': total_entropy_loss,
                f'{mode}/total_envs_loss': total_envs_loss,
                    },step=global_step)
            else:
                wandb_logger.log({
                            f'{mode}/actor_lr': agent.actor_optimizer.param_groups[0]["lr"],
                            f'{mode}/critic_lr': agent.critic_optimizer.param_groups[0]["lr"],
                            f'{mode}/total_value_loss': total_value_loss,
                            f'{mode}/total_policy_loss': total_policy_loss,
                            f'{mode}/total_entropy_loss': total_entropy_loss,
                            f'{mode}/total_envs_loss': total_envs_loss,
                            },step=global_step)

            if self.cfg.experiment.save_ckpt and rank == 0:
                if update_idx in intervals:
                    logger.info(f"ckpt is saved: {update_idx}. dir: {self.ckpt_dir}")
                    if self.is_distributed:
                        agent.module.save_checkpoint(self.ckpt_dir, update_idx, mode=mode)
                    else:
                        agent.save_checkpoint(self.ckpt_dir, update_idx, mode=mode)
                
        for env in env_list:
            env.close()
        # self.finish()
        if rank == 0:
            if self.is_distributed:
                agent.module.save_checkpoint(self.ckpt_dir, update_idx, mode=mode)
            else:
                agent.save_checkpoint(self.ckpt_dir, update_idx, mode=mode)
                
    
    def run(self, wandb_logger):
        # ! #################
        # ! 1. Pretraining phase
        # ! #################
        print(self.pretraining_env_ids)

        rank = 0
        cfg = self.cfg
        exp_cfg = cfg.experiment
        self.writer = SummaryWriter(str(Path(cfg.paths.log)))
        device = exp_cfg.device
        device = torch.device(f"cuda:{device}")
        
        # * set environments
        pretraining_env_list = []
        for j, env_id in enumerate(self.pretraining_env_ids):
            env = make_envpool_env(j, env_id, cfg)
            pretraining_env_list.append(env)
            print(f"{j+1}/{len(self.pretraining_env_ids)} environment {env_id} is loaded...")
        
        print(f"pretraining env ids {self.pretraining_env_ids}")
        # * set agent
        agent = PPOAgent(cfg, 
                         self.pretraining_env_ids, 
                         pretraining_env_list,
                        mode='pretraining').to(device)
        pretraining_total_num_params = sum([np.prod(p.size()) for p in agent.parameters()])
        agent = torch.compile(agent, 
                              mode="reduce-overhead",
                              disable=not self.use_compile)
        print(f"pretraining total_num_params: {pretraining_total_num_params}")
        with open(self.output_dir+'/pretraining_total_num_params.txt', mode='w') as f:
            f.write(str(pretraining_total_num_params))
        wandb_logger.config["num_params_pretraining"] = pretraining_total_num_params
        global_step = 0
        if self.single_env_learning:
            self.random_rollout(
                       wandb_logger,
                        pretraining_env_list, 
                        self.pretraining_env_ids,)
            self.train(
                    global_step,
                    0,
                   wandb_logger,
                   self.pretraining_env_ids,
                   pretraining_env_list,
                   agent, 
                   device,
                   mode='one_env'
                   )
            return 0
        else:
            self.train(
                    global_step,
                    0,
                   wandb_logger,
                   self.pretraining_env_ids,
                   pretraining_env_list,
                   agent, 
                   device,
                   mode='pretraining'
                   )
        # ! #################
        # ! 2. Finetuning phase
        # ! #################
        print(f"pretraining env ids {self.pretraining_env_ids}")
        # * load checkpoint
        agent = PPOAgent(cfg, 
                         self.pretraining_env_ids, 
                         pretraining_env_list,
                         mode='finetuning').to(device)
        agent.load_checkpoint(self.ckpt_dir, device, mode='finetuning')
        # * set new environments
        finetuning_env_list = []
        for j, env_id in enumerate(self.finetuning_env_ids):
            env = make_envpool_env(j, env_id, cfg)
            finetuning_env_list.append(env)
            print(f"{j+1}/{len(self.finetuning_env_ids)} environment {env_id} is loaded...")
        
        # ! add environments [IMPORTANT]
        agent.set_finetuning_envs(self.pretraining_env_ids,
                                  finetuning_env_list)
        agent = agent.to(device)
        
        finetuning_total_num_params = sum([np.prod(p.size()) for p in agent.parameters()])
        agent = torch.compile(agent, 
                              mode="reduce-overhead",
                              disable=not self.use_compile)
        print(f"finetuning total_num_params: {finetuning_total_num_params}")
        with open(self.output_dir+'/finetuning_total_num_params.txt', mode='w') as f:
            f.write(str(finetuning_total_num_params))
        wandb_logger.config["num_params_finetuning"] = finetuning_total_num_params
        global_step += cfg.experiment.total_timesteps
        self.train(global_step,
                    0,
                   wandb_logger,
                   self.finetuning_env_ids,
                   finetuning_env_list,
                   agent, 
                   device,
                   mode='finetuning'
                   )
       
        # ! #################
        # ! 3. from scratch phase
        # ! #################
        
        # * set agent
        agent = PPOAgent(cfg, self.finetuning_env_ids, finetuning_env_list).to(device)
        from_scratch_total_num_params = sum([np.prod(p.size()) for p in agent.parameters()])
        wandb_logger.config["num_params_from_scratch"] = from_scratch_total_num_params
        agent = torch.compile(agent, 
                              mode="reduce-overhead",
                              disable=not self.use_compile)
        global_step += cfg.experiment.total_timesteps
        self.train(global_step,
                   0,
                   wandb_logger,
                   self.finetuning_env_ids,
                   finetuning_env_list,
                   agent, 
                   device,
                   mode='scratch'
                   )
        self.writer.close()
        
    
    def setup(self, rank, cfg):
        os.environ['MASTER_ADDR'] = 'localhost'
        # port = random.randint(10000, 13000)
        os.environ['MASTER_PORT'] = str(cfg.distributed.port) # '12358'
        # os.environ['MASTER_PORT'] = str(port)
        world_size = cfg.distributed.world_size
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        dist.barrier()
    
    def distributed_run(self, rank, wandb_logger=None, *args,):
        # ! #################
        # ! 1. Pretraining phase
        # ! #################
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
        
        sub_env_ids = self.pre_ranked_env_ids[rank]
        print(f"rank: {rank}. sub_env_ids: {sub_env_ids}")
        
        # # * set agent
        all_pretraining_env_list = []
        for j, env_id in enumerate(self.pretraining_env_ids):
            train_envs = make_envpool_env(j, env_id, cfg)
            all_pretraining_env_list.append(train_envs)
        dist.barrier()
        agent = PPOAgent(cfg, self.pretraining_env_ids, all_pretraining_env_list).to(device)
        pretraining_total_num_params = sum([np.prod(p.size()) for p in agent.parameters()])
        agent = DDP(agent, device_ids=[device], output_device=device)
        agent = torch.compile(agent, 
                              mode="reduce-overhead",
                              disable=not self.use_compile)
        print(f"train total_num_params: {pretraining_total_num_params}")
        if rank == 0:
            wandb_logger.config["num_params_pretraining"] = pretraining_total_num_params
        with open(self.output_dir+'/total_num_params.txt', mode='w') as f:
            f.write(str(pretraining_total_num_params))
        print(f"rank: {rank}. agent is initialized")

        # * set vector environments
        sub_pretraining_env_ids = self.pre_ranked_env_ids[rank]
        print(f"Rank: {rank}. {sub_pretraining_env_ids}")
        sub_pretraining_env_list = []
        for j, env_id in enumerate(sub_pretraining_env_ids):
            train_envs = make_envpool_env(j, env_id, cfg)
            sub_pretraining_env_list.append(train_envs)
            print(f"Rank: {rank}. {j+1}/{len(sub_pretraining_env_ids)} environment {env_id} is loaded...")
        global_step = 0
        self.train(global_step,
                   rank,
                   wandb_logger,
                   sub_pretraining_env_ids, 
                   sub_pretraining_env_list,
                   agent, 
                   device,
                   mode='pretraining')

        dist.barrier()
        
        # ! #################
        # ! 2. Finetuning phase
        # ! #################
        print(f"pretraining env ids {self.pretraining_env_ids}")
        # * load checkpoint
        agent = PPOAgent(cfg, 
                         self.pretraining_env_ids, 
                         all_pretraining_env_list,
                         mode='finetuning').to(device)
        agent.load_checkpoint(self.ckpt_dir, device, mode='finetuning')
        # * set new environments
        all_finetuning_env_list = []
        for j, env_id in enumerate(self.finetuning_env_ids):
            env = make_envpool_env(j, env_id, cfg)
            all_finetuning_env_list.append(env)
        
        # ! add environments [IMPORTANT]
        agent.set_finetuning_envs(self.pretraining_env_ids,
                                  all_finetuning_env_list)
        agent = agent.to(device)
        finetuning_total_num_params = sum([np.prod(p.size()) for p in agent.parameters()])
        if rank == 0:
            wandb_logger.config["num_params_finetuning"] = finetuning_total_num_params
        agent = DDP(agent, device_ids=[device], output_device=device)
        agent = torch.compile(agent, 
                              mode="reduce-overhead",
                              disable=not self.use_compile)
        
        # * training sub environments on rank
        sub_finetuning_env_ids = self.fine_ranked_env_ids[rank]
        sub_finetuning_env_list = []
        for j, env_id in enumerate(sub_finetuning_env_ids):
            env = make_envpool_env(j, env_id, cfg)
            sub_finetuning_env_list.append(env)
            print(f"Rank:{rank}. {j+1}/{len(sub_finetuning_env_ids)} environment {env_id} is loaded...")
        global_step += cfg.experiment.total_timesteps
        self.train(global_step,
                   rank,
                   wandb_logger,
                   sub_finetuning_env_ids,
                   sub_finetuning_env_list,
                   agent, 
                   device,
                   mode='finetuning'
                   )
        
        dist.barrier()

        # ! #################
        # ! 3. from scratch phase
        # ! #################
        agent = PPOAgent(cfg, self.finetuning_env_ids, all_finetuning_env_list).to(device)
        from_scratch_total_num_params = sum([np.prod(p.size()) for p in agent.parameters()])
        if rank == 0:
            wandb_logger.config["num_params_from_scratch"] = from_scratch_total_num_params
        agent = DDP(agent, device_ids=[device], output_device=device)
        agent = torch.compile(agent, 
                              mode="reduce-overhead",
                              disable=not self.use_compile)
        global_step += cfg.experiment.total_timesteps
        self.train(global_step,
                   rank,
                   wandb_logger,
                   sub_finetuning_env_ids,
                   sub_finetuning_env_list,
                   agent, 
                   device,
                   mode='scratch'
                   )
        
    # @torch.compile
    def rollout(self, 
                rank,
                agent, 
                train_different_env_list, 
                sub_env_ids,
                envs_storages,
                next_obs_dict, 
                next_done_dict, 
                envs_returns, 
                envs_lengths,
                device,
                update_idx,
                global_step,
                mode,):
        for i, env in enumerate(train_different_env_list):
            env_id = sub_env_ids[i]
            for step in range(0, self.cfg.experiment.num_rollout_steps):
                envs_storages[env_id]["obs"][step] = next_obs_dict[env_id]
                envs_storages[env_id]["dones"][step] = next_done_dict[env_id]

                with torch.no_grad():
                    if self.is_distributed:
                        action, logprob, _, value = agent.module.get_action_and_value(env, next_obs_dict[env_id])
                    else:
                        action, logprob, _, value = agent.get_action_and_value(env, next_obs_dict[env_id])
                        
                envs_storages[env_id]["values"][step] = value.flatten()
                envs_storages[env_id]["actions"][step] = action
                envs_storages[env_id]["logprobs"][step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                if self.cfg.experiment.env_type == "gymnasium":
                    next_obs, reward, terminated, truncated, infos = env.step(action.cpu().numpy())
                    done = np.logical_or(terminated, truncated)
                elif self.cfg.experiment.env_type == "gym":
                    next_obs, reward, done, infos = env.step(action.cpu().numpy())
                envs_storages[env_id]["rewards"][step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
                next_obs_dict[env_id] = next_obs
                next_done_dict[env_id] = next_done
                episodic_returns = []
                episodic_lengths = []
                for k, d in enumerate(done):
                    if d:
                        episodic_returns.append(infos["r"][k])
                        episodic_lengths.append(infos["l"][k])
                mean_episodic_return = 0
                if len(episodic_returns) > 0:
                    mean_episodic_return = np.mean(np.array(episodic_returns))
                    # logger.info(f"rank: {rank}, [{update_idx}/{self.total_num_updates}] Train. env_name:{sub_env_ids[i]}, global_step={global_step}, mean_episodic_return={mean_episodic_return}")
                    mean_episodic_length = np.mean(np.array(episodic_lengths))
                    std_episodic_return = np.std(np.array(episodic_returns))
                    envs_returns[f'{mode}/' + env_id +'/mean_episodic_return'] = mean_episodic_return
                    # envs_returns[f'{mode}/' + env_id +'/std_episodic_return'] = std_episodic_return
                    # envs_lengths[f'{mode}/' + env_id +'/mean_episodic_length'] = mean_episodic_length
                    
            print(f"Rank: {rank}, [{update_idx}/{self.total_num_updates}] {mode}. env_name:{sub_env_ids[i]}, global_step={global_step}, mean_episodic_return={mean_episodic_return}")
    
    def random_rollout(self, 
                       wandb_logger,
                train_different_env_list, 
                sub_env_ids):
        exp_cfg = self.cfg.experiment
        for i, env in enumerate(train_different_env_list):
            env_id = sub_env_ids[i]
            if exp_cfg.env_type == "gymnasium":
                next_obs, infos = env.reset()
            elif exp_cfg.env_type == "gym":
                next_obs = env.reset()
            
        for i, env in enumerate(train_different_env_list):
            env_id = sub_env_ids[i]
            episodic_returns = []
            while len(episodic_returns) < 100:
                action = np.array([env.action_space.sample() for i in range(self.cfg.experiment.num_envs)])
                # TRY NOT TO MODIFY: execute the game and log data.
                if self.cfg.experiment.env_type == "gymnasium":
                    next_obs, reward, terminated, truncated, infos = env.step(action)
                    done = np.logical_or(terminated, truncated)
                elif self.cfg.experiment.env_type == "gym":
                    next_obs, reward, done, infos = env.step(action)
                for k, d in enumerate(done):
                    if d:
                        random_return = infos["r"][k]
                        episodic_returns.append(random_return)
                        print(f"random return: {random_return}")
            mean_episodic_return = np.array(episodic_returns).mean()
            wandb_logger.log({f"rand_mean_episodic_return/{env_id}":mean_episodic_return})

    def calculate_advantages(self, 
                             cfg,
                             sub_env_ids: List, 
                             train_different_env_list: List,
                             agent,
                             next_obs_dict: TensorDict,
                             next_done_dict: TensorDict,
                             envs_storages: TensorDict,
                             device,
                             ):
        exp_cfg = self.cfg.experiment
        with torch.no_grad():
            for i, env in enumerate(train_different_env_list):
                env_id = sub_env_ids[i]
                if self.is_distributed:
                    next_value = agent.module.get_value(env, next_obs_dict[env_id]).reshape(1, -1)
                else:
                    next_value = agent.get_value(env, next_obs_dict[env_id]).reshape(1, -1)
                rewards = envs_storages[env_id]['rewards']
                envs_storages[env_id]['advantages'] = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(exp_cfg.num_rollout_steps)):
                    if t == exp_cfg.num_rollout_steps - 1:
                        nextnonterminal = 1.0 - next_done_dict[env_id]
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - envs_storages[env_id]["dones"][t + 1]
                        nextvalues =  envs_storages[env_id]['values'][t + 1]
                    delta = envs_storages[env_id]['rewards'][t] + cfg.ppo.gamma * nextvalues * nextnonterminal - envs_storages[env_id]['values'][t]
                    envs_storages[env_id]['advantages'][t] = lastgaelam = delta + cfg.ppo.gamma * cfg.ppo.gae_lambda * nextnonterminal * lastgaelam
                envs_storages[env_id]['returns'] = envs_storages[env_id]['advantages'] + envs_storages[env_id]['values']
            
    # @torch.compile
    def eval_agent(self, 
                   agent, 
                   test_different_envs, 
                   sub_env_ids,
                   device, 
                   update_idx, 
                   global_step: int) -> Dict:
        agent.eval()
        exp_cfg = self.cfg.experiment
        eval_cfg = self.cfg.evaluation
        test_envs_returns = dict()
        test_envs_lengths = dict()
        for i, test_env in enumerate(test_different_envs):
            ith_test_episodic_returns = []
            ith_test_episodic_lengths = []
            next_obs = test_env.reset()
            next_obs = torch.Tensor(next_obs).to(device)
            dones = np.zeros(eval_cfg.num_test_envs)
            while len(ith_test_episodic_returns) < eval_cfg.num_eval:
                with torch.no_grad():
                    if self.distributed:
                        output = agent.module.get_action_and_value(test_env.action_space, next_obs)
                    else:
                        output = agent.get_action_and_value(test_env.action_space, next_obs)
                    action, logprob, _, value = output
                    next_obs, reward, dones, info = test_env.step(action.cpu().numpy())
                    next_obs = torch.Tensor(next_obs).to(device)
                    if True in dones:
                        done_env_indices = np.where(dones==True)
                        total_return = info['r'][done_env_indices]
                        total_length = info['l'][done_env_indices]
                        ith_test_episodic_returns += [*total_return]
                        ith_test_episodic_lengths += [*total_length]
            ith_test_episodic_returns = np.array(ith_test_episodic_returns)
            ith_test_episodic_lengths = np.array(ith_test_episodic_lengths)
            # test_envs_returns[self.pretraining_env_ids[i]+'/eval'] = ith_test_episodic_returns
            
            for batch_index in range(eval_cfg.num_eval):
                test_envs_returns[sub_env_ids[i]+f'/eval_{batch_index}'] = ith_test_episodic_returns[batch_index]
            test_envs_returns[sub_env_ids[i]+f'/eval_mean'] = ith_test_episodic_returns.mean()
            test_envs_returns[sub_env_ids[i]+'/eval_std'] = ith_test_episodic_returns.std()
            test_envs_lengths[sub_env_ids[i]+'/eval_length'] = ith_test_episodic_lengths.mean()
            logger.info(f"[{update_idx}/{self.total_num_updates}] Evaluation. env_name:{sub_env_ids[i]}, global_step={global_step}, mean_episodic_return={ith_test_episodic_returns.mean()}")
            print(f"[{update_idx}/{self.total_num_updates}] Evaluation. env_name:{sub_env_ids[i]}, global_step={global_step}, mean_episodic_return={ith_test_episodic_returns.mean()}")
            
            # self.writer.add_scalars('charts/mean_test_episodic_return', test_envs_returns, global_step)
            # self.writer.add_scalars('charts/mean_test_episodic_length', test_envs_lengths, global_step)
        agent.train()
        return test_envs_returns
    
    # @torch.compile
    def train_agent(self, reshaped_storages, agent, train_different_env_list, sub_env_ids):
        b_inds = np.arange(self.batch_size)
        total_epochs_envs_loss = 0.0
        total_epochs_value_loss = 0.0
        total_epochs_policy_loss = 0.0
        total_epochs_entropy_loss = 0.0
        for epoch in range(self.cfg.ppo.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                total_envs_loss = 0.0
                total_value_loss = 0.0
                total_policy_loss = 0.0
                total_entropy_loss = 0.0
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]
                if self.is_distributed:
                    agent.module.optim_zero_grad()  
                else:
                    agent.optim_zero_grad()  
                for i, env in enumerate(train_different_env_list):
                    env_id = sub_env_ids[i]
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]
                    pg_loss, entropy_loss, v_loss = self.calculate_ppo_loss(reshaped_storages, agent, env_id, env, mb_inds)
                    const_coef = self.cfg.ppo.const_coef
                    if isinstance(env.single_action_space, GymBox) or isinstance(env.single_action_space, GymnasiumBox):
                        ent_coef = self.cfg.ppo.ent_coef
                        if self.is_distributed:
                            a_mean_weights_norm = matrix_norm(agent.module.actor.a_mean_weights).mean()
                            a_logstd_weights_norm = matrix_norm(agent.module.actor.a_logstd_weights).mean()
                            value_weights_norm = matrix_norm(agent.module.critic.value_weights).mean()
                        else:
                            a_mean_weights_norm = matrix_norm(agent.actor.a_mean_weights).mean()
                            a_logstd_weights_norm = matrix_norm(agent.actor.a_logstd_weights).mean()
                            value_weights_norm = matrix_norm(agent.critic.value_weights).mean()
                        pg_loss = pg_loss + const_coef * (a_mean_weights_norm + a_logstd_weights_norm) - entropy_loss * ent_coef
                        v_loss = self.cfg.ppo.vf_coef * v_loss + const_coef * value_weights_norm
                    else:
                        ent_coef = self.cfg.ppo.ent_coef_discrete
                        if self.is_distributed:
                            a_prob_weights_norm = agent.module.actor.a_prob_weights.mean()
                            value_weights_norm = matrix_norm(agent.module.critic.value_weights).mean()
                        else:
                            a_prob_weights_norm = agent.actor.a_prob_weights.mean()
                            value_weights_norm = matrix_norm(agent.critic.value_weights).mean()
                        pg_loss = pg_loss + const_coef * a_prob_weights_norm - entropy_loss * ent_coef
                        v_loss = self.cfg.ppo.vf_coef * v_loss + const_coef * value_weights_norm
                    total_loss = pg_loss + v_loss
                    total_value_loss += v_loss
                    total_policy_loss += pg_loss
                    total_entropy_loss += entropy_loss
                    total_envs_loss += total_loss
                total_envs_loss.backward()
                if self.is_distributed:
                    dist.barrier()
                    agent.module.optim_step()
                else:
                    agent.optim_step()
                total_epochs_envs_loss += total_envs_loss
                total_epochs_value_loss += total_value_loss
                total_epochs_policy_loss += total_policy_loss
                total_epochs_entropy_loss += total_entropy_loss
                
        total_epochs_envs_loss = total_epochs_envs_loss / self.cfg.ppo.update_epochs / self.cfg.ppo.num_minibatches
        total_epochs_value_loss = total_epochs_value_loss / self.cfg.ppo.update_epochs / self.cfg.ppo.num_minibatches
        total_epochs_policy_loss = total_epochs_policy_loss / self.cfg.ppo.update_epochs / self.cfg.ppo.num_minibatches
        total_epochs_entropy_loss = total_epochs_entropy_loss / self.cfg.ppo.update_epochs / self.cfg.ppo.num_minibatches
        
        return total_epochs_envs_loss, total_epochs_value_loss, total_epochs_policy_loss, total_epochs_entropy_loss
    
    def calculate_ppo_loss(self, reshaped_storages, agent, env_id, env, mb_inds):
        mb_obs = reshaped_storages[env_id]['obs'][mb_inds]
        mb_actions = reshaped_storages[env_id]['actions'][mb_inds]
        mb_logprobs = reshaped_storages[env_id]['logprobs'][mb_inds]
        if self.is_distributed:
            _, newlogprob, entropy, newvalue = agent.module.get_action_and_value(env, mb_obs, mb_actions)
        else:
            _, newlogprob, entropy, newvalue = agent.get_action_and_value(env, mb_obs, mb_actions)
        logratio = newlogprob - mb_logprobs
        ratio = logratio.exp()

        mb_advantages = reshaped_storages[env_id]['advantages'][mb_inds]
        if self.cfg.ppo.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
            

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.cfg.ppo.clip_coef, 1 + self.cfg.ppo.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        mb_returns = reshaped_storages[env_id]['returns'][mb_inds].view(-1)
        if self.cfg.ppo.norm_return:
            mb_returns = (mb_returns - mb_returns.mean()) / (mb_returns.std() + 1e-8)
            
        mb_values = reshaped_storages[env_id]['values'][mb_inds].view(-1)
        newvalue = newvalue.view(-1)
        if self.cfg.ppo.clip_vloss:
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_clipped = mb_values + torch.clamp(
                newvalue - mb_values,
                -self.cfg.ppo.clip_coef,
                self.cfg.ppo.clip_coef,
            )
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

        # regularization term                            
        entropy_loss = entropy.mean()
        return pg_loss, entropy_loss, v_loss
    
    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: batch[k].to(self.device) for k in batch}
                                                        
    def finish(self) -> None:
        wandb.finish()
        