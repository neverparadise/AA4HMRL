import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf
from modules.networks.actor_critic_agnostic import AgnosticStochasticActor, AgnosticVNetwork, AgnosticDiscreteTwinQ, AgnosticContinuousTwinQ
from modules.networks.actor_critic_specific import SpecificActor, SpecificVNetwork, SpecificDiscreteTwinQ, SpecificContinuousTwinQ
from torch.optim.lr_scheduler import CosineAnnealingLR
from gym.spaces import Box as GymBox
from gym.spaces import Discrete as GymDiscrete
from gymnasium.spaces import Box as GymnasiumBox
from gymnasium.spaces import Discrete as GymnasiumDiscrete
from gym.core import Env as GymEnv
from gymnasium.core import Env as GymnasiumEnv
from typing import TypeVar, List, Dict, Any
import copy

GymSpace = TypeVar('GymSpace', GymBox, GymDiscrete, GymnasiumBox, GymnasiumDiscrete)
Env = TypeVar('Env', GymEnv, GymnasiumEnv)
TensorBatch = List[torch.Tensor]


EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class IQLAgent(nn.Module):
    def __init__(self, 
                 cfg: OmegaConf, 
                 env_ids: List[str], 
                 env_list: List[Env]):
        super().__init__()
        self.cfg = cfg        
        self.discount = cfg.iql.gamma
        self.beta = cfg.iql.beta
        self.tau = cfg.iql.tau
        self.iql_tau = cfg.iql.iql_tau
        self.is_distributed = cfg.distributed.world_size > 1 and cfg.distributed.multiprocessing_distributed
        self.use_compile = cfg.nn.actor_critic.use_compile
        if cfg.nn.actor_critic.encoder_net_1d == 's4' or \
            cfg.nn.actor_critic.decoder_net == 's4':
                self.use_compile = False # S4 does not support torch compile
        
        if cfg.nn.env_specific_enc_dec:
            self.actor = SpecificActor(cfg, env_ids, env_list)
            self.vf = SpecificVNetwork(cfg, env_ids, env_list) 
            self.cqf = SpecificContinuousTwinQ(cfg, env_ids, env_list)
            self.cqf_target = copy.deepcopy(self.cqf).requires_grad_(False)
            self.dqf = SpecificDiscreteTwinQ(cfg, env_ids, env_list)
            self.dqf_target = copy.deepcopy(self.dqf).requires_grad_(False)
        else:
            self.actor = AgnosticStochasticActor(cfg, env_ids, env_list)
            self.vf = AgnosticVNetwork(cfg, env_ids, env_list)
            self.cqf = AgnosticContinuousTwinQ(cfg, env_ids, env_list)
            self.cqf_target = copy.deepcopy(self.cqf).requires_grad_(False)
            self.dqf = AgnosticDiscreteTwinQ(cfg, env_ids, env_list)
            self.dqf_target = copy.deepcopy(self.dqf).requires_grad_(False)
            
        self.cfg = cfg
        self.use_grad_clip = cfg.iql.use_grad_clip

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.iql.actor_lr, weight_decay=cfg.iql.actor_weight_decay)
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, cfg.experiment.total_timesteps)
        self.v_optimizer = optim.Adam(self.vf.parameters(), lr=cfg.iql.critic_lr, weight_decay=cfg.iql.critic_weight_decay)
        self.q_optimizer = optim.Adam(list(self.cqf.parameters()) + \
                                                list(self.dqf.parameters())
                                      , lr=cfg.iql.critic_lr, weight_decay=cfg.iql.critic_weight_decay)

    
    def optim_zero_grad(self):
        self.actor_optimizer.zero_grad()
        self.v_optimizer.zero_grad()
        self.q_optimizer.zero_grad()
        
    def optim_step(self):
        if self.use_grad_clip:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.iql.max_grad_norm)
            nn.utils.clip_grad_norm_(self.vf.parameters(), self.cfg.iql.max_grad_norm)
            nn.utils.clip_grad_norm_(self.dqf.parameters(), self.cfg.iql.max_grad_norm)
            nn.utils.clip_grad_norm_(self.cqf.parameters(), self.cfg.iql.max_grad_norm)
        self.actor_optimizer.step()
        self.v_optimizer.step()
        self.q_optimizer.step()
        self.actor_lr_schedule.step()
        
    def update_target_net(self):
        soft_update(self.cqf_target, self.cqf, self.tau)
        soft_update(self.dqf_target, self.dqf, self.tau)
        
    @torch.no_grad()
    def act(self, env, state: np.ndarray, device: str = "cpu"):
        action_space = env.action_space
        if isinstance(action_space, GymBox) or isinstance(action_space, GymnasiumBox):
            max_action = torch.tensor(action_space.high).to(device)
            state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
            dist, _ = self.actor(env, state)
            action = dist.mean if not self.training else dist.sample()
            action = torch.clamp(max_action * action, -max_action, max_action)
            return action.cpu().data.numpy().flatten()
        elif isinstance(action_space, GymDiscrete) or isinstance(action_space, GymnasiumDiscrete):
            state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
            dist, _ = self.actor(env, state)
            action = dist.sample()
            return action.cpu().data.numpy().flatten()

    def _compute_v_loss(self, 
                  env: Env,
                  observations: torch.Tensor, 
                  actions: torch.Tensor, 
                  log_dict: Dict) -> torch.Tensor:
        env_id = env.env_id
        action_space = env.action_space
        if isinstance(action_space, GymBox) or isinstance(action_space, GymnasiumBox):
            min_target_q = self.cqf_target(env, observations, actions)

        elif isinstance(action_space, GymDiscrete) or isinstance(action_space, GymnasiumDiscrete):
            target_dq1, target_dq2 = self.dqf_target.both(env, observations)
            target_dq1_a = target_dq1.gather(1, actions.long())
            target_dq2_a = target_dq2.gather(1, actions.long())
            min_target_q = torch.min(target_dq1_a, target_dq2_a)
        
        v = self.vf(env, observations)
        adv = min_target_q.detach() - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict[f"{env_id}/value_loss"] = v_loss.item()
        return v_loss, adv
    
    def _compute_q_loss(self,
                env: Env,
                next_v: torch.Tensor,
                observations: torch.Tensor,
                actions: torch.Tensor,
                rewards: torch.Tensor,
                terminals: torch.Tensor,
                log_dict: Dict,
                ):
        env_id = env.env_id
        action_space = env.action_space
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach().view(-1)
        
        if isinstance(action_space, GymBox) or isinstance(action_space, GymnasiumBox):
            qs = self.cqf.both(env, observations, actions)
            q_loss = sum(F.mse_loss(q.view(-1), targets) for q in qs) / len(qs)
            log_dict[f"{env_id}/continuous_q_loss"] = q_loss.item()
            # soft_update(self.cqf_target, self.cqf, self.tau)
            
        elif isinstance(action_space, GymDiscrete) or isinstance(action_space, GymnasiumDiscrete):
            q1, q2 = self.dqf(env, observations)
            q1_a, q2_a = q1.gather(1, actions.long()), q2.gather(1, actions.long())
            q_loss1 = F.mse_loss(q1_a, targets).mean()
            q_loss2 = F.mse_loss(q2_a, targets).mean()
            q_loss = q_loss1 + q_loss2 
            log_dict[f"{env_id}/discrete_q_loss"] = q_loss.item()
            # soft_update(self.dqf_target, self.dqf, self.tau)
        return q_loss
    
    def _compute_policy_loss(
        self,
        env: Env,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
        ):
        env_id = env.env_id
        action_space = env.action_space
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        if isinstance(action_space, GymBox) or isinstance(action_space, GymnasiumBox):
            policy_out, _ = self.actor(env, observations)
            if isinstance(policy_out, torch.distributions.Distribution): # stochastic actor
                bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
            elif torch.is_tensor(policy_out): # deterministic actor
                if policy_out.shape != actions.shape:
                    raise RuntimeError("Actions shape missmatch")
                bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
            else:
                raise NotImplementedError
            policy_loss = torch.mean(exp_adv * bc_losses)
            log_dict[f"{env_id}/continuous_actor_loss"] = policy_loss.item()
            
        elif isinstance(action_space, GymDiscrete) or isinstance(action_space, GymnasiumDiscrete):
            policy_out, _ = self.actor(env, observations)
            if isinstance(policy_out, torch.distributions.Distribution): # stochastic actor
                bc_losses = -policy_out.log_prob(actions.squeeze(-1))
            policy_loss = torch.mean(exp_adv * bc_losses)
            log_dict[f"{env_id}/discrete_actor_loss"] = policy_loss.item()
        return policy_loss
    
    def compute_loss(self, env, batch: TensorBatch) -> Dict[str, float]:
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}
        compute_v_loss = torch.compile(self._compute_v_loss, disable=not self.use_compile)
        compute_q_loss = torch.compile(self._compute_q_loss, disable=not self.use_compile)
        compute_policy_loss = torch.compile(self._compute_policy_loss, disable=not self.use_compile)
        
        next_v = self.vf(env, next_observations)
        # Update value function
        v_loss, adv = compute_v_loss(env, observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        q_loss = compute_q_loss(env, next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        policy_loss = compute_policy_loss(env, adv, observations, actions, log_dict)
        per_env_sum_loss = policy_loss + q_loss + v_loss
        return per_env_sum_loss, log_dict

    def get_state_dict(self, t) -> Dict[str, Any]:
        return {
            "cqf": self.cqf.state_dict(),
            "dqf": self.dqf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "t": t,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)
        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])
        self.total_it = state_dict["total_it"]
        
    def set_finetuning_envs(self, 
                            envs: List[Env]):
        pass