import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, MultivariateNormal
from omegaconf import OmegaConf
from modules.networks.blocks import get_encoder_1d, get_encoder_2d, get_decoder
from modules.networks.blocks import Resnet
from modules.networks.transformer import TransformerBlock
from modules.networks.mlp import ResidualMLP
from modules.utils import get_activation
from gym.spaces import Box as GymBox
from gym.spaces import Discrete as GymDiscrete
from gymnasium.spaces import Box as GymnasiumBox
from gymnasium.spaces import Discrete as GymnasiumDiscrete
from gym.core import Env as GymEnv
from gymnasium.core import Env as GymnasiumEnv
import math
from typing import TypeVar, List, Tuple
GymSpace = TypeVar('GymSpace', GymBox, GymDiscrete, GymnasiumBox, GymnasiumDiscrete)
Env = TypeVar('Env', GymEnv, GymnasiumEnv)


def bmm_input(b_weight: torch.Tensor, b_input: torch.Tensor, scale=False):
    batch_size, feature_dim = b_input.shape
    if scale:
        bmm = torch.einsum('nfh, nf -> nh', b_weight, b_input) / math.sqrt(feature_dim)
    else:
        bmm = torch.einsum('nfh, nf -> nh', b_weight, b_input)
    return bmm


def bmm_output(b_weight: torch.Tensor, b_input: torch.Tensor, scale=False):
    batch_size, output_dim, d_model = b_weight.shape
    batch_size, d_model = b_input.shape
    # [batch_size, 6, 32], [batch_size, 32]
    if scale:
        bmm = torch.einsum('noh, nh -> no', b_weight, b_input)  / math.sqrt(d_model)
    else:
        bmm = torch.einsum('noh, nh -> no', b_weight, b_input)
    return bmm


class SpecificBase(nn.Module):
    def __init__(self, 
                 cfg: OmegaConf,
                env_ids: List[str], 
                 env_list: List[Env],):
        super().__init__()
        self.cfg = cfg
        self.env_ids = env_ids[:]
        self.env_list = env_list[:]
        activation_name: str = cfg.nn.actor_critic.activation
        self.act_func: torch.nn.Module = get_activation(activation_name)()
        self.d_model = cfg.nn.actor_critic.d_model
        self.encoders_dict = nn.ModuleDict()
        encoders_dict = dict()
        for env_id, env in zip(env_ids, env_list):
            obs_dim = env.observation_space.shape
            # for 1-D input: MLP
            if len(obs_dim) < 2:
                obs_dim = np.prod(obs_dim)
                obs_encoder = ResidualMLP(cfg, obs_dim, self.d_model)
            # for 2-D input: CNN
            elif len(obs_dim) == 3: # [C, H, W]
                obs_encoder = Resnet(cfg)
            encoders_dict[env_id] = obs_encoder
        self.encoders_dict = nn.ModuleDict(encoders_dict)
        self.use_mlp = cfg.nn.actor_critic.use_mlp
        if self.use_mlp:
            input_dim = cfg.nn.actor_critic.d_model
            output_dim = cfg.nn.actor_critic.d_model
            self.shared_mlp: torch.nn.Module = ResidualMLP(cfg, input_dim, output_dim)
        
        self.task_id_int_dict = dict()
        for task_id, number in zip(env_ids, range(len(env_ids))):
            self.task_id_int_dict[task_id] = number
        self.task_embedding = nn.Embedding(len(self.task_id_int_dict), self.d_model)

    def add_envionments(self, 
                        pretraining_env_ids,
                        finetuning_env_list: List[Env]):
        # self.env_ids: id list of environments
        # self.task_id_int_dict: key: id, value: index of environment
        
        # generate new integers from new environments
        pretrained_num_embeddings = len(pretraining_env_ids) # 8
        finetuning_env_ids = []
        for env in finetuning_env_list:
            env_id = env.env_id
            self.env_ids.append(env_id)
            finetuning_env_ids.append(env_id)
        new_integers = range(pretrained_num_embeddings, len(self.env_ids)) # 

        # add the key and value of new environment
        for task_id, number in zip(finetuning_env_ids, new_integers):
            self.task_id_int_dict[task_id] = number
        
        # extend task embedding layer
        extended_num_embeddings = len(self.env_ids)
        new_task_embedding = nn.Embedding(extended_num_embeddings, self.d_model).to(self.task_embedding.weight.device)
        with torch.no_grad():
            new_task_embedding.weight[:pretrained_num_embeddings] = self.task_embedding.weight
        self.task_embedding = new_task_embedding
        
    def add_obs_encoders(self,
                    new_envs: List[Env]):
        # add new_encoders
        for env in new_envs:
            obs_dim = env.observation_space.shape
            # for 1-D input: MLP
            if len(obs_dim) < 2:
                obs_dim = np.prod(obs_dim)
                obs_encoder = ResidualMLP(self.cfg, obs_dim, self.d_model)
            # for 2-D input: CNN
            elif len(obs_dim) == 3:
                obs_encoder = Resnet(self.cfg)
            self.encoders_dict[env.env_id] = obs_encoder
    
    def add_decoders(self,
                    new_envs: List[Env]):
        pass

    def encoding(self,
                 env: Env,
                 x: torch.Tensor):
        # x: [batch_size, feature_dim] or [batch_size, num_frames, H, W]
        env_id = env.env_id
        h = self.act_func(self.encoders_dict[env_id](x))
        task_number = self.task_id_int_dict[env_id]
        task_number = torch.tensor(task_number, dtype=torch.int).to(x.device)
        task_embed = self.task_embedding(task_number)
        h = h + task_embed
        if self.use_mlp:
            h = self.act_func(self.shared_mlp(h))
        else:
            h = self.act_func(h)
        return h
    
    def decoding(self):
        pass
    
    def forward(self):
        pass
        
class SpecificActor(SpecificBase):
    def __init__(self, 
                 cfg: OmegaConf,
                env_ids: List[str], 
                 env_list: List[Env],):
        super().__init__(cfg, env_ids, env_list)
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5
        mean_decoders_dict = dict()
        logstd_decoders_dict = dict()
        prob_decoders_dict = dict()        
        for env_id, env in zip(env_ids, env_list):
            if isinstance(env.action_space, GymBox) or \
                isinstance(env.action_space, GymnasiumBox):
                act_dim = np.prod(env.action_space.shape)
                mean_decoder = ResidualMLP(cfg, self.d_model, act_dim) # ResidualMLP(cfg, obs_dim, 256)
                logstd_decoder = ResidualMLP(cfg, self.d_model, act_dim) 
                mean_decoders_dict[env_id] = mean_decoder
                logstd_decoders_dict[env_id] = logstd_decoder
                
            elif isinstance(env.action_space, GymDiscrete) or \
                isinstance(env.action_space, GymnasiumDiscrete):
                act_dim = env.action_space.n
                prob_decoder = ResidualMLP(cfg, self.d_model, act_dim) 
                prob_decoders_dict[env_id] = prob_decoder
            
        self.mean_decoders_dict = nn.ModuleDict(mean_decoders_dict)
        self.logstd_decoders_dict = nn.ModuleDict(logstd_decoders_dict)
        self.prob_decoders_dict = nn.ModuleDict(prob_decoders_dict)
    
    def add_policy_decoders(self,
                    new_envs: List[Env]):
        for env in new_envs:
            env_id = env.env_id
            if isinstance(env.action_space, GymBox) or \
                isinstance(env.action_space, GymnasiumBox):
                act_dim = np.prod(env.action_space.shape)
                mean_decoder = ResidualMLP(self.cfg, self.d_model, act_dim) # ResidualMLP(cfg, obs_dim, 256)
                logstd_decoder = ResidualMLP(self.cfg, self.d_model, act_dim) 
                self.mean_decoders_dict[env_id] = mean_decoder
                self.logstd_decoders_dict[env_id] = logstd_decoder
                
            elif isinstance(env.action_space, GymDiscrete) or \
                isinstance(env.action_space, GymnasiumDiscrete):
                act_dim = env.action_space.n
                prob_decoder = ResidualMLP(self.cfg, self.d_model, act_dim) 
                self.prob_decoders_dict[env_id] = prob_decoder

    def decoding(self,
                 env: Env,
                 h: torch.Tensor):
        # ? Decoding continuous action
        env_id = env.env_id
        action_space = env.action_space
        if isinstance(action_space, GymBox) or isinstance(action_space, GymnasiumBox):
            a_mean = self.mean_decoders_dict[env_id](h)
            a_logstd = self.logstd_decoders_dict[env_id](h)
            self.a_mean_weights = a_mean
            self.a_logstd_weights = a_logstd
            a_logstd = torch.tanh(a_logstd)
            a_logstd = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (a_logstd + 1)
            actor_std = a_logstd.exp()
            dist = Normal(a_mean, actor_std)
            return dist, a_mean

        # ? Decoding discrete action
        elif isinstance(action_space, GymDiscrete) or isinstance(action_space, GymnasiumDiscrete):
            # get num_discretes
            a_probs = self.prob_decoders_dict[env_id](h)
            self.a_prob_weights = a_probs
            logits = F.softmax(a_probs, dim=-1)
            # get categorical distribution
            dist = Categorical(logits=logits)
            return dist, logits
    
    def forward(self, 
                env: GymSpace,
                x: torch.Tensor):
        # ? Encoding
        h = self.encoding(env, x)
        
        # ? Decoding
        dist, _ = self.decoding(env, h)
        return dist, _
    
    def set_finetuning_envs(self, 
                            pretraining_env_ids,
                            finetuning_env_list: List[Env]):
        self.add_envionments(pretraining_env_ids, finetuning_env_list)
        self.add_obs_encoders(finetuning_env_list)
        self.add_policy_decoders(finetuning_env_list)

        
class SpecificVNetwork(SpecificBase):
    def __init__(self, 
                 cfg: OmegaConf,
                env_ids: List[str], 
                 env_list: List[Env],):
        super().__init__(cfg, env_ids, env_list)
        decoders_dict = dict()
        for env_id, env in zip(env_ids, env_list):
            val_decoder = ResidualMLP(cfg, self.d_model, 1)
            decoders_dict[env_id] = val_decoder
        self.value_decoder_dict = nn.ModuleDict(decoders_dict)

    def add_v_decoders(self,
                    new_envs: List[Env]):
        for env in new_envs:
            env_id = env.env_id
            val_decoder = ResidualMLP(self.cfg, self.d_model, 1)
            self.value_decoder_dict[env_id] = val_decoder
    
    def decoding(self, 
                 env: Env,
                 h:torch.Tensor):
        env_id = env.env_id
        value = self.value_decoder_dict[env_id](h)
        self.value_weights = value 
        return value
        
    def forward(self, 
                env,
                x:torch.Tensor):
        # ? Encoding
        h = self.encoding(env, x)
        
        # ? Decoding
        value = self.decoding(env, h)
        return value

    def set_finetuning_envs(self, 
                            pretraining_env_ids,
                            finetuning_env_list: List[Env]):
        self.add_envionments(pretraining_env_ids, finetuning_env_list)
        self.add_obs_encoders(finetuning_env_list)
        self.add_v_decoders(finetuning_env_list)

class SpecificDiscreteQNetwork(SpecificBase):
    def __init__(self, 
                 cfg: OmegaConf,
                env_ids: List[str], 
                 env_list: List[Env],):
        super().__init__(cfg, env_ids, env_list)
        decoders_dict = dict()
        for env_id, env in zip(env_ids, env_list):
            if isinstance(env.action_space, GymDiscrete) or \
                isinstance(env.action_space, GymnasiumDiscrete):
                q_decoder = ResidualMLP(cfg, self.d_model, env.action_space.n)  
                decoders_dict[env_id] = q_decoder
            else:
                continue
        self.value_decoder_dict = nn.ModuleDict(decoders_dict)

    def add_q_decoders(self,
                    new_envs: List[Env]):
        for env in new_envs:
            env_id = env.env_id
            if isinstance(env.action_space, GymDiscrete) or \
                isinstance(env.action_space, GymnasiumDiscrete):
                q_decoder = ResidualMLP(self.cfg, self.d_model, env.action_space.n)  
                self.value_decoder_dict[env_id] = q_decoder
            else:
                continue
            
    def decoding(self, 
                 env: Env,
                 h:torch.Tensor):
        env_id = env.env_id
        value = self.value_decoder_dict[env_id](h)
        self.value_weights = value 
        return value
        
    def forward(self, 
                env,
                x:torch.Tensor):
        # ? Encoding
        h = self.encoding(env, x)
        
        # ? Decoding
        value = self.decoding(env, h)
        return value


class SpecificContinuousQNetwork(SpecificBase):
    def __init__(self, 
                 cfg: OmegaConf,
                env_ids: List[str], 
                 env_list: List[Env],):
        super().__init__(cfg, env_ids, env_list)
        act_encoders_dict = dict()
        decoders_dict = dict()
        for env_id, env in zip(env_ids, env_list):
            if isinstance(env.action_space, GymBox) or \
                isinstance(env.action_space, GymnasiumBox): 
                act_encoder = ResidualMLP(cfg, np.prod(env.action_space.shape), self.d_model)
                act_encoders_dict[env_id] = act_encoder
                q_decoder = ResidualMLP(cfg, self.d_model, 1)
                decoders_dict[env_id] = q_decoder
            else:
                continue
        self.act_encoder_dict = nn.ModuleDict(act_encoders_dict)
        self.value_decoder_dict = nn.ModuleDict(decoders_dict)
    
    def add_act_encoders_q_decoders(self,
                            new_envs: List[Env]):
        for env in new_envs:
            env_id = env.env_id
            if isinstance(env.action_space, GymBox) or \
                isinstance(env.action_space, GymnasiumBox): 
                act_encoder = ResidualMLP(self.cfg, np.prod(env.action_space.shape), self.d_model)
                self.act_encoders_dict[env_id] = act_encoder
                q_decoder = ResidualMLP(self.cfg, self.d_model, 1)
                self.value_decoder_dict[env_id] = q_decoder
            else:
                continue
        
    def encoding(self,
                 env: Env,
                 x: torch.Tensor,
                 a: torch.Tensor):
        # x: [batch_size, feature_dim] or [batch_size, num_frames, H, W]
        env_id = env.env_id
        h = self.act_func(self.encoders_dict[env_id](x))
        a_h = self.act_func(self.act_encoder_dict[env_id](a))
        task_number = self.task_id_int_dict[env_id]
        task_number = torch.tensor(task_number, dtype=torch.int).to(x.device)
        task_embed = self.task_embedding(task_number)
        h = h + task_embed + a_h
        if self.use_mlp:
            h = self.act_func(self.shared_mlp(h))
        else:
            h = self.act_func(h)
        return h
    
    def decoding(self, 
                 env: Env,
                 h:torch.Tensor):
        env_id = env.env_id
        value = self.value_decoder_dict[env_id](h)
        self.value_weights = value 
        return value
        
    def forward(self, 
                env,
                x:torch.Tensor,
                a: torch.Tensor):
        # ? Encoding
        h = self.encoding(env, x, a)
        
        # ? Decoding
        value = self.decoding(env, h)
        return value


class SpecificContinuousTwinQ(nn.Module):
    def __init__(self, 
                 cfg: OmegaConf,
                env_ids: List[str], 
                 env_list: List[Env],
                ):
        super().__init__()
        self.cq1 = SpecificContinuousQNetwork(cfg, env_ids, env_list)
        self.cq2 = SpecificContinuousQNetwork(cfg, env_ids, env_list)

    def both(self, 
             env: Env,
             s:torch.Tensor,
             a:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cq1(env, s, a), self.cq2(env, s, a)

    def forward(self, 
             env: Env,
             s:torch.Tensor,
             a:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.min(*self.both(env, s, a))
    
    def set_finetuning_envs(self, 
                            pretraining_env_ids,
                            new_envs: List[Env]):
        self.cq1.add_envionments(pretraining_env_ids, new_envs)
        self.cq1.add_obs_encoders(new_envs)
        self.cq1.add_act_encoders_q_decoders(new_envs)
        self.cq2.add_envionments(pretraining_env_ids, new_envs)
        self.cq2.add_obs_encoders(new_envs)
        self.cq2.add_act_encoders_q_decoders(new_envs)
        
        
class SpecificDiscreteTwinQ(nn.Module):
    def __init__(self, 
                 cfg: OmegaConf,
                env_ids: List[str], 
                 env_list: List[Env],
                ):
        super().__init__()
        self.dq1 = SpecificDiscreteQNetwork(cfg, env_ids, env_list)
        self.dq2 = SpecificDiscreteQNetwork(cfg, env_ids, env_list)

    def both(self, 
             env: Env,
             s:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dq1(env, s), self.dq2(env, s)

    def forward(self, 
             env: Env,
             s:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.min(*self.both(env, s))
    
    def set_finetuning_envs(self, 
                            pretraining_env_ids,
                            new_envs: List[Env]):
        self.dq1.add_envionments(pretraining_env_ids, new_envs)
        self.dq1.add_obs_encoders(new_envs)
        self.dq1.add_q_decoders(new_envs)
        self.dq2.add_envionments(pretraining_env_ids, new_envs)
        self.dq2.add_obs_encoders(new_envs)
        self.dq2.add_q_decoders(new_envs)
        