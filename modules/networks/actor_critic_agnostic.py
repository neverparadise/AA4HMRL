import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, MultivariateNormal
from omegaconf import OmegaConf
from modules.networks.blocks import get_encoder_1d, get_encoder_2d, get_decoder
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


def bmm_input(b_weight: torch.Tensor, b_input: torch.Tensor, scale=False) -> torch.Tensor:
    batch_size, feature_dim = b_input.shape
    if scale:
        bmm = torch.einsum('nfh, nf -> nh', b_weight, b_input) / math.sqrt(feature_dim)
    else:
        bmm = torch.einsum('nfh, nf -> nh', b_weight, b_input)
    return bmm


def bmm_output(b_weight: torch.Tensor, b_input: torch.Tensor, scale=False) -> torch.Tensor:
    batch_size, output_dim, d_model = b_weight.shape
    batch_size, d_model = b_input.shape
    # [batch_size, 6, 32], [batch_size, 32]
    if scale:
        bmm = torch.einsum('noh, nh -> no', b_weight, b_input)  / math.sqrt(d_model)
    else:
        bmm = torch.einsum('noh, nh -> no', b_weight, b_input)
    return bmm


class AgnosticBase(nn.Module):
    def __init__(self, 
                 cfg: OmegaConf,
                env_ids: List[str], 
                 env_list: List[Env],):
        super().__init__()
        self.d_model = cfg.nn.actor_critic.d_model
        activation_name: str = cfg.nn.actor_critic.activation
        self.act_func: torch.nn.Module = get_activation(activation_name)()
        self.use_transformer: bool = cfg.nn.actor_critic.use_transformer
        self.obs_encoder_1d: torch.nn.Module = get_encoder_1d(cfg)
        if self.use_transformer:
            self.obs_transformer: torch.nn.Module = TransformerBlock(cfg) # policy mean
        contain_2d_env = cfg.nn.contain_2d_env 
        if contain_2d_env: 
            self.obs_encoder_2d: torch.nn.Module = get_encoder_2d(cfg)
        self.input_to_hidden: bool = cfg.nn.actor_critic.input_to_hidden
        self.hidden_to_output: bool = cfg.nn.actor_critic.hidden_to_output
        self.use_mlp: bool = cfg.nn.actor_critic.use_mlp
        self.last_op_name: str = cfg.nn.actor_critic.last_operation
        if self.last_op_name == "mean":
            self.last_op = torch.mean
        elif self.last_op_name == "sum":
            self.last_op = torch.sum
    
        if self.use_mlp:
            input_dim = self.d_model
            output_dim = self.d_model
            self.res_mlp: torch.nn.Module = ResidualMLP(cfg, input_dim, output_dim)
        self.env_ids = env_ids[:]
        self.task_id_int_dict = dict()
        for task_id, number in zip(env_ids, range(len(env_ids))):
            self.task_id_int_dict[task_id] = number
        self.task_embedding = nn.Embedding(len(self.task_id_int_dict), self.d_model)
    
    def encoding(self,
                 env: Env,
                 x: torch.Tensor):
        env_id = env.env_id
        # ? Encoding from 1D 
        # x: [batch_size, feature_dim]
        if len(x.shape) == 2:
            h = self.obs_encoder_1d(x)
            if self.use_transformer:
                h, obs_attn_maps = self.obs_transformer(h)
            if self.input_to_hidden== 'pooling':
                h = h.mean(dim=1, keepdim=False)
            elif self.input_to_hidden == 'bmm':
                h = bmm_input(h, x)
                
        # ? Encoding from 2D
        # x: [batch_size, num_frames, H, W]
        elif len(x.shape) == 4:
            h = self.obs_encoder_2d(x)
        task_number = self.task_id_int_dict[env_id]
        task_number = torch.tensor(task_number, dtype=torch.int).to(x.device)
        task_embed = self.task_embedding(task_number)
        h = h + task_embed
        if self.use_mlp:
            h = self.act_func(self.res_mlp(h))
        else:
            h = self.act_func(h)
        return h
    
    def set_finetuning_envs(self, 
                            pretraining_env_ids,
                            finetuning_env_list: List[Env]):
        # self.env_ids: id list of environments
        # self.task_id_int_dict: key: id, value: index of environment
        
        # generate new integers from new environments
        print(f"self env ids {self.env_ids}")
        pretrained_num_embeddings = len(pretraining_env_ids)
        finetuning_env_ids = []
        for env in finetuning_env_list:
            env_id = env.env_id
            self.env_ids.append(env_id)
            finetuning_env_ids.append(env_id)
        print(f"self env ids {self.env_ids}")
        new_integers = range(len(pretraining_env_ids), len(self.env_ids))

        # add the key and value of new environment
        for task_id, number in zip(finetuning_env_ids, new_integers):
            self.task_id_int_dict[task_id] = number
        
        # extend task embedding layer
        extended_num_embeddings = len(self.env_ids)
        print(f"extended_num_embeddings {extended_num_embeddings}")
        new_task_embedding = nn.Embedding(extended_num_embeddings, self.d_model).to(self.task_embedding.weight.device)
        print(f"new_task_embedding shape: {new_task_embedding.weight.shape}")
        with torch.no_grad():
            new_task_embedding.weight[:pretrained_num_embeddings] = self.task_embedding.weight
        self.task_embedding = new_task_embedding
    
    def decoding(self):
        pass
    
    def forward(self):
        pass
        
class AgnosticStochasticActor(AgnosticBase):
    def __init__(self, 
                 cfg: OmegaConf,
                env_ids: List[str], 
                 env_list: List[Env],
                 ):
        super().__init__(cfg, env_ids, env_list)
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5
        self.policy_mean_decoder: torch.nn.Module = get_decoder(cfg)
        self.policy_logstd_decoder: torch.nn.Module = get_decoder(cfg)
        self.policy_prob_decoder: torch.nn.Module = get_decoder(cfg)
        if self.use_transformer:
            self.pmd_transformer: torch.nn.Module = TransformerBlock(cfg) # policy mean
            self.pld_transformer: torch.nn.Module = TransformerBlock(cfg) # policy logstd
            self.ppd_transformer: torch.nn.Module = TransformerBlock(cfg) # policy prob

    def decoding(self,
                 action_space: GymSpace,
                 h: torch.Tensor):
        # ? Decoding continuous action
        if isinstance(action_space, GymBox) or isinstance(action_space, GymnasiumBox):
            act_dim = np.prod(action_space.shape)
            a_mean = self.policy_mean_decoder(act_dim, h)
            a_logstd = self.policy_logstd_decoder(act_dim, h)
            if self.use_transformer:
                a_mean, mean_attn_maps = self.pmd_transformer(a_mean)
                a_logstd, logstd_attn_maps = self.pld_transformer(a_logstd)
            self.a_mean_weights = a_mean
            self.a_logstd_weights = a_logstd
            # a_mean, a_logstd: [batch_size, act_dim, d_model]
            if self.hidden_to_output == 'pooling':
                a_mu = self.last_op(a_mean, dim=2, keepdim=False) # out: [batch_size, act_dim]
                a_logstd = self.last_op(a_logstd, dim=2, keepdim=False) # out: [batch_size, act_dim]
            elif self.hidden_to_output == 'bmm':
                a_mu = bmm_output(a_mean, h, scale=True) # out: [batch_size, act_dim]
                a_logstd = bmm_output(a_logstd, h, scale=True) # out: [batch_size, act_dim]
            a_logstd = torch.tanh(a_logstd)
            a_logstd = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (a_logstd + 1)
            actor_std = a_logstd.exp()
            dist = Normal(a_mu, actor_std)
            return dist, a_mu

        # ? Decoding discrete action
        elif isinstance(action_space, GymDiscrete) or isinstance(action_space, GymnasiumDiscrete):
            # get num_discretes
            num_discretes = action_space.n
            # generate policy weights
            a_probs = self.policy_prob_decoder(num_discretes, h)
            if self.use_transformer:
                a_probs, prob_attn_maps = self.ppd_transformer(a_probs)
            self.a_prob_weights = a_probs
            if self.hidden_to_output == 'pooling':
                logits = self.last_op(a_probs, dim=2, keepdim=False)
            elif self.hidden_to_output == 'bmm':
                logits = bmm_output(a_probs, h) # out: [batch_size, act_dim]
            # get categorical distribution
            dist = Categorical(logits=logits)
            return dist, logits
    
    def forward(self, 
                env: Env,
                x: torch.Tensor):
        action_space = env.action_space
        # ? Encoding
        # h: [batch_size, d_model]
        h = self.encoding(env, x)
        
        # ? Decoding
        dist, _ = self.decoding(action_space, h)
        return dist, _


#TODO: Implement deterministic continuous action actor and discrete action actor
# class AgnosticDeterministicActor(AgnosticBase):
#     def __init__(self, 
#                  cfg: OmegaConf,
#                 env_ids: List[str], 
#                  env_list: List[Env],
#                  ):
#         super().__init__(cfg, env_ids, env_list)
#         self.env_specific_enc_dec: bool = cfg.nn.actor_critic.env_specific_enc_dec
#         self.LOG_STD_MAX = 2
#         self.LOG_STD_MIN = -5
#         self.policy_mean_decoder: torch.nn.Module = get_decoder(cfg)
#         self.policy_prob_decoder: torch.nn.Module = get_decoder(cfg)
#         if self.use_transformer:
#             self.pmd_transformer: torch.nn.Module = TransformerBlock(cfg) # policy mean
#             self.pld_transformer: torch.nn.Module = TransformerBlock(cfg) # policy logstd
#             self.ppd_transformer: torch.nn.Module = TransformerBlock(cfg) # policy prob

#     def decoding(self,
#                  action_space: GymSpace,
#                  h: torch.Tensor):
#         # ? Decoding continuous action
#         if isinstance(action_space, GymBox) or isinstance(action_space, GymnasiumBox):
#             act_dim = np.prod(action_space.shape)
#             a_mean = self.policy_mean_decoder(act_dim, h)
#             a_logstd = self.policy_logstd_decoder(act_dim, h)
#             if self.use_transformer:
#                 a_mean, mean_attn_maps = self.pmd_transformer(a_mean)
#                 a_logstd, logstd_attn_maps = self.pld_transformer(a_logstd)
#             self.a_mean_weights = a_mean
#             self.a_logstd_weights = a_logstd
#             # a_mean, a_logstd: [batch_size, act_dim, d_model]
#             if self.hidden_to_output == 'pooling':
#                 a_mu = self.last_op(a_mean, dim=2, keepdim=False) # out: [batch_size, act_dim]
#                 a_logstd = self.last_op(a_logstd, dim=2, keepdim=False) # out: [batch_size, act_dim]
#             elif self.hidden_to_output == 'bmm':
#                 a_mu = bmm_output(a_mean, h, scale=True) # out: [batch_size, act_dim]
#                 a_logstd = bmm_output(a_logstd, h, scale=True) # out: [batch_size, act_dim]
#             a_logstd = torch.tanh(a_logstd)
#             a_logstd = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (a_logstd + 1)
#             actor_std = a_logstd.exp()
#             dist = Normal(a_mu, actor_std)
#             return dist, a_mu

#         # ? Decoding discrete action
#         elif isinstance(action_space, GymDiscrete) or isinstance(action_space, GymnasiumDiscrete):
#             # get num_discretes
#             num_discretes = action_space.n
#             # generate policy weights
#             a_probs = self.policy_prob_decoder(num_discretes, h)
#             if self.use_transformer:
#                 a_probs, prob_attn_maps = self.ppd_transformer(a_probs)
#             self.a_prob_weights = a_probs
#             if self.hidden_to_output == 'pooling':
#                 logits = self.last_op(a_probs, dim=2, keepdim=False)
#             elif self.hidden_to_output == 'bmm':
#                 logits = bmm_output(a_probs, h) # out: [batch_size, act_dim]
#             # get categorical distribution
#             dist = Categorical(logits=logits)
#             return dist, logits
    
#     def forward(self, 
#                 env: Env,
#                 x: torch.Tensor):
#         action_space = env.action_space
#         # ? Encoding
#         # h: [batch_size, d_model]
#         h = self.encoding(env, x)
        
#         # ? Decoding
#         dist, _ = self.decoding(action_space, h)
#         return dist, _

        
class AgnosticVNetwork(AgnosticBase):
    def __init__(self, 
                 cfg: OmegaConf,
                env_ids: List[str], 
                 env_list: List[Env],
                 ):
        super().__init__(cfg, env_ids, env_list)
        self.value_decoder: torch.nn.Module = get_decoder(cfg)
        if self.use_transformer:
            self.value_transformer: torch.nn.Module = TransformerBlock(cfg) # policy mean

    def decoding(self, 
                 h:torch.Tensor):
        value = self.value_decoder(1, h)
        if self.use_transformer:
            value, value_attn_maps = self.value_transformer(value)
        self.value_weights = value 
        if self.hidden_to_output == 'pooling':
            value = self.last_op(value, dim=2, keepdim=False) 
        elif self.hidden_to_output == 'bmm':
            value = bmm_output(value, h)
        return value
        
    def forward(self, 
                env: Env,
                x:torch.Tensor):
        # x: [batch_size, feature_dim]
        # ? Encoding
        h = self.encoding(env, x)
        # h: [batch_size, d_model]
    
        # ? Decoding
        value = self.decoding(h)
        return value


class AgnosticDiscreteQNetwork(AgnosticBase):
    def __init__(self, 
                 cfg: OmegaConf,
                env_ids: List[str], 
                 env_list: List[Env],
                 ):
        super().__init__(cfg, env_ids, env_list)
        self.value_decoder: torch.nn.Module = get_decoder(cfg)
        if self.use_transformer:
            self.value_transformer: torch.nn.Module = TransformerBlock(cfg) # policy mean

    def decoding(self, 
                 num_discretes,
                 h:torch.Tensor):
        value = self.value_decoder(num_discretes, h)
        if self.use_transformer:
            value, value_attn_maps = self.value_transformer(value)
        self.value_weights = value 
        if self.hidden_to_output == 'pooling':
            value = self.last_op(value, dim=2, keepdim=False) 
        elif self.hidden_to_output == 'bmm':
            value = bmm_output(value, h)
        return value
        
    def forward(self, 
                env: Env,
                x:torch.Tensor):
        # x: [batch_size, feature_dim]
        # ? Encoding
        h = self.encoding(env, x)
        # h: [batch_size, d_model]
        num_discrete = env.action_space.n
        # ? Decoding
        value = self.decoding(num_discrete, h)
        return value


class AgnosticContinuousQNetwork(AgnosticBase):
    def __init__(self, 
                 cfg: OmegaConf,
                env_ids: List[str], 
                 env_list: List[Env],
                 ):
        super().__init__(cfg, env_ids, env_list)
        self.act_encoder_1d: torch.nn.Module = get_encoder_1d(cfg)
        self.value_decoder: torch.nn.Module = get_decoder(cfg)
        if self.use_transformer:
            self.obs_transformer: torch.nn.Module = TransformerBlock(cfg) # policy mean
            self.act_transformer: torch.nn.Module = TransformerBlock(cfg) # policy mean
            self.value_transformer: torch.nn.Module = TransformerBlock(cfg) # policy mean

    # method overriding
    def encoding(self,
                 env: Env,
                 x: torch.Tensor,
                 a: torch.Tensor):
        env_id = env.env_id
        # ? Encoding from 1D 
        # x: [batch_size, feature_dim]
        if len(x.shape) == 2:
            h = self.obs_encoder_1d(x)
            a_h = self.act_encoder_1d(a)
            if self.use_transformer:
                h, obs_attn_maps = self.obs_transformer(h)
                a_h, act_attn_maps = self.act_transformer(a_h)
            if self.input_to_hidden== 'pooling':
                h = h.mean(dim=1, keepdim=False)
                a_h = a_h.mean(dim=1, keepdim=False)
            elif self.input_to_hidden == 'bmm':
                h = bmm_input(h, x)
                a_h = bmm_input(a_h, a)
            
        # ? Encoding from 2D
        # x: [batch_size, num_frames, H, W]
        elif len(x.shape) == 4:
            h = self.obs_encoder_2d(x)
        task_number = self.task_id_int_dict[env_id]
        task_number = torch.tensor(task_number, dtype=torch.int).to(x.device)
        task_embed = self.task_embedding(task_number)
        h = h + task_embed + a_h
        if self.use_mlp:
            h = self.act_func(self.res_mlp(h))
        else:
            h = self.act_func(h)
        return h
    
    def decoding(self, 
                 h:torch.Tensor):
        value = self.value_decoder(1, h)
        if self.use_transformer:
            value, value_attn_maps = self.value_transformer(value)
        self.value_weights = value 
        if self.hidden_to_output == 'pooling':
            value = self.last_op(value, dim=2, keepdim=False) 
        elif self.hidden_to_output == 'bmm':
            value = bmm_output(value, h)
        return value
        
    def forward(self, 
                env: Env,
                x:torch.Tensor,
                a: torch.Tensor):
        # x: [batch_size, feature_dim]
        # ? Encoding
        h = self.encoding(env, x, a)
        # h: [batch_size, d_model]
    
        # ? Decoding
        value = self.decoding(h)
        return value


class AgnosticContinuousTwinQ(nn.Module):
    def __init__(self, 
                 cfg: OmegaConf,
                env_ids: List[str], 
                 env_list: List[Env],
                ):
        super().__init__()
        self.cq1 = AgnosticContinuousQNetwork(cfg, env_ids, env_list)
        self.cq2 = AgnosticContinuousQNetwork(cfg, env_ids, env_list)

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
    
        
class AgnosticDiscreteTwinQ(nn.Module):
    def __init__(self, 
                 cfg: OmegaConf,
                env_ids: List[str], 
                 env_list: List[Env],
                ):
        super().__init__()
        self.dq1 = AgnosticDiscreteQNetwork(cfg, env_ids, env_list)
        self.dq2 = AgnosticDiscreteQNetwork(cfg, env_ids, env_list)

    def both(self, 
             env: Env,
             s:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dq1(env, s), self.dq2(env, s)

    def forward(self, 
             env: Env,
             s:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.min(*self.both(env, s))
    
        