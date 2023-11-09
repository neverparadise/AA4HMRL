from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import StepCounter
from torchrl.collectors import MultiSyncDataCollector
from torchrl.collectors import SyncDataCollector
from torchrl.envs import TransformedEnv
from tensordict.nn import TensorDictModule
import torch
from torch import nn
from omegaconf import OmegaConf
from modules.networks.blocks import get_encoder_1d, get_encoder_2d, get_decoder
from modules.networks.mlp import ResidualMLP
from modules.utils import get_activation
from torch.distributions import Normal, Categorical
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)

cfg_path = "/home/kukjin/kukjin/Projects/MultiEnvRL/DARL_transformer/configs/ppo_trainer.yaml"
nn_cfg_path = "/home/kukjin/kukjin/Projects/MultiEnvRL/DARL_transformer/configs/nn/nn.yaml"
ppo_cfg_path = "/home/kukjin/kukjin/Projects/MultiEnvRL/DARL_transformer/configs/ppo/ppo.yaml"
ccnn_cfg_img_path = "/home/kukjin/kukjin/Projects/MultiEnvRL/DARL_transformer/configs/ccnn_img/ccnn_img.yaml"
ccnn_cfg_seq_path = "/home/kukjin/kukjin/Projects/MultiEnvRL/DARL_transformer/configs/ccnn_seq/ccnn_seq.yaml"

cfg = OmegaConf.load(cfg_path)
nn_cfg = OmegaConf.load(nn_cfg_path)
ppo_cfg = OmegaConf.load(ppo_cfg_path)
ccnn_seq_cfg = OmegaConf.load(ccnn_cfg_img_path)
ccnn_img_cfg = OmegaConf.load(ccnn_cfg_seq_path)

cfg.nn = nn_cfg
cfg.ppo = ppo_cfg
cfg.ccnn_seq = ccnn_seq_cfg
cfg.ccnn_img = ccnn_img_cfg

print(cfg)

def bmm_input(b_weight, b_input):
    batch_size, feature_dim = b_input.shape
    bmm = torch.einsum('nfh, nf -> nh', b_weight, b_input) / feature_dim
    return bmm


def bmm_output(b_weight, b_input):
    batch_size, output_dim, shared_output_dim = b_weight.shape
    batch_size, shared_output_dim = b_input.shape
    # [batch_size, 6, 32], [batch_size, 32]
    bmm = torch.einsum('noh, nh -> no', b_weight, b_input)  / shared_output_dim
    return bmm

class Actor(nn.Module):
    def __init__(self, cfg: OmegaConf):
        super().__init__()
        activation_name = cfg.nn.actor_critic.activation
        self.act_func = get_activation(activation_name)()
        self.obs_encoder_1d = get_encoder_1d(cfg)
        self.policy_mean_decoder = get_decoder(cfg)
        self.policy_logstd_decoder = get_decoder(cfg)
        self.policy_prob_decoder = get_decoder(cfg)
        self.LOG_STD_MAX = 3
        self.LOG_STD_MIN = -5 
        self.input_to_hidden = cfg.nn.actor_critic.input_to_hidden
        self.hidden_to_output = cfg.nn.actor_critic.hidden_to_output
        self.use_mlp = cfg.nn.actor_critic.use_mlp
        if self.use_mlp:
            self.res_mlp = ResidualMLP(cfg)
    
    def forward(self, is_continuous, output_dim, x):
        print(f"is_continuous: {is_continuous}")
        print(f"output_dim: {output_dim}")
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            is_continuous = is_continuous.unsqueeze(0)
            output_dim = output_dim.unsqueeze(0)
        # x: [batch_size, feature_dim]
        # ? Encoding
        h = self.get_hidden_from_1d_bmm(x)
        # h: [batch_size, d_model]
        h = self.act_func(h) 
        # ? Shared MLP
        if self.use_mlp:
            h = self.act_func(self.res_mlp(h))
        # ? Decoding
        dist, _ = self.get_dist_with_bmm(is_continuous, output_dim, h)
        action = dist.sample()
        action = action.squeeze()
        print(action)
        return action
    
    def get_hidden_from_1d_bmm(self, x):
        h = self.obs_encoder_1d(x)
        h = bmm_input(h, x)
        return h
    
    def get_dist_with_bmm(self, is_continuous, output_dim, h):
        out_dim = output_dim[0]
        if is_continuous[0]:
            a_mean_weights = self.policy_mean_decoder(out_dim, h)
            a_logstd_weights = self.policy_logstd_decoder(out_dim, h)
            # a_mean_weights, a_logstd_weights: [batch_size, act_dim, d_model]
            a_mu = bmm_output(a_mean_weights, h) # out: [batch_size, act_dim]
            a_logstd = bmm_output(a_logstd_weights, h) # out: [batch_size, act_dim]
            a_logstd = torch.tanh(a_logstd)
            a_logstd = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (a_logstd + 1)
            actor_std = a_logstd.exp()
            dist = Normal(a_mu, actor_std)
            return dist, a_mu
        else:
            # generate policy weights
            a_probs_weight = self.policy_prob_decoder(out_dim, h)
            self.policy_prob_weights = a_probs_weight
            logits = bmm_output(a_probs_weight, h) # out: [batch_size, act_dim]
            # get categorical distribution
            dist = Categorical(logits=logits)
            return dist, logits
        
exp_cfg = cfg.experiment
device = torch.device(f"cuda:{exp_cfg.device}" \
                     if torch.cuda.is_available() and exp_cfg.cuda else "cpu")
actor = Actor(cfg).to(device)

if __name__ == '__main__':
    # env_maker1 = lambda: TransformedEnv(GymEnv("Pendulum-v1", device="cpu"),
    env_maker1 = lambda: TransformedEnv(GymEnv("CartPole-v1", device="cpu"),
                                        Compose(
                                            DoubleToFloat(in_keys=["observation"]),
                                            )
                                        )
    env_maker2 = lambda: TransformedEnv(GymEnv("HalfCheetah-v4", device="cpu"),
                                        Compose(
                                            DoubleToFloat(in_keys=["observation"]),
                                            )
                                        )
    # env_maker2 = lambda: TransformedEnv(GymEnv("Pendulum-v1", device="cpu"))
    env = TransformedEnv(GymEnv("Pendulum-v1", device="cpu"))
    policy = TensorDictModule(actor, in_keys=["is_continuous", "output_dim", "observation"], out_keys=["action"])
    # policy = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])
    
    # collector = SyncDataCollector(
    # env,
    # policy,
    # frames_per_batch=50,
    # total_frames=200,
    # split_trajs=False,
    # device='cpu',
    # )
    
    collector = MultiSyncDataCollector(
        create_env_fn=[env_maker1, env_maker2],
        policy=policy,
        total_frames=256,
        # max_frames_per_traj=50,
        frames_per_batch=64,
        init_random_frames=-1,
        reset_at_each_iter=False,
        devices=device,
        storing_devices="cpu",
    )

    for i, data in enumerate(collector):
        print(data)
        if i == 2:
            break