
import envpool
from modules.utils import set_seed, make_batched_env
import torch
from tensordict.tensordict import TensorDict
from omegaconf import OmegaConf

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

cfg


env_ids = ["CartPole-v1", "HalfCheetah-v4"]
train_different_envs = []
for j, env_id in enumerate(env_ids):
    train_envs = make_batched_env(j, env_id, cfg, mode='train')
    train_different_envs.append(train_envs)
    print(f"{j+1}/{len(env_ids)}environment {env_id} is loaded...")


device = 'cpu'

envs_storages = TensorDict({}, batch_size=[128, 64])
for i, envs in enumerate(train_different_envs):
    env_id = env_ids[i]
    obs = torch.zeros((128, 64) \
        + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((128, 64) \
        + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((128, 64)).to(device)
    rewards = torch.zeros((128, 64)).to(device)
    dones = torch.zeros((128, 64)).to(device)
    values = torch.zeros((128, 64)).to(device)
    storage = TensorDict({
                "obs": obs,
                "actions": actions,
                "logprobs": logprobs,
                "rewards": rewards,
                "dones": dones,
                "values": values
                }, batch_size=[128, 64])
    envs_storages[env_id] = storage
    
raise ValueError