from omegaconf import OmegaConf
import numpy as np

cfg_path = "/home/kukjin/Projects/DARL_transformer/configs/sac_trainer.yaml"
nn_cfg_path = "/home/kukjin/Projects/DARL_transformer/configs/nn/nn.yaml"
sac_cfg_path = "/home/kukjin/Projects/DARL_transformer/configs/sac/sac.yaml"
ccnn_cfg_img_path = "/home/kukjin/Projects/DARL_transformer/configs/ccnn_img/ccnn_img.yaml"
ccnn_cfg_seq_path = "/home/kukjin/Projects/DARL_transformer/configs/ccnn_seq/ccnn_seq.yaml"

cfg = OmegaConf.load(cfg_path)
nn_cfg = OmegaConf.load(nn_cfg_path)
sac_cfg = OmegaConf.load(sac_cfg_path)
ccnn_seq_cfg = OmegaConf.load(ccnn_cfg_img_path)
ccnn_img_cfg = OmegaConf.load(ccnn_cfg_seq_path)

cfg.nn = nn_cfg
cfg.sac = sac_cfg
cfg.ccnn_seq = ccnn_seq_cfg
cfg.ccnn_img = ccnn_img_cfg

cfg

from modules.utils import set_seed, make_batched_env

train_envs = make_batched_env(0, "CartPole-v1", cfg, mode='train')
obs = train_envs.reset()
print(obs)
action = np.array([train_envs.action_space.sample()])
next_obs, reward, done, info = train_envs.step(action)
print(next_obs)
print(info)