from ccnn2.model_constructor import construct_models
from omegaconf import OmegaConf
import torch

cfg_path = "/home/kukjin/kukjin/Projects/MultiEnvRL/DARL_transformer/configs/ppo_trainer.yaml"
nn_cfg_path = "/home/kukjin/kukjin/Projects/MultiEnvRL/DARL_transformer/configs/nn/nn.yaml"
ppo_cfg_path = "/home/kukjin/kukjin/Projects/MultiEnvRL/DARL_transformer/configs/ppo/ppo.yaml"
ccnn_cfg_img_path = "/home/kukjin/kukjin/Projects/MultiEnvRL/DARL_transformer/configs/nn/ccnn_img.yaml"
ccnn_cfg_seq_path = "/home/kukjin/kukjin/Projects/MultiEnvRL/DARL_transformer/configs/nn/ccnn_seq.yaml"

cfg = OmegaConf.load(cfg_path)
nn_cfg = OmegaConf.load(nn_cfg_path)
ppo_cfg = OmegaConf.load(ppo_cfg_path)
ccnn_img_cfg = OmegaConf.load(ccnn_cfg_img_path)
ccnn_seq_cfg = OmegaConf.load(ccnn_cfg_seq_path)

cfg.nn = nn_cfg
cfg.ppo = ppo_cfg
cfg.ccnn_seq = ccnn_seq_cfg
cfg.ccnn_img = ccnn_img_cfg

cfg

img_model, seq_model = construct_models(cfg)

seq_input = torch.randn([32, 1, 1211])
seq_out = seq_model(seq_input)
print(seq_out.shape)

img_input = torch.randn([32, 4, 64, 64])
out = img_model(img_input)
print(out.shape)