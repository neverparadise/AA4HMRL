# torch
import torch
import pytorch_lightning as pl

# project
# import ckconv
import ccnn.models as models
# from ccnn.models.lightning_wrappers import (
#     ClassificationWrapper,
#     PyGClassificationWrapper
# )

# typing
from omegaconf import OmegaConf


# def construct_model(
#     cfg: OmegaConf,
#     datamodule: pl.LightningDataModule,
# ):
#     """
#     :param cfg: configuration file
#     :return: An instance of torch.nn.Module
#     """
#     # Get parameters of model from task type
#     data_dim = datamodule.data_dim
#     in_channels = datamodule.input_channels
#     out_channels = datamodule.output_channels
#     data_type = datamodule.data_type

#     # Get type of model from task type
#     net_type = f"{cfg.net.type}_{data_type}"

#     # Overwrite data_dim in cfg.net
#     cfg.net.data_dim = data_dim

#     # Print automatically derived model parameters.
#     print(
#         f"Automatic Parameters:\n dataset = {cfg.dataset.name}, "
#         f"net_name = {net_type},"
#         f" data_dim = {data_dim}"
#         f" in_channels = {in_channels},"
#         f" out_chanels = {out_channels}."
#     )
#     if out_channels == 2:
#         print(
#             "The model will output one single channel. We use BCEWithLogitsLoss for training."
#         )

#     # Create and return model
#     net_type = getattr(models, net_type)
#     network = net_type(
#         in_channels=in_channels,
#         out_channels=out_channels if out_channels != 2 else 1,
#         net_cfg=cfg.net,
#         kernel_cfg=cfg.kernel,
#         conv_cfg=cfg.conv,
#         mask_cfg=cfg.mask,
#     )

#     # Wrap in PytorchLightning
#     if cfg.dataset.name in ["ModelNet"]:
#         Wrapper = PyGClassificationWrapper
#     else:
#         Wrapper = ClassificationWrapper
#     model = Wrapper(
#         network=network,
#         cfg=cfg,
#     )
#     model = Wrapper(
#         network=network,
#         cfg=cfg,
#     )
#     # return model
#     return model


def construct_model_img(
    cfg: OmegaConf,
):
    """
    :param cfg: configuration file
    :return: An instance of torch.nn.Module
    """
    # Get parameters of model from task type
    data_dim = cfg.ccnn_img.net.data_dim
    in_channels = 4
    out_channels = 256
    data_type = "image"

    # Get type of model from task type
    net_type = f"{cfg.ccnn_img.net.type}_{data_type}"

    # Overwrite data_dim in cfg.net
    # cfg.net.data_dim = data_dim

    # Print automatically derived model parameters.
    print(
        # f"Automatic Parameters:\n dataset = {cfg.dataset.name}, "
        f"net_name = {net_type},"
        f" data_dim = {data_dim}"
        f" in_channels = {in_channels},"
        f" out_chanels = {out_channels}."
    )
    if out_channels == 2:
        print(
            "The model will output one single channel. We use BCEWithLogitsLoss for training."
        )

    # Create and return model
    net_type = getattr(models, net_type)
    network = net_type(
        in_channels=in_channels,
        out_channels=out_channels if out_channels != 2 else 1,
        net_cfg=cfg.ccnn_img.net,
        kernel_cfg=cfg.ccnn_img.kernel,
        conv_cfg=cfg.ccnn_img.conv,
        mask_cfg=cfg.ccnn_img.mask,
    )

    return network


def construct_model_seq(
    cfg: OmegaConf,
):
    """
    :param cfg: configuration file
    :return: An instance of torch.nn.Module
    """
    # Get parameters of model from task type
    data_dim = cfg.ccnn_seq.net.data_dim
    in_channels = 1
    out_channels = 256
    data_type = "sequence"

    # Get type of model from task type
    net_type = f"{cfg.ccnn_seq.net.type}_{data_type}"

    # Overwrite data_dim in cfg.net
    # cfg.net.data_dim = data_dim

    # Print automatically derived model parameters.
    print(
        # f"Automatic Parameters:\n dataset = {cfg.dataset.name}, "
        f"net_name = {net_type},"
        f" data_dim = {data_dim}"
        f" in_channels = {in_channels},"
        f" out_chanels = {out_channels}."
    )
    if out_channels == 2:
        print(
            "The model will output one single channel. We use BCEWithLogitsLoss for training."
        )

    # Create and return model
    net_type = getattr(models, net_type)
    network = net_type(
        in_channels=in_channels,
        out_channels=out_channels if out_channels != 2 else 1,
        net_cfg=cfg.ccnn_seq.net,
        kernel_cfg=cfg.ccnn_seq.kernel,
        conv_cfg=cfg.ccnn_seq.conv,
        mask_cfg=cfg.ccnn_seq.mask,
    )

    return network


from omegaconf import OmegaConf

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

img_model = construct_model_img(cfg)
seq_model = construct_model_seq(cfg)

seq_input = torch.randn([32, 1, 1211])
out = seq_model(seq_input)
print(out.shape)

img_input = torch.randn([32, 4, 256, 256])
out = img_model(img_input)
print(out.shape)
