import hydra
from omegaconf import OmegaConf
from modules.ddiql_trainer import DDIQLTrainer
import pprint
import os
import torch
import torch.multiprocessing as mp
import numpy as np
import wandb
from pathlib import Path
from modules.iql_agent import IQLAgent
from modules.utils import set_seed, make_envpool_env
import warnings
import gym
import d4rl

warnings.filterwarnings("ignore", category=DeprecationWarning) 

@hydra.main(version_base=None, config_path="./configs", config_name="ddiql_trainer") 
def main(cfg: OmegaConf):
    
    # * distributed setting
    pprint.pprint(cfg)
    print(f'CUDA is available: {torch.cuda.is_available()}')
    print(f'Number of devices: {torch.cuda.device_count()}')
    is_distributed = cfg.distributed.world_size > 1 and cfg.distributed.multiprocessing_distributed
    algorithm = "DDIQL" if is_distributed else "IQL"
    
    # os.environ["CUDA_DEVICE-ORDER"] = "PCI_BUS_ID"
    # ids = str(cfg.distributed.device_ids).strip(']').strip('[')
    # print(ids)
    # os.environ["CUDA_VISIBLE_DEVICES"] = ids
    
    # * wandb init
    wandb_name = algorithm + \
                        '_' + str(cfg.nn.actor_critic.d_model) + \
                        '_' + str(cfg.nn.actor_critic.encoder_net_1d) + \
                        '_'  + str(cfg.nn.actor_critic.encoder_net_2d) + \
                        '_' + str(cfg.nn.actor_critic.decoder_net)
    cfg.wandb.name = wandb_name
    output_dir = str(Path(cfg.paths.dir))
    wandb_logger = wandb.init(
                    dir=output_dir,
                    config=OmegaConf.to_container(cfg, resolve=True),
                    # reinit=True,
                    resume=True,
                    **cfg.wandb
                    )
    
    trainer = DDIQLTrainer(cfg)    
    if is_distributed:
        mp.spawn(
        trainer.distributed_run,
        args=(wandb_logger, ),
        nprocs=cfg.distributed.world_size
    )
        
    else:
        trainer.run(wandb_logger)

if __name__ == "__main__":
    main()
