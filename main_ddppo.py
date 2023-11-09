import hydra
from omegaconf import OmegaConf
from modules.ddppo_trainer import DDPPOTrainer
import pprint
import os
import torch
import torch.multiprocessing as mp
import numpy as np
import wandb
from pathlib import Path
from modules.ppo_agent import PPOAgent
from modules.utils import set_seed, make_envpool_env
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

@hydra.main(version_base=None, config_path="./configs", config_name="ddppo_trainer") 
def main(cfg: OmegaConf):
    
    # * distributed setting
    pprint.pprint(cfg)
    print(f'CUDA is available: {torch.cuda.is_available()}')
    print(f'Number of devices: {torch.cuda.device_count()}')
    cfg.distributed.world_size = torch.cuda.device_count()
    
    is_distributed = cfg.distributed.world_size > 1 and cfg.distributed.multiprocessing_distributed
    algorithm = "DDPPO" if is_distributed else "PPO"
    
    # os.environ["CUDA_DEVICE-ORDER"] = "PCI_BUS_ID"
    # ids = str(cfg.distributed.device_ids).strip(']').strip('[')
    # print(ids)
    # os.environ["CUDA_VISIBLE_DEVICES"] = ids
    
    # * wandb init
    if cfg.nn.env_specific_enc_dec and cfg.experiment.single_env_learning:
        architecture = 'single'
        wandb_name = algorithm + \
                            '_' + architecture + \
                            '_' + str(cfg.experiment.seed) + \
                            '_' + str(cfg.experiment.pretraining_env_ids) + \
                            '_' + str(cfg.nn.actor_critic.d_model) 
        cfg.wandb.group = list(cfg.experiment.pretraining_env_ids)[0]
    elif cfg.nn.env_specific_enc_dec and not cfg.experiment.single_env_learning:
        architecture = 'specific'
        wandb_name = algorithm + \
                            '_' + architecture + \
                            '_' + str(cfg.experiment.seed) + \
                            '_' + str(cfg.experiment.finetuning_type) + \
                            '_' + str(cfg.distributed.port) + \
                            '_' + str(cfg.nn.actor_critic.d_model) 
        cfg.wandb.group = architecture + str(cfg.experiment.finetuning_type)
                            
    else:
        architecture = 'agnostic'
        wandb_name = algorithm + \
                            '_' + architecture + \
                            '_' + str(cfg.experiment.seed) + \
                            '_' + str(cfg.experiment.finetuning_type) + \
                            '_' + str(cfg.distributed.port) + \
                            '_' + str(cfg.nn.actor_critic.d_model) + \
                            '_' + str(cfg.nn.actor_critic.encoder_net_1d) + \
                            '_'  + str(cfg.nn.actor_critic.encoder_net_2d) + \
                            '_' + str(cfg.nn.actor_critic.decoder_net)
        cfg.wandb.group = architecture + str(cfg.experiment.finetuning_type)
                            
    cfg.wandb.name = wandb_name
    output_dir = str(Path(cfg.paths.dir))
    wandb_logger = wandb.init(
                    dir=output_dir,
                    config=OmegaConf.to_container(cfg, resolve=True),
                    # reinit=True,
                    resume=True,
                    **cfg.wandb
                    )
    
    trainer = DDPPOTrainer(cfg)    
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
