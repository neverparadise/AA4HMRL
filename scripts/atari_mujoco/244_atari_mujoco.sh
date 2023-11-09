nohup xvfb-run --auto-servernum python main_ddppo.py experiment.finetuning_type='atari_mujoco_mix-to-mix' experiment.seed=244 nn.env_specific_enc_dec=False nn.actor_critic.encoder_net_1d=s4 nn.actor_critic.decoder_net=s4 distributed.port=12244 'experiment.pretraining_env_ids=[Pong-v5, Swimmer-v4, Freeway-v5, HalfCheetah-v4, Kaboom-v5, Walker2d-v4, Breakout-v5, Ant-v4]' 'experiment.finetuning_env_ids=[Frogger-v5, Humanoid-v4, Seaquest-v5, BipedalWalker-v3, Pacman-v5, InvertedDoublePendulum-v4, BeamRider-v5, Hopper-v4]'
