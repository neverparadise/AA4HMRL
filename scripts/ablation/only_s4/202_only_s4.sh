nohup xvfb-run --auto-servernum python main_ddppo.py wandb.group='only_s4' experiment.finetuning_type='classic-to-mujoco' experiment.seed=202 nn.env_specific_enc_dec=False nn.actor_critic.encoder_net_1d=s4 nn.actor_critic.decoder_net=s4 nn.actor_critic.use_transformer=False distributed.port=12202 'experiment.pretraining_env_ids=[CartPole-v1, LunarLander-v2, BipedalWalker-v3, Acrobot-v1, LunarLanderContinuous-v2, Pendulum-v1, MountainCarContinuous-v0, BipedalWalkerHardcore-v3]' 'experiment.finetuning_env_ids=[ Ant-v4, Hopper-v4, HalfCheetah-v4, InvertedPendulum-v4, Reacher-v4, InvertedDoublePendulum-v4, Walker2d-v4, Humanoid-v4, Swimmer-v4, Pusher-v4, HumanoidStandup-v4]'
nohup xvfb-run --auto-servernum python main_ddppo.py wandb.group='only_s4' experiment.finetuning_type='mujoco-to-classic' experiment.seed=202 nn.env_specific_enc_dec=False nn.actor_critic.encoder_net_1d=s4 nn.actor_critic.decoder_net=s4 nn.actor_critic.use_transformer=False distributed.port=12202 'experiment.pretraining_env_ids=[Ant-v4, Hopper-v4, HalfCheetah-v4, InvertedPendulum-v4, Reacher-v4, InvertedDoublePendulum-v4, Walker2d-v4, Humanoid-v4, Swimmer-v4, Pusher-v4, HumanoidStandup-v4]' 'experiment.finetuning_env_ids=[CartPole-v1, LunarLander-v2, BipedalWalker-v3, Acrobot-v1, LunarLanderContinuous-v2, Pendulum-v1, MountainCarContinuous-v0, BipedalWalkerHardcore-v3]'
nohup xvfb-run --auto-servernum python main_ddppo.py wandb.group='only_s4' experiment.finetuning_type='mix-to-mix' experiment.seed=202 nn.env_specific_enc_dec=False nn.actor_critic.encoder_net_1d=s4 nn.actor_critic.decoder_net=s4 nn.actor_critic.use_transformer=False distributed.port=12202 'experiment.pretraining_env_ids=[CartPole-v1, LunarLander-v2, BipedalWalker-v3, Ant-v4, Hopper-v4, HalfCheetah-v4, InvertedPendulum-v4, Reacher-v4,MountainCarContinuous-v0, HumanoidStandup-v4]' 'experiment.finetuning_env_ids=[Acrobot-v1, LunarLanderContinuous-v2, InvertedDoublePendulum-v4, Walker2d-v4, Humanoid-v4, Swimmer-v4, Pusher-v4, Pendulum-v1, BipedalWalkerHardcore-v3]'
