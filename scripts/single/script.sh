#!/bin/bash

# Define the base command
declare -a seeds=("186" "201" "202" "213" "244")
declare -a devices=("0" "1" "2" "3" "4" "5" "6" "7")

for seed in "${!seeds[@]}"; do
    base_cmd="nohup xvfb-run --auto-servernum python main_ddppo.py experiment.single_env_learning=True nn.env_specific_enc_dec=True distributed.multiprocessing_distributed=False distributed.port=12186 nn.actor_critic.use_compile=True"

    # Define the device and environment pairs
    declare -a env_ids=("BipedalWalker-v3" "MountainCarContinuous-v0" "BipedalWalkerHardcore-v3" "Ant-v4" "Hopper-v4" "HalfCheetah-v4" "Humanoid-v4" "Walker2d-v4")

    # Loop through the devices and environments
    for i in "${!devices[@]}"; do
        device=${devices[$i]}
        env_id=${env_ids[$i]}
        
        # Construct the command
        cmd="$base_cmd experiment.device=$device experiment.seed=$seed 'experiment.pretraining_env_ids=[$env_id]'"
        
        # Execute the command in a new terminal window
        gnome-terminal -- bash -c "$cmd; bash"
    done

    declare -a env_ids2=("CartPole-v1" "Acrobot-v1" "HumanoidStandup-v4" "Pendulum-v1" "InvertedPendulum-v4" "InvertedDoublePendulum-v4" "Swimmer-v4" "Pusher-v4")
    for i in "${!devices[@]}"; do
        device=${devices[$i]}
        env_id=${env_ids2[$i]}
        
        # Construct the command
        cmd="$base_cmd experiment.device=$device experiment.seed=$seed 'experiment.pretraining_env_ids=[$env_id]'"
        
        # Execute the command in a new terminal window
        gnome-terminal -- bash -c "$cmd; bash"
    done

    # Wait for all processes to finish
    base_cmd2="nohup xvfb-run --auto-servernum python main_ddppo.py experiment.num_envs=4 ppo.num_minibatches=8 experiment.single_env_learning=True nn.env_specific_enc_dec=True distributed.multiprocessing_distributed=False distributed.port=12186 nn.actor_critic.use_compile=True"
    declare -a env_ids3=("Pong-v5" "Freeway-v5" "Kaboom-v" "BeamRider-v5" "Breakout-v5" "Frogger-v5" "Seaquest-v5" "Pacman-v5")
    for i in "${!devices[@]}"; do
        device=${devices[$i]}
        env_id=${env_ids2[$i]}
        
        # Construct the command
        cmd="$base_cmd experiment.device=$device experiment.seed=$seed 'experiment.pretraining_env_ids=[$env_id]'"
        
        # Execute the command in a new terminal window
        gnome-terminal -- bash -c "$cmd; bash"
    done
    wait
done
