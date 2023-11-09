import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium
from gymnasium.wrappers.record_video import RecordVideo as GymnasiumRecordVideo
from gym.wrappers.record_video import RecordVideo as GymRecordVideo

import gym
from gym.spaces import Box as GymBox
from gym.spaces import Discrete as GymDiscrete
from gymnasium.spaces import Box as GymnasiumBox
from gymnasium.spaces import Discrete as GymnasiumDiscrete
from gym.spaces.dict import Dict as GymDict
# from gym.wrappers.record_video import RecordVideo
# from gymnasium.experimental.wrappers.rendering import RecordVideoV0 as RecordVideo
from omegaconf import OmegaConf
from typing import Dict, OrderedDict
import pathlib
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
import os
import envpool
from typing import Callable, Tuple, List

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

# cfg, env_name, seed + i, i,

gym_envs = [
'CartPole-v1',
'MountainCar-v0',
'MountainCarContinuous-v0',
'Pendulum-v1',
'Acrobot-v1',
'BipedalWalker-v3',
'BipedalWalkerHardcore-v3',
'LunarLander-v2',
'LunarLanderContinuous-v2',
]
dmc_envs = [
'AcrobotSwingup-v1',
 'AcrobotSwingupSparse-v1',
 'BallInCupCatch-v1',
'CartpoleBalance-v1',
 'CartpoleBalanceSparse-v1',
 'CartpoleSwingup-v1',
 'CartpoleSwingupSparse-v1',
 'CartpoleThreePoles-v1',
 'CartpoleTwoPoles-v1',
  'CheetahRun-v1',
 'FingerSpin-v1',
 'FingerTurnEasy-v1',
 'FingerTurnHard-v1',
 'FishSwim-v1',
 'FishUpright-v1',
 'HopperHop-v1',
 'HopperStand-v1',
 'HumanoidRun-v1',
 'HumanoidRunPureState-v1',
 'HumanoidStand-v1',
 'HumanoidWalk-v1',
 'HumanoidCMURun-v1',
 'HumanoidCMUStand-v1',
 'ManipulatorBringBall-v1',
 'ManipulatorBringPeg-v1',
 'ManipulatorInsertBall-v1',
 'ManipulatorInsertPeg-v1',
 'PendulumSwingup-v1',
 'PointMassEasy-v1',
 'PointMassHard-v1',
 'ReacherEasy-v1',
 'ReacherHard-v1',
 'SwimmerSwimmer6-v1',
 'SwimmerSwimmer15-v1',
 'WalkerRun-v1',
 'WalkerStand-v1',
 'WalkerWalk-v1', 
]
mujoco_envs = [ 
'Ant-v4',
'HalfCheetah-v4',
'Hopper-v4',
'InvertedDoublePendulum-v4',
'InvertedPendulum-v4',
    'Humanoid-v4',
'HumanoidStandup-v4',
'Pusher-v4',
'Reacher-v4',
'Swimmer-v4',
'Walker2d-v4',]

image_envs = [
'Adventure-v5',
 'AirRaid-v5',
 'Alien-v5',
 'Amidar-v5',
 'Assault-v5',
 'Asterix-v5',
 'Asteroids-v5',
 'Atlantis-v5',
 'Atlantis2-v5',
 'Backgammon-v5',
 'BankHeist-v5',
 'BasicMath-v5',
 'BattleZone-v5',
 'BeamRider-v5',
 'Berzerk-v5',
 'Blackjack-v5',
 'Bowling-v5',
 'Boxing-v5',
 'Breakout-v5',
 'Carnival-v5',
 'Casino-v5',
 'Centipede-v5',
 'ChopperCommand-v5',
 'CrazyClimber-v5',
 'Crossbow-v5',
 'Darkchambers-v5',
 'Defender-v5',
 'DemonAttack-v5',
 'DonkeyKong-v5',
 'DoubleDunk-v5',
 'Earthworld-v5',
 'ElevatorAction-v5',
 'Enduro-v5',
 'Entombed-v5',
 'Et-v5',
 'FishingDerby-v5',
 'FlagCapture-v5',
 'Freeway-v5',
 'Frogger-v5',
 'Frostbite-v5',
 'Galaxian-v5',
 'Gopher-v5',
 'Gravitar-v5',
 'Hangman-v5',
 'HauntedHouse-v5',
 'Hero-v5',
 'HumanCannonball-v5',
 'IceHockey-v5',
 'Jamesbond-v5',
 'JourneyEscape-v5',
 'Kaboom-v5',
 'Kangaroo-v5',
 'KeystoneKapers-v5',
 'KingKong-v5',
 'Klax-v5',
 'Koolaid-v5',
 'Krull-v5',
 'KungFuMaster-v5',
 'LaserGates-v5',
 'LostLuggage-v5',
 'MarioBros-v5',
 'MiniatureGolf-v5',
 'MontezumaRevenge-v5',
 'MrDo-v5',
 'MsPacman-v5',
 'NameThisGame-v5',
 'Othello-v5',
 'Pacman-v5',
 'Phoenix-v5',
 'Pitfall-v5',
 'Pitfall2-v5',
 'Pong-v5',
 'Pooyan-v5',
 'PrivateEye-v5',
 'Qbert-v5',
 'Riverraid-v5',
 'RoadRunner-v5',
 'Robotank-v5',
 'Seaquest-v5',
 'SirLancelot-v5',
 'Skiing-v5',
 'Solaris-v5',
 'SpaceInvaders-v5',
 'SpaceWar-v5',
 'StarGunner-v5',
 'Superman-v5',
 'Surround-v5',
 'Tennis-v5',
 'Tetris-v5',
 'TicTacToe3d-v5',
 'TimePilot-v5',
 'Trondead-v5',
 'Turmoil-v5',
 'Tutankham-v5',
 'UpNDown-v5',
 'Venture-v5',
 'VideoCheckers-v5',
 'VideoChess-v5',
 'VideoCube-v5',
 'VideoPinball-v5',
 'WizardOfWor-v5',
 'WordZapper-v5',
 'YarsRevenge-v5',
 'Zaxxon-v5',
 'CarRacing-v2',
]




class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )
        
def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class FlattenRecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        if isinstance(env.observation_space, GymDict) or \
            isinstance(env.observation_space, Dict) or \
            isinstance(env.observation_space, OrderedDict):
            size, highs, lows = self.get_obs_space(env.observation_space)
            self.observation_space = GymBox(low=lows, high=highs)
            # print("observation is changed")
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        if isinstance(observations, Dict) or \
            isinstance(observations, GymDict) or \
                isinstance(observations, OrderedDict):
            observations =  self.flatten_dict(observations)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        if isinstance(observations, Dict) or \
            isinstance(observations, GymDict) or \
                isinstance(observations, OrderedDict):
            observations = self.flatten_dict(observations)
        self.episode_returns += rewards #infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - dones # infos["terminated"]
        self.episode_lengths *= 1 - dones # infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        
        return (
            observations,
            rewards,
            dones,
            infos,
        )
        
    def flatten_dict(self, obs):
        obs_pieces = []
        for v in obs.values():
            flat = np.array([v]) if np.isscalar(v) else v.ravel()
            obs_pieces.append(flat)
        obs = np.concatenate(obs_pieces, axis=-1)
        obs = obs.reshape(self.num_envs, -1)
        return obs

    def get_obs_space(self, observation_space):
        shapes = []
        highs = []
        lows = []
        for key, box in observation_space.items():
            if len(box.shape) == 0:
                shapes.append(1)
                # highs.append()
                # lows.append()
                highs += [*np.expand_dims(box.high, axis=-1)]
                lows += [*np.expand_dims(box.low, axis=-1)]
            elif len(box.shape) == 1:
                shapes += [*box.shape]
                highs += [*box.high]
                lows += [*box.low]
            elif len(box.shape) > 1:
                # print(box)
                shapes += [np.prod(box.shape)]
                highs += [*(box.high.reshape(-1))]
                lows += [*(box.low.reshape(-1))]
        # print(shapes)
        # print(highs)
        # print(lows)
        size = (np.sum(shapes, dtype=np.int32))
        highs = np.array(highs)
        lows = np.array(lows)        
        return size, highs, lows

class GymnasiumNormalizedFlattenRecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, cfg, deque_size=100):
        super().__init__(env)
        if isinstance(env.observation_space, GymDict) or \
            isinstance(env.observation_space, Dict) or \
            isinstance(env.observation_space, OrderedDict):
            size, highs, lows = self.get_obs_space(env.observation_space)
            self.observation_space = GymBox(low=lows, high=highs)
            # print("observation is changed")
        self.observation_space = GymnasiumBox(low=env.observation_space.low, high=env.observation_space.high, shape=env.observation_space._shape)
        print(f"gymnasium env observation space: {env.observation_space}")
        print(f"gymnasium env observation space type: {type(env.observation_space)}")
        print(f"gymnasium env action space: {env.action_space}")
        print(f"gymnasium env action space type: {type(env.action_space)}")
        
        if isinstance(env.action_space, GymBox) or isinstance(env.action_space, GymnasiumBox):
            self.action_space = GymnasiumBox(low=env.action_space.low, high=env.action_space.high, shape=env.action_space._shape)
        elif isinstance(env.action_space, GymDiscrete) or isinstance(env.action_space, GymnasiumDiscrete):
            self.action_space = GymnasiumDiscrete(env.action_space.n)
        
        
        self.env_type = cfg.experiment.env_type
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None
        self.obs_rms = RunningMeanStd(shape=(self.num_envs, *self.observation_space.shape))
        self.return_rms = RunningMeanStd(shape=(self.num_envs, ))
        self.gamma = 0.98
        self.epsilon = 1e-8
        
    def normalize_obs(self, obs):
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
    
    def normalize_rew(self, rews):
        self.return_rms.update(self.returned_episode_returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)
        
    def reset(self, **kwargs):
        # print(f"Wrapper reset kwargs: {kwargs}")
        observations, infos = super().reset()
        if isinstance(observations, Dict) or \
            isinstance(observations, GymDict) or \
                isinstance(observations, OrderedDict):
            observations =  self.flatten_dict(observations)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        observations = self.normalize_obs(observations)
        return observations, infos

    def step(self, action):
        observations, rewards, terminated, truncated, infos = super().step(action)
        dones = np.logical_or(terminated, truncated)
        if isinstance(observations, Dict) or \
            isinstance(observations, GymDict) or \
                isinstance(observations, OrderedDict):
            observations = self.flatten_dict(observations)
        self.episode_returns += rewards #infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - dones # infos["terminated"]
        self.episode_lengths *= 1 - dones # infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        observations = self.normalize_obs(observations)
        rewards = self.normalize_rew(rewards)
        return (
                observations,
                rewards,
                terminated,
                truncated,
                infos,
            )
        
    def flatten_dict(self, obs):
        obs_pieces = []
        for v in obs.values():
            flat = np.array([v]) if np.isscalar(v) else v.ravel()
            obs_pieces.append(flat)
        obs = np.concatenate(obs_pieces, axis=-1)
        obs = obs.reshape(self.num_envs, -1)
        return obs

    def get_obs_space(self, observation_space):
        shapes = []
        highs = []
        lows = []
        for key, box in observation_space.items():
            if len(box.shape) == 0:
                shapes.append(1)
                # highs.append()
                # lows.append()
                highs += [*np.expand_dims(box.high, axis=-1)]
                lows += [*np.expand_dims(box.low, axis=-1)]
            elif len(box.shape) == 1:
                shapes += [*box.shape]
                highs += [*box.high]
                lows += [*box.low]
            elif len(box.shape) > 1:
                # print(box)
                shapes += [np.prod(box.shape)]
                highs += [*(box.high.reshape(-1))]
                lows += [*(box.low.reshape(-1))]
        # print(shapes)
        # print(highs)
        # print(lows)
        size = (np.sum(shapes, dtype=np.int32))
        highs = np.array(highs)
        lows = np.array(lows)        
        return size, highs, lows


class GymNormalizedFlattenRecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, cfg, deque_size=100):
        super().__init__(env)
        if isinstance(env.observation_space, GymDict) or \
            isinstance(env.observation_space, Dict) or \
            isinstance(env.observation_space, OrderedDict):
            size, highs, lows = self.get_obs_space(env.observation_space)
            self.observation_space = GymBox(low=lows, high=highs)
            # print("observation is changed")
        self.env_type = cfg.experiment.env_type
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None
        self.obs_rms = RunningMeanStd(shape=(self.num_envs, *self.observation_space.shape))
        self.return_rms = RunningMeanStd(shape=(self.num_envs, ))
        self.gamma = 0.98
        self.epsilon = 1e-8
        
    def normalize_obs(self, obs):
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
    
    def normalize_rew(self, rews):
        self.return_rms.update(self.returned_episode_returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)
        
    def reset(self, **kwargs):
        print(f"kwargs: {kwargs}")
        observations= super().reset(**kwargs)
        if isinstance(observations, Dict) or \
            isinstance(observations, GymDict) or \
                isinstance(observations, OrderedDict):
            observations =  self.flatten_dict(observations)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        observations = self.normalize_obs(observations)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        if isinstance(observations, Dict) or \
            isinstance(observations, GymDict) or \
                isinstance(observations, OrderedDict):
            observations = self.flatten_dict(observations)
        self.episode_returns += rewards #infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - dones # infos["terminated"]
        self.episode_lengths *= 1 - dones # infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        observations = self.normalize_obs(observations)
        rewards = self.normalize_rew(rewards)
        return (
                observations,
                rewards,
                dones,
                infos,
            )
        
    def flatten_dict(self, obs):
        obs_pieces = []
        for v in obs.values():
            flat = np.array([v]) if np.isscalar(v) else v.ravel()
            obs_pieces.append(flat)
        obs = np.concatenate(obs_pieces, axis=-1)
        obs = obs.reshape(self.num_envs, -1)
        return obs

    def get_obs_space(self, observation_space):
        shapes = []
        highs = []
        lows = []
        for key, box in observation_space.items():
            if len(box.shape) == 0:
                shapes.append(1)
                # highs.append()
                # lows.append()
                highs += [*np.expand_dims(box.high, axis=-1)]
                lows += [*np.expand_dims(box.low, axis=-1)]
            elif len(box.shape) == 1:
                shapes += [*box.shape]
                highs += [*box.high]
                lows += [*box.low]
            elif len(box.shape) > 1:
                # print(box)
                shapes += [np.prod(box.shape)]
                highs += [*(box.high.reshape(-1))]
                lows += [*(box.low.reshape(-1))]
        # print(shapes)
        # print(highs)
        # print(lows)
        size = (np.sum(shapes, dtype=np.int32))
        highs = np.array(highs)
        lows = np.array(lows)        
        return size, highs, lows

            
def make_envpool_env(env_index, env_id, cfg):
    env_type = cfg.experiment.env_type
    if env_id not in image_envs:
        envs = envpool.make(
                        env_id,
                        env_type=env_type,
                        num_envs=cfg.experiment.num_envs,
                        seed=cfg.experiment.seed,
                    )
        envs.is_image_obs = False
        
    else:
        envs = envpool.make(
                        env_id,
                        env_type=env_type,
                        img_height=84,
                        img_width=84,
                        num_envs=cfg.experiment.num_envs,
                        episodic_life=True,
                        reward_clip=True,
                        seed=cfg.experiment.seed,
                    )
        envs.is_image_obs = True
    envs.num_envs = cfg.experiment.num_envs
    if env_type == "gymnasium":
        envs = GymnasiumNormalizedFlattenRecordEpisodeStatistics(envs, cfg)
        wrappers = gymnasium.wrappers
        # wrappers = gym.wrappers
        
    elif env_type == "gym":
        envs = GymNormalizedFlattenRecordEpisodeStatistics(envs, cfg)
        wrappers = gym.wrappers
    if env_id in mujoco_envs or env_id in dmc_envs:
        envs = wrappers.ClipAction(envs)
        envs = wrappers.TransformObservation(envs, lambda obs: np.clip(obs, -10, 10))
        envs = wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
        # envs = FlattenRecordEpisodeStatistics(envs)
    # elif env_id in gym_envs or env_id in image_envs:
        # envs = FlattenRecordEpisodeStatistics(envs)
    # envs = FlattenRecordEpisodeStatistics(envs)
    envs.single_observation_space = envs.observation_space 
    envs.single_action_space = envs.action_space
    envs.env_id = env_id
    return envs

def make_gym_env(env_index, env_id, cfg):
    capture_video = cfg.experiment.capture_video
    seed = cfg.experiment.seed + env_index
    def thunk():
        # TODO: Atari
        if "asterix" in env_id or \
            "breakout" in env_id or \
            "freeway" in env_id or \
            "seaquest" in env_id or \
            "space_invaders" in env_id:
            env = gym.make(env_id)
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)
        else:
            env = gym.make(env_id)
        
        if capture_video and env_index == 0:
            video_path = pathlib.Path(cfg.paths.video)
            train_path = pathlib.Path("train")
            video_save_path = str(video_path / train_path / env_id)
            print(f"{env_id}: is being recorded")
            env = GymRecordVideo(env, video_save_path, disable_logger=True)
        
        # env = gymnasium.wrappers.TimeLimit(env, cfg.experiment.max_episode_steps)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if isinstance(env.action_space, GymBox):
            env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=cfg.experiment.gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.env_id = env_id
        # env.seed(seed)
        # env.action_space.seed(seed)
        # env.observation_space.seed(seed)
        return env
    return thunk


def make_gymnasium_env(env_index, env_id, cfg):
    capture_video = cfg.experiment.capture_video
    seed = cfg.experiment.seed + env_index
    def thunk():

        # TODO: Atari
        if "Breakout" in env_id or \
            "Breakout" in env_id or \
            "Breakout" in env_id or \
            "Breakout" in env_id or \
            "Breakout" in env_id or \
            "Breakout" in env_id or \
            "Breakout" in env_id or \
            "Breakout" in env_id:
            env = gymnasium.make(env_id)
            env = gymnasium.wrappers.RecordEpisodeStatistics(env)
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = ClipRewardEnv(env)
            env = gymnasium.wrappers.ResizeObservation(env, (84, 84))
            env = gymnasium.wrappers.GrayScaleObservation(env)
            env = gymnasium.wrappers.FrameStack(env, 4)
            return env
        else:
            env = gymnasium.make(env_id)
        
        if capture_video and env_index == 0:
            video_path = pathlib.Path(cfg.paths.video)
            train_path = pathlib.Path("train")
            video_save_path = str(video_path / train_path / env_id)
            print(f"{env_id}: is being recorded")
            env = GymnasiumRecordVideo(env, video_save_path, disable_logger=True)
        
        # env = gymnasium.wrappers.TimeLimit(env, cfg.experiment.max_episode_steps)
        # env = gymnasium.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)
        
        if isinstance(env.action_space, GymnasiumBox):
            env = gymnasium.wrappers.ClipAction(env)
        env = gymnasium.wrappers.NormalizeObservation(env)
        env = gymnasium.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gymnasium.wrappers.NormalizeReward(env, gamma=cfg.experiment.gamma)
        env = gymnasium.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

        # env.seed(seed)
        # env.action_space.seed(seed)
        # env.observation_space.seed(seed)
        return env
    return thunk


def get_activation(activation_name: str):
    if activation_name == 'tanh':
        activation = nn.Tanh
    elif activation_name == 'relu':
        activation = nn.ReLU
    elif activation_name == 'leakyrelu':
        activation = nn.LeakyReLU
    elif activation_name == "prelu":
        activation = nn.PReLU
    elif activation_name == 'gelu':
        activation = nn.GELU
    elif activation_name == 'sigmoid':
        activation = nn.Sigmoid
    elif activation_name in [ None, 'id', 'identity', 'linear', 'none' ]:
        activation = nn.Identity
    elif activation_name == 'elu':
        activation = nn.ELU
    elif activation_name in ['swish', 'silu']:
        activation = nn.SiLU
    elif activation_name == 'softplus':
        activation = nn.Softplus
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation_name))
    return activation
