# Environment-Agnostic Architecture for Heterogeneous Multi-Environment Reinforcement Learning
Official pytorch code implemenataion of the paper: **Environment-Agnostic Architecture for Heterogeneous Multi-Environment Reinforcement Learning**.
 
## Installation
1. Install pytorch 2.1 from https://pytorch.org/
2. Create conda environment and install the dependencies with pip.
```
conda create -n AA4HMRL python=3.10
conda activate AA4HMRL
pip install -r requirements.txt
```
3. Install a kernel of structured state space model
```
python ./extensions/kernels/setup.py install 
```

## Usage

### Env agnostic-architecture experiment
```
python main_ddppo.py experiment.seed=0 nn.env_specific_enc_dec=False nn.actor_critic.encoder_net_1d=s4 nn.actor_critic.decoder_net=s4 
python main_ddppo.py experiment.seed=0 nn.env_specific_enc_dec=False nn.actor_critic.encoder_net_1d=rnn nn.actor_critic.decoder_net=rnn 
```

### Env specific-architecture experiment
```
python main_ddppo.py experiment.seed=0 nn.env_specific_enc_dec=True 
```


