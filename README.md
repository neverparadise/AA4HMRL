# Environment-Agnostic Architecture for Heterogeneous Multi-Environment Reinforcement Learning
Official pytorch code implemenataion of the paper: **Environment-Agnostic Architecture for Heterogeneous Multi-Environment Reinforcement Learning**.

The experiments of the paper were tested on Ubuntu 22.04, 96 CPUs, 8 A6000 GPUs.
 
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

4. Download D4RL datasets [Link] and unzip 

## Usage


## Citation

