# Deep reinforcement learning with PyTorch and OpenAI Gym

This repository is a collection of deep learning algorithms implemented with [Pytorch](https://pytorch.org/) and using environments provided by [OpenAI Gym](https://gym.openai.com/).

## Current implementations
1. DQN with ε-greedy exploration, inpired by the original [DQN paper](https://arxiv.org/abs/1312.5602) and [PyTorch tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#sphx-glr-intermediate-reinforcement-q-learning-py)

## Setup
To start working with agents create a conda environment using the `environment.yml` file as follows

```
conda env create -f environment.yml
```

This will create a conda environment named `deep-rl` which can then be activated

```
conda activate deep-rl
```