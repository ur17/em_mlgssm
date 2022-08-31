# MLGSSM : Time series clustering with mixtures of Linear Gaussian State Space Models

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ur17/em_mlgssm/blob/main/LICENSE)

In the preprint ([arXiv](https://arxiv.org/abs/2208.11907)), I proposed a novel model-based time series clustering method with mixtures of linear Gaussian state space models (MLGSSMs), as a corresponding author. This repository contains [Python code of learning algorithm for MLGSSMs](https://github.com/ur17/em_mlgssm/tree/main/em_mlgssm/mlgssm).


## Installation 
```bash
git clone git@github.com:ur17/em_mlgssm.git
pip install -e em_mlgssm
```

## Testing 
```bash
cd em_mlgssm
python -m unittest discover -v
```

## Usage
```python
from em_mlgssm.mlgssm import EM_mlgssm


model = EM_mlgssm(
    cluster_num = K, data_len = N, time_len = T, 
    state_dim = d_x, obs_dim = d_y
)

# set initial parameters
model.param_init(
    random_param_init = "default"
)

# training
max_iter = 100
summary = model.training(
    dataset, 
    max_iter = max_iter
)

# summary
best_params = summary[0]
loglikelihoods_of_each_step= summary[1]
clustering_results_of_each_step = summary[2]

# clustering
pred, prob = model.clustering()
```