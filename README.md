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
from em_mlgssm.mlgssm import get_initial_value


# get init parameters
init_params = get_initial_value(
    dataset = dataset, time = time, cluster_num = 3, state_dim = 2
)


model = EM_mlgssm(
    cluster_num = K, data_num = N, time_len = T, 
    state_dim = d_x, obs_dim = d_y
)

# initialization
model.param_init(
    state_mat = init_params[0], state_cov = init_params[1], 
    obs_mat = init_params[2], obs_cov = init_params[3],
    init_state_mean = init_params[4], init_state_cov = init_params[5], 
    weights = init_params[6],
    random_param_init = False
)
model.kalman_init()

# training
max_iter = 100
summary = model.training(
    dataset = dataset, 
    max_iter = max_iter
)

# summary
best_params = summary[0]
loglikelihoods_of_each_step= summary[1]
clustering_results_of_each_step = summary[2]

# clustering
pred, prob = model.clustering()
```