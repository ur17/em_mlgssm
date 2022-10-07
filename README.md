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
or
```bash
python -m unittest tests.test_kalmanfilter_and_smoother.Test_filter_and_smoother
python -m unittest tests.test_em_lgssm.Test_EM_lgssm
python -m unittest tests.test_em_mlgssm.Test_EM_mlgssm
```