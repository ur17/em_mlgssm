# Time Series Clustering with an EM algorithm for Mixtures of Linear Gaussian State Space Models (accepted by Pattern Recognit in 2023)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ur17/em_mlgssm/blob/main/LICENSE)

This is the origin Python implementation of the following paper: [Time Series Clustering with an EM algorithm for Mixtures of Linear Gaussian State Space Models](https://doi.org/10.1016/j.patcog.2023.109375).


## Installation 
```bash
git clone git@github.com:ur17/em_mlgssm.git
pip install -e em_mlgssm
```

## Testing
If you want to run test cases, install following Python library "pykalman".
```bash
pip install pykalman
```
Run all test cases
```bash
python -m unittest discover -v
```
Run specific test case
```bash
python -m unittest tests.test_kalmanfilter_and_smoother.Test_filter_and_smoother
python -m unittest tests.test_em_algorithm_for_lgssm.Test_EM_lgssm
python -m unittest tests.test_em_algorithm_for_mlgssm.Test_EM_mlgssm
```