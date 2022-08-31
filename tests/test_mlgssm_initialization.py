import unittest

import numpy as np

from em_mlgssm.mlgssm.param_initialization import get_initial_value



class Test_param_initialization(unittest.TestCase):

    def setUp(self):

        self.dataset = np.ones(50).reshape(10, 5)
        self.cluster_num = 3
        self.data_num = 10
        self.time_len = 5
        self.state_dim = 2
        self.obs_dim = 1
        
    def initialization(self):
        init_params = get_initial_value(
            dataset=self.dataset, 
            time=self.time_len, 
            cluster_num=self.cluster_num, 
            state_dim=self.state_dim, 
            num=2, tuning_num=5, seed=0
        )


if __name__ == '__main__':
    unittest.main()