import unittest

import numpy as np
import numpy.testing as npt

from em_mlgssm.lgssm.em import EM_lgssm



class Test_lgssm_EM(unittest.TestCase):

    def setUp(self):
        
        self.state_dim = 2
        self.obs_dim = 1
        self.max_iter = 5
        self.tuning = 3

        self.time_series = np.asarray([1, 1, 1, 1, 1])

        self.state_mat = np.asarray([
            [1, 0],
            [0, 1],
        ])

        self.obs_mat = np.asarray([1, 1])
        self.obs_cov = np.asarray([0.1])

        self.any_state_mean = np.asarray([
            [1, 1],
        ])
        self.any_state_cov = np.asarray([
            [0.1, 0],
            [0, 0.1],
        ])


    def test_training_default(self):

        model = EM_lgssm(
            time = len(self.time_series), state_dim = self.state_dim, 
            obs_dim = self.obs_dim, obs_mat_fix = "default"
        )

        model.param_init(
            state_mat = self.state_mat, state_cov = self.any_state_cov, 
            obs_cov = self.obs_cov, init_state_mean = self.any_state_mean, 
            init_state_cov =self.any_state_cov, obs_mat = None
        )

        param = model.param_update( 
            time_series = self.time_series, max_iter = self.max_iter, 
            epsilon = -1, log = True
        )

        self.assertEqual(len(model.param_diff), self.max_iter)


    def test_tuning(self):

        model = EM_lgssm(
            time = len(self.time_series), state_dim = self.state_dim, 
            obs_dim = self.obs_dim, obs_mat_fix = "default"
        )

        (_, likelihood_list, params_list) = model.param_tuning( 
            self.time_series, num = 2, max_iter = self.max_iter, seed = 0, 
            tuning_num = self.tuning, epsilon = -1, log = True
        )

        self.assertEqual(len(likelihood_list), self.tuning)
        self.assertEqual(len(params_list), self.tuning)




if __name__ == '__main__':
    unittest.main()