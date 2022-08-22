import unittest

import numpy as np
import numpy.testing as npt

from em_mlgssm.mlgssm.em import EM_mlgssm



class Test_mlgssm_em(unittest.TestCase):

    def setUp(self):

        self.dataset = np.ones(50).reshape(10, 5)
        self.cluster_num = 3
        self.data_num = 10
        self.time_len = 5
        self.state_dim = 2
        self.obs_dim = 1
        
        self.model = EM_mlgssm(
            cluster_num = self.cluster_num, data_num = self.data_num, time_len = self.time_len, 
            state_dim = self.state_dim, obs_dim = self.obs_dim, loglikelihood=False, 
            obs_mat_fix = False
        )
        self.model.param_init(random_param_init = "default")
        self.model.kalman_init()


    def test_kalman_filter_smoother_shape(self):
        self.model.kalman_filter(self.dataset)
        self.model.kalman_smoother(self.dataset)

        for k in range(self.cluster_num):

            # filter
            self.assertEqual(
                np.asarray(self.model.filt_mean[f"cluster{k}"]).shape, 
                (self.data_num, self.time_len, self.state_dim)
            )
            self.assertEqual(
                np.asarray(self.model.filt_cov[f"cluster{k}"]).shape, 
                (self.data_num, self.time_len, self.state_dim, self.state_dim)
            )
            self.assertEqual(
                np.asarray(self.model.filt_pred_mean[f"cluster{k}"]).shape, 
                (self.data_num, self.time_len, self.state_dim)
            )
            self.assertEqual(
                np.asarray(self.model.filt_pred_cov[f"cluster{k}"]).shape, 
                (self.data_num, self.time_len, self.state_dim, self.state_dim)
            )
            self.assertEqual(
                np.asarray(self.model.filt_gain[f"cluster{k}"]).shape, 
                (self.data_num, self.time_len, self.state_dim, self.obs_dim)
            )

            # smoother
            self.assertEqual(
                np.asarray(self.model.smooth_mean[f"cluster{k}"]).shape, 
                (self.data_num, self.time_len, self.state_dim)
            )
            self.assertEqual(
                np.asarray(self.model.smooth_cov[f"cluster{k}"]).shape, 
                (self.data_num, self.time_len, self.state_dim, self.state_dim)
            )
            self.assertEqual(
                np.asarray(self.model.smooth_gain[f"cluster{k}"]).shape, 
                (self.data_num, self.time_len - 1, self.state_dim, self.state_dim)
            )

    
    def test_e_step(self):
        self.model.e_step(self.dataset)

    def test_m_step(self):
        self.model.e_step(self.dataset)
        next_params = self.model.m_step(self.dataset)

        npt.assert_array_equal(next_params[0], self.model.state_mat)
        npt.assert_array_equal(next_params[1], self.model.state_cov)
        npt.assert_array_equal(next_params[2], self.model.obs_mat)
        npt.assert_array_equal(next_params[3], self.model.obs_cov)
        npt.assert_array_equal(next_params[4], self.model.init_state_mean)
        npt.assert_array_equal(next_params[5], self.model.init_state_cov)
        npt.assert_array_equal(next_params[6], self.model.weights)

    def test_em_training(self):
        summary = self.model.training(
            self.dataset, max_iter = 10, epsilon_param = -1, epsilon_ll = -1, log = True
        )



if __name__ == '__main__':
    unittest.main()