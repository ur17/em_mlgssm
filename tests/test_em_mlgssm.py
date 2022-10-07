import unittest

import numpy as np
import numpy.testing as npt

from em_mlgssm import EM_mlgssm



class Test_EM_mlgssm(unittest.TestCase):

    def setUp(self):

        self.dataset = np.ones(50).reshape(10, 5)
        self.cluster_num = 3
        self.state_dim = 2
        self.obs_dim = 1

        self.state_mats = np.asarray([np.eye(self.state_dim)] * self.cluster_num)
        self.state_covs = np.asarray([np.eye(self.state_dim)] * self.cluster_num)
        self.obs_mats = np.ones((self.cluster_num, self.obs_dim, self.state_dim))
        self.obs_covs = np.asarray([np.eye(self.obs_dim)] * self.cluster_num)
        self.init_state_means = np.asarray([np.zeros(self.state_dim)] * self.cluster_num) 
        self.init_state_covs = np.asarray([np.eye(self.state_dim) * 1e+4] * self.cluster_num)
        self.weights = np.ones(self.cluster_num) / self.cluster_num
        
        self.model = EM_mlgssm(
            time_series_set = self.dataset, cluster_num = self.cluster_num, 
            state_dim = self.state_dim, obs_dim = self.obs_dim
        )
        self.model.param_init(
            state_mats = self.state_mats, state_covs = self.state_covs, 
            obs_mats = self.obs_mats, obs_covs = self.obs_covs, 
            init_state_means = self.init_state_means, 
            init_state_covs = self.init_state_covs, 
            weights = self.weights,
        )


    def test_e_step_shape(self):
        (posterior_prob, e_zn, e_znzn, e_znzn_1) = self.model.e_step()

        self.assertEqual(
            posterior_prob.shape, (len(self.dataset), self.cluster_num, 1, 1)
        )
        self.assertEqual(
            e_zn.shape, 
            (self.cluster_num, len(self.dataset), len(self.dataset[0]), self.state_dim, 1)
        )
        self.assertEqual(
            e_znzn.shape, 
            (self.cluster_num, len(self.dataset), len(self.dataset[0]), self.state_dim, self.state_dim)
        )
        self.assertEqual(
            e_znzn_1.shape, 
            (self.cluster_num, len(self.dataset), len(self.dataset[0]) - 1, self.state_dim, self.state_dim)
        )


    def test_m_step_shape(self, k:int = 0):
        (posterior_prob, e_zn, e_znzn, e_znzn_1) = self.model.e_step()

        init_state_mean = self.model._update_init_state_means(
            posterior_prob[:,k], e_zn[k]
        )
        init_state_cov = self.model._update_init_state_covs(
            posterior_prob[:,k], e_zn[k], e_znzn[k], init_state_mean
        )
        state_mat = self.model._update_state_mat(
            posterior_prob[:,k], e_zn[k], e_znzn[k], e_znzn_1[k]
        )
        state_cov = self.model._update_state_cov(
            posterior_prob[:,k], e_zn[k], e_znzn[k], e_znzn_1[k], state_mat
        )
        obs_mat = self.model._update_obs_mat(
            posterior_prob[:,k], e_zn[k], e_znzn[k]
        )
        obs_cov = self.model._update_obs_cov(
            posterior_prob[:,k], e_zn[k], e_znzn[k], obs_mat
        )

        self.assertEqual(
            init_state_mean.shape, (self.state_dim, 1)
        )
        self.assertEqual(
            init_state_cov.shape, 
            (self.state_dim, self.state_dim)
        )
        self.assertEqual(
            state_mat.shape, 
            (self.state_dim, self.state_dim)
        )
        self.assertEqual(
            state_cov.shape, 
            (self.state_dim, self.state_dim)
        )
        self.assertEqual(
            obs_mat.shape, 
            (self.obs_dim, self.state_dim)
        )
        self.assertEqual(
            obs_cov.shape, 
            (self.obs_dim, self.obs_dim)
        )


    def test_m_step(self):
        (posterior_prob, e_zn, e_znzn, e_znzn_1) = self.model.e_step()
        params = self.model.m_step(posterior_prob, e_zn, e_znzn, e_znzn_1)

    
    def test_fit(self):
        summary = self.model.fit(max_iter = 2, param_epsilon = -1, log = False)



if __name__ == '__main__':
    unittest.main()