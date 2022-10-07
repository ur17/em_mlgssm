import unittest

import numpy as np
import numpy.testing as npt

from pykalman import KalmanFilter
from pykalman.standard import _em_transition_matrix
from pykalman.standard import _em_transition_covariance
from pykalman.standard import _em_observation_matrix
from pykalman.standard import _em_observation_covariance
from pykalman.standard import _em_transition_offset
from pykalman.standard import _em_observation_offset
from pykalman.standard import _smooth_pair

from em_mlgssm import EM_lgssm



class Test_EM_lgssm(unittest.TestCase):

    def setUp(self):
        
        self.state_dim = 2
        self.obs_dim = 1

        self.state_mat = np.asarray([[1, 0],[0, 1]])
        self.state_cov = np.asarray([[0.5, 0],[0, 0.5]])
        self.obs_mat = np.asarray([2, 2])
        self.obs_cov = np.asarray([0.5])
        self.init_state_mean = np.asarray([1, 1])
        self.init_state_cov = np.asarray([[0.5, 0],[0, 0.5]])

        self.time_series = np.asarray([1, 1, 1, 1, 1])
        

    def test_param_update(self):
        model = EM_lgssm(
            time_series = self.time_series, 
            state_dim = self.state_dim, obs_dim = self.obs_dim
        )

        model.param_init(
            state_mat = self.state_mat, state_cov = self.state_cov, 
            obs_mat = self.obs_mat, obs_cov = self.obs_cov,
            init_state_mean = self.init_state_mean, 
            init_state_cov = self.init_state_cov
        )

        (e_zn, e_znzn, e_znzn_1) = model.e_step()

        state_mat = model._update_state_mat(e_zn, e_znzn, e_znzn_1)
        state_cov = model._update_state_cov(e_zn, e_znzn, e_znzn_1)
        obs_mat = model._update_obs_mat(e_zn, e_znzn)
        obs_cov = model._update_obs_cov(e_zn, e_znzn)

        pairwise_covariances = _smooth_pair(
            smoothed_state_covariances = model.smooth_state_covs, 
            kalman_smoothing_gain = model.smooth_gains
        )

        test_state_mat = _em_transition_matrix(
            transition_offsets = np.asarray([0., 0.]), 
            smoothed_state_means = e_zn.reshape(len(self.time_series), self.state_dim), 
            smoothed_state_covariances = model.smooth_state_covs, 
            pairwise_covariances = pairwise_covariances
        )
        test_state_cov = _em_transition_covariance(
            transition_matrices = self.state_mat, 
            transition_offsets = np.asarray([0., 0.]),
            smoothed_state_means = e_zn.reshape(len(self.time_series), self.state_dim), 
            smoothed_state_covariances = model.smooth_state_covs, 
            pairwise_covariances = pairwise_covariances
        )
        test_obs_mat = _em_observation_matrix(
            observations = self.time_series.reshape(len(self.time_series), self.obs_dim), 
            observation_offsets = np.asarray([0.]),
            smoothed_state_means = e_zn.reshape(len(self.time_series), self.state_dim), 
            smoothed_state_covariances = model.smooth_state_covs
        )
        test_obs_cov = _em_observation_covariance(
            observations = self.time_series.reshape(len(self.time_series), self.obs_dim), 
            observation_offsets = np.asarray([0.]),
            transition_matrices = self.obs_mat.reshape(self.obs_dim, self.state_dim), 
            smoothed_state_means = e_zn.reshape(len(self.time_series), self.state_dim), 
            smoothed_state_covariances = model.smooth_state_covs
        )
        npt.assert_array_almost_equal(state_mat, test_state_mat, decimal=8)
        npt.assert_array_almost_equal(state_cov, test_state_cov, decimal=8)
        npt.assert_array_almost_equal(obs_mat, test_obs_mat, decimal=8)
        npt.assert_array_almost_equal(obs_cov, test_obs_cov, decimal=8)


    def test_em_fit(self):
        model = EM_lgssm(
            time_series = self.time_series, 
            state_dim = self.state_dim, obs_dim = self.obs_dim
        )

        model.param_init(
            state_mat = self.state_mat, state_cov = self.state_cov, 
            obs_mat = self.obs_mat, obs_cov = self.obs_cov,
            init_state_mean = self.init_state_mean, 
            init_state_cov = self.init_state_cov
        )

        (params, param_diff) = model.fit(
            max_iter = 5, param_epsilon = -1, param_fix = []
        )

        kf = KalmanFilter(
            transition_matrices = self.state_mat, observation_matrices = self.obs_mat,
            transition_covariance = self.state_cov, observation_covariance = self.obs_cov,
            transition_offsets = np.asarray([0., 0.]), observation_offsets = np.asarray([0.]),
            initial_state_mean = self.init_state_mean, initial_state_covariance = self.init_state_cov,
            em_vars = ['transition_matrices', 'observation_matrices', 'transition_covariance', 
            'observation_covariance', 'initial_state_mean','initial_state_covariance'], 
            n_dim_state = self.state_dim, n_dim_obs = self.obs_dim
        )

        kf = kf.em(X = self.time_series.reshape(len(self.time_series), self.obs_dim), n_iter = 5)

        npt.assert_array_almost_equal(model.init_state_mean, kf.initial_state_mean, decimal=8)
        npt.assert_array_almost_equal(model.init_state_cov, kf.initial_state_covariance, decimal=8)

        npt.assert_array_almost_equal(model.state_mat, kf.transition_matrices, decimal=8)
        npt.assert_array_almost_equal(model.state_cov, kf.transition_covariance, decimal=8)

        npt.assert_array_almost_equal(model.obs_mat, kf.observation_matrices, decimal=8)
        npt.assert_array_almost_equal(model.obs_cov, kf.observation_covariance, decimal=8)



if __name__ == '__main__':
    unittest.main()