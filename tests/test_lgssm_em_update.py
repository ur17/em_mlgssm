import unittest

import numpy as np
import numpy.testing as npt

from pykalman.standard import _em_transition_matrix
from pykalman.standard import _em_transition_covariance
from pykalman.standard import _em_observation_matrix
from pykalman.standard import _em_observation_covariance
from pykalman.standard import _em_transition_offset
from pykalman.standard import _em_observation_offset

from em_mlgssm.lgssm.e_step import _compute_statistics_for_estep
from em_mlgssm.lgssm.m_step import _update_state_mat
from em_mlgssm.lgssm.m_step import _update_state_cov
from em_mlgssm.lgssm.m_step import _update_obs_mat
from em_mlgssm.lgssm.m_step import _update_obs_cov
from em_mlgssm.lgssm.m_step import _update_input_state_mat
from em_mlgssm.lgssm.m_step import _update_input_obs_mat



class Test_lgssm_EM_steps(unittest.TestCase):

    def setUp(self):
        
        self.state_dim = 2
        self.obs_dim = 1

        self.time_series = np.asarray([1, 1, 1, 1, 1])

        self.state_mat = np.asarray([
            [1, 0],
            [0, 1],
        ])
        self.obs_mat = np.asarray([1, 1])

        self.e_zn = np.ones(self.state_dim * len(self.time_series)).reshape(len(self.time_series), self.state_dim)
        self.e_znzn = np.asarray([np.identity(self.state_dim)] * len(self.time_series))
        self.e_znzn_1 = np.asarray([np.identity(self.state_dim)] * (len(self.time_series) - 1))

        self.smooth_state_covs = (
            self.e_znzn 
            - np.dot(
                self.e_zn[0].reshape(self.state_dim, self.obs_dim), 
                self.e_zn[0].reshape(self.obs_dim, self.state_dim)
            )
        )

        self.input_state_series = np.asarray([0.5, 0.5, 0.5, 0.5, 0.5])
        self.input_obs_series = np.asarray([0.5, 0.5, 0.5, 0.5, 0.5])
        self.input_state_mat = np.asarray([
            [1, 1]
        ])
        self.input_obs_mat = np.asarray([1])

        self.input_state_series_fix = np.asarray([1, 1, 1, 1, 1])
        self.input_obs_series_fix = np.asarray([1, 1, 1, 1, 1])
        

    def test_compute_statistics_for_estep(self):
        (e_zn, e_znzn, e_znzn_1) = _compute_statistics_for_estep(
            self.e_zn, 
            self.e_znzn, 
            self.e_znzn_1
        )

        self.assertEqual(e_zn.shape, self.e_zn.shape)
        self.assertEqual(e_znzn.shape, self.e_znzn.shape)
        self.assertEqual(e_znzn_1.shape, self.e_znzn_1.shape)


    def test_update_state_mat(self):
        state_mat = _update_state_mat(self.e_znzn, self.e_znzn_1)
        test_state_mat = _em_transition_matrix(
            np.zeros(self.state_dim), self.e_zn, 
            self.smooth_state_covs, self.smooth_state_covs
        )
        npt.assert_array_almost_equal(state_mat, test_state_mat, decimal=8)


        state_mat = _update_state_mat(
            self.e_znzn, self.e_znzn_1, 
            self.e_zn, self.input_state_mat, self.input_state_series
        )
        test_state_mat = _em_transition_matrix(
            np.asarray([0.5, 0.5]), self.e_zn, 
            self.smooth_state_covs, self.smooth_state_covs
        )
        npt.assert_array_almost_equal(state_mat, test_state_mat, decimal=8)


    def test_update_state_cov(self):
        state_cov = _update_state_cov(self.state_mat, self.e_znzn, self.e_znzn_1)
        test_state_cov = _em_transition_covariance(
            self.state_mat, np.zeros(self.state_dim), self.e_zn, self.smooth_state_covs, self.smooth_state_covs
        )
        npt.assert_array_almost_equal(state_cov, test_state_cov, decimal=8)

        state_cov = _update_state_cov(
            self.state_mat, self.e_znzn, self.e_znzn_1,
            self.e_zn, self.input_state_mat, self.input_state_series
        )
        test_state_cov = _em_transition_covariance(
            self.state_mat, np.asarray([0.5, 0.5]), self.e_zn, self.smooth_state_covs, self.smooth_state_covs
        )
        npt.assert_array_almost_equal(state_cov, test_state_cov, decimal=8)


    def test_update_obs_mat(self):
        obs_mat = _update_obs_mat(self.time_series, self.e_zn, self.e_znzn)
        test_obs_mat = _em_observation_matrix(
            self.time_series.reshape(len(self.time_series), self.obs_dim), np.zeros(self.obs_dim), 
            self.e_zn, self.smooth_state_covs
        )
        npt.assert_array_almost_equal(obs_mat, test_obs_mat, decimal=8)

        obs_mat = _update_obs_mat(
            self.time_series, self.e_zn, self.e_znzn,
            self.input_obs_mat, self.input_obs_series
        )
        test_obs_mat = _em_observation_matrix(
            self.time_series.reshape(len(self.time_series), self.obs_dim), np.asarray([0.5]), 
            self.e_zn, self.smooth_state_covs
        )
        npt.assert_array_almost_equal(obs_mat, test_obs_mat, decimal=8)


    def test_update_obs_cov(self):
        obs_cov = _update_obs_cov(self.time_series, self.obs_mat, self.e_zn, self.e_znzn)
        test_obs_cov = _em_observation_covariance(
            self.time_series.reshape(len(self.time_series), self.obs_dim), np.zeros(self.obs_dim), 
            self.obs_mat.reshape(self.obs_dim, self.state_dim), self.e_zn, self.smooth_state_covs
        )
        npt.assert_array_almost_equal(obs_cov, test_obs_cov, decimal=8)

        obs_cov = _update_obs_cov(
            self.time_series, self.obs_mat, self.e_zn, self.e_znzn,
            self.input_obs_mat, self.input_obs_series
        )
        test_obs_cov = _em_observation_covariance(
            self.time_series.reshape(len(self.time_series), self.obs_dim), np.asarray([0.5]), 
            self.obs_mat.reshape(self.obs_dim, self.state_dim), self.e_zn, self.smooth_state_covs
        )
        npt.assert_array_almost_equal(obs_cov, test_obs_cov, decimal=8)


    def test_update_input_state_mat(self):
        input_state_mat = _update_input_state_mat( 
            self.state_mat, self.e_zn, self.input_state_series_fix
        )
        test_input_state_mat = _em_transition_offset(self.state_mat, self.e_zn)
        npt.assert_array_almost_equal(input_state_mat, test_input_state_mat, decimal=8)


    def test_update_input_obs_mat(self):
        input_obs_mat = _update_input_obs_mat( 
            self.time_series, self.input_obs_series_fix, self.obs_mat, self.e_zn
        )
        test_input_obs_mat = _em_observation_offset(
            self.obs_mat.reshape(self.obs_dim, self.state_dim), self.e_zn, 
            self.time_series.reshape(len(self.time_series), self.obs_dim)
        ).reshape(self.obs_dim, self.obs_dim)
        npt.assert_array_almost_equal(input_obs_mat, test_input_obs_mat, decimal=8)



if __name__ == '__main__':
    unittest.main()