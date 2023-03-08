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

from em_mlgssm import EMlgssm



class Test_EM_lgssm(unittest.TestCase):


    def setUp(self):

        self.y = np.asarray([.1, .2, .3, .4, .5])
        self.u_x = np.asarray([1., 1., 1., 1., 1.])
        self.u_y = np.asarray([1., 1., 1., 1., 1.])
        self.d_x = 2
        self.d_y = 1
        self.d_ux = 1
        self.d_uy = 1
        self.A = np.asarray([[1., 0.],[0., 1.]])
        self.Gamma = np.asarray([[.5, 0.],[0., .5]])
        self.C = np.asarray([2., 2.])
        self.Sigma = np.asarray([.5])
        self.mu = np.asarray([1., 1.])
        self.P = np.asarray([[.5, 0.],[0., .5]])
        self.B = np.asarray([[1., 1.]])
        self.D = np.asarray([1.])

        self.model = EMlgssm(
            state_mat=self.A, 
            state_cov=self.Gamma, 
            obs_mat=self.C.reshape(self.d_y, self.d_x), 
            obs_cov=self.Sigma.reshape(self.d_y, self.d_y),
            init_state_mean=self.mu.reshape(self.d_x, 1), 
            init_state_cov=self.P
        )

        self.model_add_input = EMlgssm(
            state_mat=self.A, 
            state_cov=self.Gamma, 
            obs_mat=self.C.reshape(self.d_y, self.d_x), 
            obs_cov=self.Sigma.reshape(self.d_y, self.d_y),
            init_state_mean=self.mu.reshape(self.d_x, 1), 
            init_state_cov=self.P,
            input_state_mat=self.B.reshape(self.d_x, self.d_ux), 
            input_obs_mat=self.D.reshape(self.d_y, self.d_uy)
        )

        

    def test_m_step(self):
        
        self.model.y = self.y.reshape(len(self.y), self.d_y, 1)
        self.model.u_x = np.empty(len(self.y))
        self.model.u_y = np.empty(len(self.y))
        self.model.fix = []
        self.model.run_e_step().run_m_step()

        pairwise_covariances = _smooth_pair(
            smoothed_state_covariances=self.model.covs_s, 
            kalman_smoothing_gain=self.model.gains_s
        )

        test_A = _em_transition_matrix(
            transition_offsets=np.asarray([0., 0.]), 
            smoothed_state_means=self.model.e_xt.reshape(len(self.y), self.d_x), 
            smoothed_state_covariances=self.model.covs_s, 
            pairwise_covariances=pairwise_covariances
        )
        test_Gamma = _em_transition_covariance(
            transition_matrices=test_A, 
            transition_offsets=np.asarray([0., 0.]),
            smoothed_state_means=self.model.e_xt.reshape(len(self.y), self.d_x), 
            smoothed_state_covariances=self.model.covs_s, 
            pairwise_covariances=pairwise_covariances
        )
        test_C = _em_observation_matrix(
            observations=self.y.reshape(len(self.y), self.d_y), 
            observation_offsets=np.asarray([0.]),
            smoothed_state_means=self.model.e_xt.reshape(len(self.y), self.d_x), 
            smoothed_state_covariances=self.model.covs_s
        )
        test_Sigma = _em_observation_covariance(
            observations=self.y.reshape(len(self.y), self.d_y), 
            observation_offsets=np.asarray([0.]),
            transition_matrices=test_C.reshape(self.d_y, self.d_x), 
            smoothed_state_means=self.model.e_xt.reshape(len(self.y), self.d_x), 
            smoothed_state_covariances=self.model.covs_s
        )
        npt.assert_array_almost_equal(self.model.A, test_A, decimal=8)
        npt.assert_array_almost_equal(self.model.Gamma, test_Gamma, decimal=8)
        npt.assert_array_almost_equal(self.model.C, test_C, decimal=8)
        npt.assert_array_almost_equal(self.model.Sigma, test_Sigma, decimal=8)



    def test_input_m_step(self):

        self.model_add_input.y = self.y.reshape(len(self.y), self.d_y, 1)
        self.model_add_input.u_x = self.u_x.reshape(len(self.y), self.d_ux, 1)
        self.model_add_input.u_y = self.u_y.reshape(len(self.y), self.d_uy, 1)
        self.model_add_input.fix = []
        self.model_add_input.run_e_step().run_m_step()

        pairwise_covariances = _smooth_pair(
            smoothed_state_covariances=self.model_add_input.covs_s, 
            kalman_smoothing_gain=self.model_add_input.gains_s
        )

        test_A = _em_transition_matrix(
            transition_offsets=np.asarray([1., 1.]), 
            smoothed_state_means=self.model_add_input.e_xt.reshape(len(self.y), self.d_x), 
            smoothed_state_covariances=self.model_add_input.covs_s, 
            pairwise_covariances=pairwise_covariances
        )
        test_Gamma = _em_transition_covariance(
            transition_matrices=test_A, 
            transition_offsets=np.asarray([1., 1.]),
            smoothed_state_means=self.model_add_input.e_xt.reshape(len(self.y), self.d_x), 
            smoothed_state_covariances=self.model_add_input.covs_s, 
            pairwise_covariances=pairwise_covariances
        )
        test_C = _em_observation_matrix(
            observations=self.y.reshape(len(self.y), self.d_y), 
            observation_offsets=np.asarray([1.]),
            smoothed_state_means=self.model_add_input.e_xt.reshape(len(self.y), self.d_x), 
            smoothed_state_covariances=self.model_add_input.covs_s
        )
        test_Sigma = _em_observation_covariance(
            observations=self.y.reshape(len(self.y), self.d_y), 
            observation_offsets=np.asarray([1.]),
            transition_matrices=test_C.reshape(self.d_y, self.d_x), 
            smoothed_state_means=self.model_add_input.e_xt.reshape(len(self.y), self.d_x), 
            smoothed_state_covariances=self.model_add_input.covs_s
        )
        test_B = _em_transition_offset(
            transition_matrices=test_A.reshape(self.d_x, self.d_x), 
            smoothed_state_means=self.model_add_input.e_xt.reshape(len(self.y), self.d_x)
        ).reshape(self.d_x, self.d_y)
        test_D = _em_observation_offset(
            observation_matrices=test_C.reshape(self.d_y, self.d_x), 
            smoothed_state_means=self.model_add_input.e_xt.reshape(len(self.y), self.d_x),
            observations=self.y.reshape(len(self.y), self.d_y)
        ).reshape(self.d_y, self.d_y)

        npt.assert_array_almost_equal(self.model_add_input.A, test_A, decimal=8)
        npt.assert_array_almost_equal(self.model_add_input.Gamma, test_Gamma, decimal=8)
        npt.assert_array_almost_equal(self.model_add_input.C, test_C, decimal=8)
        npt.assert_array_almost_equal(self.model_add_input.Sigma, test_Sigma, decimal=8)
        npt.assert_array_almost_equal(self.model_add_input.B, test_B, decimal=8)
        npt.assert_array_almost_equal(self.model_add_input.D, test_D, decimal=8)



    def test_em_fit(self):

        params = self.model.fit(
            y=self.y.reshape(len(self.y), self.d_y, 1), max_iter=5, epsilon=-1
        )

        kf = KalmanFilter(
            transition_matrices=self.A, observation_matrices=self.C,
            transition_covariance=self.Gamma, observation_covariance=self.Sigma,
            transition_offsets=np.asarray([0., 0.]), observation_offsets=np.asarray([0.]),
            initial_state_mean=self.mu, initial_state_covariance=self.P,
            em_vars=['transition_matrices', 'observation_matrices', 'transition_covariance', 
            'observation_covariance', 'initial_state_mean','initial_state_covariance'], 
            n_dim_state=self.d_x, n_dim_obs=self.d_y
        )

        kf = kf.em(X=self.y.reshape(len(self.y), self.d_y), n_iter=5)
        kf.initial_state_mean = kf.initial_state_mean.reshape(self.d_x, 1)
        kf.observation_matrices = kf.observation_matrices.reshape(self.d_y, self.d_x)
        kf.observation_covariance = kf.observation_covariance.reshape(self.d_y, self.d_y)

        npt.assert_array_almost_equal(params['mu'], kf.initial_state_mean, decimal=8)
        npt.assert_array_almost_equal(params['P'], kf.initial_state_covariance, decimal=8)
        npt.assert_array_almost_equal(params['A'], kf.transition_matrices, decimal=8)
        npt.assert_array_almost_equal(params['Gamma'], kf.transition_covariance, decimal=8)
        npt.assert_array_almost_equal(params['C'], kf.observation_matrices, decimal=8)
        npt.assert_array_almost_equal(params['Sigma'], kf.observation_covariance, decimal=8)



if __name__ == '__main__':
    unittest.main()