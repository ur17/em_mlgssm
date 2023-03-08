import unittest

import numpy as np
import numpy.testing as npt

from pykalman.standard import _filter
from pykalman.standard import _smooth
from pykalman.standard import _loglikelihoods

from em_mlgssm import KalmanFS



class Test_filter_and_smoother(unittest.TestCase):


    def setUp(self):

        self.y = np.asarray([.1, .2, .3, .4, .5])
        self.u_x = np.asarray([1., 1., 1., 1., 1.])
        self.u_y = np.asarray([1., 1., 1., 1., 1.])
        self.y_t = np.asarray([10.])
        self.d_x = 2
        self.d_y = 1
        self.d_ux = 1
        self.d_uy = 1
        self.A = np.asarray([[1., 0.],[0., 1.]])
        self.C = np.asarray([1., 1.])
        self.y_cov = np.asarray([.1])
        self.x_mean = np.asarray([[1., 1.]])
        self.x_cov = np.asarray([[.1, 0.],[0., .1]])
        self.B = np.asarray([[1., 1.]])
        self.D = np.asarray([1.])
        self.input = np.asarray([2.])

        self.kfs = KalmanFS(
            state_mat=self.A, 
            state_cov=self.x_cov, 
            obs_mat=self.C.reshape(self.d_y, self.d_x), 
            obs_cov=self.y_cov.reshape(self.d_y, self.d_y),
            init_state_mean=self.x_mean.reshape(self.d_x, 1), 
            init_state_cov=self.x_cov
        )

        self.kfs_add_input = KalmanFS(
            state_mat=self.A, 
            state_cov=self.x_cov, 
            obs_mat=self.C.reshape(self.d_y, self.d_x), 
            obs_cov=self.y_cov.reshape(self.d_y, self.d_y),
            init_state_mean=self.x_mean.reshape(self.d_x, 1), 
            init_state_cov=self.x_cov,
            input_state_mat=self.B.reshape(self.d_x, self.d_ux), 
            input_obs_mat=self.D.reshape(self.d_y, self.d_uy)
        )



    def test_filtering_and_smoothing(self):

        (gains, x_means_f, x_covs_f, 
        x_means_p, x_covs_p) = self.kfs.run_filter(
            y=self.y.reshape(len(self.y), self.d_y, 1)
        )

        (test_x_means_p, test_x_covs_p, test_gain, 
        test_x_means_f, test_x_covs_f) = _filter(
            transition_matrices=self.A, 
            observation_matrices=self.C.reshape(self.d_y, self.d_x),
            transition_offsets=np.zeros(self.d_x), 
            observation_offsets=np.zeros(self.d_y),
            transition_covariance=self.x_cov, 
            observation_covariance=self.y_cov.reshape(self.d_y, self.d_y),
            initial_state_mean=self.x_mean.reshape(self.d_x,), 
            initial_state_covariance=self.x_cov,
            observations=self.y.reshape(len(self.y), 1)
        )

        test_x_means_f = test_x_means_f.reshape(len(self.y), self.d_x, 1)
        test_x_means_p = test_x_means_p.reshape(len(self.y), self.d_x, 1)

        npt.assert_array_almost_equal(gains, test_gain, decimal=8)
        npt.assert_array_almost_equal(x_means_f, test_x_means_f, decimal=8)
        npt.assert_array_almost_equal(x_covs_f, test_x_covs_f, decimal=8)
        npt.assert_array_almost_equal(x_means_p, test_x_means_p, decimal=8)
        npt.assert_array_almost_equal(x_covs_p, test_x_covs_p, decimal=8)


        test_x_means_f = test_x_means_f.reshape(len(self.y), self.d_x)
        test_x_means_p = test_x_means_p.reshape(len(self.y), self.d_x)

        (gains_s, x_means_s, x_covs_s) = self.kfs.run_smoother(
            x_means_f=x_means_f, x_covs_f=x_covs_f, 
            x_means_p=x_means_p, x_covs_p=x_covs_p
        )

        (test_x_means_s, test_x_covs_s, test_gains_s) = _smooth(
            transition_matrices=self.A, 
            filtered_state_means=test_x_means_f, 
            filtered_state_covariances=test_x_covs_f,
            predicted_state_means=test_x_means_p, 
            predicted_state_covariances=test_x_covs_p
        )

        test_x_means_s = test_x_means_s.reshape(len(self.y), self.d_x, 1)

        npt.assert_array_almost_equal(gains_s, test_gains_s, decimal=8)
        npt.assert_array_almost_equal(x_means_s, test_x_means_s, decimal=8)
        npt.assert_array_almost_equal(x_covs_s, test_x_covs_s, decimal=8)



    def test_input_filtering(self):

        (gains, x_means_f, x_covs_f, 
        x_means_p, x_covs_p) = self.kfs_add_input.run_filter(
            y=self.y.reshape(len(self.y), self.d_y, 1), 
            ux=self.u_x.reshape(len(self.y), self.d_ux, 1), 
            uy=self.u_y.reshape(len(self.y), self.d_uy, 1)
        )

        (test_x_means_p, test_x_covs_p, 
        test_gain, test_x_means_f, test_x_covs_f) = _filter(
            transition_matrices=self.A, 
            observation_matrices=self.C.reshape(self.d_y, self.d_x),
            transition_offsets=np.asarray([[1, 1]]).reshape(self.d_x,), 
            observation_offsets=np.asarray([1]),
            transition_covariance=self.x_cov, 
            observation_covariance=self.y_cov.reshape(self.d_y, self.d_y),
            initial_state_mean=self.x_mean.reshape(self.d_x,), 
            initial_state_covariance=self.x_cov,
            observations=self.y.reshape(len(self.y), 1)
        )
        
        test_x_means_f = test_x_means_f.reshape(len(self.y), self.d_x, 1)
        test_x_means_p = test_x_means_p.reshape(len(self.y), self.d_x, 1)

        npt.assert_array_almost_equal(gains, test_gain, decimal=8)
        npt.assert_array_almost_equal(x_means_f, test_x_means_f, decimal=8)
        npt.assert_array_almost_equal(x_covs_f, test_x_covs_f, decimal=8)
        npt.assert_array_almost_equal(x_means_p, test_x_means_p, decimal=8)
        npt.assert_array_almost_equal(x_covs_p, test_x_covs_p, decimal=8)



    def test_loglikelihoods(self):

        (_, _, _, x_means_p, x_covs_p) = self.kfs.run_filter(
            y=self.y.reshape(len(self.y), self.d_y, 1)
        )

        loglikelihood = self.kfs.compute_likelihoods( 
            x_means_p=x_means_p, x_covs_p=x_covs_p
        )

        (test_x_means_p, test_x_covs_p, _, _, _) = _filter(
            transition_matrices=self.A, 
            observation_matrices=self.C.reshape(self.d_y, self.d_x),
            transition_offsets=np.zeros(self.d_x), 
            observation_offsets=np.zeros(self.d_y),
            transition_covariance=self.x_cov, 
            observation_covariance=self.y_cov.reshape(self.d_y, self.d_y),
            initial_state_mean=self.x_mean.reshape(self.d_x,), 
            initial_state_covariance=self.x_cov,
            observations=self.y.reshape(len(self.y), 1)
        )

        tets_loglikelihoods = _loglikelihoods(
            observation_matrices=self.C.reshape(self.d_y, self.d_x), 
            observation_offsets=np.zeros(self.d_y),
            observation_covariance=self.y_cov, 
            predicted_state_means=test_x_means_p,
            predicted_state_covariances=test_x_covs_p, 
            observations=self.y.reshape(len(self.y), 1)
        )

        npt.assert_array_almost_equal(loglikelihood, np.sum(tets_loglikelihoods), decimal=8)



    def test_likelihoods(self):

        (_, _, _, x_means_p, x_covs_p) = self.kfs.run_filter(
            y=self.y.reshape(len(self.y), self.d_y, 1)
        )

        likelihood = self.kfs.compute_likelihoods(
            x_means_p=x_means_p, x_covs_p=x_covs_p, log=False
        )

        (test_x_means_p, test_x_covs_p, _, _, _) = _filter(
            transition_matrices=self.A, 
            observation_matrices=self.C.reshape(self.d_y, self.d_x),
            transition_offsets=np.zeros(self.d_x), 
            observation_offsets=np.zeros(self.d_y),
            transition_covariance=self.x_cov, 
            observation_covariance=self.y_cov.reshape(self.d_y, self.d_y),
            initial_state_mean=self.x_mean.reshape(self.d_x,), 
            initial_state_covariance=self.x_cov,
            observations=self.y.reshape(len(self.y), 1)
        )

        tets_loglikelihoods = _loglikelihoods(
            observation_matrices=self.C.reshape(self.d_y, self.d_x), 
            observation_offsets=np.zeros(self.d_y),
            observation_covariance=self.y_cov, 
            predicted_state_means=test_x_means_p,
            predicted_state_covariances=test_x_covs_p, 
            observations=self.y.reshape(len(self.y), 1)
        )

        tets_likelihoods = np.sum(np.exp(tets_loglikelihoods))

        npt.assert_array_almost_equal(likelihood, tets_likelihoods, decimal=8)



    def test_input_loglikelihoods(self):

        (_, _, _, x_means_p, x_covs_p) = self.kfs_add_input.run_filter(
            y=self.y.reshape(len(self.y), self.d_y, 1), 
            ux=self.u_x.reshape(len(self.y), self.d_ux, 1), 
            uy=self.u_y.reshape(len(self.y), self.d_uy, 1)
        )

        loglikelihoods = self.kfs_add_input.compute_likelihoods(
            x_means_p=x_means_p, x_covs_p=x_covs_p
        )

        (test_x_means_p, test_x_covs_p, _, _, _) = _filter(
            transition_matrices=self.A, 
            observation_matrices=self.C.reshape(self.d_y, self.d_x),
            transition_offsets=np.asarray([[1, 1]]).reshape(self.d_x,), 
            observation_offsets=np.asarray([1]),
            transition_covariance=self.x_cov, 
            observation_covariance=self.y_cov.reshape(self.d_y, self.d_y),
            initial_state_mean=self.x_mean.reshape(self.d_x,), 
            initial_state_covariance=self.x_cov,
            observations=self.y.reshape(len(self.y), 1)
        )

        tets_loglikelihoods = _loglikelihoods(
            observation_matrices=self.C.reshape(self.d_y, self.d_x), 
            observation_offsets=np.asarray([1]),
            observation_covariance=self.y_cov, 
            predicted_state_means=test_x_means_p,
            predicted_state_covariances=test_x_covs_p, 
            observations=self.y.reshape(len(self.y), 1)
        )

        npt.assert_array_almost_equal(loglikelihoods, np.sum(tets_loglikelihoods), decimal=8)



if __name__ == '__main__':
    unittest.main()