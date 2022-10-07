import unittest

import numpy as np
import numpy.testing as npt

from pykalman.standard import _filter
from pykalman.standard import _smooth
from pykalman.standard import _filter_predict
from pykalman.standard import _filter_correct
from pykalman.standard import _smooth_update
from pykalman.standard import _loglikelihoods

from em_mlgssm import KalmanFilter_and_Smoother



class Test_filter_and_smoother(unittest.TestCase):

    def setUp(self):
        
        self.obs = 10
        self.state_dim = 2
        self.obs_dim = 1
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
        self.time_series = np.asarray([0.2, 0.1, 0.6, 0.5, 0.9])

        self.kf = KalmanFilter_and_Smoother(
            time_series = self.time_series, 
            state_dim = self.state_dim, 
            obs_dim = self.obs_dim
        )
        self.kf.param_init(
            state_mat = self.state_mat, 
            state_cov = self.any_state_cov, 
            obs_mat = self.obs_mat, 
            obs_cov = self.obs_cov,
            init_state_mean = self.any_state_mean, 
            init_state_cov = self.any_state_cov
        )


    def test_predict_and_filter(self):

        (pred_state_mean, pred_state_cov) = self.kf._state_predict(
            filt_state_mean = self.any_state_mean, 
            filt_state_cov = self.any_state_cov
        )
        (test_pred_state_mean, test_pred_state_cov) = _filter_predict(
            transition_matrix = self.state_mat, 
            transition_covariance = self.any_state_cov,
            transition_offset = np.zeros(self.state_dim).reshape(self.state_dim, self.obs_dim), 
            current_state_mean = self.any_state_mean.reshape(self.state_dim, self.obs_dim),
            current_state_covariance = self.any_state_cov
        )
        test_pred_state_mean = test_pred_state_mean.reshape(self.state_dim,)
        npt.assert_array_almost_equal(pred_state_mean, test_pred_state_mean, decimal=8)
        npt.assert_array_almost_equal(pred_state_cov, test_pred_state_cov, decimal=8)

        (kalman_gain, filt_state_mean, 
        filt_state_cov) = self.kf._state_filter(
            pred_state_mean = pred_state_mean, 
            pred_state_cov = pred_state_cov, 
            obs = self.obs
        )
        (test_kalman_gain, test_filt_state_mean, 
        test_filt_state_cov) = _filter_correct(
            observation_matrix = self.obs_mat.reshape(self.obs_dim, self.state_dim), 
            observation_covariance = self.obs_cov,
            observation_offset = np.zeros(self.obs_dim), 
            predicted_state_mean = test_pred_state_mean.reshape(self.state_dim,),
            predicted_state_covariance = test_pred_state_cov, 
            observation = self.obs
        )
        npt.assert_array_almost_equal(kalman_gain, test_kalman_gain, decimal=8)
        npt.assert_array_almost_equal(filt_state_mean, test_filt_state_mean, decimal=8)
        npt.assert_array_almost_equal(filt_state_cov, test_filt_state_cov, decimal=8)


    def test_state_smoother(self):
        (smooth_gain, smooth_state_mean, 
        smooth_state_cov) = self.kf._state_smoother(
            filt_state_mean = self.any_state_mean, 
            filt_state_cov = self.any_state_cov,
            pred_state_mean = self.any_state_mean, 
            pred_state_cov = self.any_state_cov,
            smooth_next_state_mean = self.any_state_mean, 
            smooth_next_state_cov = self.any_state_cov
        )

        (test_smooth_state_mean, test_smooth_state_cov, 
        test_smooth_gain) = _smooth_update(
            transition_matrix = self.state_mat, 
            filtered_state_mean = self.any_state_mean.reshape(self.state_dim,),
            filtered_state_covariance = self.any_state_cov, 
            predicted_state_mean = self.any_state_mean.reshape(self.state_dim,),
            predicted_state_covariance = self.any_state_cov, 
            next_smoothed_state_mean = self.any_state_mean.reshape(self.state_dim,),
            next_smoothed_state_covariance = self.any_state_cov
        )

        npt.assert_array_almost_equal(smooth_gain, test_smooth_gain, decimal=8)
        npt.assert_array_almost_equal(smooth_state_mean, test_smooth_state_mean, decimal=8)
        npt.assert_array_almost_equal(smooth_state_cov, test_smooth_state_cov, decimal=8)


    def test_filtering(self):
        (kalman_gains, filt_state_means, 
        filt_state_covs, _, _) = self.kf.filtering()

        (_, _, test_kalman_gain, test_filt_state_mean, 
        test_filt_state_cov) = _filter(
            transition_matrices = self.state_mat, 
            observation_matrices = self.obs_mat.reshape(self.obs_dim, self.state_dim),
            transition_offsets = np.zeros(self.state_dim), 
            observation_offsets = np.zeros(self.obs_dim),
            transition_covariance = self.any_state_cov, 
            observation_covariance = self.obs_cov.reshape(self.obs_dim, self.obs_dim),
            initial_state_mean = self.any_state_mean.reshape(self.state_dim,), 
            initial_state_covariance = self.any_state_cov,
            observations = self.time_series.reshape(len(self.time_series), 1)
        )

        npt.assert_array_almost_equal(kalman_gains, test_kalman_gain, decimal=8)
        npt.assert_array_almost_equal(filt_state_means, test_filt_state_mean, decimal=8)
        npt.assert_array_almost_equal(filt_state_covs, test_filt_state_cov, decimal=8)


    def test_smoothing(self):
        (kalman_gains, filt_state_means, filt_state_covs, 
        pred_state_means, pred_state_covs) = self.kf.filtering()

        (smooth_gains, smooth_state_means, 
        smooth_state_covs) = self.kf.smoothing(
            filt_state_means = filt_state_means, 
            filt_state_covs = filt_state_covs,
            pred_state_means = pred_state_means, 
            pred_state_covs = pred_state_covs
        )

        (py_pred_state_mean, py_pred_state_covs, 
        py_kalman_gains, py_filt_state_means, 
        py_filt_state_covs) = _filter(
            transition_matrices = self.state_mat, 
            observation_matrices = self.obs_mat.reshape(self.obs_dim, self.state_dim),
            transition_offsets = np.zeros(self.state_dim), 
            observation_offsets = np.zeros(self.obs_dim),
            transition_covariance = self.any_state_cov, 
            observation_covariance = self.obs_cov.reshape(self.obs_dim, self.obs_dim),
            initial_state_mean = self.any_state_mean.reshape(self.state_dim,), 
            initial_state_covariance = self.any_state_cov,
            observations = self.time_series.reshape(len(self.time_series), 1)
        )

        (test_smooth_state_mean, test_smooth_state_cov, 
        test_smooth_gain) = _smooth(
            transition_matrices = self.state_mat, 
            filtered_state_means = py_filt_state_means, 
            filtered_state_covariances = py_filt_state_covs,
            predicted_state_means = py_pred_state_mean, 
            predicted_state_covariances = py_pred_state_covs
        )

        npt.assert_array_almost_equal(smooth_state_means, test_smooth_state_mean, decimal=8)
        npt.assert_array_almost_equal(smooth_state_covs, test_smooth_state_cov, decimal=8)
        npt.assert_array_almost_equal(smooth_gains, test_smooth_gain, decimal=8)


    def test_compute_loglikelihoods(self):
        (kalman_gains, filt_state_means, filt_state_covs, 
        pred_state_means, pred_state_covs) = self.kf.filtering()

        loglikelihood = self.kf.compute_loglikelihoods(pred_state_means, pred_state_covs)

        tets_loglikelihoods = _loglikelihoods(
            observation_matrices = self.obs_mat.reshape(self.obs_dim, self.state_dim), 
            observation_offsets = np.zeros(self.obs_dim),
            observation_covariance = self.obs_cov, 
            predicted_state_means = pred_state_means,
            predicted_state_covariances = pred_state_covs, 
            observations = self.time_series.reshape(len(self.time_series), 1)
        )

        npt.assert_array_almost_equal(loglikelihood, np.sum(tets_loglikelihoods), decimal=8)



if __name__ == '__main__':
    unittest.main()