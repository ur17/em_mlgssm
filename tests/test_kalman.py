import unittest

import numpy as np
import numpy.testing as npt

from pykalman.standard import _filter
from pykalman.standard import _smooth
from pykalman.standard import _filter_predict
from pykalman.standard import _filter_correct
from pykalman.standard import _smooth_update

from em_mlgssm.kalman.filter import _state_predict
from em_mlgssm.kalman.filter import _state_filter
from em_mlgssm.kalman.filter import kalman_filter
from em_mlgssm.kalman.smoother import _state_smoother
from em_mlgssm.kalman.smoother import kalman_smoother



class Test_filter_and_smoother(unittest.TestCase):

    def setUp(self):
        
        self.obs = 10
        self.state_dim = 2
        self.obs_dim = 1

        self.time_series = np.asarray([1, 1])

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

        self.input_state_vec = np.asarray([
            [1, 1]
        ])

        self.input_obs_vec = np.asarray([1])

        self.input = np.asarray([2])
        

    def test_state_predict(self):
        (pred_state_mean, pred_state_cov) = _state_predict(
            state_mat = self.state_mat, state_cov = self.any_state_cov, 
            current_state_mean = self.any_state_mean, 
            current_state_cov = self.any_state_cov
        )

        (test_pred_state_mean, test_pred_state_cov) = _filter_predict(
            transition_matrix = self.state_mat, 
            transition_covariance = self.any_state_cov,
            transition_offset = np.zeros(self.state_dim).reshape(self.state_dim, self.obs_dim), 
            current_state_mean = self.any_state_mean.reshape(self.state_dim, self.obs_dim),
            current_state_covariance = self.any_state_cov
        )

        test_pred_state_mean = test_pred_state_mean.reshape(self.state_dim,)

        npt.assert_array_almost_equal(pred_state_mean, test_pred_state_mean, decimal=6)
        npt.assert_array_almost_equal(pred_state_cov, test_pred_state_cov, decimal=6)


    def test_state_predict_with_input(self):
        (pred_state_mean, pred_state_cov) = _state_predict(
            state_mat = self.state_mat, state_cov = self.any_state_cov, 
            current_state_mean = self.any_state_mean, 
            current_state_cov = self.any_state_cov,
            input_state_vec = self.input_state_vec, 
            input_state = self.input
        )

        (test_pred_state_mean, test_pred_state_cov) = _filter_predict(
            transition_matrix = self.state_mat, 
            transition_covariance = self.any_state_cov,
            transition_offset = np.asarray([[2, 2]]).reshape(self.state_dim, self.obs_dim), 
            current_state_mean = self.any_state_mean.reshape(self.state_dim, self.obs_dim),
            current_state_covariance = self.any_state_cov
        )

        test_pred_state_mean = test_pred_state_mean.reshape(self.state_dim,)

        npt.assert_array_almost_equal(pred_state_mean, test_pred_state_mean, decimal=6)
        npt.assert_array_almost_equal(pred_state_cov, test_pred_state_cov, decimal=6)


    def test_state_filter(self):
        (kalman_gain, filt_state_mean, 
        filt_state_cov, _, _) = _state_filter(
            obs_mat = self.obs_mat, obs_cov = self.obs_cov, 
            predicted_state_mean = self.any_state_mean, 
            predicted_state_cov = self.any_state_cov, 
            obs = self.obs
        )

        (test_kalman_gain, test_filt_state_mean, 
        test_filt_state_cov) = _filter_correct(
            observation_matrix = self.obs_mat.reshape(self.obs_dim, self.state_dim), 
            observation_covariance = self.obs_cov,
            observation_offset = np.zeros(self.obs_dim), 
            predicted_state_mean = self.any_state_mean.reshape(self.state_dim,),
            predicted_state_covariance = self.any_state_cov, 
            observation = self.obs
        )

        npt.assert_array_almost_equal(kalman_gain, test_kalman_gain, decimal=8)
        npt.assert_array_almost_equal(filt_state_mean, test_filt_state_mean, decimal=8)
        npt.assert_array_almost_equal(filt_state_cov, test_filt_state_cov, decimal=8)


    def test_state_filter_with_input(self):
        (kalman_gain, filt_state_mean, 
        filt_state_cov, _, _) = _state_filter(
            obs_mat = self.obs_mat, obs_cov = self.obs_cov, 
            predicted_state_mean = self.any_state_mean, 
            predicted_state_cov = self.any_state_cov, 
            obs = self.obs,
            input_obs_vec = self.input_obs_vec, 
            input_obs = self.input
        )

        (test_kalman_gain, test_filt_state_mean, 
        test_filt_state_cov) = _filter_correct(
            observation_matrix = self.obs_mat.reshape(self.obs_dim, self.state_dim), 
            observation_covariance = self.obs_cov,
            observation_offset = np.asarray([2]), 
            predicted_state_mean = self.any_state_mean.reshape(self.state_dim,),
            predicted_state_covariance = self.any_state_cov, 
            observation = self.obs
        )

        npt.assert_array_almost_equal(kalman_gain, test_kalman_gain, decimal=8)
        npt.assert_array_almost_equal(filt_state_mean, test_filt_state_mean, decimal=8)
        npt.assert_array_almost_equal(filt_state_cov, test_filt_state_cov, decimal=8)


    def test_state_smoother(self):
        (smooth_gain, smooth_state_mean, 
        smooth_state_cov) = _state_smoother(
            self.state_mat, self.any_state_cov,
            self.any_state_mean, self.any_state_cov,
            self.any_state_mean, self.any_state_cov
        )

        (test_smooth_state_mean, test_smooth_state_cov, 
        test_smooth_gain) = _smooth_update(
            self.state_mat, self.any_state_mean.reshape(self.state_dim,),
            self.any_state_cov, self.any_state_mean.reshape(self.state_dim,),
            self.any_state_cov, self.any_state_mean.reshape(self.state_dim,),
            self.any_state_cov
        )

        npt.assert_array_almost_equal(smooth_gain, test_smooth_gain, decimal=8)
        npt.assert_array_almost_equal(smooth_state_mean, test_smooth_state_mean, decimal=8)
        npt.assert_array_almost_equal(smooth_state_cov, test_smooth_state_cov, decimal=8)


    def test_kalman_filter(self):
        (pred_state_mean, pred_state_cov, kalman_gain, 
        filt_state_mean, filt_state_cov, _, _) = kalman_filter(
            self.time_series, self.state_mat, self.any_state_cov, 
            self.obs_mat, self.obs_cov, 
            self.any_state_mean, self.any_state_cov
        )

        (test_pred_state_mean, test_pred_state_cov, test_kalman_gain, 
        test_filt_state_mean, test_filt_state_cov) = _filter(
            transition_matrices = self.state_mat, 
            observation_matrices = self.obs_mat.reshape(self.obs_dim, self.state_dim),
            transition_offsets = np.zeros(self.state_dim), 
            observation_offsets = np.zeros(self.obs_dim),
            transition_covariance = self.any_state_cov, 
            observation_covariance = self.obs_cov.reshape(self.obs_dim, self.obs_dim),
            initial_state_mean = self.any_state_mean.reshape(self.state_dim,), 
            initial_state_covariance = self.any_state_cov,
            observations = self.time_series.reshape(self.state_dim, self.obs_dim)
        )

        npt.assert_array_almost_equal(pred_state_mean, test_pred_state_mean, decimal=8)
        npt.assert_array_almost_equal(pred_state_cov, test_pred_state_cov, decimal=8)
        npt.assert_array_almost_equal(kalman_gain, test_kalman_gain, decimal=8)
        npt.assert_array_almost_equal(filt_state_mean, test_filt_state_mean, decimal=8)
        npt.assert_array_almost_equal(filt_state_cov, test_filt_state_cov, decimal=8)


    def test_kalman_smoother(self):
        (pred_state_mean, pred_state_cov, _, 
        filt_state_mean, 
        filt_state_cov, _, _) = kalman_filter(
            self.time_series, self.state_mat, self.any_state_cov, 
            self.obs_mat, self.obs_cov, 
            self.any_state_mean, self.any_state_cov
        )

        (smooth_state_mean, 
        smooth_state_cov, 
        smooth_gain) = kalman_smoother(
            self.time_series, self.state_mat, 
            filt_state_mean, 
            filt_state_cov, 
            pred_state_cov
        )

        (test_smooth_state_mean, 
        test_smooth_state_cov, 
        test_smooth_gain) = _smooth(
            transition_matrices = self.state_mat, 
            filtered_state_means = filt_state_mean, 
            filtered_state_covariances = filt_state_cov,
            predicted_state_means = pred_state_mean, 
            predicted_state_covariances = pred_state_cov
        )

        npt.assert_array_almost_equal(smooth_state_mean, test_smooth_state_mean, decimal=8)
        npt.assert_array_almost_equal(smooth_state_cov, test_smooth_state_cov, decimal=8)
        npt.assert_array_almost_equal(smooth_gain, test_smooth_gain, decimal=8)



if __name__ == '__main__':
    unittest.main()