import numpy as np
from pykalman.standard import _loglikelihoods

from em_mlgssm.kalman import kalman_filter
from em_mlgssm.kalman import kalman_smoother
from em_mlgssm.utils.matrix import random_matrix
from .e_step import _compute_statistics_for_estep
from .m_step import _update_state_mat
from .m_step import _update_state_cov
from .m_step import _update_obs_mat
from .m_step import _update_obs_cov



class EM_lgssm(object):
    
    def __init__(self, time, state_dim, obs_dim, obs_mat_fix = "default", loglikelihood = False):

        self.T = time
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.obs_mat_fix = obs_mat_fix

        self.loglikelihood = loglikelihood
        if self.loglikelihood == True:
            self.loglikelihood_list = []

        self.param_diff = []


    def param_init(self,
        state_mat, state_cov, obs_cov,
        init_state_mean, init_state_cov, obs_mat = None):
        
        if self.obs_mat_fix == "default":
            self.obs_mat = np.asarray([1] * self.state_dim)
        else:
            self.obs_mat = obs_mat

        self.state_mat = state_mat
        self.state_cov = state_cov
        self.obs_cov = obs_cov
        self.init_state_mean = init_state_mean
        self.init_state_cov = init_state_cov


    def kalman_filter(self, time_series):

        (pred_state_mean, pred_state_cov, kalman_gain, 
        filt_state_mean, filt_state_cov, _, _) = kalman_filter(
            time_series, self.state_mat, self.state_cov, self.obs_mat, 
            self.obs_cov, self.init_state_mean, self.init_state_cov
        )

        self.pred_state_means = pred_state_mean
        self.pred_state_covs = pred_state_cov
        self.filt_gains = kalman_gain
        self.filt_state_means = filt_state_mean
        self.filt_state_covs = filt_state_cov


    def kalman_smoother(self, time_series):
        (smooth_state_mean, smooth_state_cov, 
        smooth_gain) = kalman_smoother(
            time_series, self.state_mat, self.filt_state_means, 
            self.filt_state_covs, self.pred_state_covs
        )

        self.smooth_state_means = smooth_state_mean
        self.smooth_state_covs = smooth_state_cov
        self.smooth_gains = smooth_gain

    
    def compute_loglikelihood(self, obs_mat, obs_cov, time_series):
        pred_state_mean = self.pred_state_means
        pred_state_cov = self.pred_state_covs
        
        loglikelihoods = _loglikelihoods(
            obs_mat.reshape(1, self.state_dim), np.zeros(1), obs_cov, 
            pred_state_mean, pred_state_cov, time_series.reshape(self.T, self.obs_dim)
        )
        if self.loglikelihood == True:
            self.loglikelihood_list.append(np.sum(loglikelihoods))
        else:
            return np.sum(loglikelihoods)
    
        
    def e_step(self, time_series):
        
        self.kalman_filter(time_series)
        self.kalman_smoother(time_series)
        
        e_zn, e_znzn, e_znzn_1 = _compute_statistics_for_estep(
            self.smooth_state_means, 
            self.smooth_state_covs, 
            self.smooth_gains
        )

        return e_zn, e_znzn, e_znzn_1
        
        
    def m_step(self, time_series, e_zn, e_znzn, e_znzn_1):

        init_state_mean = self.smooth_state_means[0]
        init_state_cov = self.smooth_state_covs[0]

        if self.obs_mat_fix == False:
            obs_mat = _update_obs_mat(time_series, e_zn, e_znzn)
        else:
            obs_mat = self.obs_mat
    
        state_mat = _update_state_mat(e_znzn, e_znzn_1)
        state_cov = _update_state_cov(state_mat, e_znzn, e_znzn_1)
        
        obs_cov = _update_obs_cov(time_series, obs_mat, e_zn, e_znzn)

        if self.loglikelihood == True:
            self.compute_loglikelihood(obs_mat, obs_cov, time_series)
        
        param_diff = (
            np.abs(self.init_state_mean - init_state_mean).sum() 
            + np.abs(self.init_state_cov - init_state_cov).sum() 
            + np.abs(self.state_mat - state_mat).sum() 
            + np.abs(self.state_cov - state_cov).sum() 
            + np.abs(self.obs_mat - obs_mat).sum()
            + np.abs(self.obs_cov - obs_cov).sum()
        )

        self.param_diff.append(param_diff)

        self.param_init(
            state_mat, state_cov, obs_cov, 
            init_state_mean, init_state_cov, obs_mat
        )
        
        return state_mat, state_cov, obs_mat, obs_cov, init_state_mean, init_state_cov
    
    
    def param_update(self, 
        time_series, max_iter, epsilon = 0.01, log = False):

        if epsilon < 0:
            if log == True:
                print("--- START ---")
            for i in range(max_iter):
                if log == True:
                    if max_iter < 10:
                        print(f"iter = {i + 1}")
                    else:
                        if (i + 1) % (max_iter // 10) == 0:
                            print(f"iter = {i + 1}")

                e_zn, e_znzn, e_znzn_1 = self.e_step(time_series)
                param = self.m_step(time_series, e_zn, e_znzn, e_znzn_1)

            if log == True:
                print("--- FINISH ---")
            return param

        else:
            if log == True:
                print("--- START ---")
            for i in range(max_iter):
                if log == True:
                    if max_iter < 10:
                        print(f"iter = {i + 1}")
                    else:
                        if (i + 1) % (max_iter // 10) == 0:
                            print(f"iter = {i + 1}")
                        
                e_zn, e_znzn, e_znzn_1 = self.e_step(time_series)
                param = self.m_step(time_series, e_zn, e_znzn, e_znzn_1)
                if self.param_diff[-1] < epsilon:
                    if log == True:
                        print("--- FINISH ---")
                    return param
        
            if log == True:
                print("--- FINISH ---")
                
            return param


    def param_tuning(self, 
        time_series, num=2, max_iter = 50, seed = 0, 
        tuning_num = 5, epsilon = 0.01, log = False):

        np.random.seed(seed)

        theta = [num * np.pi * np.random.rand() for _ in range(tuning_num)]
        params_list = []
        likelihood_list = []

        state_cov = np.eye(self.state_dim) * 0.05
        obs_mat = np.asarray([1] * self.state_dim)
        obs_cov = np.eye(1) * 0.05
        init_state_mean = np.zeros(self.state_dim)
        init_state_cov = np.eye(self.state_dim) * 1e+4
        
        for i in range(tuning_num):
            state_mat = random_matrix(self.state_dim, theta[i])

            self.__init__(
                time = self.T, state_dim = self.state_dim, 
                obs_dim = self.obs_dim, obs_mat_fix = "default"
            )
            self.param_init(
                state_mat = state_mat, state_cov = state_cov, obs_cov = obs_cov,
                init_state_mean = init_state_mean, init_state_cov = init_state_cov, obs_mat = obs_mat
            )
            params = self.param_update(
                time_series, max_iter = max_iter, epsilon = epsilon, log = log
            )

            params_list.append(params)

            loglikelihood = self.compute_loglikelihood(params[2], params[3], time_series)
            likelihood_list.append(loglikelihood)

        max_index = np.argmax(np.asarray(likelihood_list))
        best_param = params_list[max_index]

        return (best_param, likelihood_list, params_list)