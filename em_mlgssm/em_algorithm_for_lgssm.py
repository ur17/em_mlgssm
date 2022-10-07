import numpy as np

from .kalmanfilter_and_smoother import KalmanFilter_and_Smoother
from em_mlgssm.utils.pinv import pseudo_inverse



class EM_lgssm(KalmanFilter_and_Smoother):
    
    def __init__(self, time_series, state_dim, obs_dim):

        super().__init__(
            time_series = time_series, 
            state_dim = state_dim, obs_dim = obs_dim
        )
    
        
    def e_step(self):

        (_, filt_state_means, filt_state_covs, 
        pred_state_means, pred_state_covs) = self.filtering()

        (self.smooth_gains, self.smooth_state_means, 
        self.smooth_state_covs) = self.smoothing(
            filt_state_means = filt_state_means, 
            filt_state_covs = filt_state_covs,
            pred_state_means = pred_state_means, 
            pred_state_covs = pred_state_covs
        )
        
        (e_zn, e_znzn, e_znzn_1) = self.compute_statistics_for_mstep(
            smooth_gains = self.smooth_gains,
            smooth_state_means = self.smooth_state_means, 
            smooth_state_covs = self.smooth_state_covs
        )

        return (e_zn, e_znzn, e_znzn_1)


    def _update_state_mat(self, e_zn, e_znzn, e_znzn_1):

        left_mat = np.sum(e_znzn_1, axis = 0)
        right_mat = np.sum(e_znzn[:-1], axis = 0)

        return np.dot(left_mat, pseudo_inverse(right_mat))


    def _update_state_cov(self, e_zn, e_znzn, e_znzn_1):

        state_cov = np.sum(
            e_znzn[1:] 
            - np.einsum("ij,nlj->nil", self.state_mat, e_znzn_1) 
            - np.einsum("nlj,ij->nli", e_znzn_1, self.state_mat)
            + np.einsum("ij,njk,lk->nil", self.state_mat, e_znzn[:-1], self.state_mat)
            , axis = 0
        )

        return state_cov / (len(self.data) - 1)


    def _update_obs_mat(self, e_zn, e_znzn):

        left_mat = np.sum(np.einsum("nij,nkj->nik", self.data, e_zn), axis=0)
        right_mat = np.sum(e_znzn, axis=0)

        return np.dot(left_mat, pseudo_inverse(right_mat))


    def _update_obs_cov(self, e_zn, e_znzn):

        obs_cov = np.sum(
            np.einsum("nij,nkj->nik", self.data, self.data)
            - np.einsum("ij,njk,nlk->nil", self.obs_mat, e_zn, self.data) 
            - np.einsum("nlk,njk,ij->nli", self.data, e_zn, self.obs_mat) 
            + np.einsum("ij,njk,lk->nil", self.obs_mat, e_znzn, self.obs_mat)
            , axis = 0
        )

        return obs_cov / len(self.data)
        
        
    def m_step(self, e_zn, e_znzn, e_znzn_1, param_fix: list = []):
        
        if not 0 in param_fix:
            self.init_state_mean = self.smooth_state_means[0]
        if not 1 in param_fix:
            self.init_state_cov = self.smooth_state_covs[0]
        if not 2 in param_fix:
            self.state_mat = self._update_state_mat(e_zn, e_znzn, e_znzn_1)
        if not 3 in param_fix:
            self.state_cov = self._update_state_cov(e_zn, e_znzn, e_znzn_1)
        if not 4 in param_fix:
            self.obs_mat = self._update_obs_mat(e_zn, e_znzn)
        if not 5 in param_fix:
            self.obs_cov = self._update_obs_cov(e_zn, e_znzn)
            
        return (self.init_state_mean, self.init_state_cov, 
                self.state_mat, self.state_cov, self.obs_mat, self.obs_cov)


    def fit(self, max_iter, param_epsilon, param_fix: list = []):

        self.param_diff = np.empty(max_iter)

        for i in range(max_iter):

            state_mat = self.state_mat.copy()
            state_cov = self.state_cov.copy()
            obs_mat = self.obs_mat.copy() 
            obs_cov = self.obs_cov.copy()
            init_state_mean = self.init_state_mean.copy()
            init_state_cov = self.init_state_cov.copy()

            (e_zn, e_znzn, e_znzn_1) = self.e_step()
            params = self.m_step(e_zn, e_znzn, e_znzn_1, param_fix)

            self.param_diff[i] = (
                np.abs(self.init_state_mean - init_state_mean).sum() 
                + np.abs(self.init_state_cov - init_state_cov).sum() 
                + np.abs(self.state_mat - state_mat).sum() 
                + np.abs(self.state_cov - state_cov).sum() 
                + np.abs(self.obs_mat - obs_mat).sum()
                + np.abs(self.obs_cov - obs_cov).sum()
            )

            if self.param_diff[i] < param_epsilon:
                return params, self.param_diff[:i+1]

        return params, self.param_diff