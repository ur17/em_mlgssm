import numpy as np
from em_mlgssm.utils.pinv import pseudo_inverse



class KalmanFilter_and_Smoother(object):
    
    def __init__(self, time_series, state_dim, obs_dim):

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.data = np.asarray(time_series).reshape(len(time_series), self.obs_dim, 1)
    

    def param_init(self, state_mat, state_cov, obs_mat, obs_cov,
        init_state_mean, init_state_cov):

        self.state_mat = state_mat.reshape(self.state_dim, self.state_dim)
        self.state_cov = state_cov.reshape(self.state_dim, self.state_dim)
        self.obs_mat = obs_mat.reshape(self.obs_dim, self.state_dim)
        self.obs_cov = obs_cov.reshape(self.obs_dim, self.obs_dim)
        self.init_state_mean = init_state_mean.reshape(self.state_dim,)
        self.init_state_cov = init_state_cov.reshape(self.state_dim, self.state_dim)


    def _state_predict(self, filt_state_mean, filt_state_cov):

        pred_state_mean = np.dot(
            self.state_mat, 
            filt_state_mean.reshape(self.state_dim, 1)
        ).reshape(self.state_dim,)

        pred_state_cov = (
            np.dot(self.state_mat, np.dot(filt_state_cov, self.state_mat.T)) 
            + self.state_cov
        ).reshape(self.state_dim, self.state_dim)

        return (pred_state_mean, pred_state_cov)


    def _obs_predict(self, pred_state_mean, pred_state_cov):

        pred_obs_mean = np.dot(
            self.obs_mat, 
            pred_state_mean.reshape(self.state_dim, 1)
        ).reshape(self.obs_dim, self.obs_dim)

        pred_obs_cov = (
            np.dot(self.obs_mat, np.dot(pred_state_cov, self.obs_mat.T)) 
            + self.obs_cov
        ).reshape(self.obs_dim, self.obs_dim)

        return (pred_obs_mean, pred_obs_cov)


    def _state_filter(self, pred_state_mean, pred_state_cov, obs):

        (pred_obs_mean, pred_obs_cov) = self._obs_predict(
            pred_state_mean, pred_state_cov
        )

        kalman_gain = np.dot(
            pred_state_cov, 
            np.dot(self.obs_mat.T, pseudo_inverse(pred_obs_cov))
        )
        
        filt_state_mean = (
            pred_state_mean.reshape(self.state_dim, 1)
            + np.dot(kalman_gain, obs - pred_obs_mean)
        ).reshape(self.state_dim,)

        filt_state_cov = (
            pred_state_cov
            - np.dot(kalman_gain, np.dot(self.obs_mat, pred_state_cov))
        ).reshape(self.state_dim, self.state_dim)

        return (kalman_gain, filt_state_mean, filt_state_cov)

    
    def _state_smoother(self, filt_state_mean, filt_state_cov, pred_state_mean, 
        pred_state_cov, smooth_next_state_mean, smooth_next_state_cov):

        smooth_gain = np.dot(
            filt_state_cov, 
            np.dot(self.state_mat.T, pseudo_inverse(pred_state_cov))
        ).reshape(self.state_dim, self.state_dim)

        mean_diff = (
            smooth_next_state_mean.reshape(self.state_dim, 1) 
            - pred_state_mean.reshape(self.state_dim, 1)
        )
        
        smooth_state_mean = (
            filt_state_mean.reshape(self.state_dim, 1) 
            + np.dot(smooth_gain, mean_diff)
        ).reshape(self.state_dim,)
        
        smooth_state_cov = (
            filt_state_cov 
            + np.dot(
                smooth_gain, 
                np.dot(smooth_next_state_cov - pred_state_cov, smooth_gain.T))
        ).reshape(self.state_dim, self.state_dim)

        return (smooth_gain, smooth_state_mean, smooth_state_cov)


    def filtering(self):
        
        kalman_gains = np.empty((len(self.data), self.state_dim, self.obs_dim))
        filt_state_means = np.empty((len(self.data), self.state_dim))
        filt_state_covs = np.empty((len(self.data), self.state_dim, self.state_dim))
        pred_state_means = np.empty((len(self.data), self.state_dim))
        pred_state_covs = np.empty((len(self.data), self.state_dim, self.state_dim))

        pred_state_means[0], pred_state_covs[0] = self.init_state_mean, self.init_state_cov
        for t in range(len(self.data)-1):
            (kalman_gains[t], filt_state_means[t], 
            filt_state_covs[t]) = self._state_filter( 
                pred_state_means[t], pred_state_covs[t], self.data[t]
            )

            (pred_state_means[t+1], pred_state_covs[t+1]) = self._state_predict(
                filt_state_means[t], filt_state_covs[t]
            )
        (kalman_gains[-1], filt_state_means[-1], 
        filt_state_covs[-1]) = self._state_filter( 
            pred_state_means[-1], pred_state_covs[-1], self.data[-1]
        )

        return (kalman_gains, filt_state_means, filt_state_covs, pred_state_means, pred_state_covs)


    def smoothing(self, filt_state_means, filt_state_covs, pred_state_means, pred_state_covs):
        
        smooth_gains = np.empty((len(self.data) - 1, self.state_dim, self.state_dim))
        smooth_state_means = np.empty((len(self.data), self.state_dim))
        smooth_state_covs = np.empty((len(self.data), self.state_dim, self.state_dim))

        smooth_state_means[-1], smooth_state_covs[-1] = filt_state_means[-1], filt_state_covs[-1]
        for t in reversed(range(len(self.data) - 1)):
            (smooth_gains[t], smooth_state_means[t], 
            smooth_state_covs[t]) = self._state_smoother(
                filt_state_means[t], filt_state_covs[t],
                pred_state_means[t+1], pred_state_covs[t+1],
                smooth_state_means[t+1], smooth_state_covs[t+1]
        )

        return (smooth_gains, smooth_state_means, smooth_state_covs)


    def compute_statistics_for_mstep(self, smooth_gains, smooth_state_means, smooth_state_covs):

        e_zn = smooth_state_means.reshape(len(self.data), self.state_dim, 1)

        e_znzn = smooth_state_covs + np.einsum("nij,nlj->nil", e_zn, e_zn)
        e_znzn_1 = (
            np.einsum("nij,nlj->nil", smooth_state_covs[1:], smooth_gains) 
            + np.einsum("nij,nlj->nil", e_zn[1:], e_zn[:-1])
        )

        return (e_zn, e_znzn, e_znzn_1)


    def compute_loglikelihoods(self, pred_state_means, pred_state_covs):

        pred_state_means = pred_state_means.reshape(len(self.data), self.state_dim, 1)

        pred_obs_means = np.einsum("ij,njl->nil", self.obs_mat, pred_state_means)     
        pred_obs_covs = (
            np.einsum("ij,njl,kl->nik", self.obs_mat, pred_state_covs, self.obs_mat) 
            + self.obs_cov
        )
        root_covs = np.sqrt(pred_obs_covs)

        log_exp = - 0.5 * (self.data - pred_obs_means)**2 / pred_obs_covs
        log_coef = - 0.5 * np.log(2*np.pi) - np.log(root_covs)

        return np.sum(log_exp + log_coef, axis = 0)