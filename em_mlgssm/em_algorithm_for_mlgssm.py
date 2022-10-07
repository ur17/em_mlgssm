import numpy as np
import collections

from .kalmanfilter_and_smoother import KalmanFilter_and_Smoother
from em_mlgssm.utils.pinv import pseudo_inverse



class EM_mlgssm(object):

    def __init__(self, time_series_set, cluster_num, state_dim, obs_dim):

        self.cluster_num = cluster_num
        self.data_num = len(time_series_set)
        self.time_num = len(time_series_set[0])

        self.state_dim, self.obs_dim = state_dim, obs_dim

        self.dataset = np.asarray(
            time_series_set
        ).reshape(self.data_num, self.time_num, self.obs_dim, 1)

    
    def param_init(self, state_mats, state_covs, obs_mats, obs_covs, 
        init_state_means, init_state_covs, weights):

        self.state_mats = np.asarray(
            state_mats.copy()
        ).reshape(self.cluster_num, self.state_dim, self.state_dim)
        self.state_covs = np.asarray(
            state_covs.copy()
        ).reshape(self.cluster_num, self.state_dim, self.state_dim)
        self.obs_mats = np.asarray(
            obs_mats.copy()
        ).reshape(self.cluster_num, self.obs_dim, self.state_dim)
        self.obs_covs = np.asarray(
            obs_covs.copy()
        ).reshape(self.cluster_num, self.obs_dim, self.obs_dim)
        self.init_state_means = np.asarray(
            init_state_means.copy()
        ).reshape(self.cluster_num, self.state_dim, 1)
        self.init_state_covs = np.asarray(
            init_state_covs.copy()
        ).reshape(self.cluster_num, self.state_dim, self.state_dim)
        self.weights = np.asarray(
            weights.copy()
        ).reshape(self.cluster_num, 1, 1)

        
    def kalman_init(self, data_index, cluster_index):

        kfs = KalmanFilter_and_Smoother(
            time_series = self.dataset[data_index], 
            state_dim = self.state_dim, obs_dim = self.obs_dim
        )

        kfs.param_init(
            state_mat = self.state_mats[cluster_index], 
            state_cov = self.state_covs[cluster_index], 
            obs_mat = self.obs_mats[cluster_index], 
            obs_cov = self.obs_covs[cluster_index],
            init_state_mean = self.init_state_means[cluster_index], 
            init_state_cov = self.init_state_covs[cluster_index]
        )

        return kfs


    def e_step(self):
        
        posterior_prob = np.empty((self.data_num, self.cluster_num, 1, 1))
        e_zn = np.empty(
            (self.cluster_num, self.data_num, self.time_num, self.state_dim, 1)
        )
        e_znzn = np.empty(
            (self.cluster_num, self.data_num, self.time_num, self.state_dim, self.state_dim)
        )
        e_znzn_1 = np.empty(
            (self.cluster_num, self.data_num, self.time_num - 1, self.state_dim, self.state_dim)
        )
        for n in range(self.data_num):

            loglikelihoods = np.empty((self.cluster_num, 1, 1))
            for k in range(self.cluster_num):

                kfs = self.kalman_init(data_index = n, cluster_index = k)

                (_, filt_means, filt_covs, 
                pred_state_means, pred_state_covs) = kfs.filtering()

                (smooth_gains, smooth_means, 
                smooth_covs) = kfs.smoothing(
                    filt_state_means = filt_means, 
                    filt_state_covs = filt_covs,
                    pred_state_means = pred_state_means, 
                    pred_state_covs = pred_state_covs
                )

                (e_zn[k][n], e_znzn[k][n], 
                e_znzn_1[k][n]) = kfs.compute_statistics_for_mstep(
                    smooth_gains = smooth_gains,
                    smooth_state_means = smooth_means, 
                    smooth_state_covs = smooth_covs
                )

                loglikelihoods[k] = (
                    kfs.compute_loglikelihoods(pred_state_means, pred_state_covs) 
                    + np.log(self.weights[k] + 1e-7)
                )

            loglikelihoods -= np.max(loglikelihoods)
            posterior_prob[n] = (
                (np.exp(loglikelihoods) / np.sum(np.exp(loglikelihoods))) + 1e-7
            )

        return (posterior_prob, e_zn, e_znzn, e_znzn_1)


    def _update_init_state_means(self, posterior_prob_k, e_zn_k):

        init_state_means = np.sum(
            np.einsum("nil,ndk->ndk", posterior_prob_k, e_zn_k[:,0])
            , axis = 0
        )

        return init_state_means / np.sum(posterior_prob_k, axis = 0)


    def _update_init_state_covs(self, posterior_prob_k, e_zn_k, e_znzn_k, init_state_mean_k):

        lgssm_mstep = (
            e_znzn_k[:,0]
            - np.einsum("il,nkl->nik", init_state_mean_k, e_zn_k[:,0])
            - np.einsum("nkl,il->nki", e_zn_k[:,0], init_state_mean_k)
            + np.dot(init_state_mean_k, init_state_mean_k.T)
        )
        init_state_covs = np.sum(
            np.einsum("nil,nkj->nkj", posterior_prob_k, lgssm_mstep)
            , axis = 0
        )

        return init_state_covs / np.sum(posterior_prob_k, axis = 0)


    def _update_state_mat(self, posterior_prob_k, e_zn_k, e_znzn_k, e_znzn_1_k):
        
        left_mat = np.sum(
            np.einsum("nil,ntdk->ntdk", posterior_prob_k, e_znzn_1_k)
            , axis = (0,1)
        )

        right_mat = np.sum(
            np.einsum("nil,ntdk->ntdk", posterior_prob_k, e_znzn_k[:,:-1])
            , axis = (0,1)
        )

        return np.dot(left_mat, pseudo_inverse(right_mat))


    def _update_state_cov(self, posterior_prob_k, e_zn_k, e_znzn_k, e_znzn_1_k, state_mat_k):

        lgssm_mstep = (
            e_znzn_k[:,1:]
            - np.einsum("il,ntkl->ntik", state_mat_k, e_znzn_1_k)
            - np.einsum("ntkl,il->ntki", e_znzn_1_k, state_mat_k)
            + np.einsum("ik,ntkl,jl->ntij", state_mat_k, e_znzn_k[:,:-1], state_mat_k)
        )

        state_cov = np.sum(
            np.einsum("nil,ntdk->ntdk", posterior_prob_k, lgssm_mstep)
            , axis = (0,1)
        )

        return state_cov / (np.sum(posterior_prob_k) * (self.time_num - 1))


    def _update_obs_mat(self, posterior_prob_k, e_zn_k, e_znzn_k):

        left_mat = np.sum(
            np.einsum("nil,ntdk->ntdk", 
                posterior_prob_k, np.einsum("ntij,ntdj->ntid", self.dataset, e_zn_k))
            , axis = (0,1)
        )

        right_mat = np.sum(
            np.einsum("nil,ntdk->ntdk", posterior_prob_k, e_znzn_k)
            , axis = (0,1)
        )

        return np.dot(left_mat, pseudo_inverse(right_mat))


    def _update_obs_cov(self, posterior_prob_k, e_zn_k, e_znzn_k, obs_mat_k):

        lgssm_mstep = (
            np.einsum("ntij,ntkj->ntik", self.dataset, self.dataset)
            - np.einsum("ij,ntjk,ntlk->ntil", obs_mat_k, e_zn_k, self.dataset)
            - np.einsum("ntlk,ntjk,ij->ntli", self.dataset, e_zn_k, obs_mat_k)
            + np.einsum("ij,ntjk,lk->ntil", obs_mat_k, e_znzn_k, obs_mat_k)
        )

        obs_cov = np.sum(
            np.einsum("nil,ntdk->ntdk", posterior_prob_k, lgssm_mstep)
            , axis = (0,1)
        )

        return obs_cov / (np.sum(posterior_prob_k) * self.time_num)


    def _update_weight(self, posterior_prob_k):

        return np.mean(posterior_prob_k)


    def m_step(self, posterior_prob, e_zn, e_znzn, e_znzn_1, param_fix: list = []):

        for k in range(self.cluster_num):

            self.weights[k] = self._update_weight(posterior_prob[:,k])
            
            if not 0 in param_fix:
                self.init_state_means[k] = self._update_init_state_means(
                    posterior_prob[:,k], e_zn[k]
                )
            if not 1 in param_fix:
                self.init_state_covs[k] = self._update_init_state_covs(
                    posterior_prob[:,k], e_zn[k], e_znzn[k], self.init_state_means[k]
                )
            if not 2 in param_fix:
                self.state_mats[k] = self._update_state_mat(
                    posterior_prob[:,k], e_zn[k], e_znzn[k], e_znzn_1[k]
                )
            if not 3 in param_fix:
                self.state_covs[k] = self._update_state_cov(
                    posterior_prob[:,k], e_zn[k], e_znzn[k], e_znzn_1[k], self.state_mats[k]
                )
            if not 4 in param_fix:
                self.obs_mats[k] = self._update_obs_mat(
                    posterior_prob[:,k], e_zn[k], e_znzn[k]
                )
            if not 5 in param_fix:
                self.obs_covs[k] = self._update_obs_cov(
                    posterior_prob[:,k], e_zn[k], e_znzn[k], self.obs_mats[k]
                )

        return (self.init_state_means, self.init_state_covs, self.state_mats, self.state_covs, 
                self.obs_mats, self.obs_covs, self.weights)


    def fit(self, max_iter, param_epsilon, param_fix: list = [], log = False):
    
        if log:
            print("--- START ---")

        param_diff = np.empty(max_iter)

        # E step
        (posterior_prob, e_zn, e_znzn, e_znzn_1) = self.e_step()

        for i in range(max_iter):

            state_mats = self.state_mats.copy()
            state_covs = self.state_covs.copy()
            obs_mats = self.obs_mats.copy() 
            obs_covs = self.obs_covs.copy()
            init_state_means = self.init_state_means.copy()
            init_state_covs = self.init_state_covs.copy()

            # M step
            params = self.m_step(posterior_prob, e_zn, e_znzn, e_znzn_1, param_fix)

            # E step
            (posterior_prob, e_zn, e_znzn, e_znzn_1) = self.e_step()
            
            # clustering
            pred_cluster = np.argmax(posterior_prob, axis = 1)
            cluster_count = collections.Counter(pred_cluster.reshape(self.data_num,))

            param_diff[i] = (
                np.abs(self.init_state_means - init_state_means).sum() 
                + np.abs(self.init_state_covs - init_state_covs).sum() 
                + np.abs(self.state_mats - state_mats).sum() 
                + np.abs(self.state_covs - state_covs).sum() 
                + np.abs(self.obs_mats - obs_mats).sum()
                + np.abs(self.obs_covs - obs_covs).sum()
            )
                
            if log:
                print(f"iter = {i+1:02d}; {cluster_count.most_common()}; p-diff = {np.round(param_diff[i], decimals=3)}")
                
            if param_diff[i] < param_epsilon:
                if log:
                    print("--- FINISH ---")
                return params, param_diff[:i+1], pred_cluster.reshape(self.data_num)

        if log:
            print("--- FINISH ---")
        return params, param_diff, pred_cluster.reshape(self.data_num)