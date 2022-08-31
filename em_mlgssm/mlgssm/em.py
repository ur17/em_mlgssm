import numpy as np
import collections

from pykalman.standard import _loglikelihoods

from em_mlgssm.kalman import kalman_filter
from em_mlgssm.kalman import kalman_smoother
from .likelihood  import _compute_mlgssm_likelihoods
from .m_step import _update_state_mat_k
from .m_step import _update_state_cov_k
from .m_step import _update_obs_mat_k
from .m_step import _update_obs_cov_k
from .m_step import _update_init_state_mean
from .m_step import _update_init_state_cov
from .m_step import _update_weight



def _make_dict(cluster_num):
    save_dict = {}
    for i in range(cluster_num):
        save_dict[f"cluster{i}"] = []

    return save_dict


class EM_mlgssm(object):
    
    def __init__(self, 
        cluster_num = None, data_num = None, time_len = None, 
        state_dim = None, obs_dim = None, loglikelihood=False, 
        obs_mat_fix = "default"
        ):
        
        self.cluster_num = cluster_num
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.T = time_len
        self.N = data_num
        self.obs_mat_fix = obs_mat_fix

        self.param_diff = []
        self.loglikelihood = loglikelihood

    
    def param_init(self,
        state_mat = None, state_cov = None, obs_mat = None, obs_cov = None, 
        init_state_mean = None, init_state_cov = None, weights = None,
        random_param_init = "default"
        ):
        
        if random_param_init == "default":
            self.state_mat = np.asarray([np.eye(self.state_dim)] * self.cluster_num)
            self.state_cov = np.asarray([np.eye(self.state_dim)] * self.cluster_num)
            self.obs_mat = np.ones(self.cluster_num * self.state_dim).reshape(self.cluster_num, self.state_dim)
            self.obs_cov = np.asarray([np.eye(self.obs_dim)] * self.cluster_num)
            self.init_state_mean = np.asarray([np.zeros(self.state_dim)] * self.cluster_num) 
            self.init_state_cov = np.asarray([np.eye(self.state_dim) * 1e+4] * self.cluster_num)
            self.weights = np.ones(self.cluster_num) / self.cluster_num
            
        elif random_param_init == False:
            self.state_mat = state_mat
            self.state_cov = state_cov
            self.obs_mat = obs_mat.reshape(self.cluster_num, self.state_dim)
            self.obs_cov = obs_cov
            self.init_state_mean = init_state_mean
            self.init_state_cov = init_state_cov
            self.weights = weights


    def kalman_init(self):
        self.filt_pred_mean = _make_dict(self.cluster_num)
        self.filt_pred_cov = _make_dict(self.cluster_num)
        self.filt_mean = _make_dict(self.cluster_num)
        self.filt_cov = _make_dict(self.cluster_num)
        self.filt_gain = _make_dict(self.cluster_num)

        self.smooth_mean = _make_dict(self.cluster_num)
        self.smooth_cov = _make_dict(self.cluster_num)
        self.smooth_gain = _make_dict(self.cluster_num)

        self.posterior_prob = np.zeros((self.N, self.cluster_num))


    def kalman_filter(self, dataset):
        for i in range(self.N):
            for k in range(self.cluster_num):

                state_mat_k =  self.state_mat[k]
                state_cov_k = self.state_cov[k]
                obs_mat_k = self.obs_mat[k]
                obs_cov_k = self.obs_cov[k]
                init_state_mean_k = self.init_state_mean[k]
                init_state_cov_k =  self.init_state_cov[k]


                (pred_state_means, pred_state_covs, kalman_gains, 
                filt_state_means, filt_state_covs, 
                _, _) = kalman_filter(
                    dataset[i], state_mat_k, state_cov_k, 
                    obs_mat_k, obs_cov_k, init_state_mean_k, init_state_cov_k
                )
                
                self.filt_mean[f"cluster{k}"].append(filt_state_means)
                self.filt_cov[f"cluster{k}"].append(filt_state_covs)
                self.filt_pred_mean[f"cluster{k}"].append(pred_state_means)
                self.filt_pred_cov[f"cluster{k}"].append(pred_state_covs)
                self.filt_gain[f"cluster{k}"].append(kalman_gains)
    

    def kalman_smoother(self, dataset):
        for i in range(self.N):
            for k in range(self.cluster_num):

                state_mat_k =  self.state_mat[k]

                (smooth_state_means, smooth_state_covs, 
                smooth_gains) = kalman_smoother(
                    dataset[i], state_mat_k,
                    self.filt_mean[f"cluster{k}"][i], 
                    self.filt_cov[f"cluster{k}"][i], 
                    self.filt_pred_cov[f"cluster{k}"][i]
                )

                self.smooth_mean[f"cluster{k}"].append(smooth_state_means)
                self.smooth_cov[f"cluster{k}"].append(smooth_state_covs)
                self.smooth_gain[f"cluster{k}"].append(smooth_gains)


    def compute_posterior_prob(self, dataset):
        for i in range(self.N):
            data_i = dataset[i].reshape(self.T, self.obs_dim)
            log_all_clusters_list = []

            for k in range(self.cluster_num):
                obs_mat_k = self.obs_mat[k].reshape(self.obs_dim, self.state_dim)
                obs_cov_k = self.obs_cov[k].reshape(self.obs_dim, self.obs_dim)
                weight_k = self.weights[k]

                pred_state_mean = np.asarray(self.filt_pred_mean[f"cluster{k}"][i])
                pred_state_cov = np.asarray(self.filt_pred_cov[f"cluster{k}"][i])

                loglikelihoods = _loglikelihoods(
                    obs_mat_k, np.zeros(1), obs_cov_k, 
                    pred_state_mean, pred_state_cov, data_i
                )

                log_each_cluster = np.sum(loglikelihoods) + np.log(weight_k + 1e-7)
                log_all_clusters_list.append(log_each_cluster)

            log_all_clusters_list -= np.max(log_all_clusters_list)

            posterior_prob = np.exp(log_all_clusters_list) / np.sum(np.exp(log_all_clusters_list))

            self.posterior_prob[i] = posterior_prob + 1e-7
            
            
    def compute_all_loglikelihoods(self, dataset):
        
        all_loglikelihoods = 0
        
        for i in range(self.N):
            data_i = dataset[i].reshape(self.T, self.obs_dim)

            for k in range(self.cluster_num):
                obs_mat_k = self.obs_mat[k].reshape(self.obs_dim, self.state_dim)
                obs_cov_k = self.obs_cov[k].reshape(self.obs_dim, self.obs_dim)
                weight_k = self.weights[k]

                pred_state_mean = np.asarray(self.filt_pred_mean[f"cluster{k}"][i])
                pred_state_cov = np.asarray(self.filt_pred_cov[f"cluster{k}"][i])

                loglikelihoods = _loglikelihoods(
                    obs_mat_k, np.zeros(1), obs_cov_k, 
                    pred_state_mean, pred_state_cov, data_i
                )

                all_loglikelihoods += np.sum(loglikelihoods) + np.log(weight_k + 1e-7)
                
        return all_loglikelihoods


    def compute_mlgssm_loglikelihoods(self, dataset):
        loglikelihood = _compute_mlgssm_likelihoods(
            dataset, self.cluster_num, self.state_dim, self.obs_dim,
            self.obs_mat, self.obs_cov, self.weights, 
            self.filt_pred_mean, self.filt_pred_cov
        )

        return loglikelihood


    def e_step(self, dataset):
        self.kalman_filter(dataset)
        self.kalman_smoother(dataset)
        self.compute_posterior_prob(dataset)


    def m_step(self, dataset):

        new_state_mat_list = []
        new_state_cov_list =[]
        new_obs_mat_list = []
        new_obs_cov_list = []
        new_init_state_mean_list = []
        new_init_state_cov_list = []
        new_weight_list = []

        param_diff = 0

        for cluster_num in range(self.cluster_num):

            new_state_mat_k = _update_state_mat_k(
                self.smooth_mean, self.smooth_cov, 
                self.smooth_gain, self.posterior_prob, 
                cluster_num, self.N, self.T, self.state_dim
            ).reshape(self.state_dim, self.state_dim)

            new_state_cov_k = _update_state_cov_k(
                new_state_mat_k, self.smooth_mean, 
                self.smooth_cov, self.smooth_gain, 
                self.posterior_prob, cluster_num, 
                self.N, self.T, self.state_dim
            ).reshape(self.state_dim, self.state_dim)

            if self.obs_mat_fix == False:
                new_obs_mat_k = _update_obs_mat_k(
                    dataset, self.smooth_mean, self.smooth_cov, self.smooth_gain,
                    self.posterior_prob, cluster_num, self.N, self.T, self.obs_dim, self.state_dim
                ).reshape(self.obs_dim, self.state_dim)
            else:
                new_obs_mat_k = self.obs_mat[cluster_num].reshape(self.obs_dim, self.state_dim)

            new_obs_cov_k = _update_obs_cov_k(
                dataset, new_obs_mat_k, self.smooth_mean, self.smooth_cov, 
                self.smooth_gain, self.posterior_prob, cluster_num, 
                self.N, self.T, self.obs_dim, self.state_dim
            ).reshape(self.obs_dim, self.obs_dim)

            new_init_state_mean_k = _update_init_state_mean(
                self.smooth_mean, self.posterior_prob, cluster_num, 
                self.N, self.obs_dim, self.state_dim
            ).reshape(self.obs_dim, self.state_dim)

            new_init_state_cov_k = _update_init_state_cov(
                new_init_state_mean_k, self.smooth_mean, 
                self.smooth_cov, self.posterior_prob, 
                cluster_num, self.N, self.obs_dim, self.state_dim
            ).reshape(self.state_dim, self.state_dim)

            new_weight_k = _update_weight(self.posterior_prob, cluster_num, self.N)

            param_diff += (
                np.sum(np.abs(self.state_mat[cluster_num] - new_state_mat_k)) 
                + np.sum(np.abs(self.state_cov[cluster_num] - new_state_cov_k))
                + np.sum(np.abs(self.obs_mat[cluster_num] - new_obs_mat_k))
                + np.sum(np.abs(self.obs_cov[cluster_num] - new_obs_cov_k))
                + np.sum(np.abs(self.init_state_mean[cluster_num] - new_init_state_mean_k))
                + np.sum(np.abs(self.init_state_cov[cluster_num] - new_init_state_cov_k))
            )

            new_state_mat_list.append(new_state_mat_k)
            new_state_cov_list.append(new_state_cov_k)
            new_obs_mat_list.append(new_obs_mat_k)
            new_obs_cov_list.append(new_obs_cov_k)
            new_init_state_mean_list.append(new_init_state_mean_k)
            new_init_state_cov_list.append(new_init_state_cov_k)
            new_weight_list.append(new_weight_k)
        

        self.param_init(
            state_mat = np.asarray(new_state_mat_list), state_cov = np.asarray(new_state_cov_list), 
            obs_mat = np.asarray(new_obs_mat_list), obs_cov = np.asarray(new_obs_cov_list), 
            init_state_mean = np.asarray(new_init_state_mean_list), 
            init_state_cov = np.asarray(new_init_state_cov_list), 
            weights = new_weight_list, random_param_init = False
        )

        self.param_diff.append(param_diff)
        self.kalman_init()

        return (np.asarray(new_state_mat_list), np.asarray(new_state_cov_list), 
                np.asarray(new_obs_mat_list), np.asarray(new_obs_cov_list), 
                np.asarray(new_init_state_mean_list), np.asarray(new_init_state_cov_list), 
                new_weight_list)


    def training(self, 
        dataset, max_iter, epsilon_param = 0.01, epsilon_ll = -1, log = True
        ):
        
        loglikelihood_list = [0]
        cluster_list = []
        if log:
            print("--- START ---")

        self.e_step(dataset)

        if epsilon_param < 0 and epsilon_ll < 0:
            for i in range(max_iter):
                param = self.m_step(dataset)
                self.e_step(dataset)

                if self.loglikelihood == True:
                    loglikelihood_list.append(self.compute_mlgssm_loglikelihoods(dataset))

                pred = np.argmax(self.posterior_prob, axis=1)
                c = collections.Counter(pred)

                if log:
                    print(f"iter = {i+1} : {c.most_common()}")

                cluster_list.append(pred)

            if log:
                print("--- FINISH ---")

            return (param, loglikelihood_list, np.asarray(cluster_list))
        

        elif epsilon_param < 0 and epsilon_ll > 0:
            for i in range(max_iter):
                param = self.m_step(dataset)
                self.e_step(dataset)

                loglikelihood_list.append(self.compute_mlgssm_loglikelihoods(dataset))

                pred = np.argmax(self.posterior_prob, axis=1)
                c = collections.Counter(pred)

                if log:
                    print(f"iter = {i+1} : {c.most_common()}")
                cluster_list.append(pred)
                
                loglikelihood_diff = loglikelihood_list[-1] - loglikelihood_list[-2]
                if loglikelihood_diff < epsilon_ll:
                    if log:
                        print("--- FINISH ---")
                    return (param, loglikelihood_list, np.asarray(cluster_list))
                    
            if log:
                print("--- FINISH ---")

            return (param, loglikelihood_list, np.asarray(cluster_list))
                

        elif epsilon_param > 0 and epsilon_ll < 0:
            for i in range(max_iter):
                param = self.m_step(dataset)
                self.e_step(dataset)

                if self.loglikelihood== True:
                    loglikelihood_list.append(self.compute_mlgssm_loglikelihoods(dataset))

                pred = np.argmax(self.posterior_prob, axis=1)
                c = collections.Counter(pred)

                if log:
                    print(f"iter = {i+1} : {c.most_common()}")
                cluster_list.append(pred)

                if self.param_diff[-1] < epsilon_param:
                    if log:
                        print("--- FINISH ---")
                    return (param, loglikelihood_list, np.asarray(cluster_list))
            if log:    
                print("--- FINISH ---")
                
            return (param, loglikelihood_list, np.asarray(cluster_list))
                


    def clustering(self):
        pred = np.argmax(self.posterior_prob, axis=1)

        c = collections.Counter(pred)
        print(c.most_common())

        return pred, self.posterior_prob