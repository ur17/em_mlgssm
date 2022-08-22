import numpy as np
from scipy.special import logsumexp
from pykalman.standard import _loglikelihoods


def _compute_mlgssm_likelihoods(
    dataset, obs_mat, obs_cov, weight, 
    filter_predicted_mean, filter_predicted_cov):

    data_len, _, _ = dataset.shape
    cluster_num, obs_dim, state_dim = obs_mat.shape

    loglikelihood_list = []
    for i in range(data_len):
        data_i = dataset[i]
        loglikelihood_i_list = []

        for k in range(cluster_num):

            obs_mat_k = obs_mat[k].reshape(obs_dim, state_dim)
            obs_cov_k = obs_cov[k].reshape(obs_dim, obs_dim)
            weight_k = weight[k]

            predicted_state_mean = np.asarray(filter_predicted_mean[f"cluster{k}"][i])
            predicted_state_cov = np.asarray(filter_predicted_cov[f"cluster{k}"][i])

            loglikelihoods = _loglikelihoods(
                obs_mat_k, np.zeros(1), obs_cov_k, 
                predicted_state_mean, predicted_state_cov, data_i
            )

            loglikelihood_ik = np.sum(loglikelihoods) + np.log(weight_k + 1e-7)
            loglikelihood_i_list.append(loglikelihood_ik)

        # from scipy.special import logsumexp
        loglikelihood_i = logsumexp(loglikelihood_i_list)
        loglikelihood_list.append(loglikelihood_i)

    return np.sum(loglikelihood_list)