import numpy as np

from em_mlgssm.lgssm.e_step import _compute_statistics_for_estep
from em_mlgssm.kalman.pinv import pseudo_inverse



def _update_state_mat_k(
    smooth_mean, smooth_cov, smooth_gain, 
    posterior_prob, cluster_num, data_num, time_len, state_dim
    ):

    left_mat = np.zeros((state_dim, state_dim))
    right_mat = np.zeros((state_dim, state_dim))

    for i in range(data_num):
        _, e_znzn, e_znzn_1 = _compute_statistics_for_estep(
            smooth_mean[f"cluster{cluster_num}"][i], 
            smooth_cov[f"cluster{cluster_num}"][i], 
            smooth_gain[f"cluster{cluster_num}"][i]
        )
        posterior_prob_k_for_i = posterior_prob[i][cluster_num]

        left_mat += posterior_prob_k_for_i * np.sum(e_znzn_1, axis=0)
        right_mat += posterior_prob_k_for_i * np.sum(e_znzn[:-1], axis=0)

    return np.dot(left_mat, pseudo_inverse(right_mat))


def _update_state_cov_k(
    state_mat, smooth_mean, smooth_cov, smooth_gain, 
    posterior_prob, cluster_num, data_num, time_len, state_dim
    ):

    new_state_cov = np.zeros((state_dim, state_dim))

    for i in range(data_num):
        _, e_znzn, e_znzn_1 = _compute_statistics_for_estep(
            smooth_mean[f"cluster{cluster_num}"][i], 
            smooth_cov[f"cluster{cluster_num}"][i], 
            smooth_gain[f"cluster{cluster_num}"][i]
        )
        posterior_prob_k_for_i = posterior_prob[i][cluster_num]
        
        new_state_cov += (posterior_prob_k_for_i * np.sum(
            e_znzn[1:]
            - np.einsum("ij,nkj->nik", state_mat, e_znzn_1)
            - np.einsum("nij,kj->nik", e_znzn_1, state_mat)
            + np.einsum("ij,njk,lk->nil", state_mat, e_znzn[:-1], state_mat)
            , axis=0
            )
        )

    return new_state_cov / (np.sum(posterior_prob[:,cluster_num]) * (time_len - 1))


def _update_obs_mat_k(
    dataset, smooth_mean, smooth_cov, smooth_gain,
    posterior_prob, cluster_num, data_num, time_len, obs_dim, state_dim
    ):

    dataset = dataset.reshape(data_num, time_len, obs_dim)

    left_mat = np.zeros((obs_dim, state_dim))
    right_mat = np.zeros((state_dim, state_dim))

    for i in range(data_num):
        e_zn, e_znzn, _ = _compute_statistics_for_estep(
            smooth_mean[f"cluster{cluster_num}"][i], 
            smooth_cov[f"cluster{cluster_num}"][i], 
            smooth_gain[f"cluster{cluster_num}"][i]
        )
        posterior_prob_k_for_i = posterior_prob[i][cluster_num]

        left_mat += (posterior_prob_k_for_i * np.sum(
            np.einsum("ni,nj->nij", e_zn.reshape(time_len, state_dim), dataset[i]), axis=0)
        ).reshape(obs_dim, state_dim)

        right_mat += posterior_prob_k_for_i * np.sum(e_znzn, axis=0)

    return np.dot(left_mat, pseudo_inverse(right_mat))


def _update_obs_cov_k(
    dataset, obs_mat, smooth_mean, smooth_cov, smooth_gain,
    posterior_prob, cluster_num, data_num, time_len, obs_dim, state_dim
    ):

    dataset = dataset.reshape(data_num, time_len, obs_dim)
    obs_mat = obs_mat.reshape(obs_dim, state_dim)
    
    new_obs_cov = np.zeros((obs_dim, obs_dim))

    for i in range(data_num):
        e_zn, e_znzn, _ = _compute_statistics_for_estep(
            smooth_mean[f"cluster{cluster_num}"][i], 
            smooth_cov[f"cluster{cluster_num}"][i], 
            smooth_gain[f"cluster{cluster_num}"][i]
        )
        posterior_prob_k_for_i = posterior_prob[i][cluster_num]

        new_obs_cov += (posterior_prob_k_for_i * np.sum(
            np.einsum("ni,nj->nij", dataset[i], dataset[i])
            - np.einsum("ij,nj,nk->nik", obs_mat, e_zn.reshape(time_len, state_dim), dataset[i])
            - np.einsum("ni,nj,kj->nik", dataset[i], e_zn.reshape(time_len, state_dim), obs_mat)
            + np.einsum("ij,njk,lk->nil", obs_mat, e_znzn, obs_mat)
            , axis = 0  
            )
        )

    return new_obs_cov / (np.sum(posterior_prob[:,cluster_num]) * time_len)


def _update_init_state_mean(
    smooth_mean, posterior_prob, cluster_num, 
    data_num, obs_dim, state_dim
    ):

    new_init_state_mean = np.zeros((state_dim, obs_dim))

    time = 0
    for i in range(data_num):
        e_z1 = np.asarray(smooth_mean[f"cluster{cluster_num}"][i][time]).reshape(state_dim, obs_dim)
        posterior_prob_k_for_i = posterior_prob[i][cluster_num]

        new_init_state_mean += posterior_prob_k_for_i * e_z1

    return new_init_state_mean / np.sum(posterior_prob[:,cluster_num])


def _update_init_state_cov(
    init_state_mean, smooth_mean, smooth_cov, posterior_prob, 
    cluster_num, data_num, obs_dim, state_dim
    ):

    new_init_state_cov = np.zeros((state_dim, state_dim))

    time = 0
    for i in range(data_num):

        e_z1_i = np.asarray(
            smooth_mean[f"cluster{cluster_num}"][i][time]
        ).reshape(state_dim, obs_dim)

        e_z1z1_i = (
            np.asarray(smooth_cov[f"cluster{cluster_num}"][i][time]).reshape(state_dim, state_dim)
            + np.outer(e_z1_i, e_z1_i)
        )

        posterior_prob_k_for_i = posterior_prob[i][cluster_num]

        new_init_state_cov += (posterior_prob_k_for_i * (
            e_z1z1_i 
            - np.outer(init_state_mean, e_z1_i)
            - np.outer(init_state_mean, e_z1_i).T
            + np.outer(init_state_mean, init_state_mean)
        ))

    return new_init_state_cov / np.sum(posterior_prob[:,cluster_num])


def _update_weight(posterior_prob, cluster_num, data_num):

    posterior_prob_k = 0.0
    for i in range(data_num):

        posterior_prob_k += posterior_prob[i][cluster_num]

    return posterior_prob_k / data_num