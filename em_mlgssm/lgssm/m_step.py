import numpy as np
from em_mlgssm.kalman.pinv import pseudo_inverse



def _update_state_mat(e_znzn, e_znzn_1):
    
    n_time, n_state_dim, _ = e_znzn_1.shape

    left_mat = np.zeros((n_state_dim, n_state_dim))
    right_mat = np.zeros((n_state_dim, n_state_dim))
    for t in range(n_time):
        left_mat += e_znzn_1[t]
        right_mat += e_znzn[t]

    return np.dot(left_mat, pseudo_inverse(right_mat))


def _update_state_cov(state_mat, e_znzn, e_znzn_1):

    state_mat = state_mat.reshape(max(state_mat.shape), max(state_mat.shape))
    n_time, n_state_dim, _ = e_znzn_1.shape

    new_state_cov = np.zeros((n_state_dim, n_state_dim))
    for t in range(n_time):
        new_state_cov += e_znzn[t + 1]
        new_state_cov -= np.dot(e_znzn_1[t], state_mat.T)
        new_state_cov -= np.dot(e_znzn_1[t], state_mat.T).T
        new_state_cov += np.dot(state_mat, np.dot(e_znzn[t], state_mat.T))

    return new_state_cov / n_time


def _update_obs_mat(dataset, e_zn, e_znzn):

    if len(e_zn.shape) == 2:
        n_time, n_state_dim = e_zn.shape
        n_obs_dim = 1
    elif len(e_zn.shape) == 3:
        n_time, n_state_dim, n_obs_dim = e_zn.shape

    left_mat = np.zeros((n_obs_dim, n_state_dim))
    right_mat = np.zeros((n_state_dim, n_state_dim))
    for t in range(n_time):
        left_mat += np.dot(dataset[t], e_zn[t].reshape(n_obs_dim, n_state_dim))
        right_mat += e_znzn[t]

    return np.dot(left_mat, pseudo_inverse(right_mat))


def _update_obs_cov(dataset, obs_mat, e_zn, e_znzn):

    obs_mat = obs_mat.reshape(1, max(obs_mat.shape))
    
    if len(e_zn.shape) == 2:
        n_time, n_state_dim = e_zn.shape
        n_obs_dim = 1
    elif len(e_zn.shape) == 3:
        n_time, n_state_dim, n_obs_dim = e_zn.shape

    new_state_cov = np.zeros((n_obs_dim, n_obs_dim))
    for t in range(n_time):
        new_state_cov += np.dot(dataset[t], dataset[t].T)
        new_state_cov -= np.dot(obs_mat, np.dot(e_zn[t], dataset[t].T))
        new_state_cov -= np.dot(obs_mat, np.dot(e_zn[t], dataset[t].T)).T
        new_state_cov += np.dot(obs_mat, np.dot(e_znzn[t], obs_mat.T))

    return new_state_cov / n_time