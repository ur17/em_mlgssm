import numpy as np


def _compute_statistics_for_estep(smoother_state_mean_list, smoother_state_cov_list, smoother_gain_list):
    
    smoother_state_cov = np.asarray(smoother_state_cov_list)
    smoother_state_gain = np.asarray(smoother_gain_list)

    n_time, n_state_dim, _ = smoother_state_cov.shape

    e_zn = np.asarray(smoother_state_mean_list).reshape(n_time, n_state_dim, 1)

    e_znzn_list = []
    e_znzn_1_list = []
    for t in range(n_time):
        e_znzn = smoother_state_cov[t] + np.outer(e_zn[t], e_zn[t])
        e_znzn_list.append(e_znzn)
        if t < (n_time - 1):
            e_znzn_1 = (
                np.dot(smoother_state_cov[t+1], smoother_state_gain[t].T) 
                + np.outer(e_zn[t+1], e_zn[t])
            )
            e_znzn_1_list.append(e_znzn_1)

    e_zn = e_zn.reshape(n_time, n_state_dim,)

    return e_zn, np.asarray(e_znzn_list), np.asarray(e_znzn_1_list)