import numpy as np
from em_mlgssm.kalman.pinv import pseudo_inverse



def _update_state_mat(
    e_znzn, e_znzn_1, 
    e_zn = None, input_state_mat = None, input_state_series = None
    ):
    
    time, state_dim, _ = e_znzn_1.shape

    left_mat = np.zeros((state_dim, state_dim))
    right_mat = np.zeros((state_dim, state_dim))

    if str(e_zn) == "None":
        for t in range(time):
            left_mat += e_znzn_1[t]
            right_mat += e_znzn[t]
    else:
        input_state_series = np.asarray(input_state_series).reshape(time + 1,)
        input_state_mat = input_state_mat.reshape(state_dim, 1)
        for t in range(time):
            left_mat += e_znzn_1[t]
            left_mat -= np.dot(
                np.dot(input_state_mat, input_state_series[t].reshape(1, 1)), 
                e_zn[t].reshape(1, state_dim)
            )
            right_mat += e_znzn[t]

    return np.dot(left_mat, pseudo_inverse(right_mat))


def _update_state_cov(
    state_mat, e_znzn, e_znzn_1,
    e_zn = None, input_state_mat = None, input_state_series = None
    ):

    state_mat = state_mat.reshape(max(state_mat.shape), max(state_mat.shape))
    time, state_dim, _ = e_znzn_1.shape

    new_state_cov = np.zeros((state_dim, state_dim))

    if str(e_zn) == "None":
        for t in range(time):
            new_state_cov += e_znzn[t + 1]
            new_state_cov -= np.dot(e_znzn_1[t], state_mat.T)
            new_state_cov -= np.dot(e_znzn_1[t], state_mat.T).T
            new_state_cov += np.dot(state_mat, np.dot(e_znzn[t], state_mat.T))
    
    else:
        input_state_series = np.asarray(input_state_series).reshape(time + 1,)
        input_state_mat = input_state_mat.reshape(state_dim, 1)
        for t in range(time):
            new_state_cov += e_znzn[t + 1]
            new_state_cov -= np.dot(e_znzn_1[t], state_mat.T)
            new_state_cov -= np.dot(e_znzn_1[t], state_mat.T).T
            new_state_cov += np.dot(state_mat, np.dot(e_znzn[t], state_mat.T))
            new_state_cov -= np.dot(
                np.dot(input_state_mat, input_state_series[t].reshape(1, 1)),
                e_zn[t + 1].reshape(1, state_dim)
            )
            new_state_cov -= np.dot(
                np.dot(input_state_mat, input_state_series[t].reshape(1, 1)),
                e_zn[t + 1].reshape(1, state_dim)
            ).T
            new_state_cov += np.dot(
                np.dot(
                    np.dot(input_state_mat, input_state_series[t].reshape(1, 1)),
                    e_zn[t].reshape(1, state_dim)
                ),
                state_mat.T
            )
            new_state_cov += np.dot(
                np.dot(
                    np.dot(input_state_mat, input_state_series[t].reshape(1, 1)),
                    e_zn[t].reshape(1, state_dim)
                ),
                state_mat.T
            ).T
            new_state_cov += np.dot(
                np.dot(input_state_mat, input_state_series[t].reshape(1, 1)),
                np.dot(input_state_mat, input_state_series[t].reshape(1, 1)).T
            )

    return new_state_cov / time


def _update_obs_mat(
    time_series, e_zn, e_znzn, 
    input_obs_mat = None, input_obs_series = None
    ):

    if len(e_zn.shape) == 2:
        time, state_dim = e_zn.shape
        obs_dim = 1
    elif len(e_zn.shape) == 3:
        time, state_dim, obs_dim = e_zn.shape

    left_mat = np.zeros((obs_dim, state_dim))
    right_mat = np.zeros((state_dim, state_dim))

    if str(input_obs_mat) == "None":
        for t in range(time):
            left_mat += np.dot(time_series[t], e_zn[t].reshape(obs_dim, state_dim))
            right_mat += e_znzn[t]

    else:
        input_obs_series = np.asarray(input_obs_series).reshape(time,)
        input_obs_mat = input_obs_mat.reshape(obs_dim, 1)
        for t in range(time):
            left_mat += np.dot(time_series[t], e_zn[t].reshape(obs_dim, state_dim))
            left_mat -= np.dot(
                np.dot(input_obs_mat, input_obs_series[t].reshape(1, 1)),
                e_zn[t].reshape(obs_dim, state_dim)
            )
            right_mat += e_znzn[t]

    return np.dot(left_mat, pseudo_inverse(right_mat))


def _update_obs_cov(
    time_series, obs_mat, e_zn, e_znzn,
    input_obs_mat = None, input_obs_series = None
    ):

    obs_mat = obs_mat.reshape(1, max(obs_mat.shape))
    
    if len(e_zn.shape) == 2:
        time, state_dim = e_zn.shape
        obs_dim = 1
    elif len(e_zn.shape) == 3:
        time, state_dim, obs_dim = e_zn.shape

    new_state_cov = np.zeros((obs_dim, obs_dim))

    if str(input_obs_mat) == "None":
        for t in range(time):
            new_state_cov += np.dot(time_series[t], time_series[t].T)
            new_state_cov -= np.dot(obs_mat, np.dot(e_zn[t], time_series[t].T))
            new_state_cov -= np.dot(obs_mat, np.dot(e_zn[t], time_series[t].T)).T
            new_state_cov += np.dot(obs_mat, np.dot(e_znzn[t], obs_mat.T))

    else:
        input_obs_series = np.asarray(input_obs_series).reshape(time,)
        input_obs_mat = input_obs_mat.reshape(obs_dim, 1)
        for t in range(time):
            new_state_cov += np.dot(time_series[t], time_series[t].T)
            new_state_cov -= np.dot(
                obs_mat, 
                np.dot(e_zn[t].reshape(state_dim, obs_dim), time_series[t].T)
            )
            new_state_cov -= np.dot(
                obs_mat, 
                np.dot(e_zn[t].reshape(state_dim, obs_dim), time_series[t].T)
            ).T
            new_state_cov += np.dot(obs_mat, np.dot(e_znzn[t], obs_mat.T))
            new_state_cov -= np.dot(
                np.dot(input_obs_mat, input_obs_series[t].reshape(1, 1)),
                time_series[t].T
            )
            new_state_cov -= np.dot(
                np.dot(input_obs_mat, input_obs_series[t].reshape(1, 1)),
                time_series[t].T
            ).T
            new_state_cov += np.dot(
                np.dot(input_obs_mat, input_obs_series[t].reshape(1, 1)),
                np.dot(e_zn[t].reshape(obs_dim, state_dim), obs_mat.T)
            )
            new_state_cov += np.dot(
                np.dot(input_obs_mat, input_obs_series[t].reshape(1, 1)),
                np.dot(e_zn[t].reshape(obs_dim, state_dim), obs_mat.T)
            ).T
            new_state_cov += np.dot(
                np.dot(input_obs_mat, input_obs_series[t].reshape(1, 1)),
                np.dot(input_obs_mat, input_obs_series[t].reshape(1, 1)).T
            )

    return new_state_cov / time


def _update_input_state_mat( 
    state_mat, e_zn, input_state_series
    ):
    
    state_mat = state_mat.reshape(max(state_mat.shape), max(state_mat.shape))

    if len(e_zn.shape) == 2:
        time, state_dim = e_zn.shape
    elif len(e_zn.shape) == 3:
        time, state_dim, _ = e_zn.shape

    mat = np.zeros((state_dim, 1))

    input_state_series = np.asarray(input_state_series).reshape(time,)
    for t in range(time - 1):
        mat += e_zn[t + 1].reshape(state_dim, 1)
        mat -= np.dot(state_mat, e_zn[t].reshape(state_dim, 1))

    mat = mat.reshape(state_dim,)

    return mat / np.sum(input_state_series[1:])


def _update_input_obs_mat( 
    time_series, input_obs_series, obs_mat, e_zn
    ):

    if len(e_zn.shape) == 2:
        time, state_dim = e_zn.shape
        obs_dim = 1
    elif len(e_zn.shape) == 3:
        time, state_dim, obs_dim = e_zn.shape

    obs_mat = obs_mat.reshape(1, state_dim)

    mat = np.zeros((obs_dim)).reshape(obs_dim, obs_dim)

    input_obs_series = np.asarray(input_obs_series).reshape(time,)
    for t in range(time):
        mat += time_series[t]
        mat -= np.dot(obs_mat, e_zn[t].reshape(state_dim, 1))

    mat = mat.reshape(obs_dim, obs_dim)

    return mat / np.sum(input_obs_series)