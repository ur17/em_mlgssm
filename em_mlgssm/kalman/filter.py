import numpy as np
from .pinv import pseudo_inverse


def _state_predict(
    state_mat, state_cov, current_state_mean, current_state_cov, 
    input_state_vec = None, input_state = None
    ):

    state_mat = state_mat.reshape(max(state_mat.shape), max(state_mat.shape))
    state_cov = state_cov.reshape(max(state_cov.shape), max(state_cov.shape))
    current_state_mean = current_state_mean.reshape(max(current_state_mean.shape),)
    current_state_cov = current_state_cov.reshape(max(current_state_cov.shape), max(current_state_cov.shape))

    if str(input_state_vec) == "None" and str(input_state) == "None":
        pred_state_mean = np.dot(state_mat, current_state_mean)

    else:
        input_state = np.asarray(input_state)
        input_state = input_state.reshape(max(input_state.shape),)
        input_state_vec = input_state_vec.reshape(max(state_mat.shape), max(input_state.shape))
        pred_state_mean = (
            np.dot(state_mat, current_state_mean) 
            + np.dot(input_state_vec, input_state)
        )

    pred_state_cov = (
        np.dot(state_mat, np.dot(current_state_cov, state_mat.T)) 
        + state_cov
    )

    return (pred_state_mean, pred_state_cov)


def _obs_predict(
    obs_mat, obs_cov, predicted_state_mean, predicted_state_cov,
    input_obs_vec = None, input_obs = None
    ):

    obs_mat = obs_mat.reshape(1, max(obs_mat.shape))
    obs_cov = obs_cov.reshape(max(obs_cov.shape), max(obs_cov.shape))
    predicted_state_mean = predicted_state_mean.reshape(max(predicted_state_mean.shape),)
    predicted_state_cov = predicted_state_cov.reshape(max(predicted_state_cov.shape), max(predicted_state_cov.shape))

    if str(input_obs_vec) == "None" and str(input_obs) == "None":
        pred_obs_mean = np.dot(obs_mat, predicted_state_mean)

    else:
        input_obs = np.asarray(input_obs)
        input_obs = input_obs.reshape(max(input_obs.shape),)
        input_obs_vec = input_obs_vec.reshape(1, max(input_obs.shape))
        pred_obs_mean = (
            np.dot(obs_mat, predicted_state_mean) 
            + np.dot(input_obs_vec, input_obs)
        )

    pred_obs_cov = (
        np.dot(obs_mat, np.dot(predicted_state_cov, obs_mat.T)) 
        + obs_cov
    )

    return (pred_obs_mean, pred_obs_cov)


def _state_filter(
    obs_mat, obs_cov, predicted_state_mean, predicted_state_cov, obs,
    input_obs_vec = None, input_obs = None
    ):

    obs_mat = obs_mat.reshape(1, max(obs_mat.shape))
    obs_cov = obs_cov.reshape(max(obs_cov.shape), max(obs_cov.shape))
    predicted_state_mean = predicted_state_mean.reshape(max(predicted_state_mean.shape),)
    predicted_state_cov = predicted_state_cov.reshape(max(predicted_state_cov.shape), max(predicted_state_cov.shape))

    (pred_obs_mean, pred_obs_cov) = _obs_predict(
        obs_mat, obs_cov, predicted_state_mean, predicted_state_cov,
        input_obs_vec, input_obs
    )

    kalman_gain = np.dot(
        predicted_state_cov, 
        np.dot(obs_mat.T, pseudo_inverse(pred_obs_cov))
    )
    
    filt_state_mean = (
        predicted_state_mean
        + np.dot(kalman_gain, obs - pred_obs_mean)
    )

    filt_state_cov = (
        predicted_state_cov
        - np.dot(kalman_gain, np.dot(obs_mat, predicted_state_cov))
    )

    return (kalman_gain, filt_state_mean, filt_state_cov, pred_obs_mean, pred_obs_cov)


def kalman_filter(
    time_series, state_mat, state_cov, obs_mat, obs_cov, init_state_mean, init_state_cov
    ):
    
    state_mat = state_mat.reshape(max(state_mat.shape), max(state_mat.shape))
    state_cov = state_cov.reshape(max(state_cov.shape), max(state_cov.shape))
    obs_mat = obs_mat.reshape(1, max(obs_mat.shape))
    obs_cov = obs_cov.reshape(max(obs_cov.shape), max(obs_cov.shape))
    init_state_mean = init_state_mean.reshape(max(init_state_mean.shape),)
    init_state_cov = init_state_cov.reshape(max(init_state_cov.shape), max(init_state_cov.shape))
    
    filt_state_mean_list = []
    filt_state_cov_list = []
    pred_state_mean_list = []
    pred_state_cov_list = []
    pred_obs_mean_list = []
    pred_obs_cov_list = []
    kalman_gain_list = []

    for t in range(len(time_series)):

        if t == 0:
            pred_state_mean, pred_state_cov = init_state_mean, init_state_cov

        else:
            current_state_mean, current_state_cov = filt_state_mean_list[-1], filt_state_cov_list[-1]

            (pred_state_mean, pred_state_cov) = _state_predict(
                state_mat, state_cov, current_state_mean, current_state_cov
            )

        pred_state_mean_list.append(pred_state_mean)
        pred_state_cov_list.append(pred_state_cov)


        (pred_obs_mean, pred_obs_cov) = _obs_predict(
            obs_mat, obs_cov, pred_state_mean, pred_state_cov
        )

        pred_obs_mean_list.append(pred_obs_mean)
        pred_obs_cov_list.append(pred_obs_cov)
        
        (kalman_gain, filt_state_mean, filt_state_cov, 
        _, _) = _state_filter(
            obs_mat, obs_cov, pred_state_mean, pred_state_cov, time_series[t]
        )

        kalman_gain_list.append(kalman_gain)
        filt_state_mean_list.append(filt_state_mean)
        filt_state_cov_list.append(filt_state_cov)


    return (np.asarray(pred_state_mean_list), np.asarray(pred_state_cov_list), np.asarray(kalman_gain_list), 
            np.asarray(filt_state_mean_list), np.asarray(filt_state_cov_list),
            np.asarray(pred_obs_mean_list), np.asarray(pred_obs_cov_list))