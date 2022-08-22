import numpy as np
from .pinv import pseudo_inverse


def _state_smoother(
    state_mat, predict_next_state_cov,
    filtered_state_mean, filtered_state_cov,
    smootherd_next_state_mean, smootherd_next_state_cov
    ):

    state_mat = state_mat.reshape(max(state_mat.shape), max(state_mat.shape))
    predict_next_state_cov = predict_next_state_cov.reshape(max(predict_next_state_cov.shape), max(predict_next_state_cov.shape))
    filtered_state_mean = filtered_state_mean.reshape(max(filtered_state_mean.shape),)
    filtered_state_cov = filtered_state_cov.reshape(max(filtered_state_cov.shape), max(filtered_state_cov.shape))
    smootherd_next_state_mean = smootherd_next_state_mean.reshape(max(smootherd_next_state_mean.shape),)
    smootherd_next_state_cov = smootherd_next_state_cov.reshape(max(smootherd_next_state_cov.shape), max(smootherd_next_state_cov.shape))



    smooth_gain = np.dot(
        filtered_state_cov, 
        np.dot(state_mat.T, pseudo_inverse(predict_next_state_cov))
    )
    
    smooth_state_mean = (
        filtered_state_mean 
        + np.dot(smooth_gain, smootherd_next_state_mean - np.dot(state_mat, filtered_state_mean))
    )
    
    smooth_state_cov = (
        filtered_state_cov 
        + np.dot(smooth_gain, np.dot(smootherd_next_state_cov - predict_next_state_cov, smooth_gain.T))
    )

    return (smooth_gain, smooth_state_mean, smooth_state_cov)


def kalman_smoother(
    time_series, state_mat, filtered_state_mean_list, 
    filtered_state_cov_list, predict_state_cov_list
    ):
    
    state_mat = state_mat.reshape(max(state_mat.shape), max(state_mat.shape))
    
    smooth_state_mean_list = []
    smooth_state_cov_list = []
    smooth_gain_list = []

    smooth_state_mean_list.append(filtered_state_mean_list[-1])
    smooth_state_cov_list.append(filtered_state_cov_list[-1])

    for t in range(1, len(time_series)):
        smooth_next_state_mean = smooth_state_mean_list[0]
        smooth_next_state_cov = smooth_state_cov_list[0]
        
        pred_next_state_cov = predict_state_cov_list[- t]

        filt_state_mean = filtered_state_mean_list[- (t + 1)]
        filt_state_cov = filtered_state_cov_list[- (t + 1)]
        
        (smooth_gain, smooth_state_mean, 
        smooth_state_cov) = _state_smoother(
            state_mat, pred_next_state_cov,
            filt_state_mean, filt_state_cov,
            smooth_next_state_mean, smooth_next_state_cov
        )
        
        smooth_state_mean_list.insert(0, smooth_state_mean)
        smooth_state_cov_list.insert(0, smooth_state_cov)
        smooth_gain_list.insert(0, smooth_gain)

    return (np.asarray(smooth_state_mean_list), np.asarray(smooth_state_cov_list), np.asarray(smooth_gain_list))