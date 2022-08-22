import numpy as np



def lgssm_generator(init_state_mean, init_state_cov, state_mat, state_cov, 
                    obs_mat, obs_cov, time = 100, noise = True):

    init_state_mean = init_state_mean.reshape(max(init_state_mean.shape),)
    init_state_cov = init_state_cov.reshape(max(init_state_cov.shape), max(init_state_cov.shape))
    state_mat = state_mat.reshape(max(state_mat.shape), max(state_mat.shape))
    state_cov = state_cov.reshape(max(state_cov.shape), max(state_cov.shape))
    obs_mat = obs_mat.reshape(1, max(obs_mat.shape))
    obs_cov = obs_cov.reshape(max(obs_cov.shape), max(obs_cov.shape))
    
    state_dim = state_mat.shape[0]
    mean = [0]*state_dim
    
    if noise == True:        
        x_t_1 = (init_state_mean.reshape(1,state_dim) 
                 + np.random.multivariate_normal(mean, init_state_cov)).reshape(1,state_dim)
        x_t = ((state_mat @ x_t_1.T).reshape(1,state_dim) 
               + np.random.multivariate_normal(mean, state_cov, size=1)).reshape(1,state_dim)
        y_t = obs_mat @ x_t.T + np.random.normal(loc=0, scale=obs_cov)
        
    else:
        x_t_1 = init_state_mean.reshape(1,state_dim)
        x_t = (state_mat @ x_t_1.T).reshape(1,state_dim)
        y_t = obs_mat @ x_t.T
    
    state_mat_array = x_t
    output_array = y_t
    x_t_1 = x_t
    
    for t in range(time - 1):
        
        if noise == True:
            x_t = ((state_mat @ x_t_1[0].T).reshape(1,state_dim) 
                   + np.random.multivariate_normal(mean, state_cov, size=1)).reshape(1,state_dim)
            y_t = obs_mat @ x_t.T + np.random.normal(loc=0, scale=obs_cov)
        else:
            x_t = (state_mat @ x_t_1[0].T).reshape(1,state_dim)
            y_t = obs_mat @ x_t.T
        
        state_mat_array = np.vstack((state_mat_array, x_t))
        output_array = np.vstack((output_array, y_t))
        x_t_1 = x_t
    
    return state_mat_array, output_array