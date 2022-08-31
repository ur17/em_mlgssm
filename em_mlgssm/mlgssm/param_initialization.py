import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

from em_mlgssm.lgssm.em import EM_lgssm


def get_initial_value(dataset, time=1000, cluster_num=3, state_dim=2, num=2, tuning_num=5, seed=0, random_state:int=0):
    dataset = dataset.reshape(dataset.shape[0], dataset.shape[1])
    all_params_list = []
    for data in tqdm(dataset):
        params_list = []
        model = EM_lgssm(time = time, state_dim = state_dim, obs_dim = 1)
        summary = model.param_tuning(
            data, num=num, max_iter=50, seed=seed, tuning_num=tuning_num, epsilon=0.01, log = False
        )
        params = summary[0]
        
        s_mat = params[0].reshape(state_dim**2,).tolist()
        s_cov = params[1].reshape(state_dim**2,).tolist()
        o_mat = params[2].reshape(state_dim,).tolist()
        o_cov = params[3].reshape(1,).tolist()
        init_s_mean = params[4].reshape(state_dim,).tolist()
        init_s_cov = params[5].reshape(state_dim**2,).tolist()
        
        params_list += s_mat + s_cov + o_cov + init_s_mean + init_s_cov
        all_params_list.append(params_list)
        
    kmeans = KMeans(
        n_clusters = cluster_num, random_state = random_state, 
        algorithm = "full", n_init = 30, max_iter = 100
    ).fit(np.asarray(all_params_list))
    
    labels = kmeans.labels_.tolist()
    weight = [labels.count(i) / len(dataset) for i in range(cluster_num)]
    
    new_state_mat, new_state_cov, new_obs_cov = [], [], []
    new_init_state_mean, new_init_state_cov = [], []

    obs_mat = np.asarray([1]*(cluster_num*state_dim)).reshape(cluster_num,state_dim,1)

    for i in range(cluster_num):
        state_mat = kmeans.cluster_centers_[i][:state_dim**2].reshape(state_dim,state_dim)
        state_cov = kmeans.cluster_centers_[i][state_dim**2:2*state_dim**2].reshape(state_dim,state_dim)
        obs_cov = kmeans.cluster_centers_[i][2*state_dim**2:2*state_dim**2+1].reshape(1,1)
        init_state_mean = kmeans.cluster_centers_[i][2*state_dim**2+1:2*state_dim**2+1+state_dim].reshape(state_dim,)
        init_state_cov = kmeans.cluster_centers_[i][2*state_dim**2+1+state_dim:3*state_dim**2+1+state_dim].reshape(state_dim,state_dim)
        
        new_state_mat.append(state_mat)
        new_state_cov.append(state_cov)
        new_obs_cov.append(obs_cov)
        new_init_state_mean.append(init_state_mean)
        new_init_state_cov.append(init_state_cov)
        
    return (np.asarray(new_state_mat), np.asarray(new_state_cov), obs_mat, np.asarray(new_obs_cov),
            np.asarray(new_init_state_mean), np.asarray(new_init_state_cov), weight)