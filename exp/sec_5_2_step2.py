import numpy as np
from sklearn.metrics import confusion_matrix
import pickle

from em_mlgssm import EMmlgssm, InitEMmlgssm

import toml
# You only need to change 'NAME'.
# Ex) NAME = 'income', 'temperature', or 'population'
NAME = 'ecg'
config = toml.load('config.toml')[NAME]



DATA_DIR = config['DATA_DIR']
LABEL_DIR = config['LABEL_DIR']
MAX_BIC_NUM_CLUSTER = config['MAX_BIC_NUM_CLUSTER']
MAX_BIC_DIM_X = config['MAX_BIC_DIM_X']
DIM_Y = config['DIM_Y']
NUM_DATA = config['NUM_DATA']
LENGTH = config['LENGTH']
NUM_CPU = config['NUM_CPU']
NUM_LGSSM = config['NUM_LGSSM']
MAX_ITER = config['MAX_ITER']
EPSILON = config['EPSILON']
FIX = config['FIX']
NUM_PERFORM = config['NUM_PERFORM']

def sim(cm):
    score = 0
    for i in range(2):
        sim1 = 2*cm[i][1] / (np.sum(cm[i]) + np.sum(cm[:,1:]))
        sim2 = 2*cm[i][0] / (np.sum(cm[i]) + np.sum(cm[:,:1]))
        if sim1 < sim2:
            score += sim2
        else:
            score += sim1
            
    return score/2



if __name__ == "__main__": 
    
    dataset = np.load(DATA_DIR).reshape(NUM_DATA, LENGTH, DIM_Y, 1)
    label = np.load(LABEL_DIR)

    score = []
    
    for i in range(NUM_PERFORM):

        init_em = InitEMmlgssm(
            n_clusters=MAX_BIC_NUM_CLUSTER, 
            dim_x=MAX_BIC_DIM_X, 
            dim_y=DIM_Y, 
            n_cpu=NUM_CPU
        )

        init_params = init_em.fit_tuning( 
            Y=dataset, 
            fix_param=FIX, 
            n_lgssm=NUM_LGSSM, 
        )

        model = EMmlgssm(
            state_mats=init_params["A"], 
            state_covs=init_params["Gamma"], 
            obs_mats=init_params["C"], 
            obs_covs=init_params["Sigma"], 
            init_state_means=init_params["mu"], 
            init_state_covs=init_params["P"], 
            weights=init_params["weight"]
        )

        result = model.fit(
            Y=dataset, 
            max_iter=MAX_ITER, 
            epsilon=EPSILON, 
            n_cpu=NUM_CPU, 
            fix_param=FIX,
            bic=False
        )

        cm = confusion_matrix(label, result['label'])
        score.append(sim(cm))

        with open(f"result/{NAME}_parameter_{i}.pickle", mode="wb") as f:
            pickle.dump(result['parameter'], f)
        with open(f"result/{NAME}_label_{i}.pickle", mode="wb") as f:
            pickle.dump(result['label'], f)


    print(f'Max score is {np.max(score)}')
    print(f'Mean score is {np.mean(score)}')