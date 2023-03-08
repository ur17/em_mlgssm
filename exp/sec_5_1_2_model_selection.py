import numpy as np
import pickle

from em_mlgssm import EMmlgssm, InitEMmlgssm

import toml
NAME = 'model_selection'
config = toml.load('config.toml')[NAME]



DATA_DIR = config['DATA_DIR']
LABEL_DIR = config['LABEL_DIR']
NUM_CLUSTER = config['NUM_CLUSTER']
DIM_X = config['DIM_X']
DIM_Y = config['DIM_Y']
NUM_DATA = config['NUM_DATA']
LENGTH = config['LENGTH']
NUM_CPU = config['NUM_CPU']
NUM_LGSSM = config['NUM_LGSSM']
MAX_ITER = config['MAX_ITER']
EPSILON = config['EPSILON']
FIX = config['FIX']



if __name__ == "__main__": 

    dataset = np.load(DATA_DIR).reshape(NUM_DATA, LENGTH, DIM_Y, 1)
    label = np.load(LABEL_DIR)

    for k in NUM_CLUSTER:
        for d in DIM_X:

            init_em = InitEMmlgssm(
                n_clusters=k, 
                dim_x=d, 
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
                bic=True
            )

            with open(f"result/bic_{k}_{d}.pickle", mode="wb") as f:
                pickle.dump(result['bic'], f)