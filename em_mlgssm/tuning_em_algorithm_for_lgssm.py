import numpy as np

from .kalmanfilter_and_smoother import add_input_to_state, add_input_to_obs
from .em_algorithm_for_lgssm import EMlgssm
from .utils.matrix import random_qr_matrix



class TuningEMlgssm(object):
    """
    Tuning EM algorithm for Linear Gaussian State Space Model.
    """
    
    def __init__(self, dim_x, dim_y, dim_ux=None, dim_uy=None):
        """        
        Arguments
        ---------
        dim_x : int
            Dimension of state variable.
        dim_y : int
            Dimension of observed variable.
        dim_ux : int, default=None
            Dimension of input state variable.
        dim_uy : int, default=None
            Dimension of input observed variable.
        """

        self.d_x=dim_x
        self.d_y=dim_y
        self.d_ux=dim_ux
        self.d_uy=dim_uy

    
    def _init_em_lgssm(self):
        """
        Create 'EMlgssm'-instance.

        Returns
        -------
        lgssm : instance.
            Instance of class 'EMlgssm'.
        """

        lgssm = EMlgssm(
            state_mat=random_qr_matrix(self.d_x), 
            state_cov=np.eye(self.d_x) * 0.05, 
            obs_mat=np.ones((self.d_y, self.d_x)), 
            obs_cov=np.eye(self.d_y) * 0.05,
            init_state_mean=np.zeros((self.d_x, 1)), 
            init_state_cov=np.eye(self.d_x) * 1e+4
        )
        if add_input_to_state(self.d_ux):
            lgssm.B = np.ones((self.d_x, self.d_ux)) * 0.5
        if add_input_to_obs(self.d_uy):
            lgssm.D = np.ones((self.d_y, self.d_uy)) * 0.5

        return lgssm

    
    def fit(self, 
        y, ux=None, uy=None, max_iter=10, epsilon=0.01, fix_param=[], n_lgssm=10, random_state=None):
        """
        Run EM algorithm.

        Arguments
        ---------
        y : np.ndarray(len_y, dim_y, 1)
            Time series.
        ux : np.ndarray(len_y, dim_ux, 1), default=None
            Input time series u_x.
        uy : np.ndarray(len_y, dim_uy, 1), default=None
            Input time series u_y.
        max_iter : int, default=10
            Maximum iteration number.
        epsilon : float, default=0.01
            Threshold for convergence judgment.
        fix_param : list, default=[]
            If you want to fix some parameters of LGSSM, you should
            add the corresponding names(str) to list
                'mu' -> mu,    'P' -> P      
                'A' -> A,  'Gamma' -> Gamma,  'B' -> B
                'C' -> C,  'Sigma' -> Sigma,  'D' -> D
            For example, "fix_param=['C']" means that observation matrix 
            is fixed (not updated).
        n_lgssm : int, default=10
            Tuning times (number of lgssms).
        random_state : int or None, default=None
            Seed.
        
        Returns
        -------
        : dict
            Estimated parameters.
        """

        u_x = ux if add_input_to_state(self.d_ux) else np.empty(len(y))
        u_y = uy if add_input_to_obs(self.d_uy) else np.empty(len(y))
        
        np.random.seed(random_state)

        params_set = []
        ll_set = []
        for j in range(n_lgssm):

            # Set parameters
            lgssm = self._init_em_lgssm()

            params_set.append(
                lgssm.fit(
                    y=y, ux=u_x, uy=u_y, 
                    max_iter=max_iter, 
                    epsilon=epsilon, 
                    fix_param=fix_param
                )
            )

            ll_set.append(lgssm.run_e_step.compute_likelihoods(
                    x_means_p=lgssm.means_p, x_covs_p=lgssm.covs_p, log=True
                )
            )

        return params_set[np.argmax(ll_set)]