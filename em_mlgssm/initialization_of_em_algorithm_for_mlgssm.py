import multiprocessing as mp
import numpy as np
from sklearn.cluster import KMeans

from .kalmanfilter_and_smoother import add_input_to_state, add_input_to_obs
from .tuning_em_algorithm_for_lgssm import TuningEMlgssm



class InitEMmlgssm(TuningEMlgssm):
    """
    Initial parameters of EM algorithm for MLGSSM.
    """
    
    def __init__(self, 
        n_clusters, dim_x, dim_y, 
        dim_ux=None, dim_uy=None, n_cpu=1
        ):
        """        
        Arguments
        ---------
        n_clusters : int
            Number of clusters.
        dim_x : int
            Dimension of state variable.
        dim_y : int
            Dimension of observed variable.
        dim_ux : int, default=None
            Dimension of input state variable.
        dim_uy : int, default=None
            Dimension of input observed variable.
        n_cpu : int, default=1
            Number of CPUs.
        """

        super().__init__(
            dim_x=dim_x, dim_y=dim_y, 
            dim_ux=dim_ux, dim_uy=dim_uy
        )

        self.M = n_clusters
        self.cores = n_cpu


    @property
    def dim_vec(self):
            
        dim = self.d_x + 3*(self.d_x**2) + self.d_y*self.d_x + self.d_y**2
        if add_input_to_state(self.d_ux):
            dim += self.d_x*self.d_ux
        if add_input_to_obs(self.d_uy):
            dim += self.d_y*self.d_uy

        return int(dim)


    @property
    def _window_mu(self):

        left = 0
        right = self.d_x
        window = np.zeros((self.M, self.dim_vec))
        window[:, left:right] += 1.

        return window == 1.


    @property
    def _window_P(self):

        left = self.d_x
        right = self.d_x + self.d_x**2
        window = np.zeros((self.M, self.dim_vec))
        window[:, left:right] += 1.

        return window == 1.


    @property
    def _window_A(self):

        left = self.d_x + self.d_x**2
        right = self.d_x + 2*self.d_x**2
        window = np.zeros((self.M, self.dim_vec))
        window[:, left:right] += 1.

        return window == 1.


    @property
    def _window_Gamma(self):

        left = self.d_x + 2*self.d_x**2
        right = self.d_x + 3*self.d_x**2
        window = np.zeros((self.M, self.dim_vec))
        window[:, left:right] += 1.

        return window == 1.


    @property
    def _window_C(self):

        left = self.d_x + 3*self.d_x**2
        right = self.d_x*(1+self.d_y) + 3*self.d_x**2
        window = np.zeros((self.M, self.dim_vec))
        window[:, left:right] += 1.

        return window == 1.


    @property
    def _window_Sigma(self):

        left = self.d_x*(1+self.d_y) + 3*self.d_x**2
        right = self.d_x*(1+self.d_y) + 3*self.d_x**2 + self.d_y*self.d_y
        window = np.zeros((self.M, self.dim_vec))
        window[:, left:right] += 1.

        return window == 1.


    @property
    def _window_B(self):

        left = self.d_x*(1+self.d_y) + 3*self.d_x**2 + self.d_y*self.d_y
        right = self.d_x*(1+self.d_y+self.d_ux) + 3*self.d_x**2 + self.d_y*self.d_y
        window = np.zeros((self.M, self.dim_vec))
        window[:, left:right] += 1.

        return window == 1.


    @property
    def _window_D(self):

        if add_input_to_state(self.d_ux):
            left = self.d_x*(1+self.d_y+self.d_ux) + 3*self.d_x**2 + self.d_y*self.d_y
            right = self.d_x*(1+self.d_y+self.d_ux) + 3*self.d_x**2 + self.d_y*(self.d_y+self.d_uy)
        else:
            left = self.d_x*(1+self.d_y) + 3*self.d_x**2 + self.d_y*self.d_y
            right = self.d_x*(1+self.d_y) + 3*self.d_x**2 + self.d_y*(self.d_y+self.d_uy)

        window = np.zeros((self.M, self.dim_vec))
        window[:, left:right] += 1.

        return window == 1.


    def _array_to_vec(self, params):
        
        vec = np.asarray([])
        for key in params:
            vec = np.hstack([vec, np.ravel(params[key])])

        return vec


    def _tuning(self, index):

        params_array = self.fit(
            y=self.set_y[index], ux=self.u_x, uy=self.u_y, 
            max_iter=10, epsilon=0.01, fix_param=self.fix, 
            n_lgssm=10, random_state=self.seed
        )

        params_vec = self._array_to_vec(params_array)

        return params_vec


    def _kmeans(self, params_vec):
        
        kmeans = KMeans(
            n_clusters=self.M, init='random', n_init=30, random_state=self.seed
        ).fit(params_vec)

        labels = kmeans.labels_.tolist()
        centers_vec = kmeans.cluster_centers_
        weights = np.asarray([labels.count(k) / len(self.set_y) for k in range(self.M)])
        
        return centers_vec, weights


    def _vec_to_array(self, centers_vec):

        set_params = {}
        set_params['mu'] = centers_vec[self._window_mu].reshape(self.M, self.d_x, 1)
        set_params['P'] = centers_vec[self._window_P].reshape(self.M, self.d_x, self.d_x)
        set_params['A'] = centers_vec[self._window_A].reshape(self.M, self.d_x, self.d_x)
        set_params['Gamma'] = centers_vec[self._window_Gamma].reshape(self.M, self.d_x, self.d_x)
        set_params['C'] = centers_vec[self._window_C].reshape(self.M, self.d_y, self.d_x)
        set_params['Sigma'] = centers_vec[self._window_Sigma].reshape(self.M, self.d_y, self.d_y)

        if add_input_to_state(self.d_ux):
            set_params['B'] = centers_vec[self._window_B].reshape(self.M, self.d_x, self.d_ux)
        if add_input_to_obs(self.d_uy):
            set_params['D'] = centers_vec[self._window_D].reshape(self.M, self.d_y, self.d_uy)

        return set_params
    

    def fit_tuning(self, 
        Y, ux=None, uy=None, max_iter=10, epsilon=0.01, fix_param=[], n_lgssm=10, random_state=None):
        """
        Run EM algorithm.

        Arguments
        ---------
        Y : np.ndarray(n_datas, len_y, dim_y, 1)
            Time series dataset.
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
        Initial_params : dict
            Estimated parameters.
        """
    
        self.set_y = Y
        self.u_x = ux
        self.u_y = uy
        self.fix = fix_param
        self.seed = random_state
        
        results = []
        if self.cores==1:
            for j in range(n_lgssm):
                results.append(self._tuning(j))
        elif self.cores>1:
            # Multiprocessing
            with mp.Pool(self.cores) as pool:
                async_result = pool.map_async(self._tuning, range(n_lgssm))
                results = async_result.get()
        else:
            raise  ValueError('n_cpu must be positive integer.')

        params_vec = np.asarray(results)

        # k-means clustering
        centers_vec, weights = self._kmeans(params_vec)

        # vec to array
        initial_params = self._vec_to_array(centers_vec)

        initial_params['weight'] = weights 

        return initial_params