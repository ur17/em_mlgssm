import multiprocessing as mp
import numpy as np

from .kalmanfilter_and_smoother import add_input_to_state, add_input_to_obs
from .em_algorithm_for_lgssm import EMlgssm
from .utils.pinv import pseudo_inverse



class EMmlgssm(object):
    """
    EM algorithm for Mixtures of Linear Gaussian State Space Models (MLGSSM).

    Model:
        x_i[t] = A^(k)*x_i[t-1] (+ B^(k)*u_x[t]) + w_i[t]
        y_i[t] = C^(k)*x_i[t] (+ D^(k)*u_y[t]) + v_i[t]  
        x_i[1] = mu^(k) + u_i
        where
        w_i[t] ~ N(0, Gamma^(k))
        v_i[t] ~ N(0, Sigma^(k))
           u_i ~ N(0, P^(k))

    Parameters: 
        A^(k), Gamma^(k), C^(k), Sigma^(k), mu^(k), P^(k)(, B^(k), D^(k))
    """
    
    def __init__(self,
        state_mats, 
        state_covs, 
        obs_mats, 
        obs_covs, 
        init_state_means, 
        init_state_covs, 
        weights,
        input_state_mats=None, 
        input_obs_mats=None
        ):
        """
        Set parameters.

        Arguments
        ---------
        state_mats : np.ndarray(n_clusters, dim_x, dim_x)
            Set of A.
        state_covs : np.ndarray(n_clusters, dim_x, dim_x)
            Set of Gamma.
        obs_mats : np.ndarray(n_clusters, dim_y, dim_x)
            Set of C.
        obs_covs : np.ndarray(n_clusters, dim_y, dim_y)
            Set of Sigma.
        init_state_means : np.ndarray(n_clusters, dim_x, 1)
            Set of mu.
        init_state_covs : np.ndarray(n_clusters, dim_x, dim_x)
            Set of P.
        weights : np.ndarray(n_clusters, 1, 1)
            Set of pi.
        input_state_mats : np.ndarray(n_clusters, dim_x, dim_ux), default=None 
            Set of B.
        input_obs_mats : np.ndarray(n_clusters, dim_y, dim_uy), default=None
            Set of D.
        """

        self.set_A = state_mats
        self.set_Gamma = state_covs
        self.set_C = obs_mats
        self.set_Sigma = obs_covs
        self.set_mu = init_state_means
        self.set_P = init_state_covs
        self.set_pi = weights
        self.set_B = input_state_mats
        self.set_D = input_obs_mats

        self.M, self.d_y, self.d_x = self.set_C.shape

        if add_input_to_state(self.set_B):
            self.d_ux = self.set_B.shape[2]
        if add_input_to_obs(self.set_D):
            self.d_uy = self.set_D.shape[2]

        
    def _init_em_lgssm(self, data_num, cluster_num):
        """
        Create 'EMlgssm'-instance.

        Arguments
        ---------
        data_num : int
            Data number.

        cluster_num : int
            Cluster number.

        Returns
        -------
        lgssm : instance.
            Instance of class 'EMlgssm'.
        """

        lgssm = EMlgssm(
            state_mat=self.set_A[cluster_num], 
            state_cov=self.set_Gamma[cluster_num], 
            obs_mat=self.set_C[cluster_num], 
            obs_cov=self.set_Sigma[cluster_num],
            init_state_mean=self.set_mu[cluster_num], 
            init_state_cov=self.set_P[cluster_num]
        )

        lgssm.y = self.set_y[data_num]
        lgssm.u_x = self.u_x
        lgssm.u_y = self.u_y

        if add_input_to_state(self.set_B):
            lgssm.B = self.set_B[cluster_num]
        if add_input_to_obs(self.set_D):
            lgssm.D = self.set_D[cluster_num]

        return lgssm
    
    
    def _e_step(self, index):
        """
        Base of E-step.

        Arguments
        ---------
        index : tuple
            (data_num, cluster_num).

        Returns
        -------
        Results of E-step for specific numbers i and k.
        """
        
        (i, k) = index
        
        # Create 'EMlgssm'-instance
        lgssm = self._init_em_lgssm(data_num=i, cluster_num=k)

        # Run E-step of EM algorithm for LGSSM
        lgssm.run_e_step()
        e_xt = lgssm.e_xt
        e_xtxt = lgssm.e_xtxt
        e_xtxt_1 = lgssm.e_xtxt_1
        
        # Sum of lgssm-log-likelihood and log-weight
        loglikelihood = (
            lgssm.compute_likelihoods(
                x_means_p=lgssm.means_p, x_covs_p=lgssm.covs_p, log=True
            ) 
            + 
            np.log(self.set_pi[k] + 1e-7)
        )
        
        # Product of lgssm-likelihood and weight
        likelihood = (
            lgssm.compute_likelihoods(
                x_means_p=lgssm.means_p, x_covs_p=lgssm.covs_p, log=False
            ) 
            * 
            self.set_pi[k]
        )
        
        return (i, k, (e_xt, e_xtxt, e_xtxt_1), loglikelihood, likelihood)
    
    
    def _compute_posterior_prob(self, ll):
        """
        Compute posterior probabilities.

        Arguments
        ---------
        ll : np.ndarray(n_datas, n_clusters, 1, 1) 
            Log-likelihoods.

        Returns
        -------
        pp : np.ndarray(n_datas, n_clusters, 1, 1)
            Posterior probabilities.
        """

        ll[:,:,0] -= np.max(ll, axis=1)
        pp = np.einsum("nmij,njl->nmil", np.exp(ll), 1 / np.sum(np.exp(ll), axis=1))
        pp += 1e-7
        
        return pp


    def _update_mu(self, k):
        """
        Update mu^(k).

        Arguments
        ---------
        k : int
            Cluster number.
        
        Returns
        -------
        : np.ndarray(dim_x, 1)
        """

        mean = np.sum(
            np.einsum("nil,ndk->ndk", self.p_prob[:,k], self.set_e_xt[:,k,0])
            , axis=0
        ) 

        return mean / np.sum(self.p_prob[:,k], axis=0)


    def _update_P(self, k):
        """
        Update P^(k).

        Arguments
        ---------
        k : int
            Cluster number.
        
        Returns
        -------
        : np.ndarray(dim_x, dim_x)
        """

        cov = (
            self.set_e_xtxt[:,k,0]
            - np.einsum("il,nkl->nik", self.set_mu[k], self.set_e_xt[:,k,0])
            - np.einsum("nkl,il->nki", self.set_e_xt[:,k,0], self.set_mu[k])
            + np.dot(self.set_mu[k], self.set_mu[k].T)
        )
        cov = np.sum(
            np.einsum("nil,nkj->nkj", self.p_prob[:,k], cov)
            , axis=0
        )

        return cov / np.sum(self.p_prob[:,k], axis=0)


    def _update_A(self, k):
        """
        Update A^(k).

        Arguments
        ---------
        k : int
            Cluster number.
        
        Returns
        -------
        : np.ndarray(dim_x, dim_x)
        """
        
        left_mat = np.sum(
            np.einsum("nil,ntdk->ntdk", self.p_prob[:,k], self.set_e_xtxt_1[:,k])
            , axis=(0,1)
        )
        right_mat = np.sum(
            np.einsum("nil,ntdk->ntdk", self.p_prob[:,k], self.set_e_xtxt[:,k,:-1])
            , axis=(0,1)
        )

        if add_input_to_state(self.set_B):
            left_mat -= np.sum(
                np.einsum("nil,ntdk->ntdk", 
                    self.p_prob[:,k], 
                    np.einsum("ld,tdj,ntij->ntli", 
                        self.set_B[k], self.u_x[1:], self.set_e_xt[:,k,:-1]
                    )
                )
                , axis=(0,1)
            )

        return np.dot(left_mat, pseudo_inverse(right_mat))


    def _update_Gamma(self, k):
        """
        Update Gamma^(k).

        Arguments
        ---------
        k : int
            Cluster number.
        
        Returns
        -------
        : np.ndarray(dim_x, dim_x)
        """

        cov = (
            self.set_e_xtxt[:,k,1:]
            - np.einsum("il,ntkl->ntik", 
                self.set_A[k], self.set_e_xtxt_1[:,k]
            )
            - np.einsum("ntkl,il->ntki", 
                self.set_e_xtxt_1[:,k], self.set_A[k]
            )
            + np.einsum("ik,ntkl,jl->ntij", 
                self.set_A[k], self.set_e_xtxt[:,k,:-1], self.set_A[k]
            )
        )
        cov = np.sum(
            np.einsum("nil,ntdk->ntdk", self.p_prob[:,k], cov)
            , axis=(0,1)
        )

        if add_input_to_state(self.set_B):
            _cov = (
                - np.einsum("il,tlj,ntkj->ntik", 
                    self.set_B[k], self.u_x[1:], self.set_e_xt[:,k,1:]
                )
                - np.einsum("ntkj,tlj,il->ntki", 
                    self.set_e_xt[:,k,1:], self.u_x[1:], self.set_B[k]
                )
                + np.einsum("il,tlj,ntkj,mk->ntim",
                    self.set_B[k], self.u_x[1:], self.set_e_xt[:,k,:-1], self.set_A[k]
                )
                + np.einsum("mk,ntkj,tlj,il->ntmi", 
                    self.set_A[k], self.set_e_xt[:,k,:-1], self.u_x[1:], self.set_B[k]
                )
            )
            cov += np.sum(
                np.einsum("nil,ntdk->ntdk", self.p_prob[:,k], _cov)
                + 
                np.einsum("njl,tmi->ntmi", 
                    self.p_prob[:,k], 
                    np.einsum("mk,tkj,tlj,il->tmi", 
                        self.set_B[k], self.u_x[1:], self.u_x[1:], self.set_B[k]
                    )
                )
                , axis=(0,1)
            )

        return cov / (np.sum(self.p_prob[:,k]) * (self.T-1))


    def _update_C(self, k):
        """
        Update C^(k).

        Arguments
        ---------
        k : int
            Cluster number.
        
        Returns
        -------
        : np.ndarray(dim_y, dim_x)
        """

        left_mat = np.sum(
            np.einsum("nil,ntdk->ntdk", 
                self.p_prob[:,k], 
                np.einsum("ntij,ntdj->ntid", self.set_y, self.set_e_xt[:,k])
            )
            , axis=(0,1)
        )
        right_mat = np.sum(
            np.einsum("nil,ntdk->ntdk", self.p_prob[:,k], self.set_e_xtxt[:,k])
            , axis=(0,1)
        )

        if add_input_to_obs(self.set_D):
            left_mat -= np.sum(
                np.einsum("nil,ntdk->ntdk", 
                    self.p_prob[:,k], 
                    np.einsum("ld,tdj,ntij->ntli", 
                        self.set_D[k], self.u_y, self.set_e_xt[:,k]
                    )
                )
                , axis=(0,1)
            )

        return np.dot(left_mat, pseudo_inverse(right_mat))


    def _update_Sigma(self, k):
        """
        Update Sigma^(k).

        Arguments
        ---------
        k : int
            Cluster number.
        
        Returns
        -------
        : np.ndarray(dim_y, dim_y)
        """

        cov = (
            np.einsum("ntij,ntkj->ntik", 
                self.set_y, self.set_y
            )
            - np.einsum("ij,ntjk,ntlk->ntil", 
                self.set_C[k], self.set_e_xt[:,k], self.set_y
            )
            - np.einsum("ntlk,ntjk,ij->ntli", 
                self.set_y, self.set_e_xt[:,k], self.set_C[k]
            )
            + np.einsum("ij,ntjk,lk->ntil", 
                self.set_C[k], self.set_e_xtxt[:,k], self.set_C[k]
            )
        )
        cov = np.sum(
            np.einsum("nil,ntdk->ntdk", self.p_prob[:,k], cov)
            , axis=(0,1)
        )

        if add_input_to_obs(self.set_D):
            _cov = (
                - np.einsum("ij,tjk,ntlk->ntil", 
                    self.set_D[k], self.u_y, self.set_y
                )
                - np.einsum("ntlk,tjk,ij->ntli", 
                    self.set_y, self.u_y, self.set_D[k]
                )
                + np.einsum("ij,tjm,ntlm,kl->ntik", 
                    self.set_D[k], self.u_y, self.set_e_xt[:,k], self.set_C[k]
                )
                + np.einsum("kl,ntlm,tjm,ij->ntki", 
                    self.set_C[k], self.set_e_xt[:,k], self.u_y, self.set_D[k]
                )
            )
            cov += np.sum(
                np.einsum("nil,ntdk->ntdk", self.p_prob[:,k], _cov)
                + 
                np.einsum("njl,tmi->ntmi", 
                    self.p_prob[:,k], 
                    np.einsum("mk,tkj,tlj,il->tmi", 
                        self.set_D[k], self.u_y, self.u_y, self.set_D[k]
                    )
                )
                , axis=(0,1)
            )

        return cov / (np.sum(self.p_prob[:,k]) * self.T)


    def _update_B(self, k):
        """
        Update B^(k).

        Arguments
        ---------
        k : int
            Cluster number.
        
        Returns
        -------
        : np.ndarray(dim_x, dim_ux)
        """

        left_mat = (
            np.einsum("ntij,tdj->ntid", 
                self.set_e_xt[:,k,1:], self.u_x[1:]
            )
            - 
            np.einsum("li,ntij,tdj->ntld", 
                self.set_A[k], self.set_e_xt[:,k,:-1], self.u_x[1:]
            )
        )
        left_mat = np.sum(
            np.einsum("nil,ntdk->ntdk", self.p_prob[:,k], left_mat)
            , axis=(0,1)
        )

        right_mat = np.sum(
            np.einsum("njl,tid->ntid", 
                self.p_prob[:,k], 
                np.einsum("tij,tdj->tid", self.u_x[1:], self.u_x[1:])
            )
            , axis=(0,1)
        )

        return np.dot(left_mat, pseudo_inverse(right_mat))


    def _update_D(self, k):
        """
        Update D^(k).

        Arguments
        ---------
        k : int
            Cluster number.
        
        Returns
        -------
        : np.ndarray(dim_y, dim_uy)
        """

        left_mat = (
            np.einsum("ntij,tdj->ntid", 
                self.set_y, self.u_y
            )
            - 
            np.einsum("li,ntij,tdj->ntld", 
                self.set_C[k], self.set_e_xt[:,k], self.u_y
            )
        )
        left_mat = np.sum(
            np.einsum("nil,ntdk->ntdk", self.p_prob[:,k], left_mat)
            , axis=(0,1)
        )

        right_mat = np.sum(
            np.einsum("njl,tid->ntid", 
                self.p_prob[:,k], 
                np.einsum("tij,tdj->tid", self.u_y, self.u_y)
            )
            , axis=(0,1)
        )

        return np.dot(left_mat, pseudo_inverse(right_mat))


    def _update_weight(self, k):
        """
        Update pi^(k).

        Arguments
        ---------
        k : int
            Cluster number.
        
        Returns
        -------
        : np.ndarray(1, 1)
        """

        return np.mean(self.p_prob[:,k])


    def run_e_step(self):
        """
        Run E-step.
        """
        
        self.N, self.T = self.set_y.shape[:2]

        index_pairs = [(i, k) for i in range(self.N) for k in range(self.M)]

        results = []
        if self.cores==1:
            for index in index_pairs:
                results.append(self._e_step(index))
        elif self.cores>1:
            # Multiprocessing
            with mp.Pool(self.cores) as pool:
                async_result = pool.map_async(self._e_step, index_pairs)
                results = async_result.get()
        else:
            raise  ValueError('n_cpu must be positive integer.')
            
        results.sort()
        
        # Expectations
        self.set_e_xt = np.asarray(
            [row[2][0] for row in results]
        ).reshape(self.N, self.M, self.T, self.d_x, 1)
        self.set_e_xtxt = np.asarray(
            [row[2][1] for row in results]
        ).reshape(self.N, self.M, self.T, self.d_x, self.d_x)
        self.set_e_xtxt_1 = np.asarray(
            [row[2][2] for row in results]
        ).reshape(self.N, self.M, self.T-1, self.d_x, self.d_x)

        # Posterior probabilities
        self.p_prob = self._compute_posterior_prob(
            ll = np.asarray([row[3] for row in results]).reshape(self.N, self.M, 1, 1)
        )

        # Likelihoods
        self.likelihoods = np.asarray(
            [row[4] for row in results]
        ).reshape(self.N, self.M)

        return self


    def run_m_step(self):
        """
        Run M-step.
        """

        for k in range(self.M):
            
            # Update pi
            self.set_pi[k] = self._update_weight(k)
            
            if not 'mu' in self.fix:
                # Update mu
                self.set_mu[k] = self._update_mu(k)
            if not 'P' in self.fix:
                # Update P
                self.set_P[k] = self._update_P(k)
            if not 'A' in self.fix:
                # Update A
                self.set_A[k] = self._update_A(k)
            if not 'Gamma' in self.fix:
                # Update Gamma
                self.set_Gamma[k] = self._update_Gamma(k)
            if not 'C' in self.fix:
                # Update C
                self.set_C[k] = self._update_C(k)
            if not 'Sigma' in self.fix:
                # Update Sigma
                self.set_Sigma[k] = self._update_Sigma(k)

            if add_input_to_state(self.set_B) and not 'B' in self.fix:
                # Update B
                self.set_B[k] = self._update_B(k)
            if add_input_to_obs(self.set_D) and not 'D' in self.fix:
                # Update D
                self.set_D[k] = self._update_D(k)

        return self


    def clustering(self):
        """
        Run Clustering.
        
        Returns
        -------
        : np.ndarray(n_datas,)
        """

        # Run E-step
        self.run_e_step()

        return np.argmax(self.p_prob, axis=1)


    def compute_bic(self):
        """
        Compute BIC for MLGSSM.
        
        Returns
        ----------
        bic : float
            BIC for MLGSSM.
        """

        # Number of default lgssm parameters
        n_param = 6
        # Number of optional lgssm parameters
        if add_input_to_state(self.set_B):
            n_param += 1
        if add_input_to_obs(self.set_D):
            n_param += 1
        # Number of fixed parameters
        n_param -= len(self.fix)

        # Log-Likelihoods of MLGSSM
        bic = np.sum(np.log(np.sum(self.likelihoods, axis=1)))
        # Penalties
        bic -= .5 * (self.M*n_param + self.M - 1.) * np.log(self.N)

        return bic


    def fit(self, Y, ux=None, uy=None, max_iter=10, epsilon=0.01, n_cpu=1, fix_param=[], bic=False):
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
        n_cpu : int, default=1
            Number of CPUs.
        fix_param : list, default=[]
            If you want to fix some parameters of LGSSM, you should
            add the corresponding names(str) to list
                'mu' -> mu,    'P' -> P      
                'A' -> A,  'Gamma' -> Gamma,  'B' -> B
                'C' -> C,  'Sigma' -> Sigma,  'D' -> D
            For example, "fix_param=['C']" means that observation matrix 
            is fixed (not updated).
        bic : bool, default=False
            If True, compute bic.
        
        Returns
        -------
        results : dict
        """

        self.set_y = Y
        self.u_x = ux
        self.u_y = uy
        self.cores = n_cpu
        self.fix = fix_param

        for i in range(max_iter):
            
            # Keep current parameters
            set_A = self.set_A.copy()
            set_Gamma = self.set_Gamma.copy()
            set_C = self.set_C.copy() 
            set_Sigma = self.set_Sigma.copy()
            set_mu = self.set_mu.copy()
            set_P = self.set_P.copy()
            if add_input_to_state(self.set_B):
                set_B = self.set_B.copy()
            if add_input_to_obs(self.set_D):
                set_D = self.set_D.copy()

            # Run E-step & M-step
            self.run_e_step().run_m_step()

            # Convergence judgment
            _diff = 0
            if add_input_to_state(self.set_B):
                _diff += np.abs((self.set_B - set_B).sum())
            if add_input_to_obs(self.set_D):
                _diff += np.abs((self.set_D - set_D).sum())
            diff = _diff + np.sum(
                np.abs(
                    [
                        (self.set_mu - set_mu).sum(),
                        (self.set_P - set_P).sum(),
                        (self.set_A - set_A).sum(),
                        (self.set_Gamma - set_Gamma).sum(),
                        (self.set_C - set_C).sum(),
                        (self.set_Sigma - set_Sigma).sum()
                    ]
                )
            )
            if diff < epsilon:
                break

        params = {
            'weight':self.set_pi, 
            'mu':self.set_mu, 
            'P':self.set_P,
            'A':self.set_A, 
            'Gamma':self.set_Gamma,
            'C':self.set_C, 
            'Sigma':self.set_Sigma
        }

        if add_input_to_state(self.set_B):
            params['B'] = self.set_B
        if add_input_to_obs(self.set_D):
            params['D'] = self.set_D

        # Clustering
        labels = self.clustering()

        results = {
            'parameter':params,
            'label':labels
        }

        # Compute BIC
        if bic:
            results['bic'] = self.compute_bic()
        
        return results