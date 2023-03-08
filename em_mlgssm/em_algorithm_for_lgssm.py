import numpy as np

from .kalmanfilter_and_smoother import KalmanFS
from .kalmanfilter_and_smoother import add_input_to_state, add_input_to_obs
from .utils.pinv import pseudo_inverse



class EMlgssm(KalmanFS):
    """
    EM algorithm for Linear Gaussian State Space Model (LGSSM).
    """

    def _compute_expectations(self, gains_s, x_means_s, x_covs_s):
        """
        Compute expectations.

        Arguments
        ---------
        gains_s : np.ndarray(len_y - 1, dim_x, dim_x)
            Smoother gains_s.
        x_means_s : np.ndarray(len_y, dim_x, 1)
            Smoothed means.
        x_covs_s : np.ndarray(len_y, dim_x, dim_x)
            Smoothed covariances.
        
        Returns
        -------
        e_xt : np.ndarray(len_y, dim_x, 1)
            Expectations {E[x[t]]|t=1,...,T}.
        e_xtxt : np.ndarray(len_y, dim_x, dim_x)
            Expectations {E[x[t]x[t]^T]|t=1,...,T}.
        e_xtxt_1 : np.ndarray(len_y - 1, dim_x, dim_x)
            Expectations {E[x[t]x[t-1]^T]|t=2,...,T}.
        """
        
        e_xt = x_means_s
        e_xtxt = x_covs_s + np.einsum("nij,nlj->nil", e_xt, e_xt)
        e_xtxt_1 = (
            np.einsum("nij,nlj->nil", x_covs_s[1:], gains_s) 
            + 
            np.einsum("nij,nlj->nil", e_xt[1:], e_xt[:-1])
        )

        return (e_xt, e_xtxt, e_xtxt_1)


    def _update_A(self):
        """
        Update A.
        
        Returns
        -------
        : np.ndarray(dim_x, dim_x)
        """

        left_mat = np.sum(self.e_xtxt_1, axis=0)
        right_mat = np.sum(self.e_xtxt[:-1], axis=0)

        if add_input_to_state(self.B):
            left_mat -= np.sum(
                np.einsum("il,nlj,nkj->nik",
                    self.B, self.u_x[1:], self.e_xt[:-1]
                )
                , axis=0
            )

        return np.dot(left_mat, pseudo_inverse(right_mat))


    def _update_Gamma(self):
        """
        Update Gamma.
        
        Returns
        -------
        : np.ndarray(dim_x, dim_x)
        """

        cov = np.sum(
            self.e_xtxt[1:] 
            - np.einsum("ij,nlj->nil", 
                  self.A, self.e_xtxt_1
            ) 
            - np.einsum("nlj,ij->nli", 
                  self.e_xtxt_1, self.A
            )
            + np.einsum("ij,njk,lk->nil", 
                  self.A, self.e_xtxt[:-1], self.A
            )
            , axis=0
        )

        if add_input_to_state(self.B):
            cov += np.sum(
                - np.einsum("il,nlj,nkj->nik", 
                    self.B, self.u_x[1:], self.e_xt[1:]
                )
                - np.einsum("nkj,nlj,il->nki", 
                    self.e_xt[1:], self.u_x[1:], self.B
                )
                + np.einsum("il,nlj,nkj,mk->nim",
                    self.B, self.u_x[1:], self.e_xt[:-1], self.A
                )
                + np.einsum("mk,nkj,nlj,il->nmi", 
                    self.A, self.e_xt[:-1], self.u_x[1:], self.B
                )
                + np.einsum("mk,nkj,nlj,il->nmi", 
                    self.B, self.u_x[1:], self.u_x[1:], self.B
                )
                , axis=0
            )

        return cov / (len(self.y) - 1)


    def _update_C(self):
        """
        Update C.
        
        Returns
        -------
        : np.ndarray(dim_y, dim_x)
        """

        left_mat = np.sum(np.einsum("nij,nkj->nik", self.y, self.e_xt), axis=0)
        right_mat = np.sum(self.e_xtxt, axis=0)

        if add_input_to_obs(self.D):
            left_mat -= np.sum(
                np.einsum("ij,njk,nlk->nil", 
                    self.D, self.u_y, self.e_xt
                )
                , axis=0
            )

        return np.dot(left_mat, pseudo_inverse(right_mat))


    def _update_Sigma(self):
        """
        Update Sigma.
        
        Returns
        -------
        : np.ndarray(dim_y, dim_y)
        """

        cov = np.sum(
            np.einsum("nij,nkj->nik", self.y, self.y)
            - np.einsum("ij,njk,nlk->nil", self.C, self.e_xt, self.y) 
            - np.einsum("nlk,njk,ij->nli", self.y, self.e_xt, self.C) 
            + np.einsum("ij,njk,lk->nil", self.C, self.e_xtxt, self.C)
            , axis=0
        )

        if add_input_to_obs(self.D):
            cov += np.sum( 
                - np.einsum("ij,njk,nlk->nil", 
                    self.D, self.u_y, self.y
                )
                - np.einsum("nlk,njk,ij->nli", 
                    self.y, self.u_y, self.D
                )
                + np.einsum("ij,njm,nlm,kl->nik", 
                    self.D, self.u_y, self.e_xt, self.C
                )
                + np.einsum("kl,nlm,njm,ij->nki", 
                    self.C, self.e_xt, self.u_y, self.D
                )
                + np.einsum("ij,njm,nlm,kl->nik", 
                    self.D, self.u_y, self.u_y, self.D
                )
                , axis=0
            )

        return cov / len(self.y)


    def _update_B(self):
        """
        Update B.
        
        Returns
        -------
        : np.ndarray(dim_x, dim_ux)
        """
        
        left_mat = np.sum(
            np.einsum("njk,nlk->njl", 
                self.e_xt[1:], self.u_x[1:]
            )
            - np.einsum("ij,njk,nlk->nil", 
                self.A, self.e_xt[:-1], self.u_x[1:]
            )
            , axis=0
        )

        right_mat = np.sum(
            np.einsum("nij,nkj->nik", 
                self.u_x[1:], self.u_x[1:]
            )
            , axis=0
        )

        return np.dot(left_mat, pseudo_inverse(right_mat))


    def _update_D(self):
        """
        Update D.
        
        Returns
        -------
        : np.ndarray(dim_y, dim_uy)
        """
        
        left_mat = np.sum(
            np.einsum("njk,nlk->njl", 
                self.y, self.u_y
            )
            - np.einsum("ij,njk,nlk->nil", 
                self.C, self.e_xt, self.u_y
            )
            , axis=0
        )

        right_mat = np.sum(
            np.einsum("nij,nkj->nik", 
                self.u_y, self.u_y
            )
            , axis=0
        )

        return np.dot(left_mat, pseudo_inverse(right_mat))


    def run_e_step(self):
        """
        Run E-step.
        """

        # Run Kalman Filter algorithm
        (_, means_f, covs_f, self.means_p, self.covs_p) = self.run_filter(
            y=self.y, ux=self.u_x, uy=self.u_y
        )
        # Run Smoother algorithm
        (self.gains_s, self.means_s, self.covs_s) = self.run_smoother(
            x_means_f=means_f, x_covs_f=covs_f,
            x_means_p=self.means_p, x_covs_p=self.covs_p
        )
        # Compute expectations
        (self.e_xt, self.e_xtxt, self.e_xtxt_1) = self._compute_expectations(
            gains_s=self.gains_s, x_means_s=self.means_s, x_covs_s=self.covs_s
        )

        return self
        

    def run_m_step(self):
        """
        Run M-step.
        """
        
        if not 'mu' in self.fix:
            # Update mu
            self.mu = self.means_s[0]
        if not 'P' in self.fix:
            # Update P
            self.P = self.covs_s[0]
        if not 'A' in self.fix:
            # Update A
            self.A = self._update_A()
        if not 'Gamma' in self.fix:
            # Update Gamma
            self.Gamma = self._update_Gamma()
        if not 'C' in self.fix:
            # Update C
            self.C = self._update_C()
        if not 'Sigma' in self.fix:
            # Update Sigma
            self.Sigma = self._update_Sigma()

        if add_input_to_state(self.B) and not 'B' in self.fix:
            # Update B
            self.B = self._update_B()
        if add_input_to_obs(self.D) and not 'D' in self.fix:
            # Update D
            self.D = self._update_D()

        return self


    def fit(self, y, ux=None, uy=None, max_iter=10, epsilon=0.01, fix_param=[]):
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
        
        Returns
        -------
        params : dict
            Estimated parameters.
        """

        self.y = y
        self.u_x = ux if add_input_to_state(self.B) else np.empty(len(self.y))
        self.u_y = uy if add_input_to_obs(self.D) else np.empty(len(self.y))
        self.fix = fix_param

        for i in range(max_iter):
            
            # Keep current parameters
            A = self.A.copy()
            Gamma = self.Gamma.copy()
            C = self.C.copy() 
            Sigma = self.Sigma.copy()
            mu = self.mu.copy()
            P = self.P.copy()
            if add_input_to_state(self.B):
                B = self.B.copy()
            if add_input_to_obs(self.D):
                D = self.D.copy()

            # Run E-step & M-step
            self.run_e_step().run_m_step()

            # Convergence judgment
            _diff = 0
            if add_input_to_state(self.B):
                _diff += np.abs((self.B - B).sum())
            if add_input_to_obs(self.D):
                _diff += np.abs((self.D - D).sum())
            diff = _diff + np.sum(
                np.abs(
                    [
                        (self.mu - mu).sum(),
                        (self.P - P).sum(),
                        (self.A - A).sum(),
                        (self.Gamma - Gamma).sum(),
                        (self.C - C).sum(),
                        (self.Sigma - Sigma).sum()
                    ]
                )
            )
            if diff < epsilon:
                break

        params = { 
            'mu':self.mu, 
            'P':self.P,
            'A':self.A, 
            'Gamma':self.Gamma,
            'C':self.C, 
            'Sigma':self.Sigma
        }

        if add_input_to_state(self.B):
            params['B'] = self.B
        if add_input_to_obs(self.D):
            params['D'] = self.D
            
        return params