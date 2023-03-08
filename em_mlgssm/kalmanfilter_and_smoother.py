import numpy as np
from .utils.pinv import pseudo_inverse



def add_input_to_state(param):
    return False if str(param)=="None" else True


def add_input_to_obs(param):
    return False if str(param)=="None" else True



class KalmanFS(object):
    """
    Kalman Filter and Smoother algorithms for Linear Gaussian State Space Model.

    Model:
        x[t] = A*x[t-1] (+ B*u_x[t]) + w[t]
        y[t] = C*x[t] (+ D*u_y[t]) + v[t]  
        x[1] = mu + u
        where
        w[t] ~ N(0, Gamma)
        v[t] ~ N(0, Sigma)
           u ~ N(0, P)

    Parameters: 
        A, Gamma, C, Sigma, mu, P(, B, D)
    """

    def __init__(self, 
        state_mat, 
        state_cov, 
        obs_mat, 
        obs_cov,
        init_state_mean, 
        init_state_cov, 
        input_state_mat=None, 
        input_obs_mat=None
        ):
        """
        Set parameters.
        
        Arguments
        ---------
        state_mat : np.ndarray(dim_x, dim_x)
            A.
        state_cov : np.ndarray(dim_x, dim_x)
            Gamma.
        obs_mat : np.ndarray(dim_y, dim_x)
            C.
        obs_cov : np.ndarray(dim_y, dim_y)
            Sigma.
        init_state_mean : np.ndarray(dim_x, 1)
            mu.
        init_state_cov : np.ndarray(dim_x, dim_x)
            P.
        input_state_mat : np.ndarray(dim_x, dim_ux), default=None 
            B.
        input_obs_mat : np.ndarray(dim_y, dim_uy), default=None
            D.
        """

        self.A = state_mat
        self.Gamma = state_cov
        self.C = obs_mat
        self.Sigma = obs_cov
        self.mu = init_state_mean
        self.P = init_state_cov
        self.B = input_state_mat
        self.D = input_obs_mat

        self.d_y, self.d_x = self.C.shape

        if add_input_to_state(self.B):
            self.d_ux = self.B.shape[1]
        if add_input_to_obs(self.D):
            self.d_uy = self.D.shape[1]


    def _predict_y(self, x_mean, x_cov, uy_t=None):
        """
        Predict y[t].

        Functions
        ---------
        y_mean <- C*x_mean (+ D*u_y[t])
        y_cov <- C*x_cov*C^T + Sigma
        
        Arguments
        ---------
        x_mean : np.ndarray(dim_x, 1)
            One-step-ahead predicted state mean.
        x_cov : np.ndarray(dim_x, dim_x)
            One-step-ahead predicted state covariance.
        uy_t : np.ndarray(dim_uy, 1), default=None
            u_y[t].

        Returns
        -------
        y_mean : np.ndarray(dim_y, 1)
            One-step-ahead predicted observation mean.
        y_cov : np.ndarray(dim_y, dim_y)
            One-step-ahead predicted observation covariance.
        """

        y_mean = np.dot(self.C, x_mean)
        y_cov = np.dot(self.C, np.dot(x_cov, self.C.T)) + self.Sigma

        if add_input_to_obs(self.D):
            y_mean += np.dot(self.D, uy_t)

        return (y_mean, y_cov)


    def _update_x(self, x_mean, x_cov, y_t, uy_t=None):
        """
        Update x[t].

        Functions
        ---------
        gain_f <- x_cov*C^T*y_cov^{-1}
        x_mean <- x_mean + gain_f*(y[t] - y_mean)
        x_cov <- (I - gain_f*C)*x_cov

        Arguments
        ---------
        x_mean : np.ndarray(dim_x, 1)
            One-step-ahead predicted state mean.
        x_cov : np.ndarray(dim_x, dim_x)
            One-step-ahead predicted state covariance.
        y_t : np.ndarray(dim_y, 1)
            y[t].
        uy_t : np.ndarray(dim_uy, 1), default=None
            u_y[t].

        Returns
        -------
        gain_f : np.ndarray(dim_x, dim_y)
            Kalman gain.
        x_mean : np.ndarray(dim_x, 1)
            Filtered mean.
        x_cov : np.ndarray(dim_x, dim_x)
            Filtered covariance.
        """

        (y_mean, y_cov) = self._predict_y(
            x_mean=x_mean, x_cov=x_cov, uy_t=uy_t
        )

        gain_f = np.dot(x_cov, np.dot(self.C.T, pseudo_inverse(y_cov)))
        x_mean = x_mean + np.dot(gain_f, y_t - y_mean)
        x_cov = x_cov - np.dot(gain_f, np.dot(self.C, x_cov))

        return (gain_f, x_mean, x_cov)


    def _predict_x(self, x_mean, x_cov, ux_t=None):
        """
        Predict x[t].
        
        Functions
        ---------
        x_mean <- A*x_mean (+ B*u_x[t])
        x_cov <- A*x_cov*A^T + Gamma

        Arguments
        ---------
        x_mean : np.ndarray(dim_x, 1)
            Filtered mean.
        x_cov : np.ndarray(dim_x, dim_x)
            Filtered covariance.
        ux_t : np.ndarray(dim_ux, 1), default=None
            u_x[t].

        Returns
        -------
        x_mean : np.ndarray(dim_x, 1)
            One-step-ahead predicted mean.
        x_cov : np.ndarray(dim_x, dim_x)
            One-step-ahead predicted covariance.
        """

        x_mean = np.dot(self.A, x_mean)
        x_cov = np.dot(self.A, np.dot(x_cov, self.A.T)) + self.Gamma

        if add_input_to_state(self.B):
            x_mean += np.dot(self.B, ux_t)

        return (x_mean, x_cov)
    
    
    def _online_filter(self, x_mean, x_cov, y_t, ux_t=None, uy_t=None):
        """
        Update and Predict x[t].
        
        Functions
        ---------
        # Predict
        x_mean <- A*x_mean (+ B*u_x[t])
        x_cov <- A*x_cov*A^T + Gamma
        # Update
        gain_f <- x_cov*C^T*y_cov^{-1}
        x_mean <- x_mean + gain_f*(y[t+1] - y_mean)
        x_cov <- (I - gain_f*C)*x_cov

        Arguments
        ---------
        x_mean : np.ndarray(dim_x, 1)
            Filtered mean.
        x_cov : np.ndarray(dim_x, dim_x)
            Filtered covariance.
        y_t : np.ndarray(dim_y, 1)
            y[t].
        ux_t : np.ndarray(dim_ux, 1), default=None
            u_x[t].
        uy_t : np.ndarray(dim_uy, 1), default=None
            u_y[t].

        Returns
        -------
        x_mean : np.ndarray(dim_x, 1)
            One-step-ahead filtered mean.
        x_cov : np.ndarray(dim_x, dim_x)
            One-step-ahead filtered covariance.
        """
        
        # Predict
        (x_mean, x_cov) = self._predict_x(
            x_mean=x_mean, x_cov=x_cov, ux_t=ux_t
        )
        # Update 
        (_, x_mean, x_cov) = self._update_x(
            x_mean=x_mean, x_cov=x_cov, y_t=y_t, uy_t=uy_t
        )
        
        return (x_mean, x_cov)

    
    def _smooth_x(self, x_mean_f, x_cov_f, x_mean_p, x_cov_p, x_mean_s, x_cov_s):
        """
        Smooth x[t].

        Functions
        ---------
        gain_s <- x_cov_f*A^T*x_cov_p^{-1}
        x_mean_s <- x_mean_f + gain_s*(x_mean_s - x_mean_p)
        x_cov_s <- x_cov_f + gain_s(x_cov_s - x_cov_p)gain_s^T

        Arguments
        ---------
        x_mean_f : np.ndarray(dim_x, 1)
            Filtered mean.
        x_cov_f : np.ndarray(dim_x, dim_x)
            Filtered covariance.
        x_mean_p : np.ndarray(dim_x, 1)
            One-step-ahead predicted mean.
        x_cov_p : np.ndarray(dim_x, dim_x)
            One-step-ahead predicted covariance.
        x_mean_s : np.ndarray(dim_x, 1)
            Smoothed mean.
        x_cov_s : np.ndarray(dim_x, dim_x)
            Smoothed covariance.

        Returns
        -------
        gain_s : np.ndarray(dim_x, dim_x)
            Smoother gain.
        x_mean_s : np.ndarray(dim_x, 1)
            One-step-behind smoothed mean.
        x_cov_s : np.ndarray(dim_x, dim_x)
            One-step-behind smoothed covariance.
        """

        gain_s = np.dot(x_cov_f, np.dot(self.A.T, pseudo_inverse(x_cov_p)))
        x_mean_s = x_mean_f + np.dot(gain_s, x_mean_s - x_mean_p)
        x_cov_s = x_cov_f + np.dot(gain_s, np.dot(x_cov_s - x_cov_p, gain_s.T))

        return (gain_s, x_mean_s, x_cov_s)


    def run_filter(self, y, ux=None, uy=None):
        """
        Run Kalman Filter algorithm.

        Arguments
        ---------
        y : np.ndarray(len_y, dim_y, 1)
            Time series.
        ux : np.ndarray(len_y, dim_ux, 1), default=None
            Input time series u_x.
        uy : np.ndarray(len_y, dim_uy, 1), default=None
            Input time series u_y.
        
        Returns
        -------
        gains_f : np.ndarray(len_y, dim_x, dim_y)
            Kalman gains.
        x_means_f : np.ndarray(len_y, dim_x, 1)
            Filtered means.
        x_covs_f : np.ndarray(len_y, dim_x, dim_x)
            Filtered covariances.
        x_means_p : np.ndarray(len_y, dim_x, 1)
            One-step-ahead predicted means.
        x_covs_p : np.ndarray(len_y, dim_x, dim_x)
            One-step-ahead predicted covariances.
        """

        self.y = y
        len_y = len(self.y)
        self.u_x = ux if add_input_to_state(self.B) else np.empty(len_y)
        self.u_y = uy if add_input_to_obs(self.D) else np.empty(len_y)
        
        gains_f = np.empty((len_y, self.d_x, self.d_y))
        x_means_f = np.empty((len_y, self.d_x, 1))
        x_covs_f = np.empty((len_y, self.d_x, self.d_x))
        x_means_p = np.empty((len_y, self.d_x, 1))
        x_covs_p = np.empty((len_y, self.d_x, self.d_x))

        # Initial one-step-ahead predict
        x_means_p[0] = self.mu
        x_covs_p[0] = self.P

        for t in range(len_y-1):
            # Update 
            (gains_f[t], x_means_f[t], x_covs_f[t]) = self._update_x( 
                x_mean=x_means_p[t], x_cov=x_covs_p[t], y_t=y[t], uy_t=self.u_y[t]
            )
            # Predict
            (x_means_p[t+1], x_covs_p[t+1]) = self._predict_x(
                x_mean=x_means_f[t], x_cov=x_covs_f[t], ux_t=self.u_x[t]
            )

        # Update
        (gains_f[-1], x_means_f[-1], x_covs_f[-1]) = self._update_x( 
            x_mean=x_means_p[-1], x_cov=x_covs_p[-1], y_t=y[-1], uy_t=self.u_y[-1]
        )

        return (gains_f, x_means_f, x_covs_f, x_means_p, x_covs_p)


    def run_smoother(self, x_means_f, x_covs_f, x_means_p, x_covs_p):
        """
        Run Smoother algorithm.

        Arguments
        ---------
        x_means_f : np.ndarray(len_y, dim_x, 1)
            Filtered means.
        x_covs_f : np.ndarray(len_y, dim_x, dim_x)
            Filtered covariances.
        x_means_p : np.ndarray(len_y, dim_x, 1)
            One-step-ahead predicted means.
        x_covs_p : np.ndarray(len_y, dim_x, dim_x)
            One-step-ahead predicted covariances.
        
        Returns
        -------
        gains_s : np.ndarray(len_y - 1, dim_x, dim_x)
            Smoother gains_s.
        x_means_s : np.ndarray(len_y, dim_x, 1)
            Smoothed means.
        x_covs_s : np.ndarray(len_y, dim_x, dim_x)
            Smoothed covariances.
        """
        
        len_y = x_means_f.shape[0]
        
        gains_s = np.empty((len_y-1, self.d_x, self.d_x))
        x_means_s = np.empty((len_y, self.d_x, 1))
        x_covs_s = np.empty((len_y, self.d_x, self.d_x))

        x_means_s[-1] = x_means_f[-1]
        x_covs_s[-1] = x_covs_f[-1]

        for t in reversed(range(len_y-1)):
            # Smooth
            (gains_s[t], x_means_s[t], x_covs_s[t]) = self._smooth_x(
                x_mean_f=x_means_f[t], x_cov_f=x_covs_f[t],
                x_mean_p=x_means_p[t+1], x_cov_p=x_covs_p[t+1],
                x_mean_s=x_means_s[t+1], x_cov_s=x_covs_s[t+1]
            )

        return (gains_s, x_means_s, x_covs_s)
    
    
    def compute_likelihoods(self, x_means_p, x_covs_p, log=True):
        """
        Compute (log)likelihoods.

        Arguments
        ---------
        x_means_p : np.ndarray(len_y, dim_x, 1)
            One-step-ahead predicted means.
        x_covs_p : np.ndarray(len_y, dim_x, dim_x)
            One-step-ahead predicted covariances.
        log : bool
            If True, log-likelihoods are obtained.
        
        Returns
        -------
        : np.ndarray(len_y, 1)
        """

        y_means_p = np.einsum("ij,njl->nil", self.C, x_means_p)            
        y_covs_p = np.einsum("ij,njl,kl->nik", self.C, x_covs_p, self.C) + self.Sigma

        if add_input_to_obs(self.D):
            y_means_p += np.einsum("ij,njl->nil", self.D, self.u_y)
            
        quadratic_form = np.einsum("nlm,nli,nij->nmj", 
            self.y - y_means_p, pseudo_inverse(y_covs_p), self.y - y_means_p
        )

        if log:
            # loglikelihood
            log_exp = - .5 * quadratic_form
            log_coef = - .5 * (
                np.log(np.linalg.det(y_covs_p)).reshape(len(self.y), 1, 1)
                + 
                np.log(2 * np.pi) * self.d_y
            )
            return np.sum(log_exp + log_coef, axis=0)
        
        else:
            # likelihood
            exp = np.exp(- .5 * quadratic_form)
            coef = 1 / np.sqrt(
                np.linalg.det(y_covs_p).reshape(len(self.y), 1, 1) 
                * 
                (2 * np.pi)**self.d_y
            )
            return np.sum(exp * coef, axis=0)