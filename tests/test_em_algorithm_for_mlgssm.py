import unittest

import numpy as np
import numpy.testing as npt

from em_mlgssm import EMmlgssm



class Test_EM_mlgssm(unittest.TestCase):


    def setUp(self):

        self.M = 3
        self.d_x = 2
        self.d_y = 1
        self.d_ux = 1
        self.d_uy = 1
        self.set_y = np.ones((10, 5, self.d_y, 1))*10
        self.u_x = np.asarray([1., 1., 1., 1., 1.]).reshape(5, self.d_ux, 1)
        self.u_y = np.asarray([1., 1., 1., 1., 1.]).reshape(5, self.d_uy, 1)
        self.set_A = np.asarray([np.eye(self.d_x)] * self.M)
        self.set_Gamma = np.asarray([np.eye(self.d_x)] * self.M)
        self.set_C = np.ones((self.M, self.d_y, self.d_x))
        self.set_Sigma = np.asarray([np.eye(self.d_y)] * self.M)
        self.set_mu = np.asarray([np.zeros((self.d_x, 1))] * self.M) 
        self.set_P = np.asarray([np.eye(self.d_x)*1e+4] * self.M)
        self.set_pi = np.ones(self.M) / self.M
        self.set_B = np.ones((self.M, self.d_x, self.d_ux))
        self.set_D = np.ones((self.M, self.d_y, self.d_uy))
        
        self.model = EMmlgssm(
            state_mats=self.set_A, 
            state_covs=self.set_Gamma, 
            obs_mats=self.set_C, 
            obs_covs=self.set_Sigma, 
            init_state_means=self.set_mu, 
            init_state_covs=self.set_P, 
            weights=self.set_pi
        )
        self.model_add_input = EMmlgssm(
            state_mats=self.set_A, 
            state_covs=self.set_Gamma, 
            obs_mats=self.set_C, 
            obs_covs=self.set_Sigma, 
            init_state_means=self.set_mu, 
            init_state_covs=self.set_P, 
            weights=self.set_pi,
            input_state_mats=self.set_B, 
            input_obs_mats=self.set_D
        )

        self.model.set_y = self.set_y
        self.model.u_x = None
        self.model.u_y = None
        self.model.fix = []
        self.model_add_input.set_y = self.set_y
        self.model_add_input.u_x = self.u_x
        self.model_add_input.u_y = self.u_y
        self.model_add_input.fix = []


    def test_em(self):
        self.model.cores = 1

        # Test EM algorithm
        self.model.run_e_step().run_m_step()
        # Test Clustering
        _ = self.model.clustering()
        # Test BIC
        _ = self.model.compute_bic()


    def test_input_em_step(self):
        self.model_add_input.cores = 1

        # Test EM algorithm
        self.model_add_input.run_e_step().run_m_step()
        # Test Clustering
        _ = self.model_add_input.clustering()
        # Test BIC
        _ = self.model_add_input.compute_bic()


    def test_fit(self):
        results = self.model.fit(
            Y=self.set_y, max_iter=2
        )
        self.assertEqual(len(results['parameter']), 7)


    def test_input_fit(self):
        results = self.model_add_input.fit(
            Y=self.set_y, max_iter=2, ux=self.u_x, uy=self.u_y
        )
        self.assertEqual(len(results['parameter']), 9)



if __name__ == '__main__':
    unittest.main()