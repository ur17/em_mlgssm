import numpy as np



def random_qr_matrix(n):
    
    return np.linalg.qr(np.random.randn(n,n))[0]