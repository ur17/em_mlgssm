import numpy as np



def rotation_matrix(theta):    

    array = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])

    return array


def random_matrix(n, theta):
    array = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
    
    out = np.zeros((n,n))
    out[:2,:2] = array
    q = np.linalg.qr(np.random.randn(n,n))[0]
    
    return q.dot(out).dot(q.T)