import numpy as np

def essential_matrix_from_fundamental_matrix(K, F):
    E = np.transpose(K).dot(F.dot(K))
    u, s, vh = np.linalg.svd(E)
    s = np.diag([1, 1, 0])
    E = u.dot(s.dot(vh))
    return E



