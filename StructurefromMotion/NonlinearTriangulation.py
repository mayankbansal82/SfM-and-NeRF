import numpy as np
from utils import *


from scipy import optimize

def nonLinear_triangulation(K, pts1, pts2, x3D, R1, C1, R2, C2):
    P1 = projection_matrix(R1,C1,K)
    P2 = projection_matrix(R2,C2,K)
    x3D_ = []
    for i in range(len(x3D)):
        optimized_params = optimize.least_squares(fun=reprojeciton_loss, x0=x3D[i], method="trf", args=[pts1[i], pts2[i], P1, P2],verbose=False)
        X1 = optimized_params.x
        x3D_.append(X1)  
    return np.array(x3D_)

def reprojeciton_loss(X, pts1, pts2, P1, P2):
    p1_1T, p1_2T, p1_3T = P1
    p1_1T, p1_2T, p1_3T = p1_1T.reshape(1,-1), p1_2T.reshape(1,-1),p1_3T.reshape(1,-1)
    p2_1T, p2_2T, p2_3T = P2
    p2_1T, p2_2T, p2_3T = p2_1T.reshape(1,-1), p2_2T.reshape(1,-1), p2_3T.reshape(1,-1)
    u1,v1 = pts1[0], pts1[1]
    u1_proj = np.divide(p1_1T.dot(X) , p1_3T.dot(X))
    v1_proj =  np.divide(p1_2T.dot(X) , p1_3T.dot(X))
    E1= np.square(v1 - v1_proj) + np.square(u1 - u1_proj)
    u2,v2 = pts2[0], pts2[1]
    u2_proj = np.divide(p2_1T.dot(X) , p2_3T.dot(X))
    v2_proj =  np.divide(p2_2T.dot(X) , p2_3T.dot(X))    
    E2= np.square(v2 - v2_proj) + np.square(u2 - u2_proj)
    error = E1 + E2
    return error.squeeze()