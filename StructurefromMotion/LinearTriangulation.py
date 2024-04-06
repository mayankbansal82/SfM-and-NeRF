import numpy as np
from utils import *
import cv2
from EstimateFundamentalMatrix import *
from GetInliersRANSAC import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from DisambiguateCameraPose import *

def linear_triangulation(K, C1, R1, C2, R2, x1, x2):
    C1 = np.reshape(C1, (3,1))
    C2 = np.reshape(C2, (3,1))

    I = np.identity(3)
    P1 = np.dot(K, np.dot(R1, np.hstack((I,-C1))))
    P2 = np.dot(K, np.dot(R2, np.hstack((I,-C2))))

    p1T = P1[0,:].reshape(1,4)
    p2T = P1[1,:].reshape(1,4)
    p3T = P1[2,:].reshape(1,4)
    p_1T = P2[0,:].reshape(1,4)
    p_2T = P2[1,:].reshape(1,4)
    p_3T = P2[2,:].reshape(1,4)

    X = []

    for i in range(x1.shape[0]):
        x = x1[i, 0]
        y = x1[i, 1]
        x_ = x2[i, 0]
        y_ = x2[i, 1]

        A = []
        A.append((y * p3T) - p2T)
        A.append(p1T - (x * p3T))
        A.append((y_ * p_3T) - p_2T)
        A.append(p_1T - (x_ * p_3T))
        A = np.array(A).reshape(4,4)

        _,_,vt = np.linalg.svd(A)
        v = vt.T
        x = v[:,-1]

        X.append(x)

    X = np.array(X)

    return np.array(X)