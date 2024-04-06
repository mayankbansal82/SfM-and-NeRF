import cv2
import numpy as np
from EstimateFundamentalMatrix import *

def ransac(pts1, pts2, idx, num_iterations, error_threshold):
    inliers_threshold = 0
    inliers_indices = []
    f_inliers = None

    for i in range(num_iterations):
        n_rows = pts1.shape[0]
        rand_indxs = np.random.choice(n_rows, 8)
        x1 = pts1[rand_indxs, :]
        x2 = pts2[rand_indxs, :]
        F = estimate_fundamental_matrix(x1, x2)
        indices = []

        if F is not None:
            for j in range(n_rows):
                x1_j = np.array([pts1[j, 0], pts1[j, 1], 1])
                x2_j = np.array([pts2[j, 0], pts2[j, 1], 1]).T
                error = np.abs(np.dot(x2_j, np.dot(F, x1_j)))
                if error < error_threshold:
                    indices.append(idx[j])

        if len(indices) > inliers_threshold:
            inliers_threshold = len(indices)
            inliers_indices = indices
            f_inliers = F

    return f_inliers, inliers_indices