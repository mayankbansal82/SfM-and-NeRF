import numpy as np
import time
import cv2
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from BuildVisibilityMatrix import *
from utils import *

def bundle_adjustment_sparsity(X_found, filtered_feature_flag, num_cameras):
    num_camera_params = num_cameras + 1
    X_index, visibility_matrix = find_visible_cam(X_found.reshape(-1), filtered_feature_flag, num_cameras)
    num_observations = np.sum(visibility_matrix)
    num_points = len(X_index[0])

    num_rows = num_observations * 2
    num_cols = num_camera_params * 6 + num_points * 3
    A = lil_matrix((num_rows, num_cols), dtype=int)
   
    obs_indices = np.arange(num_observations)
    camera_indices, point_indices = get_camera_point_indices(visibility_matrix)

    for i in range(6):
        A[2 * obs_indices, camera_indices * 6 + i] = 1
        A[2 * obs_indices + 1, camera_indices * 6 + i] = 1

    for i in range(3):
        A[2 * obs_indices, (num_cameras) * 6 + point_indices * 3 + i] = 1
        A[2 * obs_indices + 1, (num_cameras) * 6 + point_indices * 3 + i] = 1

    return A

def bundle_adjustment(X_index, visibility_matrix, X_all, X_found, feature_x, feature_y, filtered_feature_flag, R_set, C_set, K, n_cameras):
    points_3d = X_all[X_index]
    points_2d = get_2d_points(X_index, visibility_matrix, feature_x, feature_y)

    camera_params = []
    for i in range(n_cameras + 1):
        C, R = C_set[i], R_set[i]
        euler_angles = get_euler(R)
        camera_params_ = [euler_angles[0], euler_angles[1], euler_angles[2], C[0], C[1], C[2]]
        camera_params.append(camera_params_)
    camera_params = np.array(camera_params, dtype=object).reshape(-1, 6)

    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    n_points = points_3d.shape[0]

    camera_indices, point_indices = get_camera_point_indices(visibility_matrix)

    sparsity_matrix = bundle_adjustment_sparsity(X_found, filtered_feature_flag, n_cameras)

    t0 = time.time()
    result = least_squares(
        compute_residuals,
        x0,
        jac_sparsity=sparsity_matrix,
        verbose=0,
        x_scale='jac',
        ftol=1e-10,
        method='trf',
        args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K),
    )
    t1 = time.time()

    x1 = result.x
    n_cams = n_cameras + 1
    optim_cam_params = x1[:n_cams * 6].reshape((n_cams, 6))
    optim_points_3d = x1[n_cams * 6:].reshape((n_points, 3))

    optim_X_all = np.zeros_like(X_all)
    optim_X_all[X_index] = optim_points_3d

    optim_C_set, optim_R_set = [], []
    for i in range(len(optim_cam_params)):
        R = get_rotation(optim_cam_params[i, :3], 'e')
        C = optim_cam_params[i, 3:].reshape(3, 1)
        optim_C_set.append(C)
        optim_R_set.append(R)

    return optim_R_set, optim_C_set, optim_X_all