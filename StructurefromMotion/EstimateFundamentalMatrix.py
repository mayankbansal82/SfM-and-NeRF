import numpy as np


def estimate_fundamental_matrix(corresponding_points_image_1, corresponding_points_image_2):
    x1_coordinates, y1_coordinates = corresponding_points_image_1[:, 0], corresponding_points_image_1[:, 1]
    x2_coordinates, y2_coordinates = corresponding_points_image_2[:, 0], corresponding_points_image_2[:, 1]

    homogeneous_coordinates = np.ones(x1_coordinates.shape[0])

    A = [x1_coordinates * x2_coordinates, y1_coordinates * x2_coordinates, x2_coordinates, 
         x1_coordinates * y2_coordinates, y1_coordinates * y2_coordinates, y2_coordinates, 
         x1_coordinates, y1_coordinates, homogeneous_coordinates]  # N x 9
    A = np.vstack(A).T 

    U, D, V = np.linalg.svd(A)
    fundamental_matrix_vector = V[-1, :]

    fundamental_matrix_noisy = fundamental_matrix_vector.reshape(3, 3)

    UF, UD, UV = np.linalg.svd(fundamental_matrix_noisy)
    UD[-1] = 0
    fundamental_matrix = UF @ np.diag(UD) @ UV
    fundamental_matrix /= fundamental_matrix[2, 2]

    return fundamental_matrix

