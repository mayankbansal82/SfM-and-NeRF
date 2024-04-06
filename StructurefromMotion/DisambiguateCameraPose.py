import numpy as np



def disambiguate_pose(rotation_matrices, camera_centers, three_d_points):
    best_index = 0
    max_positive_depths = 0

    for i, (rotation_matrix, camera_center, three_d_point) in enumerate(zip(rotation_matrices, camera_centers, three_d_points)):
        r3 = rotation_matrix[2, :]
        three_d_point = three_d_point / three_d_point[:, 3][:, np.newaxis]
        three_d_point = three_d_point[:, :3]
        n_positive_depths = ((r3.dot((three_d_point - camera_center).T) > 0) & (three_d_point[:, 2] > 0)).sum()

        if n_positive_depths > max_positive_depths:
            best_index = i
            max_positive_depths = n_positive_depths

    best_rotation_matrix, best_camera_center, best_three_d_points = rotation_matrices[best_index], camera_centers[best_index], three_d_points[best_index]

    return best_rotation_matrix, best_camera_center, best_three_d_points