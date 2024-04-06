import numpy as np

def find_visible_cam(X_found, filtered_feature_flag, current_camera_index):
    temp_feature_flag = np.zeros((filtered_feature_flag.shape[0]), dtype = int)
    for n in range(current_camera_index + 1):
        temp_feature_flag = temp_feature_flag | filtered_feature_flag[:,n]

    visible_3d_point_indices = np.where((X_found.reshape(-1)) & (temp_feature_flag))

    visibility_matrix = X_found[visible_3d_point_indices].reshape(-1,1)
    for n in range(current_camera_index + 1):
        visibility_matrix = np.hstack((visibility_matrix, filtered_feature_flag[visible_3d_point_indices, n].reshape(-1,1)))

    _, c = visibility_matrix.shape
    return visible_3d_point_indices, visibility_matrix[:, 1:c]