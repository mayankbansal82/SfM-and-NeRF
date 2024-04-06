import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation 


def projection_matrix(R, C, K):
    C = np.reshape(C, (-1, 1))
    Rt = np.hstack((R, -R.dot(C)))
    P = K.dot(Rt)
    return P

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def estimate_epipole(F):
    _,_,V = np.linalg.svd(F)
    e = V[-1,:]
    e /= e[-1]
    return e

def plot_epipolar_lines(F, x1, x2, img1, img2):
    e1 = estimate_epipole(F)
    e2 = estimate_epipole(F.T)
    img1_copy = img1.copy()
    img2_copy = img2.copy()
    for pt in x1:
        img1_copy = cv2.line(img1_copy, tuple(pt.astype(int)), tuple(e1[:-1].astype(int)), (0,0,0), 1)
    for pt in x2:
        img2_copy = cv2.line(img2_copy, tuple(pt.astype(int)), tuple(e2[:-1].astype(int)), (0,0,0), 1)
    cv2.imshow('Epiploar_Lines_1', img1_copy)
    cv2.waitKey(0)
    cv2.imwrite('./Data/IntermediateOutputImages/epipolar_lines1.jpg', img1_copy)
    cv2.imshow('Epiploar_Lines_2', img2_copy)
    cv2.waitKey(0)
    cv2.imwrite('./Data/IntermediateOutputImages/epipolar_lines2.jpg', img2_copy)
    cv2.destroyAllWindows()

def projection_matrix(R,C,K):
    C = np.reshape(C,(3,1))
    I = np.identity(3)
    P = np.dot(K,np.dot(R,np.hstack((I,-C))))
    return P

def reprojection_error(X,pt1, pt2, R1, C1, R2, C2, K):
    P1 = projection_matrix(R1, C1, K)
    P2 = projection_matrix(R2, C2, K)
    P1_row1, P1_row2, P1_row3 = P1
    P1_row1, P1_row2, P1_row3 = P1_row1.reshape(1,4), P1_row2.reshape(1,4), P1_row3.reshape(1,4)
    P2_row1, P2_row2, P2_row3 = P2
    P2_row1, P2_row2, P2_row3 = P2_row1.reshape(1,4), P2_row2.reshape(1,4), P2_row3.reshape(1,4)
    X_homog = X.reshape(4,1)
    u1, v1 = pt1[0], pt1[1]
    u1_proj = np.divide(P1_row1.dot(X_homog), P1_row3.dot(X_homog))
    v1_proj = np.divide(P1_row2.dot(X_homog), P1_row3.dot(X_homog))
    err1 = np.square(u1 - u1_proj) + np.square(v1 - v1_proj)
    u2, v2 = pt2[0], pt2[1]
    u2_proj = np.divide(P2_row1.dot(X_homog), P2_row3.dot(X_homog))
    v2_proj = np.divide(P2_row2.dot(X_homog), P2_row3.dot(X_homog))
    err2 = np.square(u2 - u2_proj) + np.square(v2 - v2_proj)
    return err1, err2

def show_feature_matches(img1, img2, pts1, pts2,name):
    matches_img = np.hstack([img1, img2])
    keypoints1 = [cv2.KeyPoint(pt[0], pt[1], 3) for pt in pts1]
    keypoints2 = [cv2.KeyPoint(pt[0], pt[1], 3) for pt in pts2]
    good_matches = [cv2.DMatch(index, index, 0) for index in range(len(pts1))]
    matches_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, matches_img, (0, 255, 0), (0, 0, 255))
    cv2.imshow(name, matches_img)
    cv2.waitKey(0)
    cv2.imwrite(f"./Data/IntermediateOutputImages/{name}.jpg", matches_img)
    cv2.destroyAllWindows()




def plot_reprojected_points(img1, pts1, pts2, pts3D, R, C, K,name, initial_color=(0, 250, 0), reproj_color=(0, 0, 255)):
    C = C.reshape(3, 1)
    Rt = np.hstack((R, -np.dot(R, C)))
    P = np.dot(K, Rt)
    pts3D_xyz = pts3D[:, :3]
    pts3D_homog = np.hstack((pts3D_xyz, np.ones((len(pts3D_xyz), 1))))
    proj_pts = np.dot(P, pts3D_homog.T)
    proj_pts = (proj_pts[:2, :] / proj_pts[2, :]).T
    for pt1, proj_pt in zip(pts1, proj_pts):
        pt1 = tuple(map(int, pt1))
        proj_pt = tuple(map(int, proj_pt.reshape(-1)))
        cv2.circle(img1, pt1, 4, initial_color, -1)
        cv2.circle(img1, proj_pt, 4, reproj_color, -1)
    cv2.imshow(name, img1)
    cv2.waitKey(0)
    cv2.imwrite(f"./Data/IntermediateOutputImages/{name}.jpg", img1)
    cv2.destroyAllWindows()

def plot_triangulation_3d(X_list,name):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    x_coords = np.concatenate([X[:, 0] for X in X_list])
    z_coords = np.concatenate([X[:, 2] for X in X_list])
    color_array = np.zeros((x_coords.shape[0], 3))
    total_points = 0
    for i, X in enumerate(X_list):
        color = plt.cm.tab10(i)
        color_array[total_points:total_points+X.shape[0], :] = np.tile(color[:3], (X.shape[0], 1))
        total_points += X.shape[0]
    ax.scatter(x_coords, z_coords, s=1, c=color_array)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_title(name)
    plt.savefig(f"./Data/IntermediateOutputImages/{name}.png")
    plt.show()

def homogenize(pts):
    return np.hstack((pts, np.ones((pts.shape[0], 1))))

def pnp_error_3(X0, x3D, pts, K):
    Q, C = X0[:4], X0[4:].reshape(-1,1)
    R = get_rotation(Q)
    P = projection_matrix(R,C,K)
    Error = []
    for X, pt in zip(x3D, pts):
        p_1T, p_2T, p_3T = P
        p_1T, p_2T, p_3T = p_1T.reshape(1,-1), p_2T.reshape(1,-1), p_3T.reshape(1,-1)
        X = homogenize(X.reshape(1,-1)).reshape(-1,1)
        u, v = pt[0], pt[1]
        u_proj = np.divide(p_1T.dot(X) , p_3T.dot(X))
        v_proj =  np.divide(p_2T.dot(X) , p_3T.dot(X))
        E = np.square(v - v_proj) + np.square(u - u_proj)
        Error.append(E)
    sumError = np.mean(np.array(Error).squeeze())
    return sumError


def pnp_error_3(camera_params, world_points, image_points, calibration_matrix):
    quaternion, camera_center = camera_params[:4], camera_params[4:].reshape(-1, 1)
    rotation_matrix = get_rotation(quaternion)
    projection_matrix_ = projection_matrix(rotation_matrix, camera_center, calibration_matrix)
    errors = []
    for world_point, image_point in zip(world_points, image_points):
        row1, row2, row3 = projection_matrix_
        row1, row2, row3 = row1.reshape(1, -1), row2.reshape(1, -1), row3.reshape(1, -1)
        homogenous_point = homogenize(world_point.reshape(1, -1)).reshape(-1, 1)
        u, v = image_point[0], image_point[1]
        u_proj = np.divide(row1.dot(homogenous_point), row3.dot(homogenous_point))
        v_proj = np.divide(row2.dot(homogenous_point), row3.dot(homogenous_point))
        error = np.square(v - v_proj) + np.square(u - u_proj)
        errors.append(error)
    mean_error = np.mean(np.array(errors).squeeze())
    return mean_error

def pnp_error(x3D_points, image_points, camera_matrix, rotation_matrix, camera_center):
    projection_matrix_ = projection_matrix(rotation_matrix, camera_center, camera_matrix)
    errors = []
    for X, pt in zip(x3D_points, image_points):
        X_homogeneous = homogenize(X.reshape(1, -1)).reshape(-1, 1)
        p1, p2, p3 = projection_matrix_
        u_proj = np.divide(p1.dot(X_homogeneous), p3.dot(X_homogeneous))[0]
        v_proj = np.divide(p2.dot(X_homogeneous), p3.dot(X_homogeneous))[0]
        u, v = pt[0], pt[1]
        squared_error = np.square(u - u_proj) + np.square(v - v_proj)
        errors.append(squared_error)
    mean_error = np.mean(errors)
    return mean_error

def get_quaternion(R2):
    Q = Rotation.from_matrix(R2)
    return Q.as_quat()

def get_rotation(Q, type_ = 'q'):
    if type_ == 'q':
        R = Rotation.from_quat(Q)
    elif type_ == 'e':
        R = Rotation.from_rotvec(Q)
    return R.as_matrix()

def project(points_3d, camera_params, intrinsic_params):
    projected_points = []
    for i in range(len(camera_params)):
        rotation_matrix = get_rotation(camera_params[i, :3], 'e')
        translation_vector = camera_params[i, 3:].reshape(3, 1)
        projection_matrix = np.dot(intrinsic_params, np.dot(rotation_matrix, np.hstack((np.identity(3), -translation_vector))))
        homogeneous_3d_points = np.hstack((points_3d[i], 1)).T
        homogeneous_2d_points = np.dot(projection_matrix, homogeneous_3d_points)
        projected_2d_points = homogeneous_2d_points[:2] / homogeneous_2d_points[2]
        projected_points.append(projected_2d_points)
    return np.array(projected_points)


def compute_residuals(x0, num_cameras, num_points, camera_indices, point_indices, observed_points, intrinsic_matrix):
    num_total_cameras = num_cameras + 1
    camera_params = x0[:num_total_cameras * 6].reshape((num_total_cameras, 6))
    points_3d = x0[num_total_cameras * 6:].reshape((num_points, 3))
    projected_points = project(points_3d[point_indices], camera_params[camera_indices], intrinsic_matrix)
    residuals = (projected_points - observed_points).ravel()
    return residuals

def get_2d_points(X_index, visibility_matrix, feature_x, feature_y):
    visible_feature_x = feature_x[X_index]
    visible_feature_y = feature_y[X_index]
    row_indices, col_indices = np.where(visibility_matrix == 1)
    pts_2d = np.column_stack((visible_feature_x[row_indices, col_indices], visible_feature_y[row_indices, col_indices]))
    return pts_2d

def get_camera_point_indices(visibility_matrix):
    row_indices, col_indices = np.where(visibility_matrix == 1)
    camera_indices = col_indices
    point_indices = row_indices
    return camera_indices, point_indices

def rotate(points, rotation_vectors):
    theta = np.linalg.norm(rotation_vectors, axis=1)[:, np.newaxis]
    unit_rotation_vectors = np.divide(rotation_vectors, theta, out=np.zeros_like(rotation_vectors), where=theta != 0)
    dot_product = np.sum(points * unit_rotation_vectors, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotated_points = (cos_theta * points) + (sin_theta * np.cross(unit_rotation_vectors, points)) + ((1 - cos_theta) * dot_product * unit_rotation_vectors)
    return rotated_points

def read_matches_file(filename, img_idx):
    pts1 = []
    pts2 = []
    seen_pts = set()

    with open(filename, 'r') as f:
        lines = f.readlines()[1:]
        
        for line in lines:
            entries = line.strip().split()
            n_matches = int(entries[0]) - 1
            
            for i in range(n_matches):
                x1, y1 = float(entries[4]), float(entries[5])
                
                if int(entries[i*3+6]) == img_idx:
                    x2, y2 = float(entries[i*3+7]), float(entries[i*3+8])
                    
                    if (x1, y1, x2, y2) not in seen_pts:
                        pts1.append([x1, y1])
                        pts2.append([x2, y2])
                        seen_pts.add((x1, y1, x2, y2))

    return np.array(pts1), np.array(pts2)

def features_extraction(data_folder_path):
    num_images = 5
    feature_x = []
    feature_y = []
    feature_flag = []

    for n in range(1, num_images):
        file_path = data_folder_path + "/matching" + str(n) + ".txt"
        matching_file = open(file_path, "r")

        for i, row in enumerate(matching_file):
            if i == 0:
                row_elements = row.split(':')
            else:
                x_row = np.zeros((1,num_images))
                y_row = np.zeros((1,num_images))
                flag_row = np.zeros((1,num_images), dtype=int)

                row_elements = row.split()
                columns = [float(x) for x in row_elements]
                columns = np.asarray(columns)
                num_matches = columns[0]
             
                current_x = columns[4]
                current_y = columns[5]
                x_row[0,n-1] = current_x
                y_row[0,n-1] = current_y
                flag_row[0,n-1] = 1

                m = 1
                while num_matches > 1:
                    image_id = int(columns[5+m])
                    image_id_x = int(columns[6+m])
                    image_id_y = int(columns[7+m])
                    m = m + 3
                    num_matches = num_matches - 1

                    x_row[0, image_id - 1] = image_id_x
                    y_row[0, image_id - 1] = image_id_y
                    flag_row[0, image_id - 1] = 1

                feature_x.append(x_row)
                feature_y.append(y_row)
                feature_flag.append(flag_row)

    feature_x = np.asarray(feature_x).reshape(-1,num_images)
    feature_y = np.asarray(feature_y).reshape(-1,num_images)
    feature_flag = np.asarray(feature_flag).reshape(-1,num_images)

    return feature_x, feature_y, feature_flag


def get_euler(R2):
    euler = Rotation.from_matrix(R2)
    return euler.as_rotvec()

def pnp_error_2(feature_2d, point_3d, rotation_matrix, translation_vector, camera_matrix):
    u, v = feature_2d
    pts = point_3d.reshape(1,-1)
    point_3d_homogeneous = homogenize(pts).reshape(-1,1)
    point_3d_homogeneous = point_3d_homogeneous.reshape(4,1)
    translation_vector = translation_vector.reshape(-1,1)
    projection_matrix_ = projection_matrix(rotation_matrix, translation_vector, camera_matrix)
    row_1, row_2, row_3 = projection_matrix_
    row_1, row_2, row_3 = row_1.reshape(1,4), row_2.reshape(1,4), row_3.reshape(1,4)
    u_proj = np.divide(row_1.dot(point_3d_homogeneous), row_3.dot(point_3d_homogeneous))
    v_proj = np.divide(row_2.dot(point_3d_homogeneous), row_3.dot(point_3d_homogeneous))
    projected_2d_feature = np.hstack((u_proj, v_proj))
    observed_2d_feature = np.hstack((u, v))
    reprojection_error = np.linalg.norm(observed_2d_feature - projected_2d_feature)
    return reprojection_error