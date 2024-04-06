import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils import *
from EstimateFundamentalMatrix import *
from GetInliersRANSAC import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from NonlinearTriangulation import *
from DisambiguateCameraPose import *
from LinearPnP import *
from PnPRANSAC import *
from NonlinearPnP import *
from BuildVisibilityMatrix import *
from BundleAdjustment import *

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='Data/P3Data', help='Data directory')
    args = parser.parse_args()
    data_dir = args.data

    images = []
    for i in range(1, 6):  
        path = f"{data_dir}/{i}.png"
        image = cv2.imread(path)
        images.append(image)

    if not os.path.exists('./Data/IntermediateOutputImages'):
        os.makedirs('./Data/IntermediateOutputImages')

    feature_x, feature_y, feature_flag = features_extraction(data_dir)  

    filtered_feature_flag = np.zeros_like(feature_flag)  
    f_matrix = np.empty(shape=(5, 5), dtype=object) 

    for i in range(0, 4):  
        for j in range(i + 1, 5):
            idx = np.where(feature_flag[:, i] & feature_flag[:, j])  
            pts1, pts2 = read_matches_file(f"{data_dir}/matching{i + 1}.txt", j + 1)  
            idx = np.array(idx).reshape(-1)
            if i==0 and j==1:
                show_feature_matches(images[i], images[j], pts1, pts2,"Matched_features")

            if len(idx) > 8:  
                F_inliers, inliers_idx = ransac(pts1, pts2, idx, 1000, 0.01)  
                f_matrix[i, j] = F_inliers
                filtered_feature_flag[inliers_idx, j] = 1
                filtered_feature_flag[inliers_idx, i] = 1
                if i==0 and j==1:
                    plot_epipolar_lines(F_inliers,pts1,pts2,images[i],images[j])

    F = f_matrix[0, 1]  

    K = np.loadtxt(f"{data_dir}/calibration.txt")  
    E = essential_matrix_from_fundamental_matrix(K, F)  

    R_set, C_set = extract_camera_pose(E)  

    idx = np.where(filtered_feature_flag[:, 0] & filtered_feature_flag[:, 1])
    pts1 = np.hstack((feature_x[idx, 0].reshape((-1, 1)), feature_y[idx, 0].reshape((-1, 1))))  
    pts2 = np.hstack((feature_x[idx, 1].reshape((-1, 1)), feature_y[idx, 1].reshape((-1, 1))))
    show_feature_matches(images[0], images[1], pts1, pts2,"Matched_features_after_RANSAC")

    R1_ = np.identity(3) 
    C1_ = np.zeros((3,1)) 

    pts3D_4 = []
    for i in range(len(C_set)):
        x1 = pts1
        x2 = pts2
        X = linear_triangulation(K, C1_, R1_, C_set[i], R_set[i], x1, x2)  

        X = X / X[:, 3].reshape(-1, 1)
        pts3D_4.append(X)  

    R_best, C_best, X = disambiguate_pose(R_set, C_set, pts3D_4)  
    X = X / X[:, 3].reshape(-1, 1)
    plot_triangulation_3d([X],"Linear_Triangulation")
    plot_reprojected_points(images[0], pts1, pts2, X,R_best, C_best, K,"Reprojected_points_after_linear_triangulation")
    X_refined = nonLinear_triangulation(K, pts1, pts2, X, R1_, C1_, R_best, C_best)  
    X_refined = X_refined / X_refined[:, 3].reshape(-1, 1)
    plot_triangulation_3d([X_refined],"Non_Linear_Triangulation")
    plot_reprojected_points(images[0], pts1, pts2, X_refined, R_best, C_best, K,"Reprojected_points_after_non_linear_triangulation")

    X_all = np.zeros((feature_x.shape[0], 3))  
    cam_indices = np.zeros((feature_x.shape[0], 1), dtype=int)  
    X_found = np.zeros((feature_x.shape[0], 1), dtype=int)  
    X_all[idx] = X[:, :3]
    X_found[idx] = 1
    cam_indices[idx] = 1
    X_found[np.where(X_all[:2] < 0)] = 0

    C_set = []  
    R_set = []

    C_set_before =[]
    R_set_before =[]  

    C0 = np.zeros(3)  
    R0 = np.identity(3)  
    C_set.append(C0) 
    R_set.append(R0)
    C_set.append(C_best)
    R_set.append(R_best)

    C_set_before.append(C0)
    R_set_before.append(R0)
    C_set_before.append(C_best)
    R_set_before.append(R_best)



    for i in range(2, 5):
        feature_idx_i = np.where(X_found[:, 0] & filtered_feature_flag[:, i])  
        if len(feature_idx_i[0]) < 8:
            continue

 
        pts_i = np.hstack((feature_x[feature_idx_i, i].reshape(-1,1), feature_y[feature_idx_i, i].reshape(-1,1))) 
        X = X_all[feature_idx_i,:].reshape(-1,3)  

        R_init, C_init = pnp_ransac(K, pts_i, X, 3000, 2)  

        Ri, Ci = non_linear_pnp(K, pts_i, X, R_init, C_init)  
        C_set.append(Ci)
        R_set.append(Ri)

        C_set_before.append(Ci)
        R_set_before.append(Ri)

        for k in range(0, i):
            idx_X_pts = np.where(filtered_feature_flag[:, k] & filtered_feature_flag[:, i])  
            idx_X_pts = np.asarray(idx_X_pts)
            idx_X_pts = np.squeeze(idx_X_pts)

            if (len(idx_X_pts) < 8):
                continue

            x1 = np.hstack((feature_x[idx_X_pts, k].reshape(-1, 1), feature_y[idx_X_pts, k].reshape(-1, 1)))
            x2 = np.hstack((feature_x[idx_X_pts, i].reshape(-1, 1), feature_y[idx_X_pts, i].reshape(-1, 1)))

            X_d = linear_triangulation(K, C_set[k], R_set[k], Ci, Ri, x1, x2)

            X_d = X_d / X_d[:, 3].reshape(-1, 1)
            pts1, pts2 = x1, x2

            X = nonLinear_triangulation(K, x1, x2, X_d, R_set[k], C_set[k], Ri, Ci)
            X = X / X[:, 3].reshape(-1, 1)

            X_all[idx_X_pts] = X[:, :3]
            X_found[idx_X_pts] = 1

            X_index, visibility_matrix = find_visible_cam(X_found, filtered_feature_flag, i)

            X_all_before = X_all

            R_set_, C_set_, X_all = bundle_adjustment(X_index, visibility_matrix, X_all, X_found, feature_x, feature_y,
                                                    filtered_feature_flag, R_set, C_set, K, i)

            for k in range(0, i + 1):
                idx_X_pts = np.where(X_found[:, 0] & filtered_feature_flag[:, k])
                x = np.hstack((feature_x[idx_X_pts, k].reshape(-1, 1), feature_y[idx_X_pts, k].reshape(-1, 1)))
                X = X_all[idx_X_pts]

    X_found[X_all_before[:, 2] < 0] = 0

    feature_idx = np.where(X_found[:, 0])
    X = X_all_before[feature_idx]
    x = X[:, 0]
    z = X[:, 2]

    fig = plt.figure(figsize=(10, 10))
    plt.xlim(-4, 6)
    plt.ylim(-2, 12)
    plt.scatter(x, z, marker='.', linewidths=0.5, color='blue')

    

    for i in range(0, len(C_set_before)):
        R1 = get_euler(R_set_before[i])
        R1 = np.rad2deg(R1)
        plt.plot(C_set_before[i][0], C_set_before[i][2], marker=(3, 0, int(R1[1])), markersize=15, linestyle='None')
        
    plt.savefig('./Data/IntermediateOutputImages/Before_Bundle_Adjustment.png')
    plt.show()
    

    X_found[X_all[:, 2] < 0] = 0

    feature_idx = np.where(X_found[:, 0])
    X = X_all[feature_idx]
    x = X[:, 0]
    z = X[:, 2]

    fig = plt.figure(figsize=(10, 10))
    plt.xlim(-4, 6)
    plt.ylim(-2, 12)
    plt.scatter(x, z, marker='.', linewidths=0.5, color='blue')
    
    for i in range(0, len(C_set_)):
        R1 = get_euler(R_set_[i])
        R1 = np.rad2deg(R1)
        plt.plot(C_set_[i][0], C_set_[i][2], marker=(3, 0, int(R1[1])), markersize=15, linestyle='None')

    plt.savefig('Final.png')
    plt.show()

if __name__ == '__main__':
    main()