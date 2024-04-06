import numpy as np
from scipy.spatial.transform import Rotation 
import scipy.optimize as optimize
from utils import *

def non_linear_pnp(camera_matrix, image_points, object_points, initial_rotation, initial_translation):
    initial_quaternion = get_quaternion(initial_rotation)
    initial_params = [initial_quaternion[0], initial_quaternion[1], initial_quaternion[2], initial_quaternion[3], 
                      initial_translation[0], initial_translation[1], initial_translation[2]]
    optimized_params = optimize.least_squares(
        fun=pnp_error_3,
        x0=initial_params,
        method="dogbox",
        args=[object_points, image_points, camera_matrix],
        verbose=0)
    optimized_vector = optimized_params.x
    optimized_quaternion = optimized_vector[:4]
    optimized_translation = optimized_vector[4:]
    optimized_rotation = get_rotation(optimized_quaternion)
    return optimized_rotation, optimized_translation