# SfM

## Overview
This part contains the implementation of a Structure from Motion (SFM) pipeline capable of reconstructing 3D structures from a set of 2D images. The project demonstrates various steps involved in SFM, including feature extraction, feature matching, triangulation, and bundle adjustment.

## Prerequisites
- Python 3.7 or above
- OpenCV
- NumPy
- Matplotlib

## Results

- ![Matched Features](StructurefromMotion/Data/IntermediateOutputImages/Matched_features.jpg) Feature matching between image pairs.
- ![Matched Features](StructurefromMotion/Data/IntermediateOutputImages/Matched_features_after_RANSAC.jpg) Feature matching between image pairs after RANSAC.
- ![Non-Linear Triangulation](StructurefromMotion/Data/IntermediateOutputImages/Non_Linear_Triangulation.png) The 3D reconstruction from non-linear triangulation.
- ![Output](StructurefromMotion/Data/IntermediateOutputImages/final.png) The complete 3D reconstruction representing the culmination of the SFM pipeline.

# NeRF

## Overview
This project implements a NeRF (Neural Radiance Fields) model to create novel view synthesis of 3D scenes from a set of 2D images. It's based on the groundbreaking research that combines deep learning with a volumetric scene representation.

## Prerequisites
- Python 
- PyTorch
- CUDA (for GPU acceleration)

## Results

### Lego with positional encoding
![Lego with positional encoding](NeRF/images/lego.gif)

### Lego without positional encoding
![Lego without positional encoding](NeRF/images/lego_no_enc.gif)

### Ship with positional encoding
![Ship with positional encoding](NeRF/images/ship.gif)
