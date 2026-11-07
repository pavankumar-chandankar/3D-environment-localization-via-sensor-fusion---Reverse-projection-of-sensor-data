# RGB-D reverse projection and point-cloud processing — prototype

This repository is a collection of Python experiments for converting depth images to 3D points, creating point clouds from RGB-D data, combining point arrays, and visualizing trajectories. The code uses NumPy, OpenCV, Open3D and Matplotlib.

**Important distinction:** The current repository does not establish an operational Kinect/IMU Kalman fusion system, ROS nodes, TurtleBot integration, calibrated localization accuracy, or a complete robot navigation pipeline. These were broader project objectives described in the previous README, not demonstrated end-to-end by the checked-in code.

## Implemented components

- `Reverse_projection.py`: pinhole-camera back-projection from a depth image and an intrinsic matrix `K` to 3D camera coordinates.
- `Sensor_fusion.py`: Open3D RGB-D image-to-point-cloud conversion with hard-coded example Kinect intrinsics and a demonstration of stacking inertial sample vectors with point coordinates.
- `Visualize.py`: additional 3D visualization code.
- `track.py`: Matplotlib plotting helper for point arrays and tracks.

The function `fuse_sensor_data` is a **concatenation demonstration**, not physical sensor fusion or a Kalman filter. The example script uses randomly generated inertial values rather than actual sensor acquisition. Combining arrays this way is not a valid localization estimator.

## Installing dependencies

Create an isolated Python environment and install the packages used by these scripts:

```bash
python -m pip install numpy opencv-python open3d matplotlib
```

Source code examples reference `rgb_image.png` and `depth_image.png`, which are not included. Use compatible test images and calibrate camera intrinsics for your own sensor before evaluating the geometry. Some `__main__` examples are illustrative rather than turnkey: for example, `Reverse_projection.py` references undefined example variables `fx`, `fy`, `cx` and `cy` in its standalone block.

## Minimal mathematical contract

For each pixel `(u,v)` and depth value `z`, reverse projection estimates 3D camera coordinates using

```text
[X, Y, Z]^T = z * inverse(K) * [u, v, 1]^T
```

Depth units must match the intended output coordinate system and intrinsic values must be appropriate for the image. The code does not estimate robot pose or quantify localization error.

## CV-supported description

*Implemented Python RGB-D depth back-projection and Open3D point-cloud processing prototypes, with exploratory IMU data handling and 3D visualization.*

This replaces the unsupported description of a verified ROS-integrated RGB-D/IMU localization system.
