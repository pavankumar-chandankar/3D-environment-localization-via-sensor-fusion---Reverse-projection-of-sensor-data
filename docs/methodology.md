# Geometry and tracking

## Coordinate frames

The point `(u,v)` is back-projected using depth `z` and intrinsic matrix `K`:

\[ X=(u-c_x)z/f_x,\quad Y=(v-c_y)z/f_y,\quad Z=z. \]

The depth PNG is converted to metres before projection; zero/invalid depth is excluded. `T_world_camera` maps camera-coordinate points to world coordinates. A pose `T_current_previous` estimated from PnP must be inverted to update the world trajectory:

\[ T_{world,current}=T_{world,previous}\cdot (T_{current,previous})^{-1}. \]

## Front-end odometry

1. Parse and sort RGB/depth indices; pair frames one-to-one within 20 ms.
2. Extract ORB binary descriptors from consecutive RGB images; apply Hamming nearest-neighbor matching and Lowe's ratio filter.
3. Query *previous-frame depth* under matched keypoints and lift them into previous-frame 3D coordinates.
4. Estimate the relative transform to corresponding *current-frame* 2D keypoints using calibrated `solvePnPRansac`.
5. Reject a motion if descriptor matches, depth correspondences, RANSAC consensus or refined pose checks fail.
6. Refine the accepted pose on inliers; report reprojection RMSE.
7. Accumulate camera-to-world transforms; on a rejected transition, hold the previous transform and mark the failure in the CSV.

This is incremental RGB-D visual odometry. There is no global bundle adjustment, loop closure or dense direct odometry optimization; longer sequences can drift.

## 3D reconstruction

Two mapping options are provided:

- **Point cloud:** back-project depth with RGB colors, transform each frame with its estimated or reference pose, concatenate, average into spatial voxels, and export ASCII PLY.
- **TSDF:** optionally use Open3D's scalable TSDF volume to integrate recorded RGB/depth images at selected camera poses and extract a triangle mesh.

Reference-pose mapping and estimated-pose mapping are deliberately kept separate, because the latter includes odometry errors and the former does not.

## IMU distinction

The original exploratory script concatenated accelerometer triples with 3D surface points, which is not a physically meaningful estimator. The maintained code includes a read-only accelerometer diagnostic parser and does not use those observations to claim six-degree-of-freedom inertial fusion. A real visual-inertial estimator would additionally require calibrated sensor extrinsics, clock alignment, proper motion states, gyroscope data and gravity/bias handling.
