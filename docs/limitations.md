# Limitations and next steps

The system estimates camera poses incrementally from ORB features and depth-assisted PnP. It is not a complete SLAM system. It lacks bundle adjustment, loop closure, place recognition, motion-model filtering, geometric map consistency and multi-camera calibration. Long trajectories can accumulate drift and feature-sparse images may fail to track.

The default camera intrinsics are appropriate to the published TUM RGB-D registered data format; different sensors require explicit calibration and correct depth scaling. A hard-coded Kinect calibration should not be reused with unrelated RGB-D cameras.

TUM data is sourced from the official publisher and is not redistributed by this repository. The complete public recording is hundreds of megabytes; normal CI does not download it. A full real-data ATE result must come from running the public benchmark, not from generated unit-test fixtures.

The optional Open3D TSDF path depends on Open3D availability and machine memory. It was designed for offline reconstruction, not robot deployment at frame rate. The accelerometer parser computes diagnostics only: the dataset's accelerometer data are not integrated into an EKF, and there is no claimed ROS2 / TurtleBot navigation stack.

Useful future work includes robust photometric tracking, depth confidence modeling, loop closure, failure recovery, and a real visual-inertial module using gyro and accelerometer signals with calibrated camera-to-IMU transformations.
