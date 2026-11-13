# RGB-D Visual Odometry and 3D Reconstruction

An offline computer-vision project for estimating camera motion and reconstructing indoor scenes from color and depth images. It began as an experiment in depth-image back-projection. The maintained implementation combines timestamp association, calibrated 3D geometry, ORB/PnP visual odometry, trajectory evaluation, and point-cloud fusion.

The evaluation dataset is [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset), originally recorded using a Microsoft Kinect. In particular, the Freiburg1 XYZ recording has largely translational camera movements, which makes it useful for debugging coordinate frames and trajectory errors. Full RGB/depth recordings are downloaded from TUM, not stored in this source repository.

## Run

Requires Python 3.10 or later.

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -e '.[dev]'
    pytest -q

Download approximately 420 MB of the original RGB-D sequence:

    python scripts/download_tum.py --sequence freiburg1_xyz --destination data
    rgbd-lab inspect data/rgbd_dataset_freiburg1_xyz

Track the first 120 recorded RGB-D pairs and compare with motion-capture ground truth:

    rgbd-lab track data/rgbd_dataset_freiburg1_xyz --max-frames 120 --output results/fr1_xyz
    cat results/fr1_xyz/metrics.json

Export a calibrated RGB-D point cloud and reconstruct across multiple poses:

    rgbd-lab cloud data/rgbd_dataset_freiburg1_xyz --output results/frame.ply
    rgbd-lab reconstruct data/rgbd_dataset_freiburg1_xyz --pose-source groundtruth --every 10 --output results/map_gt.ply
    rgbd-lab reconstruct data/rgbd_dataset_freiburg1_xyz --pose-source estimated --trajectory results/fr1_xyz/trajectory.txt --max-frames 120 --every 10 --output results/map_estimated.ply

The reference-pose map and estimated-pose map answer different questions. Using motion-capture poses does not validate visual odometry.

## Tracking and geometry

- **RGB/depth association:** one-to-one nearest timestamp pairing within 20 ms, with image and index validation.
- **Reverse projection:** 16-bit TUM depth images are converted to metres by dividing by 5000, then projected into camera coordinates with a pinhole model and explicit intrinsics.
- **Pose estimation:** ORB features in consecutive RGB frames are matched, previous-frame depth lifts the keypoints to 3D, and OpenCV PnP/RANSAC estimates the relative camera transform. A consensus inlier threshold and reprojection check guard against unreliable poses.
- **Trajectory:** poses are accumulated as camera-to-world transforms. Tracking failures are recorded and the last pose held; failed steps are not silently discarded.
- **Evaluation:** estimated camera positions are rigidly aligned to interpolated TUM motion-capture trajectories. Metrics include absolute trajectory error (ATE), translation relative pose error (RPE), matched-frame counts, and the tracking-failure rate.
- **Mapping:** frames are combined into a voxel-reduced colored PLY point cloud. An optional Open3D extra supports TSDF mesh reconstruction.

The source is organized into calibration, association, geometry, odometry, evaluation and mapping modules under src/rgbd_lab. The tests check numeric projection, timestamp matching, pose conventions, failure cases and end-to-end CLI behavior.

Small generated test images are used only to verify program behavior. No numerical TUM accuracy result is presented without running the real recording. An optional manually triggered GitHub Actions workflow downloads and evaluates the first 120 frames.

## Sensor-fusion scope

The RGB and depth channels are fused geometrically to produce registered 3D information. Kinect accelerometer logs can be inspected separately, but the project does not implement a calibrated visual-inertial Kalman filter, ROS navigation or real-time robot localization.

## Documentation

See [TUM dataset](docs/dataset.md), [methodology](docs/methodology.md), [evaluation protocol](docs/evaluation.md) and [limitations](docs/limitations.md). The original experimental scripts are preserved under [legacy](legacy/).

Data attribution: J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers, “A Benchmark for the Evaluation of RGB-D SLAM Systems,” IROS 2012. The TUM RGB-D data are distributed by their original publisher (CC BY 4.0) and are not redistributed here.
