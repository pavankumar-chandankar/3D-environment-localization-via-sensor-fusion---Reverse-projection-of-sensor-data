# RGB-D Visual Odometry and 3D Reconstruction

A small research codebase for estimating camera motion and reconstructing indoor scenes from synchronized **color and depth images**. It started as a reverse-projection experiment. I developed that into a complete offline processing pipeline with frame association, metric geometry, visual odometry, trajectory evaluation, and point-cloud mapping.

The primary benchmark is **[TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)**, recorded with a Microsoft Kinect and released by Sturm and colleagues. In particular, `freiburg1_xyz` is a useful initial sequence: the camera mostly translates along the principal axes, making it easier to debug pose direction, depth scale and trajectory drift. Its archive is roughly 420 MB, so recordings are downloaded separately rather than committed here.

## Pipeline

```
RGB timestamps ──┐
                 ├─ pair RGB/depth by timestamp ─ camera calibration ─┐
Depth timestamps ┘                                                   │
                                                                     ├─ back-project depth → metric XYZ
RGB + aligned depth ─ ORB → descriptor matches → lift to XYZ ───────┤
                                                                     ├─ RANSAC 3D→2D PnP → T_current_previous
                                                                     └─ accumulate T_world_camera
                                                                              │
                                                                              ├─ compare with motion-capture GT
                                                                              ├─ ATE (rigid SE3) and RPE
                                                                              └─ fuse transformed RGB-D clouds / TSDF
```

The odometry front end uses **OpenCV ORB features**, nearest-neighbor descriptor matching, valid depth samples in the previous frame, and `solvePnPRansac` with a minimum inlier threshold. Every frame transition records its match count, inliers, reprojection error and tracking status. When tracking fails, the previous pose is held, *and the failure is counted*; it is not silently removed from the trajectory.

The geometry and evaluation modules use NumPy and SciPy. [Open3D](https://www.open3d.org/) is optional for TSDF mesh integration; a colored PLY point-cloud path works without it.

## Getting started

Python 3.10+ and a normal C++/CUDA-free environment are sufficient for the base pipeline.

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install -e '.[dev]'
pytest -q
```

Download the original `freiburg1_xyz` sequence (approximately 420 MB). The downloader checks the remote URL and rejects unsafe archive paths. The source dataset stays in `data/`, which Git ignores.

```bash
python scripts/download_tum.py --sequence freiburg1_xyz --destination data
rgbd-lab inspect data/rgbd_dataset_freiburg1_xyz
```

**Run visual odometry and evaluate against the motion-capture trajectory:**

```bash
rgbd-lab track data/rgbd_dataset_freiburg1_xyz \
  --camera fr1-default --max-frames 120 \
  --output results/fr1_xyz_120
cat results/fr1_xyz_120/metrics.json
```

Omit `--max-frames` to process the full sequence. `tracking.csv` records successful and failed transitions; `trajectory.txt` uses the standard TUM timestamp/translation/quaternion format. `metrics.json` includes SE(3)-aligned **absolute trajectory error (ATE)** and **relative pose error (RPE)** in metres when ground truth is available.

**Export a colored point cloud from one recorded RGB-D pair:**

```bash
rgbd-lab cloud data/rgbd_dataset_freiburg1_xyz \
  --output results/first_frame.ply --pixel-stride 3
```

**Fuse multiple recorded views:**

```bash
# Evaluation/oracle reconstruction using motion-capture poses
rgbd-lab reconstruct data/rgbd_dataset_freiburg1_xyz \
  --pose-source groundtruth --every 10 --output results/gt_map.ply

# Real odometry reconstruction, using the estimated trajectory
rgbd-lab reconstruct data/rgbd_dataset_freiburg1_xyz \
  --pose-source estimated --trajectory results/fr1_xyz_120/trajectory.txt \
  --max-frames 120 --every 10 --output results/estimated_map.ply
```

For **Open3D TSDF** meshes, install `pip install -e '.[reconstruction]'` and append `--tsdf`; give the output a `.ply` extension. The two pose sources are intentionally separate: reconstruction with motion-capture poses measures a different problem from reconstruction with estimated camera poses.

**Plot estimated and reference trajectories:**

```bash
python scripts/plot_trajectory.py \
  --estimated results/fr1_xyz_120/trajectory.txt \
  --groundtruth data/rgbd_dataset_freiburg1_xyz/groundtruth.txt \
  --output results/fr1_xyz_120/trajectory.png
```

## What is measured

| Output | Description |
| --- | --- |
| `tracking.csv` | Number of ORB matches, depth correspondences, RANSAC inliers, frame failures, reprojection error |
| `trajectory.txt` | Camera-to-world pose per processed frame, in TUM format |
| `metrics.json` | Rigidly aligned ATE RMSE/median/max, translation RPE, number of GT-matched frames |
| `*.ply` | Colorized world-space point cloud or optional fused TSDF mesh |

No numerical accuracy is advertised without an actual run against the public recording. Unit tests do use small generated geometry to verify math and failure behavior; **those are test fixtures, not benchmark data or research outcomes**.

## Implementation notes

- **Depth units:** TUM uses 16-bit depth PNGs, with values divided by **5000** to obtain metres. Invalid, zero, nonfinite and out-of-range samples are rejected before projection.
- **RGB/depth association:** timestamps are paired one-to-one with a default maximum gap of 20 ms; depth and color files are not assumed to have identical timestamp lists.
- **Calibration:** for pre-registered Freiburg1 RGB-D images the ROS default pinhole projection (`fx=fy=525`, `cx=319.5`, `cy=239.5`) is the default. The TUM calibrated Freiburg1 RGB intrinsics are also selectable, but correcting distortion in the registered depth stream requires care.
- **Pose convention:** `T_world_camera` transforms a camera-frame point into world coordinates. The PnP solver returns `T_current_previous`, which is inverted before accumulation.
- **Trajectory comparison:** ground-truth poses are interpolated only inside the supplied interval and within a specified time gap; ATE uses **SE(3) alignment without scale adjustment**. Unmatched timestamps are reported, not counted as perfect predictions.
- **Inertial data:** a separate Kinect accelerometer diagnostic reader is available. Accelerometer magnitude checks are **not** an IMU pose estimator; no Kalman filter or ROS robot localization is asserted by this code.

## Research notes

[Dataset and attribution](docs/dataset.md) · [Estimation methodology](docs/methodology.md) · [Evaluation protocol](docs/evaluation.md) · [Limitations](docs/limitations.md)

Original exploratory scripts are preserved under [`legacy/`](legacy/). Those snippets were useful for testing back-projection, but this package is the maintained implementation.

## Dataset citation

> J. Sturm, N. Engelhard, F. Endres, W. Burgard and D. Cremers, “A Benchmark for the Evaluation of RGB-D SLAM Systems,” *IROS*, 2012.

The dataset belongs to its original authors and is available under the license stated by its publisher (CC BY 4.0). This repository contains original processing code only; neither the raw RGB/depth recordings nor motion-capture trajectories are redistributed here.
