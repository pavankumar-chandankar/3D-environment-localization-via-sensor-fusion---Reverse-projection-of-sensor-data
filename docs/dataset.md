# TUM RGB-D benchmark

Original publisher: Computer Vision Group, Technical University of Munich.

- Official benchmark: https://cvg.cit.tum.de/data/datasets/rgbd-dataset
- File formats and camera parameters: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
- Sequence download: https://webshare.cvg.cit.tum.de/g/rgbd/dataset/
- Reference paper: J. Sturm et al., *A Benchmark for the Evaluation of RGB-D SLAM Systems*, IROS 2012.

## Data selected

`freiburg1_xyz` uses a Kinect scanning an office desk with predominantly translational camera motion. The official benchmark reports approximately 30 seconds of recording and a 7.112 m ground-truth trajectory. This sequence is used for debugging the front end; it is not a claim of full-loop SLAM performance. `freiburg1_desk` and `freiburg1_rpy` are additional manually selectable recordings.

A normal unpacked archive contains `rgb/`, `depth/`, `rgb.txt`, `depth.txt` and `groundtruth.txt`. Each RGB/depth index line contains `<timestamp> <relative image path>`. Ground truth rows are `<timestamp> tx ty tz qx qy qz qw` in metres and normalized quaternion coordinates. The actual RGB and depth timestamps differ; association is an essential preprocessing step.

Depth images are unsigned 16-bit PNGs. Divide raw pixel values by **5000** for metres (not the common 1000 used by some other RGB-D sensors). The default registered depth calibration follows the dataset's ROS default intrinsics rather than incorrectly applying color-only distortion corrections.

## Reproduction

```bash
python scripts/download_tum.py --sequence freiburg1_xyz --destination data
rgbd-lab inspect data/rgbd_dataset_freiburg1_xyz
rgbd-lab track data/rgbd_dataset_freiburg1_xyz --output results/fr1_xyz
```

The data are excluded from Git and are downloaded from the publisher. The download is not automatically run during every pull request because the original archive is approximately 420 MB. An explicit GitHub Actions `workflow_dispatch` can download and evaluate the first 120 frames.

Dataset attribution: **Sturm, Engelhard, Endres, Burgard, Cremers (IROS 2012)**; current publisher page indicates **CC BY 4.0** for the dataset. Confirm the current source terms before any redistribution of images or derived assets.
