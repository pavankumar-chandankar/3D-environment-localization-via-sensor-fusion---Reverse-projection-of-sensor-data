# Evaluation procedure

TUM ground-truth pose rows are in the format `timestamp tx ty tz qx qy qz qw`. The timestamps need not coincide with RGB timestamps. The evaluator interpolates translation linearly and orientation with quaternion spherical interpolation, only when both neighboring GT samples are close enough to the camera timestamp.

## Metrics

**ATE RMSE:** rigidly align predicted camera centers to ground-truth centers using SVD and an SE(3) transform (no scale change), then compute root-mean-squared Euclidean position residual. Report the number of matched frames along with the score.

**RPE translation RMSE:** for each chosen pair of frames, compute the relative camera motion in each trajectory and compare the translation components of `inverse(Delta_gt) @ Delta_est`. The default interval is one processed frame. This value depends on the frame stride, so comparisons must use the same stride.

**Tracking quality:** count the accepted and rejected frame transitions. A held pose following tracking failure remains in the sequence and affects ATE; failures cannot be hidden by evaluating only successful transitions.

## Experimental protocol

- Fix sequence name, intrinsics, min/max depth, association tolerance, pixel stride and maximum image count before comparing experiments.
- Run `freiburg1_xyz` first to validate coordinate conventions, then `freiburg1_rpy` for rotations and `freiburg1_desk` for more general motion.
- Store `tracking.csv`, `trajectory.txt`, `metrics.json`, full CLI command, and dependency versions for each run.
- Compare full sequences, not only an early subset, when stating benchmark-level performance.
- Report failure rate and the proportion of frames with interpolated ground truth; an ATE number without frame coverage can mislead.
- Avoid using ground-truth poses to tune the tracker and then presenting results as if evaluated on held-out data.

A GitHub workflow provides an explicit real-data download/evaluation task. Ordinary CI tests use small constructed images to verify program behavior; they are not TUM benchmark scores.
