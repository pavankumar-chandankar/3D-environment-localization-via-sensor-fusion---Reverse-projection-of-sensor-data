"""Ground-truth trajectory evaluation with rigid SE(3) alignment and RPE."""
from __future__ import annotations
import numpy as np
from .association import interpolate_poses

def align_se3(estimated: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch rigid (no scale) alignment of Nx3 estimated to reference positions."""
    if estimated.shape != reference.shape or estimated.ndim != 2 or estimated.shape[1] != 3:
        raise ValueError("Need two matching Nx3 position arrays")
    if len(estimated) < 3 or not np.isfinite(estimated).all() or not np.isfinite(reference).all():
        raise ValueError("At least three valid correspondences required")
    a, b = estimated.mean(axis=0), reference.mean(axis=0)
    U, _, Vt = np.linalg.svd((estimated - a).T @ (reference - b))
    R = Vt.T @ np.diag([1.0, 1.0, np.linalg.det(Vt.T @ U.T)]) @ U.T
    t = b - R @ a
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = R, t
    return estimated @ R.T + t, matrix

def trajectory_metrics(times: np.ndarray, estimated_poses: np.ndarray,
                       groundtruth_times: np.ndarray, groundtruth_poses: np.ndarray,
                       *, maximum_gap: float = 0.05, rpe_step: int = 1) -> dict:
    """ATE RMSE after SE(3) alignment; RPE translation on time-matched poses.

    RPE is computed from *unaligned* relative translations (norm is invariant to the
    shared world alignment). Missing groundtruth is not counted as zero error.
    """
    if len(times) != len(estimated_poses) or estimated_poses.shape[1:] != (4,4):
        raise ValueError("Pose/timestamp mismatch")
    if rpe_step < 1:
        raise ValueError("RPE step must be positive")
    reference, mask = interpolate_poses(times, groundtruth_times, groundtruth_poses, max_gap=maximum_gap)
    ids = np.flatnonzero(mask)
    if len(ids) < 3:
        raise ValueError("Fewer than three frames have valid ground truth")
    aligned, transform = align_se3(estimated_poses[ids, :3, 3], reference[ids, :3, 3])
    residual = np.linalg.norm(aligned - reference[ids, :3, 3], axis=1)
    # Relative poses are expressed in coordinate frame i; translation errors cannot
    # be computed by simple world-space displacement subtraction when camera rotates.
    rel_errors = []
    for i in range(len(ids) - rpe_step):
        a, b = ids[i], ids[i + rpe_step]
        e_delta = np.linalg.inv(estimated_poses[a]) @ estimated_poses[b]
        g_delta = np.linalg.inv(reference[a]) @ reference[b]
        error = np.linalg.inv(g_delta) @ e_delta
        rel_errors.append(float(np.linalg.norm(error[:3, 3])))
    out = {
        "matched_frames": int(len(ids)), "input_frames": int(len(times)),
        "ate_rmse_m": float(np.sqrt(np.mean(residual**2))),
        "ate_median_m": float(np.median(residual)),
        "ate_max_m": float(np.max(residual)),
        "rpe_translation_rmse_m": float(np.sqrt(np.mean(np.square(rel_errors)))) if rel_errors else None,
        "rpe_step_frames": int(rpe_step), "alignment": "SE3_rigid_no_scale",
        "max_gt_interpolation_gap_s": maximum_gap,
    }
    return out
