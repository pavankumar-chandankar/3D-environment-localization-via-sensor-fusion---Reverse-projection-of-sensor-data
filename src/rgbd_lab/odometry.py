"""ORB RGB-D visual odometry: depth-lifted 3D→2D tracks with PnP/RANSAC."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from .calibration import Calibration

@dataclass(frozen=True)
class Tracking:
    transform_current_from_previous: np.ndarray
    matches: int
    depth_correspondences: int
    inliers: int
    reprojection_rmse_px: float
    success: bool
    reason: str


def estimate_motion(previous_rgb: np.ndarray, previous_depth: np.ndarray,
                    current_rgb: np.ndarray, calibration: Calibration,
                    *, max_features: int = 1800, min_inliers: int = 12,
                    reprojection_threshold_px: float = 2.5) -> Tracking:
    """Estimate T_current_from_previous, returning an explicit failure record."""
    identity = np.eye(4, dtype=np.float64)
    def failed(reason: str, match_count: int = 0, corr: int = 0, ins: int = 0):
        return Tracking(identity.copy(), match_count, corr, ins, float("nan"), False, reason)
    if (previous_rgb.ndim != 3 or current_rgb.ndim != 3 or
        previous_rgb.shape[:2] != previous_depth.shape or
        current_rgb.shape[:2] != previous_rgb.shape[:2]):
        raise ValueError("RGB/depth frames have incompatible shapes")
    if max_features < 20 or min_inliers < 6 or reprojection_threshold_px <= 0:
        raise ValueError("Invalid feature or RANSAC settings")
    gray_a = cv2.cvtColor(previous_rgb, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=12)
    key_a, desc_a = orb.detectAndCompute(gray_a, None)
    key_b, desc_b = orb.detectAndCompute(gray_b, None)
    if desc_a is None or desc_b is None or len(key_a) < min_inliers:
        return failed("Insufficient ORB features")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(desc_a, desc_b, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    # Require unique destination features so repeated patterns do not overcount.
    unique = {}
    for m in sorted(good, key=lambda x: x.distance):
        unique.setdefault(m.trainIdx, m)
    matches = list(unique.values())
    if len(matches) < min_inliers:
        return failed("Insufficient descriptor matches", len(matches))
    z = calibration.metric_depth(previous_depth)
    object_xyz, pixels_uv = [], []
    for match in matches:
        ua, va = key_a[match.queryIdx].pt
        x, y = int(round(ua)), int(round(va))
        if not (0 <= x < z.shape[1] and 0 <= y < z.shape[0]):
            continue
        depth_m = float(z[y, x])
        if depth_m <= 0:
            continue
        point = [(ua - calibration.cx) * depth_m / calibration.fx,
                 (va - calibration.cy) * depth_m / calibration.fy, depth_m]
        object_xyz.append(point)
        pixels_uv.append(key_b[match.trainIdx].pt)
    if len(object_xyz) < min_inliers:
        return failed("Insufficient valid depth correspondences",len(matches),len(object_xyz))
    objects = np.asarray(object_xyz, dtype=np.float64)
    pixels = np.asarray(pixels_uv, dtype=np.float64)
    ok, rvec, tvec, idx = cv2.solvePnPRansac(
        objects, pixels, calibration.matrix, None, flags=cv2.SOLVEPNP_EPNP,
        iterationsCount=250, reprojectionError=reprojection_threshold_px, confidence=0.999)
    if not ok or idx is None or len(idx) < min_inliers:
        return failed("RANSAC pose rejected",len(matches),len(objects),len(idx) if idx is not None else 0)
    inlier_indices = idx.ravel()
    # Refine only from consensus observations.
    refined, rvec, tvec = cv2.solvePnP(
        objects[inlier_indices], pixels[inlier_indices], calibration.matrix, None,
        rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not refined:
        return failed("PnP refinement failed",len(matches),len(objects),len(idx))
    projection, _ = cv2.projectPoints(objects[inlier_indices], rvec, tvec, calibration.matrix, None)
    rmse = float(np.sqrt(np.mean(np.sum((projection.reshape(-1, 2) - pixels[inlier_indices])**2, axis=1))))
    R, _ = cv2.Rodrigues(rvec)
    if not np.isfinite(R).all() or not np.isfinite(tvec).all() or not np.isfinite(rmse):
        return failed("Non-finite pose",len(matches),len(objects),len(idx))
    transformation = np.eye(4)
    transformation[:3, :3], transformation[:3, 3] = R, tvec.flatten()
    return Tracking(transformation, len(matches), len(objects), len(idx), rmse, True, "tracked")


def accumulate_poses(motions: list[Tracking]) -> tuple[np.ndarray, np.ndarray]:
    """First pose is identity. Failed motion holds last pose and marks tracking false."""
    world_from_camera = np.eye(4)
    poses = [world_from_camera.copy()]
    tracked = [True]
    for motion in motions:
        if motion.success:
            world_from_camera = world_from_camera @ np.linalg.inv(motion.transform_current_from_previous)
        poses.append(world_from_camera.copy())
        tracked.append(bool(motion.success))
    return np.asarray(poses), np.asarray(tracked)
