"""Validated RGB/depth loading and TUM trajectory text output."""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

def load_rgbd(rgb_path: Path, depth_path: Path) -> tuple[np.ndarray, np.ndarray]:
    color = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if color is None or depth is None:
        raise FileNotFoundError(f"Cannot decode RGB/depth PNG: {rgb_path} {depth_path}")
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.integer):
        raise ValueError("Expected 16-bit, single-channel depth PNG")
    if color.shape[:2] != depth.shape:
        raise ValueError("RGB and depth images are not registered/aligned")
    if depth.dtype != np.uint16:
        raise ValueError("TUM raw depth PNG must be uint16")
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB), depth

def write_trajectory(path: Path, times: np.ndarray, poses: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("# timestamp tx ty tz qx qy qz qw\n")
        for t, pose in zip(times, poses):
            q = Rotation.from_matrix(pose[:3, :3]).as_quat()
            xyz = pose[:3, 3]
            file.write(f"{t:.8f} {xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f} "
                       f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}\n")
