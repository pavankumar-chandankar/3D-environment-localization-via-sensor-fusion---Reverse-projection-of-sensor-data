"""Pinhole calibration and explicit raw-depth-unit conventions."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Calibration:
    fx: float
    fy: float
    cx: float
    cy: float
    depth_factor: float = 5000.0  # TUM RGB-D PNG: depth integer / 5000 -> metres
    min_depth_m: float = 0.15
    max_depth_m: float = 5.0

    def __post_init__(self):
        vals = (self.fx, self.fy, self.cx, self.cy, self.depth_factor,
                self.min_depth_m, self.max_depth_m)
        if not all(np.isfinite(v) for v in vals):
            raise ValueError("Non-finite calibration parameter")
        if min(self.fx, self.fy, self.depth_factor, self.min_depth_m) <= 0:
            raise ValueError("Focal lengths and scales must be positive")
        if self.max_depth_m <= self.min_depth_m:
            raise ValueError("Depth range is invalid")

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def metric_depth(self, depth_raw: np.ndarray) -> np.ndarray:
        if depth_raw.ndim != 2 or not np.issubdtype(depth_raw.dtype, np.number):
            raise ValueError("Expected a single-channel numeric depth image")
        z = depth_raw.astype(np.float32) / self.depth_factor
        z[~np.isfinite(z) | (z < self.min_depth_m) | (z > self.max_depth_m)] = 0.0
        return z

# Official TUM values for RGB projection. TUM recommends the ROS default for
# pre-registered depth because re-undistorting depth images is non-trivial.
# https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
CAMERAS = {
    "fr1-default": Calibration(525.0, 525.0, 319.5, 239.5),
    "fr1-rgb-calibrated": Calibration(517.3, 516.5, 318.6, 255.3),
    "fr2-default": Calibration(525.0, 525.0, 319.5, 239.5),
    "fr3": Calibration(535.4, 539.2, 320.1, 247.6),
}
