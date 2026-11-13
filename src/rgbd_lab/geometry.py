"""Back-projection, camera/world transforms and lightweight PLY export."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .calibration import Calibration

def backproject(depth_raw: np.ndarray, calibration: Calibration, *, stride: int = 1):
    """Return (N,3) camera-frame XYZ in metres and valid pixel indices (row,col)."""
    if stride < 1:
        raise ValueError("Stride must be >= 1")
    z = calibration.metric_depth(depth_raw)[::stride, ::stride]
    v, u = np.indices(z.shape)
    u = u * stride
    v = v * stride
    valid = z > 0.0
    zp = z[valid].astype(np.float64)
    points = np.column_stack(((u[valid] - calibration.cx) * zp / calibration.fx,
                              (v[valid] - calibration.cy) * zp / calibration.fy, zp))
    pixels = np.column_stack((v[valid], u[valid])).astype(np.int32)
    return points, pixels

def color_cloud(rgb: np.ndarray, depth: np.ndarray, calibration: Calibration,
                *, stride: int = 2):
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[:2] != depth.shape:
        raise ValueError("Color and depth must be aligned HxW images")
    xyz, pixels = backproject(depth, calibration, stride=stride)
    return xyz, rgb[pixels[:, 0], pixels[:, 1]]

def transform_points(points: np.ndarray, world_from_camera: np.ndarray) -> np.ndarray:
    if world_from_camera.shape != (4, 4) or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Invalid points or transform")
    return points @ world_from_camera[:3, :3].T + world_from_camera[:3, 3]

def voxel_reduce(points: np.ndarray, colors: np.ndarray, voxel: float = 0.03):
    """Average points and RGB within fixed-size voxels, keeping color aligned."""
    if voxel <= 0 or not np.isfinite(voxel):
        raise ValueError("Voxel edge must be positive")
    if points.shape != (len(colors), 3) or colors.shape != (len(points), 3):
        raise ValueError("Points and colors must have matching Nx3 shapes")
    if len(points) == 0:
        return points.copy(), colors.copy()
    if not np.isfinite(points).all():
        raise ValueError("Point cloud contains non-finite coordinates")
    index = np.floor(points / voxel).astype(np.int64)
    _, inv, counts = np.unique(index, axis=0, return_inverse=True, return_counts=True)
    out_xyz = np.zeros((len(counts), 3), dtype=np.float64)
    out_rgb = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(out_xyz, inv, points)
    np.add.at(out_rgb, inv, colors.astype(float))
    return out_xyz / counts[:, None], np.rint(out_rgb / counts[:, None]).clip(0, 255).astype(np.uint8)

def write_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Export portable ASCII PLY; no heavy 3D library required."""
    if points.shape != (len(colors), 3) or colors.shape != (len(points), 3):
        raise ValueError("Invalid colored point cloud")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for xyz, rgb in zip(points, colors):
            file.write(f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} {int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n")
