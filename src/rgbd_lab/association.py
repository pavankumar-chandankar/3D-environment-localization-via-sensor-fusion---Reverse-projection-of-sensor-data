"""TUM RGB-D timestamp association without data-order assumptions."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class FramePair:
    rgb_t: float
    rgb_file: Path
    depth_t: float
    depth_file: Path


def read_index(path: Path) -> list[tuple[float, str]]:
    entries = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{number}: expected timestamp and relative path")
        timestamp = float(fields[0])
        if not np.isfinite(timestamp):
            raise ValueError(f"{path}:{number}: invalid timestamp")
        candidate = Path(fields[1])
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"{path}:{number}: path must be inside recording")
        entries.append((timestamp, fields[1]))
    if len(set(x[0] for x in entries)) != len(entries):
        raise ValueError(f"{path}: duplicate timestamps")
    return sorted(entries)


def pair_timestamps(rgb: list[tuple[float, str]], depth: list[tuple[float, str]],
                    tolerance: float = 0.02) -> list[tuple[int, int]]:
    """One-to-one nearest-time pairing, globally prioritized by smallest gap."""
    if tolerance <= 0:
        raise ValueError("Timestamp tolerance must be positive")
    # Use bisect and immediate neighboring candidates; avoid an O(N*M) pair list.
    import bisect
    depth_order = sorted(range(len(depth)), key=lambda index: depth[index][0])
    times = [depth[k][0] for k in depth_order]
    proposed = []
    for i, (timestamp, _) in enumerate(rgb):
        j = bisect.bisect_left(times, timestamp)
        for k in (j - 1, j):
            if 0 <= k < len(times):
                delta = abs(timestamp - times[k])
                if delta <= tolerance:
                    proposed.append((delta, i, depth_order[k]))
    used_rgb, used_depth, matched = set(), set(), []
    for delta, i, j in sorted(proposed):
        if i not in used_rgb and j not in used_depth:
            used_rgb.add(i); used_depth.add(j)
            matched.append((i, j))
    return sorted(matched)


def load_recording(folder: Path, tolerance: float = 0.02, *, check_files: bool = True) -> list[FramePair]:
    rgb, depth = read_index(folder / "rgb.txt"), read_index(folder / "depth.txt")
    matches = pair_timestamps(rgb, depth, tolerance)
    pairs = [FramePair(rgb[i][0], folder / rgb[i][1], depth[j][0], folder / depth[j][1])
             for i, j in matches]
    if not pairs:
        raise ValueError("No RGB/depth timestamps matched; check maximum time difference")
    if check_files:
        for pair in pairs:
            if not pair.rgb_file.is_file() or not pair.depth_file.is_file():
                raise FileNotFoundError(f"Missing RGB-D frame: {pair}")
    return pairs


def read_groundtruth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Timestamp -> 4x4 camera-to-world SE3 pose (translation + x,y,z,w quaternion)."""
    from scipy.spatial.transform import Rotation
    times, poses = [], []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 8:
            raise ValueError(f"groundtruth line {number}: expected eight columns")
        nums = np.asarray([float(x) for x in fields], dtype=np.float64)
        if not np.all(np.isfinite(nums)) or np.linalg.norm(nums[4:]) < 1e-8:
            raise ValueError(f"groundtruth line {number}: invalid values")
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rotation.from_quat(nums[4:]).as_matrix()
        T[:3, 3] = nums[1:4]
        times.append(nums[0]); poses.append(T)
    if not times:
        raise ValueError("No ground-truth samples")
    order = np.argsort(times)
    timestamps = np.asarray(times)[order]
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Duplicate ground-truth timestamps")
    return timestamps, np.asarray(poses)[order]


def interpolate_poses(query: np.ndarray, gt_times: np.ndarray, gt_poses: np.ndarray,
                      max_gap: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate T_world_camera; reject extrapolation or distant GT samples."""
    from scipy.spatial.transform import Rotation, Slerp
    if len(gt_times) < 2:
        raise ValueError("Need two ground-truth samples for interpolation")
    indices = np.searchsorted(gt_times, query, side="right")
    valid = (indices > 0) & (indices < len(gt_times))
    i0 = np.clip(indices - 1, 0, len(gt_times) - 2)
    i1 = i0 + 1
    valid &= (query - gt_times[i0] <= max_gap) & (gt_times[i1] - query <= max_gap)
    result = np.repeat(np.eye(4)[None], len(query), axis=0)
    for k in np.flatnonzero(valid):
        j0, j1 = int(i0[k]), int(i1[k])
        w = (query[k] - gt_times[j0]) / (gt_times[j1] - gt_times[j0])
        result[k, :3, 3] = (1 - w) * gt_poses[j0, :3, 3] + w * gt_poses[j1, :3, 3]
        rot = Rotation.from_matrix(gt_poses[[j0, j1], :3, :3])
        result[k, :3, :3] = Slerp(gt_times[[j0, j1]], rot)([query[k]]).as_matrix()[0]
    return result, valid
