"""Multi-frame colored point cloud map and optional Open3D TSDF reconstruction."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .calibration import Calibration
from .geometry import color_cloud, transform_points, voxel_reduce, write_ply
from .io import load_rgbd
from .association import FramePair

def fuse_colored_map(pairs: list[FramePair], poses: np.ndarray, calibration: Calibration,
                    *, output: Path, stride: int = 6, every: int = 10,
                    voxel: float = 0.035, max_points: int = 1000000) -> dict:
    """Combine aligned RGB-D clouds in common coordinates using known/estimated poses."""
    if len(poses) != len(pairs):
        raise ValueError("One camera pose per RGB-D pair required")
    if every < 1 or max_points < 1:
        raise ValueError("Invalid sampling controls")
    chunks, colors = [], []
    for i in range(0, len(pairs), every):
        rgb, depth = load_rgbd(pairs[i].rgb_file, pairs[i].depth_file)
        xyz, color = color_cloud(rgb, depth, calibration, stride=stride)
        if len(xyz):
            chunks.append(transform_points(xyz, poses[i]))
            colors.append(color)
    if not chunks:
        raise ValueError("No valid depth points in selected RGB-D images")
    all_xyz = np.concatenate(chunks)
    all_rgb = np.concatenate(colors)
    cloud, rgb = voxel_reduce(all_xyz, all_rgb, voxel)
    if len(cloud) > max_points:
        # Deterministic truncation after voxelization; no random point selection.
        indices = np.linspace(0, len(cloud) - 1, max_points, dtype=int)
        cloud, rgb = cloud[indices], rgb[indices]
    write_ply(output, cloud, rgb)
    return {"sampled_frames": len(chunks), "points_before_voxel": int(len(all_xyz)),
            "points_exported": int(len(cloud)), "ply": str(output)}

def integrate_tsdf(pairs: list[FramePair], poses: np.ndarray, calibration: Calibration,
                   *, mesh_path: Path, every: int = 10,
                   voxel_length: float = 0.015, sdf_trunc: float = 0.06) -> dict:
    """Optional TSDF integration. Requires Open3D installed separately."""
    import open3d as o3d
    if len(pairs) != len(poses):
        raise ValueError("Pose count mismatch")
    intr = o3d.camera.PinholeCameraIntrinsic(
        640, 480, calibration.fx, calibration.fy, calibration.cx, calibration.cy)
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length, sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    used = 0
    for i in range(0, len(pairs), every):
        color, depth = load_rgbd(pairs[i].rgb_file, pairs[i].depth_file)
        if color.shape[:2] != (480, 640):
            intr = o3d.camera.PinholeCameraIntrinsic(
                color.shape[1], color.shape[0], calibration.fx, calibration.fy, calibration.cx, calibration.cy)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color), o3d.geometry.Image(depth),
            depth_scale=calibration.depth_factor, depth_trunc=calibration.max_depth_m,
            convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, np.linalg.inv(poses[i]))
        used += 1
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(str(mesh_path), mesh):
        raise RuntimeError("Failed to export reconstructed mesh")
    return {"frames_integrated": used, "vertices": len(mesh.vertices),
            "triangles": len(mesh.triangles), "mesh": str(mesh_path)}
