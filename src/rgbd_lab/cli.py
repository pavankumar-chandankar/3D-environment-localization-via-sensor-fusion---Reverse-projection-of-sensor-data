"""Commands for the TUM RGB-D benchmark (real recordings, no bundled data)."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .association import load_recording, read_groundtruth, interpolate_poses
from .calibration import CAMERAS
from .evaluation import trajectory_metrics
from .geometry import color_cloud, write_ply, voxel_reduce
from .io import load_rgbd, write_trajectory
from .mapping import fuse_colored_map, integrate_tsdf
from .odometry import estimate_motion, accumulate_poses


def arguments() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rgbd-lab", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    def common(name, help):
        s = sub.add_parser(name, help=help)
        s.add_argument("recording", type=Path, help="Extracted TUM RGB-D sequence directory")
        s.add_argument("--camera", choices=sorted(CAMERAS), default="fr1-default")
        s.add_argument("--association-tolerance", type=float, default=0.02)
        s.add_argument("--stride-frames", type=int, default=1)
        s.add_argument("--max-frames", type=int, default=0, help="0 = entire recording")
        return s
    common("inspect", "Check index, image pairs, and groundtruth coverage")
    cloud = common("cloud", "Export a calibrated RGB-D point cloud from one real frame")
    cloud.add_argument("--output", type=Path, required=True)
    cloud.add_argument("--pixel-stride", type=int, default=3)
    cloud.add_argument("--voxel", type=float, default=0.03)
    vo = common("track", "Run ORB + depth PnP/RANSAC and write TUM trajectory")
    vo.add_argument("--output", type=Path, default=Path("results/tum-fr1-xyz"))
    vo.add_argument("--max-features", type=int, default=1800)
    vo.add_argument("--min-inliers", type=int, default=12)
    vo.add_argument("--reprojection-px", type=float, default=2.5)
    vo.add_argument("--rpe-step", type=int, default=1)
    vo.add_argument("--gt-max-gap", type=float, default=0.05)
    fusion = common("reconstruct", "Fuse multiple RGB-D scans using tracking or groundtruth poses")
    fusion.add_argument("--pose-source", choices=["estimated", "groundtruth"], default="estimated")
    fusion.add_argument("--trajectory", type=Path, help="TUM-format track output for estimated poses")
    fusion.add_argument("--output", type=Path, default=Path("results/map.ply"))
    fusion.add_argument("--every", type=int, default=10)
    fusion.add_argument("--pixel-stride", type=int, default=6)
    fusion.add_argument("--voxel", type=float, default=0.035)
    fusion.add_argument("--tsdf", action="store_true", help="Use optional Open3D TSDF integration")
    fusion.add_argument("--max-points", type=int, default=1000000)
    return p


def select_pairs(args):
    if args.stride_frames < 1 or args.max_frames < 0:
        raise ValueError("Invalid recording frame sampling")
    pairs = load_recording(args.recording, args.association_tolerance)
    pairs = pairs[::args.stride_frames]
    return pairs[:args.max_frames] if args.max_frames else pairs


def inspect(args, pairs):
    info = {"source": str(args.recording), "pairs": len(pairs),
            "first_rgb_timestamp": pairs[0].rgb_t, "last_rgb_timestamp": pairs[-1].rgb_t,
            "maximum_rgb_depth_offset_s": max(abs(x.rgb_t - x.depth_t) for x in pairs)}
    rgb, depth = load_rgbd(pairs[0].rgb_file, pairs[0].depth_file)
    cal = CAMERAS[args.camera]
    info.update({"image_resolution": [rgb.shape[1], rgb.shape[0]],
                 "valid_depth_fraction_first_frame": float(np.mean(cal.metric_depth(depth) > 0)),
                 "intrinsics": cal.matrix.tolist(), "depth_png_units_per_metre": cal.depth_factor})
    if (args.recording / "groundtruth.txt").is_file():
        gt_times, gt_poses = read_groundtruth(args.recording / "groundtruth.txt")
        _, ok = interpolate_poses(np.asarray([p.rgb_t for p in pairs]), gt_times, gt_poses)
        info["groundtruth_interpolatable_frames"] = int(ok.sum())
    return info


def cloud(args, pairs):
    rgb, depth = load_rgbd(pairs[0].rgb_file, pairs[0].depth_file)
    points, colors = color_cloud(rgb, depth, CAMERAS[args.camera], stride=args.pixel_stride)
    pts, color = voxel_reduce(points, colors, voxel=args.voxel)
    write_ply(args.output, pts, color)
    return {"frame_timestamp": pairs[0].rgb_t, "valid_depth_points":len(points),
            "exported_points":len(pts), "pointcloud":str(args.output)}


def _run_tracking(args, pairs):
    previous_rgb, previous_depth = load_rgbd(pairs[0].rgb_file, pairs[0].depth_file)
    motions = []
    cal = CAMERAS[args.camera]
    for frame in pairs[1:]:
        current_rgb, current_depth = load_rgbd(frame.rgb_file, frame.depth_file)
        motion = estimate_motion(previous_rgb, previous_depth, current_rgb, cal,
                                 max_features=args.max_features,
                                 min_inliers=args.min_inliers,
                                 reprojection_threshold_px=args.reprojection_px)
        motions.append(motion)
        previous_rgb, previous_depth = current_rgb, current_depth
    return accumulate_poses(motions), motions


def track(args, pairs):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (poses, ok), motions = _run_tracking(args, pairs)
    times = np.asarray([p.rgb_t for p in pairs])
    write_trajectory(output / "trajectory.txt", times, poses)
    with (output / "tracking.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "tracked", "orb_matches", "depth_matches", "ransac_inliers", "reprojection_rmse_px", "reason"])
        writer.writerow([f"{times[0]:.8f}", True, 0, 0, 0, "", "initial_frame"])
        for t, motion in zip(times[1:], motions):
            writer.writerow([f"{t:.8f}", motion.success, motion.matches,
                             motion.depth_correspondences, motion.inliers,
                             motion.reprojection_rmse_px if motion.success else "", motion.reason])
    summary = {"sequence": args.recording.name, "frames": len(pairs),
               "tracked_transitions": int(ok[1:].sum()),
               "failed_transitions": int((~ok[1:]).sum()),
               "trajectory": str(output / "trajectory.txt")}
    groundtruth = args.recording / "groundtruth.txt"
    if groundtruth.exists():
        gt_times, gt_poses = read_groundtruth(groundtruth)
        try:
            summary.update(trajectory_metrics(times, poses, gt_times, gt_poses,
                                             maximum_gap=args.gt_max_gap, rpe_step=args.rpe_step))
        except ValueError as error:
            summary["evaluation_error"] = str(error)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def reconstruct(args, pairs):
    times = np.asarray([p.rgb_t for p in pairs])
    if args.pose_source == "groundtruth":
        gt_times, gt_poses = read_groundtruth(args.recording / "groundtruth.txt")
        poses, valid = interpolate_poses(times, gt_times, gt_poses)
        pairs = [p for p, ok in zip(pairs, valid) if ok]
        poses = poses[valid]
        if not pairs:
            raise ValueError("No camera poses matched groundtruth")
    else:
        if not args.trajectory:
            raise ValueError("--trajectory is required with --pose-source estimated")
        times_gt, poses_gt = read_groundtruth(args.trajectory)
        # Estimated trajectories come from this same indexed stream. Forbid silent
        # association to unrelated frame timestamps.
        indices = np.searchsorted(times_gt, times)
        if np.any(indices >= len(times_gt)) or np.any(np.abs(times_gt[indices] - times) > 1e-4):
            raise ValueError("Estimated trajectory timestamps do not match selected frames")
        poses = poses_gt[indices]
    cal = CAMERAS[args.camera]
    if args.tsdf:
        result = integrate_tsdf(pairs, poses, cal, mesh_path=args.output, every=args.every)
    else:
        result = fuse_colored_map(pairs, poses, cal, output=args.output,
                                stride=args.pixel_stride, every=args.every,
                                voxel=args.voxel, max_points=args.max_points)
    return {"pose_source": args.pose_source, **result}


def main(argv=None):
    args = arguments().parse_args(argv)
    pairs = select_pairs(args)
    command = {"inspect":inspect,"cloud":cloud,"track":track,"reconstruct":reconstruct}[args.command]
    print(json.dumps(command(args, pairs), indent=2))

if __name__ == "__main__":
    main()
