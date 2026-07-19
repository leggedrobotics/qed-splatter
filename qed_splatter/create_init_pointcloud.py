"""Create a depth-backprojected point cloud for Gaussian initialization.

Two-step workflow:
1. Backproject depth maps into an uncolored point cloud (default).
2. Colorize an existing point cloud from RGB images (``--colorize``).

The output PLY is written in the same OpenGL / Nerfstudio world coordinates as
``transforms.json`` camera poses, so ``load_3D_points=True`` can use it directly
via ``ply_file_path``.

Per-frame backprojections and tree-merge intermediates are written to disk so
large datasets do not need to keep all clouds in memory.
"""

from __future__ import annotations

import gc
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
import tyro
from PIL import Image


def _load_depth(path: Path) -> np.ndarray:
    """Load a depth map from ``.npy`` / ``.npz`` or an image file."""
    suffix = path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        depth = np.load(path).astype(np.float32)
    else:
        depth = np.array(Image.open(path), dtype=np.float32)

    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth


def _load_color(path: Path) -> Optional[np.ndarray]:
    """Load an RGB image as float32 in [0, 1], or None if missing."""
    if not path.exists():
        return None
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _frame_intrinsics(contents: dict, frame: dict) -> np.ndarray:
    """Build a 3x3 intrinsics matrix, preferring frame-level values."""
    fl_x = float(frame.get("fl_x", contents["fl_x"]))
    fl_y = float(frame.get("fl_y", contents.get("fl_y", fl_x)))
    cx = float(frame.get("cx", contents["cx"]))
    cy = float(frame.get("cy", contents["cy"]))
    return np.array([[fl_x, 0.0, cx], [0.0, fl_y, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _opengl_c2w_to_opencv_w2c(c2w_opengl: np.ndarray) -> np.ndarray:
    """Convert Nerfstudio / OpenGL c2w to OpenCV w2c for Open3D / projection.

    OpenGL cameras are Y-up / Z-back; OpenCV is Y-down / Z-forward.
    Flipping the local Y/Z axes keeps points in the same world frame as the
    original ``transform_matrix`` poses.
    """
    c2w = c2w_opengl.copy()
    c2w[:3, 1:3] *= -1
    return np.linalg.inv(c2w).astype(np.float32)


def _write_pointcloud(path: Path, pcd: o3d.t.geometry.PointCloud) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.t.io.write_point_cloud(str(path), pcd)


def _read_pointcloud(path: Path) -> o3d.t.geometry.PointCloud:
    pcd = o3d.t.io.read_point_cloud(str(path))
    if "positions" not in pcd.point:
        raise RuntimeError(f"Failed to read point cloud: {path}")
    return pcd


def _maybe_downsample(
    pcd: o3d.t.geometry.PointCloud,
    voxel_size: float,
    max_points: int,
) -> o3d.t.geometry.PointCloud:
    if pcd.point.positions.shape[0] > max_points:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd


def _load_transforms(dataset_path: Path) -> dict:
    transforms_path = dataset_path / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"No transforms.json found at {transforms_path}")
    with open(transforms_path, encoding="utf-8") as f:
        return json.load(f)


def tree_merge_pointclouds_on_disk(
    ply_paths: List[Path],
    merge_dir: Path,
    voxel_size: float = 0.03,
    max_points: int = 2_000_000,
) -> Path:
    """Pairwise merge PLYs from disk, writing each level to ``merge_dir``."""
    merge_dir.mkdir(parents=True, exist_ok=True)
    current = list(ply_paths)
    level = 0

    while len(current) > 1:
        level_dir = merge_dir / f"level_{level:03d}"
        level_dir.mkdir(parents=True, exist_ok=True)
        next_level: List[Path] = []
        print(f"Tree-merge level {level}: {len(current)} clouds")

        for i in range(0, len(current), 2):
            out_path = level_dir / f"merged_{i // 2:06d}.ply"
            if out_path.exists():
                print(f"  Reusing {out_path}")
                next_level.append(out_path)
                continue

            if i + 1 < len(current):
                left = _read_pointcloud(current[i])
                right = _read_pointcloud(current[i + 1])
                merged = left + right
                del left, right
                merged = _maybe_downsample(merged, voxel_size=voxel_size, max_points=max_points)
                _write_pointcloud(out_path, merged)
                n_points = int(merged.point.positions.shape[0])
                del merged
                gc.collect()
                print(f"  Wrote {out_path} ({n_points} points)")
            else:
                shutil.copy2(current[i], out_path)
                print(f"  Carried forward {current[i]} -> {out_path}")

            next_level.append(out_path)

        current = next_level
        level += 1

    return current[0]


def backproject_frame(
    dataset_path: Path,
    contents: dict,
    frame: dict,
    depth_unit_scale_factor: float,
    depth_max: float,
    stride: int,
    frame_voxel_size: Optional[float],
) -> Optional[o3d.t.geometry.PointCloud]:
    """Backproject one depth frame (geometry only)."""
    if "depth_file_path" not in frame:
        return None

    depth_path = dataset_path / frame["depth_file_path"]
    print(f"Backprojecting {depth_path}")

    depth = _load_depth(depth_path) * depth_unit_scale_factor
    depth[~np.isfinite(depth)] = 0.0
    depth[depth <= 0.0] = 0.0
    depth = np.ascontiguousarray(depth, dtype=np.float32)

    if not np.any(depth > 0.0):
        print(f"  Skipping frame with no valid depth: {depth_path}")
        return None

    c2w = np.array(frame["transform_matrix"], dtype=np.float64)
    w2c = _opengl_c2w_to_opencv_w2c(c2w)
    intrinsic = _frame_intrinsics(contents, frame)

    depth_image = o3d.t.geometry.Image(o3d.core.Tensor(depth))
    pcd = o3d.t.geometry.PointCloud.create_from_depth_image(
        depth_image,
        o3d.core.Tensor(intrinsic),
        o3d.core.Tensor(w2c),
        depth_scale=1.0,
        depth_max=depth_max,
        stride=stride,
        with_normals=False,
    )
    del depth, depth_image

    if pcd.point.positions.shape[0] == 0:
        print(f"  Skipping empty point cloud for {depth_path}")
        return None

    if frame_voxel_size is not None and frame_voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=frame_voxel_size)

    return pcd


def create_pointcloud_from_transforms(
    dataset_path: Path,
    cache_dir: Path,
    depth_unit_scale_factor: float = 0.001,
    voxel_size: float = 0.05,
    merge_voxel_size: float = 0.03,
    frame_voxel_size: Optional[float] = 0.05,
    max_points: int = 2_000_000,
    depth_max: float = 100.0,
    stride: int = 1,
) -> o3d.t.geometry.PointCloud:
    """Backproject depths to disk, then tree-merge from disk into one cloud."""
    contents = _load_transforms(dataset_path)

    frames_dir = cache_dir / "frames"
    merge_dir = cache_dir / "merge"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: List[Path] = []
    frames = contents["frames"]
    for idx, frame in enumerate(frames):
        if "depth_file_path" not in frame:
            continue

        out_path = frames_dir / f"frame_{idx:06d}.ply"
        if out_path.exists():
            print(f"Reusing cached frame {out_path}")
            frame_paths.append(out_path)
            continue

        pcd = backproject_frame(
            dataset_path=dataset_path,
            contents=contents,
            frame=frame,
            depth_unit_scale_factor=depth_unit_scale_factor,
            depth_max=depth_max,
            stride=stride,
            frame_voxel_size=frame_voxel_size,
        )
        if pcd is None:
            continue

        n_points = int(pcd.point.positions.shape[0])
        _write_pointcloud(out_path, pcd)
        print(f"  Saved {out_path} ({n_points} points)")
        del pcd
        gc.collect()
        frame_paths.append(out_path)

    if not frame_paths:
        raise RuntimeError("No valid point clouds could be generated from the dataset.")

    print(f"Merging {len(frame_paths)} cached frame point clouds...")
    merged_path = tree_merge_pointclouds_on_disk(
        frame_paths,
        merge_dir=merge_dir,
        voxel_size=merge_voxel_size,
        max_points=max_points,
    )

    pointcloud = _read_pointcloud(merged_path)
    pointcloud = pointcloud.voxel_down_sample(voxel_size=voxel_size)
    return pointcloud


def _project_points(
    positions: np.ndarray,
    w2c: np.ndarray,
    intrinsic: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points with OpenCV extrinsics. Returns u, v, z."""
    ones = np.ones((positions.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([positions.astype(np.float32), ones], axis=1)
    pts_cam = pts_h @ w2c.T
    z = pts_cam[:, 2]
    # Avoid divide-by-zero; invalid depths are filtered later.
    z_safe = np.where(z > 1e-6, z, 1e-6)
    u = intrinsic[0, 0] * (pts_cam[:, 0] / z_safe) + intrinsic[0, 2]
    v = intrinsic[1, 1] * (pts_cam[:, 1] / z_safe) + intrinsic[1, 2]
    return u, v, z


def colorize_pointcloud(
    dataset_path: Path,
    pcd: o3d.t.geometry.PointCloud,
    depth_unit_scale_factor: float = 0.001,
    depth_max: float = 100.0,
    depth_tolerance: float = 0.05,
    depth_tolerance_rel: float = 0.02,
) -> o3d.t.geometry.PointCloud:
    """Color points by projecting them into RGB (+ depth) frames and averaging."""
    contents = _load_transforms(dataset_path)
    positions = pcd.point.positions.numpy().astype(np.float32)
    n_points = positions.shape[0]
    color_sum = np.zeros((n_points, 3), dtype=np.float64)
    color_count = np.zeros((n_points,), dtype=np.int32)

    frames = [f for f in contents["frames"] if "depth_file_path" in f and "file_path" in f]
    print(f"Colorizing {n_points} points using {len(frames)} RGB-D frames...")

    for frame_idx, frame in enumerate(frames):
        image_path = dataset_path / frame["file_path"]
        depth_path = dataset_path / frame["depth_file_path"]
        color = _load_color(image_path)
        if color is None:
            print(f"  Skipping missing RGB: {image_path}")
            continue

        depth = _load_depth(depth_path) * depth_unit_scale_factor
        depth[~np.isfinite(depth)] = 0.0
        depth[depth <= 0.0] = 0.0

        h, w = depth.shape[:2]
        if color.shape[0] != h or color.shape[1] != w:
            print(
                f"  Skipping size mismatch RGB {color.shape[:2]} vs depth {(h, w)}: {image_path}"
            )
            continue

        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        w2c = _opengl_c2w_to_opencv_w2c(c2w)
        intrinsic = _frame_intrinsics(contents, frame)
        u, v, z = _project_points(positions, w2c, intrinsic)

        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)
        valid = (
            (z > 0.0)
            & (z <= depth_max)
            & (ui >= 0)
            & (ui < w)
            & (vi >= 0)
            & (vi < h)
        )
        if not np.any(valid):
            continue

        valid_idx = np.flatnonzero(valid)
        ui_v = ui[valid_idx]
        vi_v = vi[valid_idx]
        z_v = z[valid_idx]
        measured = depth[vi_v, ui_v]
        tol = np.maximum(depth_tolerance, depth_tolerance_rel * z_v)
        consistent = (measured > 0.0) & (np.abs(measured - z_v) <= tol)
        if not np.any(consistent):
            continue

        hit_idx = valid_idx[consistent]
        sampled = color[vi_v[consistent], ui_v[consistent]]
        color_sum[hit_idx] += sampled
        color_count[hit_idx] += 1

        if (frame_idx + 1) % 50 == 0 or frame_idx + 1 == len(frames):
            observed = int(np.count_nonzero(color_count))
            print(
                f"  Frame {frame_idx + 1}/{len(frames)}: "
                f"{observed}/{n_points} points have color"
            )

        del color, depth
        gc.collect()

    colored = color_count > 0
    n_colored = int(np.count_nonzero(colored))
    if n_colored == 0:
        raise RuntimeError("No points received color from any RGB frame.")

    colors = np.zeros((n_points, 3), dtype=np.float32)
    colors[colored] = (color_sum[colored] / color_count[colored, None]).astype(np.float32)
    # Unobserved points stay black; splatfacto can still optimize them.
    print(f"Colored {n_colored}/{n_points} points "
          f"({100.0 * n_colored / n_points:.1f}%)")

    pcd.point["colors"] = o3d.core.Tensor(colors)
    return pcd


def _update_transforms_ply_path(dataset_path: Path, output_name: str) -> None:
    transforms_path = dataset_path / "transforms.json"
    with open(transforms_path, encoding="utf-8") as f:
        contents = json.load(f)
    contents["ply_file_path"] = output_name
    with open(transforms_path, "w", encoding="utf-8") as f:
        json.dump(contents, f, indent=4)
    print(f"Updated {transforms_path} with ply_file_path={output_name}")


@dataclass
class Args:
    """Create / colorize an initialization point cloud from dataset depth + RGB."""

    data: Path
    """Dataset directory, or path to transforms.json."""

    colorize: bool = False
    """If True, colorize an existing point cloud from RGB (skip backprojection)."""
    input_name: str = "sparse_pc.ply"
    """Input PLY to colorize when --colorize is set."""
    output_name: str = "sparse_pc.ply"
    """Output PLY filename written into the dataset directory."""

    depth_unit_scale_factor: float = 0.001
    """Scale raw depth values to meters (0.001 for millimeter depth)."""
    cache_dir: Optional[Path] = None
    """Directory for per-frame and merge intermediates. Default: <data>/init_pc_cache."""
    keep_cache: bool = True
    """Keep per-frame / merge cache after writing the final PLY."""
    voxel_size: float = 0.05
    """Final voxel downsample size after merging (backproject only)."""
    merge_voxel_size: float = 0.03
    """Voxel size used while tree-merging intermediate clouds."""
    frame_voxel_size: Optional[float] = 0.05
    """Voxel downsample each frame before writing to disk. Set null to disable."""
    max_points: int = 2_000_000
    """Downsample during merge when a cloud exceeds this many points."""
    depth_max: float = 100.0
    """Maximum depth (meters) to keep during backprojection / color checks."""
    stride: int = 4
    """Pixel stride when converting depth to points (higher = less memory)."""
    depth_tolerance: float = 0.05
    """Absolute depth consistency tolerance in meters for colorization."""
    depth_tolerance_rel: float = 0.02
    """Relative depth consistency tolerance (fraction of z) for colorization."""
    update_transforms: bool = True
    """If True, set transforms.json ply_file_path to the output PLY."""


def _resolve_dataset_path(data: Path) -> Path:
    """Accept a dataset directory or a path to transforms.json."""
    path = data.expanduser().resolve()
    if path.is_file() and path.name == "transforms.json":
        return path.parent
    if path.is_dir():
        return path
    raise ValueError(
        f"Expected a dataset directory or transforms.json, got: {data}"
    )


def main(args: Args) -> None:
    dataset_path = _resolve_dataset_path(args.data)
    output_path = dataset_path / args.output_name

    if args.colorize:
        input_path = dataset_path / args.input_name
        if not input_path.exists():
            raise FileNotFoundError(
                f"Input point cloud not found: {input_path}. "
                "Run without --colorize first to backproject depth."
            )
        print(f"Loading {input_path} for colorization...")
        pointcloud = _read_pointcloud(input_path)
        pointcloud = colorize_pointcloud(
            dataset_path=dataset_path,
            pcd=pointcloud,
            depth_unit_scale_factor=args.depth_unit_scale_factor,
            depth_max=args.depth_max,
            depth_tolerance=args.depth_tolerance,
            depth_tolerance_rel=args.depth_tolerance_rel,
        )
    else:
        cache_dir = (
            args.cache_dir.expanduser().resolve()
            if args.cache_dir is not None
            else dataset_path / "init_pc_cache"
        )
        pointcloud = create_pointcloud_from_transforms(
            dataset_path=dataset_path,
            cache_dir=cache_dir,
            depth_unit_scale_factor=args.depth_unit_scale_factor,
            voxel_size=args.voxel_size,
            merge_voxel_size=args.merge_voxel_size,
            frame_voxel_size=args.frame_voxel_size,
            max_points=args.max_points,
            depth_max=args.depth_max,
            stride=args.stride,
        )
        if not args.keep_cache and cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"Removed cache directory {cache_dir}")
        else:
            print(f"Cache kept at {cache_dir}")

    num_points = int(pointcloud.point.positions.shape[0])
    has_colors = "colors" in pointcloud.point
    print(
        f"Writing {num_points} points"
        f"{' with colors' if has_colors else ' (geometry only)'} to {output_path}"
    )
    o3d.t.io.write_point_cloud(str(output_path), pointcloud)
    del pointcloud
    gc.collect()

    if args.update_transforms:
        _update_transforms_ply_path(dataset_path, args.output_name)


def entrypoint() -> None:
    """CLI entrypoint registered in pyproject.toml."""
    main(tyro.cli(Args))


if __name__ == "__main__":
    entrypoint()
