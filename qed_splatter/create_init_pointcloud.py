"""Create a depth-backprojected point cloud for Gaussian initialization.

The output PLY is written in the same OpenGL / Nerfstudio world coordinates as
``transforms.json`` camera poses, so ``load_3D_points=True`` can use it directly
via ``ply_file_path``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import open3d as o3d
import tyro
from PIL import Image


def tree_merge_pointclouds(
    pointclouds: List[o3d.t.geometry.PointCloud],
    voxel_size: float = 0.03,
    max_points: int = 2_000_000,
) -> o3d.t.geometry.PointCloud:
    """Merge point clouds pairwise for O(log n) runtime."""
    merged = pointclouds
    while len(merged) > 1:
        next_level = []
        for i in range(0, len(merged), 2):
            if i + 1 < len(merged):
                pc = merged[i] + merged[i + 1]
                if pc.point.positions.shape[0] > max_points:
                    pc = pc.voxel_down_sample(voxel_size=voxel_size)
                next_level.append(pc)
            else:
                next_level.append(merged[i])
        merged = next_level
    return merged[0]


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
    image = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return image


def _frame_intrinsics(contents: dict, frame: dict) -> np.ndarray:
    """Build a 3x3 intrinsics matrix, preferring frame-level values."""
    fl_x = float(frame.get("fl_x", contents["fl_x"]))
    fl_y = float(frame.get("fl_y", contents.get("fl_y", fl_x)))
    cx = float(frame.get("cx", contents["cx"]))
    cy = float(frame.get("cy", contents["cy"]))
    return np.array([[fl_x, 0.0, cx], [0.0, fl_y, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _opengl_c2w_to_opencv_w2c(c2w_opengl: np.ndarray) -> np.ndarray:
    """Convert Nerfstudio / OpenGL c2w to OpenCV w2c for Open3D.

    OpenGL cameras are Y-up / Z-back; Open3D expects Y-down / Z-forward.
    Flipping the local Y/Z axes keeps points in the same world frame as the
    original ``transform_matrix`` poses.
    """
    c2w = c2w_opengl.copy()
    c2w[:3, 1:3] *= -1
    return np.linalg.inv(c2w).astype(np.float32)


def create_pointcloud_from_transforms(
    dataset_path: Path,
    depth_unit_scale_factor: float = 0.001,
    voxel_size: float = 0.05,
    merge_voxel_size: float = 0.03,
    max_points: int = 2_000_000,
    depth_max: float = 100.0,
    stride: int = 1,
) -> o3d.t.geometry.PointCloud:
    """Backproject depths from ``transforms.json`` into a merged point cloud."""
    transforms_path = dataset_path / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"No transforms.json found at {transforms_path}")

    with open(transforms_path, encoding="utf-8") as f:
        contents = json.load(f)

    frames = contents["frames"]
    pointclouds: List[o3d.t.geometry.PointCloud] = []

    for frame in frames:
        if "depth_file_path" not in frame:
            continue

        depth_path = dataset_path / frame["depth_file_path"]
        image_path = dataset_path / frame["file_path"]
        print(f"Backprojecting {depth_path}")

        depth = _load_depth(depth_path) * depth_unit_scale_factor
        # Invalid / missing depth: Open3D ignores zeros
        depth[~np.isfinite(depth)] = 0.0
        depth[depth <= 0.0] = 0.0
        depth = np.ascontiguousarray(depth, dtype=np.float32)

        if not np.any(depth > 0.0):
            print(f"  Skipping frame with no valid depth: {depth_path}")
            continue

        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        w2c = _opengl_c2w_to_opencv_w2c(c2w)
        intrinsic = _frame_intrinsics(contents, frame)

        depth_image = o3d.t.geometry.Image(o3d.core.Tensor(depth))
        intrinsic_tensor = o3d.core.Tensor(intrinsic)
        extrinsic_tensor = o3d.core.Tensor(w2c)

        color = _load_color(image_path)
        if color is not None and color.shape[:2] == depth.shape[:2]:
            color_image = o3d.t.geometry.Image(o3d.core.Tensor(color))
            rgbd = o3d.t.geometry.RGBDImage(color_image, depth_image)
            pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                intrinsic_tensor,
                extrinsic_tensor,
                depth_scale=1.0,
                depth_max=depth_max,
                stride=stride,
                with_normals=False,
            )
        else:
            pcd = o3d.t.geometry.PointCloud.create_from_depth_image(
                depth_image,
                intrinsic_tensor,
                extrinsic_tensor,
                depth_scale=1.0,
                depth_max=depth_max,
                stride=stride,
                with_normals=False,
            )

        if pcd.point.positions.shape[0] == 0:
            print(f"  Skipping empty point cloud for {depth_path}")
            continue

        pointclouds.append(pcd)

    if not pointclouds:
        raise RuntimeError("No valid point clouds could be generated from the dataset.")

    print(f"Merging {len(pointclouds)} point clouds...")
    pointcloud = tree_merge_pointclouds(
        pointclouds, voxel_size=merge_voxel_size, max_points=max_points
    )
    pointcloud = pointcloud.voxel_down_sample(voxel_size=voxel_size)
    return pointcloud


@dataclass
class Args:
    """Create an initialization point cloud from dataset depth maps."""

    data: Path
    """Dataset directory, or path to transforms.json."""

    depth_unit_scale_factor: float = 0.001
    """Scale raw depth values to meters (0.001 for millimeter depth)."""
    output_name: str = "sparse_pc.ply"
    """Output PLY filename written into the dataset directory."""
    voxel_size: float = 0.05
    """Final voxel downsample size after merging."""
    merge_voxel_size: float = 0.03
    """Voxel size used while tree-merging intermediate clouds."""
    max_points: int = 2_000_000
    """Downsample during merge when a cloud exceeds this many points."""
    depth_max: float = 100.0
    """Maximum depth (meters) to keep during backprojection."""
    stride: int = 1
    """Pixel stride when converting depth to points (1 = every pixel)."""
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
    pointcloud = create_pointcloud_from_transforms(
        dataset_path=dataset_path,
        depth_unit_scale_factor=args.depth_unit_scale_factor,
        voxel_size=args.voxel_size,
        merge_voxel_size=args.merge_voxel_size,
        max_points=args.max_points,
        depth_max=args.depth_max,
        stride=args.stride,
    )

    num_points = int(pointcloud.point.positions.shape[0])
    print(f"Writing {num_points} points to {output_path}")
    o3d.t.io.write_point_cloud(str(output_path), pointcloud)

    if args.update_transforms:
        transforms_path = dataset_path / "transforms.json"
        with open(transforms_path, encoding="utf-8") as f:
            contents = json.load(f)
        contents["ply_file_path"] = args.output_name
        with open(transforms_path, "w", encoding="utf-8") as f:
            json.dump(contents, f, indent=4)
        print(f"Updated {transforms_path} with ply_file_path={args.output_name}")


def entrypoint() -> None:
    """CLI entrypoint registered in pyproject.toml."""
    main(tyro.cli(Args))


if __name__ == "__main__":
    entrypoint()
