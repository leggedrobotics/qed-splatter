from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import numpy as np
import torch
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig, Nerfstudio


@dataclass
class QEDSplatterDataParserConfig(NerfstudioDataParserConfig):
    _target: Type = field(default_factory=lambda: QEDSplatterDataParser)
    depth_unit_scale_factor: float = 0.001
    # auto_scale_poses: bool = False  # If True, poses will be scaled to max extent of 1
    # center_method: Literal["poses", "focus", "none"] = "none"  # Centering method
    # orientation_method: Literal["pca", "up", "vertical", "none"] = "none"  # Orientation method


@dataclass
class QEDSplatterDataParser(Nerfstudio):
    config: QEDSplatterDataParserConfig

    def _load_3D_points(self, ply_file_path: Path, transform_matrix: torch.Tensor, scale_factor: float):
        """Load point positions and colors from a PLY file.

        Nerfstudio's default loader uses ``o3d.io.read_point_cloud``, which
        misreads colors from PLY files written by Open3D's tensor API with
        float colors in [0, 1]. We read colors via the tensor API when
        available, matching splatfacto's expected uint8 RGB format.
        """
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(ply_file_path))
        if len(pcd.points) == 0:
            return None

        points3D = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32))
        points3D = (
            torch.cat(
                (
                    points3D,
                    torch.ones_like(points3D[..., :1]),
                ),
                -1,
            )
            @ transform_matrix.T
        )
        points3D *= scale_factor
        points3D_rgb = self._load_ply_colors(ply_file_path, pcd, num_points=len(pcd.points))

        return {
            "points3D_xyz": points3D,
            "points3D_rgb": points3D_rgb,
        }

    @staticmethod
    def _load_ply_colors(ply_file_path: Path, pcd, num_points: int) -> torch.Tensor:
        """Return uint8 RGB colors for each point."""
        import open3d as o3d

        pcd_tensor = o3d.t.io.read_point_cloud(str(ply_file_path))
        if "colors" in pcd_tensor.point:
            colors = pcd_tensor.point.colors.numpy()
            if np.issubdtype(colors.dtype, np.floating):
                return torch.from_numpy((np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8))
            return torch.from_numpy(colors.astype(np.uint8))

        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            return torch.from_numpy((np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8))

        return torch.zeros((num_points, 3), dtype=torch.uint8)
