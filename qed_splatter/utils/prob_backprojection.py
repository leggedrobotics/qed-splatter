import numpy as np
from scipy.sparse import dok_matrix
from nerfstudio.cameras.cameras import Cameras
from typing import List, Tuple
import torch

def laplacian_to_3dprob_field(
    camera_edgediff_map: List[Tuple[Cameras, np.ndarray]],
    resolution: float = 0.01,
    depth_range: Tuple[float, float] = (0.5, 3.0),
    num_depth_samples: int = 100,
    edge_threshold: float = 0.1,
) -> dok_matrix:
    """
    Backprojects edge pixels from multiple camera edge maps into a sparse 3D probability field.

    Args:
        camera_edgediff_map: List of (Cameras, edge_map_numpy) tuples.
        resolution: Size of each voxel in meters.
        depth_range: Near and far bounds for ray sampling.
        num_depth_samples: How many samples along each ray to use.
        edge_threshold: Threshold to consider an edge pixel as valid.

    Returns:
        sparse_volume: A scipy sparse dok_matrix representing the 3D probability field.
    """

    # World-space bounding box and grid config
    grid_origin = np.array([0.0, 0.0, 0.0])  # TODO: parameterize if needed
    grid_size = np.array([2.0, 2.0, 2.0])
    grid_dims = np.ceil(grid_size / resolution).astype(int)
    num_voxels = np.prod(grid_dims)
    sparse_volume = dok_matrix((num_voxels, 1), dtype=np.float32)

    def flatten_index(ix, iy, iz):
        return ix * grid_dims[1] * grid_dims[2] + iy * grid_dims[2] + iz

    for cam_idx, (camera, edge_image_np) in enumerate(camera_edgediff_map):
        # Threshold edge map
        edge_pixels = np.argwhere(edge_image_np > edge_threshold)  # [[u, v], ...]

        if len(edge_pixels) == 0:
            continue  # skip if no edges

        # Get ray origins and directions from the camera
        ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True)
        origins = ray_bundle.origins.cpu().numpy()  # shape (H, W, 3)
        directions = ray_bundle.directions.cpu().numpy()  # shape (H, W, 3)

        # Loop over each edge pixel
        for (u, v) in edge_pixels:
            origin = origins[u, v]
            direction = directions[u, v]

            # Sample points along the ray
            for depth in np.linspace(depth_range[0], depth_range[1], num_depth_samples):
                point = origin + depth * direction
                rel_point = point - grid_origin
                idx = np.floor(rel_point / resolution).astype(int)

                if np.all((0 <= idx) & (idx < grid_dims)):
                    flat_idx = flatten_index(*idx)
                    sparse_volume[flat_idx, 0] += 1.0  # simple voting

    return sparse_volume
