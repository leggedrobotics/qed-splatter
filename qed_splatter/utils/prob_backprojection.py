import numpy as np
from collections import defaultdict
from typing import List, Tuple
import torch
from scipy.sparse import dok_matrix
from nerfstudio.cameras.cameras import Cameras
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn


def laplacian_to_3dprob_field(
        camera_edgediff_map: List[Tuple[Cameras, np.ndarray]],
        resolution: float = 0.01,
        depth_range: Tuple[float, float] = (0.5, 3.0),
        num_depth_samples: int = 100,
        edge_threshold: float = 0.1,
        vote_threshold: float = 1.5
) -> dok_matrix:
    grid_origin = np.array([0.0, 0.0, 0.0])
    grid_size = np.array([2.0, 2.0, 2.0])
    grid_dims = np.ceil(grid_size / resolution).astype(int)

    # Use regular dict for faster write performance
    voxel_votes = defaultdict(float)

    depths = np.linspace(depth_range[0], depth_range[1], num_depth_samples).astype(np.float32)

    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=True,
    ) as progress:
        task = progress.add_task("[blue]Backprojecting Edges", total=len(camera_edgediff_map))

        for cam_idx, (camera, edge_image_np) in enumerate(camera_edgediff_map):
            edge_coords = np.argwhere(edge_image_np > edge_threshold)
            if edge_coords.size == 0:
                progress.advance(task)
                continue

            edge_strengths = edge_image_np[edge_coords[:, 0], edge_coords[:, 1]].astype(np.float32)

            ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True)
            origins = ray_bundle.origins.cpu().numpy()
            directions = ray_bundle.directions.cpu().numpy()

            origins_edge = origins[edge_coords[:, 0], edge_coords[:, 1]]  # (N, 3)
            dirs_edge = directions[edge_coords[:, 0], edge_coords[:, 1]]  # (N, 3)

            # Vectorized sampling: (N, D, 3)
            points = origins_edge[:, None, :] + depths[None, :, None] * dirs_edge[:, None, :]

            # Convert to voxel indices
            rel_points = points - grid_origin[None, None, :]
            voxel_indices = np.floor(rel_points / resolution).astype(int)  # (N, D, 3)

            # Mask valid voxels only
            mask = np.all((voxel_indices >= 0) & (voxel_indices < grid_dims), axis=2)
            for i in range(voxel_indices.shape[0]):
                vote = edge_strengths[i]
                for j in range(num_depth_samples):
                    if not mask[i, j]:
                        continue
                    key = tuple(voxel_indices[i, j])
                    voxel_votes[key] += vote

            progress.advance(task)

    # Convert dict to dok_matrix
    num_voxels = np.prod(grid_dims)
    sparse_volume = dok_matrix((num_voxels, 1), dtype=np.float32)

    def flatten_index(ix, iy, iz):
        return ix * grid_dims[1] * grid_dims[2] + iy * grid_dims[2] + iz

    for (ix, iy, iz), value in voxel_votes.items():
        if value < vote_threshold:
            continue  # Filter out low-vote voxels
        flat_idx = flatten_index(ix, iy, iz)
        sparse_volume[flat_idx, 0] = value

    return sparse_volume


import numpy as np
import matplotlib.pyplot as plt

def visualize_sparse_volume(
    sparse_volume,
    grid_origin,
    resolution,
    grid_dims,
    min_threshold=0.0,
    max_points=100_000
):
    """
    Visualizes the sparse 3D volume as a 3D scatter plot.

    Args:
        sparse_volume: dok_matrix with shape (num_voxels, 1)
        grid_origin: np.array of shape (3,) specifying world-space origin
        resolution: voxel resolution (float)
        grid_dims: np.array of shape (3,) with voxel grid dimensions
        min_threshold: minimum value to plot a point
        max_points: max number of points to visualize (for performance)
    """
    indices, values = zip(*[(k[0], v) for k, v in sparse_volume.items() if v > min_threshold])

    if len(indices) == 0:
        print("No non-zero voxels to visualize.")
        return

    if len(indices) > max_points:
        print(f"Clipping to {max_points} points for performance.")
        sampled = np.random.choice(len(indices), max_points, replace=False)
        indices = [indices[i] for i in sampled]
        values = [values[i] for i in sampled]

    indices = np.array(indices)
    values = np.array(values)

    # Convert flat indices to 3D indices
    ix = indices // (grid_dims[1] * grid_dims[2])
    iy = (indices % (grid_dims[1] * grid_dims[2])) // grid_dims[2]
    iz = indices % grid_dims[2]

    # Map to world coordinates
    xs = grid_origin[0] + ix * resolution
    ys = grid_origin[1] + iy * resolution
    zs = grid_origin[2] + iz * resolution

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(xs, ys, zs, c=values, cmap='viridis', marker='o', s=1)
    plt.colorbar(scatter, label='Accumulated Weight')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Sparse 3D Probability Volume")
    plt.tight_layout()
    plt.show()