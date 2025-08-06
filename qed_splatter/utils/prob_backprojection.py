import torch
from nerfstudio.cameras.cameras import Cameras


def laplacian_projection_to_world(
    camera: Cameras,
    depth: torch.Tensor,
    sample_vu: torch.Tensor
) -> torch.Tensor:
    device = camera.device
    depth = depth.squeeze(-1).to(device)

    v_grid, u_grid = sample_vu[..., 0], sample_vu[..., 1]
    depth = depth[v_grid, u_grid]

    x = (u_grid - camera.cx) * depth / camera.fx
    y = (v_grid - camera.cy) * depth / camera.fy
    cam_points = torch.stack([x, y, depth], dim=0).reshape(3, -1)  # [3, H*W]
    cam_points_h = torch.cat([cam_points, torch.ones(1, cam_points.shape[1], device=device)], dim=0)  # [4, H*W]

    cam_to_world = camera.camera_to_worlds.detach().clone()
    cam_to_world[0:3, 1:3] *= -1
    world_points = cam_to_world @ cam_points_h  # [3, H*W]
    return world_points.T  # [H*W, 3]
