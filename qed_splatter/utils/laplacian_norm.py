import torch
import torch.nn.functional as F


def get_laplacian_rgb(img: torch.Tensor):
    # img: [H, W, 3] or [1, H, W, 3]
    if img.dim() == 4:
        img = img.squeeze(0)
    if img.shape[-1] != 3:
        raise ValueError("Input image must have 3 channels (RGB).")
    # [H, W, 3] -> [1, 3, H, W]
    img = img.permute(2, 0, 1).unsqueeze(0).float()
    device = img.device
    laplacian_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        device=device, dtype=img.dtype
    ).unsqueeze(0).unsqueeze(0)
    laplacian_kernel = laplacian_kernel.repeat(3, 1, 1, 1)
    laplacian = F.conv2d(img, laplacian_kernel, padding=1, groups=3)
    # Compute L2 norm across channels
    laplacian_norm = torch.linalg.vector_norm(laplacian, ord=2, dim=1)[0]
    # Zero borders
    laplacian_norm[:, 0] = 0
    laplacian_norm[:, -1] = 0
    laplacian_norm[0, :] = 0
    laplacian_norm[-1, :] = 0
    return laplacian_norm.clamp(0, 1).cpu().numpy()


def get_laplacian_depth(depth_img: torch.Tensor):
    # Accepts [H, W], [H, W, 1], [1, H, W], [1, H, W, 1], etc.
    # Squeeze all singleton dimensions except the last two (H, W)
    while depth_img.dim() > 2 and depth_img.shape[0] == 1:
        depth_img = depth_img.squeeze(0)
    if depth_img.dim() == 3 and depth_img.shape[-1] == 1:
        depth_img = depth_img.squeeze(-1)
    # Now should be [H, W]
    if depth_img.dim() != 2:
        raise ValueError(f"Depth image must be 2D after squeezing, got shape {depth_img.shape}")
    img = depth_img.unsqueeze(0).unsqueeze(0).float()
    device = img.device
    laplacian_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        device=device, dtype=img.dtype
    ).unsqueeze(0).unsqueeze(0)
    laplacian = F.conv2d(img, laplacian_kernel, padding=1)
    laplacian_norm = laplacian.abs()[0, 0]
    # Zero borders
    laplacian_norm[:, 0] = 0
    laplacian_norm[:, -1] = 0
    laplacian_norm[0, :] = 0
    laplacian_norm[-1, :] = 0
    return laplacian_norm.clamp(0, 1).cpu().numpy()
