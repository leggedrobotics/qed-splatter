import torch
import cv2
import numpy as np


def get_laplacian_rgb(img: torch.Tensor):
    with torch.no_grad():
        if img.dim() == 4:
            img_np = img.squeeze(0).cpu().numpy()
        elif img.dim() == 3:
            img_np = img.cpu().numpy()
        else:
            raise ValueError("Input image must be 3D or 4D tensor.")
        blurred = np.stack([
            cv2.GaussianBlur(img_np[..., c], (0, 0), 1.0)
            for c in range(3)
        ], axis=-1)

        laplacian = np.stack([
            cv2.Laplacian(blurred[..., c], cv2.CV_32F, ksize=3)
            for c in range(3)
        ], axis=-1)

        # Compute L2 norm across channels
        laplacian_norm = np.sqrt(np.sum(laplacian ** 2, axis=-1))
        laplacian_clamped = np.clip(laplacian_norm, 0.0, 1.0)

        return laplacian_clamped

def get_laplacian_depth(depth_img: torch.Tensor):
    with torch.no_grad():
        depth_np = depth_img.cpu().numpy()
        blurred = cv2.GaussianBlur(depth_np, (0, 0), 1.0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
        laplacian_norm = np.abs(laplacian)  # [H, W]
        laplacian_clamped = np.clip(laplacian_norm, 0.0, 1.0)
        return laplacian_clamped