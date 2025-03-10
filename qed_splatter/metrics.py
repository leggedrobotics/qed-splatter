from __future__ import annotations
import torch
import numpy as np
from scipy.spatial import cKDTree
from torch import nn
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class PDMetrics(nn.Module):
    """
    Computation of error metrics between predicted and ground truth point clouds

    @param acc: accuracy threshold
    @param cmp: completeness threshold

    @return accuracy, completeness
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.acc = calculate_accuracy
        self.cmp = calculate_completeness

    @torch.no_grad()
    def forward(self, pred, gt):
        pred_points = np.asarray(pred.points)
        gt_points = np.asarray(gt.points)
        acc_score = self.acc(pred_points, gt_points)
        cmp_score = self.cmp(pred_points, gt_points)

        return (acc_score, cmp_score)


def calculate_accuracy(reconstructed_points, reference_points, percentile=90):
    """
    Calucate accuracy: What percentage of the reconstructed point cloud is within a specific distance
    of the reference point cloud.

    @param reconstructed_points: reconstructed point cloud
    @param reference_points: reference point cloud

    @return accuracy
    """
    tree = cKDTree(reference_points)
    distances, _ = tree.query(reconstructed_points)
    return np.percentile(distances, percentile)


def calculate_completeness(reconstructed_points, reference_points, threshold=0.05):
    """
    Calucate completeness: What percentage of the reference point cloud is within a specific distance
    of the reconstructed point cloud.

    @param reconstructed_points: reconstructed point cloud
    @param reference_points: reference point cloud

    @return completeness
    """
    tree = cKDTree(reconstructed_points)
    distances, _ = tree.query(reference_points)
    within_threshold = np.sum(distances < threshold) / len(distances)
    return within_threshold * 100


def mean_angular_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Calculate the mean angular error between two sets of vectors

    @param pred: predicted vectors [B, C]
    @param gt: ground truth vectors [B, C]

    @return mean angular error
    """
    dot_products = torch.sum(gt * pred, dim=1)  # over the C dimension
    # Clamp the dot product to ensure valid cosine values (to avoid nans)
    dot_products = torch.clamp(dot_products, -1.0, 1.0)
    # Calculate the angle between the vectors (in radians)
    mae = torch.acos(dot_products)
    return mae


class RGBMetrics(nn.Module):
    """
    Computation of error metrics between predicted and ground truth RGB images

    @return psnr, ssim, lpips
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=11)
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    @torch.no_grad()
    def forward(self, pred, gt):
        self.device = pred.device
        self.psnr.to(self.device)
        self.ssim.to(self.device)
        self.lpips.to(self.device)

        # Ensure tensors are float and scaled properly
        pred = pred.float() / 255.0 if pred.dtype == torch.uint8 else pred
        gt = gt.float() / 255.0 if gt.dtype == torch.uint8 else gt

        psnr_score = self.psnr(pred, gt)
        ssim_score = self.ssim(pred, gt)
        lpips_score = self.lpips(pred, gt)

        return (psnr_score, ssim_score, lpips_score)


class DepthMetrics(nn.Module):
    """
    Computation of error metrics between predicted and ground truth depth images

    @return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
    """

    def __init__(self, tolerance: float = 0.1, **kwargs):
        self.tolerance = tolerance
        super().__init__()

    @torch.no_grad()
    def forward(self, pred, gt):
        device = pred.device
        gt = gt.to(device)

        # Filter valid pixels
        valid_mask = torch.isfinite(pred) & torch.isfinite(gt)  & (gt > self.tolerance)

        if valid_mask.sum() == 0:
            return (
                torch.tensor(float('nan'), device=device),
                torch.tensor(float('nan'), device=device),
                torch.tensor(float('nan'), device=device),
                torch.tensor(float('nan'), device=device),
                torch.tensor(float('nan'), device=device),
                torch.tensor(float('nan'), device=device),
                torch.tensor(float('nan'), device=device),
            )

        pred, gt = pred[valid_mask], gt[valid_mask]

        thresh = torch.max((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).float().mean()
        a2 = (thresh < 1.25 ** 2).float().mean()
        a3 = (thresh < 1.25 ** 3).float().mean()
        rmse = torch.sqrt(((gt - pred) ** 2).mean())
        rmse_log = torch.sqrt(((torch.log(gt) - torch.log(pred)) ** 2).nanmean())
        abs_rel = (torch.abs(gt - pred) / gt).mean()
        sq_rel = ((gt - pred) ** 2 / gt).mean()

        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3