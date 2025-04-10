from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type, Union

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

import torch
from nerfstudio.models.splatfacto import SplatfactoModelConfig, SplatfactoModel
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.misc import torch_compile
import torchvision.transforms.functional as TF
from qed_splatter.metrics import RGBMetrics, DepthMetrics

@torch_compile()
def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat


@dataclass
class QEDSplatterModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: QEDSplatterModel)
    depth_lambda: float = 0.2  # Weight for depth loss, found 0.2 to 0.3 to work well
    random_scale: float = 100.0  # Random scale for the gaussians, set to 1 if you enable scaling
    output_depth_during_training: bool = True


class QEDSplatterModel(SplatfactoModel):
    config: QEDSplatterModelConfig

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        """
        Populates the modules of the model.

        @return: None
        """
        super().populate_modules()

        self.rgb_metrics = RGBMetrics()
        self.depth_metrics = DepthMetrics()
        self.mse_loss = torch.nn.MSELoss()


    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """
        Computes the loss for the model.

        @param outputs: Dictionary containing the outputs of the model
        @param batch: Dictionary containing the batch data
        @param metrics_dict: Dictionary containing the metrics

        @return: Dictionary containing the loss
        """
        loss_dict = super().get_loss_dict(
            outputs=outputs, batch=batch, metrics_dict=metrics_dict
        )

        depth_out = outputs['depth']
        depth_batch = self.get_gt_img(batch['depth_image'])

        pred_img = outputs['rgb']
        gt_img = self.get_gt_img(batch['image']) #.clamp(min=10 / 255.0)

        if 'mask' in batch:
            mask = self.get_gt_img(batch['mask'])
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            depth_out = depth_out * mask
            depth_batch = depth_batch * mask

        # Create a validity mask: only consider finite values (i.e., exclude NaN, +Inf, -Inf)
        valid_mask = torch.isfinite(depth_out) & torch.isfinite(depth_batch)

        # Apply the mask
        valid_depth_out = depth_out[valid_mask]
        valid_depth_batch = depth_batch[valid_mask]

        if valid_depth_out.numel() > 0:  # Ensure we are not computing mean on an empty tensor
            loss = torch.abs(valid_depth_out - valid_depth_batch).mean()
        else:
            loss = torch.tensor(0.0, device=depth_out.device)  # Avoid NaN loss

        loss_dict['depth_loss'] = self.config.depth_lambda * loss

        return loss_dict

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """
        Computes the metrics for the model.

        @param outputs: Dictionary containing the outputs of the model
        @param batch: Dictionary containing the batch data

        @return: Dictionary containing the metrics
        """
        d = self._get_downscale_factor()
        if d > 1:
            # use torchvision to resize
            newsize = (batch["image"].shape[0] // d, batch["image"].shape[1] // d)
            gt_img = TF.resize(
                batch["image"].permute(2, 0, 1), newsize, antialias=None
            ).permute(1, 2, 0)

            if "depth_image" in batch:
                depth_size = (
                    batch["depth_image"].shape[0] // d,
                    batch["depth_image"].shape[1] // d,
                )
                sensor_depth_gt = TF.resize(
                    batch["depth_image"].permute(2, 0, 1), depth_size, antialias=None
                ).permute(1, 2, 0)
        else:
            gt_img = batch["image"]
            if "depth_image" in batch:
                sensor_depth_gt = batch["depth_image"]

        metrics_dict = {}
        gt_rgb = gt_img.to(self.device)  # RGB or RGBA image
        predicted_rgb = (
            outputs["rgb"][0, ...] if outputs["rgb"].dim() == 4 else outputs["rgb"]
        )

        with torch.no_grad():
            (psnr, ssim, lpips) = self.rgb_metrics(
                gt_rgb.permute(2, 0, 1).unsqueeze(0),
                predicted_rgb.permute(2, 0, 1).unsqueeze(0).to(self.device),
            )
            rgb_mse = self.mse_loss(gt_rgb.permute(2, 0, 1), predicted_rgb.permute(2, 0, 1))
            rgb_metrics = {
                "rgb_mse": float(rgb_mse),
                "rgb_psnr": float(psnr.item()),
                "rgb_ssim": float(ssim),
                "rgb_lpips": float(lpips),
            }
            metrics_dict.update(rgb_metrics)

        metrics_dict["gaussian_count"] = self.num_points


        with torch.no_grad():
            if "depth_image" in batch:
                predicted_depth = outputs["depth"]
                (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3) = self.depth_metrics(
                    predicted_depth.permute(2, 0, 1), sensor_depth_gt.permute(2, 0, 1)
                )

                depth_metrics = {
                    "depth_abs_rel": float(abs_rel.item()),
                    "depth_sq_rel": float(sq_rel.item()),
                    "depth_rmse": float(rmse.item()),
                    "depth_rmse_log": float(rmse_log.item()),
                    "depth_a1": float(a1.item()),
                    "depth_a2": float(a2.item()),
                    "depth_a3": float(a3.item()),
                }
                metrics_dict.update(depth_metrics)


        # track scales
        metrics_dict.update(
            {"avg_min_scale": torch.nanmean(torch.exp(self.scales[..., -1]))}
        )

        return metrics_dict

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Generates the outputs of the model given a camera.

        @param camera: Cameras object containing the camera parameters
        @return: Dictionary containing the outputs of the model
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+D"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        render, alpha, self.info = rasterization(
            means=means_crop,
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )
        if self.training and self.info["means2d"].requires_grad:
            self.info["means2d"].retain_grad()
        self.xys = self.info["means2d"]  # [1, N, 2]
        self.radii = self.info["radii"][0]  # [N]
        alpha = alpha[:, ...]

        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        # apply bilateral grid
        if self.config.use_bilateral_grid and self.training:
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                rgb = self._apply_bilateral_grid(rgb, camera.metadata["cam_idx"], H, W)

        if render_mode == "RGB+D":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        del render
        torch.cuda.empty_cache()

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
        }  # type: ignore
