from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Type, Union, Literal, Optional, Tuple

from qed_splatter.utils.prob_backprojection import laplacian_to_3dprob_field, visualize_sparse_volume

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

from gsplat.strategy import DefaultStrategy, MCMCStrategy
from nerfstudio.cameras.camera_optimizers import CameraOptimizer
import torch
from nerfstudio.models.splatfacto import SplatfactoModelConfig, SplatfactoModel
from nerfstudio.cameras.cameras import Cameras
import torchvision.transforms.functional as TF
import cv2
from qed_splatter.metrics import RGBMetrics, DepthMetrics
from nerfstudio.utils.math import k_nearest_sklearn, random_quat_tensor
from nerfstudio.utils.spherical_harmonics import RGB2SH, num_sh_bases
from nerfstudio.model_components.lib_bilagrid import BilateralGrid
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.scene_box import OrientedBox
from pytorch_msssim import SSIM
from qed_splatter.strategies.pixel_wise_probability import PixelWiseProbStrategy
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from nerfstudio.utils.colors import get_color
import numpy as np
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)

from pathlib import Path
import torchvision.utils as vutils


if TYPE_CHECKING:
    from nerfstudio.pipelines.base_pipeline import Pipeline

# Pre-create the flip tensor for get_viewmat to avoid tracing issues
_FLIP_GSPLAT = torch.tensor([[[1, -1, -1]]], dtype=torch.float32)

def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    flip = _FLIP_GSPLAT.to(R.device, R.dtype)
    R = R * flip
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

    depth_lambda: float = 0.2
    """Weight for depth loss, found 0.2 to 0.3 to work well."""
    # random_scale: float = 100.0  # Random scale for the gaussians, set to 1 if you enable scaling
    output_depth_during_training: bool = True
    """If True, the model will output depth during training."""
    strategy: Literal["default", "mcmc", "pwp"] = "default"
    """The default strategy will be used if strategy is not specified. Other strategies, e.g. mcmc, can be used."""
    prob_add_every: int = 500
    """Number of frames after which to add new Gaussians based on the probability map."""


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
        if self.seed_points is not None and not self.config.random_init:
            means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
        else:
            means = torch.nn.Parameter((torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale)
        distances, _ = k_nearest_sklearn(means.data, 3)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        num_points = means.shape[0]
        quats = torch.nn.Parameter(random_quat_tensor(num_points))
        dim_sh = num_sh_bases(self.config.sh_degree)

        if (
                self.seed_points is not None
                and not self.config.random_init
                # We can have colors without points.
                and self.seed_points[1].shape[0] > 0
        ):
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                shs[:, 1:, 3:] = 0.0
            else:
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            features_dc = torch.nn.Parameter(shs[:, 0, :])
            features_rest = torch.nn.Parameter(shs[:, 1:, :])
        else:
            features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
            features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        self.rgb_metrics = RGBMetrics()
        self.depth_metrics = DepthMetrics()
        self.mse_loss = torch.nn.MSELoss()

        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)
        if self.config.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                num=self.num_train_data,
                grid_X=self.config.grid_shape[0],
                grid_Y=self.config.grid_shape[1],
                grid_W=self.config.grid_shape[2],
            )

        # Strategy for GS densification
        if self.config.strategy == "default":
            # Strategy for GS densification
            self.strategy = DefaultStrategy(
                prune_opa=self.config.cull_alpha_thresh,
                grow_grad2d=self.config.densify_grad_thresh,
                grow_scale3d=self.config.densify_size_thresh,
                grow_scale2d=self.config.split_screen_size,
                prune_scale3d=self.config.cull_scale_thresh,
                prune_scale2d=self.config.cull_screen_size,
                refine_scale2d_stop_iter=self.config.stop_screen_size_at,
                refine_start_iter=self.config.warmup_length,
                refine_stop_iter=self.config.stop_split_at,
                reset_every=self.config.reset_alpha_every * self.config.refine_every,
                refine_every=self.config.refine_every,
                pause_refine_after_reset=self.num_train_data + self.config.refine_every,
                absgrad=self.config.use_absgrad,
                revised_opacity=False,
                verbose=True,
            )
            self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)
        elif self.config.strategy == "mcmc":
            self.strategy = MCMCStrategy(
                cap_max=self.config.max_gs_num,
                noise_lr=self.config.noise_lr,
                refine_start_iter=self.config.warmup_length,
                refine_stop_iter=self.config.stop_split_at,
                refine_every=self.config.refine_every,
                min_opacity=self.config.cull_alpha_thresh,
                verbose=False,
            )
            self.strategy_state = self.strategy.initialize_state()
        elif self.config.strategy == "pwp":
            self.strategy = PixelWiseProbStrategy(
                prob_add_every=self.config.prob_add_every,
                prune_opa=self.config.cull_alpha_thresh,
                grow_grad2d=self.config.densify_grad_thresh,
                grow_scale3d=self.config.densify_size_thresh,
                grow_scale2d=self.config.split_screen_size,
                prune_scale3d=self.config.cull_scale_thresh,
                prune_scale2d=self.config.cull_screen_size,
                refine_scale2d_stop_iter=self.config.stop_screen_size_at,
                refine_start_iter=self.config.warmup_length,
                refine_stop_iter=self.config.stop_split_at,
                reset_every=self.config.reset_alpha_every * self.config.refine_every,
                refine_every=self.config.refine_every,
                pause_refine_after_reset=self.num_train_data + self.config.refine_every,
                absgrad=self.config.use_absgrad,
                revised_opacity=False,
                verbose=True
            )
            self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)
        else:
            raise ValueError(f"""Splatfacto does not support strategy {self.config.strategy}
                                     Currently, the supported strategies include default and mcmc.""")

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                self.step_cb,
                args=[training_callback_attributes.optimizers],
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.step_post_backwards,
                args=[training_callback_attributes.pipeline],
            )
        )
        return cbs


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
        gt_rgb = gt_img[..., :3].to(self.device)  # RGB or RGBA image
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

    def calculate_total_edge_diff(self, pipeline: Pipeline, step) -> List[Tuple[Cameras, np.ndarray]]:
        """
        Calculates the laplacian of the RGB and GT RGB images. Then subtracts the two to get the edge difference.
        We later use this to add gaussians to the model based on the edge difference.
        """
        datamanager = pipeline.datamanager
        assert hasattr(datamanager, "fixed_indices_eval_dataloader"), (
            "datamanager must have 'fixed_indices_eval_dataloader' attribute"
        )

        pipeline.eval()
        dataloader = datamanager.fixed_indices_eval_dataloader
        num_images = len(dataloader)
        CONSOLE.log(f"Calculating edge difference for step {step} with {num_images} images")

        diff_list = []

        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                MofNCompleteColumn(),
                transient=True,
        ) as progress:
            task = progress.add_task("[green]Calculating Edge Diff", total=num_images)
            idx = 0
            for camera, batch in dataloader:
                outputs = self.get_outputs_for_camera(camera=camera)
                diff_list.append((
                    camera,
                    self.calculate_edge_diff(
                        outputs=outputs,
                        batch=batch,
                    )
                ))

                progress.advance(task)
                idx += 1

        pipeline.train()

        # Return the edge differences
        return diff_list


    def calculate_edge_diff(self, outputs, batch) -> np.ndarray:
        pred_img = outputs['rgb']
        gt_img = self.get_gt_img(batch['image'])

        with torch.no_grad():
            pred_img_np = pred_img.squeeze(0).cpu().numpy()
            gt_img_np = gt_img.squeeze(0).cpu().numpy()

            # Apply Gaussian blur and Laplacian per channel
            blurred_pred = np.stack([
                cv2.GaussianBlur(pred_img_np[..., c], (0, 0), 1.0)
                for c in range(3)
            ], axis=-1)
            pred_laplacian = np.stack([
                cv2.Laplacian(blurred_pred[..., c], cv2.CV_32F, ksize=3)
                for c in range(3)
            ], axis=-1)

            # Compute L2 norm across channels
            pred_laplacian_norm = np.sqrt(np.sum(pred_laplacian ** 2, axis=-1))
            # Ensure gt_img_np is blurred and laplacian is computed as well
            blurred_gt = np.stack([
                cv2.GaussianBlur(gt_img_np[..., c], (0, 0), 1.0)
                for c in range(3)
            ], axis=-1)
            gt_laplacian = np.stack([
                cv2.Laplacian(blurred_gt[..., c], cv2.CV_32F, ksize=3)
                for c in range(3)
            ], axis=-1)
            gt_laplacian_norm = np.sqrt(np.sum(gt_laplacian ** 2, axis=-1))

            pred_laplacian_clamped = np.clip(pred_laplacian_norm, 0.0, 1.0)
            gt_laplacian_clamped = np.clip(gt_laplacian_norm, 0.0, 1.0)

            # Difference between the two Laplacians
            edge_diff = np.maximum(gt_laplacian_clamped - pred_laplacian_clamped, 0)

            if 'mask' in batch:
                # Set edge_diff to zero where mask is zero
                mask = self.get_gt_img(batch['mask']).squeeze(0).cpu().numpy()
                edge_diff *= mask

            return edge_diff

    def step_post_backwards(self, pipeline: Pipeline, step):
        # Note: Function is called step_post_backward in splatfacto, to avoid a signature mismatch, we rename it here
        assert step == self.step

        if self.step == 1500:
            total_edge_diff = self.calculate_total_edge_diff(pipeline, step)

            CONSOLE.log(f"Total edge difference calculated for step {step}, length: {len(total_edge_diff)}")

            sparse_volume = laplacian_to_3dprob_field(
                camera_edgediff_map=total_edge_diff,
            )

            CONSOLE.log(f"Sparse volume created for step {step}, shape: {sparse_volume.shape}")

            visualize_sparse_volume(
                sparse_volume,
                grid_origin=np.array([0.0, 0.0, 0.0]),
                resolution=0.001,
                grid_dims=np.ceil(np.array([2.0, 2.0, 2.0]) / 0.001).astype(int),
                min_threshold=0.1,
                max_points=50000
            )

        if isinstance(self.strategy, DefaultStrategy):
            self.strategy.step_post_backward(
                params=self.gauss_params,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=self.step,
                info=self.info,
                packed=False,
            )
        elif isinstance(self.strategy, MCMCStrategy):
            self.strategy.step_post_backward(
                params=self.gauss_params,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=self.info,
                lr=self.schedulers["means"].get_last_lr()[0],  # the learning rate for the "means" attribute of the GS
            )
        elif isinstance(self.strategy, PixelWiseProbStrategy): # TODO: Might need to be moved up as inheritance of defaultStrategy
            if self.step % self.config.prob_add_every == 0:
                total_edge_diff = self.calculate_total_edge_diff(pipeline, step)

                sparse_volume = laplacian_to_3dprob_field(
                    camera_edgediff_map=total_edge_diff,
                )

                visualize_sparse_volume(
                    sparse_volume,
                    grid_origin=np.array([0.0, 0.0, 0.0]),
                    resolution=0.01,
                    grid_dims=np.ceil(np.array([2.0, 2.0, 2.0]) / 0.01).astype(int),
                    min_threshold=0.1,
                    max_points=50000
                )

                #self.strategy.add_gaussians(
                #    total_edge_diff
                #)

            self.strategy.step_post_backward(
                params=self.gauss_params,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=self.step,
                info=self.info,
                packed=False,
                probability_map=False
            )
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

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
            absgrad=self.strategy.absgrad if (isinstance(self.strategy, DefaultStrategy) or
                                              isinstance(self.strategy, PixelWiseProbStrategy)) else False,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )
        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
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

        # Laplacian computation
        with torch.no_grad():
            rgb_np = rgb.squeeze(0).cpu().numpy()  # [H, W, 3]
            rgb_np = np.clip(rgb_np, 0.0, 1.0)
            if rgb_np.ndim == 3 and rgb_np.shape[2] == 3:
                # Convert to float32
                rgb_np = rgb_np.astype(np.float32)
                # Apply Gaussian blur and Laplacian per channel
                blurred = np.stack([
                    cv2.GaussianBlur(rgb_np[..., c], (0, 0), 1.0)
                    for c in range(3)
                ], axis=-1)
                rgb_laplacian = np.stack([
                    cv2.Laplacian(blurred[..., c], cv2.CV_32F, ksize=3)
                    for c in range(3)
                ], axis=-1)
                # Compute L2 norm across channels
                rgb_laplacian_norm = np.sqrt(np.sum(rgb_laplacian ** 2, axis=-1))  # [H, W]
                rgb_laplacian_clamped = np.clip(rgb_laplacian_norm, 0.0, 1.0)
                rgb_laplacian_tensor = torch.from_numpy(rgb_laplacian_clamped).to(rgb.device).float()
                rgb_laplacian_tensor = rgb_laplacian_tensor.unsqueeze(-1).repeat(1, 1, 3)  # [H, W, 3]
            else:
                rgb_laplacian_tensor = torch.zeros((rgb.shape[1], rgb.shape[2], 3), device=rgb.device)

            # Depth Laplacian computation
            if depth_im is not None:
                depth_np = depth_im.cpu().numpy().astype(np.float32)  # [H, W]
                depth_blurred = cv2.GaussianBlur(depth_np, (0, 0), 1.0)
                depth_laplacian = cv2.Laplacian(depth_blurred, cv2.CV_32F, ksize=3)
                depth_laplacian_norm = np.abs(depth_laplacian)  # [H, W]
                depth_laplacian_clamped = np.clip(depth_laplacian_norm, 0.0, 1.0)
                depth_laplacian_tensor = torch.from_numpy(depth_laplacian_clamped).to(rgb.device).float()
                depth_laplacian_tensor = depth_laplacian_tensor.unsqueeze(-1).repeat(1, 1, 3)  # [H, W, 3]
            else:
                depth_laplacian_tensor = torch.zeros((rgb.shape[1], rgb.shape[2], 3), device=rgb.device)

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
            "rgb_laplacian_norm": rgb_laplacian_tensor, # type: ignore
            "depth_laplacian_norm": depth_laplacian_tensor,  # type: ignore
        }  # type: ignore
