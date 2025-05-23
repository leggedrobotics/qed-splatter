from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type, Union, Optional, Tuple, Literal
import cv2

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

import torch
import torch.nn.functional as F
from gsplat.cuda_legacy._torch_impl import quat_to_rotmat
from nerfstudio.models.splatfacto import SplatfactoModelConfig, SplatfactoModel
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.misc import torch_compile
import torchvision.transforms.functional as TF
from qed_splatter.metrics import RGBMetrics, DepthMetrics
from torch import Tensor
from gsplat import rasterize_gaussians
import random
from qed_splatter.utils.knn import knn_sk
from qed_splatter.utils.camera_utils import get_colored_points_from_depth, project_pix

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

def pcd_to_normal(xyz: Tensor):
    hd, wd, _ = xyz.shape
    bottom_point = xyz[..., 2:hd, 1 : wd - 1, :]
    top_point = xyz[..., 0 : hd - 2, 1 : wd - 1, :]
    right_point = xyz[..., 1 : hd - 1, 2:wd, :]
    left_point = xyz[..., 1 : hd - 1, 0 : wd - 2, :]
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    xyz_normal = torch.nn.functional.pad(
        xyz_normal.permute(2, 0, 1), (1, 1, 1, 1), mode="constant"
    ).permute(1, 2, 0)
    return xyz_normal

def normal_from_depth_image(
    depths: Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_size: tuple,
    c2w: Tensor,
    device: torch.device,
    smooth: bool = False,
):
    """estimate normals from depth map"""
    if smooth:
        if torch.count_nonzero(depths) > 0:
            print("Input depth map contains 0 elements, skipping smoothing filter")
        else:
            kernel_size = (9, 9)
            depths = torch.from_numpy(
                cv2.GaussianBlur(depths.cpu().numpy(), kernel_size, 0)
            ).to(device)
    means3d, _ = get_means3d_backproj(depths, fx, fy, cx, cy, img_size, c2w, device)
    means3d = means3d.view(img_size[1], img_size[0], 3)
    normals = pcd_to_normal(means3d)
    return normals


def get_camera_coords(img_size: tuple, pixel_offset: float = 0.5) -> Tensor:
    """Generates camera pixel coordinates [W,H]

    Returns:
        stacked coords [H*W,2] where [:,0] corresponds to W and [:,1] corresponds to H
    """

    # img size is (w,h)
    image_coords = torch.meshgrid(
        torch.arange(img_size[0]),
        torch.arange(img_size[1]),
        indexing="xy",  # W = u by H = v
    )
    image_coords = (
        torch.stack(image_coords, dim=-1) + pixel_offset
    )  # stored as (x, y) coordinates
    image_coords = image_coords.view(-1, 2)
    image_coords = image_coords.float()

    return image_coords

def get_means3d_backproj(
    depths: Tensor,
    fx: float,
    fy: float,
    cx: int,
    cy: int,
    img_size: tuple,
    c2w: Tensor,
    device: torch.device,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, List]:
    """Backprojection using camera intrinsics and extrinsics

    image_coords -> (x,y,depth) -> (X, Y, depth)

    Returns:
        Tuple of (means: Tensor, image_coords: Tensor)
    """

    if depths.dim() == 3:
        depths = depths.view(-1, 1)
    elif depths.shape[-1] != 1:
        depths = depths.unsqueeze(-1).contiguous()
        depths = depths.view(-1, 1)
    if depths.dtype != torch.float:
        depths = depths.float()
        c2w = c2w.float()
    if c2w.device != device:
        c2w = c2w.to(device)

    image_coords = get_camera_coords(img_size)
    image_coords = image_coords.to(device)  # note image_coords is (H,W)

    # TODO: account for skew / radial distortion
    means3d = torch.empty(
        size=(img_size[0], img_size[1], 3), dtype=torch.float32, device=device
    ).view(-1, 3)
    means3d[:, 0] = (image_coords[:, 0] - cx) * depths[:, 0] / fx  # x
    means3d[:, 1] = (image_coords[:, 1] - cy) * depths[:, 0] / fy  # y
    means3d[:, 2] = depths[:, 0]  # z

    if mask is not None:
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, device=depths.device)
        means3d = means3d[mask]
        image_coords = image_coords[mask]

    if c2w is None:
        c2w = torch.eye((means3d.shape[0], 4, 4), device=device)

    # to world coords
    means3d = means3d @ torch.linalg.inv(c2w[..., :3, :3]) + c2w[..., :3, 3]
    return means3d, image_coords

def scale_rot_to_inv_cov3d(scale, quat, return_sqrt=False):
    assert scale.shape[-1] == 3, scale.shape
    assert quat.shape[-1] == 4, quat.shape
    assert scale.shape[:-1] == quat.shape[:-1], (scale.shape, quat.shape)
    scale = 1.0 / scale.clamp(min=1e-3)
    R = quat_to_rotmat(quat)  # (..., 3, 3)
    M = R * scale[..., None, :]  # (..., 3, 3)
    if return_sqrt:
        return M
    return M @ M.transpose(-1, -2)  # (..., 3, 3)


@dataclass
class QEDSplatterModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: QEDSplatterModel)
    predict_normals: bool = True
    depth_lambda: float = 0.2  # Weight for depth loss, found 0.2 to 0.3 to work well
    random_scale: float = 1.0  # Random scale for the gaussians, set to 1 if you enable scaling
    output_depth_during_training: bool = True


class QEDSplatterModel(SplatfactoModel):
    config: QEDSplatterModelConfig

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.camera = None

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

    def get_outputs(
        self, camera: Cameras
    ) -> Dict[str, Union[torch.Tensor, List[Tensor]]]:
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
                    int(camera.width.item()),
                    int(camera.height.item()),
                    self.background_color,
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

        colors_crop = torch.cat(
            (features_dc_crop[:, None, :], features_rest_crop), dim=1
        )

        BLOCK_WIDTH = (
            16  # this controls the tile size of rasterization, 16 is a good default
        )
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

        render_mode = "RGB+D"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(
                self.step // self.config.sh_degree_interval, self.config.sh_degree
            )
        else:
            colors_crop = torch.sigmoid(colors_crop)
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
        self.depths = self.info["depths"]
        self.conics = self.info["conics"]
        self.num_tiles_hit = self.info["tiles_per_gauss"]

        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        # visible gaussians
        self.vis_indices = torch.where(self.radii > 0)[0]

        if render_mode == "RGB+D":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(
                alpha > 0, depth_im, depth_im.detach().max()
            ).squeeze(0)
        else:
            depth_im = None

        normals_im = torch.full(rgb.shape, 0.0)
        if self.config.predict_normals:
            quats_crop = quats_crop / quats_crop.norm(dim=-1, keepdim=True)
            normals = F.one_hot(
                torch.argmin(scales_crop, dim=-1), num_classes=3
            ).float()
            rots = quat_to_rotmat(quats_crop)
            normals = torch.bmm(rots, normals[:, :, None]).squeeze(-1)
            normals = F.normalize(normals, dim=1)
            viewdirs = (
                -means_crop.detach() + camera.camera_to_worlds.detach()[..., :3, 3]
            )
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            dots = (normals * viewdirs).sum(-1)
            negative_dot_indices = dots < 0
            normals[negative_dot_indices] = -normals[negative_dot_indices]
            # update parameter group normals
            self.gauss_params["normals"] = normals
            # convert normals from world space to camera space
            normals = normals @ camera.camera_to_worlds.squeeze(0)[:3, :3]

            xys = self.xys[0, ...].detach()

            normals_im: Tensor = rasterize_gaussians(  # type: ignore
                xys,
                self.depths[0, ...],
                self.radii,
                self.conics[0, ...],
                self.num_tiles_hit[0, ...],
                normals,
                torch.sigmoid(opacities_crop),
                H,
                W,
                BLOCK_WIDTH,
            )
            # convert normals from [-1,1] to [0,1]
            normals_im = normals_im / normals_im.norm(dim=-1, keepdim=True)
            normals_im = (normals_im + 1) / 2

        surface_normal = normal_from_depth_image(
            depths=depth_im.detach(),
            fx=camera.fx.item(),
            fy=camera.fy.item(),
            cx=camera.cx.item(),
            cy=camera.cy.item(),
            img_size=(
                W,
                H
            ),
            c2w=torch.eye(4, dtype=torch.float, device=depth_im.device),
            device=self.device,
            smooth=False,
        )
        surface_normal = surface_normal @ torch.diag(
            torch.tensor([1, -1, -1], device=depth_im.device, dtype=depth_im.dtype)
        )
        surface_normal = (1 + surface_normal) / 2

        return {
            "rgb": rgb.squeeze(0),
            "depth": depth_im,
            "normal": normals_im,  # predicted normal from gaussians
            "surface_normal": surface_normal,  # normal from surface / depth
            "accumulation": alpha.squeeze(0),
            "background": background,
        }

    def get_closest_gaussians(self, samples) -> torch.Tensor:
        """Get closest gaussians to samples

        Args:
            samples: tensor of 3d point samples

        Returns:
            knn gaussians
        """
        closest_gaussians = knn_sk(
            x=self.means.data.to("cuda"),
            y=samples.to("cuda"),
            k=16,
        )
        return closest_gaussians

    def get_density(
        self,
        sdf_samples: Tensor,
        closest_gaussians: Optional[Tensor] = None,
        vis_indices: Optional[Tensor] = None,
    ):
        """Estimate current density at sample points based on current gaussian distributions

        Args:
            sdf_samples: current point samples
            closest_gaussians: closest knn gaussians per current point sample
            vis_indices: visibility mask

        Returns:
            densities
        """
        if closest_gaussians is None:
            closest_gaussians = self.get_closest_gaussians(samples=sdf_samples)
        closest_gaussians_idx = closest_gaussians
        closest_gaussian_centers = self.means[closest_gaussians]

        closest_gaussian_inv_scaled_rotation = scale_rot_to_inv_cov3d(
            scale=torch.exp(self.scales[closest_gaussians_idx]),
            quat=self.quats[closest_gaussians_idx],
            return_sqrt=True,
        )  # sigma^-1
        closest_gaussian_opacities = torch.sigmoid(
            self.opacities[closest_gaussians_idx]
        )

        # Compute the density field as a sum of local gaussian opacities
        # (num_samples, knn, 3)
        dist = sdf_samples[:, None, :] - closest_gaussian_centers
        # (num_samples, knn, 3, 1)
        man_distance = (
            closest_gaussian_inv_scaled_rotation.transpose(-1, -2) @ dist[..., None]
        )
        # Mahalanobis distance
        # (num_samples, knn)
        neighbor_opacities = (
            (man_distance[..., 0] * man_distance[..., 0])
            .sum(dim=-1)
            .clamp(min=0.0, max=1e8)
        )
        # (num_samples, knn)
        neighbor_opacities = closest_gaussian_opacities[..., 0] * torch.exp(
            -1.0 / 2 * neighbor_opacities
        )
        densities = neighbor_opacities.sum(dim=-1)  # (num_samples,)

        # BUG: this seems to be quite sensitive to the EPS
        density_mask = densities >= 1.0
        densities[density_mask] = densities[density_mask] / (
            densities[density_mask].detach() + 1e-5
        )
        opacity_min_clamp = 1e-4
        clamped_densities = densities.clamp(min=opacity_min_clamp)

        return clamped_densities

    def get_sdf(
        self,
        sdf_samples: Tensor,
        closest_gaussians: Optional[Tensor] = None,
        vis_indices: Optional[Tensor] = None,
    ) -> Tensor:
        """Estimate current sdf values at sample points based on current gaussian distributions

        Args:
            sdf_samples: current point samples
            closest_gaussians: closest knn gaussians per current point sample
            vis_indices: visibility mask

        Returns:
            sdf values
        """
        densities = self.get_density(
            sdf_samples=sdf_samples,
            closest_gaussians=closest_gaussians,
            vis_indices=vis_indices,
        )
        sdf_values = 1 * torch.sqrt(-2.0 * torch.log(densities))
        return sdf_values

    @torch.no_grad()
    def compute_level_surface_points(
            self,
            camera: Cameras,
            num_samples: int,
            mask: Optional[Tensor] = None,
            surface_levels: Tuple[float, float, float] = (0.1, 0.3, 0.5),
            return_normal: Literal[
                "analytical", "closest_gaussian", "average"
            ] = "closest_gaussian",
    ) -> Tensor:
        """Compute level surface intersections and their normals

        Args:
            camera: current camera object to find surface intersections
            num_samples: number of samples per camera to target
            mask: optional mask per camera
            surface_levels: surface levels to compute
            return_normal: normal return mode

        Returns:
            level surface intersection points, normals
        """
        c2w = camera.camera_to_worlds.squeeze(0)
        c2w = c2w @ torch.diag(
            torch.tensor([1, -1, -1, 1], device=c2w.device, dtype=c2w.dtype)
        )
        outputs = self.get_outputs(camera=camera)
        assert "depth" in outputs
        depth: Tensor = outputs["depth"]  # type: ignore
        rgb: Tensor = outputs["rgb"]  # type: ignore
        W, H = camera.width.item(), camera.height.item()

        # backproject from depth map
        points, colors = get_colored_points_from_depth(
            depths=depth,
            rgbs=rgb,
            fx=camera.fx.item(),
            fy=camera.fy.item(),
            cx=camera.cx.item(),
            cy=camera.cy.item(),
            img_size=(W, H),  # img_size = (w,h)
            c2w=c2w,
        )
        points = points.view(H, W, -1)  # type: ignore
        colors = colors.view(H, W, 3)

        if mask is not None:
            mask = mask.to(points.device)
            points = points * mask
            depth = depth * mask

        no_depth_mask = (depth <= 0.0)[..., 0]
        points = points[~no_depth_mask]
        colors = colors[~no_depth_mask]

        # get closest gaussians
        closest_gaussians_idx = knn_sk(self.means.data, points, k=16)

        # compute gaussian stds along ray direction
        viewdirs = -self.means.detach() + camera.camera_to_worlds.detach()[..., :3, 3]
        viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
        quats = self.quats / self.quats.norm(dim=-1, keepdim=True)
        inv_rots = quat_to_rotmat(invert_quaternion(quat=quats))
        gaussian_standard_deviations = (
                torch.exp(self.scales) * torch.bmm(inv_rots, viewdirs[..., None])[..., 0]
        ).norm(dim=-1)
        points_stds = gaussian_standard_deviations[closest_gaussians_idx][
            ..., 0
        ]  # get first closest gaussian std

        range_size = 3
        n_points_in_range = 21
        n_points_per_pass = 2_000_000

        # sampling on ray
        points_range = (
            torch.linspace(-range_size, range_size, n_points_in_range)
            .to(self.device)
            .view(1, -1, 1)
        )  # (1, n_points_in_range, 1)
        points_range = points_range * points_stds[..., None, None].expand(
            -1, n_points_in_range, 1
        )  # (n_points, n_points_in_range, 1)
        camera_to_samples = torch.nn.functional.normalize(
            points - camera.camera_to_worlds.detach()[..., :3, 3], dim=-1
        )  # (n_points, 3)
        samples = (
                points[:, None, :] + points_range * camera_to_samples[:, None, :]
        ).view(
            -1, 3
        )  # (n_points * n_points_in_range, 3)
        samples_closest_gaussians_idx = (
            closest_gaussians_idx[:, None, :]
            .expand(-1, n_points_in_range, -1)
            .reshape(-1, 16)
        )

        densities = torch.zeros(len(samples), dtype=torch.float, device=self.device)
        gaussian_strengths = torch.sigmoid(self.opacities)
        gaussian_centers = self.means
        gaussian_inv_scaled_rotation = scale_rot_to_inv_cov3d(
            scale=torch.exp(self.scales), quat=self.quats, return_sqrt=True
        )

        # compute densities along rays
        for i in range(0, len(samples), n_points_per_pass):
            i_start = i
            i_end = min(len(samples), i + n_points_per_pass)

            pass_closest_gaussians_idx = samples_closest_gaussians_idx[i_start:i_end]

            closest_gaussian_centers = gaussian_centers[pass_closest_gaussians_idx]
            closest_gaussian_inv_scaled_rotation = gaussian_inv_scaled_rotation[
                pass_closest_gaussians_idx
            ]

            closest_gaussian_strengths = gaussian_strengths[pass_closest_gaussians_idx]
            shift = samples[i_start:i_end, None] - closest_gaussian_centers
            man_distance = (
                    closest_gaussian_inv_scaled_rotation.transpose(-1, -2)
                    @ shift[..., None]
            )
            neighbor_opacities = (
                (man_distance[..., 0] * man_distance[..., 0])
                .sum(dim=-1)
                .clamp(min=0.0, max=1e8)
            )
            neighbor_opacities = closest_gaussian_strengths[..., 0] * torch.exp(
                -1.0 / 2 * neighbor_opacities
            )
            pass_densities = neighbor_opacities.sum(dim=-1)

            pass_density_mask = pass_densities >= 1.0
            pass_densities[pass_density_mask] = pass_densities[pass_density_mask] / (
                    pass_densities[pass_density_mask].detach() + 1e-5
            )
            densities[i_start:i_end] = pass_densities

        densities = densities.reshape(
            -1, n_points_in_range
        )  # (num_samples, n_points_in_range (21))

        all_outputs = {}
        for surface_level in surface_levels:
            outputs = {}

            under_level = densities - surface_level < 0
            above_level = densities - surface_level > 0

            _, first_point_above_level = above_level.max(dim=-1, keepdim=True)
            empty_pixels = ~under_level[..., 0] + (first_point_above_level[..., 0] == 0)

            # depth as level point
            valid_densities = densities[~empty_pixels]
            valid_range = points_range[~empty_pixels][..., 0]
            valid_first_point_above_level = first_point_above_level[~empty_pixels]

            first_value_above_level = valid_densities.gather(
                dim=-1, index=valid_first_point_above_level
            ).view(-1)
            value_before_level = valid_densities.gather(
                dim=-1, index=valid_first_point_above_level - 1
            ).view(-1)

            first_t_above_level = valid_range.gather(
                dim=-1, index=valid_first_point_above_level
            ).view(-1)
            t_before_level = valid_range.gather(
                dim=-1, index=valid_first_point_above_level - 1
            ).view(-1)

            intersection_t = (surface_level - value_before_level) / (
                    first_value_above_level - value_before_level
            ) * (first_t_above_level - t_before_level) + t_before_level
            intersection_points = (
                    points[~empty_pixels]
                    + intersection_t[:, None] * camera_to_samples[~empty_pixels]
            )
            intersection_colors = colors[~empty_pixels]

            # normal
            if return_normal == "analytical":
                points_closest_gaussians_idx = closest_gaussians_idx[~empty_pixels]
                closest_gaussian_centers = gaussian_centers[
                    points_closest_gaussians_idx
                ]
                closest_gaussian_inv_scaled_rotation = gaussian_inv_scaled_rotation[
                    points_closest_gaussians_idx
                ]
                closest_gaussian_strengths = gaussian_strengths[
                    points_closest_gaussians_idx
                ]
                shift = intersection_points[:, None] - closest_gaussian_centers
                man_distance = (
                        closest_gaussian_inv_scaled_rotation.transpose(-1, -2)
                        @ shift[..., None]
                )
                neighbor_opacities = (
                    (man_distance[..., 0] * man_distance[..., 0])
                    .sum(dim=-1)
                    .clamp(min=0.0, max=1e8)
                )
                neighbor_opacities = closest_gaussian_strengths[..., 0] * torch.exp(
                    -1.0 / 2 * neighbor_opacities
                )
                density_grad = (
                        neighbor_opacities[..., None]
                        * (closest_gaussian_inv_scaled_rotation @ man_distance)[..., 0]
                ).sum(dim=-2)
                intersection_normals = -torch.nn.functional.normalize(
                    density_grad, dim=-1
                )
            elif return_normal == "closest_gaussian":
                points_closest_gaussians_idx = closest_gaussians_idx[~empty_pixels]
                intersection_normals = self.normals[
                    points_closest_gaussians_idx[..., 0]
                ]
            else:
                raise NotImplementedError

            # sample pixels for this frame
            assert intersection_points.shape[0] == intersection_normals.shape[0]
            indices = random.sample(
                range(intersection_points.shape[0]),
                (
                    num_samples
                    if num_samples < intersection_points.shape[0]
                    else intersection_points.shape[0]
                ),
            )
            samples_mask = torch.tensor(indices, device=points.device)
            intersection_points = intersection_points[samples_mask]
            intersection_normals = intersection_normals[samples_mask]
            intersection_colors = intersection_colors[samples_mask]

            outputs["points"] = intersection_points
            outputs["normals"] = intersection_normals
            outputs["colors"] = intersection_colors
            outputs["sdf"] = self.get_sdf(
                sdf_samples=intersection_points,
                closest_gaussians=closest_gaussians_idx[~empty_pixels][samples_mask]
            )
            all_outputs[surface_level] = outputs

        return all_outputs

    # @property
    def normals(self):
        return self.gauss_params["normals"]

def invert_quaternion(quat: Tensor):
    """Invert quaternion in wxyz convention

    Args:
        quaternion: quat shape (..., 4), with real part first

    Returns:
        inverse quat, shape (..., 4).
    """
    scaling = torch.tensor([1, -1, -1, -1], device=quat.device)
    return quat * scaling
