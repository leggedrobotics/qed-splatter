import json
import math
import os
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from soft_pruning_tweeks.soft_prune_utils.nerf import NerfDataset , NerfParser
from soft_pruning_tweeks.soft_prune_utils.colmap import Dataset, Parser
from soft_pruning_tweeks.soft_prune_utils.traj import (
    generate_interpolated_path,
    generate_ellipse_path_z,
    generate_spiral_path,
)
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fused_ssim import fused_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from soft_pruning_tweeks.soft_prune_utils.utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed
from soft_pruning_tweeks.soft_prune_utils.lib_bilagrid import (
    BilateralGrid,
    slice,
    color_correct,
    total_variation_loss,
)
from soft_pruning_tweeks.soft_prune_utils.open_ply_pipeline import load_splats, save_splats


from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.strategy.ops import remove

from PIL import Image


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Pruning Ratio
    pruning_ratio: float = 0.0
    # Output data format converted
    output_format: str = "ply"    
    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = False
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 10
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = True
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)




class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        ext = os.path.splitext(cfg.ckpt[0])[1].lower()
        if ext == ".ckpt" or ext == ".ply" : 
            print("this is a ckpt or ply file ")
            self.parser = NerfParser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=cfg.normalize_world_space,
                test_every=cfg.test_every,
            )
            self.trainset = NerfDataset(
                self.parser,
                split="train",
                patch_size=cfg.patch_size,
                load_depths=cfg.depth_loss,
            )
        elif ext == ".pt":
            print("this is a pt file")
            self.parser = Parser(
                    data_dir=cfg.data_dir,
                    factor=cfg.data_factor,
                    normalize=cfg.normalize_world_space,
                    test_every=cfg.test_every,
                )
            self.trainset = Dataset(
                self.parser,
                split="train",
                patch_size=cfg.patch_size,
                load_depths=cfg.depth_loss,
            )
        else:
            msg ="Invalid Model type. Use .pt, .ckpt or .ply files."
            raise TypeError(msg)
        
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = self.create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            cfg=self.cfg,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

    def create_splats_with_optimizers(
        self,
        parser: Parser,
        init_type: str = "sfm",
        init_num_pts: int = 100_000,
        init_extent: float = 3.0,
        init_opacity: float = 0.1,
        init_scale: float = 1.0,
        scene_scale: float = 1.0,
        sh_degree: int = 3,
        sparse_grad: bool = False,
        batch_size: int = 1,
        feature_dim: Optional[int] = None,
        device: str = "cuda",
        world_rank: int = 0,
        world_size: int = 1,
        cfg: Optional[List[str]] = None,
    ) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
        
        print("the path is : ", cfg.ckpt)

        self.steps, splats = load_splats(cfg.ckpt[0], device)

        # Convert to ParameterDict on the correct device
        splats = torch.nn.ParameterDict(splats).to(device)


        print("size of the splats for sh0m shN, means, opacities, scales, is : ", splats["sh0"].shape, splats["shN"].shape, splats["means"].shape, splats["opacities"].shape, splats["scales"].shape)


        # Learning rates: you need to define them since they’re not stored in the ckpt
        # Use default values from above
        default_lrs = {
            "means": 1.6e-4 * scene_scale,
            "scales": 5e-3,
            "quats": 1e-3,
            "opacities": 5e-2,
            "sh0": 2.5e-3,
            "shN": 2.5e-3 / 20,
            "features": 2.5e-3,
            "colors": 2.5e-3,
        }

        BS = batch_size * world_size
        optimizers = {
            name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
                [
                    {
                        "params": splats[name],
                        "lr": default_lrs[name] * math.sqrt(BS),
                        "name": name,
                    }
                ],
                eps=1e-15 / math.sqrt(BS),
                betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            )
            for name in splats.keys()
            if name in default_lrs
        }

        return splats, optimizers


    def reinit_optimizers(self):
        """Reinitialize optimizers after pruning Gaussians."""
        print("Reinitializing optimizers after pruning...")
        device = self.device
        cfg = self.cfg
        BS = cfg.batch_size * self.world_size

        # Recreate the optimizer dictionary with new parameters
        new_optimizers = {}
        for name, param in self.splats.items():
            lr = {
                "means": 1.6e-4 * self.scene_scale,
                "scales": 5e-3,
                "quats": 1e-3,
                "opacities": 5e-2,
                "sh0": 2.5e-3,
                "shN": 2.5e-3 / 20,
                "features": 2.5e-3,
                "colors": 2.5e-3,
            }[name]

            betas = (
                1 - BS * (1 - 0.9),
                1 - BS * (1 - 0.999),
            )

            new_optimizers[name] = torch.optim.Adam(
                [{"params": param, "lr": lr * math.sqrt(BS), "name": name}],
                eps=1e-15 / math.sqrt(BS),
                betas=betas,
            )

        # Replace old optimizers with new ones
        self.optimizers = new_optimizers


    def prune_gaussians(self, prune_ratio: float, scores):
        """Prune Gaussians based on score thresholding."""
        reduced_scores = scores.mean(dim=1)

        num_prune = int(len(reduced_scores) * prune_ratio)
        print("the number of gaussians to remove is : ",num_prune)
        print("the size of the squeezed score is : ", reduced_scores.shape)
        _, idx = torch.topk(reduced_scores, k=num_prune, largest=False)
        mask = torch.ones_like(reduced_scores, dtype=torch.bool)
        mask[idx] = False
        # Prune splats
        # pruned_splats = {}
        # for name in self.splats:
        #     pruned_splats[name] = self.splats[name].data[mask]

        # # Replace splats with pruned ones
        # self.splats = torch.nn.ParameterDict({
        #     k: torch.nn.Parameter(v.clone()) for k, v in pruned_splats.items()
        # })
        
        remove(
            params=self.splats,
            optimizers= self.optimizers,
            state= self.strategy_state,
            mask = ~mask,
        )
                     


    def save_tensors_side_by_side(self, rendered_tensor, target_tensor, filename="depth_debug/depth_comparison.png"):
        """
        Saves two PyTorch tensors (e.g., depth maps) as side-by-side PNG images.
        If the file exists, appends a number to avoid overwriting (e.g., _1, _2, ...).

        Args:
            rendered_tensor (torch.Tensor): Predicted tensor (e.g., rendered depth).
            target_tensor (torch.Tensor): Ground truth tensor (e.g., image depth).
            filename (str): Base output filename (will be modified if it exists).
        """
        def tensor_to_image(tensor):
            # Ensure tensor is on CPU and detach gradients
            tensor = tensor.cpu().detach()

            # Remove batch dimension if present (e.g., [1, H, W, 1] -> [H, W, 1])
            while tensor.dim() > 3 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)

            # At this point, should be [H, W] or [H, W, 1]
            if tensor.dim() == 3 and tensor.shape[-1] == 1:
                tensor = tensor.squeeze(-1)  # Remove channel dim

            # Now it should be [H, W]
            if tensor.dim() != 2:
                raise ValueError(f"Unexpected tensor shape after squeezing: {tensor.shape}")

            # Normalize to [0, 1]
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

            # Convert to uint8 [0, 255]
            image = (tensor.numpy() * 255).astype(np.uint8)

            return Image.fromarray(image, mode='L'), image  # L = grayscale

        # Convert both tensors to PIL images
        rendered_img, rendered_image = tensor_to_image(rendered_tensor)
        target_img, target_image = tensor_to_image(target_tensor)


        # Create a new image that combines them horizontally
        total_width = rendered_img.width + target_img.width
        max_height = max(rendered_img.height, target_img.height)

        combined_img = Image.new('L', (total_width, max_height))  # 'L' for grayscale
        combined_img.paste(rendered_img, (0, 0))
        combined_img.paste(target_img, (rendered_img.width, 0))

        # Save to disk
        combined_img.save(filename)
        # print(f"Saved comparison image to '{new_filename}'")

        return rendered_image, target_image


    @torch.enable_grad()
    def score_func(self, viewpoint_cam, scores, mask_views):
        img_scores = torch.zeros_like(scores, requires_grad=True)


        # Get camera matrices without extra dimensions
        camtoworld = viewpoint_cam["camtoworld"].to(self.device)  # shape: [4, 4]
        K = viewpoint_cam["K"].to(self.device)                    # shape: [3, 3]
        height, width = viewpoint_cam["image"].shape[1:3]

        # Forward pass
        render, _, info = self.rasterize_splats(
            camtoworlds=camtoworld,   # Add batch dim here only
            Ks=K,                     # Add batch dim here only
            width=width,
            height=height,
            sh_degree=self.cfg.sh_degree,
            image_ids=viewpoint_cam["image_id"].to(self.device)[None],
            render_mode="RGB+ED"
        )

        # Compute loss
        rendered_depth = render[..., 3:4]

        # print("testing index : ", viewpoint_cam["image_id"])

        image_depth = self.trainset[viewpoint_cam["image_id"].item()]["depth"].to(self.device)[None].unsqueeze(-1)

        # print("the id is : ", viewpoint_cam["image_id"])
        # print("the test data is : ", camtoworld)




        rendered_np = rendered_depth
        image_np = image_depth
        # print("render: ", type(rendered_depth))
        # print("base: ", type(image_depth))




        # rendered_image, target_image = self.save_tensors_side_by_side(rendered_np, image_np, f"depth_debug/00{self.trainset.indices[viewpoint_cam['image_id'].item()]}.png")

        # print(f"Rendered depth - min: {rendered_image.min()}, max: {rendered_image.max()}")
        # print(f"Image depth    - min: {target_image.min()}, max: {target_image.max()}")


        l1loss = F.l1_loss(rendered_depth, image_depth)
        ssimloss = 1.0 - fused_ssim(
            rendered_depth.permute(0, 3, 1, 2),
            image_depth.permute(0, 3, 1, 2),
            padding="valid"
        )
        loss = l1loss * (1.0 - self.cfg.ssim_lambda) + ssimloss * self.cfg.ssim_lambda

        # Backward pass
        loss.backward()

        mask_views += self.splats["means"].grad.abs().squeeze() > 0.001

        # Accumulate gradient magnitude as score (e.g., opacities)
        with torch.no_grad():
            scores += self.splats["means"].grad.abs().squeeze() 


    @torch.no_grad()
    def prune(self, prune_ratio: float):
        print("Running pruning...")
        device = self.device
        scores = torch.zeros_like(self.splats["means"])
        mask_views = torch.zeros_like(self.splats["means"])

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

        i = 0 
        pbar = tqdm.tqdm(trainloader, desc="Computing pruning scores")
        for data in pbar:
            # if 10 > i : 
            #     print("the data in loop is : ", data)
            self.score_func(data, scores, mask_views)
            pbar.update(1)
            i += 1


        scores = scores / (mask_views + 1e-8)

        # Prune Gaussians
        self.prune_gaussians(prune_ratio, scores)
        


    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        render_mode: Optional[str] = "RGB",
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            render_mode=render_mode,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info


    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+D",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        # print("the K matrix is equal to : ", K)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        # print("the cam to world matrix is : ", c2w)

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    if cfg.ckpt is not None:
        # run eval only
        runner = Runner(local_rank, world_rank, world_size, cfg)
                
        print("We are now hard pruning")
        if cfg.pruning_ratio != 0:
            runner.prune(prune_ratio=cfg.pruning_ratio)
        print("the size of gaussian is:", runner.splats["means"].shape)

        # save checkpoint after hard pruning
        mem = torch.cuda.max_memory_allocated() / 1024**3
        step = runner.steps
        
        name = os.path.splitext(os.path.basename(cfg.ckpt[0]))[0]
  
        data = {"step": step, "splats": runner.splats.state_dict()}
        if cfg.pose_opt:
            if world_size > 1:
                data["pose_adjust"] = runner.pose_adjust.module.state_dict()
            else:
                data["pose_adjust"] = runner.pose_adjust.state_dict()
        if cfg.app_opt:
            if world_size > 1:
                data["app_module"] = runner.app_module.module.state_dict()
            else:
                data["app_module"] = runner.app_module.state_dict()
        suffix = str(cfg.pruning_ratio).replace("0.", "")

        print("output format", cfg.output_format)

        save_splats(f"{runner.ckpt_dir}/{name}_depth_pruned_{suffix}", data, cfg.output_format)
                
    else:
        raise ValueError("ckpt cant be None")

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    cli(main, cfg, verbose=True)