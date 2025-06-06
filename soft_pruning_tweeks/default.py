from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch
from torch import Tensor

from .base import Strategy
from .ops import duplicate, remove, reset_opa, split
from typing_extensions import Literal

from .soft_prune_utils.nerf import NerfDataset , NerfParser
import os
from gsplat.rendering import rasterization
from .soft_prune_utils.utils import AppearanceOptModule

import os

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union


import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from fused_ssim import fused_ssim



@dataclass
class DefaultStrategy(Strategy):
    """A default strategy that follows the original 3DGS paper:

    `3D Gaussian Splatting for Real-Time Radiance Field Rendering <https://arxiv.org/abs/2308.04079>`_

    The strategy will:

    - Periodically duplicate GSs with high image plane gradients and small scales.
    - Periodically split GSs with high image plane gradients and large scales.
    - Periodically prune GSs with low opacity.
    - Periodically reset GSs to a lower opacity.

    If `absgrad=True`, it will use the absolute gradients instead of average gradients
    for GS duplicating & splitting, following the AbsGS paper:

    `AbsGS: Recovering Fine Details for 3D Gaussian Splatting <https://arxiv.org/abs/2404.10484>`_

    Which typically leads to better results but requires to set the `grow_grad2d` to a
    higher value, e.g., 0.0008. Also, the :func:`rasterization` function should be called
    with `absgrad=True` as well so that the absolute gradients are computed.

    Args:
        prune_opa (float): GSs with opacity below this value will be pruned. Default is 0.005.
        grow_grad2d (float): GSs with image plane gradient above this value will be
          split/duplicated. Default is 0.0002.
        grow_scale3d (float): GSs with 3d scale (normalized by scene_scale) below this
          value will be duplicated. Above will be split. Default is 0.01.
        grow_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be split. Default is 0.05.
        prune_scale3d (float): GSs with 3d scale (normalized by scene_scale) above this
          value will be pruned. Default is 0.1.
        prune_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be pruned. Default is 0.15.
        refine_scale2d_stop_iter (int): Stop refining GSs based on 2d scale after this
          iteration. Default is 0. Set to a positive value to enable this feature.
        refine_start_iter (int): Start refining GSs after this iteration. Default is 500.
        refine_stop_iter (int): Stop refining GSs after this iteration. Default is 15_000.
        reset_every (int): Reset opacities every this steps. Default is 3000.
        refine_every (int): Refine GSs every this steps. Default is 100.
        pause_refine_after_reset (int): Pause refining GSs until this number of steps after
          reset, Default is 0 (no pause at all) and one might want to set this number to the
          number of images in training set.
        absgrad (bool): Use absolute gradients for GS splitting. Default is False.
        revised_opacity (bool): Whether to use revised opacity heuristic from
          arXiv:2404.06109 (experimental). Default is False.
        verbose (bool): Whether to print verbose information. Default is False.
        key_for_gradient (str): Which variable uses for densification strategy.
          3DGS uses "means2d" gradient and 2DGS uses a similar gradient which stores
          in variable "gradient_2dgs".

    Examples:

        >>> from gsplat import DefaultStrategy, rasterization
        >>> params: Dict[str, torch.nn.Parameter] | torch.nn.ParameterDict = ...
        >>> optimizers: Dict[str, torch.optim.Optimizer] = ...
        >>> strategy = DefaultStrategy()
        >>> strategy.check_sanity(params, optimizers)
        >>> strategy_state = strategy.initialize_state()
        >>> for step in range(1000):
        ...     render_image, render_alpha, info = rasterization(...)
        ...     strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        ...     loss = ...
        ...     loss.backward()
        ...     strategy.step_post_backward(params, optimizers, strategy_state, step, info)

    """

    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    reset_every: int = 3000
    refine_every: int = 100
    pause_refine_after_reset: int = 0
    absgrad: bool = False
    revised_opacity: bool = False
    verbose: bool = False
    key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"


    print("we are in the default stategie")


    soft_pruning = bool(os.getenv("SOFT_PRUNING", False))
    #  soft pruning method : 
    if soft_pruning:
        current_dir = os.getenv("DATA_DIR", os.getcwd())
        print("Dataset Path: ", current_dir)

        parser = NerfParser(
            data_dir=current_dir,
            factor=4,
            normalize= True,
            test_every= 8,
        )
        trainset = NerfDataset(
            parser,
            split="train",
            patch_size= None,
            load_depths= False,
        )

        ssim_lambda = 0.2
        sh_degree: int = 3
        app_opt: bool = False
        app_embed_dim: int = 16
        device = "cuda"
        antialiased: bool = False
        packed: bool = False
        sparse_grad: bool = False
        camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"
        pruning_ratio = float(os.getenv("PRUNING_RATIO", 0.3))
        initialize = None 




        feature_dim = 32 if app_opt else None
        if app_opt:
            assert feature_dim is not None
            app_module = AppearanceOptModule(
                        len(trainset), feature_dim, app_embed_dim, sh_degree
                    ).to(device)


    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` functions.
        """
        # Postpone the initialization of the state to the first step so that we can
        # put them on the correct device.
        # - grad2d: running accum of the norm of the image plane gradients for each GS.
        # - count: running accum of how many time each GS is visible.
        # - radii: the radii of the GSs (normalized by the image resolution).
        state = {"grad2d": None, "count": None, "scene_scale": scene_scale}
        if self.refine_scale2d_stop_iter > 0:
            state["radii"] = None

        return state

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers.

        Check if:
            * `params` and `optimizers` have the same keys.
            * Each optimizer has exactly one param_group, corresponding to each parameter.
            * The following keys are present: {"means", "scales", "quats", "opacities"}.

        Raises:
            AssertionError: If any of the above conditions is not met.

        .. note::
            It is not required but highly recommended for the user to call this function
            after initializing the strategy to ensure the convention of the parameters
            and optimizers is as expected.
        """

        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Callback function to be executed before the `loss.backward()` call."""
        assert (
            self.key_for_gradient in info
        ), "The 2D means of the Gaussians is required but missing."
        info[self.key_for_gradient].retain_grad()

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """Callback function to be executed after the `loss.backward()` call."""
        if step >= self.refine_stop_iter:
            return

        self._update_state(params, state, info, packed=packed)

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            # grow GSs
            n_dupli, n_split = self._grow_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                    f"Now having {len(params['means'])} GSs."
                )

            # prune GSs
            n_prune = self._prune_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_prune} GSs pruned. "
                    f"Now having {len(params['means'])} GSs."
                )

            # reset running stats
            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()

        if self.soft_pruning:
            if self.initialize is None : 
                self.initialize = params["means"].shape[0]
            else : 
                if step % 1000 == 0 : 
                    self.initialize = params["means"].shape[0]


            if step % 500 == 0 and step > 0 and params["means"].shape[0] > self.initialize  and self.refine_stop_iter > step :
                self.means = params["means"].reshape(-1, 3)  # [N, 3]
                self.quats = params["quats"]  # [N, 4]
                self.scales = params["scales"].reshape(-1, 3)  # [N, 3]
                self.opacities = params["opacities"].reshape(-1)  # [N,]
                self.features_dc = torch.tensor(params["features_dc"], dtype=torch.float32).reshape(-1, 1, 3)
                self.features_rest = torch.tensor(params["features_rest"], dtype=torch.float32).reshape(-1, 15, 3)

                print("it is now soft pruning at step ", step)
                self.prune(self.pruning_ratio, params, optimizers, state)


        if step % self.reset_every == 0:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,
            )

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        for key in [
            "width",
            "height",
            "n_cameras",
            "radii",
            "gaussian_ids",
            self.key_for_gradient,
        ]:
            assert key in info, f"{key} is required but missing."

        # normalize grads to [-1, 1] screen space
        if self.absgrad:
            grads = info[self.key_for_gradient].absgrad.clone()
        else:
            grads = info[self.key_for_gradient].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        # initialize state on the first run
        n_gaussian = len(list(params.values())[0])

        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)
        if self.refine_scale2d_stop_iter > 0 and state["radii"] is None:
            assert "radii" in info, "radii is required but missing."
            state["radii"] = torch.zeros(n_gaussian, device=grads.device)

        # update the running state
        if packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz]
            radii = info["radii"]  # [nnz]
        else:
            # grads is [C, N, 2]
            sel = info["radii"] > 0.0  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            grads = grads[sel]  # [nnz, 2]
            radii = info["radii"][sel]  # [nnz]

        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )
        if self.refine_scale2d_stop_iter > 0:
            # Should be ideally using scatter max
            state["radii"][gs_ids] = torch.maximum(
                state["radii"][gs_ids],
                # normalize radii to [0, 1] screen space
                radii / float(max(info["width"], info["height"])),
            )

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> Tuple[int, int]:
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        is_grad_high = grads > self.grow_grad2d
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        # Print shapes
        print("Shape of is_grad_high:", is_grad_high.shape)
        print("Shape of is_small:", is_small.shape)
        is_dupli = is_grad_high & is_small
        n_dupli = is_dupli.sum().item()

        is_large = ~is_small
        is_split = is_grad_high & is_large
        if step < self.refine_scale2d_stop_iter:
            is_split |= state["radii"] > self.grow_scale2d
        n_split = is_split.sum().item()

        # first duplicate
        if n_dupli > 0:
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

        # new GSs added by duplication will not be split
        is_split = torch.cat(
            [
                is_split,
                torch.zeros(n_dupli, dtype=torch.bool, device=device),
            ]
        )

        # then split
        if n_split > 0:
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                revised_opacity=self.revised_opacity,
            )
        return n_dupli, n_split
    

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        params, 
        masks: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.means # [N, 3]
        # quats = F.normalize(params["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.quats  # [N, 4]
        scales = torch.exp(self.scales) # [N, 3]
        opacities = torch.sigmoid(self.opacities) # [N,]


        image_ids = kwargs.pop("image_ids", None)
        if self.app_opt:
            colors = self.app_module(
                features=params["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.sh_degree),
            )
            colors = colors + params["colors"]
            colors = torch.sigmoid(colors)
        else:


            colors = torch.cat([self.features_dc, self.features_rest], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.antialiased else "classic"
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
            packed=self.packed,
            absgrad=self.absgrad,
            sparse_grad=self.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=False,
            camera_model=self.camera_model,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info



    def prune_gaussians(self, prune_ratio: float, scores, params, optimizers, state):
        """Prune Gaussians based on score thresholding."""
        num_prune = int(len(scores) * prune_ratio)
        _, idx = torch.topk(scores.squeeze(), k=num_prune, largest=False)
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask[idx] = False

        
        remove(
            params=params,
            optimizers= optimizers,
            state= state,
            mask = ~mask,
        )
                    
    
    @torch.enable_grad()
    def score_func(self, viewpoint_cam, scores, mask_views, device, params):
        img_scores = torch.zeros_like(scores, requires_grad=True)

        # Get camera matrices without extra dimensions
        camtoworld = viewpoint_cam["camtoworld"].to(device)  # shape: [4, 4]
        K = viewpoint_cam["K"].to(device)                    # shape: [3, 3]
        height, width = viewpoint_cam["image"].shape[1:3]

        # Forward pass
        colors, _, _ = self.rasterize_splats(
            camtoworlds=camtoworld,   # Add batch dim here only
            Ks=K,                     # Add batch dim here only
            width=width,
            height=height,
            params = params, 
            sh_degree=self.sh_degree,
            image_ids=viewpoint_cam["image_id"].to(device)[None],
        )

        # Compute loss
        gt_image = viewpoint_cam["image"].to(device) / 255.0
        l1loss = F.l1_loss(colors, gt_image)
        ssimloss = 1.0 - fused_ssim(
            colors.permute(0, 3, 1, 2),
            gt_image.permute(0, 3, 1, 2),
            padding="valid"
        )
        loss = l1loss * (1.0 - self.ssim_lambda) + ssimloss * self.ssim_lambda

        # Backward pass
        loss.backward()

        mask_views += params["opacities"].grad.abs().squeeze() > 0.001

        # Accumulate gradient magnitude as score (e.g., opacities)
        with torch.no_grad():
            scores += params["opacities"].grad.abs().squeeze() 


    @torch.no_grad()
    def prune(self, prune_ratio: float, params, optimizers, state):
        print("Running pruning...")
        device = self.device
        scores = torch.zeros_like(self.opacities)
        mask_views = torch.zeros_like(self.opacities)

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

        pbar = tqdm.tqdm(trainloader, desc="Computing pruning scores")
        for data in pbar:
            self.score_func(data, scores, mask_views, device, params)
            pbar.update(1)

        scores = scores / (mask_views + 1e-8)

        print("the mask is equal to : ", mask_views)

        # Prune Gaussians
        self.prune_gaussians(prune_ratio, scores, params, optimizers, state)

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
        if step > self.reset_every:
            is_too_big = (
                torch.exp(params["scales"]).max(dim=-1).values
                > self.prune_scale3d * state["scene_scale"]
            )
            # The official code also implements sreen-size pruning but
            # it's actually not being used due to a bug:
            # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
            # We implement it here for completeness but set `refine_scale2d_stop_iter`
            # to 0 by default to disable it.
            if step < self.refine_scale2d_stop_iter:
                is_too_big |= state["radii"] > self.prune_scale2d

            is_prune = is_prune | is_too_big

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune
