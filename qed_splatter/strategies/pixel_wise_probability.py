import torch
from gsplat.strategy import DefaultStrategy

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

@torch.no_grad()

@dataclass
class PixelWiseProbStrategy(DefaultStrategy):
    """
    Originally Implemented in:
    "On-the-fly Reconstruction for Large-Scale Novel View Synthesis from Unposed Images"
    https://repo-sam.inria.fr/nerphys/on-the-fly-nvs/

    Computes a pixel-wise probability map to guide the placement of new Gaussians.

    This probability map highlights visually informative regions in the keyframe image,
    favoring areas with high-frequency content such as edges or texture. It is computed
    by applying a Laplacian-based norm to a downsampled and smoothed
    version of the keyframe image.

    The resulting probability values are later used to stochastically sample pixels where
    new Gaussians should be added, promoting effiscient and meaningful updates to the scene
    """


    def add_gaussians(self, params, optimizers, valid_gs_mask: torch.Tensor, new_gaussians: dict):
        """
        Adds new Gaussians to the model parameters
        """
        for par_name in params.keys():
            optimizer = optimizers[par_name]

            for i, param_group in enumerate(optimizer.param_groups):
                param = param_group["params"][0]
                extension_tensor = new_gaussians[par_name]

                # Copy optimizer state for the new parameter
                p_state = optimizer.state[param]
                del optimizer.state[param]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = torch.cat([p_state[key][valid_gs_mask], torch.zeros_like(extension_tensor)], dim=0).contiguous()

                param_new = torch.nn.Parameter(torch.cat([param[valid_gs_mask], extension_tensor]))
                optimizer.param_groups[i]["params"] = [param_new]
                optimizer.state[param_new] = p_state
                params[par_name] = param_new
