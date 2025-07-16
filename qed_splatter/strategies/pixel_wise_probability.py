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

    prob_add_every: int = 500
    """Number of frames after which to add new Gaussians based on the probability map."""

    def step_post_backward(
            self,
            params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
            optimizers: Dict[str, torch.optim.Optimizer],
            state: Dict[str, Any],
            step: int,
            info: Dict[str, Any],
            packed: bool = False,
            probability_map: bool = False
    ):
        super().step_post_backward(params, optimizers, state, step, info, packed)

        if step % self.prob_add_every == 0:


