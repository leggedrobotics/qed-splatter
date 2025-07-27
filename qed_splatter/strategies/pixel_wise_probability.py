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


    def add_gaussians(self, total_edge_diff):
        pass

    @torch.no_grad()
    def _insert_gaussians(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        positions: torch.Tensor,  # [M, 3]
        scales: torch.Tensor,  # [M, 3]
        rotations: torch.Tensor,  # [M, 4] or [M, 3, 3]
        opacities: torch.Tensor,  # [M]
        colors: torch.Tensor  # [M, 3]
    ):
        """
        Inserts new Gaussians into the parameter set.
        """
        device = params["positions"].device
        dtype = params["positions"].dtype

        M = positions.shape[0]

        # Sanity checks
        assert positions.shape == (M, 3)
        assert scales.shape == (M, 3)
        assert rotations.shape[0] == M
        assert opacities.shape == (M,)
        assert colors.shape == (M, 3)

        # Encode to match parameter space if needed
        encoded_scales = torch.log(scales.clamp(min=1e-6)).to(device=device, dtype=dtype)
        encoded_opacities = torch.logit(opacities.clamp(1e-4, 1 - 1e-4)).to(device=device, dtype=dtype)
        encoded_positions = positions.to(device=device, dtype=dtype)
        encoded_colors = colors.to(device=device, dtype=dtype)

        # Handle rotations
        if rotations.shape[-1] == 4:
            encoded_rotations = rotations / rotations.norm(dim=-1, keepdim=True)  # quaternion
        else:
            encoded_rotations = rotations  # assume already normalized 3x3 matrices or similar

        encoded_rotations = encoded_rotations.to(device=device, dtype=dtype)

        # Append to parameters
        for key, new_data in zip(
                ["positions", "scales", "rotations", "opacities", "colors"],
                [encoded_positions, encoded_scales, encoded_rotations, encoded_opacities, encoded_colors]
        ):
            param = params[key]
            updated_param = torch.cat([param.data, new_data], dim=0)
            params[key] = torch.nn.Parameter(updated_param, requires_grad=True)

            # Replace in optimizer if needed
            if key in optimizers:
                del optimizers[key]
                optimizers[key] = torch.optim.Adam([params[key]], lr=optimizers[key].defaults["lr"])

        # Expand state entries for new Gaussians
        for key in ["grad2d", "count", "radii"]:
            if key in state and state[key] is not None:
                state[key] = torch.cat([
                    state[key],
                    torch.zeros(M, device=state[key].device, dtype=state[key].dtype)
                ], dim=0)



