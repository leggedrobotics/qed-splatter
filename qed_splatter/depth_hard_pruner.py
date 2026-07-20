"""Depth hard pruning via nerfstudio splatfacto ``get_outputs`` gradients."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from qed_splatter.pruning.base import HardPruneConfig, HardPruner, launch_cli


@dataclass
class DepthHardPruneConfig(HardPruneConfig):
    """Prune gaussians using depth reconstruction loss gradients."""

    def main(self) -> None:
        DepthHardPruner(self).run()


class DepthHardPruner(HardPruner):
    def output_stem(self) -> str:
        suffix = str(self.cfg.pruning_ratio).replace("0.", "")
        return f"{self.cfg.load_config.parent.name}_depth_pruned_{suffix}"

    def compute_scores(self) -> torch.Tensor:
        scores = torch.zeros(len(self.model.means), device=self.device)
        hits = torch.zeros_like(scores)

        for _, camera, batch in self.iter_train_views():
            if "depth_image" not in batch:
                raise KeyError(
                    "Train batch has no 'depth_image'. Use a depth-aware method "
                    "(e.g. qed-splatter) when running depth pruning."
                )
            outputs = self.render(camera)
            assert outputs["depth"] is not None, "Model did not return depth"
            loss = self.depth_loss(outputs["depth"], batch["depth_image"])
            if loss.numel() == 0 or not torch.isfinite(loss):
                continue
            loss.backward()

            means_grad = self.model.gauss_params["means"].grad.abs().mean(dim=-1)
            with torch.no_grad():
                active = means_grad > 1e-8
                scores += means_grad
                hits += active.float()

        return scores / (hits + 1e-8)


def entrypoint() -> None:
    launch_cli(DepthHardPruneConfig)


if __name__ == "__main__":
    entrypoint()
