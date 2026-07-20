"""RGB hard pruning via nerfstudio splatfacto ``get_outputs`` gradients."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from qed_splatter.pruning.base import HardPruneConfig, HardPruner, launch_cli


@dataclass
class RGBHardPruneConfig(HardPruneConfig):
    """Prune gaussians using RGB reconstruction loss gradients."""

    def main(self) -> None:
        RGBHardPruner(self).run()


class RGBHardPruner(HardPruner):
    def output_stem(self) -> str:
        suffix = str(self.cfg.pruning_ratio).replace("0.", "")
        return f"{self.cfg.load_config.parent.name}_rgb_pruned_{suffix}"

    def compute_scores(self) -> torch.Tensor:
        scores = torch.zeros(len(self.model.means), device=self.device)
        for _, camera, batch in self.iter_train_views():
            outputs = self.render(camera)
            loss = self.rgb_loss(outputs["rgb"], batch["image"])
            loss.backward()

            gp = self.model.gauss_params
            combined = (
                gp["opacities"].grad.abs().squeeze(-1)
                + gp["means"].grad.abs().mean(dim=-1)
                + gp["scales"].grad.abs().mean(dim=-1)
                + gp["features_dc"].grad.abs().mean(dim=-1)
                + gp["features_rest"].grad.abs().flatten(1).mean(dim=-1)
            )
            with torch.no_grad():
                scores += combined
        return scores


def entrypoint() -> None:
    launch_cli(RGBHardPruneConfig)


if __name__ == "__main__":
    entrypoint()
