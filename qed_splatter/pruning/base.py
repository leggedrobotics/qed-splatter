"""Shared hard-pruning config and runner (dn-splatter / nerfstudio style).

Loads a past splatfacto / qed-splatter training via ``eval_setup``, scores
gaussians with ``model.get_outputs``, prunes ``model.gauss_params``, and
exports with nerfstudio's ``ExportGaussianSplat.write_ply``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Type

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.scripts.exporter import ExportGaussianSplat
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from torchmetrics.image import StructuralSimilarityIndexMeasure


@dataclass
class HardPruneConfig:
    """Shared CLI config for hard pruning."""

    load_config: Path
    """Path to the trained nerfstudio config.yml."""
    result_dir: Path = Path("./pruned/")
    """Directory for pruned outputs."""
    pruning_ratio: float = 0.1
    """Fraction of gaussians to remove (lowest scores)."""
    output_format: str = "ply"
    """``ply`` (nerfstudio gaussian-splat) or ``ckpt``."""
    ssim_lambda: float = 0.2
    """Weight of SSIM vs L1 in the scoring loss."""
    eval_only: bool = False
    """If True, skip pruning and only report gaussian count."""

    def main(self) -> None:
        raise NotImplementedError


class HardPruner(ABC):
    """Base runner: eval_setup → score → prune → export."""

    def __init__(self, cfg: HardPruneConfig) -> None:
        self.cfg = cfg
        self.cfg.result_dir.mkdir(parents=True, exist_ok=True)

        self.config, self.pipeline, self.checkpoint_path, self.step = eval_setup(
            cfg.load_config, test_mode="val"
        )
        assert isinstance(self.pipeline.model, SplatfactoModel), (
            f"Expected SplatfactoModel (or subclass), got {type(self.pipeline.model).__name__}"
        )
        self.model: SplatfactoModel = self.pipeline.model
        self.device = self.pipeline.device
        self.train_dataset = self.pipeline.datamanager.train_dataset
        assert self.train_dataset is not None

        # Gradients through rasterization; keep eval path (no densify strategy hooks).
        self.model.eval()
        for p in self.model.gauss_params.parameters():
            p.requires_grad_(True)

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        CONSOLE.print(
            f"Loaded {len(self.model.means)} gaussians from {self.checkpoint_path} (step {self.step})"
        )

    def run(self) -> None:
        if self.cfg.eval_only or self.cfg.pruning_ratio <= 0:
            CONSOLE.print(f"Skipping prune (eval_only={self.cfg.eval_only}, ratio={self.cfg.pruning_ratio})")
            CONSOLE.print(f"Number of GS: {len(self.model.means)}")
            if not self.cfg.eval_only:
                self.save()
            return

        CONSOLE.print("Computing pruning scores...")
        scores = self.compute_scores()
        self.prune(scores)
        CONSOLE.print(f"Number of GS after prune: {len(self.model.means)}")
        self.save()

    @abstractmethod
    def compute_scores(self) -> torch.Tensor:
        """Return a per-gaussian score (higher = keep)."""

    def iter_train_views(self):
        """Yield ``(image_idx, camera, batch)`` like dn-splatter's dataset loop."""
        cameras: Cameras = self.train_dataset.cameras  # type: ignore[attr-defined]
        for image_idx in tqdm.trange(len(self.train_dataset), desc="Views"):
            batch = self.train_dataset[image_idx]
            camera = cameras[image_idx : image_idx + 1].to(self.device)
            yield image_idx, camera, batch

    def render(self, camera: Cameras) -> Dict[str, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        return self.model.get_outputs(camera)  # type: ignore[return-value]

    def rgb_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        gt = self.model.get_gt_img(gt)
        if gt.shape[-1] == 4:
            gt = self.model.composite_with_background(gt, self.model._get_background_color())
        l1 = F.l1_loss(pred, gt)
        ssim = self.ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
        return l1 * (1.0 - self.cfg.ssim_lambda) + (1.0 - ssim) * self.cfg.ssim_lambda

    def depth_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if gt.ndim == 2:
            gt = gt.unsqueeze(-1)
        gt = gt.to(pred.device, dtype=pred.dtype)
        # Match shapes if needed
        if pred.shape[:2] != gt.shape[:2]:
            gt = F.interpolate(
                gt.permute(2, 0, 1)[None],
                size=pred.shape[:2],
                mode="nearest",
            )[0].permute(1, 2, 0)
        valid = torch.isfinite(pred) & torch.isfinite(gt) & (gt > 0)
        if valid.sum() == 0:
            return pred.new_zeros(())
        pred_v, gt_v = pred[valid], gt[valid]
        l1 = F.l1_loss(pred_v, gt_v)
        # SSIM on dense maps (fill invalid with 0 for structural term)
        pred_f = torch.where(valid, pred, torch.zeros_like(pred))
        gt_f = torch.where(valid, gt, torch.zeros_like(gt))
        ssim = self.ssim(pred_f.permute(2, 0, 1)[None], gt_f.permute(2, 0, 1)[None])
        return l1 * (1.0 - self.cfg.ssim_lambda) + (1.0 - ssim) * self.cfg.ssim_lambda

    @torch.no_grad()
    def prune(self, scores: torch.Tensor) -> None:
        n = scores.numel() if scores.ndim == 1 else scores.shape[0]
        num_prune = int(n * self.cfg.pruning_ratio)
        if num_prune <= 0:
            return
        flat = scores.reshape(n, -1).mean(dim=1)
        _, idx = torch.topk(flat, k=num_prune, largest=False)
        keep = torch.ones(n, dtype=torch.bool, device=flat.device)
        keep[idx] = False
        self._apply_keep_mask(keep)

    def _apply_keep_mask(self, keep: torch.Tensor) -> None:
        with torch.no_grad():
            for name, param in list(self.model.gauss_params.items()):
                self.model.gauss_params[name] = torch.nn.Parameter(param.data[keep].clone())
                self.model.gauss_params[name].requires_grad_(True)

    def save(self) -> None:
        stem = self.output_stem()
        if self.cfg.output_format == "ply":
            path = self.cfg.result_dir / f"{stem}.ply"
            export_gaussian_ply(self.model, path)
            CONSOLE.print(f"Wrote {path}")
        elif self.cfg.output_format == "ckpt":
            path = self.cfg.result_dir / f"{stem}.ckpt"
            save_pruned_ckpt(self.pipeline, self.step, path)
            CONSOLE.print(f"Wrote {path}")
        else:
            raise ValueError(f"Unknown output_format: {self.cfg.output_format}")

    @abstractmethod
    def output_stem(self) -> str:
        ...


def export_gaussian_ply(model: SplatfactoModel, filename: Path) -> None:
    """Export gaussians using nerfstudio's ``ExportGaussianSplat.write_ply``."""
    map_to_tensors: OrderedDict[str, np.ndarray] = OrderedDict()
    with torch.no_grad():
        positions = model.means.detach().cpu().numpy()
        count = positions.shape[0]
        n = count
        map_to_tensors["x"] = positions[:, 0]
        map_to_tensors["y"] = positions[:, 1]
        map_to_tensors["z"] = positions[:, 2]
        map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
        map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
        map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)

        shs_0 = model.shs_0.contiguous().cpu().numpy()
        for i in range(shs_0.shape[1]):
            map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

        if model.config.sh_degree > 0:
            shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy().reshape((n, -1))
            for i in range(shs_rest.shape[-1]):
                map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]

        map_to_tensors["opacity"] = model.opacities.data.cpu().numpy()
        scales = model.scales.data.cpu().numpy()
        for i in range(3):
            map_to_tensors[f"scale_{i}"] = scales[:, i, None]
        quats = model.quats.data.cpu().numpy()
        for i in range(4):
            map_to_tensors[f"rot_{i}"] = quats[:, i, None]

        select = np.ones(n, dtype=bool)
        for k, t in map_to_tensors.items():
            select = np.logical_and(select, np.isfinite(t).all(axis=-1))
        if np.sum(select) < n:
            for k in map_to_tensors:
                map_to_tensors[k] = map_to_tensors[k][select]
            count = int(np.sum(select))

    ExportGaussianSplat.write_ply(str(filename), count, map_to_tensors)


def save_pruned_ckpt(pipeline: Pipeline, step: int, path: Path) -> None:
    """Save a nerfstudio-style checkpoint after in-place pruning."""
    torch.save({"step": step, "pipeline": pipeline.state_dict()}, path)


def launch_cli(config_cls: Type[HardPruneConfig]) -> None:
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(config_cls).main()
