from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Type
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig, Nerfstudio


@dataclass
class QEDSplatterDataParserConfig(NerfstudioDataParserConfig):
    _target: Type = field(default_factory=lambda: QEDSplatterDataParser)
    depth_unit_scale_factor: float = 1
    # auto_scale_poses: bool = False  # If True, poses will be scaled to max extent of 1
    # center_method: Literal["poses", "focus", "none"] = "none"  # Centering method
    # orientation_method: Literal["pca", "up", "vertical", "none"] = "none"  # Orientation method

@dataclass
class QEDSplatterDataParser(Nerfstudio):
    config: QEDSplatterDataParserConfig
