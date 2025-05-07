from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Type
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig, Nerfstudio


@dataclass
class QEDSplatterDataParserConfig(NerfstudioDataParserConfig):
    _target: Type = field(default_factory=lambda: QEDSplatterDataParser)
    depth_unit_scale_factor: float = 0.001
    auto_scale_poses: bool = True  # If True, poses will be scaled to max extent of 1
    center_method: Literal["poses", "focus", "none"] = "poses"  # Centering method
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"  # Orientation method

@dataclass
class QEDSplatterDataParser(Nerfstudio):
    config: QEDSplatterDataParserConfig
