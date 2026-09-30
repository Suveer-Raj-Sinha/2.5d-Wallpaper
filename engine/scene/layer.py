"""
Scene layer definition.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Layer:
    """
    Independent 2.5D visual layer positioned at a specific depth with parallax properties.
    """
    id: str
    name: str
    texture: str  # Relative archive path (e.g. "layers/layer_0_sky.png")
    depth: float  # Normalized depth in [0.0, 1.0] (0=far, 1=near)
    parallax: float = 1.0  # Layer-specific parallax response multiplier
    position: List[float] = field(default_factory=lambda: [0.0, 0.0])  # [offset_x, offset_y]
    scale: List[float] = field(default_factory=lambda: [1.0, 1.0])  # [scale_x, scale_y]
    rotation: float = 0.0  # degrees
    opacity: float = 1.0  # [0.0, 1.0]
    visible: bool = True
    blend_mode: str = "normal"  # "normal", "additive", "multiply"
    mask: Optional[str] = None  # Relative archive path to mask (e.g. "masks/layer_0_sky.png")
    animation: Dict[str, Any] = field(default_factory=dict)
