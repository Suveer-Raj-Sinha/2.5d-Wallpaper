"""
Scene object representation for isolated segmented items.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class SceneObject:
    """
    Salient extracted object that can have independent physics, animation, or interactivity.
    """
    id: str
    name: str
    type: str = "object"  # "character", "vehicle", "prop", "foliage", "object"
    depth: float = 0.5
    texture: Optional[str] = None
    mask: Optional[str] = None
    bbox: List[int] = field(default_factory=lambda: [0, 0, 0, 0])  # [x, y, w, h]
    metadata: Dict[str, Any] = field(default_factory=dict)
