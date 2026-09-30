"""
Scene visual effects definitions (particles, fog, lighting).
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Effect:
    """Base visual effect definition attached to a 4DW scene."""
    type: str  # "fog", "particles", "lighting", "bloom"
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
