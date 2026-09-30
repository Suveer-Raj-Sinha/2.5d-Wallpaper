"""
Virtual camera model for 2.5D/3D parallax rendering.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Camera:
    """
    Virtual camera parameters controlling viewport perspective and parallax dynamics.
    """
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    rotation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    fov: float = 45.0  # degrees
    parallax_strength: float = 0.08  # global multiplier for layer displacement
    limits: Tuple[float, float, float, float] = (-0.5, 0.5, -0.5, 0.5)  # min_x, max_x, min_y, max_y
    focal_depth: float = 0.5  # depth plane that remains stationary during parallax (0=far, 1=near)

    def clamp_position(self):
        """Clamp camera X and Y within safety bounds."""
        min_x, max_x, min_y, max_y = self.limits
        self.position[0] = max(min_x, min(max_x, self.position[0]))
        self.position[1] = max(min_y, min(max_y, self.position[1]))
        self.position[2] = max(0.2, min(3.0, self.position[2]))  # zoom limits
