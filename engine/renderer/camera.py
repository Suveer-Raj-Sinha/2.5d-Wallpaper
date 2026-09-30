"""
Interactive smooth camera controller with mouse tracking, damping,
and smartphone gyroscope sway simulation.
"""

import math
import time
from typing import Tuple, List


class SmoothCameraController:
    """
    Manages camera coordinates with smooth spring-damper interpolation,
    mouse input mapping, and automatic live gyroscope simulation.
    """

    def __init__(
        self,
        parallax_strength: float = 0.08,
        smoothing: float = 0.15,
        limits: Tuple[float, float, float, float] = (-0.4, 0.4, -0.4, 0.4),
        sensitivity: float = 1.0,
        damping: float = 15.0,
    ):
        self.parallax_strength = parallax_strength
        self.smoothing = smoothing
        self.limits = limits
        self.sensitivity = sensitivity
        self.damping = damping

        # Target camera position [x, y, z]
        self.target_x: float = 0.0
        self.target_y: float = 0.0
        self.target_z: float = 1.0  # 1.0 = default zoom

        # Current interpolated position
        self.current_x: float = 0.0
        self.current_y: float = 0.0
        self.current_z: float = 1.0

        # Gyroscope simulation state
        self.simulate_gyro: bool = False
        self._gyro_start_time: float = time.time()
        self.gyro_speed: float = 1.2
        self.gyro_amplitude_x: float = 0.18
        self.gyro_amplitude_y: float = 0.12

        # Exploded 3D view factor (0.0 = normal, 1.0 = fully separated layers)
        self.target_explode: float = 0.0
        self.current_explode: float = 0.0

    def reset(self):
        """Reset camera to default centered position."""
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 1.0
        self.target_explode = 0.0

    def on_mouse_move(self, norm_x: float, norm_y: float):
        """
        Handle normalized mouse coordinates [-1.0, 1.0].
        """
        if not self.simulate_gyro:
            min_x, max_x, min_y, max_y = self.limits
            scale = 0.35 * self.sensitivity
            self.target_x = max(min_x, min(max_x, norm_x * scale))
            self.target_y = max(min_y, min(max_y, -norm_y * scale))

    def on_scroll(self, dy: float):
        """Handle mouse wheel zoom."""
        self.target_z = max(0.5, min(2.5, self.target_z - dy * 0.1))

    def toggle_gyro(self) -> bool:
        """Toggle automatic phone gyroscope sway simulation."""
        self.simulate_gyro = not self.simulate_gyro
        if self.simulate_gyro:
            self._gyro_start_time = time.time()
        return self.simulate_gyro

    def update(self, dt: float):
        """Update smooth interpolation."""
        if self.simulate_gyro:
            t = (time.time() - self._gyro_start_time) * self.gyro_speed
            # Figure-eight Lissajous curve mimicking natural hand tilt
            self.target_x = math.sin(t) * self.gyro_amplitude_x * self.sensitivity
            self.target_y = math.sin(t * 2.0) * (self.gyro_amplitude_y * 0.5 * self.sensitivity)

        # Smooth exponential approach (lerp)
        alpha = min(1.0, 1.0 - math.exp(-dt * self.damping))
        self.current_x += (self.target_x - self.current_x) * alpha
        self.current_y += (self.target_y - self.current_y) * alpha
        self.current_z += (self.target_z - self.current_z) * alpha
        self.current_explode += (self.target_explode - self.current_explode) * alpha

    @property
    def position(self) -> Tuple[float, float, float]:
        return self.current_x, self.current_y, self.current_z
