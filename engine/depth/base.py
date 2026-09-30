"""
Base depth estimation interface and data structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Tuple, Optional
import numpy as np
from PIL import Image
import cv2


@dataclass
class DepthResult:
    """
    Standardized depth estimation result.

    Convention:
      0.0 = furthest background (infinity)
      1.0 = nearest foreground
    """
    # Normalized depth in range [0.0, 1.0], shape (H, W), dtype float32
    depth_map: np.ndarray
    original_size: Tuple[int, int]  # (width, height)
    confidence: Optional[np.ndarray] = None

    @property
    def uint16_depth(self) -> np.ndarray:
        """Convert normalized depth [0.0, 1.0] to 16-bit unsigned integer [0, 65535]."""
        clipped = np.clip(self.depth_map, 0.0, 1.0)
        return (clipped * 65535.0).astype(np.uint16)

    @property
    def uint8_depth(self) -> np.ndarray:
        """Convert normalized depth [0.0, 1.0] to 8-bit unsigned integer [0, 255]."""
        clipped = np.clip(self.depth_map, 0.0, 1.0)
        return (clipped * 255.0).astype(np.uint8)

    def save_16bit(self, filepath: Union[str, Path]) -> Path:
        """Save depth map as 16-bit grayscale PNG."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self.uint16_depth, [cv2.IMWRITE_PNG_COMPRESSION, 4])
        return path

    def save_colormapped(self, filepath: Union[str, Path], colormap: int = cv2.COLORMAP_INFERNO) -> Path:
        """Save depth map as 8-bit colormapped preview image."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        colored = cv2.applyColorMap(self.uint8_depth, colormap)
        cv2.imwrite(str(path), colored)
        return path

    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> "DepthResult":
        """Load depth map from an existing 16-bit or 8-bit image file."""
        img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to read depth map from {filepath}")

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img.dtype == np.uint16:
            depth_map = img.astype(np.float32) / 65535.0
        else:
            depth_map = img.astype(np.float32) / 255.0

        h, w = depth_map.shape[:2]
        return cls(depth_map=depth_map, original_size=(w, h))


class BaseDepthEngine(ABC):
    """Abstract base class for monocular depth estimation models."""

    @abstractmethod
    def estimate(self, image: Union[np.ndarray, Image.Image, str, Path]) -> DepthResult:
        """
        Estimate a depth map from an input RGB image.

        Args:
            image: Input RGB image (np.ndarray HWC uint8, PIL Image, or path)

        Returns:
            DepthResult with normalized depth in range [0.0, 1.0] (0=far, 1=near)
        """
        pass
