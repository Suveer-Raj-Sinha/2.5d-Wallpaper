"""
Base segmentation interfaces and data structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union, Optional

import numpy as np
import cv2


@dataclass
class Segment:
    """
    Represents a segmented region/layer extracted from a 2.5D scene.

    Fields
    ------
    mask       : uint8 HxW array (values 0..255, feathered at edges)
    label      : human-readable name e.g. "sky", "far", "near", "foreground"
    depth_mean : mean depth value [0=far, 1=near]
    depth_std  : standard deviation of depth within the segment
    parallax_factor : signed parallax strength [-1=far, +1=near]
    bbox       : (x, y, w, h) bounding box in pixels
    layer_index: ordering index (0 = furthest)
    metadata   : arbitrary extra data
    """
    mask: np.ndarray                          # HxW uint8, 0–255
    label: str
    depth_mean: float
    depth_std: float = 0.0
    parallax_factor: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    layer_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Convenience aliases (backward compat) ─────────────────────────────
    @property
    def id(self) -> str:
        """Stable string identifier derived from label + layer_index."""
        return f"{self.label}_{self.layer_index:02d}"

    @property
    def name(self) -> str:
        return self.label

    @property
    def mean_depth(self) -> float:
        return self.depth_mean

    @property
    def median_depth(self) -> float:
        return self.depth_mean  # approximation

    @property
    def area_ratio(self) -> float:
        total = self.mask.size
        return float((self.mask > 0).sum()) / total if total > 0 else 0.0

    @property
    def uint8_mask(self) -> np.ndarray:
        """Binary uint8 mask (0 or 255)."""
        return np.where(self.mask > 127, 255, 0).astype(np.uint8)

    def save_mask(self, filepath: Union[str, Path]) -> Path:
        """Save mask as PNG."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self.mask)
        return path


class BaseSegmenter(ABC):
    """Abstract base for depth-guided segmenters."""

    @abstractmethod
    def segment(
        self,
        depth_map: np.ndarray,
        original_image: Optional[np.ndarray] = None,
    ) -> List[Segment]:
        """
        Segment a depth map into distinct scene layers.

        Args:
            depth_map:      Float32 HxW in [0=far, 1=near].
            original_image: Optional HxWx3 uint8 RGB for colour-guided refinement.

        Returns:
            List of Segment objects ordered from furthest (index 0) to nearest.
        """
        ...
