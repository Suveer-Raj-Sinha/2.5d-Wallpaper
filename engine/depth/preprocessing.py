"""
Preprocessing and postprocessing utilities for depth estimation.
"""

from pathlib import Path
from typing import Union, Tuple
import numpy as np
from PIL import Image
import cv2


def load_image_rgb(image_input: Union[np.ndarray, Image.Image, str, Path]) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Load an image as RGB uint8 numpy array (H, W, 3).
    Returns (rgb_array, (width, height)).
    """
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {img_path}")
        pil_img = Image.open(str(img_path)).convert("RGB")
        arr = np.array(pil_img, dtype=np.uint8)
        w, h = pil_img.size
        return arr, (w, h)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
        w, h = pil_img.size
        return np.array(pil_img, dtype=np.uint8), (w, h)
    elif isinstance(image_input, np.ndarray):
        arr = image_input
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            # Assume RGB if uint8
            pass
        else:
            raise ValueError(f"Unsupported image array shape: {arr.shape}")
        h, w = arr.shape[:2]
        return arr.astype(np.uint8), (w, h)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")


def normalize_depth(
    raw_depth: np.ndarray,
    invert: bool = False,
    percentile_min: float = 1.0,
    percentile_max: float = 99.0,
) -> np.ndarray:
    """
    Robust min-max normalization to [0.0, 1.0] using percentiles to eliminate extreme outliers.
    Convention: 0.0 = furthest, 1.0 = nearest.
    """
    raw_float = raw_depth.astype(np.float32)

    # Use percentiles for robust scaling against outliers
    p_min = float(np.percentile(raw_float, percentile_min))
    p_max = float(np.percentile(raw_float, percentile_max))

    if p_max - p_min > 1e-6:
        normalized = (raw_float - p_min) / (p_max - p_min)
        normalized = np.clip(normalized, 0.0, 1.0)
    else:
        normalized = np.zeros_like(raw_float)

    if invert:
        normalized = 1.0 - normalized

    return normalized.astype(np.float32)


def smooth_depth(
    depth_map: np.ndarray,
    rgb_guide: np.ndarray = None,
    diameter: int = 5,
    sigma_color: float = 0.1,
    sigma_space: float = 5.0,
) -> np.ndarray:
    """
    Apply edge-preserving bilateral filtering to smooth noise while keeping object boundaries crisp.
    If rgb_guide is provided, uses joint bilateral filtering (depth edge guided by RGB edges).
    """
    depth_float = np.clip(depth_map, 0.0, 1.0).astype(np.float32)

    if rgb_guide is not None and hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "jointBilateralFilter"):
        try:
            guide = rgb_guide.astype(np.uint8)
            refined = cv2.ximgproc.jointBilateralFilter(
                guide, depth_float, diameter, sigma_color, sigma_space
            )
            return np.clip(refined, 0.0, 1.0)
        except Exception:
            pass

    # Standard bilateral filter on single channel float
    filtered = cv2.bilateralFilter(depth_float, diameter, sigma_color, sigma_space)
    return np.clip(filtered, 0.0, 1.0).astype(np.float32)
