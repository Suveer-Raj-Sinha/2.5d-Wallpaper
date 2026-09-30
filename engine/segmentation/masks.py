"""
Mask processing, refinement, feathering, and morphological cleaning.
"""

from typing import Tuple
import numpy as np
import cv2


def clean_mask(
    mask: np.ndarray,
    min_area_ratio: float = 0.005,
    morph_kernel_size: int = 5,
) -> np.ndarray:
    """
    Remove small pinhole noise, close internal gaps, and remove tiny speckles.
    """
    u8 = (mask > 0).astype(np.uint8) * 255
    h, w = u8.shape[:2]
    min_pixels = int(h * w * min_area_ratio)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    # Close small holes inside segments
    closed = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kernel)
    # Open to remove small outside speckles
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # Filter connected components by size
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    cleaned = np.zeros_like(opened)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_pixels:
            cleaned[labels == i] = 255

    return cleaned > 0


def feather_mask(
    mask: np.ndarray,
    radius: int = 3,
    sigma: float = 1.5,
) -> np.ndarray:
    """
    Produce smooth anti-aliased alpha boundary transitions [0.0, 1.0].
    """
    u8 = (mask > 0).astype(np.uint8) * 255
    if radius <= 0:
        return (u8 / 255.0).astype(np.float32)

    ksize = radius * 2 + 1
    blurred = cv2.GaussianBlur(u8.astype(np.float32), (ksize, ksize), sigma)
    alpha = np.clip(blurred / 255.0, 0.0, 1.0)
    return alpha


def dilate_mask(mask: np.ndarray, pixels: int = 3) -> np.ndarray:
    """Dilate mask by a given pixel radius for overlap and hole-filling."""
    if pixels <= 0:
        return mask
    u8 = (mask > 0).astype(np.uint8) * 255
    ksize = pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(u8, kernel)
    return dilated > 0


def compute_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Calculate (x, y, width, height) bounding box of positive mask pixels."""
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return 0, 0, 0, 0
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)
