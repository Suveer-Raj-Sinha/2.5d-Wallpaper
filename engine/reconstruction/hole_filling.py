"""
Occlusion handling and hole-filling strategies for multi-layer parallax scenes.
"""

from enum import Enum
from typing import Tuple, Optional
import numpy as np
import cv2

from engine.segmentation.masks import dilate_mask, feather_mask


class InpaintMethod(str, Enum):
    TELEA = "telea"
    NAVIER_STOKES = "ns"
    EDGE_BLEED = "edge_bleed"
    LAMA = "lama"
    NONE = "none"


def inpaint_occlusion(
    image_rgb: np.ndarray,
    hole_mask: np.ndarray,
    method: InpaintMethod = InpaintMethod.TELEA,
    inpaint_radius: int = 5,
) -> np.ndarray:
    """
    Inpaint missing/occluded regions of an image.

    Args:
        image_rgb: (H, W, 3) RGB uint8
        hole_mask: (H, W) boolean or uint8 mask where True/255 represents occluded area to fill
        method: Inpainting algorithm to use
        inpaint_radius: Pixel radius for neighborhood sampling

    Returns:
        (H, W, 3) RGB uint8 reconstructed image
    """
    u8_mask = (hole_mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(u8_mask) == 0 or method == InpaintMethod.NONE:
        return image_rgb.copy()

    # Convert RGB to BGR for OpenCV
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if method == InpaintMethod.TELEA:
        inpainted_bgr = cv2.inpaint(bgr, u8_mask, inpaint_radius, cv2.INPAINT_TELEA)
    elif method == InpaintMethod.NAVIER_STOKES:
        inpainted_bgr = cv2.inpaint(bgr, u8_mask, inpaint_radius, cv2.INPAINT_NS)
    elif method == InpaintMethod.EDGE_BLEED:
        # Fast push-pull or repeated dilation for large occlusions
        dilated = bgr.copy()
        current_mask = u8_mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for _ in range(8):
            if np.count_nonzero(current_mask) == 0:
                break
            blurred = cv2.GaussianBlur(dilated, (7, 7), 0)
            dilated[current_mask > 0] = blurred[current_mask > 0]
            current_mask = cv2.erode(current_mask, kernel)
        inpainted_bgr = dilated
    elif method == InpaintMethod.LAMA:
        try:
            from engine.reconstruction.lama_inpainter import get_lama_inpainter
            inpainter = get_lama_inpainter()
            return inpainter.inpaint(image_rgb, u8_mask)
        except Exception as e:
            print(f"Warning: LAMA model failed to run: {e}. Falling back to EDGE_BLEED.")
            dilated = bgr.copy()
            current_mask = u8_mask.copy()
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            for _ in range(8):
                if np.count_nonzero(current_mask) == 0:
                    break
                blurred = cv2.GaussianBlur(dilated, (7, 7), 0)
                dilated[current_mask > 0] = blurred[current_mask > 0]
                current_mask = cv2.erode(current_mask, kernel)
            inpainted_bgr = dilated
    else:
        inpainted_bgr = bgr

    return cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)


def apply_overscan(
    rgba_image: np.ndarray,
    overscan_margin: float = 0.08,
) -> Tuple[np.ndarray, float]:
    """
    Expand texture slightly beyond standard canvas bounds (overscan)
    so parallax camera shifts do not expose empty borders.

    Args:
        rgba_image: (H, W, 4) RGBA uint8
        overscan_margin: Fractional margin expansion (e.g. 0.08 = 8% larger)

    Returns:
        (expanded_rgba, scale_factor)
    """
    if overscan_margin <= 0.0:
        return rgba_image, 1.0

    h, w = rgba_image.shape[:2]
    new_w = int(w * (1.0 + overscan_margin * 2))
    new_h = int(h * (1.0 + overscan_margin * 2))

    # Resize with cubic interpolation
    expanded = cv2.resize(rgba_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    scale = (1.0 + overscan_margin * 2)
    return expanded, scale
