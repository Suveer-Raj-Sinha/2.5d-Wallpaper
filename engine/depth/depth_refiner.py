import cv2
import numpy as np
from typing import Optional

def refine_depth_map(depth_map: np.ndarray, rgb_img: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Refines the depth map by applying edge-preserving smoothing.
    If RGB is provided, uses guided filtering or bilateral filtering.
    """
    # depth_map is float32 [0.0, 1.0]
    depth_u8 = (np.clip(depth_map, 0.0, 1.0) * 255.0).astype(np.uint8)
    
    if rgb_img is not None:
        # Use Guided Filter (implemented via ximgproc if available, else bilateral)
        try:
            # OpenCV extended image processing module
            refined_u8 = cv2.ximgproc.guidedFilter(
                guide=rgb_img, 
                src=depth_u8, 
                radius=15, 
                eps=0.01
            )
        except AttributeError:
            # Fallback to bilateral filtering
            refined_u8 = cv2.bilateralFilter(depth_u8, d=9, sigmaColor=75, sigmaSpace=75)
    else:
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(depth_u8, cv2.MORPH_CLOSE, kernel)
        refined_u8 = cv2.bilateralFilter(closed, d=9, sigmaColor=75, sigmaSpace=75)
        
    return refined_u8.astype(np.float32) / 255.0
