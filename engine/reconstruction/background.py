"""
Background reconstruction subsystem.
Ensures seamless, hole-free infinite backdrop coverage behind foreground elements.
"""

from typing import List
import numpy as np

from engine.reconstruction.hole_filling import inpaint_occlusion, InpaintMethod


class BackgroundReconstructor:
    """
    Reconstructs continuous full-screen background layers
    by filling occluded regions left by foreground objects.
    """

    def __init__(self, inpaint_method: InpaintMethod = InpaintMethod.TELEA, radius: int = 7):
        self.inpaint_method = inpaint_method
        self.radius = radius

    def reconstruct(
        self,
        rgb_image: np.ndarray,
        occlusion_masks: List[np.ndarray],
    ) -> np.ndarray:
        """
        Synthesize full background by inpainting the combined occlusions of all foreground layers.

        Args:
            rgb_image: Original RGB image (H, W, 3)
            occlusion_masks: List of boolean masks of foreground objects that occlude the background

        Returns:
            Clean RGB background without foreground cutouts
        """
        if not occlusion_masks:
            return rgb_image.copy()

        # Combine all foreground masks
        h, w = rgb_image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=bool)
        for m in occlusion_masks:
            combined_mask = combined_mask | (m > 0)

        if np.count_nonzero(combined_mask) == 0:
            return rgb_image.copy()

        # Inpaint foreground occlusions
        inpainted_bg = inpaint_occlusion(
            rgb_image,
            combined_mask,
            method=self.inpaint_method,
            inpaint_radius=self.radius,
        )
        return inpainted_bg
