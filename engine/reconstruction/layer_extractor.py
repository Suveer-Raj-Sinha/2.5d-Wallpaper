"""
Layer extractor: constructs multi-depth RGBA layers from segmentation masks
and depth data, with alpha feathering and occlusion inpainting.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2

from engine.segmentation.base import Segment
from engine.segmentation.masks import feather_mask, dilate_mask
from engine.reconstruction.hole_filling import inpaint_occlusion, apply_overscan, InpaintMethod
from engine.reconstruction.background import BackgroundReconstructor


@dataclass
class ExtractedLayer:
    """In-memory representation of an extracted scene layer."""
    id: str
    name: str
    texture_rgba: np.ndarray  # (H, W, 4) uint8
    depth: float  # [0.0, 1.0], 0.0 = furthest, 1.0 = nearest
    parallax_factor: float  # relative displacement multiplier
    scale: Tuple[float, float] = (1.0, 1.0)
    offset: Tuple[float, float] = (0.0, 0.0)
    visible: bool = True
    mask: Optional[np.ndarray] = None


class LayerExtractor:
    """
    Transforms segmented scene regions into depth-aware 2.5D visual layers
    with alpha transparency and occlusion reconstruction.
    """

    def __init__(
        self,
        base_parallax: float = 0.08,
        feather_radius: int = 3,
        dilation_pixels: int = 2,
        overscan_margin: float = 0.08,
        inpaint_method: InpaintMethod = InpaintMethod.LAMA,
    ):
        self.base_parallax = base_parallax
        self.feather_radius = feather_radius
        self.dilation_pixels = dilation_pixels
        self.overscan_margin = overscan_margin
        self.inpaint_method = inpaint_method
        self.bg_reconstructor = BackgroundReconstructor(inpaint_method=inpaint_method)

    def extract(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        segments: List[Segment],
    ) -> List[ExtractedLayer]:
        """
        Extract ordered list of layers from background to foreground.
        """
        if not segments:
            raise ValueError("No segments provided to LayerExtractor.")

        # Iterative inpainting from nearest to furthest
        # 1. Reverse sort segments to go from Foreground -> Background
        h, w = rgb_image.shape[:2]
        extracted_layers: List[ExtractedLayer] = []
        
        # We need original sorting for parallax calculations
        depths = [s.median_depth for s in segments]
        min_depth = min(depths)
        max_depth = max(depths)
        depth_range = max(1e-4, max_depth - min_depth)
        
        # Sort furthest to nearest for parallax math, but we iterate backwards
        sorted_segs = sorted(segments, key=lambda s: s.median_depth)
        
        # We maintain a running RGB image that has all foregrounds inpainted out
        current_rgb = rgb_image.copy()
        
        # To avoid running expensive LAMA for every single layer (which takes forever),
        # we can use EDGE_BLEED/TELEA for the midground, and only use LAMA for the final base background
        from engine.reconstruction.hole_filling import inpaint_occlusion, InpaintMethod
        
        # Process from foreground (nearest) down to background (furthest)
        for i in range(len(sorted_segs) - 1, -1, -1):
            seg = sorted_segs[i]
            is_base_background = (i == 0)
            
            norm_depth_ratio = (seg.median_depth - min_depth) / depth_range
            parallax = 0.15 + (norm_depth_ratio * 0.85)
            
            if is_base_background:
                # The running current_rgb has ALREADY had all foreground layers inpainted out of it!
                rgba = np.dstack([current_rgb, np.full((h, w), 255, dtype=np.uint8)])
            else:
                # Extract the current layer from the running RGB (which has no foregrounds in front of it)
                dilated = dilate_mask(seg.mask, pixels=self.dilation_pixels)
                alpha_norm = feather_mask(dilated, radius=self.feather_radius)
                alpha_u8 = (alpha_norm * 255.0).astype(np.uint8)
                rgba = np.dstack([current_rgb.copy(), alpha_u8])
                
                # IMPORTANT: Inpaint this layer out of the running RGB image so layers BEHIND it don't duplicate it.
                # Dilate the mask slightly before inpainting to ensure we erase the feathered edges completely.
                occlusion_mask = dilate_mask(seg.mask, pixels=self.feather_radius + 3)
                
                # Use LAMA (or selected method) for ALL layers to prevent blurry TELEA "empty spaces"
                current_rgb = inpaint_occlusion(
                    current_rgb, 
                    occlusion_mask > 0, 
                    method=self.inpaint_method, 
                    inpaint_radius=self.dilation_pixels
                )
                
            # Apply overscan to ALL layers to prevent hard screen-edge cutoffs when panning
            if self.overscan_margin > 0.0:
                rgba, scale_factor = apply_overscan(rgba, self.overscan_margin)
                scale = (scale_factor, scale_factor)
            else:
                scale = (1.0, 1.0)
                
            extracted_layers.append(ExtractedLayer(
                id=seg.id, name=seg.name, texture_rgba=rgba,
                depth=float(seg.median_depth), parallax_factor=float(parallax),
                scale=scale, offset=(0.0, 0.0), visible=True, mask=seg.mask,
            ))

        # Restore the original far-to-near ordering
        extracted_layers.reverse()
        return extracted_layers
