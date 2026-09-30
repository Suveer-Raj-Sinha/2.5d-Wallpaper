"""
Improved heuristic depth estimator that produces better results for
complex real-world scenes (anime, cityscapes, outdoor photography).
Combines multiple depth cues: gradient, color temperature, blur radius,
saturation, vertical position, and edge contrast.
"""

from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image
import cv2

from engine.depth.base import BaseDepthEngine, DepthResult
from engine.depth.preprocessing import load_image_rgb, normalize_depth, smooth_depth


class HeuristicDepthEngine(BaseDepthEngine):
    """
    Multi-cue heuristic depth estimator for offline/testing environments.
    Produces plausible depth separation for natural scenes without neural weights.
    """

    def __init__(self, smoothing: bool = True, smoothing_passes: int = 2):
        self.smoothing = smoothing
        self.smoothing_passes = smoothing_passes

    def estimate(self, image: Union[np.ndarray, Image.Image, str, Path]) -> DepthResult:
        rgb, (w, h) = load_image_rgb(image)

        # ---- Cue 1: Vertical perspective gradient (strong prior: top=far, bottom=near)
        y_coords = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        vertical_cue = np.tile(y_coords, (1, w))

        # ---- Cue 2: Sky/horizon detection via blue-grey dominance in upper region
        # Sky pixels should be mapped to VERY far depth (0.0)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        h_chan = hsv[:, :, 0] / 180.0
        s_chan = hsv[:, :, 1] / 255.0
        v_chan = hsv[:, :, 2] / 255.0

        # Sky heuristic: low saturation OR blue hue, in upper 60% of image
        sky_candidate = (s_chan < 0.35) | ((h_chan > 0.5) & (h_chan < 0.8) & (s_chan < 0.7))
        upper_region = np.zeros((h, w), dtype=bool)
        upper_region[:int(h * 0.62), :] = True
        sky_mask = sky_candidate & upper_region

        # ---- Cue 3: Color temperature (cool=far, warm=near)
        r_norm = rgb[:, :, 0].astype(np.float32) / 255.0
        b_norm = rgb[:, :, 2].astype(np.float32) / 255.0
        # Red-blue ratio: more red -> nearer (warm), more blue -> farther (cool)
        color_temp_cue = np.clip((r_norm - b_norm + 1.0) / 2.0, 0.0, 1.0)

        # ---- Cue 4: Local contrast / texture energy (high frequency = near objects)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        edge_energy = cv2.GaussianBlur(laplacian, (31, 31), 0)
        edge_cue = normalize_depth(edge_energy)

        # ---- Cue 5: Saturation cue (vivid, saturated objects tend to be foreground)
        sat_cue = s_chan.copy()

        # ---- Cue 6: Focus/blur estimation (blurry = far in natural scenes)
        blur_response = cv2.GaussianBlur(laplacian, (11, 11), 0)
        blur_cue = normalize_depth(blur_response)

        # ---- Cue 7: Value / luminance contrast with neighborhood
        local_mean = cv2.blur(v_chan, (41, 41))
        lum_contrast = np.abs(v_chan - local_mean)
        lum_cue = normalize_depth(lum_contrast)

        # ---- Combine all cues with weighted blending
        # Reduced vertical_cue weight — it was causing a hard horizontal cliff
        # that made the depth map look like two flat slabs (sky=black, scene=orange).
        raw_depth = (
            vertical_cue   * 0.20  +  # soft perspective prior (was 0.40)
            color_temp_cue * 0.15  +
            edge_cue       * 0.30  +  # texture/focus is the most reliable cue
            sat_cue        * 0.12  +
            blur_cue       * 0.15  +
            lum_cue        * 0.08
        )

        # Sky region hard-coded to far depth
        raw_depth[sky_mask] = raw_depth[sky_mask] * 0.25  # push sky very far

        # Normalize
        norm = normalize_depth(raw_depth)

        # Re-enforce sky at near-zero depth
        if np.any(sky_mask):
            sky_depth_vals = norm[sky_mask]
            sky_max = np.percentile(sky_depth_vals, 85)
            norm[sky_mask] = np.minimum(norm[sky_mask], sky_max * 0.25)

        # Smoothing passes for cleaner layer boundaries
        if self.smoothing:
            for _ in range(self.smoothing_passes):
                norm = smooth_depth(norm, rgb)

        return DepthResult(depth_map=norm, original_size=(w, h))
