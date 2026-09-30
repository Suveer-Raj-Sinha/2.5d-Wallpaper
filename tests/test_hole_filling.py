"""
Unit tests for hole filling, inpainting, and overscan padding.
"""

import numpy as np
import pytest

from engine.reconstruction.hole_filling import inpaint_occlusion, apply_overscan, InpaintMethod
from engine.reconstruction.background import BackgroundReconstructor


def test_inpaint_occlusion():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    hole_mask = np.zeros((100, 100), dtype=bool)
    hole_mask[40:60, 40:60] = True
    img[hole_mask] = 0  # Black hole

    inpainted = inpaint_occlusion(img, hole_mask, method=InpaintMethod.TELEA, inpaint_radius=5)
    assert inpainted.shape == (100, 100, 3)
    # The hole center should no longer be pure black
    assert inpainted[50, 50, 0] > 50


def test_apply_overscan():
    img = np.full((100, 100, 4), 255, dtype=np.uint8)
    overscan, scale = apply_overscan(img, overscan_margin=0.10)
    assert scale > 1.0
    assert overscan.shape[0] > 100
    assert overscan.shape[1] > 100
    assert overscan.shape[2] == 4


def test_background_reconstructor():
    img = np.full((120, 120, 3), 200, dtype=np.uint8)
    mask1 = np.zeros((120, 120), dtype=bool)
    mask1[30:50, 30:50] = True
    img[mask1] = 10

    reconstructor = BackgroundReconstructor()
    bg = reconstructor.reconstruct(img, [mask1])
    assert bg.shape == (120, 120, 3)
    assert bg[40, 40, 0] > 50
