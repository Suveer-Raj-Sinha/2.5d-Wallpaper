"""
Unit tests for segmentation subsystem and mask utilities.
"""

import numpy as np
import pytest

from engine.segmentation.masks import clean_mask, feather_mask, dilate_mask, compute_bbox
from engine.segmentation.depth_segmenter import DepthSegmenter


def test_mask_utils():
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 20:60] = True

    # compute_bbox
    x, y, w, h = compute_bbox(mask)
    assert (x, y, w, h) == (20, 30, 40, 40)

    # dilate_mask
    dilated = dilate_mask(mask, pixels=3)
    assert np.count_nonzero(dilated) > np.count_nonzero(mask)

    # feather_mask
    feathered = feather_mask(mask, radius=3)
    assert feathered.shape == (100, 100)
    assert feathered.min() >= 0.0
    assert feathered.max() <= 1.0


def test_depth_segmenter():
    h, w = 200, 300
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # Depth map with 3 steps: far (0.1), mid (0.5), near (0.9)
    depth = np.full((h, w), 0.1, dtype=np.float32)
    depth[60:140, :] = 0.5
    depth[140:, :] = 0.9

    segmenter = DepthSegmenter(min_area_ratio=0.01, detect_sky=False)
    segments = segmenter.segment(rgb, depth, num_layers=3)

    assert len(segments) >= 2
    # Verify sorted by median depth
    for i in range(1, len(segments)):
        assert segments[i].median_depth >= segments[i - 1].median_depth
        assert segments[i].area_ratio > 0.0
