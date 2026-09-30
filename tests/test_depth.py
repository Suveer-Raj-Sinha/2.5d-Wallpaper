"""
Unit tests for Depth Engine and preprocessing utilities.
"""

import numpy as np
import pytest
from pathlib import Path

from engine.depth.base import DepthResult
from engine.depth.preprocessing import normalize_depth, smooth_depth, load_image_rgb
from engine.depth.fallback import HeuristicDepthEngine


def test_normalize_depth():
    raw = np.array([[-10.0, 50.0], [100.0, 200.0]], dtype=np.float32)
    norm = normalize_depth(raw)
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0
    assert norm.dtype == np.float32

    # Inverted
    norm_inv = normalize_depth(raw, invert=True)
    assert norm_inv.min() >= 0.0
    assert norm_inv.max() <= 1.0


def test_depth_result_16bit_roundtrip(tmp_path: Path):
    h, w = 100, 150
    # Gradient depth
    depth_grid = np.linspace(0.0, 1.0, h * w, dtype=np.float32).reshape(h, w)
    res = DepthResult(depth_map=depth_grid, original_size=(w, h))

    u16 = res.uint16_depth
    assert u16.shape == (h, w)
    assert u16.dtype == np.uint16
    assert u16[0, 0] == 0
    assert u16[-1, -1] == 65535

    # Save and reload
    save_file = tmp_path / "test_depth.png"
    res.save_16bit(save_file)
    assert save_file.exists()

    reloaded = DepthResult.from_file(save_file)
    assert reloaded.original_size == (w, h)
    assert np.allclose(reloaded.depth_map, depth_grid, atol=1e-4)


def test_heuristic_depth_engine():
    engine = HeuristicDepthEngine()
    test_img = np.zeros((120, 160, 3), dtype=np.uint8)
    # Put bright object in center
    test_img[40:80, 50:110] = [255, 200, 50]

    result = engine.estimate(test_img)
    assert result.depth_map.shape == (120, 160)
    assert result.depth_map.min() >= 0.0
    assert result.depth_map.max() <= 1.0
    # Top background should be farther (lower depth) than bottom/center
    assert result.depth_map[5, 5] < result.depth_map[60, 80]
