"""
Unit tests for scene format serialization, deserialization, and .4dw packaging.
"""

from pathlib import Path
import numpy as np
import pytest

from engine.scene.camera import Camera
from engine.scene.layer import Layer
from engine.scene.scene import Canvas, Scene
from engine.depth.base import DepthResult
from engine.format.scene_format import scene_to_manifest, manifest_to_scene
from engine.format.package import Package4DW


def test_manifest_roundtrip():
    scene = Scene(
        canvas=Canvas(width=1440, height=2560),
        camera=Camera(fov=50.0, parallax_strength=0.12),
        layers=[
            Layer(id="l0", name="Background", texture="layers/l0.png", depth=0.1, parallax=0.2),
            Layer(id="l1", name="Subject", texture="layers/l1.png", depth=0.8, parallax=1.0),
        ],
        metadata={"author": "Test"},
    )

    manifest = scene_to_manifest(scene)
    assert manifest.canvas.width == 1440
    assert len(manifest.layers) == 2
    assert manifest.camera.fov == 50.0

    restored_scene = manifest_to_scene(manifest)
    assert restored_scene.canvas.width == 1440
    assert len(restored_scene.layers) == 2
    assert restored_scene.layers[1].depth == 0.8
    assert restored_scene.layers[1].parallax == 1.0


def test_package_4dw_lifecycle(tmp_path: Path):
    h, w = 80, 120
    color = np.full((h, w, 3), 180, dtype=np.uint8)
    depth_arr = np.linspace(0.0, 1.0, h * w, dtype=np.float32).reshape(h, w)
    depth_res = DepthResult(depth_map=depth_arr, original_size=(w, h))

    scene = Scene(
        canvas=Canvas(width=w, height=h),
        layers=[
            Layer(id="bg", name="BG", texture="layers/bg.png", depth=0.1),
            Layer(id="fg", name="FG", texture="layers/fg.png", depth=0.9),
        ],
    )

    layer_textures = {
        "layers/bg.png": np.full((h, w, 4), 255, dtype=np.uint8),
        "layers/fg.png": np.full((h, w, 4), 200, dtype=np.uint8),
    }

    masks = {
        "masks/bg.png": np.ones((h, w), dtype=bool),
        "masks/fg.png": np.ones((h, w), dtype=bool),
    }

    pkg_path = tmp_path / "test_scene.4dw"
    Package4DW.save(
        scene=scene,
        output_path=pkg_path,
        color_image_rgb=color,
        depth_result=depth_res,
        layer_textures=layer_textures,
        masks=masks,
    )

    assert pkg_path.exists()

    # Test inspect
    info = Package4DW.inspect(pkg_path)
    assert info["format"] == "4DW"
    assert info["num_layers"] == 2
    assert info["canvas"]["width"] == w

    # Test load
    with Package4DW.load(pkg_path) as pkg:
        assert len(pkg.scene.layers) == 2
        bg_rgba = pkg.read_image_rgba("layers/bg.png")
        assert bg_rgba.shape == (h, w, 4)
        loaded_depth = pkg.read_depth_map()
        assert loaded_depth.original_size == (w, h)
