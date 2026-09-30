"""
End-to-end integration test for full ConversionPipeline.
"""

from pathlib import Path
import numpy as np
import pytest

from engine.depth.fallback import HeuristicDepthEngine
from engine.segmentation.depth_segmenter import DepthSegmenter
from engine.reconstruction.layer_extractor import LayerExtractor
from engine.pipeline import ConversionPipeline
from engine.format.package import Package4DW


def test_conversion_pipeline_end_to_end(tmp_path: Path):
    h, w = 180, 240
    # Create simple synthetic test scene
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky (top)
    img[:60, :] = [200, 100, 50]
    # Midground (hills)
    img[60:120, :] = [50, 120, 50]
    # Foreground
    img[120:, :] = [30, 30, 80]
    # Hero object in center
    img[80:150, 100:140] = [255, 255, 255]

    pipeline = ConversionPipeline(
        depth_engine=HeuristicDepthEngine(),
        segmenter=DepthSegmenter(min_area_ratio=0.01),
        layer_extractor=LayerExtractor(base_parallax=0.08),
        num_layers=3,
    )

    out_file = tmp_path / "output_scene.4dw"
    dbg_dir = tmp_path / "debug_dir"

    created = pipeline.run(
        input_image=img,
        output_package_path=out_file,
        debug_dir=dbg_dir,
    )

    assert created.exists()
    assert out_file.exists()

    # Inspect package
    info = Package4DW.inspect(out_file)
    assert info["format"] == "4DW"
    assert info["canvas"]["width"] == w
    assert info["canvas"]["height"] == h
    assert info["num_layers"] >= 2

    # Verify debug artifacts were written
    assert (dbg_dir / "depth_16bit.png").exists()
    assert (dbg_dir / "depth_preview.png").exists()
    assert (dbg_dir / "masks").exists()
