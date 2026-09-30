"""
Debug script: runs the full segmentation pipeline on an image and saves
all masks as PNGs so we can see exactly what the segmenter is producing.

Usage:
  .venv\Scripts\python.exe debug_seg.py "path/to/image.jpg"
"""
import sys
import logging
import numpy as np
import cv2
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

if len(sys.argv) < 2:
    print("Usage: python debug_seg.py <image_path>")
    sys.exit(1)

img_path = Path(sys.argv[1])
out_dir  = Path("debug_masks")
out_dir.mkdir(exist_ok=True)

# ── Load image ────────────────────────────────────────────────────────────────
from engine.depth import create_depth_engine, load_image_rgb
rgb, (w, h) = load_image_rgb(img_path)
print(f"Image: {img_path.name}  {w}x{h}")

# ── Depth estimation ──────────────────────────────────────────────────────────
print("Running depth estimation...")
depth_engine = create_depth_engine()
depth_result = depth_engine.estimate(rgb)
depth = depth_result.depth_map

# Save depth preview
depth_u8 = (depth * 255).astype(np.uint8)
depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
cv2.imwrite(str(out_dir / "depth.png"), depth_color)
print(f"  Saved: {out_dir}/depth.png")

# ── Segmentation ──────────────────────────────────────────────────────────────
from engine.segmentation import create_segmenter, FastSAMSegmenter

segmenter = create_segmenter(method="auto")
print(f"\nSegmenter: {type(segmenter).__name__}")

segments = segmenter.segment(depth_map=depth, original_image=rgb)
print(f"\nTotal segments: {len(segments)}")

# ── Save each mask overlaid on the original image ────────────────────────────
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

colors = [
    (0, 255, 255), (0, 128, 255), (0, 255, 0), (255, 0, 128),
    (255, 0, 0),   (128, 0, 255), (0, 200, 100), (200, 200, 0),
]

composite = bgr.copy()

for i, seg in enumerate(segments):
    mask = (seg.mask > 0.5) if seg.mask.dtype == np.float32 else (seg.mask > 127)
    color = colors[i % len(colors)]

    # Per-mask file
    overlay = bgr.copy()
    overlay[mask] = (
        overlay[mask] * 0.4 + np.array(color, dtype=np.float32) * 0.6
    ).astype(np.uint8)
    fname = out_dir / f"seg_{i:02d}_{seg.label}_depth{seg.depth_mean:.2f}.png"
    cv2.imwrite(str(fname), overlay)

    # Add to composite
    composite[mask] = (
        composite[mask] * 0.5 + np.array(color, dtype=np.float32) * 0.5
    ).astype(np.uint8)

    px_count = int(mask.sum())
    print(f"  [{i:02d}] label={seg.label:<12}  depth={seg.depth_mean:.3f}"
          f"  parallax={seg.parallax_factor:+.3f}  pixels={px_count:,}")

cv2.imwrite(str(out_dir / "composite.png"), composite)
print(f"\nSaved all masks to: {out_dir}/")
print(f"Composite overview: {out_dir}/composite.png")
