"""
Segmentation and scene understanding subsystem package.
"""

from engine.segmentation.base import BaseSegmenter, Segment
from engine.segmentation.foreground_ai import ForegroundDetector
from engine.segmentation.masks import (
    clean_mask,
    feather_mask,
    dilate_mask,
    compute_bbox,
)

__all__ = [
    "BaseSegmenter",
    "Segment",
    "ForegroundDetector",
    "clean_mask",
    "feather_mask",
    "dilate_mask",
    "compute_bbox",
]
