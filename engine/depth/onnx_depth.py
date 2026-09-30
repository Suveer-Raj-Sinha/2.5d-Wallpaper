"""
ONNX-based Depth Anything V2 depth estimator.
Uses onnxruntime — no PyTorch / Transformers required.

Supported models (place in models/ directory):
  - depth_anything_v2_vits.onnx  (Small, ~94 MB, recommended)
  - depth_anything_v2_vitb.onnx  (Base, ~390 MB)
  - depth_anything_v2_vitl.onnx  (Large, ~1.3 GB)

Download from:
  https://github.com/fabio-sim/Depth-Anything-ONNX/releases/tag/v2.0.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

from engine.depth.base import BaseDepthEngine, DepthResult
from engine.depth.preprocessing import load_image_rgb, normalize_depth, smooth_depth

logger = logging.getLogger(__name__)

# ImageNet normalization constants (used by Depth Anything V2)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Default input size for the static ONNX model
_DEFAULT_INPUT_SIZE = 518


class OnnxDepthEngine(BaseDepthEngine):
    """
    Runs Depth Anything V2 via ONNX Runtime (CPU or CUDA EP).
    Produces metric-style relative depth maps, normalized to [0=far, 1=near].
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        input_size: int = _DEFAULT_INPUT_SIZE,
        use_cuda: bool = True,
        smoothing: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.use_cuda = use_cuda
        self.smoothing = smoothing
        self._session = None  # lazy-load on first inference

    def _load_session(self):
        """Lazy-load the ONNX session on first call."""
        if self._session is not None:
            return

        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime is not installed. Run: pip install onnxruntime"
            ) from e

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {self.model_path}.\n"
                f"Download from: https://github.com/fabio-sim/Depth-Anything-ONNX/releases/tag/v2.0.0"
            )

        providers = []
        if self.use_cuda:
            available = [p for p in ort.get_available_providers()]
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
                logger.info("OnnxDepthEngine: using CUDA EP")
            else:
                logger.info("OnnxDepthEngine: CUDA EP unavailable, using CPU")

        providers.append("CPUExecutionProvider")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.inter_op_num_threads = 4
        opts.intra_op_num_threads = 4

        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=providers,
        )

        # Cache input/output names
        self._input_name  = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        logger.info(
            "OnnxDepthEngine: loaded %s (input=%s, output=%s)",
            self.model_path.name,
            self._input_name,
            self._output_name,
        )

    # ------------------------------------------------------------------
    # Preprocessing / Postprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
        """
        Convert HxWx3 uint8 RGB → 1x3xSxS float32 tensor (ImageNet normalised).
        """
        s = self.input_size
        resized = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_LINEAR)
        img_f = resized.astype(np.float32) / 255.0
        img_norm = (img_f - _IMAGENET_MEAN) / _IMAGENET_STD
        # HWC → CHW → BCHW
        tensor = img_norm.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        return tensor

    def _postprocess(self, raw_output: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
        """
        Convert raw model output → float32 depth map normalised to [0=far, 1=near].
        Depth Anything V2 outputs larger values for closer pixels.
        """
        depth = raw_output.squeeze()  # → [H, W]

        # Resize back to original resolution
        if depth.shape != (orig_h, orig_w):
            depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1] preserving monotonicity (larger=closer stays larger)
        d_min, d_max = float(depth.min()), float(depth.max())
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

        return depth.astype(np.float32)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def estimate(self, image: Union[np.ndarray, Image.Image, str, Path]) -> DepthResult:
        self._load_session()

        rgb, (orig_w, orig_h) = load_image_rgb(image)

        tensor   = self._preprocess(rgb)
        output   = self._session.run([self._output_name], {self._input_name: tensor})[0]
        depth    = self._postprocess(output, orig_w, orig_h)

        if self.smoothing:
            depth = smooth_depth(depth, rgb)

        return DepthResult(depth_map=depth, original_size=(orig_w, orig_h))

    @classmethod
    def is_available(cls, model_path: Union[str, Path]) -> bool:
        """Return True if onnxruntime is installed AND model file exists."""
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False
        return Path(model_path).exists()
