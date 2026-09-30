"""
Depth Anything V2 monocular depth estimation backend.
Supports Hugging Face Transformers with CUDA acceleration.
"""

from pathlib import Path
from typing import Union, Optional
import numpy as np
from PIL import Image
import cv2

from engine.depth.base import BaseDepthEngine, DepthResult
from engine.depth.preprocessing import load_image_rgb, normalize_depth, smooth_depth


class DepthAnythingV2Engine(BaseDepthEngine):
    """
    Depth Anything V2 depth estimation engine.
    Uses Hugging Face Transformers pipeline for robust weight loading and inference.
    """

    MODEL_ALIASES = {
        "small": "depth-anything/Depth-Anything-V2-Small-hf",
        "base": "depth-anything/Depth-Anything-V2-Base-hf",
        "large": "depth-anything/Depth-Anything-V2-Large-hf",
    }

    def __init__(
        self,
        model_name: str = "small",
        device: Optional[str] = None,
        use_fp16: bool = True,
        smoothing: bool = True,
    ):
        """
        Initialize Depth Anything V2 engine.

        Args:
            model_name: "small", "base", "large", or full HuggingFace model repo id
            device: "cuda", "cpu", or None for auto-detection
            use_fp16: Use FP16 precision on CUDA for faster inference & lower VRAM
            smoothing: Apply edge-preserving smoothing post-inference
        """
        self.model_id = self.MODEL_ALIASES.get(model_name.lower(), model_name)
        self.device = device
        self.use_fp16 = use_fp16
        self.smoothing = smoothing
        self._pipe = None

    def _load_model(self):
        """Lazy load model pipeline on first inference request."""
        if self._pipe is not None:
            return

        try:
            import torch
            from transformers import pipeline

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            torch_dtype = torch.float16 if (self.device == "cuda" and self.use_fp16) else torch.float32

            print(f"Loading Depth Anything V2 ({self.model_id}) on {self.device}...")
            self._pipe = pipeline(
                task="depth-estimation",
                model=self.model_id,
                device=self.device,
                torch_dtype=torch_dtype,
            )
            print("Depth Anything V2 loaded successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Depth Anything V2 model ({self.model_id}): {e}\n"
                "Ensure torch and transformers are installed, or use HeuristicDepthEngine for offline fallback."
            ) from e

    def estimate(self, image: Union[np.ndarray, Image.Image, str, Path]) -> DepthResult:
        self._load_model()

        rgb_arr, (w, h) = load_image_rgb(image)
        pil_img = Image.fromarray(rgb_arr)

        # Run inference
        result = self._pipe(pil_img)
        # result["depth"] is a PIL Image representing predicted depth
        depth_pil = result["depth"]

        # Convert to numpy float32
        raw_depth = np.array(depth_pil, dtype=np.float32)

        # Resize to original resolution if pipeline resized it
        if raw_depth.shape[:2] != (h, w):
            raw_depth = cv2.resize(raw_depth, (w, h), interpolation=cv2.INTER_CUBIC)

        # In Depth Anything V2, higher values correspond to nearer objects
        # Normalize to [0.0, 1.0] where 0.0 is far and 1.0 is near
        norm_depth = normalize_depth(raw_depth, invert=False)

        if self.smoothing:
            norm_depth = smooth_depth(norm_depth, rgb_arr)

        return DepthResult(depth_map=norm_depth, original_size=(w, h))
