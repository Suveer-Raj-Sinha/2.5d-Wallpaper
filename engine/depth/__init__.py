"""
Depth estimation subsystem package.
Priority order:
  1. OnnxDepthEngine  — if models/depth_anything_v2_vits.onnx exists (best quality, no torch needed)
  2. DepthAnythingV2Engine — if torch + transformers are installed
  3. HeuristicDepthEngine — pure-Python fallback (always works)
"""

from pathlib import Path
from typing import Optional

from engine.depth.base import BaseDepthEngine, DepthResult
from engine.depth.depth_anything import DepthAnythingV2Engine
from engine.depth.fallback import HeuristicDepthEngine
from engine.depth.preprocessing import load_image_rgb, normalize_depth, smooth_depth

# Default search paths for ONNX models (relative to repo root, then absolute)
_ONNX_MODEL_CANDIDATES = [
    Path(__file__).parents[2] / "models" / "depth_anything_v2_vits.onnx",
    Path(__file__).parents[2] / "models" / "depth_anything_v2_vitb.onnx",
    Path(__file__).parents[2] / "models" / "depth_anything_v2_vitl.onnx",
]


def _find_onnx_model() -> Optional[Path]:
    """Return first existing ONNX model path, or None."""
    for p in _ONNX_MODEL_CANDIDATES:
        if p.exists():
            return p
    return None


def create_depth_engine(
    model_name: str = "auto",
    device: Optional[str] = None,
    fallback_on_error: bool = True,
    onnx_model_path: Optional[str] = None,
) -> BaseDepthEngine:
    """
    Factory function — instantiates the best available depth engine.

    Args:
        model_name: "auto" | "onnx" | "depth_anything_v2[_small|_base|_large]" | "heuristic"
        device:     "cuda" | "cpu" | None (auto-detect)
        fallback_on_error: Fall back to heuristic if preferred engine fails to load.
        onnx_model_path: Override path to .onnx file (overrides auto-detection).
    """
    m_lower = model_name.lower()

    # ── Explicit heuristic request ────────────────────────────────────────────
    if m_lower in ("heuristic", "fallback", "dummy"):
        return HeuristicDepthEngine()

    # ── Explicit ONNX request ─────────────────────────────────────────────────
    if m_lower in ("onnx", "onnx_depth"):
        from engine.depth.onnx_depth import OnnxDepthEngine
        path = Path(onnx_model_path) if onnx_model_path else _find_onnx_model()
        if path is None:
            raise FileNotFoundError(
                "No ONNX model found. Place depth_anything_v2_vits.onnx in the models/ directory.\n"
                "Download: https://github.com/fabio-sim/Depth-Anything-ONNX/releases/tag/v2.0.0"
            )
        return OnnxDepthEngine(model_path=path, use_cuda=(device != "cpu"))

    # ── Auto mode: try ONNX → transformers → heuristic ───────────────────────
    if m_lower == "auto":
        # 1. Try ONNX model (best quality, no torch needed)
        onnx_path = Path(onnx_model_path) if onnx_model_path else _find_onnx_model()
        if onnx_path is not None:
            try:
                from engine.depth.onnx_depth import OnnxDepthEngine
                engine = OnnxDepthEngine(model_path=onnx_path, use_cuda=(device != "cpu"))
                engine._load_session()  # validate immediately
                print(f"[DepthEngine] Using ONNX model: {onnx_path.name}")
                return engine
            except Exception as e:
                print(f"[DepthEngine] ONNX load failed ({e}), trying transformers …")

        # 2. Try transformers / PyTorch
        try:
            variant = "small"
            engine = DepthAnythingV2Engine(model_name=variant, device=device)
            print("[DepthEngine] Using Depth Anything V2 (transformers/torch)")
            if fallback_on_error:
                return _make_safe_engine(engine)
            return engine
        except Exception as e:
            print(f"[DepthEngine] Transformers load failed ({e}), using HeuristicDepthEngine.")

        # 3. Pure-Python fallback
        print("[DepthEngine] Using HeuristicDepthEngine (offline fallback)")
        return HeuristicDepthEngine()

    # ── Named transformers model ──────────────────────────────────────────────
    variant = "small"
    if "base" in m_lower:
        variant = "base"
    elif "large" in m_lower:
        variant = "large"

    engine = DepthAnythingV2Engine(model_name=variant, device=device)
    if fallback_on_error:
        return _make_safe_engine(engine)
    return engine


def _make_safe_engine(primary: BaseDepthEngine) -> BaseDepthEngine:
    """Wrap an engine with a HeuristicDepthEngine fallback."""

    class SafeDepthEngine(BaseDepthEngine):
        def __init__(self):
            self._primary = primary
            self._fallback = HeuristicDepthEngine()

        def estimate(self, image):
            try:
                return self._primary.estimate(image)
            except Exception as e:
                print(f"[DepthEngine] Primary engine failed ({e}), falling back to heuristic.")
                return self._fallback.estimate(image)

    return SafeDepthEngine()


__all__ = [
    "BaseDepthEngine",
    "DepthResult",
    "DepthAnythingV2Engine",
    "HeuristicDepthEngine",
    "create_depth_engine",
    "load_image_rgb",
    "normalize_depth",
    "smooth_depth",
]
