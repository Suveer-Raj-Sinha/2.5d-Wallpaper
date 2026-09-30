from pathlib import Path
from typing import Union, Optional, Callable, Dict, Any
from datetime import datetime, timezone
import numpy as np

from engine.depth.base import BaseDepthEngine, DepthResult
from engine.depth import create_depth_engine, load_image_rgb
from engine.depth.depth_refiner import refine_depth_map
from engine.layers.depth_bands import generate_adaptive_depth_bands
from engine.segmentation.foreground_ai import ForegroundDetector
from engine.segmentation.base import Segment
from engine.reconstruction.layer_extractor import LayerExtractor
from engine.reconstruction.hole_filling import InpaintMethod
from engine.scene.camera import Camera
from engine.scene.layer import Layer
from engine.scene.object import SceneObject
from engine.scene.scene import Canvas, Scene
from engine.format.package import Package4DW

class ConversionPipelineV2:
    """
    V2 Pipeline focused on depth-first approach (MVP 2).
    Image -> Depth -> Refinement -> Depth Bands -> Layers
    """
    def __init__(
        self,
        depth_engine: Optional[BaseDepthEngine] = None,
        layer_extractor: Optional[LayerExtractor] = None,
        num_layers: int = 5,
        base_parallax: float = 0.08,
        inpaint_method: InpaintMethod = InpaintMethod.LAMA,
        overscan_margin: float = 0.08,
    ):
        self.depth_engine = depth_engine or create_depth_engine()
        self.foreground_detector = ForegroundDetector()
        self.layer_extractor = layer_extractor or LayerExtractor(
            base_parallax=base_parallax,
            overscan_margin=overscan_margin,
            inpaint_method=inpaint_method,
        )
        self.num_layers = num_layers
        self.base_parallax = base_parallax

    def run(
        self,
        input_image: Union[str, Path, np.ndarray],
        output_package_path: Union[str, Path],
        debug_dir: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        foreground_approval_callback: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None,
    ) -> Path:
        def report(stage: str, ratio: float):
            if progress_callback:
                progress_callback(stage, ratio)
            print(f"[{int(ratio * 100):3d}%] {stage}")

        out_path = Path(output_package_path)
        dbg_path = Path(debug_dir) if debug_dir else None
        if dbg_path:
            dbg_path.mkdir(parents=True, exist_ok=True)

        # 1. Load Image
        report("Validating and loading image", 0.05)
        rgb_img, (w, h) = load_image_rgb(input_image)

        # 2. Depth Estimation
        report("Estimating depth map", 0.20)
        depth_result = self.depth_engine.estimate(rgb_img)

        # 3. Depth Refinement (Phase 3)
        report("Refining depth map", 0.35)
        refined_depth = refine_depth_map(depth_result.depth_map, rgb_img)
        depth_result.depth_map = refined_depth

        if dbg_path:
            depth_result.save_16bit(dbg_path / "depth_16bit_refined.png")
            depth_result.save_colormapped(dbg_path / "depth_preview_refined.png")

        # 4. Optional Phase 5: AI Object Detection for primary foreground
        report("Detecting primary foreground object", 0.45)
        fg_mask = self.foreground_detector.get_foreground_mask(rgb_img, depth_result.depth_map)
        
        if fg_mask is not None and np.any(fg_mask) and foreground_approval_callback:
            report("Waiting for user to approve AI Foreground...", 0.47)
            approved = foreground_approval_callback(rgb_img, fg_mask)
            if not approved:
                report("AI Foreground rejected by user, falling back to pure depth.", 0.48)
                fg_mask = None
        
        # 5. Generate Parallax Bands (Phase 4 & 6)
        report("Generating parallax bands", 0.50)
        depth_bands = generate_adaptive_depth_bands(depth_result.depth_map, num_bands=self.num_layers, foreground_mask=fg_mask)
        
        # Convert DepthBands to Segments for LayerExtractor compatibility
        segments = []
        for i, band in enumerate(depth_bands):
            # Compute bbox for compatibility
            y_idx, x_idx = np.nonzero(band.mask)
            if len(x_idx) > 0 and len(y_idx) > 0:
                x, y = int(np.min(x_idx)), int(np.min(y_idx))
                w_box, h_box = int(np.max(x_idx) - x), int(np.max(y_idx) - y)
                bbox = (x, y, w_box, h_box)
            else:
                bbox = (0, 0, w, h)
                
            segments.append(Segment(
                mask=(band.mask.astype(np.uint8) * 255),
                label=band.name,
                depth_mean=band.mean_depth,
                layer_index=i,
                bbox=bbox,
                parallax_factor=band.parallax_factor * self.base_parallax
            ))
        
        if dbg_path:
            masks_dir = dbg_path / "masks"
            masks_dir.mkdir(parents=True, exist_ok=True)
            for seg in segments:
                seg.save_mask(masks_dir / f"{seg.id}.png")

        # 5. Extract Layers & Inpaint (Phases 7, 8, 9)
        report("Extracting layers & inpainting", 0.65)
        extracted_layers = self.layer_extractor.extract(rgb_img, depth_result.depth_map, segments)

        # 6. Scene Construction
        report("Constructing scene graph", 0.80)
        canvas = Canvas(width=w, height=h)
        camera = Camera(
            position=[0.0, 0.0, 1.0],
            rotation=[0.0, 0.0, 0.0],
            fov=45.0,
            parallax_strength=self.base_parallax,
            limits=(-0.4, 0.4, -0.4, 0.4),
            focal_depth=0.5,
        )

        scene_layers = []
        layer_textures_dict = {}
        masks_dict = {}

        for elay in extracted_layers:
            tex_rel_path = f"layers/{elay.id}.png"
            mask_rel_path = f"masks/{elay.id}.png"
            layer_textures_dict[tex_rel_path] = elay.texture_rgba
            if elay.mask is not None:
                masks_dict[mask_rel_path] = elay.mask

            scene_layers.append(Layer(
                id=elay.id, name=elay.name, texture=tex_rel_path,
                depth=elay.depth, parallax=elay.parallax_factor,
                position=list(elay.offset), scale=list(elay.scale),
                rotation=0.0, opacity=1.0, visible=elay.visible, blend_mode="normal",
                mask=mask_rel_path if elay.mask is not None else None,
            ))

        scene_objects = []
        metadata = {
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "generator": "4DW Engine V2",
            "layerCount": len(scene_layers),
            "originalResolution": [w, h],
        }

        scene = Scene(
            canvas=canvas, camera=camera, layers=scene_layers, objects=scene_objects,
            color_asset="color.png", depth_asset="depth.png", preview_asset="preview.jpg", metadata=metadata,
        )

        # 7. Package
        report(f"Packaging scene into {out_path.name}", 0.95)
        created_file = Package4DW.save(
            scene=scene, output_path=out_path, color_image_rgb=rgb_img,
            depth_result=depth_result, layer_textures=layer_textures_dict, masks=masks_dict,
        )

        report("Conversion V2 completed successfully!", 1.0)
        return created_file
