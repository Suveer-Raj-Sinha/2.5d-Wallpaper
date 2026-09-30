"""
4DW package container reader and writer.
Encapsulates scene manifest, textures, 16-bit depth maps, and masks in a portable .4dw ZIP package.
"""

from pathlib import Path
from typing import Union, Dict, Any, List, Optional
import io
import json
import zipfile
import numpy as np
from PIL import Image
import cv2

from engine.scene.scene import Scene
from engine.depth.base import DepthResult
from engine.format.scene_format import ManifestModel, scene_to_manifest, manifest_to_scene


class LoadedScenePackage:
    """
    In-memory representation of an opened .4dw package file,
    providing direct streaming access to internal textures, depth maps, and metadata.
    """

    def __init__(self, zip_file: zipfile.ZipFile, manifest: ManifestModel, scene: Scene):
        self._zip = zip_file
        self.manifest = manifest
        self.scene = scene

    def close(self):
        self._zip.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def list_files(self) -> List[str]:
        return self._zip.namelist()

    def read_bytes(self, archive_path: str) -> bytes:
        norm_path = archive_path.replace("\\", "/")
        try:
            return self._zip.read(norm_path)
        except KeyError:
            raise FileNotFoundError(f"Asset '{archive_path}' not found in .4dw package.")

    def read_image_rgba(self, archive_path: str) -> np.ndarray:
        raw_bytes = self.read_bytes(archive_path)
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")
        return np.array(pil_img, dtype=np.uint8)

    def read_image_rgb(self, archive_path: str) -> np.ndarray:
        raw_bytes = self.read_bytes(archive_path)
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return np.array(pil_img, dtype=np.uint8)

    def read_depth_map(self) -> DepthResult:
        """Reads 16-bit depth map from package and returns normalized DepthResult."""
        depth_path = self.manifest.assets.depth
        raw_bytes = self.read_bytes(depth_path)
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to decode depth image from {depth_path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img.dtype == np.uint16:
            depth_map = img.astype(np.float32) / 65535.0
        else:
            depth_map = img.astype(np.float32) / 255.0

        h, w = depth_map.shape[:2]
        return DepthResult(depth_map=depth_map, original_size=(w, h))

    def read_preview(self) -> np.ndarray:
        return self.read_image_rgb(self.manifest.assets.preview)


class Package4DW:
    """
    Handles serialization, packaging, unpacking, and inspection of .4dw files.
    """

    @classmethod
    def save(
        cls,
        scene: Scene,
        output_path: Union[str, Path],
        color_image_rgb: np.ndarray,
        depth_result: DepthResult,
        preview_rgb: Optional[np.ndarray] = None,
        layer_textures: Optional[Dict[str, np.ndarray]] = None,
        masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> Path:
        """
        Package a complete scene into a .4dw container.

        Args:
            scene: Scene graph object
            output_path: Target .4dw filepath
            color_image_rgb: (H, W, 3) uint8 original image
            depth_result: DepthResult object
            preview_rgb: Optional preview thumbnail image
            layer_textures: Dict mapping layer archive path to RGBA uint8 ndarray
            masks: Dict mapping mask archive path to uint8/bool ndarray
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        manifest = scene_to_manifest(scene)
        manifest_json_str = manifest.model_dump_json(indent=2)

        # Prepare preview thumbnail if not provided
        if preview_rgb is None:
            # Resize color image to max 800px on long side
            h, w = color_image_rgb.shape[:2]
            scale = min(1.0, 800.0 / max(h, w))
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            preview_rgb = cv2.resize(color_image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Write ZIP container
        with zipfile.ZipFile(str(out_file), "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # 1. Manifest
            zf.writestr("manifest.json", manifest_json_str)

            # 2. Color image (PNG or JPG)
            success, color_encoded = cv2.imencode(".png", cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR))
            if success:
                zf.writestr(manifest.assets.color, color_encoded.tobytes())

            # 3. 16-bit Depth map (PNG)
            u16_depth = depth_result.uint16_depth
            success, depth_encoded = cv2.imencode(".png", u16_depth, [cv2.IMWRITE_PNG_COMPRESSION, 4])
            if success:
                zf.writestr(manifest.assets.depth, depth_encoded.tobytes())

            # 4. Preview image (JPG)
            success, prev_encoded = cv2.imencode(".jpg", cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
            if success:
                zf.writestr(manifest.assets.preview, prev_encoded.tobytes())

            # 5. Layer textures
            if layer_textures:
                for rel_path, tex in layer_textures.items():
                    norm_path = rel_path.replace("\\", "/")
                    # Convert RGBA to BGRA for OpenCV
                    if tex.shape[2] == 4:
                        bgra = cv2.cvtColor(tex, cv2.COLOR_RGBA2BGRA)
                    else:
                        bgra = cv2.cvtColor(tex, cv2.COLOR_RGB2BGR)
                    succ, enc = cv2.imencode(".png", bgra, [cv2.IMWRITE_PNG_COMPRESSION, 4])
                    if succ:
                        zf.writestr(norm_path, enc.tobytes())

            # 6. Segment masks
            if masks:
                for rel_path, m in masks.items():
                    norm_path = rel_path.replace("\\", "/")
                    u8_mask = (m > 0).astype(np.uint8) * 255 if m.dtype == bool else m.astype(np.uint8)
                    succ, enc = cv2.imencode(".png", u8_mask, [cv2.IMWRITE_PNG_COMPRESSION, 4])
                    if succ:
                        zf.writestr(norm_path, enc.tobytes())

        return out_file

    @classmethod
    def load(cls, package_path: Union[str, Path]) -> LoadedScenePackage:
        """Open a .4dw package file for reading."""
        path = Path(package_path)
        if not path.exists():
            raise FileNotFoundError(f"Package file not found: {path}")

        zf = zipfile.ZipFile(str(path), "r")
        try:
            manifest_bytes = zf.read("manifest.json")
            manifest_dict = json.loads(manifest_bytes.decode("utf-8"))
            manifest = ManifestModel.model_validate(manifest_dict)
            scene = manifest_to_scene(manifest)
            return LoadedScenePackage(zf, manifest, scene)
        except Exception as e:
            zf.close()
            raise ValueError(f"Invalid or corrupted .4dw package ({path}): {e}") from e

    @classmethod
    def extract(cls, package_path: Union[str, Path], output_dir: Union[str, Path]) -> Path:
        """Extract all contents of a .4dw package to a directory."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(package_path), "r") as zf:
            zf.extractall(str(out))
        return out

    @classmethod
    def inspect(cls, package_path: Union[str, Path]) -> Dict[str, Any]:
        """Inspect and return comprehensive metadata summary of a .4dw file."""
        with cls.load(package_path) as pkg:
            m = pkg.manifest
            files = pkg.list_files()
            file_stats = []
            for f in files:
                info = pkg._zip.getinfo(f)
                file_stats.append({
                    "name": f,
                    "uncompressed_bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                })

            summary = {
                "format": m.format,
                "version": m.version,
                "canvas": {"width": m.canvas.width, "height": m.canvas.height},
                "camera": {
                    "fov": m.camera.fov,
                    "parallaxStrength": m.camera.parallaxStrength,
                    "focalDepth": m.camera.focalDepth,
                },
                "num_layers": len(m.layers),
                "layers": [
                    {
                        "id": l.id,
                        "name": l.name,
                        "depth": round(l.depth, 4),
                        "parallax": round(l.parallax, 4),
                        "texture": l.texture,
                        "visible": l.visible,
                    }
                    for l in m.layers
                ],
                "num_objects": len(m.objects),
                "effects": [e.type for e in m.effects if e.enabled],
                "files_count": len(files),
                "files": file_stats,
                "metadata": m.metadata,
            }
            return summary
