 """
4DW manifest schema serialization and deserialization.
"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

from engine.scene.camera import Camera
from engine.scene.layer import Layer
from engine.scene.object import SceneObject
from engine.scene.effects import Effect
from engine.scene.scene import Canvas, Scene


class CanvasModel(BaseModel):
    width: int = 1920
    height: int = 1080


class CameraModel(BaseModel):
    position: List[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0])
    rotation: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    fov: float = 45.0
    parallaxStrength: float = 0.08
    limits: Tuple[float, float, float, float] = (-0.5, 0.5, -0.5, 0.5)
    focalDepth: float = 0.5


class AssetsModel(BaseModel):
    color: str = "color.png"
    depth: str = "depth.png"
    preview: str = "preview.jpg"


class LayerModel(BaseModel):
    id: str
    name: str
    texture: str
    depth: float
    parallax: float = 1.0
    position: List[float] = Field(default_factory=lambda: [0.0, 0.0])
    scale: List[float] = Field(default_factory=lambda: [1.0, 1.0])
    rotation: float = 0.0
    opacity: float = 1.0
    visible: bool = True
    blendMode: str = "normal"
    mask: Optional[str] = None
    animation: Dict[str, Any] = Field(default_factory=dict)


class SceneObjectModel(BaseModel):
    id: str
    name: str
    type: str = "object"
    depth: float = 0.5
    texture: Optional[str] = None
    mask: Optional[str] = None
    bbox: List[int] = Field(default_factory=lambda: [0, 0, 0, 0])
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EffectModel(BaseModel):
    type: str
    enabled: bool = True
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ManifestModel(BaseModel):
    format: str = "4DW"
    version: int = 1
    canvas: CanvasModel = Field(default_factory=CanvasModel)
    camera: CameraModel = Field(default_factory=CameraModel)
    assets: AssetsModel = Field(default_factory=AssetsModel)
    layers: List[LayerModel] = Field(default_factory=list)
    objects: List[SceneObjectModel] = Field(default_factory=list)
    effects: List[EffectModel] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


def scene_to_manifest(scene: Scene) -> ManifestModel:
    """Converts in-memory Scene graph to serializable Pydantic ManifestModel."""
    return ManifestModel(
        format="4DW",
        version=1,
        canvas=CanvasModel(width=scene.canvas.width, height=scene.canvas.height),
        camera=CameraModel(
            position=list(scene.camera.position),
            rotation=list(scene.camera.rotation),
            fov=scene.camera.fov,
            parallaxStrength=scene.camera.parallax_strength,
            limits=scene.camera.limits,
            focalDepth=scene.camera.focal_depth,
        ),
        assets=AssetsModel(
            color=scene.color_asset,
            depth=scene.depth_asset,
            preview=scene.preview_asset,
        ),
        layers=[
            LayerModel(
                id=l.id,
                name=l.name,
                texture=l.texture,
                depth=l.depth,
                parallax=l.parallax,
                position=list(l.position),
                scale=list(l.scale),
                rotation=l.rotation,
                opacity=l.opacity,
                visible=l.visible,
                blendMode=l.blend_mode,
                mask=l.mask,
                animation=dict(l.animation),
            )
            for l in scene.layers
        ],
        objects=[
            SceneObjectModel(
                id=o.id,
                name=o.name,
                type=o.type,
                depth=o.depth,
                texture=o.texture,
                mask=o.mask,
                bbox=list(o.bbox),
                metadata=dict(o.metadata),
            )
            for o in scene.objects
        ],
        effects=[
            EffectModel(type=e.type, enabled=e.enabled, parameters=dict(e.parameters))
            for e in scene.effects
        ],
        metadata=dict(scene.metadata),
    )


def manifest_to_scene(manifest: ManifestModel) -> Scene:
    """Reconstructs in-memory Scene graph from validated ManifestModel."""
    canvas = Canvas(width=manifest.canvas.width, height=manifest.canvas.height)
    camera = Camera(
        position=list(manifest.camera.position),
        rotation=list(manifest.camera.rotation),
        fov=manifest.camera.fov,
        parallax_strength=manifest.camera.parallaxStrength,
        limits=manifest.camera.limits,
        focal_depth=manifest.camera.focalDepth,
    )
    layers = [
        Layer(
            id=l.id,
            name=l.name,
            texture=l.texture,
            depth=l.depth,
            parallax=l.parallax,
            position=list(l.position),
            scale=list(l.scale),
            rotation=l.rotation,
            opacity=l.opacity,
            visible=l.visible,
            blend_mode=l.blendMode,
            mask=l.mask,
            animation=dict(l.animation),
        )
        for l in manifest.layers
    ]
    objects = [
        SceneObject(
            id=o.id,
            name=o.name,
            type=o.type,
            depth=o.depth,
            texture=o.texture,
            mask=o.mask,
            bbox=list(o.bbox),
            metadata=dict(o.metadata),
        )
        for o in manifest.objects
    ]
    effects = [
        Effect(type=e.type, enabled=e.enabled, parameters=dict(e.parameters))
        for e in manifest.effects
    ]
    return Scene(
        canvas=canvas,
        camera=camera,
        layers=layers,
        objects=objects,
        effects=effects,
        color_asset=manifest.assets.color,
        depth_asset=manifest.assets.depth,
        preview_asset=manifest.assets.preview,
        metadata=dict(manifest.metadata),
    )
