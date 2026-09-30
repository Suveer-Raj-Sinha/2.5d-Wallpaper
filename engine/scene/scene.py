"""
Core scene representation and scene graph.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from engine.scene.camera import Camera
from engine.scene.layer import Layer
from engine.scene.object import SceneObject
from engine.scene.effects import Effect


@dataclass
class Canvas:
    width: int = 1920
    height: int = 1080

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(1, self.height)


@dataclass
class Scene:
    """
    Comprehensive representation of a depth-aware 2.5D/3D scene.
    """
    canvas: Canvas = field(default_factory=Canvas)
    camera: Camera = field(default_factory=Camera)
    layers: List[Layer] = field(default_factory=list)
    objects: List[SceneObject] = field(default_factory=list)
    effects: List[Effect] = field(default_factory=list)
    color_asset: str = "color.png"
    depth_asset: str = "depth.png"
    preview_asset: str = "preview.jpg"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_layer_by_id(self, layer_id: str) -> Optional[Layer]:
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        return None

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def add_object(self, obj: SceneObject):
        self.objects.append(obj)
