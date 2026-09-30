"""
ModernGL-based GPU Interactive Scene Renderer.
Supports multi-layer 2.5D parallax, depth visualization, 3D exploded view,
Windows global desktop mouse tracking, and Windows Live Wallpaper (WorkerW) mode.
"""

from pathlib import Path
from typing import Union, Optional, List, Dict
import time
import sys
import numpy as np
import moderngl
import pygame

from engine.format.package import Package4DW, LoadedScenePackage
from engine.renderer.camera import SmoothCameraController
from engine.platform.windows.mouse import WindowsMouseTracker
from engine.platform.windows.wallpaper import WindowsWallpaperManager


class RenderMode:
    PARALLAX = 1
    DEPTH = 2
    EXPLODE = 3
    ORIGINAL = 4


class SceneRenderer:
    """
    Hardware-accelerated OpenGL desktop renderer for .4dw depth scenes.
    """

    def __init__(
        self,
        package_or_path: Union[str, Path, LoadedScenePackage],
        window_width: int = 1280,
        window_height: int = 720,
        title: str = "4DW Scene Renderer",
        use_global_mouse: bool = True,
        wallpaper_mode: bool = False,
    ):
        if isinstance(package_or_path, (str, Path)):
            self.pkg = Package4DW.load(package_or_path)
            self._owns_pkg = True
        else:
            self.pkg = package_or_path
            self._owns_pkg = False

        self.scene = self.pkg.scene
        self.win_w = window_width
        self.win_h = window_height
        self.base_title = title
        self.use_global_mouse = use_global_mouse and (sys.platform == "win32")
        self.wallpaper_mode = wallpaper_mode and (sys.platform == "win32")

        # Windows platform integrations
        self.mouse_tracker = WindowsMouseTracker() if self.use_global_mouse else None
        self.wallpaper_manager = WindowsWallpaperManager() if self.wallpaper_mode else None

        self.camera_ctrl = SmoothCameraController(
            parallax_strength=self.scene.camera.parallax_strength,
            limits=self.scene.camera.limits,
            sensitivity=1.0,
            damping=15.0,
        )

        self.mode = RenderMode.PARALLAX
        self.running = False

    def _init_gl(self):
        """Initialize Pygame and ModernGL context."""
        pygame.init()
        pygame.display.set_caption(self.base_title)

        # Request OpenGL 3.3 Core
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )

        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if self.wallpaper_mode:
            # Fullscreen borderless for wallpaper background
            sw, sh = self.mouse_tracker.screen_size if self.mouse_tracker else (self.win_w, self.win_h)
            self.win_w, self.win_h = sw, sh
            flags |= pygame.NOFRAME

        self.screen = pygame.display.set_mode((self.win_w, self.win_h), flags | pygame.RESIZABLE)

        # Reparent to Windows WorkerW if in live wallpaper mode
        if self.wallpaper_mode and self.wallpaper_manager:
            wm_info = pygame.display.get_wm_info()
            hwnd = wm_info.get("window")
            if hwnd:
                attached = self.wallpaper_manager.attach_to_desktop(hwnd)
                if attached:
                    print("[4DW] Successfully attached to Windows WorkerW desktop background.")
                else:
                    print("[4DW] Falling back to standard desktop window.")

        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Quad mesh geometry: (pos_x, pos_y, uv_u, uv_v)
        vertices = np.array([
            -1.0, -1.0,  0.0,  1.0,
             1.0, -1.0,  1.0,  1.0,
            -1.0,  1.0,  0.0,  0.0,
             1.0,  1.0,  1.0,  0.0,
        ], dtype="f4")

        self.vbo = self.ctx.buffer(vertices.tobytes())

        # Load Shaders
        shaders_dir = Path(__file__).parent / "shaders"
        layer_vert = (shaders_dir / "layer.vert").read_text(encoding="utf-8")
        layer_frag = (shaders_dir / "layer.frag").read_text(encoding="utf-8")
        depth_vert = (shaders_dir / "depth.vert").read_text(encoding="utf-8")
        depth_frag = (shaders_dir / "depth.frag").read_text(encoding="utf-8")

        self.layer_prog = self.ctx.program(vertex_shader=layer_vert, fragment_shader=layer_frag)
        self.depth_prog = self.ctx.program(vertex_shader=depth_vert, fragment_shader=depth_frag)

        self.layer_vao = self.ctx.simple_vertex_array(
            self.layer_prog, self.vbo, "in_position", "in_uv"
        )
        self.depth_vao = self.ctx.simple_vertex_array(
            self.depth_prog, self.vbo, "in_position", "in_uv"
        )

        # Upload Textures
        self._load_textures()

    def _load_textures(self):
        """Upload image and depth textures to GPU."""
        # 1. Original 2D Color Image Texture
        color_rgb = self.pkg.read_image_rgb(self.scene.color_asset)
        h, w = color_rgb.shape[:2]
        self.tex_color = self.ctx.texture((w, h), 3, color_rgb.tobytes())
        self.tex_color.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # 2. 16-bit Depth Map Texture
        depth_res = self.pkg.read_depth_map()
        dh, dw = depth_res.depth_map.shape[:2]
        depth_f32 = depth_res.depth_map.astype("f4")
        self.tex_depth = self.ctx.texture((dw, dh), 1, depth_f32.tobytes(), dtype="f4")
        self.tex_depth.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # 3. Layer Textures
        self.layer_textures: List[Dict] = []
        for l in self.scene.layers:
            rgba = self.pkg.read_image_rgba(l.texture)
            lh, lw = rgba.shape[:2]
            tex = self.ctx.texture((lw, lh), 4, rgba.tobytes())
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.layer_textures.append({
                "layer": l,
                "texture": tex,
            })

    def run(self):
        """Main rendering and event loop."""
        self._init_gl()
        self.running = True
        clock = pygame.time.Clock()
        prev_time = time.time()

        while self.running:
            curr_time = time.time()
            dt = max(1e-4, curr_time - prev_time)
            prev_time = curr_time

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_1:
                        self.mode = RenderMode.PARALLAX
                        self.camera_ctrl.target_explode = 0.0
                    elif event.key == pygame.K_2:
                        self.mode = RenderMode.DEPTH
                        self.camera_ctrl.target_explode = 0.0
                    elif event.key == pygame.K_3:
                        self.mode = RenderMode.EXPLODE
                        self.camera_ctrl.target_explode = 1.0
                    elif event.key == pygame.K_4:
                        self.mode = RenderMode.ORIGINAL
                        self.camera_ctrl.target_explode = 0.0
                    elif event.key == pygame.K_r:
                        self.camera_ctrl.reset()
                    elif event.key == pygame.K_SPACE:
                        self.camera_ctrl.toggle_gyro()
                    elif event.key == pygame.K_m:
                        # Toggle global mouse tracking vs window tracking
                        self.use_global_mouse = not self.use_global_mouse
                        if self.use_global_mouse and self.mouse_tracker is None:
                            self.mouse_tracker = WindowsMouseTracker()
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        self.camera_ctrl.sensitivity = min(3.0, self.camera_ctrl.sensitivity + 0.15)
                    elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                        self.camera_ctrl.sensitivity = max(0.2, self.camera_ctrl.sensitivity - 0.15)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        self.camera_ctrl.parallax_strength = min(0.3, self.camera_ctrl.parallax_strength + 0.02)
                    elif event.key == pygame.K_LEFTBRACKET:
                        self.camera_ctrl.parallax_strength = max(0.01, self.camera_ctrl.parallax_strength - 0.02)
                elif event.type == pygame.MOUSEMOTION and not self.use_global_mouse:
                    # Window-relative mouse coordinates
                    mx, my = event.pos
                    norm_x = (mx / self.win_w) * 2.0 - 1.0
                    norm_y = (my / self.win_h) * 2.0 - 1.0
                    self.camera_ctrl.on_mouse_move(norm_x, norm_y)
                elif event.type == pygame.MOUSEWHEEL:
                    self.camera_ctrl.on_scroll(event.y)
                elif event.type == pygame.VIDEORESIZE and not self.wallpaper_mode:
                    self.win_w, self.win_h = event.w, event.h
                    self.screen = pygame.display.set_mode(
                        (self.win_w, self.win_h),
                        pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
                    )
                    self.ctx.viewport = (0, 0, self.win_w, self.win_h)

            # Global Windows mouse tracking (polls system cursor even across other windows)
            if self.use_global_mouse and self.mouse_tracker:
                gnx, gny = self.mouse_tracker.get_normalized_pos()
                self.camera_ctrl.on_mouse_move(gnx, gny)

            # Update camera
            self.camera_ctrl.update(dt)

            # Render frame
            self._render_frame()

            # Swap buffers & cap at 60 FPS
            pygame.display.flip()
            clock.tick(60)

            # Update window caption with stats
            if not self.wallpaper_mode:
                mode_names = {
                    RenderMode.PARALLAX: "[1] 2.5D Parallax",
                    RenderMode.DEPTH: "[2] Depth Map",
                    RenderMode.EXPLODE: "[3] 3D Explode",
                    RenderMode.ORIGINAL: "[4] 2D Flat",
                }
                tracking_str = "Global Desktop" if self.use_global_mouse else "Window"
                fps = int(clock.get_fps())
                pygame.display.set_caption(
                    f"{self.base_title} | FPS: {fps} | Mode: {mode_names[self.mode]} | Mouse: {tracking_str} | Sens: {self.camera_ctrl.sensitivity:.1f}x | [M] Mouse [+/-] Sens [Space] Sway"
                )

        self._cleanup()

    def _render_frame(self):
        self.ctx.clear(0.08, 0.09, 0.11, 1.0)
        cam_pos = self.camera_ctrl.position
        explode = self.camera_ctrl.current_explode

        if self.mode in (RenderMode.PARALLAX, RenderMode.EXPLODE):
            # Render layers back to front
            self.layer_prog["u_camera"].value = cam_pos
            self.layer_prog["u_parallax_strength"].value = self.camera_ctrl.parallax_strength
            self.layer_prog["u_explode"].value = explode
            self.layer_prog["u_debug_tint"].value = 1 if self.mode == RenderMode.EXPLODE else 0

            for item in self.layer_textures:
                layer = item["layer"]
                if not layer.visible:
                    continue

                tex = item["texture"]
                tex.use(location=0)
                self.layer_prog["u_texture"].value = 0
                self.layer_prog["u_layer_depth"].value = float(layer.depth)
                self.layer_prog["u_parallax_factor"].value = float(layer.parallax)
                self.layer_prog["u_scale"].value = tuple(layer.scale)
                self.layer_prog["u_offset"].value = tuple(layer.position)
                self.layer_prog["u_opacity"].value = float(layer.opacity)

                self.layer_vao.render(moderngl.TRIANGLE_STRIP)

        elif self.mode == RenderMode.DEPTH:
            self.depth_prog["u_camera"].value = cam_pos
            self.tex_depth.use(location=0)
            self.depth_prog["u_depth_texture"].value = 0
            self.depth_vao.render(moderngl.TRIANGLE_STRIP)

        elif self.mode == RenderMode.ORIGINAL:
            # Render single flat color image
            self.layer_prog["u_camera"].value = cam_pos
            self.layer_prog["u_parallax_strength"].value = 0.0
            self.layer_prog["u_explode"].value = 0.0
            self.layer_prog["u_debug_tint"].value = 0
            self.layer_prog["u_layer_depth"].value = 0.5
            self.layer_prog["u_parallax_factor"].value = 0.0
            self.layer_prog["u_scale"].value = (1.0, 1.0)
            self.layer_prog["u_offset"].value = (0.0, 0.0)
            self.layer_prog["u_opacity"].value = 1.0

            self.tex_color.use(location=0)
            self.layer_prog["u_texture"].value = 0
            self.layer_vao.render(moderngl.TRIANGLE_STRIP)

    def _cleanup(self):
        if self.wallpaper_mode and self.wallpaper_manager:
            self.wallpaper_manager.detach_from_desktop()
        pygame.quit()
        if self._owns_pkg:
            self.pkg.close()
