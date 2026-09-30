# 4DW Engine: 2D to 2.5D/3D Depth Scene Conversion Engine

A modular Python-based conversion engine that transforms conventional 2D images into interactive, depth-aware 2.5D/3D scenes packaged as `.4dw` containers, rendered using modern GPU shaders.

```text
2D Image
   ↓
AI Monocular Depth Estimation (Depth Anything V2)
   ↓
Scene Understanding & Depth Segmentation
   ↓
Multi-Plane Layer Extraction & Occlusion Inpainting
   ↓
Scene Graph & Packaging (.4dw container)
   ↓
Hardware-Accelerated GPU Preview (ModernGL / OpenGL)
```

---

## Key Features

- **Offline Single-Pass Processing**: Heavy AI inference happens **once** during scene creation. The generated `.4dw` package is self-contained and renders at 60+ FPS on consumer hardware with zero runtime AI overhead.
- **Monocular Depth Subsystem**: Pluggable architecture supporting **Depth Anything V2** (Small, Base, Large) with CUDA acceleration, plus a rule-based heuristic fallback for headless CI and offline environments.
- **16-bit Depth Pipeline**: Depth maps are normalized to `[0.0, 1.0]` (`0.0 = furthest, 1.0 = nearest`) and stored as high-precision 16-bit PNGs (`[0, 65535]`).
- **Depth-Aware Scene Segmentation**: Slices scenes into depth strata and salient objects (Sky, Distant Background, Midground structures, Foreground characters).
- **Occlusion Handling & Hole-Filling**:
  - **Strategy 1 (Overscan)**: Canvas boundary expansion preventing edge clipping during parallax shifts.
  - **Strategy 2 (Edge Bleed & Feathering)**: Morphological boundary dilation and soft alpha feathering.
  - **Strategy 3 (Background Inpainting)**: Fast Telea / Navier-Stokes reconstruction behind occluded regions.
- **Portable `.4dw` Scene Container**: Open, ZIP-based container containing `manifest.json`, color image, 16-bit depth map, preview thumbnail, layer RGBA textures, and object masks.
- **Windows Desktop Application (`app.py`)**: A sleek, dark-themed GUI launcher to browse images, convert them to `.4dw`, configure parallax sensitivity, and run either windowed or directly as a Windows Live Wallpaper.
- **Global Desktop Mouse Tracking**: Uses Win32 `GetCursorPos` via `ctypes` so moving your cursor across any monitor, application, or game shifts the wallpaper parallax in real time.
- **Windows Live Wallpaper Mode (`WorkerW`)**: Reparents the OpenGL viewport directly behind Windows desktop icons using the native Win32 `WorkerW` mechanism (identical to Wallpaper Engine).
- **Interactive GPU Viewport**:
  - Multi-layer 2.5D parallax shader with smooth camera interpolation.
  - **4 View Modes**:
    1. `[1]` Interactive 2.5D Parallax Scene
    2. `[2]` Colormapped Depth Map (Google Turbo colormap)
    3. `[3]` 3D Exploded Layer Stack (visualizing depth planes in 3D perspective)
    4. `[4]` Original 2D Flat Image
  - **Virtual Camera & Gyroscope Simulation**: Mouse pan and scroll wheel zoom, plus a live smartphone gyroscope tilt sway simulator (`[Space]`).
  - **In-Viewport Tuning**: Adjust sensitivity (`[+] / [-]`) and parallax strength (`[[] / []]`) on the fly.

---

## Directory Structure

```text
4d wallpaper/
│
├── engine/
│   ├── depth/
│   │   ├── base.py              # BaseDepthEngine & DepthResult (16-bit encoder/decoder)
│   │   ├── depth_anything.py    # Depth Anything V2 implementation (PyTorch/Transformers)
│   │   ├── fallback.py          # Heuristic depth estimator for offline/CI
│   │   └── preprocessing.py     # Bilateral smoothing, normalization & conversions
│   │
│   ├── segmentation/
│   │   ├── base.py              # BaseSegmenter & Segment dataclass
│   │   ├── depth_segmenter.py   # Adaptive depth strata clustering & sky detector
│   │   └── masks.py             # Dilation, feathering & contour cleaning
│   │
│   ├── reconstruction/
│   │   ├── layer_extractor.py   # RGBA multi-plane extraction with depth-proportional parallax
│   │   ├── background.py        # Hole-free continuous background synthesizer
│   │   └── hole_filling.py      # Telea/NS inpainting and overscan extension
│   │
│   ├── scene/
│   │   ├── scene.py             # Scene & Canvas scene graph representations
│   │   ├── layer.py             # Layer definition (transform, parallax, blend mode)
│   │   ├── object.py            # Segmented salient object model
│   │   ├── camera.py            # Virtual camera parameters and bounds
│   │   └── effects.py           # Post-processing & environment effects model
│   │
│   ├── format/
│   │   ├── scene_format.py      # Manifest v1 schema & Pydantic serializer
│   │   └── package.py           # .4dw ZIP package writer, reader & inspector
│   │
│   ├── renderer/
│   │   ├── renderer.py          # ModernGL/Pygame desktop GPU viewport
│   │   ├── camera.py            # Inertial camera controller & gyro simulation
│   │   └── shaders/             # GLSL vertex & fragment shaders
│   │       ├── layer.vert / layer.frag
│   │       └── depth.vert / depth.frag
│   │
│   └── pipeline.py              # End-to-end ConversionPipeline
│
├── examples/
│   └── generate_sample.py       # Multi-depth synthetic scene generator
├── tests/                       # Complete Pytest test suite
├── cli.py                       # Unified command-line interface
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Installation

Create and activate a Python 3.11+ virtual environment:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Generate Sample Test Scene

```bash
python examples/generate_sample.py
```

This creates `examples/sample.png` containing a synthetic scene with sunset sky, mountains, buildings, and foreground hero.

### 3. Launch Windows Desktop Application (GUI)

```bash
python app.py
```

This launches the native dark-themed Windows desktop application:
- Select any 2D image (`.jpg`, `.png`, `.webp`)
- Adjust layer count, parallax intensity, and mouse sensitivity
- One-click convert to `.4dw`
- Set directly as **Windows Live Wallpaper** or open in interactive windowed preview!

---

## CLI Usage

### Convert 2D Image to `.4dw` Scene

```bash
# Using Depth Anything V2 (or automatic fallback if weights not present)
python cli.py convert input.jpg --output scene.4dw --layers 4

# With debug artifact outputs (raw depth, individual masks)
python cli.py convert input.jpg -o scene.4dw --debug-dir debug_output/
```

### Interactive GPU Desktop Preview

```bash
# Standard window with global desktop mouse tracking
python cli.py preview scene.4dw

# Set directly as Windows Live Wallpaper behind desktop icons!
python cli.py preview scene.4dw --wallpaper
```

**Viewport Controls**:
- **Desktop Cursor**: System-wide global mouse tracking drives parallax across Windows
- **Mouse Scroll**: Zoom camera in / out
- **[M]**: Toggle between Global Desktop and Window-Only mouse tracking
- **[+ / -]**: Increase / decrease mouse sensitivity
- **[[ / ]]**: Decrease / increase parallax displacement strength
- **[Space]**: Toggle simulated sway / gyroscope tilt
- **[1]**: 2.5D Parallax Interactive Scene
- **[2]**: Depth Map View (Turbo colormap)
- **[3]**: 3D Exploded Layer Stack
- **[4]**: Original 2D Flat Image
- **[R]**: Reset camera to center
- **[ESC]**: Exit preview

### Inspect `.4dw` Archive

```bash
python cli.py inspect scene.4dw
```

Output:
```text
========================================
 4DW Scene Archive: sample.4dw
========================================
Format:       4DW v1
Canvas:       1280 x 720
Camera FOV:   45.0 deg
Parallax Mul: 0.08
Total Layers: 4
Total Files:  12
----------------------------------------
LAYERS:
  * [layer_0_sky] Sky            depth=0.081  parallax=0.15 (visible)
  * [layer_1_background] Background     depth=0.320  parallax=0.46 (visible)
  * [layer_2_midground] Midground      depth=0.595  parallax=0.81 (visible)
  * [layer_3_foreground] Foreground     depth=0.747  parallax=1.00 (visible)
----------------------------------------
CONTAINER ASSETS:
  - manifest.json             (   0.9 KB)
  - color.png                 (  11.3 KB)
  - depth.png                 ( 550.8 KB)
  - preview.jpg               (  27.4 KB)
  - layers/layer_0_sky.png    (1115.1 KB)
  ...
========================================
```

### Extract Depth Map Only

```bash
python cli.py depth input.jpg -o depth.png --colormap
```

### Extract Visual Layers

```bash
python cli.py layers input.jpg -o output_layers/
```

---

## The `.4dw` File Specification

A `.4dw` file is a ZIP container with a structured manifest:

```text
scene.4dw
├── manifest.json
├── color.png
├── depth.png             # 16-bit grayscale (0 = far, 65535 = near)
├── preview.jpg           # Thumbnail
├── masks/
│   ├── layer_0_sky.png
│   └── layer_1_foreground.png
└── layers/
    ├── layer_0_sky.png   # RGBA texture
    └── layer_1_foreground.png
```

### `manifest.json` Example

```json
{
  "format": "4DW",
  "version": 1,
  "canvas": {
    "width": 1440,
    "height": 3200
  },
  "camera": {
    "position": [0.0, 0.0, 1.0],
    "rotation": [0.0, 0.0, 0.0],
    "fov": 45.0,
    "parallaxStrength": 0.08,
    "limits": [-0.4, 0.4, -0.4, 0.4],
    "focalDepth": 0.5
  },
  "assets": {
    "color": "color.png",
    "depth": "depth.png",
    "preview": "preview.jpg"
  },
  "layers": [
    {
      "id": "layer_0_sky",
      "name": "Sky",
      "texture": "layers/layer_0_sky.png",
      "depth": 0.05,
      "parallax": 0.15,
      "position": [0.0, 0.0],
      "scale": [1.16, 1.16],
      "rotation": 0.0,
      "opacity": 1.0,
      "visible": true,
      "blendMode": "normal"
    }
  ]
}
```

---

## Future Android Integration Roadmap

The engine is platform-independent. To consume `.4dw` scenes on Android:
1. **Unpack `.4dw`**: Unzip assets to internal storage.
2. **OpenGL ES / Vulkan Viewport**: Use the identical vertex and fragment shaders located in `engine/renderer/shaders/`.
3. **Sensor Integration**: Hook `Sensor.TYPE_ROTATION_VECTOR` or `Sensor.TYPE_GYROSCOPE` to supply camera `(x, y)` offsets exactly matching the `SmoothCameraController` logic.
