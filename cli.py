"""
Command-Line Interface for the 4DW 2D-to-2.5D/3D Depth Scene Conversion Engine.
"""

import argparse
import sys
from pathlib import Path
import json

from engine.depth import create_depth_engine, load_image_rgb
from engine.segmentation import create_segmenter
from engine.reconstruction.layer_extractor import LayerExtractor
from engine.pipeline import ConversionPipeline
from engine.format.package import Package4DW
from engine.renderer.renderer import SceneRenderer


def cmd_convert(args):
    """Convert a 2D image into an interactive .4dw depth scene package."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input image not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_suffix(".4dw")

    print(f"=== 4DW Scene Conversion ===")
    print(f"Input:        {input_path}")
    print(f"Output:       {output_path}")
    print(f"Depth Engine: {args.depth_engine}")
    print(f"Layers:       {args.layers}")
    print(f"Parallax:     {args.parallax}")
    print("---------------------------------")

    depth_engine = create_depth_engine(model_name=args.depth_engine, device=args.device)
    segmenter = create_segmenter(method="depth")
    layer_extractor = LayerExtractor(base_parallax=args.parallax)

    pipeline = ConversionPipeline(
        depth_engine=depth_engine,
        segmenter=segmenter,
        layer_extractor=layer_extractor,
        num_layers=args.layers,
        base_parallax=args.parallax,
    )

    created = pipeline.run(
        input_image=input_path,
        output_package_path=output_path,
        debug_dir=args.debug_dir,
    )

    print("---------------------------------")
    print(f"Success! Generated: {created}")
    print(f"Preview with: python cli.py preview {created}")


def cmd_preview(args):
    """Launch GPU interactive preview for a .4dw scene package."""
    scene_path = Path(args.scene)
    if not scene_path.exists():
        print(f"Error: Scene file not found: {scene_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Launching interactive GPU preview for: {scene_path}")
    print("Controls:")
    print("  Desktop Cursor : Global mouse tracking drives parallax across Windows")
    print("  Mouse Wheel    : Zoom in / out")
    print("  [M]            : Toggle Global Desktop vs Window Mouse tracking")
    print("  [+ / -]        : Adjust sensitivity")
    print("  [[ / ]]        : Adjust parallax strength")
    print("  [Space]        : Toggle phone gyroscope sway simulation")
    print("  [1]            : 2.5D Parallax Interactive Scene")
    print("  [2]            : Depth Map (Colormapped)")
    print("  [3]            : 3D Exploded Layer Stack")
    print("  [4]            : Original 2D Flat Image")
    print("  [R]            : Reset Camera")
    print("  [ESC]          : Exit")

    use_global = not getattr(args, "window_mouse", False)

    renderer = SceneRenderer(
        package_or_path=scene_path,
        window_width=args.width,
        window_height=args.height,
        title=f"4DW Preview - {scene_path.name}",
        use_global_mouse=use_global,
        wallpaper_mode=getattr(args, "wallpaper", False),
    )
    if hasattr(args, "sensitivity") and args.sensitivity:
        renderer.camera_ctrl.sensitivity = args.sensitivity
    renderer.run()


def cmd_depth(args):
    """Estimate and export depth map from a 2D image."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input image not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_depth.png")

    print(f"Estimating depth for {input_path} using {args.depth_engine}...")
    depth_engine = create_depth_engine(model_name=args.depth_engine, device=args.device)
    result = depth_engine.estimate(input_path)

    saved_path = result.save_16bit(output_path)
    print(f"Saved 16-bit depth map: {saved_path}")

    if args.colormap:
        color_path = output_path.with_name(f"{output_path.stem}_color.png")
        result.save_colormapped(color_path)
        print(f"Saved colormapped preview: {color_path}")


def cmd_layers(args):
    """Extract individual depth layers and save them to a directory."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input image not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else Path(f"{input_path.stem}_layers")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {args.layers} layers for {input_path}...")
    rgb_img, (w, h) = load_image_rgb(input_path)

    depth_engine = create_depth_engine(model_name=args.depth_engine, device=args.device)
    depth_res = depth_engine.estimate(rgb_img)

    segmenter = create_segmenter(method="depth")
    segments = segmenter.segment(rgb_img, depth_res.depth_map, num_layers=args.layers)

    layer_extractor = LayerExtractor()
    layers = layer_extractor.extract(rgb_img, depth_res.depth_map, segments)

    import cv2
    for l in layers:
        layer_file = out_dir / f"{l.id}.png"
        bgra = cv2.cvtColor(l.texture_rgba, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(layer_file), bgra)
        print(f"  Exported layer '{l.name}' (depth={l.depth:.3f}, parallax={l.parallax_factor:.2f}) -> {layer_file}")

    print(f"Done! All layers saved to {out_dir}")


def cmd_inspect(args):
    """Inspect and display metadata of a .4dw scene package."""
    scene_path = Path(args.scene)
    if not scene_path.exists():
        print(f"Error: Scene file not found: {scene_path}", file=sys.stderr)
        sys.exit(1)

    info = Package4DW.inspect(scene_path)
    print("========================================")
    print(f" 4DW Scene Archive: {scene_path.name}")
    print("========================================")
    print(f"Format:       {info['format']} v{info['version']}")
    print(f"Canvas:       {info['canvas']['width']} x {info['canvas']['height']}")
    print(f"Camera FOV:   {info['camera']['fov']} deg")
    print(f"Parallax Mul: {info['camera']['parallaxStrength']}")
    print(f"Total Layers: {info['num_layers']}")
    print(f"Total Files:  {info['files_count']}")
    print("----------------------------------------")
    print("LAYERS:")
    for l in info["layers"]:
        vis = "visible" if l["visible"] else "hidden"
        print(f"  * [{l['id']}] {l['name']:<14} depth={l['depth']:.3f}  parallax={l['parallax']:.2f} ({vis})")
    print("----------------------------------------")
    print("CONTAINER ASSETS:")
    for f in info["files"]:
        size_kb = f["compressed_bytes"] / 1024.0
        print(f"  - {f['name']:<25} ({size_kb:6.1f} KB)")
    print("========================================")


def main():
    parser = argparse.ArgumentParser(
        prog="4dw",
        description="2D to 2.5D/3D Depth Scene Conversion Engine",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. convert
    p_conv = subparsers.add_parser("convert", help="Convert 2D image into .4dw scene package")
    p_conv.add_argument("input", help="Path to input 2D image (JPG, PNG)")
    p_conv.add_argument("-o", "--output", help="Output .4dw file path")
    p_conv.add_argument("--depth-engine", default="depth_anything_v2", help="Depth engine: depth_anything_v2 or heuristic")
    p_conv.add_argument("--device", default=None, help="Device: cuda or cpu")
    p_conv.add_argument("--layers", type=int, default=4, help="Number of depth layers to extract (default: 4)")
    p_conv.add_argument("--parallax", type=float, default=0.08, help="Base parallax strength factor (default: 0.08)")
    p_conv.add_argument("--debug-dir", help="Directory to save debug intermediate artifacts")
    p_conv.set_defaults(func=cmd_convert)

    # 2. preview
    p_prev = subparsers.add_parser("preview", help="Interactive GPU desktop preview for .4dw scene")
    p_prev.add_argument("scene", help="Path to .4dw scene file")
    p_prev.add_argument("--width", type=int, default=1280, help="Window width (default: 1280)")
    p_prev.add_argument("--height", type=int, default=720, help="Window height (default: 720)")
    p_prev.add_argument("--wallpaper", action="store_true", help="Run as Windows Live Wallpaper behind desktop icons (WorkerW)")
    p_prev.add_argument("--window-mouse", action="store_true", help="Disable global Windows mouse tracking and use window-only tracking")
    p_prev.add_argument("--sensitivity", type=float, default=1.0, help="Mouse sensitivity multiplier (default: 1.0)")
    p_prev.set_defaults(func=cmd_preview)

    # 3. depth
    p_depth = subparsers.add_parser("depth", help="Generate 16-bit depth map from image")
    p_depth.add_argument("input", help="Input image file")
    p_depth.add_argument("-o", "--output", help="Output 16-bit PNG depth map path")
    p_depth.add_argument("--depth-engine", default="depth_anything_v2", help="Depth engine: depth_anything_v2 or heuristic")
    p_depth.add_argument("--device", default=None, help="Device: cuda or cpu")
    p_depth.add_argument("--colormap", action="store_true", help="Also generate colormapped RGB preview")
    p_depth.set_defaults(func=cmd_depth)

    # 4. layers
    p_layers = subparsers.add_parser("layers", help="Extract and export visual layers from image")
    p_layers.add_argument("input", help="Input image file")
    p_layers.add_argument("-o", "--output-dir", help="Directory to save extracted layers")
    p_layers.add_argument("--depth-engine", default="depth_anything_v2", help="Depth engine to use")
    p_layers.add_argument("--device", default=None, help="Device: cuda or cpu")
    p_layers.add_argument("--layers", type=int, default=4, help="Number of layers to extract")
    p_layers.set_defaults(func=cmd_layers)

    # 5. inspect
    p_insp = subparsers.add_parser("inspect", help="Inspect .4dw scene archive metadata")
    p_insp.add_argument("scene", help="Path to .4dw scene file")
    p_insp.set_defaults(func=cmd_inspect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
