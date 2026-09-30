"""
4DW Engine - Windows Desktop Application.
Interactive GUI to convert images into .4dw depth scenes, preview with global mouse tracking,
and set scenes as live Windows desktop wallpapers behind desktop icons.
"""

import sys
import os
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk

from engine.depth import create_depth_engine

from engine.reconstruction.layer_extractor import LayerExtractor
from engine.pipeline.conversion_pipeline import ConversionPipelineV2 as ConversionPipeline
from engine.format.package import Package4DW
from engine.renderer.renderer import SceneRenderer
from engine.platform.windows.mouse import WindowsMouseTracker


class WindowsAppGUI:
    """Windows GUI controller for 4DW Scene Conversion and Live Wallpaper."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("4DW Engine - Windows Parallax Wallpaper")
        self.root.geometry("860x720")
        self.root.minsize(780, 640)

        # State
        self.current_image_path: Path = None
        self.current_scene_path: Path = None
        self.active_renderer_thread: threading.Thread = None
        self.active_renderer: SceneRenderer = None
        self.mouse_tracker = WindowsMouseTracker()

        # Styles
        self._init_theme()

        # Build UI layout
        self._build_ui()

        # Check for pre-existing sample
        sample_path = Path(__file__).parent / "examples" / "sample.4dw"
        if sample_path.exists():
            self._load_scene_path(sample_path)

    def _init_theme(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Dark theme color palette
        self.bg_color = "#18191f"
        self.card_bg = "#23242b"
        self.card_border = "#2d2f38"
        self.accent_color = "#00d2ff"
        self.accent_hover = "#00a3cc"
        self.text_color = "#f0f2f5"
        self.text_dim = "#9aa0a6"

        self.root.configure(bg=self.bg_color)

        style.configure("TFrame", background=self.bg_color)
        style.configure("Card.TFrame", background=self.card_bg, relief="flat")
        style.configure("TLabel", background=self.bg_color, foreground=self.text_color, font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=self.card_bg, foreground=self.text_color, font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=self.bg_color, foreground=self.text_color, font=("Segoe UI", 16, "bold"))
        style.configure("Sub.TLabel", background=self.bg_color, foreground=self.text_dim, font=("Segoe UI", 10))
        style.configure("CardSub.TLabel", background=self.card_bg, foreground=self.text_dim, font=("Segoe UI", 9))
        style.configure("Section.TLabel", background=self.card_bg, foreground=self.accent_color, font=("Segoe UI", 11, "bold"))

        style.configure(
            "Accent.TButton",
            font=("Segoe UI", 10, "bold"),
            background=self.accent_color,
            foreground="#0b0e14",
            borderwidth=0,
            focuscolor="none",
        )
        style.map("Accent.TButton", background=[("active", self.accent_hover), ("disabled", "#3a3d45")])

        style.configure(
            "Secondary.TButton",
            font=("Segoe UI", 10),
            background="#333742",
            foreground=self.text_color,
            borderwidth=0,
            focuscolor="none",
        )
        style.map("Secondary.TButton", background=[("active", "#404654")])

    def _build_ui(self):
        main_container = ttk.Frame(self.root, padding=20)
        main_container.pack(fill="both", expand=True)

        # Header
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill="x", pady=(0, 15))

        title_label = ttk.Label(header_frame, text="4DW Engine", style="Title.TLabel")
        title_label.pack(side="left")

        sub_label = ttk.Label(
            header_frame,
            text="2D → 2.5D/3D Depth Scene Conversion & Windows Live Wallpaper",
            style="Sub.TLabel",
        )
        sub_label.pack(side="left", padx=(15, 0), pady=(4, 0))

        # Two-column layout: Left (File & Preview), Right (Settings & Actions)
        body_frame = ttk.Frame(main_container)
        body_frame.pack(fill="both", expand=True)

        # Left Column (Preview & File picker)
        left_card = ttk.Frame(body_frame, style="Card.TFrame", padding=15)
        left_card.pack(side="left", fill="both", expand=True, padx=(0, 10))

        ttk.Label(left_card, text="IMAGE / SCENE ASSET", style="Section.TLabel").pack(anchor="w")

        # Preview Canvas / Image Box
        self.preview_canvas = tk.Canvas(left_card, bg="#121316", highlightthickness=1, highlightbackground=self.card_border, height=260)
        self.preview_canvas.pack(fill="x", pady=10)
        self.preview_canvas.bind("<Configure>", lambda e: self._redraw_preview())

        # Buttons to select file
        file_btn_frame = ttk.Frame(left_card, style="Card.TFrame")
        file_btn_frame.pack(fill="x", pady=(5, 10))

        ttk.Button(file_btn_frame, text="Choose 2D Image (.jpg/.png)", style="Secondary.TButton", command=self._on_choose_image).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(file_btn_frame, text="Open .4dw Scene", style="Secondary.TButton", command=self._on_choose_scene).pack(side="right", fill="x", expand=True, padx=(5, 0))

        self.file_info_label = ttk.Label(left_card, text="No image or scene selected.", style="CardSub.TLabel")
        self.file_info_label.pack(anchor="w", pady=(0, 10))

        # Log & Status Console
        ttk.Label(left_card, text="PROCESSING CONSOLE", style="Section.TLabel").pack(anchor="w", pady=(10, 5))
        self.log_text = tk.Text(left_card, height=8, bg="#111215", fg="#d1d5db", font=("Consolas", 9), relief="flat", padx=8, pady=8)
        self.log_text.pack(fill="both", expand=True)
        self.log_text.insert("end", "[System] Ready.\n")
        self.log_text.configure(state="disabled")

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(left_card, variable=self.progress_var, maximum=100.0)
        self.progress_bar.pack(fill="x", pady=(8, 0))

        # Right Column (Parameters & Live Wallpaper Actions)
        right_card = ttk.Frame(body_frame, style="Card.TFrame", padding=15)
        right_card.pack(side="right", fill="both", expand=False, padx=(10, 0))
        right_card.configure(width=340)

        ttk.Label(right_card, text="CONVERSION SETTINGS", style="Section.TLabel").pack(anchor="w", pady=(0, 10))

        # Depth Engine
        ttk.Label(right_card, text="Depth Engine:", style="Card.TLabel").pack(anchor="w")
        self.depth_engine_var = tk.StringVar(value="depth_anything_v2")
        depth_combo = ttk.Combobox(
            right_card,
            textvariable=self.depth_engine_var,
            values=["depth_anything_v2", "heuristic (fast / offline)"],
            state="readonly",
        )
        depth_combo.pack(fill="x", pady=(2, 10))

        # Number of layers
        ttk.Label(right_card, text="Layers Count:", style="Card.TLabel").pack(anchor="w")
        self.layers_var = tk.IntVar(value=4)
        layer_scale = tk.Scale(
            right_card,
            from_=2,
            to=8,
            orient="horizontal",
            variable=self.layers_var,
            bg=self.card_bg,
            fg=self.text_color,
            highlightthickness=0,
            troughcolor="#111215",
        )
        layer_scale.pack(fill="x", pady=(0, 10))

        # Parallax Strength
        ttk.Label(right_card, text="Parallax Strength:", style="Card.TLabel").pack(anchor="w")
        self.parallax_var = tk.DoubleVar(value=0.08)
        parallax_scale = tk.Scale(
            right_card,
            from_=0.02,
            to=0.20,
            resolution=0.01,
            orient="horizontal",
            variable=self.parallax_var,
            bg=self.card_bg,
            fg=self.text_color,
            highlightthickness=0,
            troughcolor="#111215",
        )
        parallax_scale.pack(fill="x", pady=(0, 10))

        # Mouse Sensitivity
        ttk.Label(right_card, text="Windows Mouse Sensitivity:", style="Card.TLabel").pack(anchor="w")
        self.sens_var = tk.DoubleVar(value=1.0)
        sens_scale = tk.Scale(
            right_card,
            from_=0.2,
            to=2.5,
            resolution=0.1,
            orient="horizontal",
            variable=self.sens_var,
            bg=self.card_bg,
            fg=self.text_color,
            highlightthickness=0,
            troughcolor="#111215",
        )
        sens_scale.pack(fill="x", pady=(0, 10))

        # Global Mouse Tracking Checkbox
        self.global_mouse_var = tk.BooleanVar(value=True)
        global_chk = tk.Checkbutton(
            right_card,
            text="Global Desktop Mouse Tracking",
            variable=self.global_mouse_var,
            bg=self.card_bg,
            fg=self.text_color,
            selectcolor="#111215",
            activebackground=self.card_bg,
            activeforeground=self.accent_color,
        )
        global_chk.pack(anchor="w", pady=(0, 5))

        # AI Foreground Detection Checkbox
        self.use_ai_fg_var = tk.BooleanVar(value=False)
        ai_fg_chk = tk.Checkbutton(
            right_card,
            text="Use AI Foreground (FastSAM)",
            variable=self.use_ai_fg_var,
            bg=self.card_bg,
            fg=self.text_color,
            selectcolor="#111215",
            activebackground=self.card_bg,
            activeforeground=self.accent_color,
        )
        ai_fg_chk.pack(anchor="w", pady=(0, 15))

        # Action Buttons
        ttk.Label(right_card, text="ACTIONS", style="Section.TLabel").pack(anchor="w", pady=(5, 10))

        self.btn_convert = ttk.Button(
            right_card,
            text="⚡ Convert 2D Image to .4dw",
            style="Accent.TButton",
            command=self._on_convert,
        )
        self.btn_convert.pack(fill="x", pady=(0, 8))

        self.btn_preview = ttk.Button(
            right_card,
            text="▶ Launch Interactive Preview",
            style="Secondary.TButton",
            command=self._on_preview_window,
        )
        self.btn_preview.pack(fill="x", pady=(0, 8))

        self.btn_wallpaper = ttk.Button(
            right_card,
            text="🖥 Set as Windows Live Wallpaper",
            style="Secondary.TButton",
            command=self._on_wallpaper_mode,
        )
        self.btn_wallpaper.pack(fill="x", pady=(0, 8))

    def _log(self, message: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _ask_foreground_approval(self, rgb_img: np.ndarray, fg_mask: np.ndarray) -> bool:
        """
        Pauses pipeline thread and shows a Tkinter dialog to cross-check AI foreground mask.
        Returns True if approved, False to discard.
        """
        import threading
        import cv2
        from PIL import Image, ImageTk

        result = [True]
        event = threading.Event()

        def show_dialog():
            top = tk.Toplevel(self.root)
            top.title("Cross-Check AI Foreground")
            top.geometry("600x600")
            top.configure(bg=self.bg_color)
            top.transient(self.root)
            top.grab_set()

            ttk.Label(top, text="AI Foreground Detected", style="Title.TLabel").pack(pady=(15, 5))
            ttk.Label(top, text="Is this the correct main foreground subject?", style="Sub.TLabel").pack(pady=(0, 15))

            # Create red overlay
            overlay = rgb_img.copy()
            overlay[fg_mask > 0] = [255, 0, 0] # Red
            blended = cv2.addWeighted(rgb_img, 0.5, overlay, 0.5, 0)
            
            h, w = blended.shape[:2]
            ratio = min(500 / w, 400 / h)
            nw, nh = max(1, int(w * ratio)), max(1, int(h * ratio))
            blended_resized = cv2.resize(blended, (nw, nh))

            pil_img = Image.fromarray(blended_resized)
            tk_img = ImageTk.PhotoImage(pil_img)

            img_lbl = tk.Label(top, image=tk_img, bg="#121316")
            img_lbl.image = tk_img
            img_lbl.pack(pady=10)

            btn_frame = ttk.Frame(top, style="TFrame")
            btn_frame.pack(fill="x", pady=20, padx=20)

            def on_approve():
                result[0] = True
                event.set()
                top.destroy()

            def on_discard():
                result[0] = False
                event.set()
                top.destroy()

            ttk.Button(btn_frame, text="✅ Approve (Keep as solid layer)", style="Accent.TButton", command=on_approve).pack(side="left", expand=True, fill="x", padx=5)
            ttk.Button(btn_frame, text="❌ Discard (Fallback to pure depth bands)", style="Secondary.TButton", command=on_discard).pack(side="right", expand=True, fill="x", padx=5)

            top.protocol("WM_DELETE_WINDOW", on_discard)

        self.root.after(0, show_dialog)
        event.wait()
        return result[0]

    def _on_choose_image(self):
        filetypes = [("Image Files", "*.png;*.jpg;*.jpeg;*.webp"), ("All Files", "*.*")]
        path_str = filedialog.askopenfilename(title="Select 2D Image", filetypes=filetypes)
        if path_str:
            self.current_image_path = Path(path_str)
            self.current_scene_path = None
            self._load_image_preview(self.current_image_path)
            self.file_info_label.configure(
                text=f"Image: {self.current_image_path.name} ({self.current_image_path.stat().st_size // 1024} KB)"
            )
            self._log(f"[Loaded] Image: {self.current_image_path}")

    def _on_choose_scene(self):
        filetypes = [("4DW Scene Package", "*.4dw"), ("All Files", "*.*")]
        path_str = filedialog.askopenfilename(title="Select .4dw Scene", filetypes=filetypes)
        if path_str:
            self._load_scene_path(Path(path_str))

    def _load_scene_path(self, path: Path):
        self.current_scene_path = path
        try:
            info = Package4DW.inspect(path)
            with Package4DW.load(path) as pkg:
                prev_img = pkg.read_preview()
                self._display_preview_image(Image.fromarray(prev_img))
            self.file_info_label.configure(
                text=f"Scene: {path.name} ({info['num_layers']} layers, {path.stat().st_size // 1024} KB)"
            )
            self._log(f"[Loaded] .4dw Scene: {path.name} ({info['num_layers']} layers)")
        except Exception as e:
            self._log(f"[Error] Failed to load .4dw: {e}")

    def _load_image_preview(self, path: Path):
        try:
            pil_img = Image.open(path).convert("RGB")
            self._display_preview_image(pil_img)
        except Exception as e:
            self._log(f"[Error] Failed to display preview: {e}")

    def _display_preview_image(self, pil_img: Image.Image):
        self._raw_preview_img = pil_img
        self._redraw_preview()

    def _redraw_preview(self):
        if not hasattr(self, "_raw_preview_img") or not self._raw_preview_img:
            return
        cw = max(100, self.preview_canvas.winfo_width())
        ch = max(100, self.preview_canvas.winfo_height())
        iw, ih = self._raw_preview_img.size
        ratio = min(cw / iw, ch / ih)
        nw, nh = max(1, int(iw * ratio)), max(1, int(ih * ratio))

        resized = self._raw_preview_img.resize((nw, nh), Image.Resampling.LANCZOS)
        self._tk_preview = ImageTk.PhotoImage(resized)

        self.preview_canvas.delete("all")
        cx = cw // 2
        cy = ch // 2
        self.preview_canvas.create_image(cx, cy, image=self._tk_preview)

    def _on_convert(self):
        if not self.current_image_path or not self.current_image_path.exists():
            messagebox.showwarning("Notice", "Please select a 2D image first!")
            return

        engine_choice = "depth_anything_v2" if "anything" in self.depth_engine_var.get() else "heuristic"
        layers_count = self.layers_var.get()
        parallax = self.parallax_var.get()
        out_scene = self.current_image_path.with_suffix(".4dw")

        self.btn_convert.configure(state="disabled")
        self.progress_var.set(0.0)

        def worker():
            try:
                self._log(f"[Starting] Converting {self.current_image_path.name}...")
                depth_engine = create_depth_engine(model_name=engine_choice)

                extractor = LayerExtractor(base_parallax=parallax)

                def on_progress(stage, r):
                    self.root.after(0, lambda: self.progress_var.set(r * 100.0))
                    self.root.after(0, lambda: self._log(f"[{int(r*100)}%] {stage}"))

                pipeline = ConversionPipeline(
                    depth_engine=depth_engine,
                    layer_extractor=extractor,
                    num_layers=layers_count,
                    base_parallax=parallax,
                )
                pipeline.run(
                    input_image=self.current_image_path,
                    output_package_path=out_scene,
                    progress_callback=on_progress,
                    foreground_approval_callback=self._ask_foreground_approval,
                )

                self.root.after(0, lambda: self._on_convert_finished(out_scene))
            except Exception as e:
                self.root.after(0, lambda: self._log(f"[Error] Conversion failed: {e}"))
                self.root.after(0, lambda: self.btn_convert.configure(state="normal"))

        threading.Thread(target=worker, daemon=True).Thread = None
        threading.Thread(target=worker, daemon=True).start()

    def _on_convert_finished(self, out_scene: Path):
        self.btn_convert.configure(state="normal")
        self._log(f"[Complete] Scene saved: {out_scene.name}")
        self._load_scene_path(out_scene)
        messagebox.showinfo("Success", f"Converted successfully to {out_scene.name}!")

    def _on_preview_window(self):
        if not self.current_scene_path or not self.current_scene_path.exists():
            messagebox.showwarning("Notice", "Please open or convert a .4dw scene first!")
            return

        def render_thread():
            try:
                renderer = SceneRenderer(
                    package_or_path=self.current_scene_path,
                    window_width=1280,
                    window_height=720,
                    title=f"4DW Interactive View - {self.current_scene_path.name}",
                    use_global_mouse=self.global_mouse_var.get(),
                    wallpaper_mode=False,
                )
                renderer.camera_ctrl.sensitivity = self.sens_var.get()
                renderer.run()
            except Exception as e:
                self.root.after(0, lambda: self._log(f"[Renderer Error] {e}"))

        threading.Thread(target=render_thread, daemon=True).start()
        self._log("[Viewport] Launched interactive desktop preview.")

    def _on_wallpaper_mode(self):
        if not self.current_scene_path or not self.current_scene_path.exists():
            messagebox.showwarning("Notice", "Please open or convert a .4dw scene first!")
            return

        def wallpaper_thread():
            try:
                renderer = SceneRenderer(
                    package_or_path=self.current_scene_path,
                    title=f"4DW Live Wallpaper",
                    use_global_mouse=True,
                    wallpaper_mode=True,
                )
                renderer.camera_ctrl.sensitivity = self.sens_var.get()
                self._log("[Wallpaper] Running behind desktop icons (WorkerW). Press ESC in app or taskbar to close.")
                renderer.run()
            except Exception as e:
                self.root.after(0, lambda: self._log(f"[Wallpaper Error] {e}"))

        threading.Thread(target=wallpaper_thread, daemon=True).start()


def main():
    root = tk.Tk()
    app = WindowsAppGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
