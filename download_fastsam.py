"""
Setup FastSAM ONNX model for the 4D Wallpaper engine.

Steps:
  1. Installs ultralytics (needed only for the export, not runtime)
  2. Downloads FastSAM-s.pt weights from GitHub releases
  3. Exports to ONNX at models/FastSAM-s.onnx
  4. Uninstalls ultralytics (optional cleanup)

Run once:
  .venv\Scripts\python.exe download_fastsam.py
"""
import subprocess
import sys
import urllib.request
from pathlib import Path

DEST_ONNX = Path("models/FastSAM-s.onnx")
DEST_PT   = Path("models/FastSAM-s.pt")

# Official GitHub release weights
PT_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-s.pt"
)


def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name} ...")

    def _progress(block, block_size, total):
        downloaded = block * block_size
        pct = min(downloaded / total * 100, 100) if total > 0 else 0
        mb  = downloaded / 1_048_576
        print(f"\r  {pct:5.1f}%  ({mb:.1f} MB)    ", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print(f"\n  Saved: {dest}  ({dest.stat().st_size / 1_048_576:.1f} MB)")


def main():
    if DEST_ONNX.exists():
        print(f"[OK] {DEST_ONNX} already exists — nothing to do.")
        return

    # 1. Install ultralytics temporarily
    print("Installing ultralytics for ONNX export ...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "ultralytics", "--quiet"]
    )

    # 2. Download .pt weights
    if not DEST_PT.exists():
        download_file(PT_URL, DEST_PT)
    else:
        print(f"[OK] {DEST_PT} already cached.")

    # 3. Export to ONNX
    print(f"Exporting {DEST_PT.name} -> {DEST_ONNX.name} ...")
    from ultralytics import FastSAM
    model = FastSAM(str(DEST_PT))
    exported = model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
    exported_path = Path(exported)

    # ultralytics saves next to .pt; move to models/
    if exported_path != DEST_ONNX:
        import shutil
        shutil.move(str(exported_path), str(DEST_ONNX))

    print(f"\n[Done] {DEST_ONNX}  ({DEST_ONNX.stat().st_size / 1_048_576:.1f} MB)")
    print("You can now convert images with AI segmentation!")


if __name__ == "__main__":
    main()
