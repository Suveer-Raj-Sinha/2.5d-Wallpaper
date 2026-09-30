import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from engine.pipeline.conversion_pipeline import ConversionPipelineV2 as ConversionPipeline

def main():
    img_path = r"C:\Users\srs94\.gemini\antigravity\brain\ccbd07dc-1a09-4173-9073-9d5e53990ad3\.user_uploaded\media_1790535597806.jpg"
    out_path = "user_image.4dw"
    debug_dir = "debug_user_image"

    print(f"Starting conversion for: {img_path}")
    pipeline = ConversionPipeline()
    
    result = pipeline.run(
        input_image=img_path,
        output_package_path=out_path,
        debug_dir=debug_dir
    )
    
    print(f"Success! Output saved to: {result}")

if __name__ == "__main__":
    main()
