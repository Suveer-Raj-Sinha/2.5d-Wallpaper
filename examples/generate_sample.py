"""
Generate a synthetic test image with clearly stratified depth layers:
- Far Sky / Sunset Gradient
- Distant Mountains / Horizon
- Midground Cyberpunk Buildings & Trees
- Prominent Foreground Hero Character
"""

from pathlib import Path
import numpy as np
import cv2


def generate_sample_scene(width: int = 1280, height: int = 720) -> np.ndarray:
    """Generate a clean synthetic 2D scene with distinct depth cues."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # 1. Sky: Sunset gradient from deep violet to warm orange
    for y in range(height):
        ratio = y / height
        # Gradient: Top = (30, 10, 60), Horizon = (255, 120, 60)
        r = int(30 * (1 - ratio) + 255 * ratio)
        g = int(10 * (1 - ratio) + 120 * ratio)
        b = int(60 * (1 - ratio) + 60 * ratio)
        img[y, :] = [r, g, b]

    # 2. Distant Mountains (Dark purple)
    horizon_y = int(height * 0.55)
    mountain_pts = [
        (0, horizon_y),
        (int(width * 0.2), horizon_y - 120),
        (int(width * 0.35), horizon_y - 60),
        (int(width * 0.5), horizon_y - 150),
        (int(width * 0.7), horizon_y - 80),
        (int(width * 0.85), horizon_y - 130),
        (width, horizon_y),
        (width, height),
        (0, height),
    ]
    cv2.fillPoly(img, [np.array(mountain_pts, dtype=np.int32)], (60, 25, 75))

    # 3. Midground City Skyline (Dark indigo skyscrapers with glowing windows)
    building_y = int(height * 0.65)
    buildings = [
        (int(width * 0.05), int(width * 0.20), 220),
        (int(width * 0.22), int(width * 0.38), 300),
        (int(width * 0.40), int(width * 0.55), 180),
        (int(width * 0.58), int(width * 0.75), 260),
        (int(width * 0.78), int(width * 0.95), 200),
    ]
    for left, right, h_b in buildings:
        cv2.rectangle(img, (left, building_y - h_b), (right, height), (35, 30, 50), -1)
        # Windows
        for wy in range(building_y - h_b + 20, building_y - 20, 25):
            for wx in range(left + 15, right - 15, 20):
                if (wx + wy) % 3 == 0:
                    cv2.rectangle(img, (wx, wy), (wx + 8, wy + 12), (255, 220, 100), -1)

    # 4. Midground Ground / Road Platform
    cv2.rectangle(img, (0, building_y), (width, height), (20, 20, 30), -1)

    # 5. Foreground Subject (Silhouette of character standing on ledge looking out)
    char_cx = int(width * 0.5)
    char_base_y = int(height * 0.85)

    # Legs & Coat
    coat_pts = [
        (char_cx - 40, char_base_y),
        (char_cx - 20, char_base_y - 140),
        (char_cx + 20, char_base_y - 140),
        (char_cx + 40, char_base_y),
    ]
    cv2.fillPoly(img, [np.array(coat_pts, dtype=np.int32)], (15, 15, 20))

    # Torso
    cv2.rectangle(img, (char_cx - 25, char_base_y - 200), (char_cx + 25, char_base_y - 130), (15, 15, 20), -1)

    # Head
    cv2.circle(img, (char_cx, char_base_y - 225), 22, (15, 15, 20), -1)

    # Neon edge highlight on character (rim light)
    cv2.ellipse(img, (char_cx, char_base_y - 225), (22, 22), 0, 180, 360, (0, 220, 255), 2)
    cv2.line(img, (char_cx - 25, char_base_y - 200), (char_cx - 40, char_base_y), (0, 220, 255), 2)
    cv2.line(img, (char_cx + 25, char_base_y - 200), (char_cx + 40, char_base_y), (0, 220, 255), 2)

    # Foreground railing / edge
    cv2.rectangle(img, (0, int(height * 0.92)), (width, height), (10, 10, 15), -1)
    cv2.line(img, (0, int(height * 0.92)), (width, int(height * 0.92)), (0, 200, 255), 3)

    return img


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    sample_file = out_dir / "sample.png"
    sample_img = generate_sample_scene()
    bgr = cv2.cvtColor(sample_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(sample_file), bgr)
    print(f"Sample test image generated at: {sample_file}")
