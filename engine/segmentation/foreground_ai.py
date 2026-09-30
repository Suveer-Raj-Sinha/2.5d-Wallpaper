import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class ForegroundDetector:
    """
    Identifies the main foreground object(s) using YOLO/FastSAM.
    """
    def __init__(self, model_path: str = "models/FastSAM-s.pt"):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            # Fallback to YOLOv8n-seg if FastSAM is missing
            self.model_path = Path("yolov8n-seg.pt")
        self.model = YOLO(str(self.model_path))

    def get_foreground_mask(self, rgb_img: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        Returns a boolean mask of the primary foreground subject.
        """
        results = self.model(rgb_img, conf=0.25, iou=0.5, verbose=False)
        result = results[0]
        
        if result.masks is None or len(result.masks.data) == 0:
            return np.zeros(rgb_img.shape[:2], dtype=bool)

        h, w = rgb_img.shape[:2]
        masks = result.masks.data.cpu().numpy() # (N, H, W)
        
        # Resize masks to original image size
        combined_fg_mask = np.zeros((h, w), dtype=bool)
        
        for mask_data in masks:
            mask_resized = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
            
            # Score based on size and depth
            size_ratio = np.sum(mask_resized) / (h * w)
            if size_ratio < 0.005: 
                continue # ignore tiny floating artifacts
                
            depth_score = np.median(depth_map[mask_resized]) if np.any(mask_resized) else 0.0
            
            # If the object is physically near the camera, merge it into the foreground layer!
            if depth_score > 0.75:
                combined_fg_mask |= mask_resized

        # Cleanup the combined mask using morphological operations
        if np.any(combined_fg_mask):
            mask_u8 = combined_fg_mask.astype(np.uint8) * 255
            # Close small holes (e.g., between the guy and the railing)
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel_close)
            # Open to remove tiny speckles
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel_open)
            combined_fg_mask = mask_u8 > 127

        return combined_fg_mask
