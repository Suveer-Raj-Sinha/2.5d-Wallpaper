import numpy as np
import cv2
from pathlib import Path
import urllib.request
import logging

logger = logging.getLogger(__name__)

class LamaInpainter:
    """
    LaMa (Large Mask Inpainting) using PyTorch TorchScript.
    Downloads the model automatically if missing.
    """
    def __init__(self, model_path: str = "models/big-lama.pt"):
        self.model_path = Path(model_path)
        self.model = None
        self.device = None

    def _load_session(self):
        if self.model is not None:
            return

        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.model_path.exists():
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
            logger.info(f"Downloading PyTorch LaMa model from {url}...")
            print(f"Downloading LaMa PyTorch model to {self.model_path} (this is ~195MB, may take a minute)...")
            urllib.request.urlretrieve(url, self.model_path)
            print("Download complete.")

        print("Loading LaMa TorchScript model...")
        self.model = torch.jit.load(self.model_path, map_location="cpu")
        self.model.eval()
        self.model.to(self.device)

    def inpaint(self, rgb_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint missing regions in rgb_image defined by mask.
        """
        if np.count_nonzero(mask) == 0:
            return rgb_image.copy()

        self._load_session()
        import torch

        # Prepare inputs. LaMa expects image to be multiple of 8.
        h, w = rgb_image.shape[:2]
        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8
        
        if pad_h > 0 or pad_w > 0:
            img_padded = cv2.copyMakeBorder(rgb_image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            mask_padded = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        else:
            img_padded = rgb_image.copy()
            mask_padded = mask.copy()

        img_tensor = img_padded.astype(np.float32) / 255.0
        img_tensor = np.transpose(img_tensor, (2, 0, 1))
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(self.device)

        mask_tensor = mask_padded.astype(np.float32)
        if mask_tensor.max() > 1.0:
            mask_tensor = mask_tensor / 255.0
        mask_tensor = mask_tensor > 0.0  # Ensure binary
        mask_tensor = mask_tensor.astype(np.float32)
        mask_tensor = torch.from_numpy(mask_tensor).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor, mask_tensor)
            
        res = output[0].cpu().numpy()  # (3, H, W)
        res = np.transpose(res, (1, 2, 0))
        res = np.clip(res * 255.0, 0, 255).astype(np.uint8)

        # Crop back to original size
        res = res[:h, :w]
        
        # Composite: use original pixels where mask is 0
        final = np.where(np.expand_dims(mask > 0, -1), res, rgb_image)
        
        return final

# Global instance for reuse
_lama_inpainter = None

def get_lama_inpainter() -> LamaInpainter:
    global _lama_inpainter
    if _lama_inpainter is None:
        _lama_inpainter = LamaInpainter()
    return _lama_inpainter
