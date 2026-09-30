import numpy as np
import cv2
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class DepthBand:
    id: str
    name: str
    min_depth: float
    max_depth: float
    mean_depth: float
    mask: np.ndarray  # Boolean mask of the band
    parallax_factor: float = 0.0
    z_index: int = 0

def generate_adaptive_depth_bands(
    depth_map: np.ndarray, 
    num_bands: int = 5,
    foreground_mask: np.ndarray = None
) -> List[DepthBand]:
    """
    Analyzes the depth histogram to identify major depth regions and 
    slices the depth map into exactly `num_bands` discrete layers.
    If foreground_mask is provided, it extracts the foreground object 
    as its own layer to prevent it from being sliced in two.
    """
    bands = []
    layer_names = ["Sky", "Background", "Midground", "Foreground", "Close-up"]
    if num_bands != 5:
        layer_names = [f"Layer {i}" for i in range(num_bands)]
        
    valid_depths_mask = np.ones_like(depth_map, dtype=bool)
    
    # 1. Extract Foreground Object (Phase 6: Hybrid Layer Construction)
    fg_band = None
    if foreground_mask is not None and np.any(foreground_mask):
        valid_depths_mask = ~foreground_mask
        fg_depth_mean = float(np.median(depth_map[foreground_mask]))
        fg_band = DepthBand(
            id=f"band_{num_bands-1}",
            name="Primary Subject",
            min_depth=float(np.min(depth_map[foreground_mask])),
            max_depth=float(np.max(depth_map[foreground_mask])),
            mean_depth=fg_depth_mean,
            mask=foreground_mask,
            parallax_factor=fg_depth_mean,
            z_index=num_bands - 1
        )
        num_bands -= 1 # Reserve one band for the foreground
        
    # 2. Adaptive Depth Bands for the rest of the scene
    valid_depths = depth_map[valid_depths_mask]
    if len(valid_depths) > 0:
        # Subsample for speed
        subsample = np.random.choice(valid_depths, size=min(10000, len(valid_depths)), replace=False)
        
        # 1D K-Means to find depth clusters
        k = num_bands
        centroids = np.linspace(subsample.min(), subsample.max(), k)
        for _ in range(20):
            dists = np.abs(subsample[:, None] - centroids)
            labels = np.argmin(dists, axis=1)
            for j in range(k):
                if np.any(labels == j):
                    centroids[j] = np.mean(subsample[labels == j])
                    
        centroids = np.sort(centroids)
        # Thresholds are midpoints between centroids
        thresholds = [0.0]
        for j in range(k - 1):
            thresholds.append((centroids[j] + centroids[j+1]) / 2.0)
        thresholds.append(1.0)
    else:
        thresholds = np.linspace(0.0, 1.0, num_bands + 1)
        
    thresholds[0] = -0.01
    thresholds[-1] = 1.01

    for i in range(num_bands):
        lo = thresholds[i]
        hi = thresholds[i+1]
        
        # Boolean mask for this depth slice, excluding the foreground object
        mask = (depth_map > lo) & (depth_map <= hi) & valid_depths_mask
        
        if np.any(mask):
            mean_d = float(np.mean(depth_map[mask]))
        else:
            mean_d = float((lo + hi) / 2.0)
            
        bands.append(DepthBand(
            id=f"band_{i}",
            name=layer_names[i] if i < len(layer_names) else f"Layer {i}",
            min_depth=float(lo),
            max_depth=float(hi),
            mean_depth=mean_d,
            mask=mask,
            parallax_factor=mean_d,
            z_index=i
        ))
        
    if fg_band is not None:
        bands.append(fg_band)
        
    return bands
