"""
Windows platform integration package.
"""

from engine.platform.windows.mouse import WindowsMouseTracker
from engine.platform.windows.wallpaper import WindowsWallpaperManager

__all__ = [
    "WindowsMouseTracker",
    "WindowsWallpaperManager",
]
