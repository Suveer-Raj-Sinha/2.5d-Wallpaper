"""
Unit tests for Windows platform mouse tracking and wallpaper manager.
"""

import sys
import pytest
from engine.platform.windows.mouse import WindowsMouseTracker
from engine.platform.windows.wallpaper import WindowsWallpaperManager


def test_windows_mouse_tracker_initialization():
    tracker = WindowsMouseTracker()
    assert tracker.screen_size[0] > 0
    assert tracker.screen_size[1] > 0

    if sys.platform == "win32":
        # Check normalized pos is within [-1.0, 1.0]
        nx, ny = tracker.get_normalized_pos()
        assert -1.0 <= nx <= 1.0
        assert -1.0 <= ny <= 1.0


def test_windows_wallpaper_manager_initialization():
    manager = WindowsWallpaperManager()
    assert manager.is_windows == (sys.platform == "win32")
