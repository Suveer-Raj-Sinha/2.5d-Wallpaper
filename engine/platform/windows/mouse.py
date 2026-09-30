"""
Windows global mouse tracker using Win32 API.
Enables system-wide cursor tracking across all monitors and applications,
even when the 4DW window is unfocused or running as a live desktop wallpaper.
"""

import sys
from typing import Tuple, Optional
import ctypes
from ctypes import wintypes


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class WindowsMouseTracker:
    """
    Polls the Windows system cursor position via ctypes.windll.user32.GetCursorPos.
    Normalizes coordinates to [-1.0, 1.0] relative to screen center.
    """

    def __init__(self):
        self.is_windows = (sys.platform == "win32")
        self._user32 = ctypes.windll.user32 if self.is_windows else None
        self._cached_screen_w = 1920
        self._cached_screen_h = 1080
        self._update_screen_size()

    def _update_screen_size(self):
        """Fetch primary screen resolution from Windows metrics."""
        if self.is_windows and self._user32:
            # SM_CXSCREEN = 0, SM_CYSCREEN = 1
            w = self._user32.GetSystemMetrics(0)
            h = self._user32.GetSystemMetrics(1)
            if w > 0 and h > 0:
                self._cached_screen_w = w
                self._cached_screen_h = h

    def get_cursor_pos(self) -> Optional[Tuple[int, int]]:
        """Return raw screen coordinates (x, y) or None if unsupported."""
        if not self.is_windows or not self._user32:
            return None
        pt = POINT()
        if self._user32.GetCursorPos(ctypes.byref(pt)):
            return int(pt.x), int(pt.y)
        return None

    def get_normalized_pos(self) -> Tuple[float, float]:
        """
        Returns normalized coordinates (nx, ny) in range [-1.0, 1.0].
        (0.0, 0.0) is screen center.
        -1.0 is left/bottom, +1.0 is right/top.
        """
        raw = self.get_cursor_pos()
        if raw is None:
            return 0.0, 0.0

        raw_x, raw_y = raw
        w = max(1, self._cached_screen_w)
        h = max(1, self._cached_screen_h)

        # Normalize to [-1.0, 1.0]
        nx = (raw_x / (w * 0.5)) - 1.0
        # Invert Y so up is positive
        ny = -((raw_y / (h * 0.5)) - 1.0)

        # Clamp within bounds
        nx = max(-1.0, min(1.0, nx))
        ny = max(-1.0, min(1.0, ny))

        return nx, ny

    @property
    def screen_size(self) -> Tuple[int, int]:
        return self._cached_screen_w, self._cached_screen_h
