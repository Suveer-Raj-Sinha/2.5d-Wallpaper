"""
Windows Live Wallpaper integration via Win32 WorkerW reparenting.
Embeds the OpenGL viewport directly behind desktop icons, exactly like Wallpaper Engine.
"""

import sys
from typing import Optional, Tuple
import ctypes
from ctypes import wintypes


class WindowsWallpaperManager:
    """
    Manages embedding a Win32 window behind Windows desktop icons into the WorkerW layer.
    """

    def __init__(self):
        self.is_windows = (sys.platform == "win32")
        self.user32 = ctypes.windll.user32 if self.is_windows else None
        self.original_parent: Optional[int] = None
        self.target_workerw: Optional[int] = None
        self.embedded_hwnd: Optional[int] = None

    def find_workerw(self) -> Optional[int]:
        """
        Send undocumented 0x052C message to Progman to spawn the WorkerW wallpaper window,
        then locate and return its HWND.
        """
        if not self.is_windows or not self.user32:
            return None

        # 1. Find Progman
        progman = self.user32.FindWindowW("Progman", None)
        if not progman:
            return None

        # 2. Send 0x052C to Progman to spawn WorkerW
        result = wintypes.DWORD()
        self.user32.SendMessageTimeoutW(
            progman,
            0x052C,
            0x0000000D,
            0,
            0x0002,  # SMTO_NORMAL
            1000,
            ctypes.byref(result),
        )

        # 3. Enumerate top-level windows to locate WorkerW
        workerw_hwnd = None

        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def enum_windows_callback(hwnd, lparam):
            nonlocal workerw_hwnd
            # Look for window containing SHELLDLL_DefView
            shell_view = self.user32.FindWindowExW(hwnd, 0, "SHELLDLL_DefView", None)
            if shell_view != 0:
                # The target WorkerW window is the next sibling
                next_worker = self.user32.FindWindowExW(0, hwnd, "WorkerW", None)
                if next_worker != 0:
                    workerw_hwnd = next_worker
            return True

        cb = WNDENUMPROC(enum_windows_callback)
        self.user32.EnumWindows(cb, 0)

        # If not found via sibling, try direct WorkerW
        if not workerw_hwnd:
            workerw_hwnd = self.user32.FindWindowExW(0, 0, "WorkerW", None)

        self.target_workerw = workerw_hwnd
        return workerw_hwnd

    def attach_to_desktop(self, hwnd: int) -> bool:
        """
        Reparent the given window handle behind desktop icons.
        """
        if not self.is_windows or not self.user32:
            return False

        workerw = self.find_workerw()
        if not workerw:
            print("Warning: Could not find Windows WorkerW desktop handle.")
            return False

        self.embedded_hwnd = hwnd
        self.original_parent = self.user32.GetParent(hwnd)

        # Reparent window
        self.user32.SetParent(hwnd, workerw)

        # Adjust window size to cover the desktop
        screen_w = self.user32.GetSystemMetrics(0)
        screen_h = self.user32.GetSystemMetrics(1)
        SWP_SHOWWINDOW = 0x0040
        self.user32.SetWindowPos(hwnd, 0, 0, 0, screen_w, screen_h, SWP_SHOWWINDOW)
        return True

    def detach_from_desktop(self):
        """Restore window to normal state before closing."""
        if self.embedded_hwnd and self.user32:
            self.user32.SetParent(self.embedded_hwnd, self.original_parent or 0)
            self.embedded_hwnd = None
