import ctypes
import tkinter as tk


class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class Window(tk.Tk):

    TITLEBAR_AND_TASKBAR = 110
    Y_OFFSET = 3

    def __init__(self, title, design_w, design_h):
        self._enable_dpi_awareness()
        super().__init__()
        self.withdraw()
        self.title(title)

        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.scale = min(1.0, (sw - 20) / design_w,
                              (sh - self.TITLEBAR_AND_TASKBAR) / design_h)
        self.W = int(design_w * self.scale)
        self.H = int(design_h * self.scale)

        self.geometry(f"{self.W}x{self.H}+0+0")
        self.resizable(False, False)
        self.update_idletasks()

        border = self.winfo_rootx() - self.winfo_x()
        titlebar = self.winfo_rooty() - self.winfo_y()
        outer_w = self.W + 2 * border
        outer_h = self.H + titlebar + border

        left, top, right, bottom = self._work_area(sw, sh)
        x = left + (right - left - outer_w) // 2
        y = top + (bottom - top - outer_h) // 2 + self.Y_OFFSET
        y = max(top, min(y, bottom - outer_h))

        self.geometry(f"{self.W}x{self.H}+{x}+{y}")
        self.deiconify()

    @staticmethod
    def _work_area(sw, sh):
        try:
            r = _RECT()
            ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(r), 0)   # SPI_GETWORKAREA
            return r.left, r.top, r.right, r.bottom
        except Exception:
            return 0, 0, sw, sh

    @staticmethod
    def _enable_dpi_awareness():
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass