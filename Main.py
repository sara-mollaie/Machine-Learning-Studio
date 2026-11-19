import ctypes
from PIL import Image

from UserInterface.Window import Window
from UserInterface.MainView import MainView
from UserInterface.HomeModule import BG_PATH
from UserInterface.Theme import theme


def apply_titlebar_theme(window):
    """نوار عنوان ویندوز را با تم هماهنگ می‌کند (تیره یا روشن)."""
    try:
        window.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        val = ctypes.c_int(1 if theme.name == "dark" else 0)
        for attr in (20, 19):                     # 20: ویندوز جدید، 19: نسخه‌های قدیمی‌تر ۱۰
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(val), ctypes.sizeof(val))
        ctypes.windll.user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, 0x37)    # بازرسم نوار عنوان
    except Exception:
        pass


def main():
    design_w, design_h = Image.open(BG_PATH).size
    window = Window("ML Studio", design_w, design_h)
    view = MainView(window, window.scale)
    view.pack()

    window.update()
    apply_titlebar_theme(window)
    theme.subscribe(lambda: apply_titlebar_theme(window))

    window.mainloop()


if __name__ == "__main__":
    main()