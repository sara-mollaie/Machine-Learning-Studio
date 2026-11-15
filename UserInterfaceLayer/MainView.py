import tkinter as tk

from UserInterfaceLayer.HomeModule import HomeModule
from UserInterfaceLayer.Dialogs import InfoDialog, SettingsDialog
from UserInterfaceLayer.Theme import theme, FONT


class ComingSoon(tk.Canvas):
    """صفحهٔ موقت برای ماژول‌هایی که هنوز ساخته نشده‌اند (با تم هماهنگ می‌شود)."""

    def __init__(self, master, name, size, on_back):
        self.w, self.h = size
        super().__init__(master, width=self.w, height=self.h, highlightthickness=0, bd=0)
        self.page_name = name
        self._btn = tk.Button(self, text="← Home", command=on_back, bd=0, padx=20, pady=8,
                              bg="#a3305f", fg="white", font=(FONT, 12, "bold"), cursor="hand2")
        theme.subscribe(self._draw)
        self._draw()

    def _draw(self):
        self.delete("all")
        self.configure(bg=theme.c("dlg_bg"))
        self.create_text(self.w // 2, self.h // 2 - 20, text=f"{self.page_name.title()} page",
                         fill=theme.c("title"), font=(FONT, -40, "bold"))
        self.create_text(self.w // 2, self.h // 2 + 30, text="Coming soon...",
                         fill=theme.c("body"), font=(FONT, -22))
        self.create_window(self.w // 2, self.h // 2 + 100, window=self._btn)


class MainView(tk.Frame):
    """قاب اصلی: صفحه‌ها را ثبت می‌کند و بین آن‌ها جابه‌جا می‌شود."""

    def __init__(self, master, scale):
        super().__init__(master, bd=0, highlightthickness=0)
        self.pages = {}
        self.current = None

        home = HomeModule(self, scale=scale, on_navigate=self.navigate)
        self.size = (home.W, home.H)
        self.register("home", home)

        for name in ("datasets", "algorithms"):
            self.register(name, ComingSoon(self, name, self.size, lambda: self.navigate("home")))

        self.navigate("home")

    def register(self, name, page):
        self.pages[name] = page

    def navigate(self, name):
        # info و settings پنجرهٔ کوچک باز می‌کنند، نه صفحهٔ کامل
        if name == "info":
            InfoDialog(self.winfo_toplevel())
            return
        if name == "settings":
            SettingsDialog(self.winfo_toplevel())
            return

        page = self.pages.get(name)
        if page is None:
            return
        if self.current is not None:
            self.current.pack_forget()
        page.pack()
        self.current = page