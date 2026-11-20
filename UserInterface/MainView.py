import tkinter as tk
from tkinter import messagebox

from UserInterface.HomeModule import HomeModule
from UserInterface.Datasets import DatasetsModule
from UserInterface.Algorithms import AlgorithmsModule
from UserInterface.Dialogs import InfoDialog, SettingsDialog
from UserInterface.Theme import theme, FONT


class ComingSoon(tk.Canvas):
    """صفحهٔ موقت برای ماژول‌هایی که هنوز ساخته نشده‌اند."""

    def __init__(self, master, name, size, on_back):
        self.w, self.h = size
        super().__init__(
            master, width=self.w, height=self.h,
            highlightthickness=0, bd=0
        )
        self.page_name = name
        self._btn = tk.Button(
            self, text="← Home", command=on_back, bd=0,
            padx=20, pady=8, bg="#a3305f", fg="white",
            font=(FONT, 12, "bold"), cursor="hand2"
        )
        theme.subscribe(self._draw)
        self._draw()

    def _draw(self):
        self.delete("all")
        self.configure(bg=theme.c("dlg_bg"))
        self.create_text(
            self.w // 2, self.h // 2 - 20,
            text=f"{self.page_name.title()} page",
            fill=theme.c("title"), font=(FONT, -40, "bold")
        )
        self.create_text(
            self.w // 2, self.h // 2 + 30,
            text="Coming soon...", fill=theme.c("body"),
            font=(FONT, -22)
        )
        self.create_window(
            self.w // 2, self.h // 2 + 100,
            window=self._btn
        )


class MainView(tk.Frame):
    """قاب اصلی: صفحه‌ها را ثبت می‌کند و بین آن‌ها جابه‌جا می‌شود."""

    def __init__(self, master, scale):
        super().__init__(master, bd=0, highlightthickness=0)
        self.pages = {}
        self.current = None

        # Home Page
        home = HomeModule(
            self, scale=scale, on_navigate=self.navigate
        )
        self.size = (home.W, home.H)
        self.register("home", home)

        # Datasets Page
        datasets = DatasetsModule(
            self,
            scale=scale,
            on_navigate=self.navigate,
            on_open=self.open_dataset
        )
        self.register("datasets", datasets)

        # Algorithms Page
        algorithms = AlgorithmsModule(
            self,
            scale=scale,
            on_navigate=self.navigate,
            on_explore=self.open_algorithm_datasets
        )
        self.register("algorithms", algorithms)

        self.navigate("home")

    def register(self, name, page):
        self.pages[name] = page

    def navigate(self, name):
        # Info و Settings پنجرهٔ کوچک باز می‌کنند، نه صفحهٔ کامل
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

        # به‌روزرسانی صفحه هنگام نمایش
        if hasattr(page, "on_show"):
            page.on_show()

    def open_algorithm_datasets(self, task):
        """با کلیک روی Browse، صفحه Datasets را با فیلتر همان الگوریتم باز می‌کند."""
        datasets = self.pages.get("datasets")
        if datasets is not None and hasattr(datasets, "set_task_filter"):
            datasets.set_task_filter(task)
        self.navigate("datasets")

    def open_dataset(self, ds):
        """با کلیک روی کارت دیتاست، خلاصهٔ اطلاعات آن را نمایش می‌دهد."""
        messagebox.showinfo(
            ds.name,
            f"{ds.description}\n\n"
            f"Level: {ds.level}\n"
            f"Domain: {ds.domain}\n"
            f"Task: {ds.task}\n"
            f"Rows: {ds.rows:,}   Columns: {ds.columns:,}",
            parent=self.winfo_toplevel()
        )
