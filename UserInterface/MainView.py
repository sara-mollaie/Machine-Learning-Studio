import tkinter as tk
from tkinter import messagebox

from UserInterface.HomeModule import HomeModule
from UserInterface.Datasets import DatasetsModule
from UserInterface.Algorithms import AlgorithmsModule
from UserInterface.PredictorModule import PredictorModule
from UserInterface.EvaluateModule import EvaluateModule
from UserInterface.VisualizationModule import VisualizationModule
from UserInterface.Dialogs import InfoDialog, SettingsDialog
from UserInterface.Theme import theme, FONT


class ComingSoon(tk.Canvas):
    """صفحهٔ موقت برای ماژول‌هایی که هنوز ساخته نشده‌اند."""

    def __init__(self, master, name, size, on_back, back_text="← Home"):
        self.w, self.h = size
        super().__init__(
            master, width=self.w, height=self.h,
            highlightthickness=0, bd=0
        )
        self.page_name = name
        self._btn = tk.Button(
            self, text=back_text, command=on_back, bd=0,
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

    # صفحه‌هایی که فقط وقتی یک دیتاست باز است معنی دارند
    DATASET_PAGES = ("evaluate", "visualization")

    def __init__(self, master, scale):
        super().__init__(master, bd=0, highlightthickness=0)
        self.pages = {}
        self.current = None
        self.dataset = None          # دیتاستی که الان Predictor/Evaluate رویش کار می‌کند

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
            on_open=self.open_dataset,
            on_predict=self.open_predictor
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

        # Predictor Page
        predictor = PredictorModule(
            self,
            scale=scale,
            on_navigate=self.navigate
        )
        self.register("predictor", predictor)

        # Evaluate Page (هم‌دیتاست با Predictor)
        evaluate = EvaluateModule(
            self,
            scale=scale,
            on_navigate=self.navigate
        )
        self.register("evaluate", evaluate)

        # Visualization Page (نمودارهای همان دیتاست)
        visualization = VisualizationModule(
            self,
            scale=scale,
            on_navigate=self.navigate
        )
        self.register("visualization", visualization)

        self.navigate("home")

        # بعد از نمایش Home، مدل‌ها و تصویرهای صفحهٔ Predictor در پس‌زمینه آماده می‌شوند
        self.after(1200, self._warm_predictor)

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

        # Evaluate / Visualization بدون دیتاست بازشده معنی ندارند
        if name in self.DATASET_PAGES and self.dataset is None:
            return

        page = self.pages.get(name)
        if page is None:
            return

        # Evaluate همیشه با دیتاستی ساخته می‌شود که Predictor رویش باز بوده
        if name in self.DATASET_PAGES:
            page.load_dataset(self.dataset)

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

    def _warm_predictor(self):
        """بارگذاری پیش‌دستی: باز شدن صفحهٔ Predictor را فوری می‌کند."""
        predictor = self.pages.get("predictor")
        datasets = self.pages.get("datasets")
        if predictor is None:
            return
        clustering = [d for d in getattr(datasets, "all", []) if d.task == "Clustering"]
        predictor.preload(clustering)
        predictor.prewarm()

    def open_predictor(self, ds):
        """با زدن Open predictor در پنجرهٔ Info، صفحهٔ Predictor باز می‌شود."""
        if ds.task != "Clustering":
            messagebox.showinfo(
                "Predictor",
                f"The predictor for “{ds.name}” will be available soon.",
                parent=self.winfo_toplevel()
            )
            return

        datasets = self.pages.get("datasets")
        if datasets is not None and hasattr(datasets, "_close_info"):
            datasets._close_info()          # پنجرهٔ Info بسته شود

        self.dataset = ds                   # Evaluate و Visualization از همین استفاده می‌کنند

        predictor = self.pages["predictor"]
        predictor.load_dataset(ds)
        self.navigate("predictor")