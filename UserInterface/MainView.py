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

    def __init__(self, master, name, size, on_back,
                 back_text='← Home'):

        self.w, self.h = size

        super().__init__(
            master,
            width=self.w,
            height=self.h,
            highlightthickness=0,
            bd=0
        )

        self.page_name = name

        self._btn = tk.Button(
            self,
            text=back_text,
            command=on_back,
            bd=0,
            padx=20,
            pady=8,
            bg='#a3305f',
            fg='white',
            font=(FONT, 12, 'bold'),
            cursor='hand2'
        )

        theme.subscribe(self._draw)
        self._draw()

    def _draw(self):

        self.delete('all')
        self.configure(bg=theme.c('dlg_bg'))

        self.create_text(
            self.w // 2,
            self.h // 2 - 20,
            text=f'{self.page_name.title()} page',
            fill=theme.c('title'),
            font=(FONT, -40, 'bold')
        )

        self.create_text(
            self.w // 2,
            self.h // 2 + 30,
            text='Coming soon...',
            fill=theme.c('body'),
            font=(FONT, -22)
        )

        self.create_window(
            self.w // 2,
            self.h // 2 + 100,
            window=self._btn
        )


class MainView(tk.Frame):

    DATASET_PAGES = (
        'evaluate',
        'visualization'
    )

    PREDICTOR_PAGES = {
        'clustering': 'predictor',
        'regression': 'regression_predictor'
    }

    def __init__(self, master, scale):

        super().__init__(
            master,
            bd=0,
            highlightthickness=0
        )

        self.pages = {}
        self.current = None
        self.dataset = None
        self.scale = scale

        # ==========================================
        # Home
        # ==========================================

        home = HomeModule(
            self,
            scale=scale,
            on_navigate=self.navigate
        )

        self.size = (home.W, home.H)

        self.register('home', home)

        # ==========================================
        # Datasets
        # ==========================================

        self.register(
            'datasets',
            DatasetsModule(
                self,
                scale=scale,
                on_navigate=self.navigate,
                on_open=self.open_dataset,
                on_predict=self.open_predictor
            )
        )

        # ==========================================
        # Algorithms
        # ==========================================

        self.register(
            'algorithms',
            AlgorithmsModule(
                self,
                scale=scale,
                on_navigate=self.navigate,
                on_explore=self.open_algorithm_datasets
            )
        )

        # ==========================================
        # Clustering Pages
        # ==========================================

        for name, cls in [
            ('predictor', PredictorModule),
            ('evaluate', EvaluateModule),
            ('visualization', VisualizationModule)
        ]:

            self.register(
                name,
                cls(
                    self,
                    scale=scale,
                    on_navigate=self.navigate
                )
            )

        self.navigate('home')

        self.after(1200, self._warm_predictor)

    # ==============================================
    # Register Page
    # ==============================================

    def register(self, name, page):

        self.pages[name] = page

    # ==============================================
    # Dataset Task
    # ==============================================

    def _get_dataset_task(self):

        if self.dataset is None:
            return None

        return str(
            self.dataset.task
        ).strip().lower()

    # ==============================================
    # Regression Predictor
    # ==============================================

    def _get_predictor(self, name):

        if (
            name == 'regression_predictor'
            and name not in self.pages
        ):

            from UserInterface.RegressionPredictor import (
                RegressionPredictor
            )

            self.register(
                name,
                RegressionPredictor(
                    self,
                    scale=self.scale,
                    on_navigate=self.navigate
                )
            )

        return self.pages[name]

    # ==============================================
    # Regression Evaluation
    # ==============================================

    def _get_regression_evaluate(self):

        name = 'regression_evaluate'

        if name not in self.pages:

            from UserInterface.RegressionEvaluate import (
                RegressionEvaluate
            )

            self.register(
                name,
                RegressionEvaluate(
                    self,
                    scale=self.scale,
                    on_navigate=self.navigate
                )
            )

        return self.pages[name]

    # ==============================================
    # Navigation
    # ==============================================

    def _get_regression_visualization(self):
        name = 'regression_visualization'
        if name not in self.pages:
            from UserInterface.RegressionVisualization import RegressionVisualization
            self.register(name, RegressionVisualization(self, scale=self.scale,
                                                        on_navigate=self.navigate))
        return self.pages[name]

    def navigate(self, name):

        # Information Dialog

        if name == 'info':

            InfoDialog(self.winfo_toplevel())
            return

        # Settings Dialog

        if name == 'settings':

            SettingsDialog(self.winfo_toplevel())
            return

        task = self._get_dataset_task()

        load = (
            name in self.DATASET_PAGES
            or name in (
                'regression_evaluate',
                'regression_visualization'
            )
        )

        # ==========================================
        # Evaluation / Visualization
        # ==========================================

        if load:

            if self.dataset is None:
                return

            if task not in self.PREDICTOR_PAGES:

                messagebox.showinfo(
                    'Machine Learning Studio',
                    'This page will be available soon.',
                    parent=self.winfo_toplevel()
                )

                return

            if task == 'regression':

                if name in self.DATASET_PAGES:
                    name = 'regression_' + name

                if name == 'regression_evaluate':

                    try:

                        self._get_regression_evaluate()

                    except Exception as exc:

                        messagebox.showerror(
                            'Evaluation Error',
                            f'{self.dataset.name}: {exc}',
                            parent=self.winfo_toplevel()
                        )

                        return

                if name == 'regression_visualization':
                    try:
                        self._get_regression_visualization()
                    except Exception as exc:
                        messagebox.showerror('Visualization Error', str(exc),
                                             parent=self.winfo_toplevel())
                        return

            elif name.startswith('regression_'):

                name = name.removeprefix('regression_')

            if name not in self.pages:

                messagebox.showinfo(
                    'Machine Learning Studio',
                    'This page will be available soon.',
                    parent=self.winfo_toplevel()
                )

                return

        # ==========================================
        # Predictor Routing
        # ==========================================

        if name == 'predictor' and task:

            name = self.PREDICTOR_PAGES.get(
                task,
                name
            )

        if name in self.PREDICTOR_PAGES.values():

            try:

                self._get_predictor(name)

            except Exception as exc:

                messagebox.showerror(
                    'Predictor Error',
                    str(exc),
                    parent=self.winfo_toplevel()
                )

                return

        page = self.pages.get(name)

        if page is None:
            return

        # ==========================================
        # Load Selected Dataset
        # ==========================================

        if load:

            try:

                page.load_dataset(self.dataset)

            except Exception as exc:

                messagebox.showerror(
                    'Dataset Error',
                    f'{self.dataset.name}: {exc}',
                    parent=self.winfo_toplevel()
                )

                return

        # ==========================================
        # Switch Page
        # ==========================================

        previous = self.current

        try:

            if previous is not None:
                previous.pack_forget()

            page.pack()

            self.current = page

            if hasattr(page, 'on_show'):
                page.on_show()

        except Exception:

            page.pack_forget()

            self.current = previous

            if previous is not None:
                previous.pack()

            raise

    # ==============================================
    # Open Algorithm Datasets
    # ==============================================

    def open_algorithm_datasets(self, task):

        datasets = self.pages.get('datasets')

        if (
            datasets is not None
            and hasattr(datasets, 'set_task_filter')
        ):

            datasets.set_task_filter(task)

        self.navigate('datasets')

    # ==============================================
    # Dataset Information
    # ==============================================

    def open_dataset(self, ds):

        messagebox.showinfo(
            ds.name,
            f'{ds.description}\n\n'
            f'Level: {ds.level}\n'
            f'Domain: {ds.domain}\n'
            f'Task: {ds.task}\n'
            f'Rows: {ds.rows:,}   '
            f'Columns: {ds.columns:,}',
            parent=self.winfo_toplevel()
        )

    # ==============================================
    # Clustering Predictor Preload
    # ==============================================

    def _warm_predictor(self):

        predictor = self.pages.get('predictor')
        datasets = self.pages.get('datasets')

        if predictor is None or datasets is None:
            return

        clustering = [

            d for d in getattr(datasets, 'all', [])

            if str(d.task).strip().lower()
            == 'clustering'

        ]

        predictor.preload(clustering)
        predictor.prewarm()

    # ==============================================
    # Open Predictor
    # ==============================================

    def open_predictor(self, ds):

        task = str(ds.task).strip().lower()

        name = self.PREDICTOR_PAGES.get(task)

        if name is None:

            messagebox.showinfo(
                'Predictor',
                f'The predictor for "{ds.name}" '
                f'will be available soon.',
                parent=self.winfo_toplevel()
            )

            return

        previous_dataset = self.dataset

        try:

            predictor = self._get_predictor(name)

            predictor.load_dataset(ds)

            self.dataset = ds

            self.navigate(name)

        except Exception as exc:

            self.dataset = previous_dataset

            messagebox.showerror(
                'Predictor Error',
                f'{ds.name}: {exc}',
                parent=self.winfo_toplevel()
            )

            return

        datasets = self.pages.get('datasets')

        if (
            datasets is not None
            and hasattr(datasets, '_close_info')
        ):

            datasets._close_info()
