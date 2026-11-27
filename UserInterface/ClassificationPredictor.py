"""Classification UI: reuses the Regression predictor (features panel, pagination,
missing-value highlighting, ...) and replaces only the algorithm bar and result panel."""
import math

from UserInterface.HomeModule import BASE
from UserInterface.PredictorModule import PY, PH, RX, RW
from UserInterface.Theme import theme

from .ClassificationData import ClassificationData
from .RegressionPredictor import RegressionPredictor


class ClassificationPredictor(RegressionPredictor):
    DATA_CLASS = ClassificationData

    MODEL_NAMES = {
        'KNN': 'K-Nearest Neighbors',
        'Logistic': 'Logistic Regression',
        'Decision Tree': 'Decision Tree Classifier',
        'Random Forest': 'Random Forest Classifier',
        'SVM': 'Support Vector Machine',
        'Naive Bayes': 'Gaussian Naive Bayes',
    }

    # ---------- background ----------
    def _load_pred_bg(self, name):
        """Use 'predict Class_background_<theme>.png' if present, else the Regression one."""
        from PIL import Image
        folder = BASE / 'assets' / 'page_images'
        for stem in (f'predict Class_background_{name}.png', f'predict Reg_background_{name}.png'):
            path = folder / stem
            if path.is_file():
                with Image.open(path) as image:
                    return image.convert('RGB').resize((self.W, self.H), Image.Resampling.LANCZOS)
        raise FileNotFoundError(f'Classification background not found in: {folder}')

    # ---------- metrics ----------
    def _metric(self, key, suffix=''):
        try:
            v = self.data.accuracy_pct if key.lower() == 'accuracy' else float(self.data.best[key])
            return f'{v:,.2f}{suffix}' if v is not None and math.isfinite(v) else '—'
        except (KeyError, TypeError, ValueError):
            return '—'

    # ---------- feature importance (from metadata, no heavy computation) ----------
    @staticmethod
    def _compute_importance(data, metadata_path):
        info = data.meta.get('feature_importance') or {}
        ranked = []
        for item in info.get('results') or []:
            try:
                ranked.append((str(item['Feature']), abs(float(item['Importance']))))
            except (KeyError, TypeError, ValueError):
                continue
        ranked.sort(key=lambda kv: -kv[1])
        status = str(info.get('method') or 'Feature importance') if ranked else 'Feature importance unavailable'
        return ranked[:3], status

    # ---------- algorithm bar ----------
    def _draw_algorithm(self, px, pw, pal):
        c, S = theme.c, self.s
        x, inner = px + 48, pw - 96
        top = PY + PH - 168
        aw = inner - 170
        dark = theme.name == 'dark'
        border = ('#7a5aa6' if dark else '#bfaedf', 255)
        self.create_line(S(x), S(top - 52), S(x + inner), S(top - 52), fill=pal['sep'], width=max(1, S(1.5)))
        self._section_icon(x, top - 39, 'algorithm', '#f3a4d0' if dark else '#a3306f', 30)
        self.text(x + 42, top - 24, 'Algorithm', 20, c('title'), True)
        self.text(x + aw + 20, top - 24, 'Accuracy', 20, c('title'), True)
        for xx, w, label in [(x, aw, 'Classification'), (x + aw + 20, 150, self._metric('Accuracy', '%'))]:
            self.create_image(S(xx), S(top), image=self._shape(w, 52, 26, pal['search_bg'], border=border), anchor='nw')
            self.text(xx + 26, top + 26, self._fit(label, 22, True, w - 40), 22, c('title'), True)

    # ---------- result panel ----------
    def _draw_result(self):
        if not self.bundle:
            return
        S, T = self.s, ('res',)
        left, right = RX + 48, RX + RW - 48
        width = right - left
        center = (left + right) / 2
        muted, accent = '#e1c9ed', '#ffc0df'

        def txt(x, y, value, size=18, bold=False, color='white', anchor='w'):
            self.text(x, PY + y, str(value), size, color, bold, anchor=anchor, tags=T)

        def card(x, y, w, h):
            self.create_image(S(x), S(PY + y), image=self._shape(w, h, 18, '#ffffff', alpha=28), anchor='nw', tags=T)

        def line(points, color, weight=2):
            self.create_line(*[S(v) for x, y in points for v in (x, PY + y)],
                             fill=color, width=max(1, S(weight)), tags=T)

        def icon(x, y, kind):
            if kind == 'award':
                self.create_oval(S(x - 7), S(PY + y - 10), S(x + 7), S(PY + y + 4),
                                 outline=accent, width=max(1, S(2)), tags=T)
                line([(x - 5, y + 3), (x - 7, y + 11), (x, y + 7), (x + 7, y + 11), (x + 5, y + 3)], accent, 2)
            elif kind == 'target':
                for r in (10, 5):
                    self.create_oval(S(x - r), S(PY + y - r), S(x + r), S(PY + y + r),
                                     outline=accent, width=max(1, S(2)), tags=T)
            else:
                for dy in (-6, 0, 6):
                    line([(x - 10, y + dy), (x, y + dy - 5), (x + 10, y + dy), (x, y + dy + 5), (x - 10, y + dy)],
                         accent, 1.5)

        # --- header: icon, title, illustrative classification decision boundary
        self._section_icon(left, PY + 23, 'prediction', '#f3a4d0', 36, tags=T)
        txt(left + 48, 41, 'Prediction', 28, True)
        # Compact classification icon with clear space below.
        line([(center - 43, 60), (center - 43, 96),
              (center + 43, 96)], '#b77bad', 1.5)
        self.create_line(S(center - 22), S(PY + 94),
                         S(center + 22), S(PY + 60),
                         fill=muted, dash=(4, 3),
                         width=max(1, S(1.5)), tags=T)
        class_a = [(-33, 70), (-21, 78), (-13, 64),
                   (-32, 87), (-7, 72)]
        class_b = [(10, 87), (23, 73), (32, 84),
                   (35, 64), (25, 91)]
        for dx, y in class_a:
            x, r = center + dx, 3
            self.create_oval(S(x - r), S(PY + y - r),
                             S(x + r), S(PY + y + r),
                             fill=accent, outline='', tags=T)
        for dx, y in class_b:
            x, r = center + dx, 3
            self.create_rectangle(S(x - r), S(PY + y - r),
                                  S(x + r), S(PY + y + r),
                                  fill='#b9a0ff', outline='', tags=T)

        # --- predicted class
        txt(center, 124, self._fit('Predicted ' + str(self.data.target), 19, False, width), 19, False, muted, 'center')
        value = '—' if self.result is None else str(self.result)
        value_size = 48
        while value_size > 1 and self.text_width(value, value_size, True) > width:
            value_size -= 1
        txt(center, 163, value, value_size, True, accent, 'center')

        confidence = getattr(self.result, 'confidence', None)
        if self.notice:
            msg = self.notice
        elif self.result is None:
            msg = 'Set every feature value, then press Predict.'
        elif confidence is not None:
            msg = f'Confidence {confidence * 100:.1f}%'
        else:
            msg = 'Prediction generated'
        txt(center, 203, self._fit(msg, 17, False, width), 17, False, muted, 'center')
        line([(left, 224), (right, 224)], '#795184', 1)

        # --- recommended model
        model = str(self.data.best['Model'])
        card(left, 238, width, 83)
        icon(left + 25, 257, 'award')
        txt(left + 43, 257, 'Recommended Model', 17, False, muted)
        txt(left + 16, 284, self._fit(model, 24, True, width - 32), 24, True)
        full = self.MODEL_NAMES.get(model, '')
        if full and full != model:
            txt(left + 16, 306, self._fit(full, 16, False, width - 32), 16, False, muted)

        # --- accuracy / F1
        card(left, 333, width, 61)
        txt(left + 16, 352, 'Accuracy', 19, True)
        txt(right - 16, 352, self._metric('Accuracy', '%'), 24, True, accent, 'e')
        f1 = self.data.f1_pct
        txt(left + 16, 377, f'F1 (weighted) {f1:.2f}%' if f1 is not None else 'Test-set accuracy',
            16, False, muted)

        # --- key predictors
        card(left, 408, width, 187)
        icon(left + 25, 429, 'layers')
        txt(left + 46, 429, 'Key Predictors', 21, True)
        ranked = self._importance_data or []
        if ranked:
            peak = max(abs(v) for _, v in ranked) or 1.0
            for i, (feature, importance) in enumerate(ranked[:3]):
                y = 459 + i * 40
                txt(left + 18, y, f'{i + 1:02d}', 15, True, accent)
                txt(left + 48, y, self._fit(self._pretty(feature), 17, True, width - 70), 17, True)
                bx, ex = left + 48, right - 20
                by = y + 15
                self.create_line(S(bx), S(PY + by), S(ex), S(PY + by), fill='#795184',
                                 width=max(1, S(6)), capstyle='round', tags=T)
                self.create_line(S(bx), S(PY + by), S(bx + (ex - bx) * importance / peak), S(PY + by),
                                 fill=accent, width=max(1, S(6)), capstyle='round', tags=T)
            txt(left + 18, 581, self._fit(self._importance_status or '', 13, False, width - 36), 13, False, muted)
        else:
            txt(center, 505, self._fit(self._importance_status or 'Feature importance unavailable', 17, False, width - 32),
                17, False, muted, 'center')

        line([(left, 610), (right, 610)], '#795184', 1)
        txt(left, 632, 'Learning Type', 18, True)
        txt(right, 632, 'Supervised Learning', 17, True, muted, 'e')
