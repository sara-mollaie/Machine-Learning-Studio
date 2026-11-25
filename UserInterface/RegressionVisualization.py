"""Regression charts from saved metadata and held-out predictions; no retraining."""
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from PIL import Image, ImageTk

from UserInterface.RegressionFeatureRanking import ranked_coefficients
from UserInterface.Theme import theme, FONT
from UserInterface.VisualizationModule import VisualizationModule, MX, CW, PY, PH, D
from UserInterface.VisualizationModule import _corr_color, _lum, PILL_ON_LIGHT, PILL_BORDER_LIGHT

TABS = [('importance', 'Feature Importance'), ('corr', 'Correlation'), ('actual', 'Actual vs Predicted'),
        ('accuracy', 'Model Accuracy'), ('residual', 'Residual Distribution')]

TITLES = {
    'importance': 'Feature Importance',
    'corr': 'Correlation Heatmap',
    'actual': 'Actual vs Predicted',
    'accuracy': 'Model Accuracy Comparison',
    'residual': 'Residual Error Distribution',
}

# ---- پالت چارت‌ها (برای هر دو تم، با کنتراست بالا روی کارت شیشه‌ای) ----
PALETTE = {
    'light': dict(fg='#2b1a52', muted='#65567f', grid='#a897c8', purple='#7547e8', purple_soft='#a98bf2',
                  pink='#e03a8c', card='#7547e8', pt_face='#7547e8', pt_edge='#ffffff', line='#e03a8c',
                  card_fill='#ffffff', card_fill_a=.58, card_edge='#c4b0ec', card_edge_a=.95),
    'dark': dict(fg='#f8efff', muted='#dccbf3', grid='#c4aee6', purple='#ad90ff', purple_soft='#8a6bd8',
                 pink='#ff7cb8', card='#c9b0ff', pt_face='#cfb8ff', pt_edge='#ffffff', line='#ff7cb8',
                 card_fill='#c9b0ff', card_fill_a=.12, card_edge='#dccbf3', card_edge_a=.45),
}

CARD_RECT = [0.685, 0.06, 0.295, 0.74]       # کادر راست: بزرگ‌تر تا متن به لبه نچسبد
_FONTS_READY = False


def _font_family():
    """فونت برنامه برای نمودارها (فایل‌های assets/fonts هم به matplotlib معرفی می‌شوند)."""
    global _FONTS_READY
    from matplotlib import font_manager as fm
    if not _FONTS_READY:
        _FONTS_READY = True
        try:
            from UserInterface.HomeModule import BASE
            for pattern in ('*.ttf', '*.otf'):
                for path in (BASE / 'assets' / 'fonts').rglob(pattern):
                    fm.fontManager.addfont(str(path))
        except Exception:
            pass
    try:
        fm.findfont(FONT, fallback_to_default=False)
        return FONT
    except Exception:
        return 'DejaVu Sans'


def _num(value, digits=1):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return '—'
    return f'{v:,.{digits}f}' if np.isfinite(v) else '—'


def _short(name, limit=16):
    """نام بلند را کوتاه می‌کند (وسط را با … حذف می‌کند) تا از کادر بیرون نزند."""
    name = str(name).replace('self_reference_', 'ref_')
    if len(name) <= limit:
        return name
    keep = (limit - 1) // 2
    return name[:keep + 1] + '…' + name[-keep:]


def _density(x, y, bins=64):
    """چگالی نسبی هر نقطه (برای رنگ‌آمیزی نمودار پراکندگی)."""
    h, xe, ye = np.histogram2d(x, y, bins=bins)
    xi = np.clip(np.digitize(x, xe[:-1]) - 1, 0, bins - 1)
    yi = np.clip(np.digitize(y, ye[:-1]) - 1, 0, bins - 1)
    return h[xi, yi]


class RegressionVisualization(VisualizationModule):
    # =====================================================
    # بارگذاری داده
    # =====================================================
    def load_dataset(self, ds):
        self.ds = ds
        self._menu = None
        self._cache.clear()
        self.meta = json.loads(Path(ds.path).read_text(encoding='utf-8-sig'))
        m = self.meta
        ev = m.get('model_evaluation') or {}
        fs = m.get('feature_selection') or {}
        self.data = SimpleNamespace(
            name=m.get('name') or ds.name, model=ev.get('recommended_model', 'Unknown'),
            target=fs.get('target_column'), rows=ev.get('results') or [],
            importance=(m.get('feature_importance') or {}).get('results') or [],
            actual=None, predicted=None, corr=None, error='')
        d = self.data
        files = m.get('output_files') or {}
        root = Path(ds.path).parent.parent
        try:
            rel = files.get('processed_dataset') or files.get('rankings')
            if not rel:
                raise ValueError('Saved prediction file is not recorded.')
            p = root / rel.replace('\\', '/')
            cols = pd.read_csv(p, nrows=0).columns.tolist()
            low = {v: v.lower() for v in cols}
            pred = next((v for v in cols if low[v].startswith('predicted')), None) \
                or next((v for v in cols if 'predict' in low[v]), None)
            actual = d.target if d.target in cols else (
                next((v for v in cols if low[v].startswith('actual')), None))
            features = [v for v in fs.get('selected_features', []) if v in cols]
            raw = fs.get('raw_target_column')
            wanted = set(features + [actual, pred, raw])
            frame = pd.read_csv(p, usecols=lambda v: v in wanted)
            if actual and pred:
                pairs = frame[[actual, pred]].apply(pd.to_numeric, errors='coerce') \
                    .replace([np.inf, -np.inf], np.nan).dropna()
                d.actual, d.predicted = pairs[actual].to_numpy(), pairs[pred].to_numpy()
            if actual and features and not files.get('rankings') \
                    and not str(d.name).lower().startswith('household'):
                names = list(dict.fromkeys(features + ([raw] if d.name == 'Abalone' and raw in frame else [])
                                           + [actual]))
                numeric = frame[names].select_dtypes(include='number')
                corr = numeric.corr()
                if actual in corr and len(corr) > 12:
                    top = corr[actual].drop(actual).abs().nlargest(9).index.tolist() + [actual]
                    corr = corr.loc[top, top]
                if len(corr) > 1:
                    d.corr = corr
        except Exception as exc:
            d.error = str(exc)
        cm = m.get('correlation_matrix')
        if isinstance(cm, dict) and cm:
            try:
                d.corr = pd.DataFrame(cm, dtype=float)
            except (ValueError, TypeError):
                pass
        self.chart = 'importance'
        self._dirty = True
        if self.winfo_ismapped():
            self._build()

    # =====================================================
    # سرصفحه و تب‌ها
    # =====================================================
    def _draw_back_button(self):
        S = self.s
        dark = theme.name == "dark"
        color = "#f1d5fa" if dark else "#ffffff"
        bg = "#50336f" if dark else "#713b91"
        edge = "#805ba1" if dark else "#713b91"
        w, h = 90, 30
        x, y = MX + 4, 103
        tag = "viz_back"
        self.create_image(S(x), S(y), image=self._shape(w, h, 10, bg, border=(edge, 255)), anchor="nw", tags=(tag,))
        cy = y + h / 2
        self.create_line(S(x+29), S(cy), S(x+14), S(cy), fill=color, width=max(1,S(2)), capstyle="round", tags=(tag,))
        self.create_line(S(x+21), S(cy-7), S(x+14), S(cy), S(x+21), S(cy+7), fill=color, width=max(1,S(2)), capstyle="round", joinstyle="round", tags=(tag,))
        self.text(x+37, cy, "Back", 16, color, True, tags=(tag,))
        self.hover(tag)
        self.tag_bind(tag, "<Button-1>", lambda e: self.on_navigate("regression_evaluate"))

    def _draw_head(self):
        c = theme.c
        S = self.s
        x, y = MX + 4, 118
        self._draw_back_button()
        x += 104
        self.text(x, y, 'Datasets', 20, c('accent'), True, tags=('crumb_ds',))
        self.hover('crumb_ds')
        self.tag_bind('crumb_ds', '<Button-1>', lambda e: self.on_navigate('datasets'))
        x += self.text_width('Datasets', 20, True) + 10
        self.text(x, y, f'/  {self.data.name}  /  Visualization', 20, c('body'))
        self.text(MX + 6, 212, 'Explore feature influence, prediction quality and regression model performance',
                  20, c('body'))

    def _tabs(self):
        return TABS

    def _draw_pills(self):
        x = MX
        c = theme.c
        for key, label in self._tabs():
            w = self.text_width(label, 19, True) + 46
            tag = 'regviz_' + key
            selected = key == self.chart
            light = theme.name != 'dark'
            if selected and light:
                img = self._shape(w, 44, 22, *PILL_ON_LIGHT)
            else:
                border = PILL_BORDER_LIGHT if light else c('card_border')
                if isinstance(border, str):
                    border = (border, 255)
                elif isinstance(border, list):
                    border = tuple(border)
                img = self._shape(w, 44, 22, '#a3305f' if selected else c('row_bg'),
                                  border=border)
            self.create_image(self.s(x), self.s(254), image=img, anchor='nw', tags=D + (tag,))
            self.text(x + w / 2, 276, label, 19, 'white' if selected else c('title'), True,
                      anchor='center', tags=D + (tag,))
            self.hover(tag)
            self.tag_bind(tag, '<Button-1>', lambda e, k=key: self._set_chart(k))
            x += w + 10

    def _draw_main(self):
        self._glass_card(MX, PY, CW, PH)
        key = (self.chart, theme.name, self.k)
        if key not in self._cache:
            try:
                self._cache[key] = self._render()
            except Exception as exc:
                self.text(MX + 40, PY + 55, 'Chart unavailable', 26, theme.c('title'), True, tags=D)
                for i, line in enumerate(self._wrap(str(exc), 18, False, CW - 80, 4)):
                    self.text(MX + 40, PY + 105 + i * 28, line, 18, theme.c('body'), tags=D)
                return
        self.create_image(self.s(MX + CW / 2), self.s(PY + PH / 2), image=self._cache[key],
                          anchor='center', tags=D)

    # =====================================================
    # رسم نمودارها
    # =====================================================
    def _render(self):
        d, key = self.data, self.chart
        dark = theme.name == 'dark'
        pal = PALETTE['dark' if dark else 'light']
        W, H = CW - 50, PH - 30                      # اندازهٔ طراحی (پیکسل)
        with matplotlib.rc_context({'font.family': _font_family(), 'axes.unicode_minus': True}):
            fig = Figure(figsize=(W / 100, H / 100), dpi=100 * self.k, facecolor='none')
            FigureCanvasAgg(fig)
            ctx = SimpleNamespace(fig=fig, W=W, H=H, pal=pal, dark=dark, d=d, k=self.k)
            builder = {'importance': self._importance, 'corr': self._corr, 'accuracy': self._accuracy,
                       'actual': self._actual, 'residual': self._residual}[key]
            subtitle = builder(ctx)
            fig.text(0.035, 0.935, f'{d.name} — {TITLES[key]}', ha='left', va='center', color=pal['fg'],
                     fontsize=20, fontweight='bold')
            fig.text(0.035, 0.872, subtitle, ha='left', va='center', color=pal['muted'], fontsize=12)
            canvas = fig.canvas
            canvas.draw()
            return ImageTk.PhotoImage(Image.fromarray(np.asarray(canvas.buffer_rgba()).copy()))

    # ---- ابزارهای مشترک ----
    @staticmethod
    def _style_axes(ax, pal, grid='y'):
        ax.set_facecolor('none')
        ax.tick_params(colors=pal['fg'], labelsize=12.5, length=0)
        ax.xaxis.label.set_color(pal['fg'])
        ax.yaxis.label.set_color(pal['fg'])
        ax.xaxis.label.set_size(13.5)
        ax.yaxis.label.set_size(13.5)
        ax.xaxis.label.set_weight('bold')
        ax.yaxis.label.set_weight('bold')
        for name, spine in ax.spines.items():
            spine.set_visible(name in ('left', 'bottom'))
            spine.set_color(pal['muted'])
            spine.set_alpha(.65)
        ax.set_axisbelow(True)
        if grid:
            ax.grid(axis=grid, alpha=.30, color=pal['grid'], linewidth=.9)

    @staticmethod
    def _measure(ctx, text, size, weight='normal'):
        """عرض واقعی متن (به پیکسل طراحی)."""
        t = ctx.fig.text(0, 0, text, fontsize=size, fontweight=weight)
        w = t.get_window_extent(ctx.fig.canvas.get_renderer()).width / ctx.k
        t.remove()
        return w

    @classmethod
    def _wrap_px(cls, ctx, text, size, max_px, weight='normal'):
        lines, cur = [], ''
        for word in text.split():
            trial = (cur + ' ' + word).strip()
            if cls._measure(ctx, trial, size, weight) <= max_px or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines

    @classmethod
    def _card(cls, ctx, rect, heading, entries, note=''):
        """کادر خلاصهٔ سمت راست: عنوان، معیارها با فاصلهٔ کافی و توضیح که از لبه بیرون نمی‌زند."""
        pal = ctx.pal
        pw, ph = rect[2] * ctx.W, rect[3] * ctx.H
        pad = 30
        inner = pw - 2 * pad
        ax = ctx.fig.add_axes(rect)
        ax.set_xlim(0, pw)
        ax.set_ylim(0, ph)
        ax.axis('off')
        ax.add_patch(FancyBboxPatch((1, 1), pw - 2, ph - 2, boxstyle='round,pad=0,rounding_size=24',
                                    facecolor=pal['card_fill'], alpha=pal['card_fill_a'], edgecolor='none'))
        ax.add_patch(FancyBboxPatch((1, 1), pw - 2, ph - 2, boxstyle='round,pad=0,rounding_size=24',
                                    facecolor='none', edgecolor=pal['card_edge'], alpha=pal['card_edge_a'], linewidth=1.3))
        size = 15
        while cls._measure(ctx, heading, size, 'bold') > inner and size > 11:
            size -= 1
        ax.text(pad, ph - 38, heading, color=pal['pink'], fontsize=size, fontweight='bold', va='center')
        ax.plot([pad, pw - pad], [ph - 64, ph - 64], color=pal['card_edge'], alpha=min(1, pal['card_edge_a'] + .1),
                linewidth=1.1)

        note_lines = cls._wrap_px(ctx, note, 11, inner) if note else []
        note_h = len(note_lines) * 20 + 22 if note_lines else 10
        top = ph - 104
        avail = top + 12 - note_h - 8
        vsize = 20
        while True:                                   # مقدار بلند به چند خط شکسته می‌شود، نه بیرون‌زدن از کادر
            lh = vsize * 1.35
            blocks = [(label, cls._wrap_px(ctx, value, vsize, inner, 'bold'), color)
                      for label, value, color in entries]
            heights = [31 + lh * (len(lines) - 1) + 20 for _, lines, _ in blocks]
            if sum(heights) <= avail or vsize <= 13:
                break
            vsize -= 1
        gap = max(0, min(30, (avail - sum(heights)) / max(len(blocks), 1)))
        y = top
        for (label, lines, color), h in zip(blocks, heights):
            ax.text(pad, y + 12, label, color=pal['muted'], fontsize=12, va='center')
            for i, line in enumerate(lines):
                ax.text(pad, y - 19 - i * lh, line, color=color or pal['fg'], fontsize=vsize,
                        fontweight='bold', va='center')
            y -= h + gap
        for i, line in enumerate(note_lines):
            ax.text(pad, 24 + (len(note_lines) - 1 - i) * 20, line, color=pal['muted'], fontsize=11,
                    va='bottom')

    # ---- ۱) اهمیت ویژگی‌ها ----
    def _importance(self, ctx):
        pal, d = ctx.pal, ctx.d
        rows = ranked_coefficients(self.meta)
        if not rows:
            raise ValueError('Saved feature coefficients are missing.')
        rows = [(str(n), float(v)) for n, v in rows]
        by_strength = sorted(rows, key=lambda r: abs(r[1]), reverse=True)
        rows = by_strength                                         # قوی‌ترین تأثیر در بالا، ضعیف‌ترین در پایین
        names = [r[0] for r in rows]
        values = np.array([r[1] for r in rows])

        ax = ctx.fig.add_axes([0.135, 0.12, 0.50, 0.61])
        self._style_axes(ax, pal, grid='x')
        ypos = np.arange(len(rows))[::-1]
        colors = [pal['purple'] if v >= 0 else pal['pink'] for v in values]
        ax.barh(ypos, values, color=colors, height=.64)
        ax.set_yticks(ypos, names)
        ax.axvline(0, color=pal['muted'], linewidth=1.2, alpha=.8)
        span = max(abs(values).max(), 1e-9)
        ax.set_xlim(min(0, values.min()) - span * .20, max(0, values.max()) + span * .20)
        for y, v in zip(ypos, values):
            ax.text(v + (span * .02 if v >= 0 else -span * .02), y, f'{v:+.2f}', va='center',
                    ha='left' if v >= 0 else 'right', color=pal['fg'], fontsize=11.5, fontweight='bold')
        ax.set_xlabel('Standardized regression coefficient')
        # راهنما بیرون از نمودار و در یک ردیف بالای آن (مثل بقیهٔ نمودارها)، پس هرگز روی میله‌ها نمی‌افتد
        ctx.fig.legend(handles=[Patch(color=pal['purple'], label='Increases the prediction'),
                                Patch(color=pal['pink'], label='Decreases the prediction')],
                       loc='center left', bbox_to_anchor=(0.135, 0.775), ncol=2, frameon=False, fontsize=12.5,
                       labelcolor=pal['fg'], handlelength=1.4, columnspacing=1.8)

        top3 = by_strength[:3]
        entries = [(f'#{i + 1}  {"raises" if v >= 0 else "lowers"} {d.target or "the target"}',
                    f'{n}  {v:+.2f}', pal['purple'] if v >= 0 else pal['pink'])
                   for i, (n, v) in enumerate(top3)]
        self._card(ctx, CARD_RECT, 'Top drivers', entries,
                   'Inputs are standardized, so bar lengths are directly comparable across features.')
        return 'Signed coefficients from the saved linear regression model'

    # ---- ۲) همبستگی ----
    def _corr(self, ctx):
        pal, d, dark = ctx.pal, ctx.d, ctx.dark
        if d.corr is None:
            raise ValueError('Correlation matrix is not saved and suitable feature data is unavailable.')
        corr = d.corr
        if len(corr) > 12:
            target = d.target if d.target in corr.columns else corr.columns[-1]
            names = corr[target].drop(target).abs().nlargest(9).index.tolist() + [target]
            corr = corr.loc[names, names]
        n = len(corr)
        fig, W, H = ctx.fig, ctx.W, ctx.H

        side_h, side_w = .60, .37
        x0, y0 = .155, .19
        ax = fig.add_axes([x0, y0, side_w, side_h])
        ax.set_facecolor('none')
        ax.set_xlim(-.5, n - .5)
        ax.set_ylim(n - .5, -.5)
        cell_w, cell_h = side_w * W / n, side_h * H / n          # اندازهٔ هر خانه به پیکسل
        fs = max(8.5, min(13, .26 * min(cell_w, cell_h * 1.6)))
        aspect = cell_h / cell_w                                   # گوشه‌های گرد، بدون کشیدگی
        gap = 5                                                    # فاصلهٔ بین خانه‌ها (پیکسل، یکسان در افقی و عمودی)
        gx, gy = gap / cell_w, gap / cell_h
        for i in range(n):
            for j in range(n):
                val = corr.iloc[i, j]
                col = _corr_color(float(val), dark) if np.isfinite(val) else '#777777'
                ax.add_patch(FancyBboxPatch((j - .5 + gx / 2, i - .5 + gy / 2), 1 - gx, 1 - gy,
                                            boxstyle='round,pad=0,rounding_size=.12', mutation_aspect=aspect,
                                            facecolor=col, edgecolor='none', alpha=.55 if i == j else 1))
                if i != j:
                    txt = f'{val:.2f}'.replace('0.', '.', 1) if np.isfinite(val) else '—'
                    ax.text(j, i, txt, ha='center', va='center', fontsize=fs, fontweight='bold',
                            color='#1b1030' if _lum(col) > .55 else '#ffffff')
        ax.set_xticks(range(n), [_short(c_) for c_ in corr.columns])
        ax.set_yticks(range(n), [_short(c_) for c_ in corr.index])
        ax.tick_params(length=0, labelsize=10, colors=pal['fg'])
        for lbl in ax.get_xticklabels():                       # برچسب‌های محور X همیشه مورب
            lbl.set_rotation(40)
            lbl.set_ha('right')
            lbl.set_rotation_mode('anchor')
        for s_ in ax.spines.values():
            s_.set_visible(False)

        cmap = LinearSegmentedColormap.from_list('regcorr', [_corr_color(-1, dark), _corr_color(0, dark),
                                                             _corr_color(1, dark)])
        cax = fig.add_axes([x0 + side_w + .018, y0, .011, side_h])
        cb = fig.colorbar(ScalarMappable(norm=Normalize(-1, 1), cmap=cmap), cax=cax, ticks=[-1, -.5, 0, .5, 1])
        cb.ax.tick_params(colors=pal['fg'], labelsize=10.5, length=0)
        cb.set_label('Pearson r', color=pal['fg'], fontsize=11.5)
        cb.outline.set_visible(False)

        target = d.target if d.target in corr.columns else corr.columns[-1]
        links = corr[target].drop(target)
        links = links.reindex(links.abs().sort_values(ascending=False).index)[:3]
        entries = [(f'#{i + 1}  strongest link', f'{name}  {val:+.2f}',
                    pal['purple'] if val >= 0 else pal['pink']) for i, (name, val) in enumerate(links.items())]
        self._card(ctx, CARD_RECT, f'Linked to {target}', entries,
                   'Closer to ±1 means a stronger linear relationship. Purple = positive, pink = negative.')
        return 'Pearson correlation between inputs and the target'

    # ---- ۳) دقت مدل‌ها ----
    def _accuracy(self, ctx):
        pal, d = ctx.pal, ctx.d
        rows = []
        for r in d.rows:
            try:
                acc = float(r.get('Accuracy'))
            except (TypeError, ValueError):
                continue
            if np.isfinite(acc):
                rows.append((str(r.get('Model')), acc, r))
        rows.sort(key=lambda r: r[1], reverse=True)
        if not rows:
            raise ValueError('Saved model accuracy is missing.')
        names = [r[0] for r in rows]
        values = np.array([r[1] for r in rows])

        ax = ctx.fig.add_axes([0.075, 0.25, 0.575, 0.54])
        self._style_axes(ax, pal, grid='y')
        xs = np.arange(len(rows))
        bars = ax.bar(xs, values, width=.56,
                      color=[pal['pink'] if n == d.model else pal['purple_soft'] for n in names])
        top = min(max(100, values.max() + 24), 130)
        ax.set_ylim(min(0, values.min() * 1.1), top)
        ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= 100])
        ax.set_ylabel('Accuracy score (%)')
        ax.set_xticks(xs, [n.replace(' ', '\n') for n in names])
        ax.tick_params(axis='x', labelsize=11.5, pad=6)
        for tick in ax.get_xticklabels():
            tick.set_fontweight('bold')
            tick.set_linespacing(1.15)
        for x, (n, acc, r) in zip(xs, rows):
            ax.text(x, acc + top * .015, f'{acc:.1f}%', ha='center', va='bottom', color=pal['fg'],
                    fontsize=14, fontweight='bold')
            if n == d.model:
                ax.text(x, acc + top * .135, 'RECOMMENDED', ha='center', va='bottom', fontsize=9.5,
                        fontweight='bold', color='white',
                        bbox=dict(boxstyle='round,pad=0.32', facecolor=pal['pink'], edgecolor='none'))
            ax.text(x, -.21, f'R² {_num(r.get("R2"), 3)}\nRMSE {_num(r.get("RMSE"), 1)}', ha='center', va='top',
                    transform=ax.get_xaxis_transform(), color=pal['muted'], fontsize=10.5, linespacing=1.4)

        best = next((r for r in rows if r[0] == d.model), rows[0])
        info = best[2]
        entries = [('Accuracy score', f'{best[1]:.1f}%', pal['pink']),
                   ('R² (variance explained)', _num(info.get('R2'), 3), None),
                   ('RMSE (typical error)', _num(info.get('RMSE'), 1), None),
                   ('MAE (average error)', _num(info.get('MAE'), 1), None)]
        definition = (self.meta.get('model_evaluation') or {}).get('accuracy_definition')
        note = 'Accuracy = 100 × (1 − RMSE / mean |actual|), a descriptive score, not classification accuracy.' \
            if definition else 'Accuracy is the saved RMSE-based score for each model.'
        self._card(ctx, CARD_RECT, best[0], entries, note)
        return 'Held-out test results • the pink bar is the recommended model'

    # ---- ۴) واقعی در برابر پیش‌بینی ----
    def _actual(self, ctx):
        pal, d, dark = ctx.pal, ctx.d, ctx.dark
        a, p = self._pairs(d)
        idx = np.random.default_rng(42).choice(len(a), min(6000, len(a)), replace=False)
        x, y = a[idx], p[idx]
        n = len(idx)
        lo, hi = min(a.min(), p.min()), max(a.max(), p.max())
        pad = (hi - lo) * .04

        ax = ctx.fig.add_axes([0.075, 0.12, 0.575, 0.61])
        self._style_axes(ax, pal, grid='both')
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=pal['line'], linewidth=3.0, zorder=2,
                solid_capstyle='round')
        if n <= 600:                                  # نمونهٔ کم: نقطه‌های درشت با لبهٔ سفید
            ax.scatter(x, y, s=78 if n <= 200 else 44, c=pal['pt_face'], edgecolors=pal['pt_edge'],
                       linewidths=1.1, alpha=.97, zorder=3)
        else:                                         # نمونهٔ زیاد: رنگ بر اساس تراکم
            z = _density(x, y)
            order = np.argsort(z)
            cmap = LinearSegmentedColormap.from_list(
                'dens', ['#b9a1ff', '#ff7ab8'] if dark else ['#6a35c9', '#d1327d'])
            ax.scatter(x[order], y[order], c=z[order], cmap=cmap, s=20, alpha=.85, edgecolors='none',
                       rasterized=True, zorder=3)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel('Actual values')
        ax.set_ylabel('Predicted values')
        ctx.fig.legend(handles=[Line2D([0], [0], color=pal['line'], linewidth=3.0, label='Perfect prediction'),
                                Line2D([0], [0], marker='o', linestyle='', markersize=9,
                                       markerfacecolor=pal['pt_face'], markeredgecolor=pal['pt_edge'],
                                       markeredgewidth=1.1, label='Model predictions')],
                       loc='center left', bbox_to_anchor=(0.075, 0.775), ncol=2, frameon=False, fontsize=12.5,
                       labelcolor=pal['fg'], handlelength=1.8, columnspacing=2.2)

        err = a - p
        ss_res, ss_tot = float((err ** 2).sum()), float(((a - a.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot else float('nan')
        entries = [('R² (variance explained)', _num(r2, 3), pal['pink']),
                   ('RMSE (typical error)', _num(np.sqrt((err ** 2).mean()), 1), None),
                   ('MAE (average error)', _num(np.abs(err).mean(), 1), None),
                   ('Saved test predictions', f'{len(a):,}', None)]
        self._card(ctx, CARD_RECT, d.model, entries,
                   'Points close to the diagonal line are accurate predictions.')
        return f'{d.model} • {len(idx):,} of {len(a):,} saved prediction pairs'

    # ---- ۵) توزیع خطا ----
    def _residual(self, ctx):
        pal, d, dark = ctx.pal, ctx.d, ctx.dark
        a, p = self._pairs(d)
        err = a - p
        mean, std = float(err.mean()), float(err.std())
        rmse = float(np.sqrt((err ** 2).mean()))

        ax = ctx.fig.add_axes([0.075, 0.12, 0.575, 0.61])
        self._style_axes(ax, pal, grid='y')
        counts, edges, patches = ax.hist(err, bins=36, edgecolor=pal['fg'], linewidth=.5)
        centers = (edges[:-1] + edges[1:]) / 2
        for c_, patch in zip(centers, patches):
            patch.set_facecolor(pal['purple'] if abs(c_ - mean) <= std else pal['purple_soft'])
            patch.set_alpha(.95 if abs(c_ - mean) <= std else .6)
        ax.axvline(0, color=pal['pink'], linewidth=2.4)
        for s_ in (-std, std):
            ax.axvline(mean + s_, color=pal['fg'], linewidth=1.3, linestyle=(0, (5, 4)), alpha=.75)
        ax.set_xlabel('Residual (actual − predicted)')
        ax.set_ylabel('Number of predictions')
        ctx.fig.legend(handles=[Line2D([0], [0], color=pal['pink'], linewidth=2.4, label='Zero error'),
                                Patch(color=pal['purple'], label='Within ±1 standard deviation'),
                                Line2D([0], [0], color=pal['fg'], linewidth=1.3, linestyle=(0, (5, 4)),
                                       label='±1 std boundary')],
                       loc='center left', bbox_to_anchor=(0.075, 0.775), ncol=3, frameon=False, fontsize=12.5,
                       labelcolor=pal['fg'], handlelength=1.6, columnspacing=1.8)

        within = float((np.abs(err) <= rmse).mean() * 100)
        entries = [('Mean error (bias)', f'{mean:+.2f}', pal['pink']),
                   ('Standard deviation', _num(std, 1), None),
                   (f'Within ± RMSE ({_num(rmse, 1)})', f'{within:.0f}% of cases', None),
                   ('Saved test predictions', f'{len(a):,}', None)]
        self._card(ctx, CARD_RECT, 'Error summary', entries,
                   'A histogram centered on zero means the model has no systematic over- or under-prediction.')
        return f'{d.model} • {len(a):,} saved prediction pairs'

    @staticmethod
    def _pairs(d):
        if d.actual is None or not len(d.actual):
            raise ValueError(d.error or 'Saved actual/predicted pairs are missing.')
        return d.actual, d.predicted
