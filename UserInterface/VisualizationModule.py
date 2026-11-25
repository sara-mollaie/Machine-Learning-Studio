import json
import math
import tkinter as tk
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
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter
from PIL import Image, ImageTk

from UserInterface.Datasets import (
    DatasetsModule, BG_LIGHT, BG_DARK, DESIGN_W, DESIGN_H, MX, BTN_DARK, BTN_LIGHT, CONTENT_W,
)
from UserInterface.DatasetNav import draw_dataset_navbar
from UserInterface.HomeModule import BASE, _rgb
from UserInterface.Theme import theme, FONT

# ---- چیدمان (واحد: پیکسل طراحی ۱۶۷۲×۹۴۱؛ مطابق طرح مرجع) ----
PY, PH = 316, 576                 # بالا و ارتفاع کارت اصلی
CW = CONTENT_W                    # کارت اصلی تمام عرض محتوا را می‌گیرد

# پس‌زمینه‌های مخصوص این صفحه
VIZ_BG = {
    "light": BASE / "assets" / "page_images" / "Visualization_background_light.png",
    "dark": BASE / "assets" / "page_images" / "Visualization_background_dark.png",
}

# اگر عنوان «Visualization» داخل عکس پس‌زمینه چاپ شده است True بماند
# (مثل «Predict the Cluster» در صفحهٔ Predictor)؛ اگر عکس عنوان ندارد False کنید.
TITLE_IN_BACKGROUND = True

D = ("dyn",)

CHARTS = [("scatter", "Scatter"), ("pca", "PCA"),
          ("sizes", "Cluster sizes"), ("corr", "Correlation")]

CLUSTER_COLORS = [
    "#8b5cc7", "#e0679f", "#2ca6a4", "#f0a83a", "#4a8fe0",
    "#8dbb3f", "#b05cc6", "#e0644d", "#5c6cd6", "#7a7a8c",
]

MAX_POINTS = 6000

# ---- پالت نمودارها: دقیقاً همان پالت RegressionVisualization ----
PALETTE = {
    "light": dict(fg="#2b1a52", muted="#65567f", grid="#a897c8", purple="#7547e8", purple_soft="#a98bf2",
                  pink="#e03a8c", card="#7547e8", pt_face="#7547e8", pt_edge="#ffffff", line="#e03a8c",
                  card_fill="#ffffff", card_fill_a=.58, card_edge="#c4b0ec", card_edge_a=.95),
    "dark": dict(fg="#f8efff", muted="#dccbf3", grid="#c4aee6", purple="#ad90ff", purple_soft="#8a6bd8",
                 pink="#ff7cb8", card="#c9b0ff", pt_face="#cfb8ff", pt_edge="#ffffff", line="#ff7cb8",
                 card_fill="#c9b0ff", card_fill_a=.12, card_edge="#dccbf3", card_edge_a=.45),
}

# گیره‌های انتخاب‌شده در تم روشن: گرادیان بنفش ← رز، و حاشیهٔ لیلکی برای گیره‌های غیرفعال
PILL_ON_LIGHT = ("#6d3ee8", "#e0408f")
PILL_BORDER_LIGHT = ("#cbb8ee", 255)

CARD_RECT = [0.685, 0.06, 0.295, 0.74]       # کادر خلاصهٔ سمت راست
_FONTS_READY = False


# =====================================================
# HELPERS
# =====================================================

def _mix(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _lum(color):
    r, g, b = _rgb(color)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255


def _cluster_key(v):
    s = str(v)
    return (int(s) if s.isdigit() else 10 ** 6, s)


def _nice_ticks(lo, hi, target=6):
    span = (hi - lo) or 1.0
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    step = mag
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if span / step <= target:
            break
    ticks, v = [], math.ceil(lo / step - 1e-9) * step
    while v <= hi + 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks


def _fmt_tick(v):
    if abs(v - round(v)) < 1e-9 or abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:g}"


def _corr_color(v, dark):
    """
    هرچه قدر مطلق همبستگی بزرگ‌تر باشد رنگ پررنگ‌تر؛ مثبت = بنفش، منفی = صورتی.
    در تم روشن رنگ سه‌مرحله‌ای است (لیلکی روشن ← بنفش میانی ← بنفش عمیق) تا میانه‌ها گل‌آلود نشوند.
    """
    t = min(1.0, abs(v))
    if dark:
        lo = "#b9a0ec"
        hi = "#5a2ea6" if v >= 0 else "#a3305f"
        return _mix(lo, hi, t)
    lo, mid = "#e6dcfa", ("#a98bf2" if v >= 0 else "#f08ab8")
    hi = "#4a1d9e" if v >= 0 else "#b3195f"
    return _mix(lo, mid, t * 2) if t < .5 else _mix(mid, hi, (t - .5) * 2)


def _pretty(name, labels):
    if labels.get(name):
        return str(labels[name])
    text = str(name).replace("_", " ")
    return text.upper() if text.islower() and len(text) <= 4 else (text.title() if text.islower() else text)


def _pill_on():
    return (BTN_DARK, BTN_LIGHT) if theme.name == "dark" else PILL_ON_LIGHT


def _pill_border(c):
    border = c("card_border") if theme.name == "dark" else PILL_BORDER_LIGHT
    if isinstance(border, str):
        return (border, 255)
    return tuple(border) if border else None


# ---------- matplotlib helpers (هم‌ظاهر با RegressionVisualization) ----------

def _font_family():
    """فونت برنامه برای نمودارها (فایل‌های assets/fonts هم به matplotlib معرفی می‌شوند)."""
    global _FONTS_READY
    from matplotlib import font_manager as fm
    if not _FONTS_READY:
        _FONTS_READY = True
        try:
            for pattern in ("*.ttf", "*.otf"):
                for path in (Path(BASE) / "assets" / "fonts").rglob(pattern):
                    fm.fontManager.addfont(str(path))
        except Exception:
            pass
    try:
        fm.findfont(FONT, fallback_to_default=False)
        return FONT
    except Exception:
        return "DejaVu Sans"


def _short_label(name, limit=16):
    """نام بلند را کوتاه می‌کند (وسط را با … حذف می‌کند) تا از کادر بیرون نزند."""
    name = str(name)
    if len(name) <= limit:
        return name
    keep = (limit - 1) // 2
    return name[:keep + 1] + "…" + name[-keep:]


def _mpl_style_axes(ax, pal, grid="y"):
    ax.set_facecolor("none")
    ax.tick_params(colors=pal["fg"], labelsize=12.5, length=0)
    ax.xaxis.label.set_color(pal["fg"])
    ax.yaxis.label.set_color(pal["fg"])
    ax.xaxis.label.set_size(13.5)
    ax.yaxis.label.set_size(13.5)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_weight("bold")
    for name, spine in ax.spines.items():
        spine.set_visible(name in ("left", "bottom"))
        spine.set_color(pal["muted"])
        spine.set_alpha(.65)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, alpha=.30, color=pal["grid"], linewidth=.9)


def _mpl_measure(ctx, text, size, weight="normal"):
    """عرض واقعی متن (به پیکسل طراحی)."""
    t = ctx.fig.text(0, 0, text, fontsize=size, fontweight=weight)
    w = t.get_window_extent(ctx.fig.canvas.get_renderer()).width / ctx.k
    t.remove()
    return w


def _mpl_wrap(ctx, text, size, max_px, weight="normal"):
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if _mpl_measure(ctx, trial, size, weight) <= max_px or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _mpl_card(ctx, rect, heading, entries, note=""):
    """کادر خلاصهٔ سمت راست؛ مقدارهای بلند به خط بعد می‌روند و از لبه بیرون نمی‌زنند."""
    pal = ctx.pal
    pw, ph = rect[2] * ctx.W, rect[3] * ctx.H
    pad = 30
    inner = pw - 2 * pad
    ax = ctx.fig.add_axes(rect)
    ax.set_xlim(0, pw)
    ax.set_ylim(0, ph)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((1, 1), pw - 2, ph - 2, boxstyle="round,pad=0,rounding_size=24",
                                facecolor=pal["card_fill"], alpha=pal["card_fill_a"], edgecolor="none"))
    ax.add_patch(FancyBboxPatch((1, 1), pw - 2, ph - 2, boxstyle="round,pad=0,rounding_size=24",
                                facecolor="none", edgecolor=pal["card_edge"], alpha=pal["card_edge_a"], linewidth=1.3))
    size = 15
    while _mpl_measure(ctx, heading, size, "bold") > inner and size > 11:
        size -= 1
    ax.text(pad, ph - 38, heading, color=pal["pink"], fontsize=size, fontweight="bold", va="center")
    ax.plot([pad, pw - pad], [ph - 64, ph - 64], color=pal["card_edge"], alpha=min(1, pal["card_edge_a"] + .1),
            linewidth=1.1)

    note_lines = _mpl_wrap(ctx, note, 11, inner) if note else []
    note_h = len(note_lines) * 20 + 22 if note_lines else 10
    top = ph - 104
    avail = top + 12 - note_h - 8
    vsize = 20
    while True:
        lh = vsize * 1.35
        blocks = [(label, _mpl_wrap(ctx, value, vsize, inner, "bold"), color)
                  for label, value, color in entries]
        heights = [31 + lh * (len(lines) - 1) + 20 for _, lines, _ in blocks]
        if sum(heights) <= avail or vsize <= 13:
            break
        vsize -= 1
    gap = max(0, min(30, (avail - sum(heights)) / max(len(blocks), 1)))
    y = top
    for (label, lines, color), h in zip(blocks, heights):
        ax.text(pad, y + 12, label, color=pal["muted"], fontsize=12, va="center")
        for i, line in enumerate(lines):
            ax.text(pad, y - 19 - i * lh, line, color=color or pal["fg"], fontsize=vsize,
                    fontweight="bold", va="center")
        y -= h + gap
    for i, line in enumerate(note_lines):
        ax.text(pad, 24 + (len(note_lines) - 1 - i) * 20, line, color=pal["muted"], fontsize=11,
                va="bottom")


def _cluster_legend(ctx, handles, x=0.075):
    """راهنمای خوشه‌ها در یک ردیف بالای نمودار؛ اگر زیاد بود دو ردیف. تعداد ردیف‌ها را برمی‌گرداند."""
    pal = ctx.pal
    n = len(handles)
    ncol = n if n <= 7 else math.ceil(n / 2)
    rows = math.ceil(n / ncol)
    ctx.fig.legend(handles=handles, loc="center left", bbox_to_anchor=(x, 0.775 if rows == 1 else 0.795),
                   ncol=ncol, frameon=False, fontsize=12.5 if rows == 1 else 11.5, labelcolor=pal["fg"],
                   handlelength=1.6, columnspacing=1.8 if rows == 1 else 1.4)
    return rows


# =====================================================
# PAGE
# =====================================================

class VisualizationModule(DatasetsModule):
    """
    صفحهٔ Visualization برای دیتاست‌های Clustering.
    چهار نمودار یکسان برای همهٔ دیتاست‌های Clustering:
        Scatter (دو ویژگی دلخواه) · PCA · اندازهٔ خوشه‌ها · Correlation
    نقطه‌ها از processed_dataset و بقیه از متادیتا خوانده می‌شوند.
    ظاهر نمودارها (matplotlib) مثل صفحهٔ RegressionVisualization است.
    """

    def __init__(self, master, scale=1.0, on_navigate=None):
        self._light = self._load_background(BG_LIGHT)
        self.k = scale
        self.W, self.H = int(DESIGN_W * scale), int(DESIGN_H * scale)
        tk.Canvas.__init__(self, master, width=self.W, height=self.H, highlightthickness=0, bd=0)

        self.on_navigate = on_navigate or (lambda page: print("goto", page))
        self._refs, self._bgs, self._cache = {}, {}, {}
        self._poll_job = None
        self._info = None

        self.ds = None
        self.data = None
        self.chart = "scatter"
        self.xf = self.yf = None
        self._menu = None
        self._pill_boxes = {}
        self._ds_sig = None
        self._dirty = True

        theme.subscribe(self._on_theme)

    # ---------- پس‌زمینه ----------

    def _bg_photo(self):
        name = theme.name

        if name not in self._bgs:
            path = VIZ_BG[name]

            if path.exists():
                img = Image.open(path).convert("RGB")
            else:                                         # اگر فایل نبود، پس‌زمینهٔ Datasets
                fallback = self._load_background(BG_DARK) if name == "dark" and BG_DARK.exists() else self._light
                img = fallback.convert("RGB")

            self._bgs[name] = ImageTk.PhotoImage(img.resize((self.W, self.H), Image.LANCZOS))

        return self._bgs[name]

    # ---------- چرخهٔ حیات ----------

    def destroy(self):
        try:
            theme.unsubscribe(self._on_theme)
        except Exception:
            pass
        tk.Canvas.destroy(self)

    def _on_theme(self):
        self._dirty = True
        if self.winfo_ismapped():
            self._build()

    def on_show(self):
        if self._dirty:
            self._build()

    def load_dataset(self, ds):
        try:
            sig = (str(ds.path), ds.path.stat().st_mtime_ns)
        except OSError:
            sig = (str(ds.path), None)

        if self.ds is not None and sig == self._ds_sig:
            return

        self._ds_sig, self.ds = sig, ds
        self._menu = None

        try:
            meta = json.loads(ds.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = ds.raw or {}

        self.data = self._load_data(ds, meta if isinstance(meta, dict) else {})
        d = self.data
        self.xf, self.yf = d.default_axes
        self.chart = next((key for key, _ in CHARTS if self._available(key)), "sizes")
        self._dirty = True

        if self.winfo_ismapped():
            self._build()

    # ---------- داده ----------

    def _load_data(self, ds, meta):

        cl = meta.get("Clustering") or meta.get("clustering") or {}
        fs = meta.get("feature_selection") or {}
        feats = list(fs.get("selected_features") or cl.get("input_features") or [])
        analysis = list(fs.get("analysis_only_columns") or [])
        lab_col = cl.get("cluster_label_column") or "ClusterLabel"
        names = meta.get("feature_labels") or {}

        d = SimpleNamespace(
            name=str(meta.get("name") or ds.name),
            labels_map=names,
            cols=[], X=None, labels=None, idx=None,
            pca=None, centers=None, default_axes=(None, None),
            points_error=None,
            sizes={}, perc={}, corr_names=[], corr=None,
            missing=(meta.get("dataset_statistics") or {}).get("total_missing_values"),
            ids=[],
        )

        # ----- نقطه‌ها: از فایل داده‌های پردازش‌شده -----
        try:
            root = ds.path.parent.parent
            files = meta.get("output_files") or {}
            rel = files.get("processed_dataset") or files.get("original_dataset")
            if not rel:
                raise FileNotFoundError("processed_dataset is missing in metadata output_files")

            df = pd.read_csv(root / rel)
            if lab_col not in df.columns:
                raise ValueError(f"'{lab_col}' column not found in the processed dataset")

            cols = [c for c in feats + analysis if c in df.columns]
            if not cols:
                raise ValueError("no feature columns found in the processed dataset")

            X = df[cols].apply(pd.to_numeric, errors="coerce")
            X = X.fillna(X.median())
            labels = df[lab_col].to_numpy(dtype=int)

            d.cols, d.X, d.labels = cols, X, labels

            rng = np.random.default_rng(42)
            d.idx = (np.arange(len(df)) if len(df) <= MAX_POINTS
                     else np.sort(rng.choice(len(df), MAX_POINTS, replace=False)))

            # دو ویژگی‌ای که خوشه‌ها را بیشتر از هم جدا می‌کنند = محورهای پیش‌فرض Scatter
            M = X.to_numpy(float)
            lo, hi = M.min(0), M.max(0)
            Z = (M - lo) / np.where(hi - lo == 0, 1, hi - lo)
            total = ((Z - Z.mean(0)) ** 2).sum(0)
            between = np.zeros(Z.shape[1])
            for c in np.unique(labels):
                part = Z[labels == c]
                between += len(part) * (part.mean(0) - Z.mean(0)) ** 2
            score = np.where(total > 0, between / np.where(total == 0, 1, total), 0)
            top = [cols[i] for i in np.argsort(-score)[:2]]
            if len(top) == 2:
                d.default_axes = (top[0], top[1])

            # PCA روی همان ویژگی‌های خوشه‌بندی (مقیاس MinMax) با SVD
            pca_cols = [c for c in feats if c in cols] or cols
            if len(pca_cols) >= 2:
                P = X[pca_cols].to_numpy(float)
                plo, phi = P.min(0), P.max(0)
                PZ = (P - plo) / np.where(phi - plo == 0, 1, phi - plo)
                mu = PZ.mean(0)
                _, S, Vt = np.linalg.svd(PZ - mu, full_matrices=False)
                comp = (PZ - mu) @ Vt[:2].T
                ratio = (S ** 2) / np.sum(S ** 2) * 100
                centers = {int(c): ((PZ[labels == c].mean(0) - mu) @ Vt[:2].T) for c in np.unique(labels)}
                d.pca = SimpleNamespace(comp=comp, ev=ratio[:2], centers=centers)

            d.centers = {int(c): X[labels == c].mean().to_numpy(float) for c in np.unique(labels)}

        except Exception as error:
            d.points_error = str(error)

        # ----- اندازهٔ خوشه‌ها -----
        d.sizes = {str(k): int(v) for k, v in (cl.get("cluster_sizes") or {}).items()}
        d.perc = {str(k): float(v) for k, v in (cl.get("cluster_percentages") or {}).items()}

        if not d.sizes and d.labels is not None:
            u, n = np.unique(d.labels, return_counts=True)
            d.sizes = {str(int(a)): int(b) for a, b in zip(u, n)}
            d.perc = {k: v / sum(d.sizes.values()) * 100 for k, v in d.sizes.items()}

        d.ids = sorted(d.sizes.keys(), key=_cluster_key)

        # ----- همبستگی: از متادیتا (یا محاسبه از داده) -----
        cm = meta.get("correlation_matrix")
        if isinstance(cm, dict) and cm:
            names_c = list(cm.keys())
            try:
                d.corr_names = names_c
                d.corr = np.array([[float(cm[a].get(b, 0)) for b in names_c] for a in names_c])
            except Exception:
                d.corr_names, d.corr = [], None
        elif d.X is not None:
            d.corr_names = list(d.X.columns)
            d.corr = d.X.corr().to_numpy(float)

        return d

    def _available(self, key):
        d = self.data
        if d is None:
            return False
        if key == "scatter":
            return d.X is not None and len(d.cols) >= 2
        if key == "pca":
            return d.pca is not None
        if key == "sizes":
            return bool(d.sizes)
        if key == "corr":
            return d.corr is not None
        return False

    def _color(self, cid):
        ids = self.data.ids
        i = ids.index(str(cid)) if str(cid) in ids else int(cid)
        return CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

    def _pn(self, name):
        return _pretty(name, self.data.labels_map)

    # =====================================================
    # رسم
    # =====================================================

    def _build(self):
        self._dirty = False
        self.delete("all")
        self.config(cursor="")
        self.create_image(0, 0, image=self._bg_photo(), anchor="nw", tags=("bgimg",))
        self.tag_bind("bgimg", "<Button-1>", lambda e: self._close_menu())
        draw_dataset_navbar(self, active="visualization")
        if self.ds is not None:
            self._draw_head()
            self._draw_dynamic()

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
        self.tag_bind(tag, "<Button-1>", lambda e: self.on_navigate("evaluate"))

    def _draw_head(self):

        c = theme.c
        d = self.data
        crumb_y, x = 118, MX + 4
        self._draw_back_button()
        x += 104

        self.text(x, crumb_y, "Datasets", 20, c("accent"), True, tags=("crumb_ds",))
        self.hover("crumb_ds")
        self.tag_bind("crumb_ds", "<Button-1>", lambda e: self.on_navigate("datasets"))
        x += self.text_width("Datasets", 20, True) + 10

        self.text(x, crumb_y, "/", 20, c("body"))
        x += self.text_width("/", 20) + 10

        self.text(x, crumb_y, d.name, 20, c("body"), tags=("crumb_name",))
        self.hover("crumb_name")
        self.tag_bind("crumb_name", "<Button-1>", lambda e: self.on_navigate("predictor"))
        x += self.text_width(d.name, 20) + 10

        self.text(x, crumb_y, "/  Visualization", 20, c("body"))

        if not TITLE_IN_BACKGROUND:
            self.text(MX + 4, 168, "Visualization", 58, c("title"), True)

        self.text(MX + 6, 212,
                  f"Explore {d.name}: how the clusters separate, how big they are and how features relate",
                  20, c("body"))

    def _draw_dynamic(self):
        self.delete("dyn")
        self._draw_pills()
        self._draw_main()
        if self._menu:
            self._draw_menu()

    # ---------- گیره‌ها (انتخاب نوع نمودار) ----------

    def _draw_pills(self):

        c, pal = theme.c, self._pal()
        x, y, h = MX, 254, 44
        self._pill_boxes = {}

        for key, label in CHARTS:
            avail = self._available(key)
            sel = key == self.chart
            w = self.text_width(label, 19, True) + 46
            tag = f"vz_{key}"

            if sel:
                img = self._shape(w, h, h / 2, *_pill_on())
            else:
                img = self._shape(w, h, h / 2, pal["pill_off"], alpha=pal["pill_off_a"], border=_pill_border(c))

            self.create_image(self.s(x), self.s(y), image=img, anchor="nw", tags=D + (tag,))
            self.text(x + w / 2, y + h / 2, label, 19,
                      "white" if sel else (c("title") if avail else c("body")), True,
                      anchor="center", tags=D + (tag,))

            if avail:
                self.hover(tag)
                self.tag_bind(tag, "<Button-1>", lambda e, k=key: self._set_chart(k))
            x += w + 10

        # انتخاب محور Scatter (فقط در حالت Scatter)
        if self.chart == "scatter" and self._available("scatter"):
            right = MX + 1544
            for axis, name in (("y", self.yf), ("x", self.xf)):
                label = f"{axis.upper()}  {self._pn(name)}"
                w = self.text_width(label, 19, True) + 70
                tag = f"vz_axis_{axis}"
                active = self._menu == axis

                if active:
                    img = self._shape(w, h, h / 2, *_pill_on())
                else:
                    img = self._shape(w, h, h / 2, pal["pill_off"], alpha=pal["pill_off_a"], border=_pill_border(c))

                self.create_image(self.s(right - w), self.s(y), image=img, anchor="nw", tags=D + (tag,))
                self.text(right - w + 24, y + h / 2, label, 19, "white" if active else c("title"), True,
                          tags=D + (tag,))

                ax, ay = right - 26, y + h / 2
                up = 1 if active else -1
                self.create_polygon(self.s(ax - 7), self.s(ay + 3 * up), self.s(ax + 7), self.s(ay + 3 * up),
                                    self.s(ax), self.s(ay - 4 * up),
                                    fill="white" if active else c("title"), outline="", tags=D + (tag,))

                self._pill_boxes[axis] = (right - w, y + h + 8, right)
                self.hover(tag)
                self.tag_bind(tag, "<Button-1>", lambda e, a=axis: self._toggle_menu(a))
                right -= w + 10

    def _set_chart(self, key):
        self.chart, self._menu = key, None
        self._draw_dynamic()

    def _toggle_menu(self, axis):
        self._menu = None if self._menu == axis else axis
        self._draw_dynamic()

    def _close_menu(self):
        if self._menu:
            self._menu = None
            self._draw_dynamic()

    def _pick_axis(self, axis, name):
        if axis == "x":
            self.xf = name
        else:
            self.yf = name
        self._menu = None
        self._draw_dynamic()

    def _draw_menu(self):

        c = theme.c
        axis = self._menu
        cols = self.data.cols
        row_h, per_col = 40, 8
        n_cols = math.ceil(len(cols) / per_col)
        col_w = max(self.text_width(self._pn(f), 18, True) for f in cols) + 64
        pw, ph = n_cols * col_w + 24, min(len(cols), per_col) * row_h + 24
        left_, top_, right_ = self._pill_boxes[axis]
        px = right_ - pw

        img, pad = self._glass(pw, ph, 24, c("card_cols")[0], c("card_end"), 255,
                               c("shadow_a") + 30, c("shadow_rgb"), c("card_border"))
        self.create_image(self.s(px) - pad, self.s(top_) - pad, image=img, anchor="nw", tags=D + ("vz_menu",))
        self.tag_bind("vz_menu", "<Button-1>", lambda e: None)

        current = self.xf if axis == "x" else self.yf

        for i, name in enumerate(cols):
            col, row = divmod(i, per_col)
            x = px + 12 + col * col_w
            y = top_ + 12 + row * row_h
            tag = f"vz_item_{i}"
            sel = name == current

            if sel:
                self.create_image(self.s(x), self.s(y + 2),
                                  image=self._shape(col_w - 8, row_h - 4, 12, *_pill_on()),
                                  anchor="nw", tags=D + (tag,))
            self.text(x + 22, y + row_h / 2, self._pn(name), 18, "white" if sel else c("title"), True,
                      tags=D + (tag,))
            self.hover(tag)
            self.tag_bind(tag, "<Button-1>", lambda e, a=axis, n=name: self._pick_axis(a, n))

    # ---------- کارت‌های شیشه‌ای ----------

    def _glass_card(self, x, y, w, h, tags=D, r=36):
        c, S = theme.c, self.s
        img, pad = self._glass(w, h, r, c("card_cols")[0], c("card_end"), c("card_alpha"),
                               c("shadow_a"), c("shadow_rgb"), c("card_border"))
        return self.create_image(S(x) - pad, S(y) - pad, image=img, anchor="nw", tags=tags)

    # =====================================================
    # کارت اصلی
    # =====================================================

    def _draw_main(self):

        c = theme.c
        d = self.data
        x0, y0 = MX, PY

        self._glass_card(x0, y0, CW, PH, D + ("vz_main",))
        self.tag_bind("vz_main", "<Button-1>", lambda e: self._close_menu())

        key = self.chart

        if not self._available(key):
            msg = d.points_error or "This chart is not available for this dataset."
            self.text(x0 + 40, y0 + 52, "No data to show", 26, c("title"), True, tags=D)
            for i, line in enumerate(self._wrap(msg, 17, False, CW - 80, 3)):
                self.text(x0 + 40, y0 + 100 + i * 26, line, 17, c("body"), tags=D)
            return

        ck = ("vz_chart", self._ds_sig, self.k, theme.name, key, self.xf, self.yf)

        if ck not in self._cache:
            old = [k for k in self._cache if isinstance(k, tuple) and k and k[0] == "vz_chart"]
            if len(old) > 10:                                  # حافظه: قدیمی‌ترین نمودارها را نگه ندار
                del self._cache[old[0]]
            try:
                self._cache[ck] = self._vz_render()
            except Exception as exc:
                self.text(x0 + 40, y0 + 52, "Chart unavailable", 26, c("title"), True, tags=D)
                for i, line in enumerate(self._wrap(str(exc), 17, False, CW - 80, 3)):
                    self.text(x0 + 40, y0 + 100 + i * 26, line, 17, c("body"), tags=D)
                return

        self.create_image(self.s(x0 + CW / 2), self.s(y0 + PH / 2), image=self._cache[ck],
                          anchor="center", tags=D + ("vz_main",))

    # =====================================================
    # رسم نمودارها با matplotlib (مثل RegressionVisualization)
    # =====================================================

    def _vz_render(self):
        d, key = self.data, self.chart
        dark = theme.name == "dark"
        pal = PALETTE["dark" if dark else "light"]
        W, H = CW - 50, PH - 30                                # اندازهٔ طراحی (پیکسل)

        with matplotlib.rc_context({"font.family": _font_family(), "axes.unicode_minus": True}):
            fig = Figure(figsize=(W / 100, H / 100), dpi=100 * self.k, facecolor="none")
            FigureCanvasAgg(fig)
            ctx = SimpleNamespace(fig=fig, W=W, H=H, pal=pal, dark=dark, d=d, k=self.k)
            builder = {"scatter": self._vz_scatter, "pca": self._vz_pca,
                       "sizes": self._vz_sizes, "corr": self._vz_corr}[key]
            title, subtitle = builder(ctx)
            fig.text(0.035, 0.935, f"{d.name} — {title}", ha="left", va="center", color=pal["fg"],
                     fontsize=20, fontweight="bold")
            fig.text(0.035, 0.872, subtitle, ha="left", va="center", color=pal["muted"], fontsize=12)
            canvas = fig.canvas
            canvas.draw()
            return ImageTk.PhotoImage(Image.fromarray(np.asarray(canvas.buffer_rgba()).copy()))

    def _centers_xy(self, xf, yf):
        d = self.data
        ix, iy = d.cols.index(xf), d.cols.index(yf)
        return {cid: (v[ix], v[iy]) for cid, v in d.centers.items()}

    def _pct(self, cid):
        d = self.data
        if d.perc.get(cid) is not None:
            return d.perc[cid]
        total = sum(d.sizes.values()) or 1
        return d.sizes[cid] / total * 100

    def _size_entries(self, ctx):
        d, pal = ctx.d, ctx.pal
        ids = d.ids
        big = max(ids, key=lambda i: d.sizes[i])
        small = min(ids, key=lambda i: d.sizes[i])
        return [("Clusters", str(len(ids)), pal["pink"]),
                ("Records", f"{sum(d.sizes.values()):,}", None),
                ("Largest cluster", f"Cluster {big}  {self._pct(big):.1f}%", None),
                ("Smallest cluster", f"Cluster {small}  {self._pct(small):.1f}%", None)]

    # ---- ۱) Scatter ----
    def _vz_scatter(self, ctx):
        d = ctx.d
        X = d.X[self.xf].to_numpy(float)
        Y = d.X[self.yf].to_numpy(float)
        self._vz_xy(ctx, X, Y, self._centers_xy(self.xf, self.yf), self._pn(self.xf), self._pn(self.yf))
        n_rows, shown = len(d.labels), len(d.idx)
        sub = f"{n_rows:,} records coloured by cluster" + (f" • showing {shown:,}" if shown < n_rows else "")
        _mpl_card(ctx, CARD_RECT, "Cluster summary", self._size_entries(ctx),
                  "Each colour is one cluster. The × marks show the cluster centres.")
        return f"{self._pn(self.xf)} vs {self._pn(self.yf)}", sub

    # ---- ۲) PCA ----
    def _vz_pca(self, ctx):
        d, pal = ctx.d, ctx.pal
        comp, ev = d.pca.comp, d.pca.ev
        self._vz_xy(ctx, comp[:, 0], comp[:, 1], {cid: (v[0], v[1]) for cid, v in d.pca.centers.items()},
                    f"PC1 ({ev[0]:.1f}% variance)", f"PC2 ({ev[1]:.1f}% variance)")
        entries = [("PC1 (first component)", f"{ev[0]:.1f}%", pal["pink"]),
                   ("PC2 (second component)", f"{ev[1]:.1f}%", None),
                   ("PC1 + PC2 together", f"{ev.sum():.1f}%", None),
                   ("Clusters", str(len(d.ids)), None)]
        _mpl_card(ctx, CARD_RECT, "Variance explained", entries,
                  "Each point is a record projected onto the first two principal components.")
        return "PCA clustering view", f"PC1 + PC2 keep {ev.sum():.1f}% of the variance"

    def _vz_xy(self, ctx, X, Y, centers, xlab, ylab):
        """نمودار پراکندگی مشترک Scatter و PCA (رنگ هر خوشه جدا، مراکز با ×)."""
        pal, d, dark = ctx.pal, ctx.d, ctx.dark
        edge = "#1b1030" if dark else "#ffffff"

        handles = [Line2D([0], [0], marker="o", linestyle="", markersize=9, markerfacecolor=self._color(i),
                          markeredgecolor=pal["pt_edge"], markeredgewidth=1.0, label=f"Cluster {i}")
                   for i in d.ids]
        if centers:
            handles.append(Line2D([0], [0], marker="X", linestyle="", markersize=10, markerfacecolor=pal["fg"],
                                  markeredgecolor=edge, markeredgewidth=1.2, label="Centers"))
        rows = _cluster_legend(ctx, handles)

        bottom, height = (0.12, 0.61) if rows == 1 else (0.12, 0.58)
        ax = ctx.fig.add_axes([0.075, bottom, 0.575, height])
        _mpl_style_axes(ax, pal, grid="both")

        order = np.random.default_rng(7).permutation(d.idx)      # ترتیب تصادفی تا یک خوشه روی بقیه نیفتد
        colors = [self._color(d.labels[i]) for i in order]
        n = len(order)
        few = n <= 600
        ax.scatter(X[order], Y[order], s=70 if n <= 200 else (40 if few else 22), c=colors,
                   edgecolors=pal["pt_edge"], linewidths=1.0 if few else .35, alpha=.97 if few else .85,
                   zorder=3, rasterized=True)
        if centers:
            ax.scatter([v[0] for v in centers.values()], [v[1] for v in centers.values()], marker="X", s=210,
                       c=pal["fg"], edgecolors=edge, linewidths=1.6, zorder=5)

        xmin, xmax = float(X.min()), float(X.max())
        ymin, ymax = float(Y.min()), float(Y.max())
        px_, py_ = (xmax - xmin) * .05 or 1.0, (ymax - ymin) * .05 or 1.0
        ax.set_xlim(xmin - px_, xmax + px_)
        ax.set_ylim(ymin - py_, ymax + py_)
        fmt = FuncFormatter(lambda v, _: _fmt_tick(v))
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

    # ---- ۳) اندازهٔ خوشه‌ها ----
    def _vz_sizes(self, ctx):
        pal, d = ctx.pal, ctx.d
        ids = d.ids
        values = np.array([d.sizes[i] for i in ids], dtype=float)
        top = float(values.max()) * 1.22

        ax = ctx.fig.add_axes([0.075, 0.15, 0.575, 0.62])
        _mpl_style_axes(ax, pal, grid="y")
        xs = np.arange(len(ids))
        ax.bar(xs, values, width=.56, color=[self._color(i) for i in ids])
        ax.set_ylim(0, top)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _fmt_tick(v)))
        ax.set_ylabel("Number of records")
        ax.set_xlabel("Cluster")
        ax.set_xticks(xs, [f"Cluster {i}" for i in ids])
        ax.tick_params(axis="x", labelsize=12 if len(ids) <= 6 else 10.5)
        for tick in ax.get_xticklabels():
            tick.set_fontweight("bold")

        many = len(ids) > 5
        for x, cid, v in zip(xs, ids, values):
            label = f"{int(v):,}\n{self._pct(cid):.1f}%" if many else f"{int(v):,} ({self._pct(cid):.1f}%)"
            ax.text(x, v + top * .015, label, ha="center", va="bottom", color=pal["fg"],
                    fontsize=14 if not many else 11, fontweight="bold", linespacing=1.25)

        _mpl_card(ctx, CARD_RECT, "Size summary", self._size_entries(ctx),
                  "Bar height is the number of records assigned to each cluster.")
        return "Cluster Size Comparison", f"{int(values.sum()):,} records in {len(ids)} clusters"

    # ---- ۴) همبستگی ----
    def _vz_corr(self, ctx):
        pal, d, dark = ctx.pal, ctx.d, ctx.dark
        names_all, M_all = list(d.corr_names), np.asarray(d.corr, float)
        n_all = len(names_all)
        keep = list(range(n_all))
        if n_all > 14:                                         # برای خوانایی: ۱۴ ستون با بیشترین همبستگی
            A = np.abs(np.nan_to_num(M_all))
            np.fill_diagonal(A, 0)
            keep = sorted(np.argsort(-A.mean(1))[:14].tolist())
        M = M_all[np.ix_(keep, keep)]
        names = [names_all[i] for i in keep]
        n = len(names)
        fig, W, H = ctx.fig, ctx.W, ctx.H

        side_h, side_w = .60, .37
        x0, y0 = .155, .19
        ax = fig.add_axes([x0, y0, side_w, side_h])
        ax.set_facecolor("none")
        ax.set_xlim(-.5, n - .5)
        ax.set_ylim(n - .5, -.5)
        cell_w, cell_h = side_w * W / n, side_h * H / n           # اندازهٔ هر خانه به پیکسل
        fs = max(8.5, min(13, .26 * min(cell_w, cell_h * 1.6)))
        annotate = min(cell_w, cell_h * 1.6) >= 30
        aspect = cell_h / cell_w                                   # گوشه‌های گرد، بدون کشیدگی
        gap = 5                                                    # فاصلهٔ بین خانه‌ها (پیکسل)
        gx, gy = gap / cell_w, gap / cell_h
        for i in range(n):
            for j in range(n):
                val = 1.0 if i == j else float(M[i, j])
                ok = np.isfinite(val)
                col = _corr_color(val, dark) if ok else "#777777"
                ax.add_patch(FancyBboxPatch((j - .5 + gx / 2, i - .5 + gy / 2), 1 - gx, 1 - gy,
                                            boxstyle="round,pad=0,rounding_size=.12", mutation_aspect=aspect,
                                            facecolor=col, edgecolor="none", alpha=.55 if i == j else 1))
                if i != j and annotate:
                    txt = f"{val:.2f}".replace("0.", ".", 1) if ok else "—"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=fs, fontweight="bold",
                            color="#1b1030" if _lum(col) > .55 else "#ffffff")
        labels = [_short_label(self._pn(nm)) for nm in names]
        ax.set_xticks(range(n), labels)
        ax.set_yticks(range(n), labels)
        ax.tick_params(length=0, labelsize=10 if n <= 12 else 8.5, colors=pal["fg"])
        for lbl in ax.get_xticklabels():                       # برچسب‌های محور X همیشه مورب تا روی هم نیفتند
            lbl.set_rotation(40)
            lbl.set_ha("right")
            lbl.set_rotation_mode("anchor")
        for s_ in ax.spines.values():
            s_.set_visible(False)

        cmap = LinearSegmentedColormap.from_list("vzcorr", [_corr_color(-1, dark), _corr_color(0, dark),
                                                            _corr_color(1, dark)])
        cax = fig.add_axes([x0 + side_w + .018, y0, .011, side_h])
        cb = fig.colorbar(ScalarMappable(norm=Normalize(-1, 1), cmap=cmap), cax=cax, ticks=[-1, -.5, 0, .5, 1])
        cb.ax.tick_params(colors=pal["fg"], labelsize=10.5, length=0)
        cb.set_label("Pearson r", color=pal["fg"], fontsize=11.5)
        cb.outline.set_visible(False)

        pairs = sorted(((abs(M[i, j]), i, j) for i in range(n) for j in range(i + 1, n)
                        if np.isfinite(M[i, j])), reverse=True)[:3]
        entries = [(f"#{k + 1}  strongest pair", f"{self._pn(names[i])} · {self._pn(names[j])}  {M[i, j]:+.2f}",
                    pal["purple"] if M[i, j] >= 0 else pal["pink"]) for k, (_, i, j) in enumerate(pairs)]
        _mpl_card(ctx, CARD_RECT, "Strongest correlations", entries,
                  "Closer to ±1 means a stronger linear relationship. Purple = positive, pink = negative.")

        if pairs:
            _, i, j = pairs[0]
            sub = (f"Darker = stronger correlation • strongest pair: "
                   f"{self._pn(names[i])} · {self._pn(names[j])} ({M[i, j]:.2f})")
        else:
            sub = "Pearson correlation between the dataset's numeric columns"
        if n < n_all:
            sub = f"Showing the {n} most correlated of {n_all} columns • " + sub
        return "Correlation Heatmap", sub



