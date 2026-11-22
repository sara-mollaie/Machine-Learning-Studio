import json
import math
import tkinter as tk
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageTk

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
    هرچه همبستگی بزرگ‌تر باشد رنگ تیره‌تر و هرچه کوچک‌تر باشد روشن‌تر.
    مقدارهای مثبت بنفش و مقدارهای منفی (اگر باشند) صورتی هستند؛ در هر دو،
    قدر مطلق بزرگ‌تر یعنی تیره‌تر.
    """
    t = min(1.0, abs(v))
    # سر روشن عمداً سفید نیست تا روی کارت گم نشود (لیلکی/صورتی واقعی)
    lo = "#b9a0ec" if dark else "#c6aaf0"                 # شروع مشترک = نوار رنگ بدون پرش
    if v >= 0:
        hi = "#5a2ea6" if dark else "#2f1366"
    else:
        hi = "#a3305f" if dark else "#8a1f4f"
    return _mix(lo, hi, t)


def _pretty(name, labels):
    if labels.get(name):
        return str(labels[name])
    text = str(name).replace("_", " ")
    return text.upper() if text.islower() and len(text) <= 4 else (text.title() if text.islower() else text)


# ---------- PIL drawings (supersampled, transparent background) ----------

SS = 3


def _scatter_img(w, h, gx, gy, pts, cols, centers, r, grid_col, frame_col,
                 edge_col, ctr_fill, ctr_edge, k):
    img = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")

    for x in gx:
        d.line([(x * SS, 0), (x * SS, h * SS)], fill=grid_col, width=SS)
    for y in gy:
        d.line([(0, y * SS), (w * SS, y * SS)], fill=grid_col, width=SS)
    d.rectangle((0, 0, w * SS - 1, h * SS - 1), outline=frame_col, width=SS)

    for (x, y), col in zip(pts, cols):
        rr = r * SS
        d.ellipse((x * SS - rr, y * SS - rr, x * SS + rr, y * SS + rr),
                  fill=_rgb(col) + (217,), outline=edge_col, width=max(1, int(0.8 * SS)))

    for x, y in centers:
        a = 9 * k * SS
        for width, color in ((int(5.5 * k * SS), ctr_edge), (int(3 * k * SS), ctr_fill)):
            d.line([(x * SS - a, y * SS - a), (x * SS + a, y * SS + a)], fill=color, width=width)
            d.line([(x * SS - a, y * SS + a), (x * SS + a, y * SS - a)], fill=color, width=width)

    return img.resize((w, h), Image.Resampling.LANCZOS)


def _bars_img(w, h, bars, grid_y=(), grid_col=None, radius=10):
    """bars = [(x_left, bar_w, bar_h, color)] با پایهٔ صاف و سر گرد"""
    img = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")

    for y in grid_y:
        d.line([(0, y * SS), (w * SS, y * SS)], fill=grid_col, width=SS)

    for x, bw, bh, color in bars:
        if bh <= 0:
            continue
        x0, x1 = x * SS, (x + bw) * SS
        y0, y1 = (h - bh) * SS, h * SS
        rr = min(radius * SS, (x1 - x0) / 2, (y1 - y0))
        d.rounded_rectangle((x0, y0, x1, y1 + rr), radius=rr, fill=color)
        d.rectangle((x0, y1 - rr, x1, y1), fill=color)

    return img.resize((w, h), Image.Resampling.LANCZOS)


def _heat_img(size, tiles, gap, radius):
    """tiles = [(col, row, color)] روی شبکه‌ای با اندازهٔ کاشی ثابت"""
    n_c = max(c for c, _, _ in tiles) + 1
    n_r = max(r for _, r, _ in tiles) + 1
    w = n_c * size + (n_c - 1) * gap
    h = n_r * size + (n_r - 1) * gap
    img = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")

    for c, r, color in tiles:
        x0, y0 = c * (size + gap) * SS, r * (size + gap) * SS
        d.rounded_rectangle((x0, y0, x0 + size * SS - 1, y0 + size * SS - 1),
                            radius=radius * SS, fill=color)

    return img.resize((w, h), Image.Resampling.LANCZOS)


def _gradient_bar(w, h, dark):
    img = Image.new("RGBA", (w, h))
    px = img.load()
    for y in range(h):
        v = 1 - 2 * y / max(h - 1, 1)                       # بالا = +1 (تیره)، وسط = 0 (روشن)، پایین = -1
        col = _rgb(_corr_color(v, dark)) + (255,)
        for x in range(w):
            px[x, y] = col
    mask = Image.new("L", (w * 4, h * 4), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w * 4 - 1, h * 4 - 1), w * 2, fill=255)
    img.putalpha(mask.resize((w, h), Image.Resampling.LANCZOS))
    return img


# =====================================================
# PAGE
# =====================================================

class VisualizationModule(DatasetsModule):
    """
    صفحهٔ Visualization برای دیتاست‌های Clustering.
    چهار نمودار یکسان برای همهٔ دیتاست‌های Clustering:
        Scatter (دو ویژگی دلخواه) · PCA · اندازهٔ خوشه‌ها · Correlation
    نقطه‌ها از processed_dataset و بقیه از متادیتا خوانده می‌شوند.
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

    def _draw_head(self):

        c = theme.c
        d = self.data
        crumb_y, x = 118, MX + 4

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
        x, y, h = MX, 238, 44
        self._pill_boxes = {}

        for key, label in CHARTS:
            avail = self._available(key)
            sel = key == self.chart
            w = self.text_width(label, 19, True) + 46
            tag = f"vz_{key}"

            if sel:
                img = self._shape(w, h, h / 2, BTN_DARK, BTN_LIGHT)
            else:
                img = self._shape(w, h, h / 2, pal["pill_off"], alpha=pal["pill_off_a"], border=c("card_border"))

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
                    img = self._shape(w, h, h / 2, BTN_DARK, BTN_LIGHT)
                else:
                    img = self._shape(w, h, h / 2, pal["pill_off"], alpha=pal["pill_off_a"], border=c("card_border"))

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
                                  image=self._shape(col_w - 8, row_h - 4, 12, BTN_DARK, BTN_LIGHT),
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

        c, S = theme.c, self.s
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

        # ----- عنوان + توضیح -----
        n_rows = int(len(d.labels)) if d.labels is not None else sum(d.sizes.values())
        titles = {
            "scatter": (f"{self._pn(self.xf)} vs {self._pn(self.yf)}",
                        f"{n_rows:,} records coloured by cluster" + (
                            f" · showing {len(d.idx):,}" if d.idx is not None and len(d.idx) < n_rows else "")),
            "pca": ("PCA clustering view",
                    "PC1 + PC2 keep " + (f"{d.pca.ev.sum():.1f}%" if d.pca else "–") + " of the variance"),
            "sizes": ("Cluster size comparison", f"{n_rows:,} records in {len(d.ids)} clusters"),
            "corr": ("Correlation heatmap", self._corr_subtitle()),
        }
        title, sub = titles[key]
        title_w = self.text_width(title, 26, True)
        self.text(x0 + 40, y0 + 52, title, 26, c("title"), True, tags=D)
        self.text(x0 + 40, y0 + 84, sub, 16, c("body"), tags=D)

        # ----- راهنمای رنگ خوشه‌ها -----
        if key in ("scatter", "pca", "sizes"):
            self._legend(x0 + CW - 40, y0 + 52, x0 + 40 + title_w + 30, centers=key in ("scatter", "pca"))

        box = (x0 + 110, y0 + 116, x0 + CW - 50, y0 + PH - 84)

        if key == "scatter":
            self._draw_xy(box, d.X[self.xf].to_numpy(float), d.X[self.yf].to_numpy(float),
                          self._centers_xy(self.xf, self.yf), self._pn(self.xf), self._pn(self.yf))
        elif key == "pca":
            comp = d.pca.comp
            self._draw_xy(box, comp[:, 0], comp[:, 1],
                          {cid: (v[0], v[1]) for cid, v in d.pca.centers.items()},
                          f"PC1 ({d.pca.ev[0]:.1f}% variance)", f"PC2 ({d.pca.ev[1]:.1f}% variance)")
        elif key == "sizes":
            self._draw_sizes(box)
        else:
            self._draw_corr((x0 + 56, y0 + 112, x0 + CW - 40, y0 + PH - 44))

    def _corr_subtitle(self):
        d = self.data
        if d.corr is None:
            return ""
        names, M = d.corr_names, d.corr
        best, pair = 0.0, None
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if abs(M[i][j]) > abs(best):
                    best, pair = float(M[i][j]), (names[i], names[j])
        if pair is None:
            return "Pearson correlation between the dataset's numeric columns"
        return (f"Darker = stronger correlation  ·  strongest pair: "
                f"{self._pn(pair[0])} · {self._pn(pair[1])} ({best:.2f})")

    def _centers_xy(self, xf, yf):
        d = self.data
        ix, iy = d.cols.index(xf), d.cols.index(yf)
        return {cid: (v[ix], v[iy]) for cid, v in d.centers.items()}

    def _legend(self, right, cy, min_left, centers):

        c, S = theme.c, self.s
        d = self.data
        items = [(f"Cluster {i}", self._color(i)) for i in d.ids]
        if centers:
            items.append(("Centers", c("title")))

        widths = [self.text_width(label, 15, True) + 44 for label, _ in items]
        gap = 10

        while items and sum(widths) + gap * (len(items) - 1) > right - min_left:
            drop = -2 if centers and len(items) > 2 else -1       # «Centers» همیشه می‌ماند
            items.pop(drop)
            widths.pop(drop)

        x = right - sum(widths) - gap * (len(items) - 1)

        for (label, col), w in zip(items, widths):
            self.create_image(S(x), S(cy - 17), image=self._shape(w, 34, 17, c("row_bg"), border=c("card_border")),
                              anchor="nw", tags=D)
            self.create_image(S(x + 20), S(cy), image=self._shape(11, 11, 5.5, col), anchor="center", tags=D)
            self.text(x + 34, cy, label, 15, c("title"), True, tags=D)
            x += w + gap

    # ---------- نمودار نقطه‌ای (Scatter / PCA) ----------

    def _draw_xy(self, box, X, Y, centers, xlab, ylab):

        c, S = theme.c, self.s
        d = self.data
        dark = theme.name == "dark"
        L, T, R, B = box
        wpx, hpx = S(R - L), S(B - T)

        xmin, xmax = float(X.min()), float(X.max())
        ymin, ymax = float(Y.min()), float(Y.max())
        px_, py_ = (xmax - xmin) * 0.06 or 1.0, (ymax - ymin) * 0.06 or 1.0
        x0, x1, y0, y1 = xmin - px_, xmax + px_, ymin - py_, ymax + py_

        def fx(v):
            return (v - x0) / (x1 - x0) * wpx

        def fy(v):
            return hpx - (v - y0) / (y1 - y0) * hpx

        xt, yt = _nice_ticks(x0, x1, 7), _nice_ticks(y0, y1, 6)

        idx = d.idx
        pts = [(fx(X[i]), fy(Y[i])) for i in idx]
        cols = [self._color(d.labels[i]) for i in idx]
        ctr = [(fx(x), fy(y)) for x, y in centers.values()]

        card = c("card_cols")[0] if dark else "#ffffff"
        key = ("vz_xy", self._ds_sig, self.k, theme.name, self.chart, self.xf, self.yf, wpx, hpx)

        if key not in self._cache:
            self._cache[key] = ImageTk.PhotoImage(_scatter_img(
                wpx, hpx, [fx(v) for v in xt], [fy(v) for v in yt], pts, cols, ctr,
                max(3.6, 4.6 * self.k), _rgb(c("body")) + (46,), _rgb(c("body")) + (70,),
                card, c("title") if not dark else "#f5efff", card, self.k))

        self.create_image(S(L), S(T), image=self._cache[key], anchor="nw", tags=D)

        for v in xt:
            self.text(L + fx(v) / self.k, B + 18, _fmt_tick(v), 14, c("body"), anchor="center", tags=D)
        for v in yt:
            self.text(L - 12, T + fy(v) / self.k, _fmt_tick(v), 14, c("body"), anchor="e", tags=D)

        self.text((L + R) / 2, B + 52, xlab, 16, c("body"), True, anchor="center", tags=D)
        self.create_text(S(L - 70), S((T + B) / 2), text=ylab, angle=90, fill=c("body"),
                         font=(FONT, -S(16), "bold"), tags=D)

    # ---------- اندازهٔ خوشه‌ها ----------

    def _draw_sizes(self, box):

        c, S = theme.c, self.s
        d = self.data
        L, T, R, B = box
        wpx, hpx = S(R - L), S(B - T)
        ids = d.ids
        values = [d.sizes[i] for i in ids]
        vmax = max(values)
        ticks = _nice_ticks(0, vmax * 1.12, 5)
        ymax = ticks[-1] if ticks and ticks[-1] >= vmax * 1.05 else vmax * 1.15

        bars = []
        slot = wpx / len(ids)
        bw = min(slot * 0.55, 120 * self.k)
        for i, (cid, v) in enumerate(zip(ids, values)):
            bh = hpx * v / ymax
            bars.append((i * slot + (slot - bw) / 2, bw, bh, self._color(cid)))

        gy = [hpx - hpx * t / ymax for t in ticks]
        key = ("vz_sizes", self._ds_sig, self.k, theme.name, wpx, hpx, tuple(round(b[2]) for b in bars))

        if key not in self._cache:
            self._cache[key] = ImageTk.PhotoImage(_bars_img(
                wpx, hpx, bars, gy, _rgb(c("body")) + (46,), radius=int(12 * self.k)))

        self.create_image(S(L), S(T), image=self._cache[key], anchor="nw", tags=D)

        for t, y in zip(ticks, gy):
            self.text(L - 12, T + y / self.k, _fmt_tick(t), 14, c("body"), anchor="e", tags=D)

        for (bx, bw_, bh, _), cid, v in zip(bars, ids, values):
            cx = L + (bx + bw_ / 2) / self.k
            pct = d.perc.get(cid)
            label = f"{v:,}" + (f" ({pct:.1f}%)" if pct is not None else "")
            self.text(cx, T + (hpx - bh) / self.k - 16, label, 16, c("title"), True, anchor="center", tags=D)
            self.text(cx, B + 18, f"Cluster {cid}", 15, c("body"), anchor="center", tags=D)

        self.text((L + R) / 2, B + 52, "Cluster", 16, c("body"), True, anchor="center", tags=D)
        self.create_text(S(L - 70), S((T + B) / 2), text="Number of records", angle=90, fill=c("body"),
                         font=(FONT, -S(16), "bold"), tags=D)

    # ---------- Correlation (بزرگ) ----------

    def _draw_corr(self, box):

        c, S = theme.c, self.s
        d = self.data
        dark = theme.name == "dark"
        L, T, R, B = box
        names, M = d.corr_names, d.corr
        n = len(names)

        label_w = max(self.text_width(self._pn(nm), 15, True) for nm in names) + 14
        bar_zone = 84
        avail_w = (R - L) - label_w - bar_zone
        avail_h = (B - T) - 34
        tile = int(_clamp(min(avail_w / n, avail_h / n), 18, 78))
        gap = 4 if tile >= 34 else 2
        step = tile + gap
        grid_w = n * tile + (n - 1) * gap

        gx = L + label_w + (avail_w - grid_w) / 2
        gy = T + (avail_h - grid_w) / 2

        tiles = [(j, i, _corr_color(1.0 if i == j else float(M[i][j]), dark))
                 for i in range(n) for j in range(n)]

        key = ("vz_heat", self.k, theme.name, n, tile, tuple(round(float(v), 2) for v in M.flatten()))
        if key not in self._cache:
            self._cache[key] = ImageTk.PhotoImage(_heat_img(S(tile), [(a, b, col) for a, b, col in tiles],
                                                            S(gap), S(min(10, tile * 0.18))))
        self.create_image(S(gx), S(gy), image=self._cache[key], anchor="nw", tags=D)

        num_size = _clamp(tile * 0.27, 11, 17)

        for i in range(n):
            self.text(gx - 12, gy + i * step + tile / 2, self._fit(self._pn(names[i]), 15, True, label_w),
                      15, c("body"), True, anchor="e", tags=D)
            self.text(gx + i * step + tile / 2, gy + grid_w + 16,
                      self._fit(self._pn(names[i]), 14, True, tile + 16), 14, c("body"), True,
                      anchor="center", tags=D)

            if tile >= 34:
                for j in range(n):
                    if i == j:
                        continue
                    v = float(M[i][j])
                    col = _corr_color(v, dark)
                    self.text(gx + j * step + tile / 2, gy + i * step + tile / 2, f"{v:.2f}", num_size,
                              "#1b1030" if _lum(col) > 0.55 else "#ffffff", True, anchor="center", tags=D)

        # نوار رنگ
        bx = gx + grid_w + 34
        bh = grid_w
        kb = ("vz_cbar", self.k, theme.name, bh)
        if kb not in self._cache:
            self._cache[kb] = ImageTk.PhotoImage(_gradient_bar(S(16), S(bh), dark))
        self.create_image(S(bx), S(gy), image=self._cache[kb], anchor="nw", tags=D)

        for v, y in ((1, gy), (0, gy + bh / 2), (-1, gy + bh)):
            self.text(bx + 28, y, f"{v:g}", 14, c("body"), tags=D)
        self.create_text(S(bx + 62), S(gy + bh / 2), text="Correlation", angle=270, fill=c("body"),
                         font=(FONT, -S(14), "bold"), tags=D)