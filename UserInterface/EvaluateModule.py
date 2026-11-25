import json
import math
import tkinter as tk
from types import SimpleNamespace

from PIL import Image, ImageDraw, ImageTk

from UserInterface.Datasets import (
    DatasetsModule, DESIGN_W, DESIGN_H, MX,
)
from UserInterface.DatasetNav import draw_dataset_navbar
from UserInterface.HomeModule import BASE, _rgb
from UserInterface.Theme import theme

# ---- چیدمان (همان قالب صفحهٔ Predictor؛ واحد: پیکسل طراحی ۱۶۷۲×۹۴۱) ----
PY, PH = 236, 656
LW = 920
PANEL_GAP = 24
RX = MX + LW + PANEL_GAP
RW = MX + 1544 - RX

DARK_A, DARK_B = "#2a1250", "#5b2352"
FILL_A, FILL_B = "#8a2a62", "#b4376e"          # نوار K انتخاب‌شده (مثل اسلایدرهای Predictor)
BAR_OFF = ("#a58be0", "#c9b6f2")

CLUSTER_COLORS = [
    "#b79bff", "#f49bc9", "#5fe0d0", "#f2bd6b", "#7fb5f5",
    "#b5de6b", "#d08be0", "#f08a76", "#8f9cf0", "#b0b0c4",
]


# =====================================================
# HELPERS (PIL)
# =====================================================

def _mix(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _cluster_key(v):
    s = str(v)
    return (int(s) if s.isdigit() else 10 ** 6, s)


def _segbar_image(w, h, parts):
    """نوار قطعه‌قطعهٔ اندازهٔ خوشه‌ها (RGBA با لبهٔ گرد)"""
    inner = Image.new("RGB", (w, h), "#3d2a68")
    d = ImageDraw.Draw(inner)
    total = sum(p for p, _ in parts) or 1
    x = 0.0

    for i, (share, color) in enumerate(parts):
        x2 = x + share / total * w
        d.rectangle((int(round(x)), 0, int(round(x2)), h), fill=color)
        if i > 0:
            d.rectangle((int(round(x)) - 1, 0, int(round(x)) + 1, h), fill="#2f1655")
        x = x2

    ss = 4
    mask = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), h * ss // 2, fill=255)
    out = inner.convert("RGBA")
    out.putalpha(mask.resize((w, h), Image.Resampling.LANCZOS))
    return out


def _elbow_image(w, h, grid_col, line_col, area_col, dot_fill, pts, grid_ys,
                 final_i, elbow_i, final_col, elbow_col, lw, r_small, r_big):
    """نمودار Elbow روی زمینهٔ شفاف (RGBA) تا با کارت شیشه‌ای یکی شود"""

    S = 3
    img = Image.new("RGBA", (w * S, h * S), _rgb(line_col) + (0,))
    d = ImageDraw.Draw(img)
    base_y = max(grid_ys)

    for gy in grid_ys:
        d.line([(0, gy * S), (w * S, gy * S)], fill=grid_col, width=max(1, S))

    poly = [(x * S, y * S) for x, y in pts]

    if len(poly) > 1:
        d.polygon(poly + [(poly[-1][0], base_y * S), (poly[0][0], base_y * S)], fill=area_col)
        d.line(poly, fill=line_col, width=int(lw * S), joint="curve")

    if final_i is not None:
        fx, fy = pts[final_i]
        y = fy + r_big
        while y < base_y:
            d.line([(fx * S, y * S), (fx * S, min(y + 4, base_y) * S)], fill=final_col, width=max(1, S))
            y += 8

    for x, y in pts:
        r = r_small
        d.ellipse(((x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S),
                  fill=dot_fill, outline=line_col, width=int(1.8 * S))

    if elbow_i is not None and elbow_i != final_i:
        x, y = pts[elbow_i]
        r = r_big
        d.ellipse(((x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S),
                  fill=dot_fill, outline=elbow_col, width=int(2.2 * S))

    if final_i is not None:
        x, y = pts[final_i]
        r = r_big
        d.ellipse(((x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S), fill=final_col)

    return img.resize((w, h), Image.Resampling.LANCZOS)


def _insight_text(profiles, perc, feats, analysis, desc):

    if len(profiles) < 2:
        return ""

    ratios = {}
    for c, p in profiles.items():
        rs = {}
        for f in feats:
            m = (desc.get(f) or {}).get("mean")
            v = p.get(f)
            if m and v is not None:
                rs[f] = v / m
        ratios[c] = rs

    if not any(ratios.values()):
        return ""

    focus = max(ratios, key=lambda c: sum(ratios[c].values()) / max(1, len(ratios[c])))
    top = sorted(ratios[focus].items(), key=lambda kv: -kv[1])[:2]
    pct = perc.get(focus)

    head = f"Cluster {focus}" + (f" ({pct:.1f}% of samples)" if pct is not None else "")
    text = (f"{head} runs high on "
            + " and ".join(f"{f} (×{r:.1f})" for f, r in top)
            + " versus the dataset mean.")

    for a in analysis:
        v = profiles[focus].get(a)
        m = (desc.get(a) or {}).get("mean")
        if v is not None and m is not None:
            text += f" Mean {a}: {v:.1f} vs {m:.1f} overall."

    return text


# =====================================================
# PAGE
# =====================================================

class EvaluateModule(DatasetsModule):
    """
    صفحهٔ Train & Evaluate برای دیتاست‌های Clustering (K-Means).
    از DatasetsModule ابزارهای رسم، تم و مقیاس را به ارث می‌برد؛
    هدر همان هدر مخصوص صفحه‌های دیتاست است (Evaluate فعال).
    """

    def __init__(self, master, scale=1.0, on_navigate=None, on_train=None):
        # عمداً __init__ والد صدا زده نمی‌شود (Entry و polling کاتالوگ لازم نیست)
        self._light = None
        self.k = scale
        self.W, self.H = int(DESIGN_W * scale), int(DESIGN_H * scale)
        tk.Canvas.__init__(self, master, width=self.W, height=self.H, highlightthickness=0, bd=0)

        self.on_navigate = on_navigate or (lambda page: print("goto", page))
        self.on_train = on_train
        self._refs, self._bgs, self._cache = {}, {}, {}
        self._poll_job = None
        self._info = None

        self.ds = None
        self.meta = {}
        self._ds_sig = None
        self._dirty = True

        theme.subscribe(self._on_theme)

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

    def load_dataset(self, ds):
        """متادیتای همان دیتاست را از فایل JSON می‌خواند (اگر عوض نشده باشد دوباره نمی‌خواند)"""
        try:
            sig = (str(ds.path), ds.path.stat().st_mtime_ns)
        except OSError:
            sig = (str(ds.path), None)

        if self.ds is not None and sig == self._ds_sig:
            return

        self._ds_sig, self.ds = sig, ds

        try:
            meta = json.loads(ds.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = ds.raw or {}

        self.meta = meta if isinstance(meta, dict) else {}
        self._dirty = True

        if self.winfo_ismapped():
            self._build()

    def on_show(self):
        if self._dirty:
            self._build()

    # ---------- داده ----------

    def _data(self):
        meta = self.meta
        cl = meta.get("Clustering") or meta.get("clustering") or {}
        ksel = dict(meta.get("model_evaluation") or {})
        ksel.update(meta.get("k_selection") or {})
        fs = meta.get("feature_selection") or {}
        profiles = cl.get("cluster_profiles") or {}

        feats = list(fs.get("selected_features") or cl.get("input_features") or [])
        analysis = list(fs.get("analysis_only_columns") or [])

        if not feats and profiles:
            first = profiles[sorted(profiles, key=_cluster_key)[0]]
            feats = [f for f in first if f not in analysis]

        explained_variance = (cl.get("ssd") or {}).get("explained_variance_pct")
        if explained_variance is None:
            inertia_by_k = ksel.get("inertia_per_k") or {}
            metrics_by_k = ksel.get("metrics_per_k") or {}
            final_k = ksel.get("final_k", cl.get("n_clusters"))
            total_ss = inertia_by_k.get("1")
            if total_ss is None:
                total_ss = (metrics_by_k.get("1") or {}).get("Inertia")
            within_ss = inertia_by_k.get(str(final_k))
            if within_ss is None:
                within_ss = (metrics_by_k.get(str(final_k)) or {}).get("Inertia")
            if total_ss is not None and within_ss is not None:
                total_ss, within_ss = float(total_ss), float(within_ss)
                if math.isfinite(total_ss) and math.isfinite(within_ss) and total_ss > 0:
                    explained_variance = 100.0 * (1.0 - within_ss / total_ss)

        return SimpleNamespace(
            name=str(meta.get("name") or (self.ds.name if self.ds else "Dataset")),
            mpk=ksel.get("metrics_per_k") or {},
            final_k=ksel.get("final_k", cl.get("n_clusters")),
            elbow_k=ksel.get("elbow_k"),
            reason=str(ksel.get("selection_reason") or ""),
            sil=cl.get("silhouette_score"),
            ev=explained_variance,
            sizes=cl.get("cluster_sizes") or {},
            perc=cl.get("cluster_percentages") or {},
            profiles=profiles,
            desc=(meta.get("dataset_statistics") or {}).get("descriptive_statistics") or {},
            feats=feats,
            analysis=analysis,
            dec=meta.get("feature_decimals") or {},
            scaler=str(cl.get("scaler") or (meta.get("preprocessing") or {}).get("method") or "MinMaxScaler"),
            algo=str(cl.get("algorithm") or "KMeans"),
        )

    def _bg_photo(self):
        key = (theme.name, self.W, self.H)
        if key not in self._bgs:
            path = BASE / "assets" / "page_images" / (
                f"Train & Evaluate_background_{theme.name}.png"
            )
            with Image.open(path) as source:
                background = source.convert("RGB").resize(
                    (self.W, self.H), Image.Resampling.LANCZOS
                )
            self._bgs[key] = ImageTk.PhotoImage(background, master=self)
        return self._bgs[key]

    def _track(self):
        return ("#3d2a68", "#6a4f9a") if theme.name == "dark" else ("#ebe4f6", "#cbbde4")

    # =====================================================
    # رسم
    # =====================================================

    def _build(self):
        self._dirty = False
        self.delete("all")
        self.config(cursor="")
        self.create_image(0, 0, image=self._bg_photo(), anchor="nw")
        draw_dataset_navbar(self, active="evaluate")
        if self.ds is not None:
            self._draw_page()

    def _draw_back_button(self):
        S = self.s
        dark = theme.name == "dark"
        color = "#f1d5fa" if dark else "#ffffff"
        bg = "#50336f" if dark else "#713b91"
        edge = "#805ba1" if dark else "#713b91"
        w, h = 90, 30
        x, y = MX + 4, 103
        tag = "evaluate_back"
        self.create_image(S(x), S(y), image=self._shape(w, h, 10, bg, border=(edge, 255)), anchor="nw", tags=(tag,))
        cy = y + h / 2
        self.create_line(S(x+29), S(cy), S(x+14), S(cy), fill=color, width=max(1,S(2)), capstyle="round", tags=(tag,))
        self.create_line(S(x+21), S(cy-7), S(x+14), S(cy), S(x+21), S(cy+7), fill=color, width=max(1,S(2)), capstyle="round", joinstyle="round", tags=(tag,))
        self.text(x+37, cy, "Back", 16, color, True, tags=(tag,))
        self.hover(tag)
        self.tag_bind(tag, "<Button-1>", lambda e: self.on_navigate("predictor"))

    def _draw_page(self):

        c, S = theme.c, self.s
        d = self._data()

        # ----- مسیر + عنوان -----
        crumb_y = 118
        x = MX + 108
        self._draw_back_button()

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

        self.text(x, crumb_y, "/  Evaluate", 20, c("body"))

        # The heading is included in the background image.

        self._draw_stats(d)

        # ----- پنل‌ها -----
        left, lp = self._glass(LW, PH, 36, c("card_cols")[0], c("card_end"), c("card_alpha"),
                               c("shadow_a"), c("shadow_rgb"), c("card_border"))
        right, rp = self._glass(RW, PH, 36, DARK_A, DARK_B, 255,
                                c("shadow_a"), c("shadow_rgb"), None)
        self.create_image(S(MX) - lp, S(PY) - lp, image=left, anchor="nw")
        self.create_image(S(RX) - rp, S(PY) - rp, image=right, anchor="nw")

        self._draw_left(d)
        self._draw_right(d)

    # ---- کارت‌های آماری بالا-راست ----
    def _draw_stats(self, d):

        c, S = theme.c, self.s
        w, h, gap = 268, 92, 14
        x0 = MX + 1544 - 3 * w - 2 * gap
        y = 112

        stats = [
            ("clusters", str(d.final_k) if d.final_k is not None else "—", "Clusters (K)"),
            ("columns", f"{float(d.sil):.3f}" if d.sil is not None else "—", "Silhouette"),
            ("features", f"{float(d.ev):.1f}%" if d.ev is not None else "—", "Variance explained"),
        ]

        for i, (kind, value, label) in enumerate(stats):
            x = x0 + i * (w + gap)
            img, pad = self._glass(w, h, 28, c("card_cols")[0], c("card_end"), c("card_alpha"),
                                   c("shadow_a"), c("shadow_rgb"), c("card_border"), 10, 6)
            self.create_image(S(x) - pad, S(y) - pad, image=img, anchor="nw")
            self.create_image(S(x + 20), S(y + (h - 50) / 2), image=self._info_icon(kind, 50), anchor="nw")
            self.text(x + 86, y + 36, value, 30, c("title"), True)
            self.text(x + 86, y + 68, label, 17, c("body"))

    # =====================================================
    # پنل چپ: انتخاب K
    # =====================================================

    def _draw_left(self, d):

        c, S = theme.c, self.s
        dark = theme.name == "dark"
        x0, inner = MX + 48, LW - 96
        right = x0 + inner

        self.text(x0, PY + 52, "Choosing K", 30, c("title"), True)

        if d.final_k is not None:
            chip = f"Final K = {d.final_k}"
            cw = self.text_width(chip, 16, True) + 36
            self.create_image(S(right - cw), S(PY + 52 - 18),
                              image=self._shape(cw, 36, 18, c("accent"), alpha=40, border=(c("accent"), 170)),
                              anchor="nw")
            self.text(right - cw / 2, PY + 52, chip, 16, c("accent"), True, anchor="center")

        self.text(x0, PY + 90, f"{d.algo}  ·  {d.scaler}  ·  {len(d.feats)} features", 17, c("body"))

        for i, line in enumerate(self._wrap(d.reason, 16, False, inner, 2) if d.reason else []):
            self.text(x0, PY + 118 + i * 20, line, 16, c("title"))

        ks = sorted(int(v) for v in d.mpk if str(v).isdigit())

        if len(ks) < 2:
            self.text(x0, PY + 200, "No K-selection data in this dataset's metadata.", 19, c("body"))
            return

        # ----- نمودار Elbow -----
        inertia = [(d.mpk[str(v)] or {}).get("Inertia") or 0 for v in ks]

        chart_y, chart_h = PY + 152, 118
        cx0, cx1 = x0 + 40, right
        wpx, hpx = S(cx1 - cx0), S(chart_h)

        maxv = max(inertia) or 1
        mag = 10 ** math.floor(math.log10(maxv))
        step = mag
        if maxv / step < 3:
            step = mag / 2
        elif maxv / step > 6:
            step = mag * 2
        n_steps = max(1, math.ceil(maxv / step))
        ymax = n_steps * step

        mtop, mbot, inset = S(12), S(6), S(14)
        plot_h = hpx - mtop - mbot

        def yv(v):
            return mtop + (1 - v / ymax) * plot_h

        def xv(kv):
            return inset + (kv - ks[0]) / (ks[-1] - ks[0]) * (wpx - 2 * inset)

        pts = [(xv(kv), yv(v)) for kv, v in zip(ks, inertia)]
        grid_ys = [yv(g * step) for g in range(n_steps + 1)]
        final_i = ks.index(d.final_k) if d.final_k in ks else None
        elbow_i = None

        line_col = "#b79bff" if dark else "#6a3fa5"
        final_col = "#f49bc9" if dark else "#a3305f"
        dot_fill = "#2a1a47" if dark else "#ffffff"

        key = ("evl_elbow", self.k, theme.name, wpx, hpx, tuple(round(p[0]) for p in pts),
               tuple(round(p[1]) for p in pts), final_i, elbow_i)

        if key not in self._cache:
            self._cache[key] = ImageTk.PhotoImage(_elbow_image(
                wpx, hpx,
                _rgb(c("body")) + (55,), line_col, _rgb(line_col) + (42,), dot_fill,
                pts, grid_ys, final_i, elbow_i, final_col, c("accent"),
                max(2.2, 2.6 * self.k), 3.6 * self.k, 6.0 * self.k
            ))

        self.create_image(S(cx0), S(chart_y), image=self._cache[key], anchor="nw")

        for g in range(n_steps + 1):
            self.text(cx0 - 10, chart_y + yv(g * step) / self.k, f"{g * step:g}", 14, c("body"), anchor="e")

        for kv in ks:
            on = kv == d.final_k
            self.text(cx0 + xv(kv) / self.k, chart_y + chart_h + 16, str(kv), 14,
                      final_col if on else c("body"), on, anchor="center")

        self.text(cx1, chart_y + 8, "Inertia vs K", 14, c("body"), anchor="e")

        if final_i is not None:
            px_, py_ = pts[final_i]
            label = f"K = {d.final_k}"
            self.text(cx0 + px_ / self.k + 12, chart_y + py_ / self.k - 14, label, 14, final_col, True)

        # ----- جدول اعتبارسنجی -----
        head_y = chart_y + chart_h + 46
        self.text(x0, head_y, "Validation by K", 21, c("title"), True)

        hint = "best: highest Silhouette & CH, lowest DB"
        if self.text_width("Validation by K", 21, True) + self.text_width(hint, 15) + 30 < inner:
            self.text(right, head_y + 2, hint, 15, c("body"), anchor="e")

        db_x, ch_x, sil_x = right, right - 96, right - 196
        bar_x0, bar_x1 = x0 + 188, sil_x - 82
        col_y = head_y + 28

        self.text(x0, col_y, "K", 15, c("title"), True)
        self.text(sil_x, col_y, "Silhouette", 15, c("title"), True, anchor="e")
        self.text(ch_x, col_y, "CH", 15, c("title"), True, anchor="e")
        self.text(db_x, col_y, "DB", 15, c("title"), True, anchor="e")

        rows = [kv for kv in ks if (d.mpk[str(kv)] or {}).get("Silhouette") is not None]
        has_btn = callable(self.on_train)

        rows_top = col_y + 18
        rows_bottom = PY + PH - (112 if has_btn else 44)
        row_h = _clamp((rows_bottom - rows_top) / max(1, len(rows)), 20, 30)

        sils = [d.mpk[str(kv)]["Silhouette"] for kv in rows]
        max_sil = max([s for s in sils if s and s > 0] or [1])
        track_fill, track_edge = self._track()

        for i, kv in enumerate(rows):

            md = d.mpk[str(kv)]
            ry = rows_top + i * row_h
            cy = ry + row_h / 2
            sel = kv == d.final_k

            if sel:
                self.create_image(S(x0 - 12), S(ry + 2),
                                  image=self._shape(inner + 24, row_h - 4, 10, c("accent"), alpha=34),
                                  anchor="nw")

            self.text(x0, cy, f"K = {kv}", 18, c("title"), sel)

            if sel:
                bt = "Selected"
                bw = self.text_width(bt, 13, True) + 24
                self.create_image(S(x0 + 84), S(cy - 11),
                                  image=self._shape(bw, 22, 11, c("accent"), alpha=50, border=(c("accent"), 160)),
                                  anchor="nw")
                self.text(x0 + 84 + bw / 2, cy, bt, 13, c("accent"), True, anchor="center")

            sil = md.get("Silhouette") or 0
            bar_w = max(40, bar_x1 - bar_x0)

            self.create_image(S(bar_x0), S(cy), image=self._shape(bar_w, 9, 4.5, track_fill, border=(track_edge, 255)),
                              anchor="w")

            fw = max(9, bar_w * max(0, sil) / max_sil)
            c1, c2 = (FILL_A, FILL_B) if sel else BAR_OFF
            self.create_image(S(bar_x0), S(cy), image=self._shape(fw, 9, 4.5, c1, c2), anchor="w")

            tc = c("title") if sel else c("body")
            self.text(sil_x, cy, f"{sil:.3f}", 18, tc, True, anchor="e")
            self.text(ch_x, cy, f"{md.get('Calinski-Harabasz', 0):.1f}", 18, tc, True, anchor="e")
            self.text(db_x, cy, f"{md.get('Davies-Bouldin', 0):.2f}", 18, tc, True, anchor="e")

        # ----- دکمهٔ Train (فقط اگر on_train داده شده باشد) -----
        if has_btn:
            bh, pw = 58, 230
            by = PY + PH - 92
            img, pad = self._glass(pw, bh, 29, "#b4376e", "#5a1f4b", 255, 110, (163, 48, 95), None, 10, 8)
            self.create_image(S(x0) - pad, S(by) - pad, image=img, anchor="nw", tags=("btn_train",))
            self.text(x0 + pw / 2, by + bh / 2, "Train K-Means", 22, "white", True, anchor="center",
                      tags=("btn_train",))
            self.hover("btn_train")
            self.tag_bind("btn_train", "<Button-1>", lambda e: self.on_train())

    # =====================================================
    # پنل راست: پروفایل خوشه‌ها
    # =====================================================

    def _draw_right(self, d):

        S = self.s
        left, right = RX + 48, RX + RW - 48
        width = right - left

        self.text(left, PY + 52, "Cluster profiles", 26, "#ffffff", True)
        self.text(left, PY + 84, "Mean per cluster  ·  × = ratio to dataset mean", 15, "#d9c8ee")

        ids = sorted(d.profiles.keys(), key=_cluster_key)

        if not ids:
            self.text(left, PY + 150, "No cluster profiles in this dataset's metadata.", 18, "#f0e7fa")
            return

        color = {cid: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, cid in enumerate(ids)}

        # ----- اندازهٔ خوشه‌ها -----
        self.text(left, PY + 122, "Cluster sizes", 18, "#f0e7fa", True)

        parts = [(d.sizes.get(cid, 0), color[cid]) for cid in ids]

        if sum(p for p, _ in parts) > 0:
            bw_px, bh_px = S(width), S(14)
            key = ("evl_seg", self.k, bw_px, tuple(parts))
            if key not in self._cache:
                self._cache[key] = ImageTk.PhotoImage(_segbar_image(bw_px, bh_px, parts))
            self.create_image(S(left), S(PY + 140), image=self._cache[key], anchor="nw")

        ly, lx = PY + 172, left

        for cid in ids:
            n = d.sizes.get(cid, 0)
            p = d.perc.get(cid)
            txt = f"Cluster {cid}  {n:,}" + (f" · {p:.1f}%" if p is not None else "")
            item_w = 9 + 8 + self.text_width(txt, 16) + 24

            if lx + item_w > right and lx > left:
                lx, ly = left, ly + 22

            self.create_image(S(lx), S(ly - 4.5), image=self._shape(9, 9, 4.5, color[cid]), anchor="nw")
            self.text(lx + 17, ly, txt, 16, "#f0e7fa")
            lx += item_w

        legend_end = ly + 22

        # ----- متن Insight (اول حساب می‌شود تا فضای باقی‌مانده معلوم شود) -----
        insight = _insight_text(d.profiles, d.perc, d.feats, d.analysis, d.desc)
        lines = self._wrap(insight, 16, False, width - 40, 4) if insight else []
        ins_h = (14 + 22 + len(lines) * 22 + 14) if lines else 0
        ins_y = PY + PH - 40 - ins_h

        # ----- جدول پروفایل -----
        first = d.profiles[ids[0]]
        rows = [(f, False) for f in d.feats if f in first] + [(a, True) for a in d.analysis if a in first]

        label_w = max([self.text_width(f, 17, True) for f, _ in rows] or [60]) + 20
        gx0 = left + label_w
        n = len(ids)
        g = 10
        tile_w = (right - gx0 - (n - 1) * g) / n

        head_y = legend_end + 16

        for i, cid in enumerate(ids):
            cx = gx0 + i * (tile_w + g) + tile_w / 2
            name = f"Cluster {cid}" if tile_w > 90 else str(cid)
            tw = self.text_width(name, 16, True)
            dx = cx - (tw + 17) / 2
            self.create_image(S(dx), S(head_y - 4.5), image=self._shape(9, 9, 4.5, color[cid]), anchor="nw")
            self.text(dx + 17, head_y, name, 16, "#f0e7fa", True)

        tiles_top = head_y + 20
        gap_r = 8 if len(ids) <= 3 else 5
        extra = (34 if len(ids) <= 3 else 28) if any(a for _, a in rows) else 0
        rows_bottom = (ins_y - 14) if lines else (PY + PH - 40)
        avail = rows_bottom - tiles_top - extra - gap_r * (len(rows) - 1)
        tile_h = _clamp(avail / max(1, len(rows)), 20, 46)

        tile_lo, tile_hi = "#3b2766", "#7c4fe0"
        yy = tiles_top
        seen_analysis = False

        for f, is_analysis in rows:

            if is_analysis and not seen_analysis:
                seen_analysis = True
                self.create_line(S(left), S(yy + 8), S(right), S(yy + 8), fill="#6a4a8c", width=max(1, S(1)))
                self.text(left, yy + 24, "Analysis variable (not used for clustering)", 14, "#d9c8ee")
                yy += extra

            self.text(left, yy + tile_h / 2, f, 17, "#f0e7fa", True)

            mean = (d.desc.get(f) or {}).get("mean")
            dec = max(1, int(d.dec.get(f, 1)))

            for i, cid in enumerate(ids):

                v = d.profiles[cid].get(f)
                if v is None:
                    continue

                ratio = (v / mean) if mean else None
                t = _clamp((ratio - 0.7) / 1.6, 0, 1) if ratio is not None else 0.3
                tx = gx0 + i * (tile_w + g)

                self.create_image(S(tx), S(yy), image=self._shape(tile_w, tile_h, 11, _mix(tile_lo, tile_hi, t)),
                                  anchor="nw")

                if tile_w >= 110 and ratio is not None:
                    self.text(tx + 14, yy + tile_h / 2, f"{v:.{dec}f}", 18, "white", True)
                    self.text(tx + tile_w - 14, yy + tile_h / 2, f"×{ratio:.1f}", 15, "#f0e7fa", anchor="e")
                else:
                    self.text(tx + tile_w / 2, yy + tile_h / 2, f"{v:.{dec}f}", 17, "white", True, anchor="center")

            yy += tile_h + gap_r

        # ----- جعبهٔ Insight -----
        if lines:
            self.create_image(S(left), S(ins_y),
                              image=self._shape(width, ins_h, 20, "#ffffff", alpha=34, border=("#ffffff", 40)),
                              anchor="nw")
            self.text(left + 20, ins_y + 22, "INSIGHT", 15, "#f3b3d3", True)

            for i, line in enumerate(lines):
                self.text(left + 20, ins_y + 22 + 24 + i * 22 - 2, line, 16, "#eadcf6")

