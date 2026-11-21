import json
import math
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageTk

from UserInterface.Datasets import (
    DatasetsModule, BG_LIGHT, BG_DARK, DESIGN_W, DESIGN_H, MX, CONTENT_W, BTN_DARK, BTN_LIGHT,
)
from UserInterface.HomeModule import BASE, _rgb
from UserInterface.Theme import theme, FONT

# پس‌زمینه‌های مخصوص صفحهٔ Predictor (عنوان «Predict the Cluster» داخل خود عکس است)
PRED_BG = {
    "light": BASE / "assets" / "page_images" / "predict_background_light.png",
    "dark": BASE / "assets" / "page_images" / "predict_background_dark.png",
}

# ---- چیدمان (واحد: پیکسل طراحی ۱۶۷۲×۹۴۱) ----
PY, PH = 236, 656            # بالا و ارتفاع پنل‌ها
LW = 920                     # عرض پنل چپ
PANEL_GAP = 24
RX = MX + LW + PANEL_GAP     # شروع پنل راست
RW = MX + CONTENT_W - RX     # عرض پنل راست
SLIDER_COLS = 2              # تعداد ستون اسلایدرها (۱ = همه زیر هم)
KNOB_D = 24                  # قطر دستگیرهٔ اسلایدر

DARK_A, DARK_B = "#2a1250", "#5b2352"
FILL_A, FILL_B = "#8a2a62", "#b4376e"          # رنگ پر شدن اسلایدر
BAR_ON = ("#f6a9d0", "#ee86b8")                # نوار خوشهٔ پیش‌بینی‌شده
BAR_OFF = ("#a58be0", "#c9b6f2")               # نوار سایر خوشه‌ها
BAR_TRACK = "#5d4486"


class PredictorModule(DatasetsModule):
    """صفحهٔ Predictor عمومی برای همهٔ دیتاست‌های Clustering (همه یک قاعده و یک ظاهر)."""

    def __init__(self, master, scale=1.0, on_navigate=None, ds=None):
        # عمداً __init__ والد صدا زده نمی‌شود (Entry و polling کاتالوگ لازم نیست)
        self._light = None
        self.k = scale
        self.W, self.H = int(DESIGN_W * scale), int(DESIGN_H * scale)
        tk.Canvas.__init__(self, master, width=self.W, height=self.H, highlightthickness=0, bd=0)

        self.on_navigate = on_navigate or (lambda page: print("goto", page))
        self._refs, self._bgs, self._cache = {}, {}, {}
        self._poll_job = None
        self._info = None

        self.ds = None
        self.meta = {}
        self.bundle = None
        self.error = None
        self.vals = {}
        self.result = None
        self.notice = None
        self._sl = {}
        self._counter = None
        self._dirty = True

        # کارهای سنگین (خواندن عکس‌ها، import سنگین sklearn، خواندن مدل‌ها) در پس‌زمینه انجام می‌شوند
        self._pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="predictor")
        self._bg_jobs, self._bundle_jobs = {}, {}
        self._pool.submit(self._warm_imports)
        for name in [theme.name] + [n for n in ("light", "dark") if n != theme.name]:
            self._bg_jobs[name] = self._pool.submit(self._load_pred_bg, name)

        theme.subscribe(self._on_theme)
        if ds is not None:
            self.load_dataset(ds)

    # =====================================================
    # سرعت: بارگذاری پیش‌دستی
    # =====================================================
    @staticmethod
    def _warm_imports():
        import sklearn.cluster          # noqa: F401  (اولین import چند ثانیه طول می‌کشد)
        import sklearn.preprocessing    # noqa: F401

    def preload(self, datasets):
        """مدل‌های Clustering را در پس‌زمینه می‌خواند تا کلیک روی Open predictor فوری باشد."""
        for ds in datasets:
            try:
                key = self._sig(ds)
                if key not in self._bundle_jobs:
                    self._bundle_jobs[key] = self._pool.submit(self._read_bundle, ds)
            except OSError:
                pass

    def prewarm(self):
        """تصویرهای ثابت صفحه را در زمان‌های خلوت و مرحله‌به‌مرحله می‌سازد."""
        steps = [self._bg_photo, self._panel_images, self._knob_img,
                 lambda: self._ring_img(0.5, 124, 14)]

        def run(i=0):
            if i >= len(steps) or not self.winfo_exists():
                return
            try:
                steps[i]()
            except Exception as error:
                print("[Predictor] prewarm skipped:", error)
            self.after(30, run, i + 1)

        self.after(200, run)

    def destroy(self):
        try:
            theme.unsubscribe(self._on_theme)
        except Exception:
            pass
        self._pool.shutdown(wait=False)
        tk.Canvas.destroy(self)

    def _on_theme(self):
        self._dirty = True
        if self.winfo_ismapped():
            self._build()

    # =====================================================
    # پس‌زمینه
    # =====================================================
    def _load_pred_bg(self, name):
        path = PRED_BG[name]
        if path.exists():
            img = Image.open(path).convert("RGB")
        else:                                           # اگر فایل نبود، پس‌زمینهٔ Datasets
            fallback = BG_DARK if name == "dark" and BG_DARK.exists() else BG_LIGHT
            img = self._load_background(fallback).convert("RGB")
        return img.resize((self.W, self.H), Image.LANCZOS)

    def _bg_photo(self):
        name = theme.name
        if name not in self._bgs:
            job = self._bg_jobs.pop(name, None)
            img = job.result() if job is not None else self._load_pred_bg(name)
            self._bgs[name] = ImageTk.PhotoImage(img)
        return self._bgs[name]

    # =====================================================
    # بارگذاری دیتاست و مدل
    # =====================================================
    @staticmethod
    def _sig(ds):
        return (str(ds.path), ds.path.stat().st_mtime_ns)

    @staticmethod
    def _read_bundle(ds):
        """بدون استفاده از Tk (قابل اجرا در thread): متادیتا + مدل + scaler + ستون‌ها."""
        try:
            meta = json.loads(ds.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = ds.raw or {}
        root = ds.path.parent.parent                    # <Task>/Metadata/x.json → <Task>
        files = meta.get("output_files") or {}

        def path(key):
            rel = files.get(key)
            if not rel:
                raise FileNotFoundError(f"'{key}' is missing in metadata output_files")
            p = root / rel
            if not p.exists():
                raise FileNotFoundError(p.name)
            return p

        model = joblib.load(path("trained_model"))
        scaler = joblib.load(path("scaler"))
        features = list(joblib.load(path("feature_columns")))
        labels = meta.get("feature_labels") or {}
        bundle = {
            "model": model, "scaler": scaler, "features": features,
            "k": PredictorModule._find_k(model, meta),
            "range": PredictorModule._find_ranges(scaler, features, meta),
            "labels": {f: labels.get(f) or PredictorModule._pretty(f) for f in features},
        }
        bundle["quality"] = PredictorModule._compute_quality(model, scaler, features, root, files)
        return meta, bundle

    @staticmethod
    def _compute_quality(model, scaler, features, root, files):
        """SSD کل، SSD هر خوشه و درصد واریانس توضیح‌داده‌شده را از داده‌های پردازش‌شده حساب می‌کند.
        (در همان فضایی که KMeans آموزش دیده؛ مثلاً بعد از PCA.) اگر فایل نبود None برمی‌گرداند."""
        try:
            rel = files.get("processed_dataset") or files.get("original_dataset")
            if not rel:
                return None
            df = pd.read_csv(root / rel, usecols=features)[features]
            data = pd.DataFrame(scaler.transform(df), columns=features)
            steps = getattr(model, "steps", None)
            est = steps[-1][1] if steps else model
            space = data
            if steps and len(steps) > 1:
                space = data
                for _, step in steps[:-1]:
                    space = step.transform(space)
            space = np.asarray(space, dtype=float)
            names = getattr(est, "feature_names_in_", None)
            labels = est.predict(pd.DataFrame(space, columns=list(names)) if names is not None else space)
            centers = est.cluster_centers_
            per = {str(i): float(((space[labels == i] - centers[i]) ** 2).sum()) for i in range(len(centers))}
            total = sum(per.values())
            tss = float(((space - space.mean(axis=0)) ** 2).sum())
            return {"total": total, "per": per, "explained": (tss - total) / tss * 100 if tss else None}
        except Exception as error:
            print("[Predictor] SSD could not be computed:", error)
            return None

    @staticmethod
    def _final_estimator(model):
        """اگر مدل Pipeline باشد، آخرین مرحله (KMeans) را برمی‌گرداند."""
        steps = getattr(model, "steps", None)
        return steps[-1][1] if steps else model

    @staticmethod
    def _find_k(model, meta):
        est = PredictorModule._final_estimator(model)
        k = getattr(est, "n_clusters", None)
        if not k and hasattr(est, "cluster_centers_"):
            k = len(est.cluster_centers_)
        if not k:
            clu = meta.get("Clustering") or meta.get("clustering") or {}
            k = clu.get("n_clusters") or (meta.get("k_selection") or {}).get("final_k")
        return int(k)

    @staticmethod
    def _find_ranges(scaler, features, meta):
        """بازهٔ هر ویژگی: از scaler (حتی داخل Pipeline) و در غیر این صورت از آمار متادیتا."""
        parts = [step for _, step in getattr(scaler, "steps", [])] or [scaler]
        lo = hi = None
        for part in parts:
            if hasattr(part, "data_min_") and hasattr(part, "data_max_"):
                lo, hi = part.data_min_, part.data_max_
                break
        stats = (meta.get("dataset_statistics") or {}).get("descriptive_statistics") or {}
        ranges = {}
        for i, f in enumerate(features):
            if lo is not None and i < len(lo):
                ranges[f] = (float(lo[i]), float(hi[i]))
            else:
                st = stats.get(f) or {}
                a, b = st.get("min"), st.get("max")
                if a is None or b is None:
                    a, b = 0.0, 1.0
                ranges[f] = (float(a), float(b))
        return ranges

    def load_dataset(self, ds):
        self.ds, self.result, self.notice, self.error, self.bundle = ds, None, None, None, None
        try:
            key = self._sig(ds)
            job = self._bundle_jobs.get(key)
            if job is None:
                job = self._bundle_jobs[key] = self._pool.submit(self._read_bundle, ds)
            try:
                self.meta, self.bundle = job.result()
            except Exception:
                self._bundle_jobs.pop(key, None)
                raise
            # هنگام باز شدن صفحه هیچ مقداری انتخاب نشده است
            self.vals = {f: None for f in self.bundle["features"]}
        except Exception as error:                      # فایل مدل نبود / خراب بود
            self.error = f"Could not load the model files: {error}"
        self._dirty = True
        if self.winfo_ismapped():
            self._build()

    def on_show(self):
        if self._dirty:                                 # فقط یک بار ساخته می‌شود
            self._build()

    @staticmethod
    def _pretty(name):
        text = str(name).replace("_", " ")
        return text.upper() if text.islower() and len(text) <= 4 else text.title()

    @staticmethod
    def _fmt(v, lo, hi):
        span = abs(hi - lo)
        if span >= 100:
            return f"{v:,.0f}"
        return f"{v:.1f}" if span >= 10 else f"{v:.2f}"

    def _clu(self):
        return self.meta.get("Clustering") or self.meta.get("clustering") or {}

    # =====================================================
    # پیش‌بینی
    # =====================================================
    def _predict(self):
        if not self.bundle:
            return
        b = self.bundle
        missing = [f for f in b["features"] if self.vals.get(f) is None]
        if missing:
            self.result = None
            self.notice = f"Set a value for every feature first ({len(missing)} left)."
        else:
            X = pd.DataFrame([[self.vals[f] for f in b["features"]]], columns=b["features"])
            Xs = pd.DataFrame(b["scaler"].transform(X), columns=b["features"])
            label = int(b["model"].predict(Xs)[0])
            dist = b["model"].transform(Xs)[0]
            w = 1.0 / np.maximum(dist, 1e-9)            # نزدیک‌تر = سهم بیشتر
            self.result = {"label": label, "shares": (w / w.sum()).tolist()}
            self.notice = None
        self._redraw_result()

    def _reset(self):
        if not self.bundle:
            return
        c = theme.c
        for f, s in self._sl.items():
            self.vals[f] = None
            self.itemconfigure(s["fill"], state="hidden")
            self.itemconfigure(s["knob"], state="hidden")
            self.itemconfigure(s["val"], text="—", fill=c("body"))
        self.result, self.notice = None, None
        self._update_counter()
        self._redraw_result()

    def _redraw_result(self):
        self.delete("res")
        self._draw_result()

    def _update_counter(self):
        if self._counter is None or not self.bundle:
            return
        done = sum(1 for v in self.vals.values() if v is not None)
        try:
            self.itemconfigure(self._counter, text=f"{done} of {len(self.vals)} values set")
        except tk.TclError:
            pass

    # =====================================================
    # تصویرهای PIL (همه با ابَرنمونه‌گیری = لبهٔ کاملاً تیز)
    # =====================================================
    def _panel_images(self):
        key = ("pred_panels", self.k, theme.name)
        if key not in self._cache:
            c = theme.c
            left = self._glass(LW, PH, 36, c("card_cols")[0], c("card_end"), c("card_alpha"),
                               c("shadow_a"), c("shadow_rgb"), c("card_border"))
            right = self._glass(RW, PH, 36, DARK_A, DARK_B, 255, c("shadow_a"), c("shadow_rgb"), None)
            self._cache[key] = (left, right)
        return self._cache[key]

    def _pill_pil(self, w, h, c1, c2):
        """قرص گرادیانی با لبهٔ نرم (PIL، برای برش و نمایش بخشی از آن)."""
        key = ("pillpil", self.k, w, h, c1, c2)
        if key in self._cache:
            return self._cache[key]
        W, H, ss = max(2, self.s(w)), max(2, self.s(h)), 4
        a, b = _rgb(c1), _rgb(c2)
        arr = np.zeros((H, W, 4), dtype=np.uint8)
        t = np.linspace(0, 1, W)
        for i in range(3):
            arr[:, :, i] = (a[i] + (b[i] - a[i]) * t).astype(np.uint8)[None, :]
        arr[:, :, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        mask = Image.new("L", (W * ss, H * ss), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, W * ss - 1, H * ss - 1), H * ss // 2, fill=255)
        img.putalpha(mask.resize((W, H), Image.LANCZOS))
        self._cache[key] = img
        return img

    def _knob_img(self):
        """دستهٔ اسلایدر: دایرهٔ سفید + هستهٔ مارون + سایهٔ نرم."""
        key = ("knob", self.k)
        if key in self._cache:
            return self._cache[key]
        S, pad, ss = self.s(KNOB_D), self.s(6), 8
        size = (S + 2 * pad) * ss
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
        o, dy = pad * ss, self.s(2) * ss
        ImageDraw.Draw(sh).ellipse((o, o + dy, o + S * ss, o + S * ss + dy), fill=(90, 31, 75, 120))
        img = Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(self.s(2.5) * ss)))
        d = ImageDraw.Draw(img)
        d.ellipse((o, o, o + S * ss, o + S * ss), fill=(255, 255, 255, 255))
        ring = self.s(3.2) * ss
        d.ellipse((o + ring, o + ring, o + S * ss - ring, o + S * ss - ring), fill=_rgb("#a3305f") + (255,))
        photo = ImageTk.PhotoImage(img.resize((S + 2 * pad, S + 2 * pad), Image.LANCZOS))
        self._cache[key] = photo
        return photo

    def _hit(self, w, h):
        """تصویر نامرئی برای بزرگ کردن ناحیهٔ کلیک اسلایدر."""
        key = ("hit", self.k, w, h)
        if key not in self._cache:
            self._cache[key] = ImageTk.PhotoImage(
                Image.new("RGBA", (max(1, self.s(w)), max(1, self.s(h))), (0, 0, 0, 1)))
        return self._cache[key]

    def _ring_img(self, share, d, stroke):
        """حلقهٔ درصد با سرِ گرد و رنگ گرادیانی، کاملاً تیز."""
        key = ("ring", self.k, round(share * 1000), d, stroke)
        if key in self._cache:
            return self._cache[key]
        S, ss = self.s(d), 4
        big, bw = S * ss, max(4, self.s(stroke)) * ss
        bbox = (bw / 2, bw / 2, big - bw / 2, big - bw / 2)

        base = Image.new("RGBA", (big, big), (0, 0, 0, 0))
        ImageDraw.Draw(base).ellipse(bbox, outline=_rgb(BAR_TRACK) + (255,), width=bw)

        mask = Image.new("L", (big, big), 0)
        md = ImageDraw.Draw(mask)
        if share >= 0.999:
            md.ellipse(bbox, outline=255, width=bw)
        elif share > 0:
            end = -90 + 360 * share
            md.arc(bbox, start=-90, end=end, fill=255, width=bw)
            rad, r, c0 = (big - bw) / 2, bw / 2, big / 2
            for ang in (-90, end):
                a = math.radians(ang)
                px, py = c0 + rad * math.cos(a), c0 + rad * math.sin(a)
                md.ellipse((px - r, py - r, px + r, py + r), fill=255)

        yy, xx = np.mgrid[0:big, 0:big]
        t = ((xx + yy) / (2 * big)).astype(np.float32)
        c1, c2 = _rgb("#d3c0f7"), _rgb("#f49bc9")
        arr = np.zeros((big, big, 4), dtype=np.uint8)
        for i in range(3):
            arr[:, :, i] = (c1[i] + (c2[i] - c1[i]) * t).astype(np.uint8)
        arr[:, :, 3] = 255
        grad = Image.fromarray(arr, "RGBA")
        grad.putalpha(mask)
        base.alpha_composite(grad)
        photo = ImageTk.PhotoImage(base.resize((S, S), Image.LANCZOS))
        self._cache[key] = photo
        return photo

    def _ui(self):
        if theme.name == "dark":
            return {"track": "#3d2a68", "track_border": "#6a4f9a"}
        return {"track": "#ebe4f6", "track_border": "#cbbde4"}

    # =====================================================
    # رسم
    # =====================================================
    def _build(self):
        self._dirty = False
        self.delete("all")
        self.config(cursor="")
        self.create_image(0, 0, image=self._bg_photo(), anchor="nw")
        self._navbar()
        if self.ds is not None:
            self._draw_page()

    def _draw_page(self):
        c, pal, S = theme.c, self._pal(), self.s

        # مسیر (عنوان «Predict the Cluster» داخل پس‌زمینه است)
        crumb_y = 118
        self.text(MX + 4, crumb_y, "Datasets", 20, c("accent"), True, tags=("crumb",))
        self.text(MX + 4 + self.text_width("Datasets", 20, True) + 10, crumb_y, f"/  {self.ds.name}", 20, c("body"))
        self.hover("crumb")
        self.tag_bind("crumb", "<Button-1>", lambda e: self.on_navigate("datasets"))

        (left, lp), (right, rp) = self._panel_images()
        self.create_image(S(MX) - lp, S(PY) - lp, image=left, anchor="nw")
        self.create_image(S(RX) - rp, S(PY) - rp, image=right, anchor="nw")

        if self.error or not self.bundle:
            self.text(MX + 48, PY + 70, "Predictor unavailable", 30, c("title"), True)
            self.text(MX + 48, PY + 120, self._fit(self.error or "No model loaded.", 20, False, LW - 96),
                      20, c("body"))
            return
        self._draw_sliders(MX, LW)
        self._draw_algorithm(MX, LW, pal)
        self._draw_buttons(MX, pal)
        self._draw_result()

    # ---- پنل چپ ----
    def _draw_sliders(self, px, pw):
        c, S = theme.c, self.s
        ui = self._ui()
        b = self.bundle
        feats = b["features"]
        self._sl = {}
        x0, inner = px + 48, pw - 96

        self.text(x0, PY + 52, "Feature values", 28, c("title"), True)
        self._counter = self.create_text(S(px + pw - 48), S(PY + 52), text="", anchor="e", fill=c("body"),
                                         font=(FONT, -S(17)))
        self._update_counter()

        cols = max(1, SLIDER_COLS)
        rows = math.ceil(len(feats) / cols)
        top, bottom = PY + 108, PY + PH - 224
        pitch = min(86, (bottom - top) / max(rows, 1))
        off = min(36, pitch * 0.58)
        gap = 40
        colw = (inner - gap * (cols - 1)) / cols
        full_fill = self._pill_pil(colw, 12, FILL_A, FILL_B)
        track_img = self._shape(colw, 12, 6, ui["track"], border=(ui["track_border"], 255))
        hit_img, knob_img = self._hit(colw, 34), self._knob_img()

        for i, f in enumerate(feats):
            row, col = divmod(i, cols)
            x, y = x0 + col * (colw + gap), top + row * pitch
            ty = y + off
            lo, hi = b["range"][f]
            v = self.vals.get(f)
            t = 0.0 if v is None or hi <= lo else (v - lo) / (hi - lo)
            tag = f"sl_{i}"
            state = "hidden" if v is None else "normal"

            self.text(x, y, b["labels"][f], 19, c("body"))
            val_id = self.create_text(S(x + colw), S(y), text="—" if v is None else self._fmt(v, lo, hi),
                                      anchor="e", fill=c("body") if v is None else c("title"),
                                      font=(FONT, -S(19), "bold"))
            self.create_image(S(x + colw / 2), S(ty), image=hit_img, anchor="center", tags=(tag,))
            self.create_image(S(x), S(ty), image=track_img, anchor="w", tags=(tag,))
            photo = ImageTk.PhotoImage(full_fill.crop((0, 0, max(1, round(t * full_fill.width)), full_fill.height)))
            fill_id = self.create_image(S(x), S(ty), image=photo, anchor="w", tags=(tag,), state=state)
            knob_id = self.create_image(S(x + t * colw), S(ty), image=knob_img, anchor="center",
                                        tags=(tag,), state=state)
            self._sl[f] = dict(x=x, w=colw, ty=ty, lo=lo, hi=hi, val=val_id, fill=fill_id, knob=knob_id,
                               photo=photo, full=full_fill)

            self.hover(tag)
            handler = lambda e, name=f: self._drag(name, e)
            self.tag_bind(tag, "<Button-1>", handler)
            self.tag_bind(tag, "<B1-Motion>", handler)

    def _drag(self, name, event):
        S, c = self.s, theme.c
        s = self._sl[name]
        t = min(1.0, max(0.0, (event.x / self.k - s["x"]) / s["w"]))
        first = self.vals[name] is None
        self.vals[name] = s["lo"] + t * (s["hi"] - s["lo"])

        full = s["full"]
        s["photo"] = ImageTk.PhotoImage(full.crop((0, 0, max(1, round(t * full.width)), full.height)))
        self.itemconfigure(s["fill"], image=s["photo"])
        self.coords(s["knob"], S(s["x"] + t * s["w"]), S(s["ty"]))
        self.itemconfigure(s["val"], text=self._fmt(self.vals[name], s["lo"], s["hi"]))
        if first:
            self.itemconfigure(s["fill"], state="normal")
            self.itemconfigure(s["knob"], state="normal")
            self.itemconfigure(s["val"], fill=c("title"))
            self._update_counter()

    def _draw_algorithm(self, px, pw, pal):
        c, S = theme.c, self.s
        dark = theme.name == "dark"
        k = self.bundle["k"]
        algo = self._clu().get("algorithm") or "KMeans"
        x0, inner = px + 48, pw - 96
        kw = 150
        aw = inner - kw - 20
        top = PY + PH - 168
        border = ("#7a5aa6" if dark else "#bfaedf", 255)

        # خط جداکنندهٔ بخش مدل از اسلایدرها
        self.create_line(S(x0), S(top - 52), S(x0 + inner), S(top - 52), fill=pal["sep"], width=max(1, S(1.5)))
        self.text(x0, top - 24, "Algorithm", 20, c("title"), True)
        self.text(x0 + aw + 20, top - 24, "K (clusters)", 20, c("title"), True)
        for x, w, label in ((x0, aw, f"Clustering: {algo}"), (x0 + aw + 20, kw, str(k))):
            self.create_image(S(x), S(top), image=self._shape(w, 52, 26, pal["search_bg"], border=border), anchor="nw")
            self.text(x + 26, top + 26, label, 22, c("title"), True)

    def _draw_buttons(self, px, pal):
        c, S = theme.c, self.s
        dark = theme.name == "dark"
        by, bh, pw = PY + PH - 92, 58, 190
        x = px + 48
        img, pad = self._glass(pw, bh, 29, "#b4376e", "#5a1f4b", 255, 110, (163, 48, 95), None, 10, 8)
        self.create_image(S(x) - pad, S(by) - pad, image=img, anchor="nw", tags=("btn_predict",))
        self.text(x + pw / 2, by + bh / 2, "Predict", 23, "white", True, anchor="center", tags=("btn_predict",))
        self.hover("btn_predict")
        self.tag_bind("btn_predict", "<Button-1>", lambda e: self._predict())

        rw, rx = 150, x + pw + 20
        edge = "#f08fbd" if dark else BTN_LIGHT
        self.create_image(S(rx), S(by), image=self._shape(rw, bh, 29, pal["search_bg"], border=(edge, 255)),
                          anchor="nw", tags=("btn_reset",))
        self.text(rx + rw / 2, by + bh / 2, "Reset", 23, edge, True, anchor="center", tags=("btn_reset",))
        self.hover("btn_reset")
        self.tag_bind("btn_reset", "<Button-1>", lambda e: self._reset())

    # ---- پنل راست (همه با تگ res تا بدون ساخت دوبارهٔ صفحه به‌روز شود) ----
    @staticmethod
    def _first(*values):
        for v in values:
            if v is not None:
                return v
        return None

    def _quality(self, label):
        """Silhouette و SSD: اول از محاسبهٔ مستقیم، بعد از هر جای متادیتا که وجود داشته باشد."""
        q = (self.bundle or {}).get("quality") or {}
        clu = self._clu()
        ssd = clu.get("ssd") or {}
        ev = self.meta.get("model_evaluation") or {}
        sil = self._first(clu.get("silhouette_score"), ev.get("silhouette_score"))
        total = self._first(q.get("total"), ssd.get("total_ssd"), ev.get("SSD"), ev.get("ssd"))
        expl = self._first(q.get("explained"), ssd.get("explained_variance_pct"))
        per = q.get("per") or ssd.get("per_cluster_ssd") or {}
        share = None
        if label is not None and total and str(label) in per:
            share = float(per[str(label)]) / float(total) * 100
        return sil, total, expl, share

    def _draw_result(self):
        S, T = self.s, ("res",)
        b, res = self.bundle, self.result
        if not b:
            return
        k = b["k"]
        clu = self._clu()
        sizes, pcts = clu.get("cluster_sizes") or {}, clu.get("cluster_percentages") or {}
        left, right = RX + 48, RX + RW - 48
        width = right - left
        label = res["label"] if res else None

        self.text(left, PY + 52, "Predicted cluster", 22, "#d9c8ee", tags=T)
        if res:
            share = res["shares"][label]
            self.text(left, PY + 126, f"Cluster {label}", 72, "#f3b3d3", True, tags=T)
            self.text(left, PY + 190, "Closeness to this cluster's centre", 17, "#d9c8ee", tags=T)
            d = 124
            self.create_image(S(right - d / 2), S(PY + 124), image=self._ring_img(share, d, 14),
                              anchor="center", tags=T)
            self.text(right - d / 2, PY + 124, f"{share * 100:.0f}%", 28, "white", True, anchor="center", tags=T)
        else:
            self.text(left, PY + 126, "—", 72, "#f3b3d3", True, tags=T)
            if self.notice:
                self.text(left, PY + 190, self.notice, 18, "#ffb3d1", True, tags=T)
            else:
                self.text(left, PY + 190, "Set every feature value, then press Predict.", 18, "#d9c8ee", tags=T)

        self.create_line(S(left), S(PY + 222), S(right), S(PY + 222), fill="#6a4a8c", width=max(1, S(1)), tags=T)

        # نوارهای نزدیکی به هر خوشه
        top = PY + 246
        pitch = min(70, (PH - 222 - 246) / max(k, 1))
        compact = pitch < 52
        fs, toff = (17, 24) if compact else (20, 30)
        for i in range(k):
            y = top + i * pitch
            active = res is not None and i == label
            share = res["shares"][i] if res else 0.0
            col = "white" if active else "#d9c8ee"
            self.text(left, y, f"Cluster {i}", fs, col, active, tags=T)
            self.text(right, y, f"{share * 100:.0f}%" if res else "–", fs, col, active, anchor="e", tags=T)
            self.create_image(S(left), S(y + toff), image=self._shape(width, 10, 5, BAR_TRACK, border=("#7a5aa6", 120)),
                              anchor="w", tags=T)
            if res:
                c1, c2 = BAR_ON if active else BAR_OFF
                self.create_image(S(left), S(y + toff), image=self._shape(max(10, share * width), 10, 5, c1, c2),
                                  anchor="w", tags=T)

        # کاشی‌های کیفیت: Silhouette / SSD (با درصد) / سهم SSD خوشهٔ پیش‌بینی‌شده
        sil, total, expl, cl_share = self._quality(label)
        tiles = [
            ("Silhouette", "—" if sil is None else f"{float(sil):.3f}", "cluster separation"),
            ("SSD", "—" if total is None else f"{float(total):,.2f}",
             "" if expl is None else f"{float(expl):.1f}% explained"),
            ("Cluster SSD", "—" if cl_share is None else f"{cl_share:.1f}%", "of total SSD"),
        ]
        gap, th = 12, 92
        tw = (width - 2 * gap) / 3
        ty = PY + PH - 206
        for i, (name, value, sub) in enumerate(tiles):
            x = left + i * (tw + gap)
            self.create_image(S(x), S(ty), image=self._shape(tw, th, 22, "#ffffff", alpha=30, border=("#ffffff", 50)),
                              anchor="nw", tags=T)
            self.text(x + 18, ty + 22, name, 16, "#d9c8ee", tags=T)
            self.text(x + 18, ty + 52, value, 28, "white", True, tags=T)
            self.text(x + 18, ty + 77, sub, 14, "#bfa9dd", tags=T)

        # توضیح مدل
        algo = clu.get("algorithm") or "KMeans"
        text = f"Model: {algo}, K = {k}. Trained on {self.ds.rows:,} rows and {len(b['features'])} features."
        if res and str(label) in sizes:
            pct = pcts.get(str(label))
            text += f" Cluster {label} holds {sizes[str(label)]:,} records" + (f" ({pct:.1f}%)." if pct is not None else ".")
        by, bh = PY + PH - 102, 70
        self.create_image(S(left), S(by), image=self._shape(width, bh, 22, "#ffffff", alpha=34), anchor="nw", tags=T)
        for i, line in enumerate(self._wrap(text, 17, False, width - 40, 2)):
            self.text(left + 20, by + 24 + i * 24, line, 17, "#eadcf6", tags=T)