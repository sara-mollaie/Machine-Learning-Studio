import json
import math
import random
import re
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps, ImageTk

from UserInterface.HomeModule import HomeModule, BASE, _rgb, HDR_TOP, HDR_H, HDR_CY
from UserInterface.Theme import theme, FONT

BG_LIGHT = BASE / "assets" / "page_images" / "datasets_background_light.png"
BG_DARK = BASE / "assets" / "page_images" / "datasets_background_dark.png"

# ---- چیدمان (واحد: پیکسل طراحی ۱۶۷۲×۹۴۱، مثل HomeModule) ----
DESIGN_W, DESIGN_H = 1672, 941      # اندازهٔ صفحه؛ مستقل از رزولوشن فایل پس‌زمینه
MX = 64                       # حاشیهٔ چپ محتوا
CONTENT_W = 1544              # عرض محتوا (۶۴ تا ۱۶۰۸)
COLS, ROWS = 4, 2
GAP_X, GAP_Y = 24, 22
CARD_W = (CONTENT_W - GAP_X * (COLS - 1)) // COLS
CARD_H = 226
FILTER_Y = 266
GRID_Y = 332
BOTTOM_CY = 874               # مرکز ردیف پایین (دستهٔ موضوعی + صفحه‌بندی)
PER_PAGE = COLS * ROWS

# پنجرهٔ اطلاعات دیتاست (روی خود صفحه، وسط)
INFO_W, INFO_H = 900, 790
INFO_X, INFO_Y = (1672 - INFO_W) // 2, 100

LEVELS = ["Beginner", "Intermediate", "Advanced"]
TASKS = ["Regression", "Classification", "Clustering"]

LEVEL_DOT = {"Beginner": "#2fb57a", "Intermediate": "#f0a23b", "Advanced": "#e5546b"}

TASK_ACCENT = {
    "Regression": ("#6c4cd6", "#9a74f0"),
    "Classification": ("#ee5c9f", "#f58bbe"),
    "Clustering": ("#1fb8a8", "#4bd6c6"),
}
TASK_DEFAULT = ("#7a6f8e", "#a396b8")

PALETTE = {
    "light": {
        "tint": {"Regression": "#efe9fa", "Classification": "#fbe9ef", "Clustering": "#e3f5f2"},
        "tint_default": "#f1ebfa",
        "search_bg": "#ffffff", "search_border": ("#e1d3f4", 255),
        "pill_off": "#ffffff", "pill_off_a": 205,
        "badge_bg": "#ffffff",
        "tile_bg": "#f8f4fd", "step_circle": "#e2d5f6", "sep": "#e7ddf4",
        "popup_a": 255,
    },
    "dark": {
        "tint": {"Regression": "#2a1a47", "Classification": "#3a1b42", "Clustering": "#16363a"},
        "tint_default": "#271a45",
        "search_bg": "#2a1a47", "search_border": ("#4a3566", 255),
        "pill_off": "#2a1a47", "pill_off_a": 235,
        "badge_bg": "#3a2866",
        "tile_bg": "#241540", "step_circle": "#3d2a68", "sep": "#3b2a5e",
        "popup_a": 255,
    },
}

# سایهٔ رنگی و ملایم زیر کارت‌ها در تم روشن (سبز برای Clustering)
CARD_SHADOW = {
    "Regression": (108, 76, 214),
    "Classification": (238, 92, 159),
    "Clustering": (40, 214, 168),
}
CARD_SHADOW_DEFAULT = (120, 100, 170)
LIGHT_SHADOW_ALPHA = 140

BTN_DARK = "#5a1f4b"          # گرادیان دکمه‌ها: مثل دکمهٔ Get Started
BTN_LIGHT = "#a3305f"


# =====================================================
# DATA LAYER
# =====================================================

@dataclass
class DatasetInfo:
    id: str
    name: str
    level: str
    domain: str
    task: str
    rows: int
    columns: int
    description: str
    source: str
    path: Path
    raw: dict = field(default_factory=dict)

    @property
    def topics(self):
        """«Business / Retail» → ['Business', 'Retail']"""
        parts = [p.strip() for p in str(self.domain).split("/") if p.strip()]
        return parts or ["General"]

    @property
    def sort_key(self):
        level_rank = LEVELS.index(self.level) if self.level in LEVELS else len(LEVELS)
        match = re.search(r"\d+", self.id or "")
        number = int(match.group()) if match else 9999
        return (level_rank, number, self.name.lower())


def _canonical(value, allowed, default):
    text = str(value or "").strip()
    for item in allowed:
        if item.lower() == text.lower():
            return item
    return text.title() if text else default


def _to_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


class DatasetCatalog:
    """متادیتای دیتاست‌ها را از <root>/*/Metadata/*.json پیدا و بارگذاری می‌کند."""

    PATTERNS = ("*/Metadata/*.json", "Metadata/*.json")

    def __init__(self, root=None):
        self.root = Path(root) if root else BASE

    def _files(self):
        seen = set()
        for pattern in self.PATTERNS:
            for path in self.root.glob(pattern):
                if any(part.startswith(".") for part in path.relative_to(self.root).parts):
                    continue                                   # .venv و .idea
                if path not in seen:
                    seen.add(path)
                    yield path

    def signature(self):
        items = []
        for path in self._files():
            try:
                items.append((str(path), path.stat().st_mtime_ns))
            except OSError:
                pass
        return tuple(sorted(items))

    def load(self):
        datasets = []
        for path in self._files():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                datasets.append(self._parse(path, data))
            except (OSError, ValueError, TypeError) as error:
                print(f"[Datasets] Skipped {path.name}: {error}")
        datasets.sort(key=lambda d: d.sort_key)
        return datasets

    @staticmethod
    def _parse(path, data):
        if not isinstance(data, dict):
            raise ValueError("metadata root must be an object")
        stats = data.get("dataset_statistics") or {}
        return DatasetInfo(
            id=str(data.get("id") or path.stem),
            name=str(data.get("name") or path.stem.replace("_", " ")),
            level=_canonical(data.get("level"), LEVELS, "Beginner"),
            domain=str(data.get("domain") or "General"),
            task=_canonical(data.get("task"), TASKS, "Other"),
            rows=_to_int(stats.get("rows", data.get("rows"))),
            columns=_to_int(stats.get("columns", data.get("columns"))),
            description=str(data.get("description") or ""),
            source=str(data.get("source") or "Data Science Dojo"),
            path=path,
            raw=data,
        )


# =====================================================
# PAGE
# =====================================================

class DatasetsModule(HomeModule):
    """صفحهٔ Datasets؛ از HomeModule ابزارهای رسم (panel, text, pic, nav_hover ...) را به ارث می‌برد."""

    def __init__(self, master, scale=1.0, on_navigate=None, on_open=None, on_predict=None, catalog=None,
                 poll_ms=2500):
        # نکته: عمداً HomeModule.__init__ صدا زده نمی‌شود (پس‌زمینهٔ Home را بارگذاری می‌کند)
        self._light = self._load_background(BG_LIGHT)
        self.k = scale
        self.W, self.H = int(DESIGN_W * scale), int(DESIGN_H * scale)
        tk.Canvas.__init__(self, master, width=self.W, height=self.H, highlightthickness=0, bd=0)

        self.on_navigate = on_navigate or (lambda page: print("goto", page))
        self.on_open = on_open            # برای سازگاری با MainView قدیمی؛ دیگر لازم نیست
        self.on_predict = on_predict or self._predictor_soon
        self.catalog = catalog or DatasetCatalog()
        self.poll_ms = poll_ms

        self._refs, self._bgs, self._cache = {}, {}, {}
        self.all, self.filtered = [], []
        self.level_filter = self.task_filter = self.topic_filter = None
        self.page = 1
        self._popup = False
        self._info = None                 # اطلاعات پنجرهٔ باز (None = بسته)
        self._entry_item = None
        self._placeholder_on = True
        self._sig = None
        self._poll_job = None

        self._make_entry()
        self._load_data()

        self.bind_all("<Escape>", self._on_escape, add="+")

        theme.subscribe(self._build)
        self._build()
        self._poll_job = self.after(self.poll_ms, self._poll)

    # ---------- پس‌زمینه ----------
    @staticmethod
    def _load_background(path):
        if path.exists():
            return Image.open(path).convert("RGBA")
        # بدون فایل عکس هم صفحه کار می‌کند (گرادیان ساده با اندازهٔ طراحی)
        img = Image.new("RGBA", (1672, 941))
        d = ImageDraw.Draw(img)
        for y in range(941):
            t = y / 940
            d.line([(0, y), (1672, y)], fill=(int(249 - 22 * t), int(243 - 30 * t), int(254 - 8 * t), 255))
        return img

    def _bg_photo(self):
        name = theme.name
        if name not in self._bgs:
            img = self._load_background(BG_DARK) if name == "dark" and BG_DARK.exists() else self._light
            self._bgs[name] = ImageTk.PhotoImage(img.resize((self.W, self.H), Image.LANCZOS))
        return self._bgs[name]

    def _pal(self):
        return PALETTE[theme.name]

    # ---------- داده ----------
    def _load_data(self):
        self.all = self.catalog.load()
        self._sig = self.catalog.signature()
        self._apply_filters()

    def _poll(self):
        try:
            if self.catalog.signature() != self._sig:
                self._load_data()
                self._draw_dynamic()
        except Exception as error:
            print("[Datasets] refresh check failed:", error)
        self._poll_job = self.after(self.poll_ms, self._poll)

    def on_show(self):
        """هنگام باز شدن صفحه صدا زده می‌شود تا دیتاست‌های تازه بلافاصله دیده شوند."""
        self._popup = False
        if self.catalog.signature() != self._sig:
            self._load_data()
        self._draw_dynamic()

    def set_task_filter(self, task):
        """نمایش دیتاست‌های الگوریتم انتخاب‌شده از صفحه Algorithms."""
        task = _canonical(task, TASKS, "")
        if task not in TASKS:
            return

        self.task_filter = task
        self.level_filter = None
        self.topic_filter = None
        self.page = 1
        self._popup = False

        # بستن پنجرهٔ Info در صورت باز بودن
        self._info = None
        self.delete("overlay")

        # پاک کردن جست‌وجوی قبلی تا فقط فیلتر الگوریتم اعمال شود
        self.entry.delete(0, "end")
        self.entry.insert(0, "Search datasets")
        self.entry.config(fg=theme.c("body"))
        self._placeholder_on = True

        self._apply_filters()
        self._draw_dynamic()

    def destroy(self):
        if self._poll_job:
            try:
                self.after_cancel(self._poll_job)
            except Exception:
                pass
        theme.unsubscribe(self._build)
        super().destroy()

    def _query(self):
        return "" if self._placeholder_on else self.entry.get().strip().lower()

    def _apply_filters(self):
        query = self._query()

        def ok(ds):
            if self.level_filter and ds.level != self.level_filter:
                return False
            if self.task_filter and ds.task != self.task_filter:
                return False
            if self.topic_filter and self.topic_filter not in ds.topics:
                return False
            if query:
                hay = " ".join((ds.name, ds.domain, ds.task, ds.level, ds.description)).lower()
                return query in hay
            return True

        self.filtered = [d for d in self.all if ok(d)]

    def _topics(self):
        counts = {}
        for ds in self.all:
            for t in ds.topics:
                counts[t] = counts.get(t, 0) + 1
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))

    # ---------- ساخت تصویرهای PIL (با کش) ----------
    def _shape(self, w, h, r, c1, c2=None, alpha=255, border=None):
        key = ("shape", self.k, w, h, r, c1, c2, alpha, border)
        if key in self._cache:
            return self._cache[key]
        W, H, R, ss = self.s(w), self.s(h), self.s(r), 3
        a, b = _rgb(c1), _rgb(c2 or c1)
        body = Image.new("RGBA", (W, H))
        d = ImageDraw.Draw(body)
        for x in range(W):
            t = x / max(W - 1, 1)
            d.line([(x, 0), (x, H)], fill=tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3)) + (255,))
        mask = Image.new("L", (W * ss, H * ss), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, W * ss - 1, H * ss - 1), R * ss, fill=alpha)
        body.putalpha(mask.resize((W, H), Image.LANCZOS))
        if border:
            ring = Image.new("L", (W * ss, H * ss), 0)
            ImageDraw.Draw(ring).rounded_rectangle((0, 0, W * ss - 1, H * ss - 1), R * ss, outline=255,
                                                   width=max(2, self.s(2)) * ss // 2)
            ring = ring.resize((W, H), Image.LANCZOS)
            edge = Image.new("RGBA", (W, H), _rgb(border[0]) + (0,))
            edge.putalpha(ring.point(lambda v: v * border[1] // 255))
            body.alpha_composite(edge)
        photo = ImageTk.PhotoImage(body)
        self._cache[key] = photo
        return photo

    def _glass(self, w, h, r, c1, c2, alpha, shadow, shadow_rgb, border, blur=14, off=10):
        key = ("glass", self.k, w, h, r, c1, c2, alpha, shadow, tuple(shadow_rgb), border, blur, off)
        if key in self._cache:
            return self._cache[key]
        W, H, R, ss = self.s(w), self.s(h), self.s(r), 3
        pad = self.s(max(34, off + 2 * blur)) if shadow else 0
        canvas = Image.new("RGBA", (W + 2 * pad, H + 2 * pad), (0, 0, 0, 0))
        if shadow:
            sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            dy = self.s(off)
            ImageDraw.Draw(sh).rounded_rectangle((pad, pad + dy, pad + W, pad + dy + H), R,
                                                 fill=tuple(shadow_rgb) + (shadow,))
            canvas = Image.alpha_composite(canvas, sh.filter(ImageFilter.GaussianBlur(self.s(blur))))
        a, b = _rgb(c1), _rgb(c2 or c1)
        body = Image.new("RGBA", (W, H))
        d = ImageDraw.Draw(body)
        for x in range(W):
            t = x / max(W - 1, 1)
            d.line([(x, 0), (x, H)], fill=tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3)) + (255,))
        mask = Image.new("L", (W * ss, H * ss), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, W * ss - 1, H * ss - 1), R * ss, fill=alpha)
        body.putalpha(mask.resize((W, H), Image.LANCZOS))
        canvas.alpha_composite(body, (pad, pad))
        if border:
            ring = Image.new("L", (W * ss, H * ss), 0)
            ImageDraw.Draw(ring).rounded_rectangle((0, 0, W * ss - 1, H * ss - 1), R * ss, outline=255,
                                                   width=max(2, self.s(2)) * ss // 2)
            ring = ring.resize((W, H), Image.LANCZOS)
            edge = Image.new("RGBA", (W, H), _rgb(border[0]) + (0,))
            edge.putalpha(ring.point(lambda v: v * border[1] // 255))
            canvas.alpha_composite(edge, (pad, pad))
        result = (ImageTk.PhotoImage(canvas), pad)
        self._cache[key] = result
        return result

    def _task_icon(self, task, size):
        key = ("icon", self.k, task, size)
        if key in self._cache:
            return self._cache[key]
        S, ss = self.s(size), 4
        c1, c2 = (_rgb(c) for c in TASK_ACCENT.get(task, TASK_DEFAULT))
        tile = Image.new("RGBA", (S, S))
        d = ImageDraw.Draw(tile)
        for i in range(S * 2):                                 # گرادیان مورب
            t = i / (S * 2 - 1)
            d.line([(i, 0), (0, i)], fill=tuple(int(c1[k] + (c2[k] - c1[k]) * t) for k in range(3)) + (255,))
        tile.putalpha(self._round_mask(S, S, int(S * 0.28)))

        g = Image.new("RGBA", (S * ss, S * ss), (0, 0, 0, 0))
        gd = ImageDraw.Draw(g)
        u, w = S * ss, max(3, int(S * ss * 0.06))
        white = (255, 255, 255, 255)

        def ring(cx, cy, r):
            gd.ellipse((cx * u - r * u, cy * u - r * u, cx * u + r * u, cy * u + r * u), outline=white, width=w)

        if task == "Regression":
            gd.line([(0.28 * u, 0.26 * u), (0.28 * u, 0.72 * u), (0.74 * u, 0.72 * u)], fill=white, width=w, joint="curve")
            pts = [(0.36, 0.62), (0.47, 0.50), (0.57, 0.57), (0.70, 0.36)]
            gd.line([(x * u, y * u) for x, y in pts], fill=white, width=w, joint="curve")
            for x, y in pts[1:]:
                ring(x, y, 0.045)
        elif task == "Classification":
            for x, y in ((0.34, 0.34), (0.66, 0.34), (0.34, 0.66), (0.66, 0.66)):
                ring(x, y, 0.10)
        elif task == "Clustering":
            for x, y in ((0.5, 0.33), (0.33, 0.65), (0.67, 0.65)):
                ring(x, y, 0.11)
        else:
            for x, y in ((0.35, 0.35), (0.65, 0.35), (0.35, 0.65), (0.65, 0.65)):
                gd.ellipse((x * u - w, y * u - w, x * u + w, y * u + w), fill=white)
        tile.alpha_composite(g.resize((S, S), Image.LANCZOS))
        photo = ImageTk.PhotoImage(tile)
        self._cache[key] = photo
        return photo


    # =====================================================
    # پنجرهٔ اطلاعات دیتاست (روی همین صفحه)
    # =====================================================
    def _predictor_soon(self, ds):
        messagebox.showinfo("Predictor", f"The predictor for “{ds.name}” will be available soon.",
                            parent=self.winfo_toplevel())

    def _on_escape(self, _event=None):
        if self._info:
            self._close_info()

    def _open_info(self, ds):
        """متادیتای همان دیتاست را از فایل JSON می‌خواند و پنجره را نشان می‌دهد."""
        try:
            meta = json.loads(ds.path.read_text(encoding="utf-8"))      # همیشه نسخهٔ تازهٔ فایل
        except (OSError, ValueError):
            meta = ds.raw
        self._popup = False
        self._info = self._parse_info(ds, meta if isinstance(meta, dict) else {})
        self._draw_dynamic()
        self._draw_info()

    def _close_info(self):
        self._info = None
        self.delete("overlay")
        if self._entry_item is not None:
            self.itemconfigure(self._entry_item, state="normal")
        self.config(cursor="")

    @staticmethod
    def _as_list(value):
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        return [str(value)] if value else []

    @staticmethod
    def _find_metric(meta, *names):
        """مقدار یک معیار را از چند جای رایج متادیتا (evaluation / metrics / ...) پیدا می‌کند."""
        pools = [meta]
        for key in ("evaluation", "metrics", "model_evaluation", "results", "supervised"):
            if isinstance(meta.get(key), dict):
                pools.append(meta[key])
        for pool in pools:
            for name in names:
                value = pool.get(name)
                if isinstance(value, (int, float)):
                    return float(value)
        return None

    def _parse_info(self, ds, meta):
        """
        متادیتا را برای هر سه نوع الگوریتم به ساختار ثابت پنجره تبدیل می‌کند:

            Clustering     → Preparation, X (features), Algorithm (+K), Quality
            Classification → Preparation, X (features), Y (target), Algorithm, Quality
            Regression     → Preparation, X (features), Y (target), Algorithm, Quality

        فیلدِ نبود = ردیفِ نبود. (ردیف Analysis variable حذف شده است.)
        """
        stats = meta.get("dataset_statistics") or {}
        fs = meta.get("feature_selection") or {}
        pre = meta.get("preprocessing") or {}
        clu = meta.get("clustering") or {}
        task = ds.task

        rows = _to_int(stats.get("rows", ds.rows))
        cols = _to_int(stats.get("columns", ds.columns))
        target = meta.get("target") or meta.get("target_column") or fs.get("target") or meta.get("y")

        features = (self._as_list(fs.get("selected_features")) or self._as_list(clu.get("input_features"))
                    or self._as_list(pre.get("input_features")) or self._as_list(meta.get("features")))
        if not features and target and stats.get("column_names"):
            features = [col for col in self._as_list(stats["column_names"]) if col != str(target)]

        # ---- کاشی سوم: بسته به نوع الگوریتم
        classes = meta.get("n_classes") or meta.get("classes")
        if task == "Clustering" and clu.get("n_clusters"):
            third = (str(clu["n_clusters"]), "Clusters", "clusters")
        elif task == "Classification" and classes:
            n = len(classes) if isinstance(classes, (list, tuple)) else classes
            third = (str(n), "Classes", "classes")
        else:
            third = (str(len(features)), "Features", "features")

        # ---- جدول مشخصات: فیلدهای موجود در Metadata هر دیتاست
        table = []

        # Preparation: فقط روش نرمال‌سازی نمایش داده می‌شود.
        normalization = (
            pre.get("normalization")
            or pre.get("normalization_method")
            or pre.get("scaler")
            or pre.get("scaling_method")
            or pre.get("method")
        )

        # سازگاری با Metadataهای قدیمی
        if not normalization:
            legacy_prep = meta.get("preparation")
            if isinstance(legacy_prep, dict):
                normalization = (
                    legacy_prep.get("normalization")
                    or legacy_prep.get("normalization_method")
                    or legacy_prep.get("scaler")
                    or legacy_prep.get("scaling_method")
                    or legacy_prep.get("method")
                )
            elif isinstance(legacy_prep, str):
                legacy_text = legacy_prep.strip()
                if "scaler" in legacy_text.lower() or "normalizer" in legacy_text.lower():
                    normalization = legacy_text

        if isinstance(normalization, (list, tuple)):
            normalization = " · ".join(str(value).strip() for value in normalization if value)

        # نمایش نام تکنیک نرمال‌سازی به‌جای نام کلاس کتابخانه scikit-learn.
        normalization_labels = {
            "minmaxscaler": "Min-Max Normalization",
            "min-maxscaler": "Min-Max Normalization",
            "min-max scaler": "Min-Max Normalization",
            "standardscaler": "Z-Score Standardization",
            "standard scaler": "Z-Score Standardization",
            "robustscaler": "Robust Scaling (Median/IQR)",
            "robust scaler": "Robust Scaling (Median/IQR)",
            "maxabsscaler": "Max-Abs Scaling",
            "max abs scaler": "Max-Abs Scaling",
            "normalizer": "Unit-Norm Normalization",
            "quantiletransformer": "Quantile Transformation",
            "powertransformer": "Power Transformation"
        }

        if normalization:
            normalization_key = str(normalization).strip().casefold()
            normalization = normalization_labels.get(normalization_key, str(normalization).strip())
            table.append(("Preparation", normalization))

        if features:
            shown = ", ".join(features[:8]) + (f" +{len(features) - 8} more" if len(features) > 8 else "")
            table.append(("X (features)", shown))

        if task in ("Classification", "Regression") and target:  # Clustering هدف ندارد
            table.append(("Y (target)", str(target)))

        algo = clu.get("algorithm") or meta.get("algorithm")
        algo_text = f"{task} ({algo})" if algo else task
        n_clusters = clu.get("n_clusters") or (meta.get("k_selection") or {}).get("final_k")
        if task == "Clustering" and n_clusters is not None:
            algo_text += f" · K = {n_clusters}"
        table.append(("Algorithm", algo_text))

        quality = []
        if task == "Clustering":
            model_eval = meta.get("model_evaluation") or meta.get("evaluation") or {}
            sil = clu.get("silhouette_score")
            if sil is None:
                sil = model_eval.get("silhouette_score")
            if sil is not None:
                quality.append(f"Silhouette {float(sil):.3f}")

            # PCA Variance برای Travel Reviews در پنجره Info نمایش داده نمی‌شود.
            is_travel_reviews = (
                str(meta.get("id", "")).strip().upper() == "D21"
                or str(meta.get("name") or ds.name).strip().casefold() == "travel reviews"
            )

            if not is_travel_reviews:
                pca_info = (
                    meta.get("dimensionality_reduction")
                    or meta.get("pca")
                    or model_eval.get("dimensionality_reduction")
                    or model_eval.get("pca")
                    or {}
                )
                ev_pca = pca_info.get("total_explained_variance_pct")
                if ev_pca is None:
                    ev_pca = pca_info.get("explained_variance_pct")

                if ev_pca is None:
                    ratios = pca_info.get("explained_variance_ratio")
                    if isinstance(ratios, dict):
                        ev_pca = sum(float(value) for value in ratios.values()) * 100
                    elif isinstance(ratios, (list, tuple)):
                        ev_pca = sum(float(value) for value in ratios) * 100

                if ev_pca is not None:
                    ev_pca = float(ev_pca)
                    if 0 <= ev_pca <= 1:
                        ev_pca *= 100
                    quality.append(f"PCA variance {ev_pca:.1f}%")

        elif task == "Classification":
            acc = self._find_metric(meta, "accuracy", "test_accuracy")
            f1 = self._find_metric(meta, "f1", "f1_score", "f1_macro")
            best_k = self._find_metric(meta, "best_k")
            if acc is not None:
                quality.append(f"Accuracy {acc * 100:.1f}%" if acc <= 1 else f"Accuracy {acc:.1f}%")
            if f1 is not None:
                quality.append(f"F1 {f1:.3f}")
            if best_k is not None:
                quality.append(f"best k = {int(best_k)}")

        elif task == "Regression":
            r2 = self._find_metric(meta, "r2", "r2_score", "r_squared")
            mae = self._find_metric(meta, "mae", "mean_absolute_error")
            rmse = self._find_metric(meta, "rmse")
            if r2 is not None:
                quality.append(f"R² {r2:.3f}")
            if mae is not None:
                quality.append(f"MAE {mae:,.2f}")
            if rmse is not None:
                quality.append(f"RMSE {rmse:,.2f}")

        if quality:
            table.append(("Quality", " · ".join(quality)))

        return {
            "ds": ds,
            "title": str(meta.get("name") or ds.name),
            "task": str(meta.get("task") or ds.task),
            "level": ds.level,
            "domain": str(meta.get("domain") or ds.domain),
            "description": str(meta.get("description") or "No description available."),
            "stats": [(f"{rows:,}", "Rows", "rows"), (f"{cols:,}", "Columns", "columns"), third],
            "table": table[:5],
        }

    # ---- تصویرهای PIL پنجره ----
    def _scrim(self):
        """پس‌زمینهٔ محو و تیره پشت پنجره (مثل طرح: صفحه با هدرِ محو)."""
        key = ("scrim", self.k, theme.name)
        if key in self._cache:
            return self._cache[key]
        dark = theme.name == "dark"
        base = self._load_background(BG_DARK) if dark and BG_DARK.exists() else self._light
        img = base.resize((self.W, self.H), Image.LANCZOS).convert("RGBA")
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((self.s(23), self.s(HDR_TOP), self.s(1649), self.s(HDR_TOP + HDR_H)), self.s(41),
                            fill=_rgb(theme.c("hdr")[0]) + (255,))
        img = img.filter(ImageFilter.GaussianBlur(self.s(16)))
        img.alpha_composite(Image.new("RGBA", img.size, (20, 8, 45, 130) if dark else (75, 45, 120, 150)))
        photo = ImageTk.PhotoImage(img.convert("RGB"))
        self._cache[key] = photo
        return photo

    def _dataset_image_path(self, ds):
        """Find a dataset image by metadata ID inside its task-specific assets folder."""
        assets_root = BASE / "assets" / "DataSets"
        task_name = str(ds.task or "").strip().casefold()
        task_folder = assets_root / task_name

        if not task_folder.is_dir() and assets_root.is_dir():
            for folder in assets_root.iterdir():
                if folder.is_dir() and folder.name.casefold() == task_name:
                    task_folder = folder
                    break

        if not task_folder.is_dir():
            return None

        dataset_id = str(ds.id or "").strip().casefold()
        if not dataset_id:
            return None

        supported_extensions = {".png", ".jpg", ".jpeg", ".webp"}
        candidates = []
        for image_path in task_folder.iterdir():
            if not image_path.is_file() or image_path.suffix.casefold() not in supported_extensions:
                continue

            image_stem = image_path.stem.casefold()
            if (
                image_stem == dataset_id
                or image_stem.startswith(dataset_id + "_")
                or image_stem.startswith(dataset_id + "-")
            ):
                candidates.append(image_path)

        extension_order = {".png": 0, ".jpg": 1, ".jpeg": 2, ".webp": 3}
        candidates.sort(key=lambda path: (extension_order.get(path.suffix.casefold(), 9), path.name.casefold()))
        return candidates[0] if candidates else None

    def _dataset_thumbnail(self, ds, width=170, height=151):
        """Create a full-bleed header image, flush to the card's left/top/bottom edges."""
        image_path = self._dataset_image_path(ds)
        if image_path is None:
            return None

        try:
            modified = image_path.stat().st_mtime_ns
            cache_key = (
                "dataset_header_image", self.k, theme.name, str(image_path),
                modified, width, height
            )
            if cache_key in self._cache:
                return self._cache[cache_key]

            scale = 3
            out_w = self.s(width) * scale
            out_h = self.s(height) * scale
            corner = self.s(36) * scale

            with Image.open(image_path) as source_image:
                source_image = source_image.convert("RGB")
                fitted = ImageOps.fit(
                    source_image,
                    (out_w, out_h),
                    method=Image.LANCZOS,
                    centering=(0.5, 0.5)
                ).convert("RGBA")

            # Only the card's outer top-left corner is rounded. The image
            # remains flush with the header's top, bottom and left edges.
            mask = Image.new("L", (out_w, out_h), 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.rounded_rectangle(
                (0, 0, out_w - 1, out_h - 1),
                radius=corner,
                fill=255
            )
            mask_draw.rectangle((corner, 0, out_w - 1, out_h - 1), fill=255)
            mask_draw.rectangle((0, corner, corner, out_h - 1), fill=255)
            fitted.putalpha(mask)

            photo = ImageTk.PhotoImage(
                fitted.resize((self.s(width), self.s(height)), Image.LANCZOS)
            )
            self._cache[cache_key] = photo
            return photo

        except (OSError, ValueError) as error:
            print(f"[Datasets] Could not load image for {ds.id}: {error}")
            return None

    def _info_card(self, height):
        """بدنهٔ پنجره: سایه + کارت + هدر گرادیانی با درخششِ صورتی گوشهٔ بالا-راست."""
        key = ("info_card", self.k, theme.name, height)
        if key in self._cache:
            return self._cache[key]
        S, c = self.s, theme.c
        W, H, pad, R, ss = S(INFO_W), S(height), S(64), S(36), 3
        canvas = Image.new("RGBA", (W + 2 * pad, H + 2 * pad), (0, 0, 0, 0))

        sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle((pad, pad + S(20), pad + W, pad + S(20) + H), R, fill=(25, 8, 55, 150))
        canvas.alpha_composite(sh.filter(ImageFilter.GaussianBlur(S(26))))

        card = Image.new("RGBA", (W, H), _rgb(c("dlg_bg")) + (255,))
        hh = S(151)
        head = Image.new("RGBA", (W, hh))
        hd = ImageDraw.Draw(head)
        a, b = _rgb("#2a1250"), _rgb("#5b2352")
        for x in range(W):
            t = x / max(W - 1, 1)
            hd.line([(x, 0), (x, hh)], fill=tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3)) + (255,))
        glow = Image.new("RGBA", (W, hh), (0, 0, 0, 0))
        ImageDraw.Draw(glow).ellipse((int(W * 0.60), -int(hh * 0.95), int(W * 1.10), int(hh * 0.75)),
                                     fill=(196, 96, 156, 170))
        head.alpha_composite(glow.filter(ImageFilter.GaussianBlur(S(42))))
        card.paste(head, (0, 0))

        mask = Image.new("L", (W * ss, H * ss), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, W * ss - 1, H * ss - 1), R * ss, fill=255)
        card.putalpha(mask.resize((W, H), Image.LANCZOS))
        canvas.alpha_composite(card, (pad, pad))

        photo = ImageTk.PhotoImage(canvas)
        self._cache[key] = (photo, pad)
        return photo, pad

    def _info_icon(self, kind, size):
        """کاشی بنفش با نماد سفید (Rows / Columns / Clusters / Classes / Features)."""
        key = ("info_icon", self.k, kind, size)
        if key in self._cache:
            return self._cache[key]
        S, ss = self.s(size), 6
        big = S * ss
        c1, c2 = _rgb("#4b2a8a"), _rgb("#7a4fd0")
        tile = Image.new("RGBA", (big, big))
        d = ImageDraw.Draw(tile)
        for i in range(big * 2):
            t = i / (big * 2 - 1)
            d.line([(i, 0), (0, i)], fill=tuple(int(c1[k] + (c2[k] - c1[k]) * t) for k in range(3)) + (255,))
        m = Image.new("L", (big, big), 0)
        ImageDraw.Draw(m).rounded_rectangle((0, 0, big - 1, big - 1), int(big * 0.30), fill=255)
        tile.putalpha(m)

        g = ImageDraw.Draw(tile)
        w, white = max(4, int(big * 0.07)), (255, 255, 255, 255)

        def line(x1, y1, x2, y2):
            g.line([(x1 * big, y1 * big), (x2 * big, y2 * big)], fill=white, width=w)
            for x, y in ((x1, y1), (x2, y2)):
                g.ellipse((x * big - w / 2, y * big - w / 2, x * big + w / 2, y * big + w / 2), fill=white)

        def ring(x, y, r):
            g.ellipse(((x - r) * big, (y - r) * big, (x + r) * big, (y + r) * big), outline=white, width=w)

        if kind == "rows":
            for y in (0.35, 0.5, 0.65):
                line(0.3, y, 0.7, y)
        elif kind == "columns":
            for x in (0.36, 0.5, 0.64):
                line(x, 0.30, x, 0.70)
        elif kind == "clusters":
            for x, y in ((0.5, 0.34), (0.34, 0.64), (0.66, 0.64)):
                ring(x, y, 0.10)
        elif kind == "classes":
            g.rounded_rectangle((0.32 * big, 0.32 * big, 0.68 * big, 0.68 * big), int(big * 0.09),
                                outline=white, width=w)
        else:
            for x, y in ((0.36, 0.36), (0.64, 0.36), (0.36, 0.64), (0.64, 0.64)):
                g.rounded_rectangle(((x - 0.07) * big, (y - 0.07) * big, (x + 0.07) * big, (y + 0.07) * big),
                                    int(big * 0.03), fill=white)

        photo = ImageTk.PhotoImage(tile.resize((S, S), Image.LANCZOS))
        self._cache[key] = photo
        return photo

    # ---- رسم پنجره ----
    def _draw_info(self):
        info = self._info
        if not info:
            return
        c, pal, S = theme.c, self._pal(), self.s
        ov = ("overlay",)
        rows = info["table"]
        height = 362 + len(rows) * 49 + 22 + 80 + 54 + 26         # پنجره به تعداد ردیف‌ها بسته است
        x0, y0 = INFO_X, (DESIGN_H - height) // 2
        left, right = x0 + 46, x0 + INFO_W - 46
        inner_w = right - left

        self.delete("overlay")
        if self._entry_item is not None:
            self.itemconfigure(self._entry_item, state="hidden")      # ویجت Entry همیشه روی بوم می‌ماند

        # پس‌زمینهٔ محو: کلیک روی آن پنجره را می‌بندد
        self.create_image(0, 0, image=self._scrim(), anchor="nw", tags=ov + ("ov_scrim",))
        self.tag_bind("ov_scrim", "<Button-1>", lambda e: self._close_info())

        card, pad = self._info_card(height)
        self.create_image(S(x0) - pad, S(y0) - pad, image=card, anchor="nw", tags=ov + ("ov_card",))
        self.tag_bind("ov_card", "<Button-1>", lambda e: None)

        # --- تصویر تمام‌لبه در سمت چپ هدر؛ بدون قاب یا حاشیه
        thumbnail = self._dataset_thumbnail(info["ds"], width=170, height=151)
        header_left = left
        if thumbnail is not None:
            self.create_image(
                S(x0), S(y0), image=thumbnail,
                anchor="nw", tags=ov + ("ov_dataset_image",)
            )
            header_left = x0 + 170 + 26

        # --- هدر: چیپ‌ها + عنوان
        cx = header_left
        for kind, label in (("task", info["task"]), ("level", info["level"]), ("domain", info["domain"])):
            w = self.text_width(label, 17, True) + 40 + (20 if kind == "level" else 0)
            self.create_image(S(cx), S(y0 + 34), image=self._shape(w, 38, 19, "#ffffff", alpha=46),
                              anchor="nw", tags=ov)
            tx = cx + 20
            if kind == "level":
                self.create_image(S(cx + 24), S(y0 + 53), image=self._level_dot(LEVEL_DOT.get(info["level"], "#a99bc0"), 22),
                                  anchor="center", tags=ov)
                tx = cx + 40
            self.text(tx, y0 + 53, label, 17, "white", True, tags=ov)
            cx += w + 14

        title_width = right - header_left - 18
        self.text(
            header_left, y0 + 108,
            self._fit(info["title"], 40, True, title_width),
            40, "white", True, tags=ov
        )

        # --- توضیح (ایتالیک، حداکثر دو خط)
        for i, line in enumerate(self._wrap(info["description"], 22, False, inner_w, 2)):
            self.create_text(S(left), S(y0 + 180 + i * 29), text=line, anchor="w", fill=c("body"),
                             font=(FONT, -S(22), "italic"), tags=ov)

        # --- سه کاشی آماری
        tw, gap, ty = (inner_w - 2 * 14) / 3, 14, y0 + 248
        for i, (value, label, kind) in enumerate(info["stats"]):
            tx = left + i * (tw + gap)
            self.create_image(S(tx), S(ty), image=self._shape(tw, 90, 24, pal["tile_bg"], border=pal["search_border"]),
                              anchor="nw", tags=ov)
            self.create_image(S(tx + 22), S(ty + 22), image=self._info_icon(kind, 46), anchor="nw", tags=ov)
            self.text(tx + 84, ty + 33, value, 32, c("title"), True, tags=ov)
            self.text(tx + 84, ty + 65, label, 18, c("body"), tags=ov)

        # --- جدول مشخصات
        top = y0 + 362
        for i in range(len(rows) + 1):
            self.create_line(S(left), S(top + i * 49), S(right), S(top + i * 49), fill=pal["sep"],
                             width=max(1, S(1)), tags=ov)
        for i, (label, value) in enumerate(rows):
            cy = top + i * 49 + 25
            self.text(left, cy, label, 20, c("body"), tags=ov)
            self.text(left + 200, cy, self._fit(value, 20, True, inner_w - 200), 20, c("title"), True, tags=ov)

        # --- مراحل: Load / Prepare / Train / Predict
        sy = top + len(rows) * 49 + 22
        self.create_image(S(left), S(sy), image=self._shape(inner_w, 56, 28, c("row_bg")), anchor="nw", tags=ov)
        sx = left + 22
        for n, step in enumerate(("Load", "Prepare", "Train", "Predict"), start=1):
            active = n == 1
            circle = (self._shape(32, 32, 16, BTN_LIGHT, BTN_DARK) if active else
                      self._shape(32, 32, 16, pal["step_circle"]))
            self.create_image(S(sx), S(sy + 12), image=circle, anchor="nw", tags=ov)
            self.text(sx + 16, sy + 28, str(n), 16, "white" if active else c("title"), True, anchor="center", tags=ov)
            self.text(sx + 44, sy + 28, step, 20, c("title"), True, tags=ov)
            sx += 44 + self.text_width(step, 20, True) + 34

        # --- دکمه‌ها
        by, bh = sy + 80, 54
        pw = self.text_width("Open predictor", 21, True) + 70
        px = right - pw
        shadow_img, spad = self._glass(pw, bh, 27, "#b4376e", "#5a1f4b", 255, 110, (163, 48, 95), None, 10, 8)
        self.create_image(S(px) - spad, S(by) - spad, image=shadow_img, anchor="nw", tags=ov + ("ov_predict",))
        self.text(px + pw / 2, by + bh / 2, "Open predictor", 21, "white", True, anchor="center",
                  tags=ov + ("ov_predict",))
        self.hover("ov_predict")
        self.tag_bind("ov_predict", "<Button-1>", lambda e, d=info["ds"]: self.on_predict(d))

        cw = 140
        cx0 = px - 20 - cw
        self.create_image(S(cx0), S(by), image=self._shape(cw, bh, 27, c("dlg_bg"), border=pal["search_border"]),
                          anchor="nw", tags=ov + ("ov_close",))
        self.text(cx0 + cw / 2, by + bh / 2, "Close", 21, c("title"), True, anchor="center",
                  tags=ov + ("ov_close",))
        self.hover("ov_close")
        self.tag_bind("ov_close", "<Button-1>", lambda e: self._close_info())

        self.tag_raise("overlay")

    def _level_dot(self, color, size):
        """نقطهٔ رنگی سطح؛ کاملاً تیز (ابَرنمونه‌گیری) با حلقهٔ هالهٔ نرم دورش تا واضح دیده شود."""
        key = ("level_dot", self.k, color, size)
        if key in self._cache:
            return self._cache[key]
        S, ss = self.s(size), 8
        big = S * ss
        img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        r, g, b = _rgb(color)
        c0 = big / 2
        d.ellipse((0, 0, big - 1, big - 1), fill=(r, g, b, 70))                      # هاله
        core = big * 0.30
        d.ellipse((c0 - core, c0 - core, c0 + core, c0 + core), fill=(r, g, b, 255))  # نقطه
        hl = core * 0.38
        d.ellipse((c0 - core * 0.55, c0 - core * 0.6, c0 - core * 0.55 + hl, c0 - core * 0.6 + hl),
                  fill=(255, 255, 255, 150))                                          # برق کوچک
        photo = ImageTk.PhotoImage(img.resize((S, S), Image.LANCZOS))
        self._cache[key] = photo
        return photo

    def _topics_icon(self, size):
        """دیسک سفید + چهار مربع گردگوشه (با رنگ تم)؛ با ابَرنمونه‌گیری رسم می‌شود تا کاملاً تیز باشد."""
        key = ("topics_icon", self.k, size)
        if key in self._cache:
            return self._cache[key]
        S, ss = self.s(size), 6
        big = S * ss
        img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse((0, 0, big - 1, big - 1), fill=(255, 255, 255, 255))

        cell, gap = big * 0.235, big * 0.075
        start = (big - (2 * cell + gap)) / 2
        colors = ((90, 31, 75), (163, 48, 95), (163, 48, 95), (240, 161, 197))   # مربع‌ها: تیره، صورتی، صورتی، روشن
        for i, (row, col) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            x0, y0 = start + col * (cell + gap), start + row * (cell + gap)
            d.rounded_rectangle((x0, y0, x0 + cell, y0 + cell), cell * 0.30, fill=colors[i] + (255,))
        photo = ImageTk.PhotoImage(img.resize((S, S), Image.LANCZOS))
        self._cache[key] = photo
        return photo

    @staticmethod
    def _round_mask(w, h, r, ss=4):
        m = Image.new("L", (w * ss, h * ss), 0)
        ImageDraw.Draw(m).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), r * ss, fill=255)
        return m.resize((w, h), Image.LANCZOS)

    def _decor(self, task, w, h):
        """موج و نقطه‌های تزئینی پشت آیکون کارت (مثل کارت‌های صفحهٔ Algorithms)."""
        key = ("decor", self.k, task, w, h)
        if key in self._cache:
            return self._cache[key]
        W, H, ss = self.s(w), self.s(h), 3
        col = _rgb(TASK_ACCENT.get(task, TASK_DEFAULT)[0])
        light = _rgb(TASK_ACCENT.get(task, TASK_DEFAULT)[1])
        big = (W * ss, H * ss)

        def wave_y(t):
            return H * ss * (0.70 - 0.34 * t - 0.10 * math.sin(t * 7.5))

        poly = [(i, wave_y(i / big[0])) for i in range(0, big[0] + 1, ss * 3)] + [(big[0], big[1]), (0, big[1])]
        mask = Image.new("L", big, 0)
        ImageDraw.Draw(mask).polygon(poly, fill=255)
        fade_x = Image.linear_gradient("L").rotate(90).resize(big)          # چپ شفاف ← راست پررنگ
        fade_y = Image.linear_gradient("L").rotate(180).resize(big)         # پایین شفاف
        # لبهٔ راست هم نرم می‌شود تا مستطیل تیز دیده نشود
        edge_r = Image.linear_gradient("L").rotate(-90).resize(big).point(lambda v: min(255, v * 5))
        mask = ImageChops.multiply(ImageChops.multiply(mask, fade_x), ImageChops.multiply(fade_y, edge_r))
        mask = mask.point(lambda v: min(255, v * 130 // 255))
        layer = Image.new("RGBA", big, col + (0,))
        layer.putalpha(mask)

        d = ImageDraw.Draw(layer)
        rng = random.Random(task)

        def dot(x, y, r, color, a=235):
            d.ellipse((x - r, y - r, x + r, y + r), fill=color + (a,))

        if task == "Regression":
            pts = [(0.22 + 0.15 * i, wave_y(0.22 + 0.15 * i) / big[1] - 0.04) for i in range(5)]
            xy = [(x * big[0], y * big[1]) for x, y in pts]
            d.line(xy, fill=col + (230,), width=ss * 2, joint="curve")
            for x, y in xy:
                dot(x, y, ss * 4.2, (255, 255, 255), 255)
                dot(x, y, ss * 2.6, col, 255)
        elif task == "Classification":
            for i in range(14):
                t = 0.30 + 0.66 * (i / 13) ** 0.8
                dot(t * big[0] + rng.uniform(-14, 14) * ss, wave_y(t) - rng.uniform(4, 30) * ss,
                    ss * rng.uniform(2.0, 4.6), col)
        elif task == "Clustering":
            for cx, cy, color in ((0.62, 0.34, col), (0.90, 0.24, (139, 92, 199))):
                for _ in range(6):
                    dot(cx * big[0] + rng.uniform(-12, 12) * ss, cy * big[1] + rng.uniform(-9, 9) * ss,
                        ss * 2.6, color)
        else:
            for i in range(10):
                t = 0.4 + 0.55 * i / 9
                dot(t * big[0], wave_y(t) - 8 * ss, ss * 3, light)

        photo = ImageTk.PhotoImage(layer.resize((W, H), Image.LANCZOS))
        self._cache[key] = photo
        return photo

    # ---------- جزئیات رسم ----------
    def _d(self, *extra):
        return ("dyn",) + extra

    def _guard(self, fn):
        """اگر پنجرهٔ Topics باز باشد، اولین کلیک فقط آن را می‌بندد."""
        def handler(_event=None):
            if self._popup:
                self._close_popup()
            else:
                fn()
        return handler

    def _clickable(self, tag, fn):
        self.hover(tag)
        self.tag_bind(tag, "<Button-1>", self._guard(fn))

    def _wrap(self, text, size, bold, max_w, max_lines=2):
        words, lines, cur = text.split(), [], ""
        for word in words:
            trial = f"{cur} {word}".strip()
            if self.text_width(trial, size, bold) <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = lines[-1] + "…"
        return [self._fit(l, size, bold, max_w) for l in lines] or [""]

    def _fit(self, text, size, bold, max_w):
        if self.text_width(text, size, bold) <= max_w:
            return text
        while text and self.text_width(text + "…", size, bold) > max_w:
            text = text[:-1]
        return text.rstrip() + "…"

    # ---------- ساخت کامل صفحه (لایهٔ ثابت) ----------
    def _build(self):
        self.delete("all")
        self.config(cursor="")
        self._popup = False
        self._info = None

        bg = self.create_image(0, 0, image=self._bg_photo(), anchor="nw", tags=("bgimg",))
        self.tag_bind("bgimg", "<Button-1>", lambda e: self._close_popup())
        self._navbar()
        self._search_box()
        self._draw_dynamic()

    def _navbar(self):
        """هدر مثل Home؛ آیتم Home حذف شده و «Datasets» فعال است."""
        c, cy = theme.c, HDR_CY
        self.panel(23, HDR_TOP, 1626, HDR_H, 41, *c("hdr"), alpha=255)

        self.pic(88, cy, "ml_studio_logo", 56, crop=True)
        self.text(130, cy, "Machine Learning Prediction Studio", 25, "#ffffff", True)

        # آیتم‌ها دقیقاً وسطِ فاصلهٔ «عنوان ↔ آیکون info» چیده می‌شوند
        items = [("Datasets", "nav_datasets", 38, "datasets"),
                 ("Algorithms", "nav_algorithms", 42, "algorithms")]
        PAD, ICON_GAP, ITEM_GAP = 26, 12, 28           # فاصلهٔ داخلی، آیکون تا متن، بین آیتم‌ها
        widths = [PAD + isz + ICON_GAP + self.text_width(label, 22, True) + PAD for label, _, isz, _ in items]
        group_w = sum(widths) + ITEM_GAP * (len(items) - 1)

        title_end = 130 + self.text_width("Machine Learning Prediction Studio", 25, True)   # انتهای عنوان
        info_left = 1495 - 27                                                               # ابتدای دکمهٔ info
        gap_center = (title_end + info_left) / 2
        # حاشیهٔ خالیِ سمت راست آخرین آیتم دیده نمی‌شود؛ پس مرکز را روی بخش دیدنی می‌گذاریم
        left = max(gap_center - (group_w - PAD) / 2, title_end + 20)

        for (label, ic, isz, page), w in zip(items, widths):
            active = page == "datasets"
            tag = f"nav_{page}"
            cx = left + w / 2
            if active:
                self._pill = self.rounded(w, 50, 25, c("pill"))
                self.create_image(self.s(cx), self.s(cy), image=self._pill)
                self.hover(tag)
                self.tag_bind(tag, "<Button-1>", lambda e, p=page: self.on_navigate(p))
            else:
                self.nav_hover(tag, cx, cy, w, 50, page)
            self.pic(left + PAD + isz / 2, cy, ic, isz, (tag,), crop=True)
            self.text(left + PAD + isz + ICON_GAP, cy, label, 22, c("pill_text") if active else "white", True,
                      tags=(tag,))
            left += w + ITEM_GAP

        for ic, x, page in [("nav_info", 1495, "info"), ("nav_settings", 1590, "settings")]:
            tag = f"nav_{page}"
            self.nav_hover(tag, x, cy, 54, 54, page)
            self.pic(x, cy, ic, 40, (tag,), crop=True)

    def _make_entry(self):
        self.entry = tk.Entry(self, bd=0, relief="flat", highlightthickness=0)
        self.entry.insert(0, "Search datasets")
        self.entry.bind("<FocusIn>", self._entry_focus_in)
        self.entry.bind("<FocusOut>", self._entry_focus_out)
        self.entry.bind("<KeyRelease>", self._entry_changed)

    def _entry_focus_in(self, _e):
        if self._placeholder_on:
            self.entry.delete(0, "end")
            self.entry.config(fg=theme.c("title"))
            self._placeholder_on = False

    def _entry_focus_out(self, _e):
        if not self.entry.get().strip():
            self.entry.delete(0, "end")
            self.entry.insert(0, "Search datasets")
            self.entry.config(fg=theme.c("body"))
            self._placeholder_on = True

    def _entry_changed(self, _e):
        self.page = 1
        self._apply_filters()
        self._draw_dynamic()

    def _search_box(self):
        pal, c = self._pal(), theme.c
        x, y, w, h = MX, FILTER_Y, 330, 46
        cy = y + h / 2

        self._search_bg = self._shape(w, h, h / 2, pal["search_bg"], border=pal["search_border"])
        self.create_image(self.s(x), self.s(y), image=self._search_bg, anchor="nw")

        mx, lw = x + 28, max(2, self.s(2.6))
        self.create_oval(self.s(mx - 9), self.s(cy - 11), self.s(mx + 5), self.s(cy + 3),
                         outline=c("body"), width=lw)
        self.create_line(self.s(mx + 4), self.s(cy + 2), self.s(mx + 12), self.s(cy + 10),
                         fill=c("body"), width=lw, capstyle="round")

        placeholder_fg = c("body") if self._placeholder_on else c("title")
        self.entry.config(font=(FONT, -self.s(20)), bg=pal["search_bg"], fg=placeholder_fg,
                          insertbackground=c("title"))
        self._entry_item = self.create_window(self.s(x + 54), self.s(cy), window=self.entry, anchor="w",
                           width=self.s(w - 76), height=self.s(h - 14))

    # ---------- لایهٔ پویا (فیلترها، کارت‌ها، پایین صفحه) ----------
    def _draw_dynamic(self):
        self.delete("dyn")
        self._draw_filters()

        pages = max(1, math.ceil(len(self.filtered) / PER_PAGE))
        self.page = max(1, min(self.page, pages))
        visible = self.filtered[(self.page - 1) * PER_PAGE: self.page * PER_PAGE]

        if not visible:
            self._draw_empty()
        for i, ds in enumerate(visible):
            row, col = divmod(i, COLS)
            self._draw_card(ds, MX + col * (CARD_W + GAP_X), GRID_Y + row * (CARD_H + GAP_Y), i)

        self._draw_bottom(len(visible), pages)
        if self._popup:
            self._draw_popup()
        if self._info:
            self.tag_raise("overlay")

    # ---- فیلترها ----
    def _draw_filters(self):
        c, pal = theme.c, self._pal()
        x, h, y = MX + 330 + 26, 46, FILTER_Y
        groups = [(["All levels"] + LEVELS, 0), (TASKS, 1)]

        n = 0
        for labels, group in groups:
            for label in labels:
                w = self.text_width(label, 20, True) + 50
                if group == 0:
                    selected = (label == "All levels" and not self.level_filter) or label == self.level_filter
                    command = (lambda: self._set_level(None)) if label == "All levels" else (lambda lv=label: self._set_level(lv))
                else:
                    selected = label == self.task_filter
                    command = lambda t=label: self._toggle_task(t)

                tag = f"pill{n}"
                if selected:
                    img = self._shape(w, h, h / 2, BTN_DARK, BTN_LIGHT)
                else:
                    img = self._shape(w, h, h / 2, pal["pill_off"], alpha=pal["pill_off_a"], border=c("card_border"))
                self.create_image(self.s(x), self.s(y), image=img, anchor="nw", tags=self._d(tag))
                self.text(x + w / 2, y + h / 2, label, 20, "white" if selected else c("title"), True,
                          anchor="center", tags=self._d(tag))
                self._clickable(tag, command)
                x += w + 10
                n += 1
            x += 20

    def _set_level(self, level):
        self.level_filter, self.page = level, 1
        self._apply_filters()
        self._draw_dynamic()

    def _toggle_task(self, task):
        self.task_filter = None if self.task_filter == task else task
        self.page = 1
        self._apply_filters()
        self._draw_dynamic()

    def _set_topic(self, topic):
        self.topic_filter, self.page, self._popup = topic, 1, False
        self._apply_filters()
        self._draw_dynamic()

    def _go_page(self, number):
        self.page = number
        self._draw_dynamic()

    # ---- کارت ----
    def _draw_card(self, ds, x, y, index):
        c, pal = theme.c, self._pal()
        tag = f"card{index}"
        tags = self._d(tag)
        pad = 24
        tint = pal["tint"].get(ds.task, pal["tint_default"])
        accent = TASK_ACCENT.get(ds.task, TASK_DEFAULT)[0]

        if theme.name == "light":
            # در تم روشن سایهٔ کارت‌های دیتاست غیرفعال است.
            sh_rgb, sh_a, sh_blur, sh_off = (0, 0, 0), 0, 0, 0
        else:
            # در تم تاریک سایهٔ کارت‌ها حفظ می‌شود.
            sh_rgb, sh_a, sh_blur, sh_off = c("shadow_rgb"), c("shadow_a"), 14, 10

        def glass(border):
            return self._glass(CARD_W, CARD_H, 30, tint, c("card_end"), c("card_alpha"),
                               sh_a, sh_rgb, border, sh_blur, sh_off)

        normal, pad_px = glass(c("card_border"))
        self.create_image(self.s(x) - pad_px, self.s(y) - pad_px, image=normal, anchor="nw", tags=tags)

        # نشان سطح (عرضش برای جای تزئین لازم است)
        bw = self.text_width(ds.level, 17, True) + 62

        # تزئین فضای بین آیکون و نشان سطح
        gap_x = x + pad + 54 + 14
        gap_w = int(CARD_W - pad - bw - 14 - (pad + 54 + 14))
        if gap_w >= 56:
            self.create_image(self.s(gap_x), self.s(y + 18), image=self._decor(ds.task, gap_w, 58),
                              anchor="nw", tags=tags)

        self.create_image(self.s(x + pad), self.s(y + 20), image=self._task_icon(ds.task, 54), anchor="nw", tags=tags)

        # نشان سطح
        bx2, by = x + CARD_W - pad, y + 47
        level_color = LEVEL_DOT.get(ds.level, "#7a6f8e")
        badge_fill = pal["badge_bg"]
        self.create_image(self.s(bx2 - bw), self.s(by - 19),
                          image=self._shape(bw, 38, 19, badge_fill, alpha=255, border=(level_color, 150)),
                          anchor="nw", tags=tags)
        self.create_image(self.s(bx2 - bw + 21), self.s(by), image=self._level_dot(level_color, 22),
                          anchor="center", tags=tags)
        self.text(bx2 - bw + 38, by, ds.level, 17, c("title"), True, tags=tags)

        # عنوان دیتاست همراه با شناسهٔ Metadata داخل پرانتز
        dataset_title = f"{ds.name} ({str(ds.id).strip()})" if str(ds.id).strip() else ds.name
        lines = self._wrap(dataset_title, 24, True, CARD_W - 2 * pad)
        top = y + 84
        for i, line in enumerate(lines):
            self.text(x + pad, top + i * 29 + 14, line, 24, c("title"), True, tags=tags)
        sub = self._fit(f"{ds.domain} · {ds.task}", 18, False, CARD_W - 2 * pad)
        self.text(x + pad, top + len(lines) * 29 + 16, sub, 18, c("body"), tags=tags)

        # چیپ‌ها و دکمهٔ فلش
        cy = y + CARD_H - 30
        cx = x + pad
        for label in (f"{ds.rows:,} rows", f"{ds.columns:,} cols"):
            w = self.text_width(label, 17, True) + 34
            self.create_image(self.s(cx), self.s(cy - 18), image=self._shape(w, 36, 18, c("row_bg")),
                              anchor="nw", tags=tags)
            self.text(cx + w / 2, cy, label, 17, c("title"), True, anchor="center", tags=tags)
            cx += w + 10

        # دکمهٔ فلش: فقط همین دکمه پنجرهٔ اطلاعات را باز می‌کند (بقیهٔ کارت کلیک‌پذیر نیست)
        ax = x + CARD_W - pad - 24
        atag = f"arrow{index}"
        atags = self._d(tag, atag)
        normal_btn = self._shape(48, 48, 24, BTN_DARK, BTN_LIGHT)
        hover_btn = self._shape(48, 48, 24, "#c23f78", "#8a2a62", border=("#ffffff", 120))
        btn = self.create_image(self.s(ax), self.s(cy), image=normal_btn, anchor="center", tags=atags)
        arrow = dict(fill="white", width=max(2, self.s(3)), capstyle="round", joinstyle="round", tags=atags)
        self.create_line(self.s(ax - 10), self.s(cy), self.s(ax + 10), self.s(cy), **arrow)
        self.create_line(self.s(ax + 3), self.s(cy - 8), self.s(ax + 10), self.s(cy), self.s(ax + 3), self.s(cy + 8), **arrow)

        self.hover(atag,
                   enter=lambda: self.itemconfig(btn, image=hover_btn),
                   leave=lambda: self.itemconfig(btn, image=normal_btn))
        self.tag_bind(atag, "<Button-1>", self._guard(lambda d=ds: self._open_info(d)))

    def _draw_empty(self):
        c = theme.c
        w, h = CONTENT_W, ROWS * CARD_H + GAP_Y
        img, pad = self._glass(w, h, 34, c("card_cols")[0], c("card_end"), c("card_alpha"),
                               c("shadow_a"), c("shadow_rgb"), c("card_border"))
        self.create_image(self.s(MX) - pad, self.s(GRID_Y) - pad, image=img, anchor="nw", tags=self._d())
        if self.all:
            title, hint = "No datasets match your filters", "Try another search, or clear a filter."
        else:
            title, hint = "No datasets yet", "Add a dataset script that writes <Task>/Metadata/<name>.json - it appears here automatically."
        cx, cy = MX + w / 2, GRID_Y + h / 2
        self.text(cx, cy - 22, title, 32, c("title"), True, anchor="center", tags=self._d())
        self.text(cx, cy + 24, hint, 20, c("body"), anchor="center", tags=self._d())

    # ---- ردیف پایین: دستهٔ موضوعی + شمارنده + صفحه‌بندی ----
    def _draw_bottom(self, shown, pages):
        c = theme.c
        self._draw_dock()

        # صفحه‌بندی (راست)
        right, size, gap = MX + CONTENT_W, 46, 10
        tokens = self._page_tokens(pages)
        x = right - (len(tokens) * size + (len(tokens) - 1) * gap) if pages > 1 else right
        pal = self._pal()
        if pages > 1:
            for n, token in enumerate(tokens):
                cy = BOTTOM_CY
                if token == "…":
                    self.text(x + size / 2, cy, "…", 22, c("body"), True, anchor="center", tags=self._d())
                else:
                    active, tag = token == self.page, f"pg{n}"
                    img = (self._shape(size, size, size / 2, BTN_DARK, BTN_LIGHT) if active else
                           self._shape(size, size, size / 2, pal["pill_off"], alpha=pal["pill_off_a"],
                                       border=c("card_border")))
                    self.create_image(self.s(x), self.s(cy - size / 2), image=img, anchor="nw", tags=self._d(tag))
                    self.text(x + size / 2, cy, str(token), 19, "white" if active else c("title"), True,
                              anchor="center", tags=self._d(tag))
                    self._clickable(tag, lambda t=token: self._go_page(t))
                x += size + gap

        # شمارنده
        counter_right = (right - (len(tokens) * size + (len(tokens) - 1) * gap) - 28) if pages > 1 else right
        self.text(counter_right, BOTTOM_CY, f"Showing {shown} of {len(self.filtered)} datasets", 20, c("title"),
                  anchor="e", tags=self._d())

    def _page_tokens(self, pages):
        if pages <= 7:
            return list(range(1, pages + 1))
        wanted = sorted(n for n in {1, pages, self.page - 1, self.page, self.page + 1} if 1 <= n <= pages)
        tokens = []
        for n in wanted:
            if tokens and n - tokens[-1] > 1:
                tokens.append("…")
            tokens.append(n)
        return tokens

    # ---- دستهٔ موضوعی (پایین-چپ) ----
    def _draw_dock(self):
        c = theme.c
        value = self.topic_filter or "All topics"
        if self.topic_filter:
            value += f" · {sum(1 for d in self.all if self.topic_filter in d.topics)}"
        else:
            value += f" · {len(self._topics())}"

        tw = max(self.text_width("Topics", 22, True), self.text_width(value, 17, False))
        clear_w = 50 if self.topic_filter else 0
        w, h = 24 + 46 + 16 + tw + 22 + 22 + clear_w + 16, 66
        x, y, cy = MX, BOTTOM_CY - h / 2, BOTTOM_CY

        tag = "dock"
        self.create_image(self.s(x), self.s(y), image=self._shape(w, h, h / 2, BTN_DARK, BTN_LIGHT,
                                                                   border=("#ffffff", 70)),
                          anchor="nw", tags=self._d(tag))

        # آیکون دسته‌بندی (واضح و تیز)
        ix, iy = x + 24 + 23, cy
        self.create_image(self.s(ix), self.s(iy), image=self._topics_icon(46), anchor="center", tags=self._d(tag))

        tx = x + 24 + 46 + 16
        self.text(tx, cy - 12, "Topics", 22, "white", True, tags=self._d(tag))
        self.text(tx, cy + 15, value, 17, "#f1e4f7", tags=self._d(tag))

        # فلش کوچک
        ax = x + w - 28 - clear_w
        up = -1 if not self._popup else 1
        self.create_polygon(self.s(ax - 8), self.s(cy + 4 * up), self.s(ax + 8), self.s(cy + 4 * up),
                            self.s(ax), self.s(cy - 5 * up), fill="white", outline="", tags=self._d(tag))
        self._clickable(tag, self._toggle_popup)

        if self.topic_filter:
            kx = x + w - 34
            self.create_oval(self.s(kx - 16), self.s(cy - 16), self.s(kx + 16), self.s(cy + 16),
                             fill="#ffffff", outline="", tags=self._d("dockclear"))
            k = 6
            for a, b in (((-k, -k), (k, k)), ((-k, k), (k, -k))):
                self.create_line(self.s(kx + a[0]), self.s(cy + a[1]), self.s(kx + b[0]), self.s(cy + b[1]),
                                 fill=BTN_LIGHT, width=max(2, self.s(2.6)), capstyle="round",
                                 tags=self._d("dockclear"))
            self._clickable("dockclear", lambda: self._set_topic(None))

        self._dock_top = y

    def _toggle_popup(self):
        self._popup = not self._popup
        self._draw_dynamic()

    def _close_popup(self):
        if self._popup:
            self._popup = False
            self._draw_dynamic()

    def _draw_popup(self):
        c, pal = theme.c, self._pal()
        topics = [("All topics", len(self.all))] + self._topics()
        inner_w = 880
        pad, chip_h, gap = 26, 48, 12

        # چیدمان چیپ‌ها
        rows, row, rx = [], [], 0
        for label, count in topics:
            w = self.text_width(label, 19, True) + self.text_width(str(count), 16, True) + 66
            if rx + w > inner_w and row:
                rows.append(row)
                row, rx = [], 0
            row.append((label, count, w, rx))
            rx += w + gap
        rows.append(row)
        hidden = 0
        if len(rows) > 6:
            hidden = sum(len(r) for r in rows[6:])
            rows = rows[:6]

        ph = 74 + len(rows) * (chip_h + gap) + (24 if hidden else 0) + 10
        pw = inner_w + 2 * pad
        px, py = MX, self._dock_top - 16 - ph

        img, gpad = self._glass(pw, ph, 36, c("card_cols")[0], c("card_end"), pal["popup_a"],
                                c("shadow_a") + 30, c("shadow_rgb"), c("card_border"))
        self.create_image(self.s(px) - gpad, self.s(py) - gpad, image=img, anchor="nw", tags=self._d("popup"))
        self.text(px + pad, py + 36, "Browse by topic", 26, c("title"), True, anchor="w", tags=self._d("popup"))
        self.text(px + pw - pad, py + 36, "Click a topic to show its datasets", 17, c("body"), anchor="e",
                  tags=self._d("popup"))

        for r_i, row in enumerate(rows):
            for label, count, w, rx in row:
                x, y = px + pad + rx, py + 70 + r_i * (chip_h + gap)
                selected = (label == "All topics" and not self.topic_filter) or label == self.topic_filter
                tag = f"topic_{r_i}_{int(rx)}"
                chip = (self._shape(w, chip_h, chip_h / 2, BTN_DARK, BTN_LIGHT) if selected else
                        self._shape(w, chip_h, chip_h / 2, c("row_bg"), border=c("card_border")))
                self.create_image(self.s(x), self.s(y), image=chip, anchor="nw", tags=self._d("popup", tag))
                self.create_oval(self.s(x + 16), self.s(y + chip_h / 2 - 5), self.s(x + 26), self.s(y + chip_h / 2 + 5),
                                 fill="white" if selected else c("accent"), outline="", tags=self._d("popup", tag))
                self.text(x + 36, y + chip_h / 2, label, 19, "white" if selected else c("title"), True,
                          tags=self._d("popup", tag))
                self.text(x + w - 16, y + chip_h / 2, str(count), 16, "#f1e4f7" if selected else c("accent"), True,
                          anchor="e", tags=self._d("popup", tag))
                self.hover(tag)
                self.tag_bind(tag, "<Button-1>",
                              lambda e, t=(None if label == "All topics" else label): self._set_topic(t))

        if hidden:
            self.text(px + pad, py + ph - 26, f"+ {hidden} more topics", 16, c("body"), anchor="w",
                      tags=self._d("popup"))
