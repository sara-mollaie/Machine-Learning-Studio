"""
DatasetsModule.py  -  صفحهٔ Datasets (هم‌اندازه و هم‌سبک صفحهٔ Home)

* همان کانواس، همان فونت (Theme.FONT) و همان تم Home (روشن/تیره)
* کارت‌ها خودکار از فایل‌های متادیتا ساخته می‌شوند:
      <ریشهٔ پروژه>/<پوشهٔ نوع>/Metadata/*.json      مثلاً Clustering/Metadata/D21_travel_reviews.json
  هر چند ثانیه پوشه دوباره بررسی می‌شود؛ دیتاست تازه = کارت تازه، بدون تغییر کد.
* پایین-چپ: دستهٔ موضوعی (Topics) که از فیلد domain خودکار ساخته می‌شود.

فیلدهای خوانده‌شده از هر JSON:
    id, name, level, domain, task, description, source,
    dataset_statistics.rows, dataset_statistics.columns
"""

import json
import math
import random
import re
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageTk

from UserInterfaceLayer.HomeModule import HomeModule, BASE, _rgb, HDR_TOP, HDR_H, HDR_CY
from UserInterfaceLayer.Theme import theme, FONT

BG_LIGHT = BASE / "assets" / "page_images" / "datasets_background.png"
BG_DARK = BASE / "assets" / "page_images" / "datasets_background_dark.png"

# ---- چیدمان (واحد: پیکسل طراحی ۱۶۷۲×۹۴۱، مثل HomeModule) ----
MX = 64                       # حاشیهٔ چپ محتوا
CONTENT_W = 1544              # عرض محتوا (۶۴ تا ۱۶۰۸)
COLS, ROWS = 4, 2
GAP_X, GAP_Y = 24, 22
CARD_W = (CONTENT_W - GAP_X * (COLS - 1)) // COLS
CARD_H = 226
FILTER_Y = 266
GRID_Y = 332
BOTTOM_CY = 854               # مرکز ردیف پایین (دستهٔ موضوعی + صفحه‌بندی)
PER_PAGE = COLS * ROWS

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
        "popup_a": 255,
    },
    "dark": {
        "tint": {"Regression": "#2a1a47", "Classification": "#3a1b42", "Clustering": "#16363a"},
        "tint_default": "#271a45",
        "search_bg": "#2a1a47", "search_border": ("#4a3566", 255),
        "pill_off": "#2a1a47", "pill_off_a": 235,
        "popup_a": 255,
    },
}

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

    def __init__(self, master, scale=1.0, on_navigate=None, on_open=None, catalog=None, poll_ms=2500):
        # نکته: عمداً HomeModule.__init__ صدا زده نمی‌شود (پس‌زمینهٔ Home را بارگذاری می‌کند)
        self._light = self._load_background(BG_LIGHT)
        self.k = scale
        self.W, self.H = int(self._light.width * scale), int(self._light.height * scale)
        tk.Canvas.__init__(self, master, width=self.W, height=self.H, highlightthickness=0, bd=0)

        self.on_navigate = on_navigate or (lambda page: print("goto", page))
        self.on_open = on_open or (lambda ds: print("open dataset:", ds.name))
        self.catalog = catalog or DatasetCatalog()
        self.poll_ms = poll_ms

        self._refs, self._bgs, self._cache = {}, {}, {}
        self.all, self.filtered = [], []
        self.level_filter = self.task_filter = self.topic_filter = None
        self.page = 1
        self._popup = False
        self._placeholder_on = True
        self._sig = None
        self._poll_job = None

        self._make_entry()
        self._load_data()

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

    def _glass(self, w, h, r, c1, c2, alpha, shadow, shadow_rgb, border):
        key = ("glass", self.k, w, h, r, c1, c2, alpha, shadow, tuple(shadow_rgb), border)
        if key in self._cache:
            return self._cache[key]
        W, H, R, ss = self.s(w), self.s(h), self.s(r), 3
        pad = self.s(34) if shadow else 0
        canvas = Image.new("RGBA", (W + 2 * pad, H + 2 * pad), (0, 0, 0, 0))
        if shadow:
            sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            off = self.s(10)
            ImageDraw.Draw(sh).rounded_rectangle((pad, pad + off, pad + W, pad + off + H), R,
                                                 fill=tuple(shadow_rgb) + (shadow,))
            canvas = Image.alpha_composite(canvas, sh.filter(ImageFilter.GaussianBlur(self.s(14))))
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

        bg = self.create_image(0, 0, image=self._bg_photo(), anchor="nw", tags=("bgimg",))
        self.tag_bind("bgimg", "<Button-1>", lambda e: self._close_popup())
        self._navbar()
        self._search_box()
        self._draw_dynamic()

    def _navbar(self):
        """هدر دقیقاً مثل Home؛ فقط آیتم فعال «Datasets» است."""
        c, cy = theme.c, HDR_CY
        self.panel(23, HDR_TOP, 1626, HDR_H, 41, *c("hdr"), alpha=255)

        self.pic(88, cy, "ml_studio_logo", 56, crop=True)
        self.text(130, cy, "Machine Learning Prediction Studio", 25, "#ffffff", True)

        items = [("Home", "nav_home", 34, 748, 775, "home"),
                 ("Datasets", "nav_datasets", 38, 934, 964, "datasets"),
                 ("Algorithms", "nav_algorithms", 42, 1140, 1173, "algorithms")]
        for label, ic, isz, ix, tx, page in items:
            active = page == "datasets"
            tag = f"nav_{page}"
            left = ix - isz / 2 - 16
            right = tx + self.text_width(label, 22, True) + 18
            if active:
                self._pill = self.rounded(right - left + 12, 50, 25, c("pill"))
                self.create_image(self.s((left + right) / 2), self.s(cy), image=self._pill)
                self.hover(tag)
                self.tag_bind(tag, "<Button-1>", lambda e, p=page: self.on_navigate(p))
            else:
                self.nav_hover(tag, (left + right) / 2, cy, right - left, 50, page)
            self.pic(ix, cy, ic, isz, (tag,), crop=True)
            self.text(tx, cy, label, 22, c("pill_text") if active else "white", True, tags=(tag,))

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
        self.create_window(self.s(x + 54), self.s(cy), window=self.entry, anchor="w",
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

        def glass(border):
            return self._glass(CARD_W, CARD_H, 30, tint, c("card_end"), c("card_alpha"),
                               c("shadow_a"), c("shadow_rgb"), border)

        normal, pad_px = glass(c("card_border"))
        hover, _ = glass((accent, 255))
        body = self.create_image(self.s(x) - pad_px, self.s(y) - pad_px, image=normal, anchor="nw", tags=tags)

        # نشان سطح (عرضش برای جای تزئین لازم است)
        bw = self.text_width(ds.level, 17, True) + 56

        # تزئین: فقط در فاصلهٔ آزاد بین آیکون و نشان سطح
        gap_w = int(CARD_W - pad - bw - 14 - (pad + 54 + 14))
        if gap_w >= 56:
            self.create_image(self.s(x + pad + 54 + 14), self.s(y + 18), image=self._decor(ds.task, gap_w, 58),
                              anchor="nw", tags=tags)
        self.create_image(self.s(x + pad), self.s(y + 20), image=self._task_icon(ds.task, 54), anchor="nw", tags=tags)

        # نشان سطح
        bx2, by = x + CARD_W - pad, y + 47
        self.create_image(self.s(bx2 - bw), self.s(by - 18),
                          image=self._shape(bw, 36, 18, c("row_bg"), alpha=240), anchor="nw", tags=tags)
        self.create_oval(self.s(bx2 - bw + 14), self.s(by - 6), self.s(bx2 - bw + 26), self.s(by + 6),
                         fill=LEVEL_DOT.get(ds.level, "#7a6f8e"), outline="", tags=tags)
        self.text(bx2 - bw + 34, by, ds.level, 17, c("title"), True, tags=tags)

        # عنوان (حداکثر دو خط) و زیرعنوان
        lines = self._wrap(ds.name, 24, True, CARD_W - 2 * pad)
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

        ax = x + CARD_W - pad - 24
        self.create_image(self.s(ax), self.s(cy), image=self._shape(48, 48, 24, BTN_DARK, BTN_LIGHT),
                          anchor="center", tags=tags)
        arrow = dict(fill="white", width=max(2, self.s(3)), capstyle="round", joinstyle="round", tags=tags)
        self.create_line(self.s(ax - 10), self.s(cy), self.s(ax + 10), self.s(cy), **arrow)
        self.create_line(self.s(ax + 3), self.s(cy - 8), self.s(ax + 10), self.s(cy), self.s(ax + 3), self.s(cy + 8), **arrow)

        # تعامل
        self.hover(tag,
                   enter=lambda: self.itemconfig(body, image=hover),
                   leave=lambda: self.itemconfig(body, image=normal))
        self.tag_bind(tag, "<Button-1>", self._guard(lambda d=ds: self.on_open(d)))

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

        # آیکون شبکهٔ ۲×۲
        ix, iy = x + 24 + 23, cy
        self.create_oval(self.s(ix - 23), self.s(iy - 23), self.s(ix + 23), self.s(iy + 23),
                         fill="#ffffff", outline="", tags=self._d(tag))
        for dx in (-6.5, 6.5):
            for dy in (-6.5, 6.5):
                self.create_oval(self.s(ix + dx - 4.2), self.s(iy + dy - 4.2), self.s(ix + dx + 4.2),
                                 self.s(iy + dy + 4.2), fill=BTN_LIGHT, outline="", tags=self._d(tag))

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