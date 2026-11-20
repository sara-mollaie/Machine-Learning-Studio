import math
import random
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageTk

from UserInterface.HomeModule import HomeModule, BASE, _rgb, HDR_TOP, HDR_H, HDR_CY
from UserInterface.Theme import theme, FONT

BG_LIGHT = BASE / "assets" / "page_images" / "algorithms_background_light.png"
BG_DARK = BASE / "assets" / "page_images" / "algorithms_background_dark.png"

DESIGN_W, DESIGN_H = 1672, 941          # اندازهٔ صفحه؛ مستقل از رزولوشن فایل پس‌زمینه

# ---- محتوای سه کارت (مختصات: پیکسل طراحی ۱۶۷۲×۹۴۱، دقیقاً مثل عکس مرجع) ----
CATEGORIES = [
    dict(task="Regression", subtitle="Predict a number",
         algorithms=["Linear", "Ridge", "Random Forest", "Gradient Boosting", "SVR"],
         metrics=["R²", "MAE", "MSE", "RMSE"],
         x=82, y=280, w=474, h=590, featured=False,
         tile=("#4f35d6", "#8b5cf6"), accent="#8b5cf6", button=("#4b2a8a", "#2f1760"),
         glow="#a88bf0", glow_pos="bl", shadow=(110, 70, 170)),
    dict(task="Classification", subtitle="Predict a category",
         algorithms=["KNN", "Logistic", "Decision Tree", "Random Forest", "SVM", "Naive Bayes"],
         metrics=["Accuracy", "F1", "Confusion matrix"],
         x=599, y=280, w=474, h=590, featured=False,
         tile=("#ee4f97", "#f78db8"), accent="#ec4899", button=("#c0267c", "#7a1d63"),
         glow="#f26eae", glow_pos="bl", shadow=(235, 80, 150)),
    dict(task="Clustering", subtitle="Find natural groups",
         algorithms=["K-Means"],
         metrics=["Silhouette"],
         x=1116, y=280, w=474, h=590, featured=False,
         tile=("#13c3b0", "#3ddccb"), accent="#14b8a6", button=("#14b8a6", "#176c70"),
         glow="#2ad6c0", glow_pos="br", shadow=(60, 160, 150)),
]

# موقعیت‌های مطلق محتوا (برای هر سه کارت یکسان است)
DECOR_Y, DECOR_H = 314, 96
TILE_Y, TILE_SIZE = 314, 96
TITLE_Y, SUBTITLE_Y = 459, 500
ALGORITHMS_LABEL_Y = 552
CHIP_ROWS_Y = (592, 634)
CHIP_H = 40
METRICS_LABEL_Y = 692
METRIC_CHIPS_Y = 732
BUTTON_Y, BUTTON_H = 792, 64

PALETTE = {
    "light": {
        "card": {"Regression": ("#f3eefd", "#fbf9ff"), "Classification": ("#fdeef6", "#fffafc"),
                 "Clustering": ("#e8f7f5", "#fafefe")},
        "card_alpha": 238,
        "decor": {"Regression": "#ebe3fb", "Classification": "#fde2ee", "Clustering": "#dcf2ef"},
        "chip": {"Regression": "#efe9fa", "Classification": "#fbe8f1", "Clustering": "#e5f5f3"},
        "subtitle": "#6f6a7c",
        "border": ("#ffffff", 230),
        "glow_alpha": 150,
        "shadow_alpha": 60,
    },
    "dark": {
        "card": {"Regression": ("#2a1a47", "#1b1030"), "Classification": ("#3a1b42", "#1b1030"),
                 "Clustering": ("#16363a", "#1b1030")},
        "card_alpha": 242,
        "decor": {"Regression": "#35235a", "Classification": "#4a2350", "Clustering": "#1f4a4b"},
        "chip": {"Regression": "#3a2866", "Classification": "#4c2a5c", "Clustering": "#235252"},
        "subtitle": None,                      # در تم تیره از رنگ body تم استفاده می‌شود
        "border": ("#7d5bc0", 120),
        "glow_alpha": 95,
        "shadow_alpha": 130,
    },
}


class AlgorithmsModule(HomeModule):
    """صفحهٔ Algorithms؛ ابزارهای رسم (panel, text, pic, nav_hover ...) را از HomeModule به ارث می‌برد."""

    def __init__(self, master, scale=1.0, on_navigate=None, on_explore=None):
        # نکته: عمداً HomeModule.__init__ صدا زده نمی‌شود (پس‌زمینهٔ Home را بارگذاری می‌کند)
        self._light = self._load_background(BG_LIGHT)
        self.k = scale
        self.W, self.H = int(DESIGN_W * scale), int(DESIGN_H * scale)
        tk.Canvas.__init__(self, master, width=self.W, height=self.H, highlightthickness=0, bd=0)

        self.on_navigate = on_navigate or (lambda page: print("goto", page))
        self.on_explore = on_explore or self._explore_soon
        self._refs, self._bgs, self._cache = {}, {}, {}
        self.selected_task = None

        theme.subscribe(self._build)
        self._build()

    def on_show(self):
        """هنگام باز شدن صفحه صدا زده می‌شود (فعلاً کاری لازم نیست)."""

    def destroy(self):
        theme.unsubscribe(self._build)
        super().destroy()

    # ---------- پس‌زمینه ----------
    @staticmethod
    def _load_background(path):
        if path.exists():
            return Image.open(path).convert("RGBA")
        img = Image.new("RGBA", (DESIGN_W, DESIGN_H))
        d = ImageDraw.Draw(img)
        for y in range(DESIGN_H):
            t = y / (DESIGN_H - 1)
            d.line([(0, y), (DESIGN_W, y)], fill=(int(249 - 22 * t), int(243 - 30 * t), int(254 - 8 * t), 255))
        return img

    def _bg_photo(self):
        name = theme.name
        if name not in self._bgs:
            img = self._load_background(BG_DARK) if name == "dark" and BG_DARK.exists() else self._light
            self._bgs[name] = ImageTk.PhotoImage(img.resize((self.W, self.H), Image.LANCZOS))
        return self._bgs[name]

    def _pal(self):
        return PALETTE[theme.name]

    def _explore_soon(self, task):
        messagebox.showinfo("Algorithms", f"The {task} section will be available soon.",
                            parent=self.winfo_toplevel())

    def _select_card(self, task):
        """انتخاب کارت؛ کلیک دوباره روی همان کارت، انتخاب را لغو می‌کند."""
        self.selected_task = None if self.selected_task == task else task
        self._build()

    def _explore_card(self, task):
        """با کلیک روی Browse، کارت مربوطه انتخاب و سپس عملیات Explore اجرا می‌شود."""
        if self.selected_task != task:
            self.selected_task = task
            self._build()
        self.on_explore(task)

    # =====================================================
    # PIL helpers (همه با کش)
    # =====================================================
    @staticmethod
    def _lerp(c1, c2, t):
        return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

    def _mask(self, w, h, r, ss=3):
        m = Image.new("L", (w * ss, h * ss), 0)
        ImageDraw.Draw(m).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), r * ss, fill=255)
        return m.resize((w, h), Image.LANCZOS)

    def _ring(self, w, h, r, width, color, alpha, ss=3):
        ring = Image.new("L", (w * ss, h * ss), 0)
        ImageDraw.Draw(ring).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), r * ss, outline=255, width=width * ss)
        ring = ring.resize((w, h), Image.LANCZOS)
        edge = Image.new("RGBA", (w, h), _rgb(color) + (0,))
        edge.putalpha(ring.point(lambda v: v * alpha // 255))
        return edge

    def _card_image(self, cat):
        """کارت شیشه‌ای: سایه + گرادیان + درخشش رنگی گوشه + حاشیه (کارت ویژه: حاشیهٔ صورتی)."""
        selected = self.selected_task == cat["task"]
        key = ("card", self.k, theme.name, cat["task"], selected)
        if key in self._cache:
            return self._cache[key]
        S, pal = self.s, self._pal()
        dark = theme.name == "dark"
        W, H = S(cat["w"]), S(cat["h"])
        R = S(62 if cat["featured"] else 58)
        pad = S(70)
        c1, c2 = (_rgb(c) for c in pal["card"][cat["task"]])

        canvas = Image.new("RGBA", (W + 2 * pad, H + 2 * pad), (0, 0, 0, 0))

        # سایهٔ بیرونی کارت فقط در تم تاریک فعال است.
        if dark:
            sh_rgb = (0, 0, 0)
            sh_a = pal["shadow_alpha"]
            sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            ImageDraw.Draw(sh).rounded_rectangle(
                (pad, pad + S(18), pad + W, pad + S(18) + H),
                R,
                fill=sh_rgb + (sh_a,)
            )
            canvas.alpha_composite(sh.filter(ImageFilter.GaussianBlur(S(28))))

        # بدنه: گرادیان عمودی
        body = Image.new("RGBA", (W, H))
        d = ImageDraw.Draw(body)
        for y in range(H):
            d.line([(0, y), (W, y)], fill=self._lerp(c1, c2, y / max(H - 1, 1)) + (pal["card_alpha"],))

        # درخشش رنگی گوشهٔ پایین
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        gx = 0 if cat["glow_pos"] == "bl" else W
        ImageDraw.Draw(glow).ellipse((gx - int(W * 0.50), H - int(H * 0.26), gx + int(W * 0.50), H + int(H * 0.26)),
                                     fill=_rgb(cat["glow"]) + (pal["glow_alpha"],))
        body.alpha_composite(glow.filter(ImageFilter.GaussianBlur(S(38))))

        body.putalpha(ImageChops.multiply(body.getchannel("A"), self._mask(W, H, R)))
        canvas.alpha_composite(body, (pad, pad))

        # حاشیه فقط برای کارت انتخاب‌شده نمایش داده می‌شود.
        if selected:
            accent = cat["accent"]
            outer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            outer.alpha_composite(
                self._ring(W, H, R, S(6), accent, 155),
                (pad, pad)
            )
            canvas.alpha_composite(outer.filter(ImageFilter.GaussianBlur(S(8))))
            canvas.alpha_composite(
                self._ring(W, H, R, max(2, S(3)), accent, 255),
                (pad, pad)
            )
        else:
            border = pal["border"]
            canvas.alpha_composite(
                self._ring(W, H, R, max(2, S(2)), border[0], border[1]),
                (pad, pad)
            )

        result = (ImageTk.PhotoImage(canvas), pad)
        self._cache[key] = result
        return result

    def _decor_image(self, cat, w, h):
        """نوار تزئینی بالای کارت: موج + نمودار/نقطه‌ها مخصوص هر نوع الگوریتم."""
        key = ("decor", self.k, theme.name, cat["task"], w, h)
        if key in self._cache:
            return self._cache[key]
        S, pal, ss = self.s, self._pal(), 3
        W, H = S(w), S(h)
        big = (W * ss, H * ss)
        accent = _rgb(cat["accent"])
        task = cat["task"]

        layer = Image.new("RGBA", big, _rgb(pal["decor"][task]) + (255,))

        def wave_y(t):
            return big[1] * (0.82 - 0.46 * t - 0.09 * math.sin(t * 7.2))

        poly = [(i, wave_y(i / big[0])) for i in range(0, big[0] + 1, ss * 3)] + [(big[0], big[1]), (0, big[1])]
        wave = Image.new("L", big, 0)
        ImageDraw.Draw(wave).polygon(poly, fill=255)
        fade = Image.linear_gradient("L").rotate(90).resize(big)                # چپ شفاف ← راست پررنگ
        wave = ImageChops.multiply(wave, fade).point(lambda v: v * 150 // 255)
        wave_layer = Image.new("RGBA", big, accent + (0,))
        wave_layer.putalpha(wave)
        layer.alpha_composite(wave_layer)

        d = ImageDraw.Draw(layer)
        rng = random.Random(task)

        def dot(x, y, r, color, a=235):
            d.ellipse((x - r, y - r, x + r, y + r), fill=color + (a,))

        if task == "Regression":
            pts = [(0.30 + 0.095 * i, wave_y(0.30 + 0.095 * i) / big[1] - 0.05) for i in range(8)]
            xy = [(x * big[0], y * big[1]) for x, y in pts]
            d.line(xy, fill=accent + (235,), width=ss * 3, joint="curve")
            for i, (x, y) in enumerate(xy):
                if i in (1, 4, 7):
                    dot(x, y, ss * 7, (255, 255, 255), 255)
                    dot(x, y, ss * 4.6, accent, 255)
                else:
                    dot(x, y, ss * 3.2, accent, 220)
            for _ in range(7):                                                 # نقطه‌های پراکندهٔ چپ
                t = rng.uniform(0.22, 0.46)
                dot(t * big[0], wave_y(t) - rng.uniform(2, 20) * ss, ss * rng.uniform(2, 3.6), accent, 200)
        elif task == "Classification":
            for i in range(38):
                t = 0.30 + 0.66 * (i / 37) ** 0.8
                dot(t * big[0] + rng.uniform(-18, 18) * ss, wave_y(t) - rng.uniform(4, 40) * ss,
                    ss * rng.uniform(2.4, 6.4), accent, rng.randint(170, 240))
        else:
            for cx, cy, color, n in ((0.64, 0.50, accent, 15), (0.93, 0.34, (139, 92, 199), 11)):
                for _ in range(n):
                    x = cx * big[0] + rng.uniform(-30, 30) * ss
                    y = cy * big[1] + rng.uniform(-24, 24) * ss
                    r = ss * rng.uniform(3.2, 5.2)
                    d.ellipse((x - r, y - r, x + r, y + r), outline=color + (240,), width=ss * 2)

        layer.putalpha(ImageChops.multiply(layer.getchannel("A"), self._mask(W, H, S(40), ss).resize(big)))
        photo = ImageTk.PhotoImage(layer.resize((W, H), Image.LANCZOS))
        self._cache[key] = photo
        return photo

    def _tile_image(self, cat):
        """کاشی آیکون ۱۰۴×۱۰۴ با گرادیان و نماد سفید (Regression / Classification / Clustering)."""
        key = ("tile", self.k, theme.name, cat["task"])
        if key in self._cache:
            return self._cache[key]
        S, ss = self.s(TILE_SIZE), 6
        pad = self.s(22)
        big = S * ss
        c1, c2 = _rgb(cat["tile"][0]), _rgb(cat["tile"][1])

        tile = Image.new("RGBA", (big, big))
        d = ImageDraw.Draw(tile)
        for i in range(big * 2):
            d.line([(i, 0), (0, i)], fill=self._lerp(c1, c2, i / (big * 2 - 1)) + (255,))
        m = Image.new("L", (big, big), 0)
        ImageDraw.Draw(m).rounded_rectangle((0, 0, big - 1, big - 1), int(big * 0.26), fill=255)
        tile.putalpha(m)

        # برق ملایم بالای کاشی
        gloss = Image.new("RGBA", (big, big), (0, 0, 0, 0))
        ImageDraw.Draw(gloss).ellipse((-big * 0.2, -big * 0.55, big * 1.2, big * 0.45), fill=(255, 255, 255, 46))
        gloss.putalpha(ImageChops.multiply(gloss.getchannel("A"), m))
        tile.alpha_composite(gloss)

        g = ImageDraw.Draw(tile)
        w, white = max(6, int(big * 0.058)), (255, 255, 255, 255)

        def ring(x, y, r):
            g.ellipse(((x - r) * big, (y - r) * big, (x + r) * big, (y + r) * big), outline=white, width=w)

        task = cat["task"]
        if task == "Regression":
            g.line([(0.27 * big, 0.25 * big), (0.27 * big, 0.73 * big), (0.76 * big, 0.73 * big)],
                   fill=white, width=w, joint="curve")
            pts = [(0.34, 0.62), (0.47, 0.48), (0.58, 0.57), (0.72, 0.33)]
            g.line([(x * big, y * big) for x, y in pts], fill=white, width=w, joint="curve")
            for x, y in pts[1:]:
                ring(x, y, 0.048)
        elif task == "Classification":
            for x, y in ((0.34, 0.34), (0.66, 0.34), (0.34, 0.66), (0.66, 0.66)):
                ring(x, y, 0.118)
        else:
            for x, y in ((0.5, 0.31), (0.32, 0.65), (0.68, 0.65)):
                ring(x, y, 0.128)

        tile = tile.resize((S, S), Image.LANCZOS)

        # سایهٔ زیر آیکون فقط در تم تاریک نمایش داده می‌شود.
        # در تم روشن، خود آیکون بدون سایه رسم می‌شود؛ فضای شفاف اطراف
        # حفظ شده تا موقعیت و اندازهٔ آیکون تغییر نکند.
        canvas = Image.new("RGBA", (S + 2 * pad, S + 2 * pad), (0, 0, 0, 0))

        if theme.name == "dark":
            sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            ImageDraw.Draw(sh).rounded_rectangle(
                (pad, pad + self.s(10), pad + S, pad + self.s(10) + S),
                int(S * 0.26),
                fill=c1 + (110,)
            )
            canvas.alpha_composite(sh.filter(ImageFilter.GaussianBlur(self.s(9))))

        canvas.alpha_composite(tile, (pad, pad))

        result = (ImageTk.PhotoImage(canvas), pad)
        self._cache[key] = result
        return result

    def _pill_image(self, w, h, color, alpha=255):
        key = ("pill", self.k, w, h, color, alpha)
        if key in self._cache:
            return self._cache[key]
        W, H = self.s(w), self.s(h)
        img = Image.new("RGBA", (W, H), _rgb(color) + (255,))
        img.putalpha(self._mask(W, H, H // 2).point(lambda v: v * alpha // 255))
        photo = ImageTk.PhotoImage(img)
        self._cache[key] = photo
        return photo

    def _dot_image(self, color, size):
        key = ("dot", self.k, color, size)
        if key in self._cache:
            return self._cache[key]
        S, ss = self.s(size), 8
        img = Image.new("RGBA", (S * ss, S * ss), (0, 0, 0, 0))
        ImageDraw.Draw(img).ellipse((0, 0, S * ss - 1, S * ss - 1), fill=_rgb(color) + (255,))
        photo = ImageTk.PhotoImage(img.resize((S, S), Image.LANCZOS))
        self._cache[key] = photo
        return photo

    def _button_image(self, w, h, c1, c2, hover=False):
        key = ("button", self.k, w, h, c1, c2, hover)
        if key in self._cache:
            return self._cache[key]
        S = self.s
        W, H, pad = S(w), S(h), S(40)
        a, b = _rgb(c1), _rgb(c2)
        if hover:                                               # کمی روشن‌تر هنگام hover
            a = tuple(min(255, int(v * 1.22) + 8) for v in a)
            b = tuple(min(255, int(v * 1.22) + 8) for v in b)

        canvas = Image.new("RGBA", (W + 2 * pad, H + 2 * pad), (0, 0, 0, 0))
        sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle((pad, pad + S(10), pad + W, pad + S(10) + H), H // 2,
                                             fill=_rgb(c2) + (120,))
        canvas.alpha_composite(sh.filter(ImageFilter.GaussianBlur(S(13))))

        body = Image.new("RGBA", (W, H))
        d = ImageDraw.Draw(body)
        for x in range(W):
            d.line([(x, 0), (x, H)], fill=self._lerp(a, b, x / max(W - 1, 1)) + (255,))
        body.putalpha(self._mask(W, H, H // 2))
        canvas.alpha_composite(body, (pad, pad))

        result = (ImageTk.PhotoImage(canvas), pad)
        self._cache[key] = result
        return result

    def _star_image(self, cat):
        key = ("star", self.k, theme.name, cat["task"])
        if key in self._cache:
            return self._cache[key]
        w, h, ss = 83, 47, 4
        W, H, pad = self.s(w), self.s(h), self.s(28)
        big = (W * ss, H * ss)

        canvas = Image.new("RGBA", (W + 2 * pad, H + 2 * pad), (0, 0, 0, 0))
        sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle((pad, pad + self.s(6), pad + W, pad + self.s(6) + H), H // 2,
                                             fill=(236, 72, 153, 130))
        canvas.alpha_composite(sh.filter(ImageFilter.GaussianBlur(self.s(9))))

        body = Image.new("RGBA", (W, H))
        d = ImageDraw.Draw(body)
        star_c1, star_c2 = (_rgb(color) for color in cat["tile"])
        for x in range(W):
            d.line(
                [(x, 0), (x, H)],
                fill=self._lerp(star_c1, star_c2, x / max(W - 1, 1)) + (255,)
            )
        body.putalpha(self._mask(W, H, H // 2))

        star = Image.new("RGBA", big, (0, 0, 0, 0))
        cx, cy, ro, ri = big[0] / 2, big[1] / 2 + big[1] * 0.02, big[1] * 0.34, big[1] * 0.145
        pts = []
        for i in range(10):
            ang = -math.pi / 2 + i * math.pi / 5
            r = ro if i % 2 == 0 else ri
            pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
        ImageDraw.Draw(star).polygon(pts, fill=(255, 255, 255, 255))
        body.alpha_composite(star.resize((W, H), Image.LANCZOS))
        canvas.alpha_composite(body, (pad, pad))

        result = (ImageTk.PhotoImage(canvas), pad)
        self._cache[key] = result
        return result

    # =====================================================
    # ساخت صفحه
    # =====================================================
    def _build(self):
        self.delete("all")
        self.config(cursor="")
        self.create_image(0, 0, image=self._bg_photo(), anchor="nw")

        self._navbar()
        for cat in CATEGORIES:
            if not cat["featured"]:
                self._draw_card(cat)
        for cat in CATEGORIES:                                  # کارت ویژه آخر رسم می‌شود تا روی بقیه بیفتد
            if cat["featured"]:
                self._draw_card(cat)

    def _navbar(self):
        """هدر مثل Home؛ آیتم Home حذف شده و «Algorithms» فعال است."""
        c, cy = theme.c, HDR_CY
        self.panel(23, HDR_TOP, 1626, HDR_H, 41, *c("hdr"), alpha=255)

        self.pic(88, cy, "ml_studio_logo", 56, crop=True)
        self.text(130, cy, "Machine Learning Prediction Studio", 25, "#ffffff", True)

        items = [("Datasets", "nav_datasets", 38, "datasets"),
                 ("Algorithms", "nav_algorithms", 42, "algorithms")]
        PAD, ICON_GAP, ITEM_GAP = 26, 12, 28
        widths = [PAD + isz + ICON_GAP + self.text_width(label, 22, True) + PAD for label, _, isz, _ in items]
        group_w = sum(widths) + ITEM_GAP * (len(items) - 1)

        title_end = 130 + self.text_width("Machine Learning Prediction Studio", 25, True)
        info_left = 1495 - 27
        gap_center = (title_end + info_left) / 2
        left = max(gap_center - (group_w - PAD) / 2, title_end + 20)

        for (label, ic, isz, page), w in zip(items, widths):
            active = page == "algorithms"
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

    def _draw_card(self, cat):
        c, pal = theme.c, self._pal()
        S = self.s
        x, y, w, task = cat["x"], cat["y"], cat["w"], cat["task"]
        tag = f"card_{task}"

        # بدنهٔ کارت؛ حاشیه فقط در حالت انتخاب‌شده فعال است
        selected = self.selected_task == task
        img, pad = self._card_image(cat)
        self.create_image(S(x) - pad, S(y) - pad, image=img, anchor="nw", tags=tag)

        # تزئین + کاشی آیکون
        self.create_image(S(x + 28), S(DECOR_Y), image=self._decor_image(cat, w - 52, DECOR_H), anchor="nw", tags=tag)
        tile, tpad = self._tile_image(cat)
        self.create_image(S(x + 28) - tpad, S(TILE_Y) - tpad, image=tile, anchor="nw", tags=tag)

        # عنوان، زیرعنوان و توضیح کوتاه؛ ظاهر و فونت‌های اصلی حفظ شده‌اند
        self.text(x + 42, TITLE_Y, task, 44, c("title"), True, tags=tag)
        subtitle_color = pal["subtitle"] or c("body")
        self.text(x + 42, SUBTITLE_Y, cat["subtitle"], 22, subtitle_color, tags=tag)

        # عنوان بخش الگوریتم‌ها
        self.text(x + 28, ALGORITHMS_LABEL_Y, "Algorithms", 18, c("title"), True, tags=tag)

        # چیپ‌های الگوریتم‌ها (همان استایل، اندازه و رنگ اصلی)
        inner_w = w - 28 - 10
        chip_bg = pal["chip"][task]
        row, cx = 0, x + 28
        for name in cat["algorithms"]:
            cw = self.text_width(name, 18, True) + 12 + 10 + 8 + 12
            if cx + cw > x + 28 + inner_w and cx > x + 28:
                row, cx = row + 1, x + 28
            if row >= len(CHIP_ROWS_Y):
                break
            cy = CHIP_ROWS_Y[row]
            self.create_image(S(cx), S(cy - CHIP_H / 2), image=self._pill_image(cw, CHIP_H, chip_bg), anchor="nw", tags=tag)
            self.create_image(S(cx + 12 + 5), S(cy), image=self._dot_image(cat["accent"], 10), anchor="center", tags=tag)
            self.text(cx + 12 + 10 + 8, cy, name, 18, c("title"), True, tags=tag)
            cx += cw + 10

        # عنوان معیارهای ارزیابی؛ با همان تایپوگرافی عنوان Algorithms
        self.text(
            x + 28,
            METRICS_LABEL_Y,
            "Evaluation metrics",
            18,
            c("title"),
            True,
            tags=tag,
        )

        # چیپ‌های معیارهای ارزیابی؛ دقیقاً با همان استایل چیپ‌های Algorithms
        row, cx = 0, x + 28
        for name in cat["metrics"]:
            cw = self.text_width(name, 18, True) + 12 + 10 + 8 + 12
            if cx + cw > x + 28 + inner_w and cx > x + 28:
                row, cx = row + 1, x + 28
            if row >= 2:
                break
            cy = METRIC_CHIPS_Y + row * 42
            self.create_image(
                S(cx),
                S(cy - CHIP_H / 2),
                image=self._pill_image(cw, CHIP_H, chip_bg),
                anchor="nw",
                tags=tag,
            )
            self.create_image(
                S(cx + 12 + 5),
                S(cy),
                image=self._dot_image(cat["accent"], 10),
                anchor="center",
                tags=tag,
            )
            self.text(
                cx + 12 + 10 + 8,
                cy,
                name,
                18,
                c("title"),
                True,
                tags=tag,
            )
            cx += cw + 10

        # دکمهٔ Browse؛ شکل، رنگ، اندازه و افکت اصلی حفظ شده‌اند
        bw = w - 112
        bx = x + 56
        n_img, bpad = self._button_image(bw, BUTTON_H, *cat["button"])
        h_img, _ = self._button_image(bw, BUTTON_H, *cat["button"], hover=True)
        btag = f"explore_{task}"
        btn = self.create_image(S(bx) - bpad, S(BUTTON_Y) - bpad, image=n_img, anchor="nw", tags=(btag,))
        label = f"Browse {task.lower()} datasets"
        text_w = self.text_width(label, 19, True)
        group_w = text_w + 22 + 30                                  # متن + فاصله + فلش
        gx = bx + (bw - group_w) / 2
        ay = BUTTON_Y + BUTTON_H / 2
        self.text(gx, ay, label, 19, "white", True, tags=(tag, btag))

        ax = gx + group_w
        arrow = dict(fill="white", width=max(2, S(3)), capstyle="round", joinstyle="round", tags=(btag,))
        self.create_line(S(ax - 30), S(ay), S(ax), S(ay), **arrow)
        self.create_line(S(ax - 9), S(ay - 8), S(ax), S(ay), S(ax - 9), S(ay + 8), **arrow)

        self.hover(btag,
                   enter=lambda: self.itemconfig(btn, image=h_img),
                   leave=lambda: self.itemconfig(btn, image=n_img))
        self.tag_bind(btag, "<Button-1>", lambda e, t=task: self._explore_card(t))

        # تمام سطح کارت (به‌جز دکمهٔ Browse) قابل انتخاب است.
        self.tag_bind(tag, "<Button-1>", lambda e, t=task: self._select_card(t))

        # نشان ستاره فقط روی کارت انتخاب‌شده و با رنگ اختصاصی همان الگوریتم نمایش داده می‌شود.
        if selected:
            star, spad = self._star_image(cat)
            star_y = y - 26
            self.create_image(
                S(x + w / 2 - 83 / 2) - spad,
                S(star_y) - spad,
                image=star,
                anchor="nw",
                tags=(tag,),
            )
