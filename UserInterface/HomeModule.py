import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw, ImageFilter

from UserInterface.Theme import theme, FONT

BASE = Path(__file__).resolve().parent.parent
ICONS = BASE / "assets" / "icons"
BG_PATH = BASE / "assets" / "page_images" / "home_background.png"
BG_DARK_PATH = BASE / "assets" / "page_images" / "home_background_dark.png"   # اختیاری (کیفیت بهتر)

# ---- تنظیمات چیدمان (واحد: پیکسل طراحی ۱۶۷۲×۹۴۱) ----
HDR_TOP = 6
HDR_H = 82
HDR_CY = HDR_TOP + HDR_H // 2
HERO_DY = -75
FOOTER_CY = 870
FOOTER_GAP = 18
FOOTER_ICON = 44
HOVER_ALPHA = 70


def _rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def make_dark(img):
    """نسخهٔ تیرهٔ پس‌زمینه: فقط پس‌زمینهٔ روشن تیره می‌شود، بدون حلقهٔ مشکی دور اجسام."""
    try:
        import numpy as np
    except ImportError:
        from PIL import ImageEnhance
        return ImageEnhance.Brightness(img.convert("RGB")).enhance(0.35).convert("RGBA")

    rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32) / 255
    lum = arr @ np.array([.299, .587, .114], dtype=np.float32)
    mx, mn = arr.max(2), arr.min(2)
    sat = (mx - mn) / (mx + 1e-6)

    # وزن «پس‌زمینهٔ روشن»
    w = np.clip((lum - .78) / .18, 0, 1) * np.clip(1 - sat * 1.6, 0, 1)

    # فرسایش + محو: وزن از لبهٔ اجسام فاصله می‌گیرد تا دور مغز و پنل‌ها حلقهٔ سیاه نماند
    W, H = rgb.size
    wm = Image.fromarray((w * 255).astype("uint8"), "L")
    small = wm.resize((W // 2, H // 2), Image.BILINEAR)
    small = small.filter(ImageFilter.MinFilter(7)).filter(ImageFilter.GaussianBlur(3.5))
    wm = small.resize((W, H), Image.LANCZOS)
    w = np.asarray(wm, dtype=np.float32) / 255

    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    dark = np.array([.09, .05, .17], dtype=np.float32) * (1 - yy) + np.array([.15, .08, .26], dtype=np.float32) * yy
    res = (arr * (1 - w[..., None]) + dark * w[..., None]) * 0.92
    return Image.fromarray((np.clip(res, 0, 1) * 255).astype("uint8")).convert("RGBA")


class HomeModule(tk.Canvas):
    def __init__(self, master, scale=1.0, on_navigate=None):
        self._light = Image.open(BG_PATH).convert("RGBA")
        self.k = scale
        self.W, self.H = int(self._light.width * scale), int(self._light.height * scale)
        super().__init__(master, width=self.W, height=self.H, highlightthickness=0, bd=0)
        self.on_navigate = on_navigate or (lambda page: print("goto", page))
        self._refs = {}
        self._bgs = {}

        theme.subscribe(self._build)          # با تغییر تم، صفحه دوباره رسم می‌شود
        self._build()

    # ---------- ابزارها ----------
    def s(self, v):
        return int(round(v * self.k))

    def _bg_photo(self):
        name = theme.name
        if name not in self._bgs:
            if name == "dark":
                img = Image.open(BG_DARK_PATH).convert("RGBA") if BG_DARK_PATH.exists() else make_dark(self._light)
            else:
                img = self._light
            self._bgs[name] = ImageTk.PhotoImage(img.resize((self.W, self.H), Image.LANCZOS))
        return self._bgs[name]

    def icon(self, name, size, crop=False):
        key = (name, size, crop)
        if key not in self._refs:
            img = Image.open(ICONS / f"{name}.png").convert("RGBA")
            if crop:
                alpha = img.getchannel("A").point(lambda v: 255 if v > 40 else 0)
                bbox = alpha.getbbox()
                if bbox:
                    img = img.crop(bbox)
            ratio = self.s(size) / max(img.size)
            img = img.resize((max(1, round(img.width * ratio)), max(1, round(img.height * ratio))),
                             Image.LANCZOS)
            self._refs[key] = ImageTk.PhotoImage(img)
        return self._refs[key]

    def panel(self, x, y, w, h, r, c1, c2=None, alpha=255, shadow=0, shadow_rgb=(78, 35, 120), border=None):
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
        for px in range(W):
            t = px / max(W - 1, 1)
            d.line([(px, 0), (px, H)], fill=tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3)) + (255,))
        mask = Image.new("L", (W * ss, H * ss), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, W * ss - 1, H * ss - 1), R * ss, fill=alpha)
        body.putalpha(mask.resize((W, H), Image.LANCZOS))
        canvas.alpha_composite(body, (pad, pad))

        if border:
            ring = Image.new("L", (W * ss, H * ss), 0)
            ImageDraw.Draw(ring).rounded_rectangle((0, 0, W * ss - 1, H * ss - 1), R * ss,
                                                   outline=255, width=max(2, self.s(2)) * ss // 2)
            ring = ring.resize((W, H), Image.LANCZOS)
            edge = Image.new("RGBA", (W, H), _rgb(border[0]) + (0,))
            edge.putalpha(ring.point(lambda v: v * border[1] // 255))
            canvas.alpha_composite(edge, (pad, pad))

        ph = ImageTk.PhotoImage(canvas)
        self._refs[("panel", x, y)] = ph
        self.create_image(self.s(x) - pad, self.s(y) - pad, image=ph, anchor="nw")

    def rounded(self, w, h, r, c1, c2=None):
        W, H, R, ss = self.s(w), self.s(h), self.s(r), 4
        mask = Image.new("L", (W * ss, H * ss), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, W * ss - 1, H * ss - 1), R * ss, fill=255)
        mask = mask.resize((W, H), Image.LANCZOS)
        a, b = _rgb(c1), _rgb(c2 or c1)
        grad = Image.new("RGB", (W, H))
        d = ImageDraw.Draw(grad)
        for x in range(W):
            t = x / max(W - 1, 1)
            d.line([(x, 0), (x, H)], fill=tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3)))
        grad.putalpha(mask)
        return ImageTk.PhotoImage(grad)

    def glow(self, w, h, r, color="#c9b6f2", alpha=70):
        W, H, R, ss = self.s(w), self.s(h), self.s(r), 4
        mask = Image.new("L", (W * ss, H * ss), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, W * ss - 1, H * ss - 1), R * ss, fill=alpha)
        img = Image.new("RGBA", (W, H), _rgb(color) + (0,))
        img.putalpha(mask.resize((W, H), Image.LANCZOS))
        return ImageTk.PhotoImage(img)

    def text(self, x, y, txt, size, color, bold=False, anchor="w", tags=()):
        return self.create_text(self.s(x), self.s(y), text=txt, anchor=anchor, fill=color, tags=tags,
                                font=(FONT, -self.s(size), "bold" if bold else "normal"))

    def text_width(self, txt, size, bold=False):
        f = tkfont.Font(family=FONT, size=-self.s(size), weight="bold" if bold else "normal")
        return f.measure(txt) / self.k

    def row(self, x, y, parts, size, bold=False):
        f = tkfont.Font(family=FONT, size=-self.s(size), weight="bold" if bold else "normal")
        cx = self.s(x)
        for txt, color in parts:
            self.create_text(cx, self.s(y), text=txt, anchor="w", fill=color, font=f)
            cx += f.measure(txt)

    def pic(self, x, y, name, size, tags=(), crop=False):
        return self.create_image(self.s(x), self.s(y), image=self.icon(name, size, crop), tags=tags)

    def hover(self, tag, enter=None, leave=None):
        self.tag_bind(tag, "<Enter>", lambda e: (self.config(cursor="hand2"), enter and enter()))
        self.tag_bind(tag, "<Leave>", lambda e: (self.config(cursor=""), leave and leave()))

    def nav_hover(self, tag, cx, cy, w, h, page):
        r = min(w, h) / 2
        idle = self.glow(w, h, r, alpha=1)
        on = self.glow(w, h, r, alpha=HOVER_ALPHA)
        self._refs[(tag, "idle")], self._refs[(tag, "on")] = idle, on
        item = self.create_image(self.s(cx), self.s(cy), image=idle, tags=(tag,))
        self.hover(tag,
                   enter=lambda: self.itemconfig(item, image=on),
                   leave=lambda: self.itemconfig(item, image=idle))
        self.tag_bind(tag, "<Button-1>", lambda e, p=page: self.on_navigate(p))

    # ---------- ساخت صفحه ----------
    def _build(self):
        self.delete("all")
        self.config(cursor="")
        self.create_image(0, 0, image=self._bg_photo(), anchor="nw")
        self._panels()
        self._navbar()
        self._hero()
        self._cards()
        self._footer()

    def _panels(self):
        c = theme.c
        self.panel(23, HDR_TOP, 1626, HDR_H, 41, *c("hdr"), alpha=255)
        for x, w, col in zip((64, 436, 803, 1221), (344, 343, 394, 404), c("card_cols")):
            self.panel(x, 642, w, 131, 30, col, c("card_end"), alpha=c("card_alpha"),
                       shadow=c("shadow_a"), shadow_rgb=c("shadow_rgb"), border=c("card_border"))
        self.panel(23, 823, 1626, 94, 38, *c("ftr"), alpha=250, shadow=110, shadow_rgb=c("shadow_rgb"))

    def _navbar(self):
        cy = HDR_CY
        self.pic(88, cy, "ml_studio_logo", 56, crop=True)
        self.text(130, cy, "Machine Learning Prediction Studio", 25, "#ffffff", True)

        for ic, x, page in [("nav_info", 1495, "info"), ("nav_settings", 1590, "settings")]:
            tag = f"nav_{page}"
            self.nav_hover(tag, x, cy, 54, 54, page)
            self.pic(x, cy, ic, 40, (tag,), crop=True)

    def _hero(self):
        dy, c = HERO_DY, theme.c
        self.row(78, 290 + dy, [("ML ", c("title")), ("Studio", c("accent"))], 96, True)
        self.text(78, 368 + dy, "A Unified Platform for Machine Learning", 30, c("title"), True)
        for i, line in enumerate(["Explore 32 datasets, train and evaluate models, compare",
                                  "algorithms, and build real-world machine learning",
                                  "projects in one intuitive platform."]):
            self.text(79, 422 + dy + i * 30.5, line, 22, c("body"))

        self._btn_n = self.rounded(332, 74, 37, "#a3305f", "#5a1f4b")
        self._btn_h = self.rounded(332, 74, 37, "#c23f78", "#7a2a62")
        btn = self.create_image(self.s(72), self.s(529 + dy), image=self._btn_n, anchor="nw", tags=("start",))
        cy = 566 + dy
        self.text(157, cy, "Get Started", 26, "white", True, tags=("start",))
        self.create_polygon(*[self.s(v) for v in (117, cy - 13, 117, cy + 13, 137, cy)],
                            fill="white", tags=("start",))
        self.create_text(self.s(350), self.s(cy), text="→", fill="white", tags=("start",),
                         font=(FONT, -self.s(34), "bold"))
        self.hover("start",
                   enter=lambda: self.itemconfig(btn, image=self._btn_h),
                   leave=lambda: self.itemconfig(btn, image=self._btn_n))
        self.tag_bind("start", "<Button-1>", lambda e: self.on_navigate("datasets"))

    def _cards(self):
        c = theme.c
        cards = [("feature_dataset", 137, "31", "Datasets", 208, 40),
                 ("feature_algorithms", 510, "3", "Algorithms", 584, 40),
                 ("feature_analytics", 871, "Train & Evaluate", "Build and compare models", 938, 28),
                 ("feature_prediction", 1293, "Real Projects", "Apply ML to real-world data", 1361, 28)]
        for ic, ix, big, small, tx, fs in cards:
            self.pic(ix, 708, ic, 90, crop=True)
            self.text(tx, 693 if fs == 40 else 691, big, fs, c("title"), True)
            self.text(tx, 730 if fs == 40 else 728, small, 20, c("body"))

    def _footer(self):
        cy = FOOTER_CY
        for x in (325, 649, 981, 1252):
            self.create_line(self.s(x), self.s(cy - 22), self.s(x), self.s(cy + 22), fill=theme.c("foot_sep"))

        segments = [(23, 325), (325, 649), (649, 981), (981, 1252), (1252, 1649)]
        LABEL, VALUE = theme.c("foot_label"), "white"
        items = [("footer_developer", [("Developed by", 17, False, LABEL), ("Sara Mollaie", 20, True, VALUE)], 0, 0),
                 ("footer_instructor", [("Instructor", 17, False, LABEL), ("Vahid Ghorbani", 20, True, VALUE)], 0, 0),
                 ("footer_institute", [("Sematect Institute", 20, True, VALUE)], 0, 0),
                 ("footer_linkedin", [("Sara Mollaie", 20, True, VALUE)], 0, 4),
                 ("footer_email", [("saramollaie82@gmail.com", 20, True, VALUE)], 0, 0)]

        for (left, right), (ic, lines, icon_dy, text_dy) in zip(segments, items):
            ph = self.icon(ic, FOOTER_ICON, crop=True)
            iw = ph.width() / self.k
            tw = max(self.text_width(t, sz, b) for t, sz, b, _ in lines)
            x0 = (left + right) / 2 - (iw + FOOTER_GAP + tw) / 2
            self.create_image(self.s(x0 + iw / 2), self.s(cy + icon_dy), image=ph)
            ys = [cy] if len(lines) == 1 else [cy - 12.5, cy + 12.5]
            for (t, sz, b, col), y in zip(lines, ys):
                self.text(x0 + iw + FOOTER_GAP, y + text_dy, t, sz, col, b)