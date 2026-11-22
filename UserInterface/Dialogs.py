import ctypes
import math
import tempfile
import tkinter as tk
import tkinter.font as tkfont
import webbrowser

from pathlib import Path
from PIL import Image, ImageTk, ImageDraw

from UserInterface.Theme import theme, FONT


# =====================================================
# CONSTANTS
# =====================================================

SOURCE_URL = (
    "https://datasciencedojo.com/blog/datasets-data-science-skills/"
)

PURPLE_ICON = "#8250B2"

SS = 4      # ضریب سوپرسمپلینگ برای لبه‌های بدون دندانه (anti-aliasing)


def _mix(c1, c2, t):
    """ترکیب دو رنگ هگز: t=0 یعنی c1 و t=1 یعنی c2"""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _rr_image(w, h, radius, fill, border=None, bw=0, bg="#000000"):
    """
    مستطیل گوشه‌گرد با کیفیت بالا (رندر ۴ برابر و کوچک‌کردن).
    رنگ پس‌زمینه با رنگ والد یکی است تا لبه‌ها تمیز ترکیب شوند.
    """

    big = Image.new("RGBA", (w * SS, h * SS), bg)
    d = ImageDraw.Draw(big)

    box = (0, 0, w * SS - 1, h * SS - 1)

    if border and bw:

        d.rounded_rectangle(box, radius=radius * SS, fill=border)

        inset = bw * SS

        d.rounded_rectangle(
            (inset, inset, w * SS - 1 - inset, h * SS - 1 - inset),
            radius=max(1, radius - bw) * SS,
            fill=fill
        )

    else:

        d.rounded_rectangle(box, radius=radius * SS, fill=fill)

    return big


# =====================================================
# TITLE (crisp text + accent bar)
# =====================================================

class FancyTitle(tk.Canvas):
    """عنوان تمیز و خوانا: متن واضح + خط تزئینی با رنگ تم (بدون هالهٔ تار)"""

    def __init__(self, parent, text, size, bg, ui=1.0, bar=True):

        font = tkfont.Font(root=parent, family=FONT, size=size, weight="bold")

        tw = font.measure(text)
        th = font.metrics("linespace")

        def px(v):
            return int(round(v * ui))

        bar_h = px(16) if bar else 0

        super().__init__(
            parent,
            width=tw + px(6),
            height=th + bar_h + px(2),
            bg=bg,
            highlightthickness=0,
            bd=0
        )

        accent = theme.c("accent")

        self.create_text(
            1, 0, text=text, anchor="nw",
            font=font, fill=theme.c("dlg_title")
        )

        if bar:

            by = th + px(9)
            w = max(3, px(4))

            self.create_line(
                w // 2, by, px(42), by,
                width=w, capstyle="round", fill=accent
            )

            self.create_line(
                px(52), by, px(62), by,
                width=w, capstyle="round", fill=_mix(bg, accent, 0.45)
            )


# =====================================================
# WINDOW ICONS (drawn in code, full-bleed, crisp at 16px)
# =====================================================

def _make_icon(kind):
    """آیکون پر و ساده برای نوار عنوان: 'info' یا 'settings'"""

    S = 1024
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    if kind == "info":

        d.ellipse((20, 20, S - 20, S - 20), fill=PURPLE_ICON)
        d.ellipse((S / 2 - 78, 205, S / 2 + 78, 361), fill="white")
        d.rounded_rectangle(
            (S / 2 - 76, 440, S / 2 + 76, 820),
            radius=64,
            fill="white"
        )

    else:

        cx = cy = S / 2
        teeth = 8
        step = 2 * math.pi / teeth
        outer, inner = 500, 385

        points = []

        for i in range(teeth):

            a = i * step

            for frac, radius in (
                (-0.27, inner),
                (-0.15, outer),
                (0.15, outer),
                (0.27, inner)
            ):
                angle = a + frac * step
                points.append((
                    cx + radius * math.cos(angle),
                    cy + radius * math.sin(angle)
                ))

        d.polygon(points, fill=PURPLE_ICON)
        d.ellipse(
            (cx - inner - 6, cy - inner - 6, cx + inner + 6, cy + inner + 6),
            fill=PURPLE_ICON
        )
        d.ellipse(
            (cx - 175, cy - 175, cx + 175, cy + 175),
            fill=(0, 0, 0, 0)
        )

    return img.resize((256, 256), Image.Resampling.LANCZOS)


# =====================================================
# ROUNDED BUTTON
# =====================================================

class RoundedButton(tk.Canvas):
    """دکمهٔ گوشه‌گرد با لبهٔ نرم. style: 'primary' یا 'secondary'"""

    def __init__(self, parent, text, command, ui=1.0,
                 style="primary", min_width=110):

        self.ui = ui
        self.command = command
        self.style = style
        self.text = text

        def px(v):
            return int(round(v * ui))

        self.font = tkfont.Font(
            root=parent, family=FONT, size=11, weight="bold"
        )

        self.W = max(px(min_width), self.font.measure(text) + px(44))
        self.H = px(42)
        self.R = px(11)

        self.parent_bg = parent.cget("bg")

        super().__init__(
            parent, width=self.W, height=self.H,
            bg=self.parent_bg, highlightthickness=0, bd=0,
            cursor="hand2"
        )

        self._imgs = {}
        self._draw(False)

        self.bind("<Enter>", lambda e: self._draw(True))
        self.bind("<Leave>", lambda e: self._draw(False))
        self.bind("<ButtonRelease-1>", self._click)

    def _click(self, event):

        if 0 <= event.x <= self.W and 0 <= event.y <= self.H:
            self.command()

    def _draw(self, hover):

        accent = theme.c("accent")
        bg = self.parent_bg

        if self.style == "primary":

            fill = theme.c("btn_hover") if hover else theme.c("btn_bg")
            border = fill
            fg = theme.c("btn_fg")

        else:

            fill = _mix(bg, accent, 0.14) if hover else bg
            border = accent if hover else theme.c("foot_sep")
            fg = theme.c("dlg_title")

        key = (fill, border)

        if key not in self._imgs:

            big = _rr_image(
                self.W, self.H, self.R, fill,
                border, max(1, int(round(1.5 * self.ui))), bg
            )

            img = big.resize((self.W, self.H), Image.Resampling.LANCZOS)
            self._imgs[key] = ImageTk.PhotoImage(img)

        self.delete("all")
        self.create_image(0, 0, anchor="nw", image=self._imgs[key])
        self.create_text(
            self.W / 2, self.H / 2, text=self.text,
            font=self.font, fill=fg
        )


# =====================================================
# CHIP (small pill label)
# =====================================================

class Chip(tk.Canvas):

    def __init__(self, parent, text, ui=1.0):

        def px(v):
            return int(round(v * ui))

        font = tkfont.Font(root=parent, family=FONT, size=10, weight="bold")

        W = font.measure(text) + px(42)
        H = px(32)

        bg = parent.cget("bg")
        accent = theme.c("accent")

        super().__init__(
            parent, width=W, height=H, bg=bg,
            highlightthickness=0, bd=0
        )

        big = _rr_image(
            W, H, H // 2,
            _mix(bg, accent, 0.10),
            _mix(bg, accent, 0.40),
            max(1, int(round(ui))), bg
        )

        d = ImageDraw.Draw(big)

        cx, cy, r = px(16) * SS, (H * SS) / 2, px(4) * SS
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=accent)

        self._img = ImageTk.PhotoImage(
            big.resize((W, H), Image.Resampling.LANCZOS)
        )

        self.create_image(0, 0, anchor="nw", image=self._img)
        self.create_text(
            px(28), H / 2, text=text, anchor="w",
            font=font, fill=theme.c("dlg_title")
        )


# =====================================================
# BASE DIALOG
# =====================================================

class _Dialog(tk.Toplevel):

    def __init__(self, parent, title, width, height, icon_name):

        super().__init__(parent)

        self.parent = parent

        self.dialog_width = width
        self.dialog_height = height

        # ضریب مقیاس نمایش ویندوز (۱٫۰ / ۱٫۲۵ / ۱٫۵ ...)
        self.ui = max(1.0, self.winfo_fpixels("1i") / 96)

        self.withdraw()

        self.title(title)
        self.transient(parent)
        self.resizable(False, False)

        self.configure(bg=theme.c("dlg_bg"))

        self._window_icons = []
        self._set_window_icon(icon_name)

        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("<Return>", lambda e: self.destroy())
        self.bind("<Destroy>", self._on_destroy, add="+")

        self._build()

        theme.subscribe(self._on_theme)

        self.after_idle(self._show_dialog)

    # ---------- helpers ----------

    def px(self, value):
        """پیکسل طراحی ← پیکسل واقعی با توجه به مقیاس صفحه"""
        return int(round(value * self.ui))

    def button(self, parent, text, command, width=110, style="primary"):
        return RoundedButton(
            parent, text, command, self.ui, style, min_width=width
        )

    def separator(self, parent, padx=0, pady=0):

        line = tk.Frame(parent, bg=theme.c("foot_sep"), height=1)
        line.pack(fill="x", padx=padx, pady=pady)

        return line

    # ---------- window icon ----------

    def _set_window_icon(self, icon_name):

        try:

            kind = "info" if "info" in icon_name else "settings"
            icon = _make_icon(kind)

            ico_path = Path(tempfile.gettempdir()) / f"mlstudio_{kind}.ico"

            icon.save(
                ico_path,
                format="ICO",
                sizes=[(16, 16), (20, 20), (24, 24), (32, 32),
                       (40, 40), (48, 48), (64, 64), (256, 256)]
            )

            try:
                self.iconbitmap(str(ico_path))

            except Exception:

                for size in (16, 24, 32, 48, 64):
                    self._window_icons.append(
                        ImageTk.PhotoImage(
                            icon.resize((size, size), Image.Resampling.LANCZOS)
                        )
                    )

                self.iconphoto(False, *self._window_icons)

        except Exception as error:
            print("[WINDOW ICON ERROR]", error)

    # ---------- title bar theme (Windows) ----------

    def _titlebar_theme(self):

        try:

            self.update_idletasks()

            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            value = ctypes.c_int(1 if theme.name == "dark" else 0)

            for attribute in (20, 19):
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, attribute,
                    ctypes.byref(value), ctypes.sizeof(value)
                )

            ctypes.windll.user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, 0x37)

        except Exception:
            pass

    # ---------- size + position ----------

    def _fit(self, center):

        self.update_idletasks()

        width = max(self.px(self.dialog_width), self.winfo_reqwidth())
        height = self.winfo_reqheight()

        if center:

            self.parent.update_idletasks()

            x = (
                self.parent.winfo_rootx()
                + (self.parent.winfo_width() - width) // 2
            )

            y = (
                self.parent.winfo_rooty()
                + (self.parent.winfo_height() - height) // 2
            )

            x = max(0, min(x, self.winfo_screenwidth() - width))
            y = max(0, min(y, self.winfo_screenheight() - height))

        else:

            x, y = self.winfo_x(), self.winfo_y()

        self.geometry(f"{width}x{height}+{x}+{y}")
        self.minsize(width, height)
        self.maxsize(width, height)

    def _show_dialog(self):

        if not self.winfo_exists():
            return

        self._fit(center=True)

        self.deiconify()
        self.lift()
        self.grab_set()
        self.focus_force()

        self.after(30, self._titlebar_theme)

    # ---------- destroy ----------

    def _on_destroy(self, event):

        if event.widget is self:
            theme.unsubscribe(self._on_theme)

    # ---------- theme change ----------

    def _on_theme(self):

        if self.winfo_exists():
            self.after_idle(self._rebuild)

    def _rebuild(self):

        if not self.winfo_exists():
            return

        for widget in self.winfo_children():
            widget.destroy()

        self.configure(bg=theme.c("dlg_bg"))

        self._build()
        self._fit(center=False)
        self._titlebar_theme()

    def _build(self):
        raise NotImplementedError


# =====================================================
# THEME OPTION CARD
# =====================================================

class ThemeOption(tk.Canvas):

    @staticmethod
    def needed_width(parent, ui, title, subtitle):

        f_title = tkfont.Font(root=parent, family=FONT, size=12, weight="bold")
        f_sub = tkfont.Font(root=parent, family=FONT, size=9)

        def px(v):
            return int(round(v * ui))

        return max(
            px(230),
            px(88) + f_sub.measure(subtitle) + px(20),
            px(88) + f_title.measure(title) + px(56)
        )

    def __init__(self, parent, mode, title, subtitle, command,
                 ui=1.0, width=None):

        self.ui = ui

        def px(v):
            return int(round(v * ui))

        self.px = px

        self.W = width or self.needed_width(parent, ui, title, subtitle)
        self.H = px(98)

        self.parent_bg = parent.cget("bg")

        super().__init__(
            parent, width=self.W, height=self.H,
            bg=self.parent_bg, highlightthickness=0, bd=0,
            cursor="hand2"
        )

        self.mode = mode
        self.title = title
        self.subtitle = subtitle
        self.command = command

        self.f_title = tkfont.Font(
            root=parent, family=FONT, size=12, weight="bold"
        )
        self.f_sub = tkfont.Font(root=parent, family=FONT, size=9)

        self.hovered = False
        self._imgs = {}

        self._draw()

        self.bind("<Button-1>", self._select)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    # ---- render card (supersampled) ----

    def _render(self, selected, hover):

        px, ui = self.px, self.ui
        W, H = self.W, self.H

        accent = theme.c("accent")
        row_bg = theme.c("row_bg")
        sep = theme.c("foot_sep")

        if selected:
            fill = _mix(row_bg, accent, 0.14)
            border = accent
            bw = max(2, px(2))
        else:
            fill = row_bg
            border = _mix(sep, accent, 0.55) if hover else sep
            bw = max(1, px(1.5))

        big = _rr_image(W, H, px(16), fill, border, bw, self.parent_bg)
        d = ImageDraw.Draw(big)

        def B(v):
            return int(round(v * SS))

        # ----- mini preview -----
        if self.mode == "light":
            p_bg, p_top, p_acc = "#FBF5FA", "#302039", "#D65B91"
        else:
            p_bg, p_top, p_acc = "#21162E", "#100A20", "#A78BFA"

        pw = ph = px(56)
        x0 = px(18)
        y0 = (H - ph) / 2

        d.rounded_rectangle(
            (B(x0), B(y0), B(x0 + pw), B(y0 + ph)),
            radius=B(px(10)),
            fill=p_bg,
            outline=_mix(p_bg, "#808080", 0.35),
            width=SS
        )

        i5 = px(6)

        d.rounded_rectangle(
            (B(x0 + i5), B(y0 + i5), B(x0 + pw - i5), B(y0 + i5 + px(11))),
            radius=B(px(3)), fill=p_top
        )

        col_top = y0 + i5 + px(17)
        col_bot = y0 + ph - i5

        d.rounded_rectangle(
            (B(x0 + i5), B(col_top), B(x0 + i5 + px(19)), B(col_bot)),
            radius=B(px(3)), fill=p_acc
        )

        d.rounded_rectangle(
            (B(x0 + pw - i5 - px(19)), B(col_top),
             B(x0 + pw - i5), B(col_bot)),
            radius=B(px(3)), fill=p_acc
        )

        # ----- selection badge -----
        r = px(12)
        cx, cy = W - px(28), px(28)

        box = (B(cx - r), B(cy - r), B(cx + r), B(cy + r))

        if selected:

            d.ellipse(box, fill=accent)

            pts = [
                (B(cx - r * 0.45), B(cy + r * 0.02)),
                (B(cx - r * 0.10), B(cy + r * 0.38)),
                (B(cx + r * 0.48), B(cy - r * 0.32)),
            ]

            lw = B(max(2.2, ui * 2.4))

            d.line(pts, fill="#FFFFFF", width=lw, joint="curve")

            for (ex, ey) in (pts[0], pts[2]):
                d.ellipse(
                    (ex - lw // 2, ey - lw // 2, ex + lw // 2, ey + lw // 2),
                    fill="#FFFFFF"
                )

        else:

            d.ellipse(
                box, fill=fill,
                outline=_mix(sep, accent, 0.55) if hover else sep,
                width=B(max(1.5, ui * 1.6))
            )

        img = big.resize((W, H), Image.Resampling.LANCZOS)

        return ImageTk.PhotoImage(img)

    # ---- draw ----

    def _draw(self):

        selected = theme.name == self.mode
        key = (selected, self.hovered, theme.name)

        if key not in self._imgs:
            self._imgs[key] = self._render(selected, self.hovered)

        px = self.px

        self.delete("all")
        self.create_image(0, 0, anchor="nw", image=self._imgs[key])

        cy = self.H / 2

        self.create_text(
            px(90), cy - px(11), text=self.title, anchor="w",
            fill=theme.c("dlg_title"), font=self.f_title
        )

        self.create_text(
            px(90), cy + px(14), text=self.subtitle, anchor="w",
            fill=theme.c("dlg_text"), font=self.f_sub
        )

    # ---- events ----

    def _select(self, event=None):
        self.command(self.mode)

    def _on_enter(self, event=None):
        self.hovered = True
        self._draw()

    def _on_leave(self, event=None):
        self.hovered = False
        self._draw()


# =====================================================
# SETTINGS DIALOG
# =====================================================

class SettingsDialog(_Dialog):

    def __init__(self, parent):

        super().__init__(parent, "Settings", 560, 300, "settings")

    def _build(self):

        bg = theme.c("dlg_bg")
        px = self.px

        main = tk.Frame(self, bg=bg)
        main.pack(fill="both", expand=True, padx=px(28), pady=px(24))

        # فوتر اول در پایین قرار می‌گیرد تا هرگز بریده نشود
        footer = tk.Frame(main, bg=bg)
        footer.pack(side="bottom", fill="x", pady=(px(20), 0))

        tk.Label(
            footer,
            text="Changes apply immediately",
            font=(FONT, 10),
            bg=bg,
            fg=theme.c("dlg_text")
        ).pack(side="left")

        self.button(
            footer, "Close", self.destroy, width=110
        ).pack(side="right")

        self.separator(main, pady=0).pack_configure(side="bottom")

        # هدر
        FancyTitle(main, "Appearance", 20, bg, self.ui).pack(anchor="w")

        tk.Label(
            main,
            text="Choose your preferred color theme.",
            font=(FONT, 11),
            bg=bg,
            fg=theme.c("dlg_text")
        ).pack(anchor="w", pady=(px(4), px(16)))

        # کارت‌ها (هر دو هم‌اندازه)
        light_t, light_s = "Light Mode", "Soft pastel appearance"
        dark_t, dark_s = "Dark Mode", "Deep purple appearance"

        width = max(
            ThemeOption.needed_width(self, self.ui, light_t, light_s),
            ThemeOption.needed_width(self, self.ui, dark_t, dark_s)
        )

        options = tk.Frame(main, bg=bg)
        options.pack(anchor="w")

        self.light_card = ThemeOption(
            options, "light", light_t, light_s,
            self._choose_theme, self.ui, width
        )
        self.light_card.grid(row=0, column=0, padx=(0, px(14)))

        self.dark_card = ThemeOption(
            options, "dark", dark_t, dark_s,
            self._choose_theme, self.ui, width
        )
        self.dark_card.grid(row=0, column=1)

    def _choose_theme(self, mode):

        if theme.name == mode:
            return

        theme.set(mode)


# =====================================================
# INFORMATION DIALOG
# =====================================================

class InfoDialog(_Dialog):

    def __init__(self, parent):

        super().__init__(parent, "About", 720, 480, "info")

    def _build(self):

        bg = theme.c("dlg_bg")
        row_bg = theme.c("row_bg")
        px = self.px

        main = tk.Frame(self, bg=bg)
        main.pack(fill="both", expand=True, padx=px(28), pady=px(24))

        # دکمه‌ها اول در پایین قرار می‌گیرند تا هرگز بریده نشوند
        buttons = tk.Frame(main, bg=bg)
        buttons.pack(side="bottom", fill="x", pady=(px(20), 0))

        self.button(
            buttons, "Close", self.destroy, width=120, style="secondary"
        ).pack(side="right")

        self.button(
            buttons, "Open website",
            lambda: webbrowser.open(SOURCE_URL),
            width=150
        ).pack(side="right", padx=(0, px(12)))

        # هدر
        FancyTitle(main, "About ML Studio", 22, bg, self.ui).pack(anchor="w")

        tk.Label(
            main, text="Machine Learning Prediction Studio",
            font=(FONT, 12), bg=bg, fg=theme.c("link")
        ).pack(anchor="w", pady=(px(4), px(18)))

        # کارت اصلی
        card = tk.Frame(
            main, bg=row_bg, highlightthickness=1,
            highlightbackground=theme.c("foot_sep")
        )
        card.pack(fill="both", expand=True)

        pad = px(28)

        tk.Label(
            card, text="32 ready-to-use datasets",
            font=(FONT, 16, "bold"), bg=row_bg, fg=theme.c("dlg_title")
        ).pack(anchor="w", padx=pad, pady=(px(22), px(10)))

        tk.Label(
            card,
            text=(
                "All 32 datasets in ML Studio were implemented from the "
                "Data Science Dojo collection. Explore each dataset, then "
                "train and compare Regression, Classification and "
                "Clustering models, and make predictions in one place."
            ),
            font=(FONT, 11),
            bg=row_bg, fg=theme.c("dlg_text"),
            justify="left", anchor="w",
            wraplength=px(600)
        ).pack(anchor="w", padx=pad)

        # برچسب‌های نوع مسئله
        chips = tk.Frame(card, bg=row_bg)
        chips.pack(anchor="w", padx=pad, pady=(px(16), 0))

        for name in ("Regression", "Classification", "Clustering"):
            Chip(chips, name, self.ui).pack(side="left", padx=(0, px(10)))

        # جداکننده
        tk.Frame(card, bg=theme.c("foot_sep"), height=1).pack(
            fill="x", padx=pad, pady=(px(20), px(16))
        )

        # بخش منبع
        tk.Label(
            card, text="SOURCE",
            font=(FONT, 9, "bold"), bg=row_bg, fg=theme.c("accent")
        ).pack(anchor="w", padx=pad)

        link_font = tkfont.Font(family=FONT, size=12, weight="bold")

        link = tk.Label(
            card,
            text="Data Science Dojo: 32 datasets for data science skills  ↗",
            font=link_font, bg=row_bg, fg=theme.c("link"),
            cursor="hand2"
        )
        link.pack(anchor="w", padx=pad, pady=(px(6), 0))

        tk.Label(
            card, text="datasciencedojo.com",
            font=(FONT, 10), bg=row_bg, fg=theme.c("dlg_text")
        ).pack(anchor="w", padx=pad, pady=(px(2), px(22)))

        def enter(_):
            link_font.configure(underline=True)
            link.configure(fg=theme.c("link_hover"))

        def leave(_):
            link_font.configure(underline=False)
            link.configure(fg=theme.c("link"))

        link.bind("<Button-1>", lambda e: webbrowser.open(SOURCE_URL))
        link.bind("<Enter>", enter)
        link.bind("<Leave>", leave)