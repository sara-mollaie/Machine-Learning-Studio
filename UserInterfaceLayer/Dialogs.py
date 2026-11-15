import ctypes
import math
import tempfile
import tkinter as tk
import tkinter.font as tkfont
import webbrowser

from pathlib import Path
from PIL import Image, ImageTk, ImageDraw

from UserInterfaceLayer.Theme import theme, FONT


# =====================================================
# CONSTANTS
# =====================================================

SOURCE_URL = (
    "https://datasciencedojo.com/blog/datasets-data-science-skills/"
)

PURPLE_ICON = "#8250B2"


def _mix(c1, c2, t):
    """ترکیب دو رنگ هگز: t=0 یعنی c1 و t=1 یعنی c2"""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


# =====================================================
# FANCY TITLE (glow + shadow + accent bar)
# =====================================================

class FancyTitle(tk.Canvas):
    """عنوان شیک: هالهٔ نرم (glow)، سایه و خط تزئینی با رنگ تم"""

    def __init__(self, parent, text, size, bg, ui=1.0, bar=True):

        font = tkfont.Font(root=parent, family=FONT, size=size, weight="bold")

        gp = max(4, int(round(8 * ui)))                 # حاشیهٔ هاله
        tw = font.measure(text)
        th = font.metrics("linespace")
        bar_h = int(round(12 * ui)) if bar else 0

        super().__init__(
            parent,
            width=tw + 2 * gp,
            height=th + 2 * gp + bar_h,
            bg=bg,
            highlightthickness=0,
            bd=0
        )

        accent = theme.c("accent")
        title = theme.c("dlg_title")
        dark = theme.name == "dark"

        x, y = gp, gp

        # هالهٔ نرم (چند لایه با شعاع کم‌کم کمتر و رنگ پررنگ‌تر)
        for radius, strength in (
            (5 * ui, 0.10),
            (3 * ui, 0.18),
            (1.5 * ui, 0.30)
        ):
            r = max(1, int(round(radius)))
            color = _mix(bg, accent, strength)

            for dx, dy in (
                (-r, 0), (r, 0), (0, -r), (0, r),
                (-r, -r), (r, r), (-r, r), (r, -r)
            ):
                self.create_text(
                    x + dx, y + dy, text=text, anchor="nw",
                    font=font, fill=color
                )

        # سایه
        s = max(1, int(round(2 * ui)))
        self.create_text(
            x + s, y + s, text=text, anchor="nw", font=font,
            fill=_mix(bg, "#000000", 0.55 if dark else 0.22)
        )

        # متن اصلی
        self.create_text(
            x, y, text=text, anchor="nw", font=font, fill=title
        )

        # خط تزئینی دو‌تکه زیر عنوان
        if bar:

            by = gp + th + int(round(4 * ui))
            w = max(3, int(round(4 * ui)))

            self.create_line(
                x, by, x + int(46 * ui), by,
                width=w, capstyle="round", fill=accent
            )

            self.create_line(
                x + int(56 * ui), by, x + int(70 * ui), by,
                width=w, capstyle="round", fill=_mix(bg, accent, 0.5)
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

        # دایرهٔ پر + حرف i سفید
        d.ellipse((20, 20, S - 20, S - 20), fill=PURPLE_ICON)
        d.ellipse((S / 2 - 78, 205, S / 2 + 78, 361), fill="white")
        d.rounded_rectangle(
            (S / 2 - 76, 440, S / 2 + 76, 820),
            radius=64,
            fill="white"
        )

    else:

        # چرخ‌دندهٔ پر با سوراخ وسط
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
# BASE DIALOG
# =====================================================

class _Dialog(tk.Toplevel):

    def __init__(self, parent, title, width, height, icon_name):

        super().__init__(parent)

        self.parent = parent

        # عرض طراحی (پیکسل در ۱۰۰٪). ارتفاع از روی محتوا حساب می‌شود
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
        self.bind("<Destroy>", self._on_destroy, add="+")

        self._build()

        theme.subscribe(self._on_theme)

        self.after_idle(self._show_dialog)

    # =================================================
    # HELPERS
    # =================================================

    def px(self, value):
        """پیکسل طراحی ← پیکسل واقعی با توجه به مقیاس صفحه"""
        return int(round(value * self.ui))

    # =================================================
    # WINDOW ICON
    # =================================================

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

    # =================================================
    # TITLE BAR THEME (Windows)
    # =================================================

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

    # =================================================
    # SIZE + POSITION (auto fit to content)
    # =================================================

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

    # =================================================
    # DESTROY
    # =================================================

    def _on_destroy(self, event):

        if event.widget is self:
            theme.unsubscribe(self._on_theme)

    # =================================================
    # THEME CHANGE
    # =================================================

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

    # =================================================
    # BUTTON
    # =================================================

    def button(self, parent, text, command, width=14):

        bg = theme.c("btn_bg")
        fg = theme.c("btn_fg")
        hover = theme.c("btn_hover")

        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bd=0,
            relief="flat",
            cursor="hand2",
            bg=bg,
            fg=fg,
            activebackground=hover,
            activeforeground=fg,
            font=(FONT, 11, "bold"),
            width=width,
            pady=self.px(8),
            highlightthickness=0
        )

        btn.bind("<Enter>", lambda e: btn.configure(bg=hover))
        btn.bind("<Leave>", lambda e: btn.configure(bg=bg))

        return btn

    def _build(self):
        raise NotImplementedError


# =====================================================
# THEME OPTION CARD
# =====================================================

class ThemeOption(tk.Canvas):

    CARD_HEIGHT = 92          # ارتفاع کارت
    MIN_WIDTH = 200           # کمترین عرض کارت (عرض واقعی از روی متن حساب می‌شود)

    def __init__(self, parent, mode, title, subtitle, command):

        # عرض کارت از روی عرض واقعی متن‌ها
        f_title = tkfont.Font(root=parent, family=FONT, size=11, weight="bold")
        f_sub = tkfont.Font(root=parent, family=FONT, size=8)

        width = max(
            self.MIN_WIDTH,
            72 + f_sub.measure(subtitle) + 18,
            72 + f_title.measure(title) + 18 + 44      # فضای دایرهٔ تیک
        )

        self.CARD_WIDTH = int(width)

        super().__init__(
            parent,
            width=self.CARD_WIDTH,
            height=self.CARD_HEIGHT,
            bg=theme.c("row_bg"),
            highlightthickness=0,
            bd=0,
            cursor="hand2"
        )

        self.mode = mode
        self.title = title
        self.subtitle = subtitle
        self.command = command

        self.hovered = False

        self._draw()

        self.bind("<Button-1>", self._select)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    # ---- rounded rectangle ----

    def _rounded_rect(self, x1, y1, x2, y2, radius, fill, outline):

        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1
        ]

        self.create_polygon(
            points,
            smooth=True,
            splinesteps=24,
            fill=fill,
            outline=outline,
            width=2
        )

    # ---- draw ----

    def _draw(self):

        self.delete("all")

        W, H = self.CARD_WIDTH, self.CARD_HEIGHT
        cy = H / 2

        selected = theme.name == self.mode

        if selected:
            background = theme.c("pill")
            border = PURPLE_ICON
            title_color = theme.c("pill_text")
            subtitle_color = theme.c("pill_text")
        else:
            background = theme.c("dlg_bg")
            border = theme.c("foot_sep")
            title_color = theme.c("dlg_title")
            subtitle_color = theme.c("dlg_text")

        if self.hovered and not selected:
            border = PURPLE_ICON

        # کارت اصلی
        self._rounded_rect(2, 2, W - 2, H - 2, 14, background, border)

        if self.mode == "light":
            preview_bg = "#FBF5FA"
            preview_top = "#302039"
            preview_accent = "#D65B91"
        else:
            preview_bg = "#21162E"
            preview_top = "#100A20"
            preview_accent = "#A78BFA"

        # پیش‌نمایش کوچک
        self._rounded_rect(12, cy - 25, 58, cy + 25, 7, preview_bg, preview_bg)
        self._rounded_rect(16, cy - 21, 54, cy - 11, 3, preview_top, preview_top)

        self.create_rectangle(18, cy - 3, 31, cy + 17, fill=preview_accent, outline="")
        self.create_rectangle(37, cy - 3, 50, cy + 17, fill=preview_accent, outline="")

        # متن‌ها
        self.create_text(
            72, cy - 9, text=self.title, anchor="w",
            fill=title_color, font=(FONT, 11, "bold")
        )

        self.create_text(
            72, cy + 16, text=self.subtitle, anchor="w",
            fill=subtitle_color, font=(FONT, 8)
        )

        # نشانگر انتخاب: بزرگ‌تر و بالا-راست، دور از متن‌ها
        r = 11
        ox, oy = W - 26, 25

        if selected:

            self.create_oval(
                ox - r, oy - r, ox + r, oy + r,
                fill=PURPLE_ICON, outline=""
            )

            # تیک با خط (تیز در هر اندازه)
            self.create_line(
                ox - 5, oy, ox - 1.5, oy + 4, ox + 5.5, oy - 4,
                fill="#FFFFFF", width=2.6,
                capstyle="round", joinstyle="round"
            )

        else:

            self.create_oval(
                ox - r, oy - r, ox + r, oy + r,
                fill=background,
                outline=theme.c("foot_sep"),
                width=2
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

    def refresh(self):
        self.configure(bg=theme.c("row_bg"))
        self._draw()


# =====================================================
# SETTINGS DIALOG
# =====================================================

class SettingsDialog(_Dialog):

    def __init__(self, parent):

        # عرض پنجره حداقل است و اگر لازم باشد خودکار بزرگ‌تر می‌شود
        super().__init__(parent, "Settings", 440, 250, "settings")

    def _build(self):

        bg = theme.c("dlg_bg")
        row_bg = theme.c("row_bg")

        main = tk.Frame(self, bg=bg)
        main.pack(fill="both", expand=True, padx=16, pady=14)

        # دکمه‌ها اول در پایین قرار می‌گیرند تا هرگز بریده نشوند
        footer = tk.Frame(main, bg=bg)
        footer.pack(side="bottom", fill="x", pady=(10, 0))

        tk.Label(
            footer,
            text="Changes apply immediately",
            font=(FONT, 9),
            bg=bg,
            fg=theme.c("dlg_text")
        ).pack(side="left", pady=4)

        self.button(footer, "Close", self.destroy, width=10).pack(side="right")

        # کادر Appearance: فقط به اندازهٔ محتوا و وسط‌چین
        appearance = tk.Frame(
            main,
            bg=row_bg,
            highlightthickness=1,
            highlightbackground=theme.c("foot_sep")
        )
        appearance.pack(anchor="center")

        # عنوان شیک
        FancyTitle(
            appearance, "Appearance", 17, row_bg, self.ui
        ).pack(anchor="w", padx=10, pady=(8, 0))

        tk.Label(
            appearance,
            text="Choose your preferred color theme.",
            font=(FONT, 9),
            bg=row_bg,
            fg=theme.c("dlg_text")
        ).pack(anchor="w", padx=16, pady=(0, 10))

        options = tk.Frame(appearance, bg=row_bg)
        options.pack(padx=12, pady=(0, 14))

        self.light_card = ThemeOption(
            options, "light", "Light Mode",
            "Soft pastel appearance", self._choose_theme
        )
        self.light_card.grid(row=0, column=0, padx=(0, 8))

        self.dark_card = ThemeOption(
            options, "dark", "Dark Mode",
            "Deep purple appearance", self._choose_theme
        )
        self.dark_card.grid(row=0, column=1)

    def _choose_theme(self, mode):

        if theme.name == mode:
            return

        theme.set(mode)


# =====================================================
# INFORMATION DIALOG
# =====================================================

def _wrap(text, font, max_px):
    """متن را با اندازه‌گیری واقعی فونت به خط‌های مشخص می‌شکند"""

    lines, current = [], ""

    for word in text.split():

        trial = (current + " " + word).strip()

        if font.measure(trial) <= max_px:
            current = trial
        else:
            lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


class InfoDialog(_Dialog):

    def __init__(self, parent):

        super().__init__(parent, "About", 680, 480, "info")

    def _build(self):

        bg = theme.c("dlg_bg")
        row_bg = theme.c("row_bg")
        px = self.px

        main = tk.Frame(self, bg=bg)
        main.pack(fill="both", expand=True, padx=px(26), pady=px(22))

        # دکمه‌ها اول در پایین قرار می‌گیرند تا هرگز بریده نشوند
        buttons = tk.Frame(main, bg=bg)
        buttons.pack(side="bottom", fill="x", pady=(px(18), 0))

        self.button(buttons, "Close", self.destroy, width=14).pack(side="right")
        self.button(
            buttons, "Open website",
            lambda: webbrowser.open(SOURCE_URL),
            width=14
        ).pack(side="right", padx=(0, px(12)))

        # عنوان شیک
        FancyTitle(
            main, "About ML Studio", 24, bg, self.ui
        ).pack(anchor="w")

        tk.Label(
            main, text="Machine Learning Prediction Studio",
            font=(FONT, 11), bg=bg, fg=theme.c("link")
        ).pack(anchor="w", padx=px(8), pady=(px(2), px(16)))

        # کارت اصلی
        card = tk.Frame(
            main, bg=row_bg, highlightthickness=1,
            highlightbackground=theme.c("foot_sep")
        )
        card.pack(fill="both", expand=True)

        pad = px(28)

        FancyTitle(
            card, "32 ready-to-use datasets", 16, row_bg, self.ui, bar=False
        ).pack(anchor="w", padx=pad - px(8), pady=(px(18), px(8)))

        # متن اصلی: هر خط جدا، با فاصلهٔ مشخص بین خطوط
        body_font = tkfont.Font(family=FONT, size=10)

        text = (
            "All 32 datasets in ML Studio were implemented from the "
            "Data Science Dojo collection. Explore each dataset, then "
            "train and compare Regression, Classification and "
            "Clustering models, and make predictions in one place."
        )

        for line in _wrap(text, body_font, px(560)):

            tk.Label(
                card, text=line, font=body_font,
                bg=row_bg, fg=theme.c("dlg_text"),
                anchor="w", justify="left"
            ).pack(anchor="w", padx=pad, pady=(0, px(7)))

        # جداکننده
        tk.Frame(card, bg=theme.c("foot_sep"), height=1).pack(
            fill="x", padx=pad, pady=(px(14), px(16))
        )

        # بخش منبع
        tk.Label(
            card, text="SOURCE",
            font=(FONT, 8, "bold"), bg=row_bg, fg=theme.c("accent")
        ).pack(anchor="w", padx=pad)

        link_font = tkfont.Font(family=FONT, size=11, weight="bold")

        link = tk.Label(
            card,
            text="Data Science Dojo: 32 datasets for data science skills  ↗",
            font=link_font, bg=row_bg, fg=theme.c("link"),
            cursor="hand2"
        )
        link.pack(anchor="w", padx=pad, pady=(px(8), 0))

        tk.Label(
            card, text="datasciencedojo.com",
            font=(FONT, 9), bg=row_bg, fg=theme.c("dlg_text")
        ).pack(anchor="w", padx=pad, pady=(px(3), px(22)))

        def enter(_):
            link_font.configure(underline=True)
            link.configure(fg=theme.c("link_hover"))

        def leave(_):
            link_font.configure(underline=False)
            link.configure(fg=theme.c("link"))

        link.bind("<Button-1>", lambda e: webbrowser.open(SOURCE_URL))
        link.bind("<Enter>", enter)
        link.bind("<Leave>", leave)