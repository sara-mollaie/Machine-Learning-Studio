import json
from pathlib import Path

FONT = "Poppins"          # اگر نصب نیست: "Segoe UI"
SETTINGS_FILE = Path(__file__).resolve().parent.parent / "settings.json"

PALETTES = {
    "light": {
        "title": "#2b1450", "accent": "#c9578f", "body": "#555062",
        "hdr": ("#261444", "#37226a"), "ftr": ("#2f1a49", "#482659"),
        "foot_sep": "#6d5585", "foot_label": "#d9c9e6",
        "pill": "#c9b6f2", "pill_text": "#3a1d63",
        "card_cols": ["#efe9fa", "#fbe9ef", "#f1ebfa", "#fbeaf1"], "card_end": "#ffffff",
        "card_alpha": 215, "card_border": ("#ffffff", 200),
        "shadow_rgb": (78, 35, 120), "shadow_a": 70,
        "dlg_bg": "#faf5fd", "dlg_title": "#2b1450", "dlg_text": "#555062",
        "link": "#c9578f", "link_hover": "#a3305f",
        "btn_bg": "#a3305f", "btn_hover": "#c23f78", "btn_fg": "#ffffff",
        "sw_on": "#a3305f", "sw_off": "#cbbfe0", "row_bg": "#efe9fa",
    },
    "dark": {
        "title": "#f5efff", "accent": "#e07fb5", "body": "#c9bddb",
        "hdr": ("#0f081e", "#1a1030"), "ftr": ("#170d2a", "#2a1745"),
        "foot_sep": "#4a3566", "foot_label": "#b9a8d0",
        "pill": "#8e6be8", "pill_text": "#ffffff",
        "card_cols": ["#2a1a47", "#3a1b42", "#271a45", "#391c44"], "card_end": "#1b1030",
        "card_alpha": 235, "card_border": ("#7d5bc0", 110),
        "shadow_rgb": (0, 0, 0), "shadow_a": 110,
        "dlg_bg": "#1a1030", "dlg_title": "#f5efff", "dlg_text": "#c9bddb",
        "link": "#e07fb5", "link_hover": "#f3b3d3",
        "btn_bg": "#a3305f", "btn_hover": "#c23f78", "btn_fg": "#ffffff",
        "sw_on": "#c9578f", "sw_off": "#4a3566", "row_bg": "#2a1a47",
    },
}


class ThemeManager:
    def __init__(self):
        self.name = "light"
        self._listeners = []
        try:
            saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8")).get("theme")
            if saved in PALETTES:
                self.name = saved
        except Exception:
            pass

    def c(self, key):
        return PALETTES[self.name][key]

    def set(self, name):
        if name == self.name or name not in PALETTES:
            return
        self.name = name
        try:
            SETTINGS_FILE.write_text(json.dumps({"theme": name}), encoding="utf-8")
        except Exception:
            pass
        for cb in list(self._listeners):
            cb()

    def subscribe(self, cb):
        self._listeners.append(cb)

    def unsubscribe(self, cb):
        if cb in self._listeners:
            self._listeners.remove(cb)


theme = ThemeManager()