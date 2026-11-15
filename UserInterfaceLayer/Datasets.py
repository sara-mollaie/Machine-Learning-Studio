"""
DatasetsModule.py  -  Datasets page of ML Studio (pure tkinter, no extra packages).

How the page fills itself
-------------------------
Every dataset script already writes a metadata file:

    <ProjectRoot>/<Task folder>/Metadata/<anything>.json
    e.g. Clustering/Metadata/D21_travel_reviews.json

The page scans   <ProjectRoot>/*/Metadata/*.json   and builds one card per file.
It re-checks the folder every couple of seconds, so a new (or edited / deleted)
metadata file shows up on the page by itself, with no change to this module.

Fields read from each JSON (missing ones fall back to a default):
    id, name, level, domain, task, description, source,
    dataset_statistics.rows, dataset_statistics.columns

Usage
-----
    from UserInterfaceLayer.DatasetsModule import DatasetsPage

    page = DatasetsPage(parent, on_open=lambda ds: print(ds.name))
    page.pack(fill="both", expand=True)

    ds.raw  ->  the full metadata dict, handy for the Dataset Info dialog.
"""

from __future__ import annotations

import json
import math
import random
import re
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import font as tkfont
from typing import Callable, Optional

# region Constants

LEVELS = ["Beginner", "Intermediate", "Advanced"]
TASKS = ["Regression", "Classification", "Clustering"]

SOURCE_LABEL = "Data Science Dojo"
PAGE_ROWS = 2          # rows of cards per page
MAX_COLS = 4           # columns of cards (fewer on narrow windows)
MIN_CARD_W = 240
CARD_H = 198
GAP = 18

# Colors (same family as the chart theme)
INK = "#2B1744"
INK_SOFT = "#3B2159"
MUTED = "#6F6585"
BG_TOP = "#F8F3FC"
BG_BOTTOM = "#EEE2F8"
NAV_BG = "#32195A"
NAV_BTN = "#43266F"
ACTIVE_PILL = "#C8B3F2"
DARK_BTN = "#3B1F66"
CARD_BORDER = "#E8DCF6"
CHIP_BG = "#EFE8F8"
WHITE = "#FFFFFF"

LEVEL_DOT = {
    "Beginner": "#2FB57A",
    "Intermediate": "#F0A23B",
    "Advanced": "#E5546B",
}

TASK_STYLE = {
    "Regression": {"icon": "#8B5CC7", "tint": "#F5EFFC", "line": "#CDB9EF"},
    "Classification": {"icon": "#D65B91", "tint": "#FDF1F7", "line": "#F0B9D3"},
    "Clustering": {"icon": "#2BB3A6", "tint": "#EFF8F7", "line": "#9CD8D2"},
}
DEFAULT_STYLE = {"icon": "#7A6F8E", "tint": "#F6F3FA", "line": "#D5CDE2"}

FONT_CANDIDATES = ("Poppins", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans")

# endregion


# region Data layer

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
    def sort_key(self):
        level_rank = LEVELS.index(self.level) if self.level in LEVELS else len(LEVELS)
        match = re.search(r"\d+", self.id or "")
        number = int(match.group()) if match else 9999
        return (level_rank, number, self.name.lower())


def _match_case(value, allowed, default):
    """Return the canonical spelling from `allowed` (case-insensitive), else a tidy value."""
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
    """Finds and parses dataset metadata files under the project root."""

    PATTERNS = ("*/Metadata/*.json", "Metadata/*.json")

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else Path(__file__).resolve().parents[1]

    def _files(self):
        seen = set()

        for pattern in self.PATTERNS:
            for path in self.root.glob(pattern):
                relative_parts = path.relative_to(self.root).parts

                if any(part.startswith(".") for part in relative_parts):
                    continue  # .venv, .idea ...

                if path not in seen:
                    seen.add(path)
                    yield path

    def signature(self):
        """Cheap fingerprint of the metadata folder (paths + modification times)."""
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
            level=_match_case(data.get("level"), LEVELS, "Beginner"),
            domain=str(data.get("domain") or "General"),
            task=_match_case(data.get("task"), TASKS, "Other"),
            rows=_to_int(stats.get("rows", data.get("rows"))),
            columns=_to_int(stats.get("columns", data.get("columns"))),
            description=str(data.get("description") or ""),
            source=str(data.get("source") or SOURCE_LABEL),
            path=path,
            raw=data,
        )

# endregion


# region Small helpers

def _lerp_color(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _fit(text, font, max_w):
    """Shorten text with an ellipsis so it fits max_w pixels."""
    if font.measure(text) <= max_w:
        return text

    while text and font.measure(text + "…") > max_w:
        text = text[:-1]

    return text.rstrip() + "…"


def _wrap(text, font, max_w, max_lines=2):
    words = text.split()
    lines, current = [], ""

    for word in words:
        trial = f"{current} {word}".strip()

        if font.measure(trial) <= max_w or not current:
            current = trial
        else:
            lines.append(current)
            current = word

    if current:
        lines.append(current)

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = _fit(lines[-1] + "…", font, max_w)

    return [_fit(line, font, max_w) for line in lines] or [""]


def _page_tokens(pages, current):
    """1 2 3 ... 9 10 style list of page numbers."""
    if pages <= 7:
        return list(range(1, pages + 1))

    wanted = {1, pages, current - 1, current, current + 1}
    numbers = sorted(n for n in wanted if 1 <= n <= pages)

    tokens = []
    for n in numbers:
        if tokens and n - tokens[-1] > 1:
            tokens.append("…")
        tokens.append(n)

    return tokens

# endregion


# region Datasets page

class DatasetsPage(tk.Frame):
    """The Datasets page: title, search, filters, card grid, pagination."""

    NAV_ITEMS = ("Home", "Datasets", "Visualization", "Algorithms")

    def __init__(
        self,
        parent,
        catalog: Optional[DatasetCatalog] = None,
        on_open: Optional[Callable[[DatasetInfo], None]] = None,
        on_navigate: Optional[Callable[[str], None]] = None,
        show_navbar: bool = False,
        poll_ms: int = 2500,
        **kwargs,
    ):
        super().__init__(parent, bg=BG_TOP, **kwargs)

        self.catalog = catalog or DatasetCatalog()
        self.on_open = on_open or (lambda ds: print("Open dataset:", ds.name))
        self.on_navigate = on_navigate
        self.show_navbar = show_navbar
        self.poll_ms = poll_ms

        self.all: list[DatasetInfo] = []
        self.filtered: list[DatasetInfo] = []
        self.level_filter: Optional[str] = None
        self.task_filter: Optional[str] = None
        self.page = 1

        self._sig = None
        self._size = (0, 0)
        self._render_job = None
        self._poll_job = None
        self._placeholder_on = True

        self._make_fonts()

        self.cv = tk.Canvas(self, bg=BG_TOP, highlightthickness=0, bd=0)
        self.cv.pack(fill="both", expand=True)

        self._make_search_entry()

        self.cv.bind("<Configure>", self._on_configure)
        self.cv.bind("<MouseWheel>", self._on_wheel)
        self.cv.bind("<Button-4>", lambda e: self._scroll(-1))
        self.cv.bind("<Button-5>", lambda e: self._scroll(1))

        self.reload()
        self._poll_job = self.after(self.poll_ms, self._poll)

    # ------------------------------------------------------------ setup

    def _make_fonts(self):
        families = set(tkfont.families(self))
        family = next((f for f in FONT_CANDIDATES if f in families), "TkDefaultFont")

        def make(px, bold=False):
            return tkfont.Font(
                root=self, family=family, size=-px, weight="bold" if bold else "normal"
            )

        self.f_title = make(44, True)
        self.f_sub = make(15)
        self.f_pill = make(14, True)
        self.f_search = make(14)
        self.f_card_title = make(18, True)
        self.f_card_sub = make(13)
        self.f_chip = make(12, True)
        self.f_badge = make(12, True)
        self.f_footer = make(15)
        self.f_page = make(13, True)
        self.f_nav = make(15, True)
        self.f_logo = make(28, True)
        self.f_empty = make(20, True)

    def _make_search_entry(self):
        self.entry = tk.Entry(
            self.cv,
            font=self.f_search,
            bd=0,
            relief="flat",
            highlightthickness=0,
            bg=WHITE,
            fg=MUTED,
            insertbackground=INK,
        )
        self.entry.insert(0, "Search datasets")

        self.entry.bind("<FocusIn>", self._entry_focus_in)
        self.entry.bind("<FocusOut>", self._entry_focus_out)
        self.entry.bind("<KeyRelease>", self._entry_changed)

        self._search_win = self.cv.create_window(
            0, 0, window=self.entry, anchor="w", tags="keep"
        )

    def _entry_focus_in(self, _event):
        if self._placeholder_on:
            self.entry.delete(0, "end")
            self.entry.config(fg=INK)
            self._placeholder_on = False

    def _entry_focus_out(self, _event):
        if not self.entry.get().strip():
            self.entry.delete(0, "end")
            self.entry.insert(0, "Search datasets")
            self.entry.config(fg=MUTED)
            self._placeholder_on = True

    def _entry_changed(self, _event):
        self.page = 1
        self._apply_filters()
        self._schedule_render(60)

    def _query(self):
        return "" if self._placeholder_on else self.entry.get().strip().lower()

    # ------------------------------------------------------------ data

    def reload(self):
        """Re-read the metadata files and redraw."""
        self.all = self.catalog.load()
        self._sig = self.catalog.signature()
        self._apply_filters()
        self._schedule_render(0)

    refresh = reload

    def _poll(self):
        try:
            if self.catalog.signature() != self._sig:
                self.reload()
        except Exception as error:  # never let polling kill the UI
            print("[Datasets] refresh check failed:", error)

        self._poll_job = self.after(self.poll_ms, self._poll)

    def destroy(self):
        for job in (self._poll_job, self._render_job):
            if job:
                try:
                    self.after_cancel(job)
                except Exception:
                    pass
        super().destroy()

    def _apply_filters(self):
        query = self._query()

        def matches(ds):
            if self.level_filter and ds.level != self.level_filter:
                return False
            if self.task_filter and ds.task != self.task_filter:
                return False
            if query:
                haystack = " ".join(
                    (ds.name, ds.domain, ds.task, ds.level, ds.description)
                ).lower()
                return query in haystack
            return True

        self.filtered = [ds for ds in self.all if matches(ds)]

    # ------------------------------------------------------------ events

    def _on_configure(self, event):
        if (event.width, event.height) != self._size:
            self._size = (event.width, event.height)
            self._schedule_render(40)

    def _schedule_render(self, delay):
        if self._render_job:
            self.after_cancel(self._render_job)
        self._render_job = self.after(delay, self._render)

    def _on_wheel(self, event):
        self._scroll(-1 if event.delta > 0 else 1)

    def _scroll(self, direction):
        region = self.cv.cget("scrollregion").split()
        if len(region) == 4 and float(region[3]) > self.cv.winfo_height() + 1:
            self.cv.yview_scroll(direction * 3, "units")

    def _set_level(self, level):
        self.level_filter = level
        self.page = 1
        self._apply_filters()
        self._render()

    def _toggle_task(self, task):
        self.task_filter = None if self.task_filter == task else task
        self.page = 1
        self._apply_filters()
        self._render()

    def _go_page(self, number):
        self.page = number
        self._render()

    def _clickable(self, tag, command):
        self.cv.tag_bind(tag, "<Button-1>", lambda e: command())
        self.cv.tag_bind(tag, "<Enter>", lambda e: self.cv.config(cursor="hand2"))
        self.cv.tag_bind(tag, "<Leave>", lambda e: self.cv.config(cursor=""))

    # ------------------------------------------------------------ drawing primitives

    def _rr(self, x1, y1, x2, y2, r, **kw):
        """Rounded rectangle."""
        r = max(0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
        pts = [
            x1 + r, y1, x1 + r, y1, x2 - r, y1, x2 - r, y1, x2, y1,
            x2, y1 + r, x2, y1 + r, x2, y2 - r, x2, y2 - r, x2, y2,
            x2 - r, y2, x2 - r, y2, x1 + r, y2, x1 + r, y2, x1, y2,
            x1, y2 - r, x1, y2 - r, x1, y1 + r, x1, y1 + r, x1, y1,
        ]
        return self.cv.create_polygon(pts, smooth=True, **kw)

    def _text(self, x, y, text, font, fill, anchor="nw", tags=("ui",)):
        return self.cv.create_text(
            x, y, text=text, font=font, fill=fill, anchor=anchor, tags=tags
        )

    def _draw_background(self, width, height):
        self.cv.delete("bg")
        step = 8

        for y in range(0, int(height) + step, step):
            color = _lerp_color(BG_TOP, BG_BOTTOM, min(1.0, y / max(height, 1)))
            self.cv.create_rectangle(
                0, y, width, y + step + 1, fill=color, outline="", tags="bg"
            )

        self.cv.tag_lower("bg")

    # ------------------------------------------------------------ render

    def _render(self):
        self._render_job = None

        width = max(self.cv.winfo_width(), 640)
        view_h = max(self.cv.winfo_height(), 300)

        self.cv.delete("ui")

        margin = 48 if width >= 1100 else 28
        content_w = width - 2 * margin

        y = 8
        if self.show_navbar:
            self._draw_navbar(width)
            y = 76
        else:
            y = 20

        # ---- title + subtitle
        self._text(margin, y, "Datasets", self.f_title, INK)
        y += 64

        subtitle = (
            f"{len(self.all)} datasets from {SOURCE_LABEL}, sorted Beginner to Advanced"
        )
        self._text(margin, y, subtitle, self.f_sub, MUTED)
        y += 38

        # ---- search + filters
        search_w = int(min(300, max(220, content_w * 0.24)))
        layout = self._pill_layout()
        search_h = 42

        row_y = y
        beside_x = margin + search_w + 26

        placed, n_rows = self._pill_rows(layout, width - margin - beside_x)

        if n_rows == 1:                              # pills sit to the right of the search box
            pills_x, pills_y = beside_x, row_y + search_h / 2 - 18
            y = row_y + search_h + 24
        else:                                        # not enough room: pills go below the search box
            placed, n_rows = self._pill_rows(layout, content_w)
            pills_x, pills_y = margin, row_y + search_h + 12
            y = pills_y + n_rows * 36 + (n_rows - 1) * 10 + 22

        self._draw_search(margin, row_y, search_w, search_h)
        self._draw_pills(placed, pills_x, pills_y)

        # ---- grid geometry
        cols = max(1, min(MAX_COLS, int((content_w + GAP) // (MIN_CARD_W + GAP))))
        card_w = (content_w - GAP * (cols - 1)) / cols
        per_page = cols * PAGE_ROWS

        pages = max(1, math.ceil(len(self.filtered) / per_page))
        self.page = max(1, min(self.page, pages))

        start = (self.page - 1) * per_page
        visible = self.filtered[start:start + per_page]

        grid_top = y

        if not visible:
            self._draw_empty(margin, grid_top, content_w)
        else:
            for index, ds in enumerate(visible):
                row, col = divmod(index, cols)
                x = margin + col * (card_w + GAP)
                cy = grid_top + row * (CARD_H + GAP)
                self._draw_card(ds, x, cy, card_w, CARD_H, index)

        grid_bottom = grid_top + PAGE_ROWS * CARD_H + (PAGE_ROWS - 1) * GAP

        # ---- footer
        footer_y = max(grid_bottom + 30, view_h - 34)

        self._text(
            margin,
            footer_y,
            f"Showing {len(visible)} of {len(self.filtered)} datasets",
            self.f_footer,
            MUTED,
            anchor="w",
        )

        if pages > 1:
            self._draw_pagination(width - margin, footer_y, pages)

        total_h = footer_y + 34
        self.cv.configure(scrollregion=(0, 0, width, max(view_h, total_h)))

        if total_h <= view_h:
            self.cv.yview_moveto(0)

        self._draw_background(width, max(view_h, total_h))

    # ------------------------------------------------------------ navbar

    def _draw_navbar(self, width):
        x1, y1, x2, y2 = 18, 8, width - 18, 70
        cy = (y1 + y2) / 2

        self._rr(x1, y1, x2, y2, 31, fill=NAV_BG, outline="", tags="ui")

        # logo (layers icon + wordmark)
        lx = x1 + 40
        white = {"fill": "", "outline": WHITE, "width": 2, "tags": "ui"}
        self.cv.create_polygon(
            lx, cy - 13, lx + 14, cy - 6, lx, cy + 1, lx - 14, cy - 6, **white
        )
        for dy in (6, 12):
            self.cv.create_line(
                lx - 14, cy - 6 + dy, lx, cy + 1 + dy, lx + 14, cy - 6 + dy,
                fill=WHITE, width=2, tags="ui",
            )

        self._text(lx + 28, cy, "ML", self.f_logo, WHITE, anchor="w")
        ml_w = self.f_logo.measure("ML ")
        self._text(lx + 28 + ml_w, cy, "Studio", self.f_logo, "#E879B9", anchor="w")

        # round icon buttons (info, settings)
        for i, kind in enumerate(("settings", "info")):
            bx = x2 - 38 - i * 48
            self.cv.create_oval(
                bx - 19, cy - 19, bx + 19, cy + 19, fill=NAV_BTN, outline="", tags="ui"
            )
            self.cv.create_oval(
                bx - 8, cy - 8, bx + 8, cy + 8, outline=WHITE, width=2, tags="ui"
            )
            if kind == "info":
                self._text(bx, cy, "i", self.f_badge, WHITE, anchor="center")
            else:
                self.cv.create_oval(
                    bx - 3, cy - 3, bx + 3, cy + 3, outline=WHITE, width=2, tags="ui"
                )

        # menu items (right to left)
        right = x2 - 38 - 48 - 19 - 22

        for n, label in enumerate(reversed(self.NAV_ITEMS)):
            w = self.f_nav.measure(label) + 40
            left = right - w
            tag = f"nav{n}"
            active = label == "Datasets"

            if active:
                self._rr(left, cy - 21, right, cy + 21, 21, fill=ACTIVE_PILL,
                         outline="", tags=("ui", tag))

            self._text(
                (left + right) / 2, cy, label, self.f_nav,
                INK if active else WHITE, anchor="center", tags=("ui", tag),
            )

            if self.on_navigate:
                self._clickable(tag, lambda name=label: self.on_navigate(name))

            right = left - 4

    # ------------------------------------------------------------ search + filter pills

    def _draw_search(self, x, y, w, h):
        cy = y + h / 2

        self._rr(x, y, x + w, y + h, h / 2, fill=WHITE, outline="#EFE6F8", tags="ui")

        # magnifier
        mx = x + 24
        self.cv.create_oval(mx - 7, cy - 8, mx + 5, cy + 4, outline=MUTED, width=2, tags="ui")
        self.cv.create_line(mx + 4, cy + 3, mx + 10, cy + 9, fill=MUTED, width=2,
                            capstyle="round", tags="ui")

        self.cv.coords(self._search_win, x + 46, cy)
        self.cv.itemconfigure(self._search_win, width=w - 64, height=h - 14)

    def _pill_layout(self):
        """[(label, x_offset, width, group)] for the level and task pills."""
        layout, x = [], 0

        groups = (["All levels"] + LEVELS, TASKS)
        for g, labels in enumerate(groups):
            for label in labels:
                w = self.f_pill.measure(label) + 40
                layout.append((label, x, w, g))
                x += w + 8
            x += 18  # extra gap between the two groups

        return layout

    def _pill_rows(self, layout, available_w):
        """Place pills in one row if they fit, else one row per group. Returns (placed, n_rows)."""
        total = layout[-1][1] + layout[-1][2]

        if total <= available_w:
            return [(label, dx, w, g, 0) for label, dx, w, g in layout], 1

        placed, offset = [], {0: 0, 1: 0}
        first_of_group = {}

        for label, dx, w, g in layout:
            first_of_group.setdefault(g, dx)
            placed.append((label, dx - first_of_group[g], w, g, g))

        return placed, 2

    def _draw_pills(self, layout, x0, y):
        h = 36

        for n, (label, dx, w, group, row) in enumerate(layout):
            top = y + row * (h + 10)

            if group == 0:
                selected = (label == "All levels" and self.level_filter is None) or (
                    label == self.level_filter
                )
            else:
                selected = label == self.task_filter

            tag = f"pill{n}"
            left = x0 + dx

            self._rr(
                left, top, left + w, top + h, h / 2,
                fill=DARK_BTN if selected else "#FBF8FE",
                outline=DARK_BTN if selected else "#EFE6F8",
                tags=("ui", tag),
            )
            self._text(
                left + w / 2, top + h / 2, label, self.f_pill,
                WHITE if selected else INK_SOFT, anchor="center", tags=("ui", tag),
            )

            if group == 0:
                command = (lambda: self._set_level(None)) if label == "All levels" \
                    else (lambda lv=label: self._set_level(lv))
            else:
                command = lambda t=label: self._toggle_task(t)

            self._clickable(tag, command)

    # ------------------------------------------------------------ cards

    def _draw_card(self, ds, x, y, w, h, index):
        style = TASK_STYLE.get(ds.task, DEFAULT_STYLE)
        tag = f"card{index}"
        tags = ("ui", tag)
        pad = 18

        # soft shadow
        for spread, color in ((6, "#F1EAF9"), (4, "#EDE4F7"), (2, "#E9DFF5")):
            self._rr(x - spread, y - spread + 5, x + w + spread, y + h + spread + 5,
                     24, fill=color, outline="", tags=tags)

        body = self._rr(x, y, x + w, y + h, 22, fill=style["tint"],
                        outline=CARD_BORDER, width=1, tags=tags)

        # decorative sparkline (deterministic per dataset)
        rng = random.Random(ds.name)
        n_pts = 9
        sx0 = x + w * 0.55
        pts = []
        for i in range(n_pts):
            px = sx0 + (x + w - sx0) * i / (n_pts - 1)
            py = y + 64 + rng.uniform(-10, 10)
            pts += [px, py]
        self.cv.create_line(*pts, smooth=True, fill=style["line"], width=2, tags=tags)

        # icon
        ix, iy, isz = x + pad, y + pad, 38
        self._rr(ix, iy, ix + isz, iy + isz, 11, fill=style["icon"], outline="", tags=tags)
        self._draw_glyph(ds.task, ix + isz / 2, iy + isz / 2, tags)

        # level badge
        bw = self.f_badge.measure(ds.level) + 38
        bx2 = x + w - pad
        by = iy + isz / 2
        self._rr(bx2 - bw, by - 13, bx2, by + 13, 13, fill="#EEE6F8", outline="", tags=tags)
        dot = LEVEL_DOT.get(ds.level, MUTED)
        self.cv.create_oval(bx2 - bw + 11, by - 4, bx2 - bw + 19, by + 4,
                            fill=dot, outline="", tags=tags)
        self._text(bx2 - bw + 26, by, ds.level, self.f_badge, INK_SOFT,
                   anchor="w", tags=tags)

        # title (max two lines) and subtitle
        text_w = w - 2 * pad
        lines = _wrap(ds.name, self.f_card_title, text_w, 2)
        line_h = 23
        ty = y + 66

        for i, line in enumerate(lines):
            self._text(x + pad, ty + i * line_h, line, self.f_card_title, INK, tags=tags)

        sub = _fit(f"{ds.domain} · {ds.task}", self.f_card_sub, text_w)
        self._text(x + pad, ty + len(lines) * line_h + 2, sub, self.f_card_sub, MUTED, tags=tags)

        # chips
        chip_cy = y + h - 34
        cx = x + pad

        for label in (f"{ds.rows:,} rows", f"{ds.columns:,} cols"):
            cw = self.f_chip.measure(label) + 26
            self._rr(cx, chip_cy - 14, cx + cw, chip_cy + 14, 14, fill=CHIP_BG,
                     outline="", tags=tags)
            self._text(cx + cw / 2, chip_cy, label, self.f_chip, INK_SOFT,
                       anchor="center", tags=tags)
            cx += cw + 8

        # arrow button
        ax, ay, ar = x + w - pad - 16, chip_cy, 16
        self.cv.create_oval(ax - ar, ay - ar, ax + ar, ay + ar, fill=DARK_BTN,
                            outline="", tags=tags)
        self.cv.create_line(ax - 6, ay, ax + 6, ay, fill=WHITE, width=2,
                            capstyle="round", tags=tags)
        self.cv.create_line(ax + 1, ay - 5, ax + 6, ay, ax + 1, ay + 5, fill=WHITE,
                            width=2, capstyle="round", joinstyle="round", tags=tags)

        # interaction
        def enter(_e, item=body, color=style["icon"]):
            self.cv.itemconfigure(item, outline=color, width=2)
            self.cv.config(cursor="hand2")

        def leave(_e, item=body):
            self.cv.itemconfigure(item, outline=CARD_BORDER, width=1)
            self.cv.config(cursor="")

        self.cv.tag_bind(tag, "<Enter>", enter)
        self.cv.tag_bind(tag, "<Leave>", leave)
        self.cv.tag_bind(tag, "<Button-1>", lambda e, d=ds: self.on_open(d))

    def _draw_glyph(self, task, cx, cy, tags):
        kw = {"fill": WHITE, "width": 2, "capstyle": "round", "joinstyle": "round", "tags": tags}

        if task == "Regression":
            pts = [(-9, 6), (-3, -1), (2, 3), (9, -6)]
            self.cv.create_line(*[c for p in pts for c in (cx + p[0], cy + p[1])], **kw)
            for px, py in pts:
                self.cv.create_oval(cx + px - 2, cy + py - 2, cx + px + 2, cy + py + 2,
                                    fill=WHITE, outline="", tags=tags)
        elif task == "Classification":
            for dy in (-6, 6):
                self.cv.create_oval(cx - 11, cy + dy - 3.5, cx - 4, cy + dy + 3.5,
                                    outline=WHITE, width=2, tags=tags)
                self.cv.create_line(cx + 1, cy + dy, cx + 10, cy + dy, **kw)
        elif task == "Clustering":
            for dx, dy in ((0, -7), (-8, 5), (8, 5)):
                self.cv.create_oval(cx + dx - 3.5, cy + dy - 3.5, cx + dx + 3.5, cy + dy + 3.5,
                                    outline=WHITE, width=2, tags=tags)
        else:
            for dx in (-6, 6):
                for dy in (-6, 6):
                    self.cv.create_oval(cx + dx - 2.5, cy + dy - 2.5, cx + dx + 2.5, cy + dy + 2.5,
                                        fill=WHITE, outline="", tags=tags)

    # ------------------------------------------------------------ empty state + pagination

    def _draw_empty(self, x, y, w):
        h = PAGE_ROWS * CARD_H + (PAGE_ROWS - 1) * GAP
        self._rr(x, y, x + w, y + h, 24, fill="#FBF8FE", outline=CARD_BORDER, tags="ui")

        if self.all:
            title, hint = "No datasets match your filters", "Try a different search or clear a filter."
        else:
            title = "No datasets yet"
            hint = "Add a dataset script that writes <Task>/Metadata/<name>.json and it will appear here."

        self._text(x + w / 2, y + h / 2 - 14, title, self.f_empty, INK, anchor="center")
        self._text(x + w / 2, y + h / 2 + 20, hint, self.f_sub, MUTED, anchor="center")

    def _draw_pagination(self, right, cy, pages):
        size, gap = 32, 8
        tokens = _page_tokens(pages, self.page)
        x = right - len(tokens) * size - (len(tokens) - 1) * gap

        for n, token in enumerate(tokens):
            tag = f"pg{n}"
            cx = x + size / 2

            if token == "…":
                self._text(cx, cy, "…", self.f_page, MUTED, anchor="center")
            else:
                active = token == self.page
                self.cv.create_oval(
                    x, cy - size / 2, x + size, cy + size / 2,
                    fill=DARK_BTN if active else WHITE,
                    outline=DARK_BTN if active else "#EFE6F8",
                    tags=("ui", tag),
                )
                self._text(cx, cy, str(token), self.f_page, WHITE if active else INK_SOFT,
                           anchor="center", tags=("ui", tag))
                self._clickable(tag, lambda t=token: self._go_page(t))

            x += size + gap

# endregion


# region Standalone preview

if __name__ == "__main__":
    root = tk.Tk()
    root.title("ML Studio - Datasets")
    root.geometry("1280x760")
    root.configure(bg=BG_TOP)

    page = DatasetsPage(
        root,
        show_navbar=True,
        on_open=lambda ds: print("Open:", ds.name, ds.path),
        on_navigate=lambda name: print("Navigate:", name),
    )
    page.pack(fill="both", expand=True)

    root.mainloop()

# endregion