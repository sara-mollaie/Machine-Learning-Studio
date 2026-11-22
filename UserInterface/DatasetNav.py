"""Navigation for dataset prediction and evaluation pages."""
from PIL import Image, ImageDraw, ImageTk
from UserInterface.HomeModule import HDR_TOP, HDR_H, HDR_CY
from UserInterface.Theme import theme


def _line_icon(kind, size, color):
    factor = 4
    n = size * factor
    image = Image.new("RGBA", (n, n))
    draw = ImageDraw.Draw(image)
    width = max(2, round(n * 0.065))

    def line(points):
        draw.line([(round(x * n), round(y * n)) for x, y in points],
                  fill=color, width=width, joint="curve")

    if kind == "evaluate":
        draw.rounded_rectangle(
            (n * .18, n * .15, n * .82, n * .91),
            radius=round(n * .08), outline=color, width=width,
        )
        draw.rounded_rectangle(
            (n * .34, n * .06, n * .66, n * .25),
            radius=round(n * .04), fill=color,
        )
        line([(.32, .55), (.46, .69), (.70, .40)])
    else:
        line([(.12, .12), (.12, .88), (.90, .88)])
        line([(.25, .68), (.45, .45), (.62, .57), (.86, .25)])
    return image.resize((size, size), Image.Resampling.LANCZOS)


def draw_dataset_navbar(view, active=None):
    c = theme.c
    cy = HDR_CY
    view.panel(23, HDR_TOP, 1626, HDR_H, 41, *c("hdr"), alpha=255)
    view.pic(88, cy, "ml_studio_logo", 56, crop=True)
    title = "Machine Learning Prediction Studio"
    view.text(130, cy, title, 25, "#ffffff", True)

    items = [
        ("Evaluation", "evaluate", "evaluate"),
        ("Visualization", "visualization", "visualization"),
        ("Algorithms", "algorithms", "nav_algorithms"),
    ]
    # Fit the three links between the brand and the utility buttons.
    left = 130 + view.text_width(title, 25, True) + 20
    available = 1453 - left
    font_size, icon_size, padding, icon_gap, item_gap = 20, 30, 12, 8, 12
    widths = [
        2 * padding + icon_size + icon_gap + view.text_width(label, font_size, True)
        for label, _, _ in items
    ]
    total = sum(widths) + item_gap * 2
    left += max(0, (available - total) / 2)

    for (label, page, icon), width in zip(items, widths):
        tag = "nav_" + page
        center = left + width / 2
        selected = page == active
        view.nav_hover(tag, center, cy, width, 50, page)
        if selected:
            pill = view.rounded(width, 50, 25, c("pill"))
            view._refs[("dataset_nav_pill", page)] = pill
            view.create_image(view.s(center), view.s(cy), image=pill, tags=(tag,))
        color = c("pill_text") if selected else "#ffffff"
        icon_x = left + padding + icon_size / 2
        if page in ("evaluate", "visualization"):
            key = ("dataset_nav_icon", page, color, view.k)
            if key not in view._refs:
                size = max(1, int(round(view.s(icon_size))))
                view._refs[key] = ImageTk.PhotoImage(
                    _line_icon(page, size, color), master=view
                )
            view.create_image(view.s(icon_x), view.s(cy),
                              image=view._refs[key], tags=(tag,))
        else:
            view.pic(icon_x, cy, icon, icon_size, (tag,), crop=True)
        view.text(left + padding + icon_size + icon_gap, cy, label,
                  font_size, color, True, tags=(tag,))
        view.tag_bind(tag, "<Button-1>", lambda event, target=page: view.on_navigate(target))
        left += width + item_gap

    for icon, x, page in [("nav_info", 1495, "info"), ("nav_settings", 1590, "settings")]:
        tag = "nav_" + page
        view.nav_hover(tag, x, cy, 54, 54, page)
        view.pic(x, cy, icon, 40, (tag,), crop=True)
