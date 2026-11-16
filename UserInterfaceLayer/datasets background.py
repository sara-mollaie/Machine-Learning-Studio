"""
make_datasets_background.py
Creates the Datasets page background (1672x941, same size as the Home background):

    assets/page_images/datasets_background.png        (light theme)
    assets/page_images/datasets_background_dark.png   (dark theme)

Style follows the "Algorithms" reference: soft lavender/pink waves, a glass title
panel with a gradient title, three dots, a glowing pedestal at the bottom and a
small illustration at the top right.

Usage:   python make_datasets_background.py [output_folder] [Poppins-Bold.ttf]
"""

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

W, H = 1672, 941
TITLE = "Datasets"

FONT_CANDIDATES = [
    "Poppins-Bold.ttf",
    "C:/Windows/Fonts/Poppins-Bold.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


# ----------------------------------------------------------------- palettes

LIGHT = dict(
    name="light",
    grad=("#F9F3FE", "#EFE4FA", "#E3D4F6"),
    blobs=[  # cx, cy, rx, ry, color, alpha
        (250, 560, 620, 360, "#F8BEDC", 120),
        (1520, 170, 760, 470, "#BDA0F1", 150),
        (860, 120, 520, 260, "#FFFFFF", 120),
        (1300, 700, 520, 300, "#E9D4F7", 110),
    ],
    waves=[  # base_y, amplitude, freq, phase, thickness, c1, c2, alpha
        (560, 95, 0.0042, 0.4, 200, "#F6C3E0", "#CDB4F3", 105),
        (250, 70, 0.0050, 2.4, 160, "#D3BDF6", "#B79AEF", 120),
        (780, 40, 0.0070, 1.1, 90, "#FFFFFF", "#E7D7F8", 120),
    ],
    glass_top=(255, 255, 255, 170), glass_bot=(251, 240, 252, 120), glass_edge=(255, 255, 255, 235),
    shadow=(110, 60, 170), shadow_a=70,
    tile=(255, 255, 255, 185),
    title_stops=[(0.0, "#2B1450"), (0.45, "#4B1F78"), (1.0, "#E0529A")],
    icon_stops=[(0.0, "#3A1B74"), (1.0, "#C84C98")],
    line=(190, 160, 235),
    dots=("#8C6BE6", "#D8509A", "#B05BD0"),
    plat=("#E8DAFB", "#C9B1F1", "#FFFFFF"), plat_glow="#F29AC8",
    disc=("#5B33C2", "#9A6CFA", "#B99AFF"), ring="#F06AA8",
    leaf=("#9B6BE0", "#4B2A9A"), pot=("#FFFFFF", "#CDB6F0"),
    pedestal=("#F4DDF3", "#FFFFFF", "#F28DC0"),
)

DARK = dict(
    name="dark",
    grad=("#150A2B", "#211139", "#2C1749"),
    blobs=[
        (260, 560, 620, 360, "#8A2E7E", 105),
        (1520, 170, 780, 470, "#5B34B8", 150),
        (860, 120, 520, 260, "#6C48C9", 55),
        (1300, 720, 520, 300, "#3C2270", 130),
    ],
    waves=[
        (560, 95, 0.0042, 0.4, 200, "#7B2D78", "#4A2E92", 105),
        (250, 70, 0.0050, 2.4, 160, "#5B3BB0", "#3D2580", 130),
        (780, 40, 0.0070, 1.1, 90, "#8E6BE0", "#3B2470", 70),
    ],
    glass_top=(255, 255, 255, 34), glass_bot=(255, 255, 255, 16), glass_edge=(200, 170, 255, 120),
    shadow=(0, 0, 0), shadow_a=130,
    tile=(255, 255, 255, 30),
    title_stops=[(0.0, "#F7F1FF"), (0.5, "#D9C2F7"), (1.0, "#EE86BC")],
    icon_stops=[(0.0, "#C9B2FA"), (1.0, "#F08CC0")],
    line=(150, 120, 210),
    dots=("#A58BF2", "#EE6FAF", "#C47BE0"),
    plat=("#4A2F88", "#2A1A55", "#8F6BE8"), plat_glow="#C0489A",
    disc=("#6A3FDB", "#A67BFF", "#CBB3FF"), ring="#F078B4",
    leaf=("#9B6BE0", "#46298F"), pot=("#8E72D0", "#4D3590"),
    pedestal=("#4A2F88", "#8F6BE8", "#E0529A"),
)


# ----------------------------------------------------------------- helpers

def font(size):
    for p in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def stops_gradient(w, h, stops, horizontal=True):
    """RGB gradient image from [(t, '#hex'), ...]."""
    n = w if horizontal else h
    t = np.linspace(0, 1, n)
    ts = [s[0] for s in stops]
    cols = np.array([hex_rgb(s[1]) for s in stops], dtype=np.float32)
    ch = [np.interp(t, ts, cols[:, i]) for i in range(3)]
    row = np.stack(ch, axis=-1)
    arr = np.tile(row[None, :, :], (h, 1, 1)) if horizontal else np.tile(row[:, None, :], (1, w, 1))
    return Image.fromarray(arr.astype("uint8"), "RGB")


def rounded_mask(w, h, r, ss=3):
    m = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), r * ss, fill=255)
    return m.resize((w, h), Image.LANCZOS)


def radial(w, h, cx, cy, rx, ry, color, alpha):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    a = np.exp(-d * 1.6) * alpha
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    layer[..., :3] = hex_rgb(color)
    layer[..., 3] = np.clip(a, 0, 255).astype("uint8")
    return Image.fromarray(layer, "RGBA")


def paste_layer(base, layer, xy=(0, 0)):
    base.alpha_composite(layer, xy)


def with_alpha(img_rgb, mask, alpha=255):
    out = img_rgb.convert("RGBA")
    out.putalpha(mask.point(lambda v: v * alpha // 255))
    return out


def blur(img, r):
    return img.filter(ImageFilter.GaussianBlur(r))


# ----------------------------------------------------------------- background

def background(p):
    img = stops_gradient(W, H, [(0, p["grad"][0]), (0.5, p["grad"][1]), (1, p["grad"][2])],
                         horizontal=False).convert("RGBA")

    for cx, cy, rx, ry, color, alpha in p["blobs"]:
        paste_layer(img, radial(W, H, cx, cy, rx, ry, color, alpha))

    # soft ribbons
    for base_y, amp, freq, phase, thick, c1, c2, alpha in p["waves"]:
        xs = np.arange(0, W + 8, 8)
        top = base_y + amp * np.sin(xs * freq + phase) + 0.45 * amp * np.sin(xs * freq * 2.3 + phase * 1.7)
        bot = top + thick * (0.6 + 0.4 * np.sin(xs * freq * 0.7 + phase))
        mask = Image.new("L", (W, H), 0)
        poly = [(float(x), float(y)) for x, y in zip(xs, top)] + \
               [(float(x), float(y)) for x, y in zip(xs[::-1], bot[::-1])]
        ImageDraw.Draw(mask).polygon(poly, fill=255)
        mask = blur(mask, 9)
        paste_layer(img, with_alpha(stops_gradient(W, H, [(0, c1), (1, c2)]), mask, alpha))

        edge = Image.new("L", (W, H), 0)
        ImageDraw.Draw(edge).line([(float(x), float(y)) for x, y in zip(xs, top)], fill=255, width=3)
        paste_layer(img, with_alpha(Image.new("RGB", (W, H), "white"), blur(edge, 2),
                                    150 if p["name"] == "light" else 60))
    return img


# ----------------------------------------------------------------- pedestal glow (bottom)

def pedestal(img, p):
    ss = 2
    layer = Image.new("RGBA", (W * ss, 220 * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy = W // 2 * ss, 120 * ss

    c_out, c_in, c_ring = (hex_rgb(c) for c in p["pedestal"])
    d.ellipse((cx - 600 * ss, cy - 62 * ss, cx + 600 * ss, cy + 62 * ss), fill=c_out + (95,))
    d.ellipse((cx - 540 * ss, cy - 46 * ss, cx + 540 * ss, cy + 46 * ss), fill=c_in + (150,))
    d.ellipse((cx - 470 * ss, cy - 32 * ss, cx + 470 * ss, cy + 32 * ss), fill=c_in + (190,))
    d.ellipse((cx - 560 * ss, cy - 52 * ss, cx + 560 * ss, cy + 52 * ss), outline=c_ring + (150,), width=3 * ss)
    layer = blur(layer, 5 * ss).resize((W, 220), Image.LANCZOS)
    img.alpha_composite(layer, (0, 745))


# ----------------------------------------------------------------- title panel

def database_icon(size, stops):
    """Gradient-stroke database cylinder (same style as the layers logo)."""
    ss = 4
    S = size * ss
    mask = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(mask)

    cx = S / 2
    rx, ry = S * 0.36, S * 0.115
    y_top, y_bot = S * 0.24, S * 0.78
    w = int(S * 0.075)

    d.ellipse((cx - rx, y_top - ry, cx + rx, y_top + ry), outline=255, width=w)
    d.line((cx - rx, y_top, cx - rx, y_bot), fill=255, width=w)
    d.line((cx + rx, y_top, cx + rx, y_bot), fill=255, width=w)
    for y in (y_top + (y_bot - y_top) * 0.36, y_top + (y_bot - y_top) * 0.68, y_bot):
        d.arc((cx - rx, y - ry, cx + rx, y + ry), 0, 180, fill=255, width=w)

    mask = mask.resize((size, size), Image.LANCZOS)
    grad = Image.new("RGB", (size, size))
    g = stops_gradient(size * 2, size, stops).resize((size, size))
    grad.paste(g)
    # diagonal feel: blend with vertical gradient
    gv = stops_gradient(size, size, stops, horizontal=False)
    grad = Image.blend(grad, gv, 0.5)
    return with_alpha(grad, mask)


def title_panel(img, p):
    title_font = font(78)
    bbox = title_font.getbbox(TITLE)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    px, py, ph = 64, 102, 118
    tile = 88
    pw = 20 + tile + 28 + text_w + 54

    # shadow
    pad = 70
    sh = Image.new("RGBA", (pw + 2 * pad, ph + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((pad, pad + 14, pad + pw, pad + 14 + ph), 46,
                                         fill=p["shadow"] + (p["shadow_a"],))
    img.alpha_composite(blur(sh, 22), (px - pad, py - pad))

    # glass body (vertical alpha gradient)
    ss = 3
    body = Image.new("RGBA", (pw, ph))
    top, bot = p["glass_top"], p["glass_bot"]
    arr = np.zeros((ph, pw, 4), dtype=np.uint8)
    for c in range(4):
        arr[..., c] = np.linspace(top[c], bot[c], ph)[:, None]
    body = Image.fromarray(arr, "RGBA")
    mask = rounded_mask(pw, ph, 46)
    body.putalpha(ImageChops.multiply(body.getchannel("A"), mask))
    img.alpha_composite(body, (px, py))

    ring = Image.new("L", (pw * ss, ph * ss), 0)
    ImageDraw.Draw(ring).rounded_rectangle((0, 0, pw * ss - 1, ph * ss - 1), 46 * ss, outline=255, width=2 * ss)
    ring = ring.resize((pw, ph), Image.LANCZOS)
    edge = Image.new("RGBA", (pw, ph), p["glass_edge"][:3] + (0,))
    edge.putalpha(ring.point(lambda v: v * p["glass_edge"][3] // 255))
    img.alpha_composite(edge, (px, py))

    # icon tile
    tx, ty = px + 20, py + (ph - tile) // 2
    tile_img = Image.new("RGBA", (tile, tile), p["tile"][:3] + (0,))
    tile_img.putalpha(rounded_mask(tile, tile, 30).point(lambda v: v * p["tile"][3] // 255))
    img.alpha_composite(tile_img, (tx, ty))
    icon = database_icon(68, p["icon_stops"])
    img.alpha_composite(icon, (tx + (tile - 68) // 2, ty + (tile - 68) // 2))

    # gradient title text
    tw_x = tx + tile + 28
    mask = Image.new("L", (text_w + 20, text_h + 40), 0)
    ImageDraw.Draw(mask).text((10 - bbox[0], 10 - bbox[1]), TITLE, font=title_font, fill=255)
    grad = stops_gradient(mask.width, mask.height, p["title_stops"])
    layer = with_alpha(grad, mask)
    img.alpha_composite(layer, (tw_x - 10, py + (ph - text_h) // 2 - 10 - 2))

    # decorative dots + lines under the panel
    cx = px + pw // 2
    y = py + ph + 22
    for side in (-1, 1):
        fade = Image.new("RGBA", (300, 6), (0, 0, 0, 0))
        fd = ImageDraw.Draw(fade)
        for i in range(300):
            t = i / 299
            a = int(190 * (t if side == -1 else 1 - t))
            fd.line((i, 2, i, 3), fill=p["line"] + (a,))
        img.alpha_composite(fade, (cx - 38 - 300 if side == -1 else cx + 38, y - 2))

    ss = 3
    dots = Image.new("RGBA", (120 * ss, 40 * ss), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dots)
    for i, color in enumerate(p["dots"]):
        x = (60 + (i - 1) * 32) * ss
        dd.ellipse((x - 8 * ss, 20 * ss - 8 * ss, x + 8 * ss, 20 * ss + 8 * ss), fill=hex_rgb(color) + (255,))
    dots = dots.resize((120, 40), Image.LANCZOS)
    img.alpha_composite(dots, (cx - 60, y - 19))
    return pw


# ----------------------------------------------------------------- illustration (top right)

def leaf(angle, length, width, colors, ss):
    size = int(length * 2.4)
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    cx, cy = size // 2, size // 2
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    d.ellipse((cx - width, cy - length, cx + width, cy), fill=255)
    grad = stops_gradient(size, size, [(0, colors[0]), (1, colors[1])], horizontal=False)
    layer = with_alpha(grad, mask)
    return layer.rotate(angle, center=(cx, cy), resample=Image.BICUBIC)


def illustration(img, p):
    ss = 2
    ox, oy, w, h = 1090, 70, 560, 250
    cv = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(cv)

    def S(v):
        return int(v * ss)

    pc1, pc2, pc3 = (hex_rgb(c) for c in p["plat"])
    cx, cy = 300, 205

    # platform
    d.ellipse((S(cx - 245), S(cy - 38), S(cx + 245), S(cy + 52)), fill=pc2 + (230,))
    d.ellipse((S(cx - 245), S(cy - 52), S(cx + 245), S(cy + 38)), fill=pc1 + (255,))
    d.ellipse((S(cx - 215), S(cy - 44), S(cx + 215), S(cy + 22)), fill=pc3 + ((120 if p["name"] == "light" else 60),))
    d.ellipse((S(cx - 245), S(cy - 52), S(cx + 245), S(cy + 38)), outline=hex_rgb(p["plat_glow"]) + (170,), width=S(2))

    # glow under the object
    glow = Image.new("RGBA", cv.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse((S(cx - 110), S(cy - 36), S(cx + 110), S(cy + 22)), fill=hex_rgb(p["plat_glow"]) + (200,))
    cv.alpha_composite(blur(glow, S(14)))
    d = ImageDraw.Draw(cv)

    # hologram rings
    for k, a in ((120, 150), (150, 90)):
        d.ellipse((S(cx - k), S(cy - 30 - k * 0.05), S(cx + k), S(cy + 14 + k * 0.05)),
                  outline=hex_rgb(p["ring"]) + (a,), width=S(2))

    # three stacked data discs
    c_dark, c_mid, c_top = (hex_rgb(c) for c in p["disc"])
    rx, ry = 92, 24
    for i, yc in enumerate((cy - 40, cy - 92, cy - 144)):
        # body
        body = Image.new("RGBA", cv.size, (0, 0, 0, 0))
        bm = Image.new("L", cv.size, 0)
        bd = ImageDraw.Draw(bm)
        bd.ellipse((S(cx - rx), S(yc + 4 - ry), S(cx + rx), S(yc + 4 + ry)), fill=255)
        bd.rectangle((S(cx - rx), S(yc - 26), S(cx + rx), S(yc + 4)), fill=255)
        grad = stops_gradient(cv.size[0], cv.size[1],
                              [(0.0, p["disc"][0]), (0.42, p["disc"][1]), (1.0, p["disc"][0])])
        body = with_alpha(grad, bm)
        cv.alpha_composite(body)
        d = ImageDraw.Draw(cv)
        # top face
        d.ellipse((S(cx - rx), S(yc - 26 - ry), S(cx + rx), S(yc - 26 + ry)), fill=c_top + (255,))
        d.ellipse((S(cx - rx + 10), S(yc - 26 - ry + 5), S(cx + rx - 10), S(yc - 26 + ry - 5)),
                  fill=c_mid + (255,))
        # glossy rim
        d.arc((S(cx - rx), S(yc - 26 - ry), S(cx + rx), S(yc - 26 + ry)), 180, 360, fill=(255, 255, 255, 190), width=S(2))
        d.arc((S(cx - rx), S(yc + 4 - ry), S(cx + rx), S(yc + 4 + ry)), 0, 180, fill=(255, 255, 255, 80), width=S(2))

    # plant (left of the platform)
    px, py = 108, 182
    for ang, ln, wd in ((-58, 74, 15), (-34, 92, 16), (-10, 104, 17), (14, 96, 16), (38, 78, 15), (62, 60, 13)):
        lf = leaf(ang, S(ln), S(wd), p["leaf"], ss)
        cv.alpha_composite(lf, (S(px) - lf.width // 2, S(py) - lf.height // 2))
    d = ImageDraw.Draw(cv)
    c_pot1, c_pot2 = (hex_rgb(c) for c in p["pot"])
    d.polygon([(S(px - 26), S(py + 2)), (S(px + 26), S(py + 2)), (S(px + 20), S(py + 38)), (S(px - 20), S(py + 38))],
              fill=c_pot2 + (255,))
    d.ellipse((S(px - 26), S(py - 6), S(px + 26), S(py + 10)), fill=c_pot1 + (255,))

    # sparkles
    for sx, sy, sr in ((470, 60, 9), (500, 110, 6), (150, 60, 6), (420, 25, 5)):
        c = hex_rgb(p["plat_glow"]) + (230,)
        d.polygon([(S(sx), S(sy - sr)), (S(sx + sr * 0.3), S(sy - sr * 0.3)), (S(sx + sr), S(sy)),
                   (S(sx + sr * 0.3), S(sy + sr * 0.3)), (S(sx), S(sy + sr)), (S(sx - sr * 0.3), S(sy + sr * 0.3)),
                   (S(sx - sr), S(sy)), (S(sx - sr * 0.3), S(sy - sr * 0.3))], fill=c)

    k = 0.68
    cv = cv.resize((int(w * k), int(h * k)), Image.LANCZOS)
    img.alpha_composite(cv, (1236, 80))



def simple_icon(img, p):
    """One clean glossy database cylinder (replaces the busy platform illustration)."""
    ss = 3
    cx, cy = 1508, 168
    rx, ry = 62, 19
    y_top, y_bot = cy - 52, cy + 52

    # soft glow behind / under the icon
    img.alpha_composite(radial(W, H, cx, cy + 40, 170, 90, p["plat_glow"], 150 if p["name"] == "light" else 120))
    img.alpha_composite(radial(W, H, cx, cy - 10, 150, 110, "#FFFFFF" if p["name"] == "light" else "#8F6BE8",
                               110 if p["name"] == "light" else 70))

    box = (360, 300)
    ox, oy = cx - box[0] // 2, cy - box[1] // 2
    lay = Image.new("RGBA", (box[0] * ss, box[1] * ss), (0, 0, 0, 0))

    def S(v):
        return int(v * ss)

    lx, ly = box[0] // 2, box[1] // 2
    top, bot = ly - 52, ly + 52

    # body (horizontal shading) + bottom cap
    mask = Image.new("L", lay.size, 0)
    md = ImageDraw.Draw(mask)
    md.rectangle((S(lx - rx), S(top), S(lx + rx), S(bot)), fill=255)
    md.ellipse((S(lx - rx), S(bot - ry), S(lx + rx), S(bot + ry)), fill=255)
    shade = stops_gradient(lay.size[0], lay.size[1],
                           [(0.28, p["disc"][0]), (0.46, p["disc"][1]), (0.72, p["disc"][0])])
    lay.alpha_composite(with_alpha(shade, mask))

    d = ImageDraw.Draw(lay)
    # separators
    for yy in (top + (bot - top) * 0.36, top + (bot - top) * 0.68):
        d.arc((S(lx - rx), S(yy - ry), S(lx + rx), S(yy + ry)), 0, 180, fill=(255, 255, 255, 120), width=S(2.2))
    # top face
    d.ellipse((S(lx - rx), S(top - ry), S(lx + rx), S(top + ry)), fill=hex_rgb(p["disc"][2]) + (255,))
    d.ellipse((S(lx - rx + 9), S(top - ry + 4), S(lx + rx - 9), S(top + ry - 4)), fill=hex_rgb(p["disc"][1]) + (255,))
    d.arc((S(lx - rx), S(top - ry), S(lx + rx), S(top + ry)), 180, 360, fill=(255, 255, 255, 200), width=S(2))
    # gloss streak on the left side
    d.rounded_rectangle((S(lx - rx + 10), S(top + 12), S(lx - rx + 17), S(bot - 14)), S(3), fill=(255, 255, 255, 70))

    lay = lay.resize(box, Image.LANCZOS)
    img.alpha_composite(lay, (ox, oy))


# ----------------------------------------------------------------- main

def make(p):
    img = background(p)
    pedestal(img, p)
    title_panel(img, p)
    simple_icon(img, p)
    return img.convert("RGB")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("assets/page_images")
    if len(sys.argv) > 2:
        FONT_CANDIDATES.insert(0, sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    make(LIGHT).save(out / "datasets_background.png")
    make(DARK).save(out / "datasets_background_dark.png")
    print("saved to", out.resolve())