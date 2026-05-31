import os

from PIL import Image, ImageDraw, ImageFont

TRAINED_DIR = "outputs/diffusion_cfg/trained"
UNTRAINED_DIR = "outputs/diffusion_cfg/untrained"
OUT_PATH = "outputs/diffusion_cfg/diffusion_cfg_grid_combined.png"

# layout
MARGIN = 40
GAP = 28
BANNER_H = 64
TITLE_H = 130
CARD_PAD = 18
RADIUS = 22

# palette
BG = (247, 249, 252)
CARD_BG = (255, 255, 255)
CARD_BORDER = (226, 232, 240)
TITLE_COLOR = (23, 37, 84)
SUBTITLE_COLOR = (100, 116, 139)
SHADOW = (231, 236, 243)

ACCENT_TRAINED = (79, 140, 201)
ACCENT_ALL = (16, 122, 96)
ACCENT_UNTRAINED = (203, 96, 96)


def load_font(size, bold=False):
    names = ("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_rows():
    rows = [("Trained - all classes (mixed)", "Conditional sampling, random labels", ACCENT_ALL,
             os.path.join(TRAINED_DIR, "diffusion_cfg_grid_trained_all.png"))]
    for class_id in range(8):
        rows.append((f"Trained - class {class_id}", "Class-conditional sampling", ACCENT_TRAINED,
                     os.path.join(TRAINED_DIR, f"diffusion_cfg_grid_trained_c{class_id}.png")))
    rows.append(("Untrained", "Randomly initialized network", ACCENT_UNTRAINED,
                 os.path.join(UNTRAINED_DIR, "diffusion_cfg_grid_untrained.png")))
    return rows


def rounded(draw, box, radius, fill=None, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def main():
    rows = build_rows()
    items = [(title, sub, accent, Image.open(p).convert("RGB")) for title, sub, accent, p in rows]

    img_w = max(im.width for *_, im in items)
    card_w = img_w + 2 * CARD_PAD
    canvas_w = card_w + 2 * MARGIN

    title_font = load_font(48, bold=True)
    sub_font = load_font(26)
    banner_font = load_font(30, bold=True)
    banner_sub_font = load_font(20)

    total_h = MARGIN + TITLE_H + GAP
    for *_, im in items:
        total_h += BANNER_H + im.height + 2 * CARD_PAD + GAP
    total_h += MARGIN - GAP

    canvas = Image.new("RGB", (canvas_w, total_h), BG)
    draw = ImageDraw.Draw(canvas)

    # header
    draw.text((MARGIN, MARGIN), "DDPM with Classifier-Free Guidance", fill=TITLE_COLOR, font=title_font)
    draw.text((MARGIN, MARGIN + 64), "Reverse-diffusion sample trajectories on an 8-mode Gaussian mixture",
              fill=SUBTITLE_COLOR, font=sub_font)

    y = MARGIN + TITLE_H + GAP
    for title, sub, accent, im in items:
        card_h = BANNER_H + im.height + 2 * CARD_PAD
        x0 = MARGIN

        # drop shadow + card
        rounded(draw, (x0 + 4, y + 5, x0 + card_w + 4, y + card_h + 5), RADIUS, fill=SHADOW)
        rounded(draw, (x0, y, x0 + card_w, y + card_h), RADIUS, fill=CARD_BG,
                outline=CARD_BORDER, width=2)

        # accent banner
        rounded(draw, (x0, y, x0 + card_w, y + BANNER_H + RADIUS), RADIUS, fill=accent)
        draw.rectangle((x0, y + RADIUS, x0 + card_w, y + BANNER_H), fill=accent)
        # accent stripe on left edge of image area
        draw.rectangle((x0, y + BANNER_H, x0 + 8, y + card_h), fill=accent)

        draw.text((x0 + CARD_PAD, y + 8), title, fill=(255, 255, 255), font=banner_font)
        draw.text((x0 + CARD_PAD, y + 38), sub, fill=(255, 255, 255), font=banner_sub_font)

        # paste image centered
        ix = x0 + CARD_PAD + (img_w - im.width) // 2
        iy = y + BANNER_H + CARD_PAD
        canvas.paste(im, (ix, iy))

        y += card_h + GAP

    canvas.save(OUT_PATH)
    print(f"saved {OUT_PATH} ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()
