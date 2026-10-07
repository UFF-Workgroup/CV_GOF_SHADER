"""Gera montagens GT x metodos e o grafico de ganho por vista a partir de per_view.json."""
import json
import os
from argparse import ArgumentParser

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def load(path, size):
    return Image.open(path).convert("RGB").resize(size, Image.LANCZOS)


def label(img, text):
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 26)
    w = draw.textlength(text, font=font)
    draw.rectangle([0, 0, w + 24, 44], fill=(20, 24, 33))
    draw.text((12, 6), text, font=font, fill=(245, 245, 240))
    return img


def montage(view, dirs, per_view, out, tile=(640, 357)):
    panels = [("Ground truth", dirs["GT"], None)] + [(n, dirs[n], n) for n in dirs if n != "GT"]
    canvas = Image.new("RGB", (tile[0] * 3, tile[1] * 2), (20, 24, 33))
    for i, (name, d, key) in enumerate(panels[:6]):
        fname = next(f for f in sorted(os.listdir(d)) if os.path.splitext(f)[0].split("_")[0] == view)
        text = name if key is None else f"{name}  {per_view[key][view]['PSNR']:.2f} dB"
        canvas.paste(label(load(os.path.join(d, fname), tile), text), ((i % 3) * tile[0], (i // 3) * tile[1]))
    canvas.save(out)


def delta_svg(deltas, out, width=1664, height=420):
    n = len(deltas)
    lo, hi = min(min(deltas), -0.2), max(deltas)
    zero = height * hi / (hi - lo)
    bw = width / n
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" aria-label="Ganho de PSNR por vista, E1 menos B1">',
             f'<line x1="0" y1="{zero:.1f}" x2="{width}" y2="{zero:.1f}" stroke="#8A93A6" stroke-width="2"/>']
    for i, d in enumerate(deltas):
        h = abs(d) / (hi - lo) * height
        y = zero - h if d >= 0 else zero
        color = "#2E86AB" if d >= 0 else "#E07A1F"
        parts.append(f'<rect x="{i * bw + 4:.1f}" y="{y:.1f}" width="{bw - 8:.1f}" height="{max(h, 2):.1f}" fill="{color}"/>')
    parts.append("</svg>")
    with open(out, "w") as f:
        f.write("".join(parts))
    return hi, lo, zero


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--per_view", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("-m", "--method", action="append", required=True, help="NOME=PASTA, mesma ordem do compare_preds")
    parser.add_argument("--views", nargs="+", required=True, help="indices das vistas para as montagens")
    parser.add_argument("--delta", nargs=2, metavar=("A", "B"), required=True, help="metodos para o grafico B - A")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.per_view) as f:
        per_view = json.load(f)
    dirs = {"GT": args.gt}
    dirs.update(dict(m.split("=", 1) for m in args.method))
    os.makedirs(args.out, exist_ok=True)
    for v in args.views:
        montage(v, dirs, per_view, os.path.join(args.out, f"montagem_{v}.png"))

    a, b = args.delta
    keys = sorted(per_view[a])
    deltas = [per_view[b][k]["PSNR"] - per_view[a][k]["PSNR"] for k in keys]
    hi, lo, zero = delta_svg(sorted(deltas), os.path.join(args.out, "delta_psnr.svg"))
    print(f"delta min {lo:.3f} max {hi:.3f}; eixo zero em {zero:.1f}/420 px")


if __name__ == "__main__":
    main()
