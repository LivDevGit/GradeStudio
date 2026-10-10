"""Contact sheet of synthetic looks, to eyeball whether they read as
plausible grades. Rows: base photos. Columns: original, then one sampled
look per creative direction (the same look on every row of a column).

Usage: python scripts/preview_looks.py [--images DIR] [--seed 0] [--generic 2]
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.utils import make_grid

from grademl.data import data_root, load_rgb
from grademl.lut import apply_lut
from grademl.synthetic import DIRECTIONS, list_images, random_look

REPO = Path(__file__).resolve().parents[1]
TILE = (200, 300)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=data_root() / "unsplash" / "images")
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--generic", type=int, default=2, help="extra columns of generic random looks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "exports" / "looks_preview.jpg")
    args = ap.parse_args()

    paths = list_images(args.images)
    step = max(1, len(paths) // args.rows)
    imgs = torch.stack([F.interpolate(load_rgb(p)[None], size=TILE, mode="bilinear", antialias=True)[0]
                        for p in paths[::step][:args.rows]])

    g = torch.Generator().manual_seed(args.seed)
    names = ["original"] + list(DIRECTIONS) + ["generic"] * args.generic
    cols = [imgs] + [apply_lut(imgs, random_look(g, direction=n)) for n in names[1:]]
    grid = make_grid(torch.stack(cols, 1).flatten(0, 1), nrow=len(cols), padding=3, pad_value=1.0)

    # Header row with direction names.
    img = Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).round().astype("uint8"))
    canvas = Image.new("RGB", (img.width, img.height + 24), "white")
    canvas.paste(img, (0, 24))
    draw = ImageDraw.Draw(canvas)
    for i, n in enumerate(names):
        draw.text((3 + i * (TILE[1] + 3) + 6, 6), n, fill="black")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, quality=92)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
