"""Contact sheet of pro looks (grademl.pro_looks).

Columns: photos (first column = --photo if given). Rows: original, then one
sampled variation per direction (same look across a row).

Usage: python scripts/preview_pro_looks.py --photo data/samples/arch_original.webp [--seed 0]
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.utils import make_grid

from grademl.data import data_root, load_rgb
from grademl.lut import apply_lut
from grademl.pro_looks import DIRECTIONS, random_pro_look

REPO = Path(__file__).resolve().parents[1]
TILE = (200, 270)


def load_any(path: Path) -> torch.Tensor:
    """load_rgb handles png/jpg; PIL covers webp."""
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return load_rgb(path)
    img = Image.open(path).convert("RGB")
    return torch.from_numpy(__import__("numpy").asarray(img).copy()).permute(2, 0, 1).float() / 255


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", type=Path, action="append", default=[])
    ap.add_argument("--extra", type=int, default=3, help="Unsplash photos to add")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=REPO / "exports" / "pro_looks_preview.jpg")
    args = ap.parse_args()

    photos = [load_any(p) for p in args.photo]
    pool = sorted((data_root() / "unsplash" / "images_384").glob("*.jpg"))[::50]
    for i in range(args.extra):
        photos.append(load_rgb(pool[(i * 97 + 13) % len(pool)]))
    imgs = torch.stack([F.interpolate(p[None], size=TILE, mode="bilinear", antialias=True)[0] for p in photos])

    g = torch.Generator().manual_seed(args.seed)
    names = ["original", *DIRECTIONS]
    rows = [imgs] + [apply_lut(imgs, random_pro_look(g, direction=d, strength=(args.strength, args.strength)))
                     for d in DIRECTIONS]
    grid = make_grid(torch.cat(rows), nrow=len(photos), padding=3, pad_value=1.0)
    img = Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).round().astype("uint8"))
    canvas = Image.new("RGB", (img.width + 120, img.height), "white")
    canvas.paste(img, (120, 0))
    d = ImageDraw.Draw(canvas)
    for i, n in enumerate(names):
        d.text((6, 3 + i * (TILE[0] + 3) + TILE[0] // 2), n, fill="black")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, quality=92)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
