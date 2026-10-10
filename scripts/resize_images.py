"""Make a downscaled copy of an image folder (long edge <= --size, JPEG).

Training only ever sees 256px crops, so decoding the 768px originals every
step wastes CPU; the DataLoader is CPU-bound otherwise.

Usage: python scripts/resize_images.py unsplash/images unsplash/images_384 --size 384
(paths relative to $GRADEML_DATA unless absolute)
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from grademl.data import data_root
from grademl.synthetic import list_images


def resize_one(args: tuple[Path, Path, int]) -> None:
    src, dst, size = args
    if dst.exists():
        return
    img = Image.open(src).convert("RGB")
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    tmp = dst.with_suffix(".part")
    img.save(tmp, "JPEG", quality=95)
    tmp.rename(dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    src = args.src if args.src.is_absolute() else data_root() / args.src
    dst = args.dst if args.dst.is_absolute() else data_root() / args.dst
    dst.mkdir(parents=True, exist_ok=True)
    jobs = [(p, dst / (p.stem + ".jpg"), args.size) for p in list_images(src)]
    with ProcessPoolExecutor(args.workers) as pool:
        list(tqdm(pool.map(resize_one, jobs, chunksize=64), total=len(jobs), mininterval=10))
    print(f"{len(jobs)} images -> {dst}")


if __name__ == "__main__":
    main()
