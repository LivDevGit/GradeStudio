"""Download the Unsplash Lite dataset (25k photos) at reduced resolution.

1. Fetches the metadata zip (photos.tsv + terms) from unsplash.com/data.
2. Downloads every photo from the Unsplash CDN at --width px (long edge
   capped via imgix `fit=max`), as JPEG.

Output: $GRADEML_DATA/unsplash/{images/<photo_id>.jpg, photos.tsv, TERMS*}
Re-runs skip photos that are already on disk.

Usage: python scripts/download_unsplash.py [--width 768] [--workers 24] [--limit N]
"""
import argparse
import csv
import io
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from grademl.data import data_root

META_URL = "https://unsplash.com/data/lite/latest"
UA = {"User-Agent": "grademl-research/0.1"}


def fetch(url: str, timeout: float = 60) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def get_metadata(out: Path) -> Path:
    tsv = out / "photos.tsv"
    if tsv.exists():
        return tsv
    print("downloading metadata zip (~320 MB) ...")
    zf = zipfile.ZipFile(io.BytesIO(fetch(META_URL, timeout=600)))
    for name in zf.namelist():
        base = Path(name).name
        if base.startswith("photos.tsv"):
            tsv.write_bytes(zf.read(name))
        elif base.upper().startswith(("TERMS", "LICENSE", "README")):
            (out / base).write_bytes(zf.read(name))
    return tsv


def download_one(row: dict, folder: Path, width: int) -> str:
    dest = folder / f"{row['photo_id']}.jpg"
    if dest.exists():
        return "skip"
    url = f"{row['photo_image_url']}?w={width}&h={width}&fit=max&fm=jpg&q=90"
    for attempt in range(4):
        try:
            data = fetch(url)
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.rename(dest)
            return "ok"
        except Exception:
            time.sleep(2**attempt)
    return "fail"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=768, help="max long edge in px")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, help="only the first N photos (testing)")
    args = ap.parse_args()

    out = data_root() / "unsplash"
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    tsv = get_metadata(out)

    with open(tsv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    rows = rows[: args.limit] if args.limit else rows
    print(f"{len(rows)} photos -> {images}")

    stats = {"ok": 0, "skip": 0, "fail": 0}
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(download_one, r, images, args.width) for r in rows]
        for fut in tqdm(as_completed(futures), total=len(futures), mininterval=10):
            stats[fut.result()] += 1
    print(stats)


if __name__ == "__main__":
    main()
