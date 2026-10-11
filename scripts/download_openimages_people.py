"""People photos from Open Images (validation + test sets, CC BY 2.0).

Unsplash has few people (~10% of photos), so GradeNet-L sees few faces and
little variety in skin tone. Open Images has human-verified person boxes
over a much wider range of people and places. Selection:

  - a box labeled Person / Man / Woman / Boy / Girl / Human face, not a
    depiction (drawing, statue, poster: IsDepiction = 0)
  - the largest such box covers >= --min-area of the photo
  - photo at least 640 px on the short side

Photos are downloaded from the public Open Images S3 mirror, resized to 384
px on the long side (as the Unsplash training set) and the originals are
not kept. attribution.csv records author, original URL and license of every
photo kept (CC BY 2.0 asks for attribution).

Usage:
  python scripts/download_openimages_people.py openimages/images_384 [--max 25000] [--workers 32]
  (paths relative to $GRADEML_DATA; resumable)
"""
import argparse
import csv
import io
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from grademl.data import data_root

META = "https://storage.googleapis.com/openimages"
FILES = {
    "validation": (f"{META}/v5/validation-annotations-bbox.csv",
                   f"{META}/2018_04/validation/validation-images-with-rotation.csv"),
    "test": (f"{META}/v5/test-annotations-bbox.csv", f"{META}/2018_04/test/test-images-with-rotation.csv"),
}
IMAGES = "https://open-images-dataset.s3.amazonaws.com/{split}/{id}.jpg"
PEOPLE = {"/m/01g317": "Person", "/m/04yx4": "Man", "/m/03bt1vf": "Woman", "/m/01bl7v": "Boy",
          "/m/05r655": "Girl", "/m/0dzct": "Human face"}


def fetch(url: str, dst: Path) -> Path:
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dst)
    return dst


def candidates(meta_dir: Path, min_area: float) -> list[dict]:
    rows = []
    for split, (boxes_url, images_url) in FILES.items():
        boxes = fetch(boxes_url, meta_dir / Path(boxes_url).name)
        images = fetch(images_url, meta_dir / Path(images_url).name)
        area = defaultdict(float)
        with open(boxes, newline="") as f:
            for r in csv.DictReader(f):
                if r["LabelName"] in PEOPLE and r.get("IsDepiction", "0") != "1":
                    a = (float(r["XMax"]) - float(r["XMin"])) * (float(r["YMax"]) - float(r["YMin"]))
                    area[r["ImageID"]] = max(area[r["ImageID"]], a)
        with open(images, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if area.get(r["ImageID"], 0) >= min_area:
                    rows.append({"split": split, "id": r["ImageID"], "person_area": round(area[r["ImageID"]], 3),
                                 "author": r.get("Author", ""), "url": r.get("OriginalURL", ""),
                                 "license": r.get("License", ""), "rotation": r.get("Rotation", "")})
        print(f"{split}: {sum(1 for v in area.values() if v >= min_area)} photos with a person >= {min_area:.0%}")
    return rows


def download(row: dict, out: Path, size: int, min_short: int) -> bool:
    dst = out / f"oi_{row['id']}.jpg"
    if dst.exists():
        return True
    try:
        with urllib.request.urlopen(IMAGES.format(split=row["split"], id=row["id"]), timeout=30) as r:
            im = Image.open(io.BytesIO(r.read()))
            im.load()
    except Exception:
        return False
    rot = row.get("rotation") or "0"
    try:
        if float(rot):  # Open Images stores the rotation needed to display the photo upright
            im = im.rotate(-float(rot), expand=True)
    except ValueError:
        pass
    im = im.convert("RGB")
    if min(im.size) < min_short:
        return False
    s = size / max(im.size)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    tmp = dst.with_suffix(".part")
    im.save(tmp, "JPEG", quality=95)
    tmp.rename(dst)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--min-area", type=float, default=0.10)
    ap.add_argument("--min-short", type=int, default=640)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--max", type=int, default=25000)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    out = data_root() / args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = candidates(out.parent / "_meta", args.min_area)
    rows = sorted(rows, key=lambda r: r["id"])[: args.max]
    print(f"{len(rows)} candidates -> {out}")
    kept = []
    with ThreadPoolExecutor(args.workers) as ex:
        for row, ok in zip(rows, tqdm(ex.map(lambda r: download(r, out, args.size, args.min_short), rows),
                                      total=len(rows), unit="img")):
            if ok:
                kept.append(row)
    with open(out.parent / "attribution.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "author", "url", "license", "person_area", "split"])
        w.writeheader()
        for r in kept:
            w.writerow({"file": f"oi_{r['id']}.jpg", "author": r["author"], "url": r["url"], "license": r["license"],
                        "person_area": r["person_area"], "split": r["split"]})
    licenses = defaultdict(int)
    for r in kept:
        licenses[r["license"]] += 1
    print(f"kept {len(kept)} photos; licenses: {dict(licenses)}")


if __name__ == "__main__":
    main()
