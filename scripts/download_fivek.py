"""Download MIT-Adobe FiveK (input + Expert C), preprocessed to 512px PNG.

Source: huggingface.co/datasets/Vicky0522/MIT-Adobe5k-for-RSFNet
  zeroed_as_shot.zip  input rendered from RAW with Lightroom "as shot", all
                      other adjustments zeroed (same convention as Zeng et al.)
  expertC.zip         Expert C retouch (the standard FiveK target)
  train.txt/test.txt  4500 / 498 split

FiveK is licensed for research use only. Use it to validate the technique,
never as training data for a model that ships in a product.

Usage: python scripts/download_fivek.py [--root DIR]   (default: $GRADEML_DATA)
"""
import argparse
import os
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "Vicky0522/MIT-Adobe5k-for-RSFNet"
PARTS = {"zeroed_as_shot.zip": "input", "expertC.zip": "expertC"}
SPLITS = ["train.txt", "test.txt"]
N_IMAGES = 5000


def default_root() -> Path:
    env = os.environ.get("GRADEML_DATA")
    return Path(env) if env else Path(__file__).resolve().parents[1] / "data"


def extract_flat(zip_path: Path, dest: Path) -> None:
    """Extract image files into dest, dropping any folder prefix inside the zip."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            with zf.open(info) as src, open(dest / Path(info.filename).name, "wb") as dst:
                shutil.copyfileobj(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=default_root())
    ap.add_argument("--keep-zips", action="store_true")
    args = ap.parse_args()

    out = args.root / "fivek"
    cache = args.root / "_downloads"
    out.mkdir(parents=True, exist_ok=True)

    for name in SPLITS:
        p = hf_hub_download(REPO, name, repo_type="dataset", local_dir=cache)
        shutil.copy(p, out / name)

    for zname, sub in PARTS.items():
        dest = out / sub
        if dest.is_dir() and len(list(dest.glob("*.png"))) == N_IMAGES:
            print(f"{sub}: already extracted, skipping")
            continue
        print(f"downloading {zname} ...")
        zpath = Path(hf_hub_download(REPO, zname, repo_type="dataset", local_dir=cache))
        print(f"extracting -> {dest}")
        extract_flat(zpath, dest)
        if not args.keep_zips:
            zpath.unlink()

    n_in = len(list((out / "input").glob("*.png")))
    n_gt = len(list((out / "expertC").glob("*.png")))
    print(f"done: {n_in} inputs, {n_gt} targets in {out}")
    assert n_in == n_gt == N_IMAGES, "image count mismatch"


if __name__ == "__main__":
    main()
