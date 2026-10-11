"""Subject-aware grading from the command line (no model checkpoint needed).

Finds the person mask, grades subject and background differently with one
preset look (or any .cube), and writes:
  <out>/<name>_mask.png        soft person mask (white = person)
  <out>/<name>_<look>.jpg      plain look, for comparison
  <out>/<name>_<look>_split.jpg subject-aware result
  <out>/<name>_<look>_subject.cube / _background.cube   (--cube)

--mask uses a mask from elsewhere instead, e.g. one the iOS app saved, to
compare the Python and Swift compositing on the same mask.

Usage:
  python scripts/subject_grade.py photo.jpg --look portra
  python scripts/subject_grade.py photo.jpg --look my_grade.cube --background-pop 0.8 --cube
"""
import argparse
from pathlib import Path

import torch
from torchvision.io import write_jpeg, write_png

from grademl.compose import SubjectAware, compose_luts, split_luts, subject_aware_grade
from grademl.data import load_rgb
from grademl.lut import read_cube, write_cube
from grademl.mask import MODELS, coverage, load_mask, mask_to_image, person_mask
from grademl.pro_looks import DIRECTIONS, random_pro_look

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("photo", type=Path)
    ap.add_argument("--look", default="portra", help=f"a direction ({', '.join(DIRECTIONS)}) or a .cube file")
    ap.add_argument("--seed", type=int, default=0, help="preset variation")
    ap.add_argument("--intensity", type=float, default=1.0)
    ap.add_argument("--subject-strength", type=float, default=SubjectAware.subject_strength)
    ap.add_argument("--background-strength", type=float, default=SubjectAware.background_strength)
    ap.add_argument("--background-pop", type=float, default=SubjectAware.background_pop)
    ap.add_argument("--seg-model", choices=MODELS, default=MODELS[0])
    ap.add_argument("--mask", type=Path, help="use this mask image instead of segmenting")
    ap.add_argument("--cube", action="store_true", help="also write subject / background .cube files")
    ap.add_argument("--out", type=Path, default=REPO / "exports" / "subject")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = load_rgb(args.photo).to(device)
    if args.look.endswith(".cube"):
        look, look_name = read_cube(args.look).to(device), Path(args.look).stem
    elif args.look in DIRECTIONS:
        look = random_pro_look(torch.Generator().manual_seed(args.seed), direction=args.look,
                               strength=(1.0, 1.0)).to(device)
        look_name = args.look
    else:
        ap.error(f"--look must be a .cube file or one of {DIRECTIONS}")

    mask = (load_mask(args.mask, img.shape[-2:]).to(device) if args.mask
            else person_mask(img, args.seg_model))
    cfg = SubjectAware(args.subject_strength, args.background_strength, args.background_pop)
    stem = args.out / args.photo.stem
    write_png(mask_to_image(mask), f"{stem}_mask.png")

    def save(x: torch.Tensor, suffix: str) -> None:
        write_jpeg((x.clamp(0, 1) * 255).round().to(torch.uint8).cpu(), f"{stem}_{look_name}{suffix}.jpg", quality=92)

    save(subject_aware_grade(img, look, None, cfg, args.intensity), "")
    save(subject_aware_grade(img, look, mask, cfg, args.intensity), "_split")
    if args.cube:
        subj, bg, pop = split_luts(look, cfg, args.intensity)
        write_cube(subj, f"{stem}_{look_name}_subject.cube", title=f"{look_name} subject")
        write_cube(compose_luts(bg, pop), f"{stem}_{look_name}_background.cube", title=f"{look_name} background")
    c = coverage(mask)
    note = "" if c >= cfg.min_coverage else " (no subject found: the split result is the plain look)"
    print(f"subject covers {c:.1%} of the frame{note} -> {args.out}")


if __name__ == "__main__":
    main()
