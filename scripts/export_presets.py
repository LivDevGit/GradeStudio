"""Export the pro-look presets as a LUT bank for the iOS app.

For every direction, N seeded variations at strength 1.0:
  presets/<direction>_<k>.lut16   float16, little-endian, RGB triplets in
                                  .cube order (red fastest), size^3 * 3 values;
                                  CIColorCube-ready after adding alpha
  presets/<direction>_<k>.cube    optional, for desktop editors (--cube)
  presets/background_pop.lut16    the subject-aware background LUT (grademl.compose)
  presets/manifest.json           directions, families, files, subject-aware
                                  defaults (read by GradeML.swift)

Usage: python scripts/export_presets.py [--variations 6] [--size 33] [--cube]
"""
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from grademl.compose import SubjectAware, background_pop_lut
from grademl.lut import write_cube
from grademl.pro_looks import DIRECTIONS, random_pro_look
from grademl.surprise import FAMILY

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variations", type=int, default=6)
    ap.add_argument("--size", type=int, default=33)
    ap.add_argument("--cube", action="store_true", help="also write .cube files")
    ap.add_argument("--out", type=Path, default=REPO / "exports" / "presets")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = {"lut_size": args.size, "format": "float16 RGB, red fastest", "directions": {}}
    total = 0

    def save(lut: torch.Tensor, name: str) -> str:
        nonlocal total
        rgb = lut.permute(1, 2, 3, 0).reshape(-1, 3).numpy().astype("<f2")  # (b, g, r, c): red fastest
        (args.out / f"{name}.lut16").write_bytes(rgb.tobytes())
        total += rgb.nbytes
        if args.cube:
            write_cube(lut, args.out / f"{name}.cube", title=f"grademl {name}")
        return f"{name}.lut16"

    for d in DIRECTIONS:
        files = []
        for k in range(args.variations):
            seed = 1000 * DIRECTIONS.index(d) + k  # stable across exports
            lut = random_pro_look(torch.Generator().manual_seed(seed), size=args.size, direction=d,
                                  strength=(1.0, 1.0))
            files.append(save(lut, f"{d}_{k}"))
        manifest["directions"][d] = {"family": FAMILY[d], "files": files}
    manifest["subject_aware"] = {"background_pop_file": save(background_pop_lut(args.size), "background_pop"),
                                 **dataclasses.asdict(SubjectAware())}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(DIRECTIONS)} directions x {args.variations} -> {args.out} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
