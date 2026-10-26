"""How the AI editor (grademl.editor.local_top_versions) chooses, over many photos.

Runs the editor on held-out Unsplash photos (as finished by their
photographers, no faults) and reports:
  - how often each look is the #1 pick and in the top 3
  - the mean of each score part per look (critic, fit, bold, clip, skin,
    and any newer parts), which shows *why* a look wins or loses
With --calibrate it also writes the per-look statistics the editor uses to
judge every look on a level field (grademl.editor.Calibration), and with
--sheet a contact sheet of the top 3 on some photos.

Usage:
  python scripts/editor_report.py [--n 300] [--calibrate] [--sheet 8]
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from grademl.data import data_root, load_rgb
from grademl.editor import CALIBRATION, EditorWeightsL, local_top_versions, score_candidates
from grademl.grade import STYLES
from grademl.local import LocalGradeNet
from grademl.pro_looks import DIRECTIONS
from grademl.synthetic import list_images

REPO = Path(__file__).resolve().parents[1]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO / "runs/grade_local_v5/best.pt"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--calibrate", action="store_true", help=f"write {CALIBRATION.name}")
    ap.add_argument("--raw", action="store_true", help="score without the calibration file")
    ap.add_argument("--keep", type=float, default=EditorWeightsL.keep, help="color-keeping weight (0: off)")
    ap.add_argument("--sheet", type=int, default=0, help="photos in exports/editor_top3.jpg")
    ap.add_argument("--offset", type=int, default=0, help="photos to skip (calibrate and report on different ones)")
    args = ap.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu")
    net = LocalGradeNet(cond_dim=len(STYLES), **ck["cfg"]["model"])
    net.load_state_dict(ck["model"])
    net = net.to(dev).eval()

    photos = list_images(data_root() / "unsplash/images_384")[::50][args.offset:args.offset + args.n]
    first, top3 = Counter(), Counter()
    parts = defaultdict(lambda: defaultdict(list))   # style -> part -> values (all intensities)
    sheet = []
    for k, p in enumerate(photos):
        img = load_rgb(p).to(dev)
        if args.calibrate:
            for c in score_candidates(img, net, w=EditorWeightsL(keep=args.keep), calibration=None)[0]:
                for name, v in c.parts.items():
                    parts[c.style][name].append(v)
            continue
        edits = local_top_versions(img, net, n=3, w=EditorWeightsL(keep=args.keep),
                                   calibration=None if args.raw else "default")
        first[edits[0].style] += 1
        for e in edits:
            top3[e.style] += 1
        for c in edits[0].shortlist:
            for name, v in c.parts.items():
                parts[c.style][name].append(v)
        if k < args.sheet:
            fit = lambda x: F.interpolate(x[None], size=(192, 256), mode="bilinear", antialias=True)[0]  # noqa: E731
            sheet += [fit(img)] + [fit(e.render(img)) for e in edits]

    if args.calibrate:
        stats = {s: {name: [float(np.mean(v)), float(np.std(v) + 1e-6)] for name, v in d.items()}
                 for s, d in parts.items()}
        CALIBRATION.write_text(json.dumps({"ckpt": Path(args.ckpt).parent.name, "photos": len(photos),
                                           "stats": stats}, indent=1))
        print(f"-> {CALIBRATION} ({len(photos)} photos)")
        return

    names = sorted({n for d in parts.values() for n in d})
    print(f"{'look':16s} {'#1':>5s} {'top3':>5s}  " + " ".join(f"{n:>7s}" for n in names))
    for s in ["none", *DIRECTIONS]:
        row = " ".join(f"{np.mean(parts[s][n]):7.2f}" if parts[s][n] else "      -" for n in names)
        print(f"{s:16s} {first[s] / len(photos):5.0%} {top3[s] / len(photos):5.0%}  {row}")
    share_mono = (first["bw_toned"] + first["bleach_bypass"]) / len(photos)
    print(f"\nB&W / bleach bypass as #1: {share_mono:.0%}   distinct #1 looks: {len(first)}")
    if sheet:
        out = REPO / "exports/editor_top3.jpg"
        save_image(sheet, out, nrow=4, padding=4)
        print(f"-> {out}  (photo | #1 | #2 | #3)")


if __name__ == "__main__":
    main()
