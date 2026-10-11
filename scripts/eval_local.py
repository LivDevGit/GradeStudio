"""Compare GradeNet (global LUT) and GradeNet-L checkpoints on the local targets.

Held-out Unsplash photos (the every-50th split used everywhere), faulty
inputs (global + local faults), target = look + local finishing
(grademl.local_looks, current version: includes skin finishing and subject
casts, so older checkpoints are measured on the new targets too). Reports:

  PSNR            whole image
  sky / person / skin PSNR   inside those regions (where local work happens)
  skin hue error  mean |hue difference| on skin vs the target, in degrees
                  (what makes a face look green, gray or magenta)

for: preset (fixed recipe on the faulty photo), v1 (one LUT per photo) and
each --local checkpoint. --people evaluates only photos with a person
(> 3% of the frame), with more samples per photo.

Sheets:
  exports/local_eval[_people].jpg   input | v1 | each local | target
  exports/local_real.jpg            unedited camera photos (FiveK inputs, research
                                    use for evaluation only): input | v1 | each local

Usage:
  python scripts/eval_local.py --local runs/grade_local_v1/best.pt runs/grade_local_v2/best.pt [--people]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from grademl.color import rgb_to_lab
from grademl.data import data_root, load_rgb
from grademl.editor import local_top_versions
from grademl.grade import STYLES, recipe_luts, style_onehot, top_versions
from grademl.local import LocalGradeNet, apply_local
from grademl.local_looks import FinishBank, LocalGradePairs, make_local_batch, person_coverage, region_masks
from grademl.lut import apply_lut
from grademl.metrics import psnr
from grademl.model import GradeNet
from grademl.synthetic import list_images

REPO = Path(__file__).resolve().parents[1]


def load(path: str, device):
    ck = torch.load(path, map_location="cpu")
    if ck.get("arch") == "grade_local":
        net = LocalGradeNet(cond_dim=len(STYLES), **ck["cfg"]["model"])
    else:
        net = GradeNet(len(STYLES), **ck["cfg"]["model"])
    net.load_state_dict(ck["model"])
    return net.to(device).eval(), ck


def masked_psnr(a, b, m):
    """PSNR inside a soft mask (B, 1, H, W); NaN where the mask covers < 2%."""
    se = ((a.clamp(0, 1) - b) ** 2).mean(1, keepdim=True)
    w = m.sum((1, 2, 3))
    mse = (se * m).sum((1, 2, 3)) / w.clamp_min(1e-6)
    out = 10 * torch.log10(1 / mse.clamp_min(1e-10))
    return torch.where(w > 0.02 * m[0].numel(), out, torch.full_like(out, float("nan")))


def hue_error(a, b, m):
    """Mean |hue difference| in degrees inside mask m, weighted by chroma
    (hue is meaningless on gray); NaN where the mask covers < 1%."""
    la = rgb_to_lab(a.clamp(0, 1).permute(0, 2, 3, 1))
    lb = rgb_to_lab(b.permute(0, 2, 3, 1))
    d = torch.rad2deg(torch.atan2(la[..., 2], la[..., 1]) - torch.atan2(lb[..., 2], lb[..., 1]))
    d = ((d + 180) % 360 - 180).abs()
    w = m[:, 0] * (lb[..., 1:].norm(dim=-1) / 20).clamp(max=1)
    out = (d * w).sum((1, 2)) / w.sum((1, 2)).clamp_min(1e-6)
    return torch.where(m.sum((1, 2, 3)) > 0.01 * m[0].numel(), out, torch.full_like(out, float("nan")))


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default=str(REPO / "runs/grade_v1/best.pt"))
    ap.add_argument("--local", nargs="+", default=[str(REPO / "runs/grade_local_v1/best.pt")])
    ap.add_argument("--samples", type=int, default=1500)
    ap.add_argument("--people", action="store_true", help="only photos with a person")
    ap.add_argument("--real", type=int, default=8, help="FiveK inputs for the real-photo sheet (0: skip)")
    args = ap.parse_args()
    dev = torch.device("cuda")

    v1, _ = load(args.v1, dev)
    locs = [load(p, dev) for p in args.local]
    names = ["v1"] + [Path(p).parent.name for p in args.local]
    dc = locs[0][1]["cfg"]["data"]
    maps_dir = data_root() / dc["maps"]
    photos = list_images(data_root() / dc["images"])[:: dc["val_every_nth"]]
    if args.people:
        cov = person_coverage(maps_dir, photos)
        photos = [p for p, c in zip(photos, cov) if c > 0.03]
        print(f"{len(photos)} held-out photos with a person")
    val = LocalGradePairs(photos, maps_dir, train=False, length=args.samples, p_clean=dc["p_clean"],
                          p_none=dc["p_none"], p_local=dc.get("p_local", 0.6))
    recipes = recipe_luts(33, dev)
    bank = FinishBank(STYLES, 33, dev)

    keys = ["psnr", "sky", "person", "skin", "skin_hue"]
    res = {n: {k: [] for k in keys} for n in names}
    preset, no_op, sheet = [], [], []
    for batch in DataLoader(val, batch_size=50, num_workers=6):
        photo, maps, fault, local, style = (t.to(dev) for t in batch)
        inp, tgt = make_local_batch(photo, maps, fault, local, style, recipes, bank)
        oh = style_onehot(style, dev)
        outs = [apply_lut(inp, v1(inp, oh))]
        for net, _ in locs:
            outs.append(apply_local(inp, *net(inp, oh)))
        styled = style > 0
        preset += psnr(apply_lut(inp[styled], recipes[style[styled]]), tgt[styled]).tolist()
        no_op += psnr(inp, tgt).tolist()
        m = region_masks(maps, photo)
        for n, out in zip(names, outs):
            r = res[n]
            r["psnr"] += psnr(out, tgt).tolist()
            r["sky"] += masked_psnr(out, tgt, m["sky"]).tolist()
            r["person"] += masked_psnr(out, tgt, m["person"]).tolist()
            r["skin"] += masked_psnr(out, tgt, m["skin"]).tolist()
            r["skin_hue"] += hue_error(out, tgt, m["skin"]).tolist()
        for i in range(len(inp)):
            interesting = m["person"][i].mean() > 0.08 if args.people else (
                m["sky"][i].mean() > 0.15 or m["person"][i].mean() > 0.08)
            if len(sheet) < (len(names) + 2) * 10 and interesting:
                sheet += [inp[i]] + [o[i].clamp(0, 1) for o in outs] + [tgt[i]]

    print(f"\nno-op {np.mean(no_op):.2f}   fixed preset {np.mean(preset):.2f}   (PSNR, dB)")
    print(f"{'model':18s}" + "".join(f"{k:>10s}" for k in keys) + "   (skin_hue: degrees, lower is better)")
    for n in names:
        row = [np.nanmean(np.array(res[n][k], dtype=float)) for k in keys]
        print(f"{n:18s}" + "".join(f"{v:10.2f}" for v in row))
    counts = {k: int(np.isfinite(np.array(res[names[0]][k], dtype=float)).sum()) for k in keys}
    print("samples per column: " + ", ".join(f"{k} {v}" for k, v in counts.items()))

    out_dir = REPO / "exports"
    out_dir.mkdir(exist_ok=True)
    sheet_path = out_dir / ("local_eval_people.jpg" if args.people else "local_eval.jpg")
    save_image(sheet, sheet_path, nrow=len(names) + 2, padding=4)
    print(f"-> {sheet_path}  (input | {' | '.join(names)} | target)")

    if args.real:
        fivek = data_root() / "fivek"
        files = (fivek / "test.txt").read_text().split()[:: 498 // args.real][: args.real]
        real = []
        fit = lambda x: F.interpolate(x[None], size=(256, 384), mode="bilinear", antialias=True)[0]  # noqa: E731
        for f in files:
            img = load_rgb(fivek / "input" / f).to(dev)
            e1 = top_versions(img, v1, n=1)[0]
            row = [fit(img), fit(apply_lut(img[None], e1.lut)[0])]
            picks = [f"v1 {e1.best.style} {e1.best.intensity:.0%}"]
            for (net, _), n in zip(locs, names[1:]):
                el = local_top_versions(img, net, n=1)[0]
                row.append(fit(el.render(img)))
                picks.append(f"{n} {el.style} {el.intensity:.0%}")
            real += row
            print(f"{f}: " + " | ".join(picks))
        save_image(real, out_dir / "local_real.jpg", nrow=len(names) + 1, padding=4)
        print(f"-> {out_dir / 'local_real.jpg'}  (camera input | {' | '.join(names)})")


if __name__ == "__main__":
    main()
