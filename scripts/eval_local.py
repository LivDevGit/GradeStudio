"""Compare GradeNet (global LUT) and GradeNet-L (local) on the local targets.

Held-out Unsplash photos (the same every-50th split as training), faulty
inputs (global + local faults), target = look + local finishing
(grademl.local_looks). Reports PSNR overall and inside the sky / person
regions, where local work happens, plus:

  preset   the fixed recipe on the faulty photo
  v1       GradeNet v1: one LUT per photo
  local    GradeNet-L: K LUTs + bilateral grid

and writes two sheets:
  exports/local_eval.jpg   input | v1 | GradeNet-L | target   (held-out Unsplash)
  exports/local_real.jpg   unedited camera photos (FiveK inputs, research use
                           for evaluation only): input | v1 best | GradeNet-L best

Usage:
  python scripts/eval_local.py --v1 runs/grade_v1/best.pt --local runs/grade_local_v1/best.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from grademl.data import data_root, load_rgb
from grademl.editor import local_top_versions
from grademl.grade import STYLES, recipe_luts, style_onehot, top_versions
from grademl.local import LocalGradeNet, apply_local
from grademl.local_looks import FinishBank, LocalGradePairs, make_local_batch, region_masks
from grademl.lut import apply_lut
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
    """PSNR inside a soft mask (B, 1, H, W); NaN where the mask is empty."""
    se = ((a.clamp(0, 1) - b) ** 2).mean(1, keepdim=True)
    w = m.sum((1, 2, 3))
    mse = (se * m).sum((1, 2, 3)) / w.clamp_min(1e-6)
    out = 10 * torch.log10(1 / mse.clamp_min(1e-10))
    return torch.where(w > 0.05 * m[0].numel(), out, torch.full_like(out, float("nan")))


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default=str(REPO / "runs/grade_v1/best.pt"))
    ap.add_argument("--local", default=str(REPO / "runs/grade_local_v1/best.pt"))
    ap.add_argument("--samples", type=int, default=1500)
    ap.add_argument("--real", type=int, default=8, help="FiveK inputs for the real-photo sheet (0: skip)")
    args = ap.parse_args()
    dev = torch.device("cuda")

    v1, _ = load(args.v1, dev)
    loc, ck = load(args.local, dev)
    dc = ck["cfg"]["data"]
    photos = list_images(data_root() / dc["images"])
    val = LocalGradePairs(photos[:: dc["val_every_nth"]], data_root() / dc["maps"], train=False,
                          length=args.samples, p_clean=dc["p_clean"], p_none=dc["p_none"],
                          p_local=dc.get("p_local", 0.6))
    recipes = recipe_luts(33, dev)
    bank = FinishBank(STYLES, 33, dev)

    rows = {k: [] for k in ["preset", "v1", "local", "no_op"]}
    region = {f"{k}_{r}": [] for k in ["v1", "local"] for r in ["sky", "person"]}
    sheet = []
    for batch in DataLoader(val, batch_size=50, num_workers=8):
        photo, maps, fault, local, style = (t.to(dev) for t in batch)
        inp, tgt = make_local_batch(photo, maps, fault, local, style, recipes, bank)
        oh = style_onehot(style, dev)
        out_v1 = apply_lut(inp, v1(inp, oh))
        luts, grid = loc(inp, oh)
        out_l = apply_local(inp, luts, grid)
        styled = style > 0
        from grademl.metrics import psnr
        rows["preset"] += psnr(apply_lut(inp[styled], recipes[style[styled]]), tgt[styled]).tolist()
        rows["v1"] += psnr(out_v1, tgt).tolist()
        rows["local"] += psnr(out_l, tgt).tolist()
        rows["no_op"] += psnr(inp, tgt).tolist()
        m = region_masks(maps, photo)
        for name, out in [("v1", out_v1), ("local", out_l)]:
            region[f"{name}_sky"] += masked_psnr(out, tgt, m["sky"]).tolist()
            region[f"{name}_person"] += masked_psnr(out, tgt, m["person"]).tolist()
        for i in range(len(inp)):
            if len(sheet) < 4 * 10 and (m["sky"][i].mean() > 0.15 or m["person"][i].mean() > 0.08):
                sheet += [inp[i], out_v1[i].clamp(0, 1), out_l[i].clamp(0, 1), tgt[i]]

    print(f"{'':10s} {'PSNR':>7s}")
    for k, v in rows.items():
        print(f"{k:10s} {np.mean(v):7.2f}")
    for k, v in region.items():
        v = np.array(v)
        print(f"{k:13s} {np.nanmean(v):7.2f}  (n={np.isfinite(v).sum()})")
    out_dir = REPO / "exports"
    out_dir.mkdir(exist_ok=True)
    save_image(sheet, out_dir / "local_eval.jpg", nrow=4, padding=4)
    print(f"-> {out_dir / 'local_eval.jpg'}  (input | v1 | GradeNet-L | target)")

    if args.real:
        fivek = data_root() / "fivek"
        names = (fivek / "test.txt").read_text().split()[:: 498 // args.real][: args.real]
        real = []
        for n in names:
            img = load_rgb(fivek / "input" / n).to(dev)
            e1 = top_versions(img, v1, n=1)[0]
            el = local_top_versions(img, loc, n=1)[0]
            fit = lambda x: F.interpolate(x[None], size=(256, 384), mode="bilinear", antialias=True)[0]  # noqa: E731
            real += [fit(img), fit(apply_lut(img[None], e1.lut)[0]), fit(el.render(img))]
            print(f"{n}: v1 {e1.best.style} {e1.best.intensity:.0%} | local {el.style} {el.intensity:.0%}")
        save_image(real, out_dir / "local_real.jpg", nrow=3, padding=4)
        print(f"-> {out_dir / 'local_real.jpg'}  (camera input | v1 pick | GradeNet-L pick)")


if __name__ == "__main__":
    main()
