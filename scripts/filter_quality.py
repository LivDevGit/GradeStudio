"""Keep only photos that already look finished, judged by GradeNet-L itself.

Training targets are built from the clean photo (grademl.local_looks), so
the clean photo is what the model learns "finished" means. Unsplash photos
are edited by their photographers; Open Images photos are Flickr snapshots,
some badly exposed or colored. The critic (GradeNet-L with style "none":
how much it would still change a photo, in % of mean color) scores every
photo; a photo is kept when it scores no worse than the given percentile of
Unsplash photos. Black-and-white photos are dropped too (no skin color to
learn from).

Writes <out>: {"threshold", "scores", "keep"}. Usage:
  python scripts/filter_quality.py openimages/images_384 openimages/quality.json [--percentile 75]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from grademl.color import rgb_to_lab
from grademl.data import data_root, load_rgb, resize
from grademl.editor import local_critic
from grademl.grade import STYLES
from grademl.local import LocalGradeNet
from grademl.synthetic import list_images

REPO = Path(__file__).resolve().parents[1]


class Small(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return resize(load_rgb(self.paths[i]))


@torch.no_grad()
def scores(net, paths, dev) -> tuple[np.ndarray, np.ndarray]:
    """(critic %, mean chroma) per photo."""
    crit, chroma = [], []
    for x in tqdm(DataLoader(Small(paths), batch_size=64, num_workers=8), unit="batch"):
        x = x.to(dev)
        crit.append(local_critic(net, x).cpu())
        chroma.append(rgb_to_lab(x.permute(0, 2, 3, 1))[..., 1:].norm(dim=-1).mean((1, 2)).cpu())
    return torch.cat(crit).numpy(), torch.cat(chroma).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--ckpt", default=str(REPO / "runs/grade_local_v3/best.pt"))
    ap.add_argument("--reference", default="unsplash/images_384")
    ap.add_argument("--percentile", type=float, default=75)
    ap.add_argument("--min-chroma", type=float, default=4.0, help="below: treated as black and white")
    args = ap.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu")
    net = LocalGradeNet(cond_dim=len(STYLES), **ck["cfg"]["model"])
    net.load_state_dict(ck["model"])
    net = net.to(dev).eval()

    ref = list_images(data_root() / args.reference)[::10]
    ref_crit, _ = scores(net, ref, dev)
    threshold = float(np.percentile(ref_crit, args.percentile))
    paths = list_images(data_root() / args.src)
    crit, chroma = scores(net, paths, dev)
    keep = [p.name for p, c, k in zip(paths, crit, chroma) if c <= threshold and k >= args.min_chroma]
    out = data_root() / args.out
    out.write_text(json.dumps({"threshold": threshold, "percentile": args.percentile,
                               "scores": {p.name: round(float(c), 3) for p, c in zip(paths, crit)},
                               "keep": keep}))
    print(f"critic: reference median {np.median(ref_crit):.2f}, p{args.percentile:.0f} {threshold:.2f}; "
          f"photos median {np.median(crit):.2f}")
    print(f"black and white: {(chroma < args.min_chroma).sum()} of {len(paths)}")
    print(f"kept {len(keep)} of {len(paths)} -> {out}")


if __name__ == "__main__":
    main()
