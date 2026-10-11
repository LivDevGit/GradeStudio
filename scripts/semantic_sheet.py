"""Visual report: content-aware matching (grademl.semantic_match) vs the
global and palette transfers.

Pairs are realistic: for each photo the reference is another held-out
Unsplash photo of similar content (closest DINOv2 image embedding among a
pool), i.e. "make my beach photo look like this beach photo". The
reference is pushed through a random pro look first, so there is a grade
to pick up, as with a film still or a photographer's edit.

  exports/semantic_match.jpg   photo | reference | global | palette | semantic

Usage: python scripts/semantic_sheet.py [--n 8] [--seed 0]
"""
import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from grademl.data import data_root, load_rgb
from grademl.lut import apply_lut
from grademl.pro_looks import random_pro_look
from grademl.semantic_match import patch_features, semantic_match
from grademl.synthetic import list_images
from grademl.transfer import palette_transfer_lut, transfer_lut

REPO = Path(__file__).resolve().parents[1]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--pool", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda")
    rng = random.Random(args.seed)

    photos = list_images(data_root() / "unsplash/images_384")[::50]   # held-out split
    pool = rng.sample(photos, args.pool)
    imgs = [load_rgb(p).to(dev) for p in pool]
    emb = F.normalize(torch.stack([patch_features(x, 224).mean((0, 1)) for x in imgs]), dim=-1)
    sim = emb @ emb.T
    sim.fill_diagonal_(-1)

    fit = lambda x: F.interpolate(x[None], size=(192, 256), mode="bilinear", antialias=True)[0]  # noqa: E731
    rows, used = [], set()
    g = torch.Generator().manual_seed(args.seed)
    for i in rng.sample(range(len(pool)), len(pool)):
        j = int(sim[i].argmax())
        if i in used or j in used or sim[i, j] < 0.55:
            continue
        used |= {i, j}
        photo = imgs[i]
        ref = apply_lut(imgs[j][None], random_pro_look(g).to(dev))[0]
        glob = apply_lut(photo[None], transfer_lut(photo[None], ref[None]))[0]
        pal = apply_lut(photo[None], palette_transfer_lut(photo[None], ref[None]))[0]
        sem = semantic_match(photo, ref).render(photo)
        rows += [fit(photo), fit(ref), fit(glob), fit(pal), fit(sem)]
        print(f"{pool[i].stem} <- {pool[j].stem}  similarity {float(sim[i, j]):.2f}")
        if len(rows) == 5 * args.n:
            break
    out = REPO / "exports/semantic_match.jpg"
    out.parent.mkdir(exist_ok=True)
    save_image(rows, out, nrow=5, padding=4)
    print(f"-> {out}  (photo | reference | global | palette | semantic)")


if __name__ == "__main__":
    main()
