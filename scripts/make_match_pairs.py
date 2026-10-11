"""Training pairs for MatchNet-L, with the content-aware teacher precomputed.

1. DINOv2 embedding of every Unsplash photo (mean patch feature).
2. Per photo, `per_photo` pairs (photo, reference): mostly one of its
   nearest neighbours in content (a beach for a beach, a portrait for a
   portrait: how people pick references), sometimes a random photo. Train
   and validation photos (every 50th, as everywhere) are paired only within
   their own split.
3. The reference goes through a random pro look (p_look), so there is a
   grade to pick up, as with a film still; otherwise it is used as it is.
4. Teacher: grademl.semantic_match.semantic_match_batch on the 256px
   (squashed) pair; the matched photo is saved as JPEG.

Output in $GRADEML_DATA/<out>: pairs.json (photo, reference, look seed or
-1, split) and targets/<i>.jpg. Usage:
  python scripts/make_match_pairs.py unsplash/images_384 unsplash/match_teacher
"""
import argparse
import json
import random

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from grademl.data import data_root, load_rgb, resize
from grademl.lut import apply_lut
from grademl.pro_looks import random_pro_look
from grademl.semantic_match import patch_features_batch, semantic_match_batch
from grademl.synthetic import list_images


class Resized(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return resize(load_rgb(self.paths[i]))


def ref_look(seed: int, size: int = 33) -> torch.Tensor:
    return random_pro_look(torch.Generator().manual_seed(seed), size)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--per-photo", type=int, default=2)
    ap.add_argument("--p-neighbour", type=float, default=0.75)
    ap.add_argument("--p-look", type=float, default=0.8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda")
    rng = random.Random(args.seed)
    out = data_root() / args.out
    (out / "targets").mkdir(parents=True, exist_ok=True)

    paths = list_images(data_root() / args.src)
    val = set(paths[::50])
    pairs_file = out / "pairs.json"
    if pairs_file.exists():
        pairs = json.loads(pairs_file.read_text())
    else:
        embs = []
        for x in tqdm(DataLoader(Resized(paths), batch_size=64, num_workers=8), desc="embed"):
            embs.append(F.normalize(patch_features_batch(x.to(dev)).mean((1, 2)), dim=-1))
        emb = torch.cat(embs)
        pairs = []
        for split in ("train", "val"):
            idx = [i for i, p in enumerate(paths) if (p in val) == (split == "val")]
            e = emb[idx]
            for a in tqdm(range(0, len(idx), 1024), desc=f"neighbours {split}"):
                sim = e[a:a + 1024] @ e.T
                sim[torch.arange(len(sim)), torch.arange(a, a + len(sim))] = -1
                top = sim.topk(10, dim=1).indices.cpu()
                for row, i in enumerate(range(a, a + len(sim))):
                    for _ in range(args.per_photo):
                        j = (top[row, rng.randrange(10)].item() if rng.random() < args.p_neighbour
                             else rng.randrange(len(idx)))
                        seed = rng.getrandbits(31) if rng.random() < args.p_look else -1
                        pairs.append({"photo": paths[idx[i]].name, "reference": paths[idx[j]].name,
                                      "look": seed, "split": split})
        pairs_file.write_text(json.dumps(pairs))
    print(f"{len(pairs)} pairs")

    root = data_root() / args.src
    todo = [i for i in range(len(pairs)) if not (out / "targets" / f"{i}.jpg").exists()]

    class Pairs(Dataset):
        def __len__(self):
            return len(todo)

        def __getitem__(self, n):
            i = todo[n]
            p = pairs[i]
            ref = resize(load_rgb(root / p["reference"]))
            if p["look"] >= 0:
                ref = apply_lut(ref[None], ref_look(p["look"]))[0].clamp(0, 1)
            return i, resize(load_rgb(root / p["photo"])), ref

    for idx, photo, ref in tqdm(DataLoader(Pairs(), batch_size=args.batch, num_workers=8), desc="teacher"):
        tgt = semantic_match_batch(photo.to(dev), ref.to(dev))
        for i, t in zip(idx.tolist(), tgt):
            img = (t.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
            Image.fromarray(img).save(out / "targets" / f"{i}.jpg", quality=95)


if __name__ == "__main__":
    main()
