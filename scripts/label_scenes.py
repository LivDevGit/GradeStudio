"""Precompute scene maps for the local-grading training data.

For every photo: soft semantic maps (OneFormer Swin-T, ADE20K; MIT) for the
regions colorists treat separately, and relative depth (Depth Anything V2
Small; Apache-2.0). Saved at half the image size as uint8, one .npz per
photo (key "maps", shape (C, h, w), channel order CHANNELS).

The maps only build training targets (grademl.local_looks): what a
colorist would do to the sky, the subject, the foliage, the background.
The trained model never sees them and needs no segmentation on device; it
learns where to apply what from the photo itself.

Usage:
  python scripts/label_scenes.py unsplash/images_384 unsplash/scene_192
  (paths relative to $GRADEML_DATA; resumable: existing outputs are skipped)
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForDepthEstimation, OneFormerForUniversalSegmentation, OneFormerProcessor

from grademl.data import data_root
from grademl.synthetic import list_images

SEG_MODEL = "shi-labs/oneformer_ade20k_swin_tiny"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

# ADE20K class names grouped into the regions a colorist would mask.
GROUPS = {
    "sky": ["sky"],
    "person": ["person"],
    "plant": ["tree", "grass", "plant", "field", "flower", "palm"],
    "water": ["water", "sea", "river", "lake", "waterfall", "swimming pool"],
    "built": ["building", "skyscraper", "house", "wall", "road", "sidewalk", "bridge", "tower"],
}
CHANNELS = (*GROUPS, "depth")  # depth: 1 = near, 0 = far (per photo)


def group_index(id2label: dict[int, str]) -> list[list[int]]:
    """Class ids per group. An ADE20K label like 'palm, palm tree' matches 'palm'."""
    out = []
    for names in GROUPS.values():
        ids = [i for i, lab in id2label.items() if any(n in [s.strip() for s in lab.split(",")] for n in names)]
        out.append(ids)
    return out


@torch.no_grad()
def semantic_probs(model, batch) -> torch.Tensor:
    """(B, classes, h, w) soft class probabilities at 1/4 of the padded input."""
    out = model(**batch)
    cls = out.class_queries_logits.float().softmax(-1)[..., :-1]  # drop "no object"
    masks = out.masks_queries_logits.float().sigmoid()
    seg = torch.einsum("bqc,bqhw->bchw", cls, masks)
    return seg / seg.sum(1, keepdim=True).clamp_min(1e-6)


def depth_input(im: Image.Image) -> torch.Tensor:
    """Depth Anything input at the photo's aspect (long side 518, multiples of 14), normalized."""
    w, h = im.size
    s = 518 / max(w, h)
    tw, th = max(14, round(w * s / 14) * 14), max(14, round(h * s / 14) * 14)
    x = torch.from_numpy(np.array(im.resize((tw, th), Image.BICUBIC))).permute(2, 0, 1).float() / 255
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def depth_map(model, x: torch.Tensor) -> torch.Tensor:
    """Relative inverse depth of one photo, normalized to [0, 1] (1 = near)."""
    d = model(pixel_values=x[None].half()).predicted_depth.float()[0]
    lo, hi = torch.quantile(d.flatten()[::7], torch.tensor([0.01, 0.99], device=d.device))
    return ((d - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)


class Photos(torch.utils.data.Dataset):
    """Decoding and both models' preprocessing, in DataLoader workers (the
    CPU side was the bottleneck)."""

    def __init__(self, paths, proc):
        self.paths, self.proc = paths, proc

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        seg = self.proc(images=[im], task_inputs=["semantic"], return_tensors="pt")
        return i, im.size, seg["pixel_values"][0], seg["task_inputs"][0], depth_input(im)


def collate(items):
    """Pad the segmentation inputs to one batch (bottom / right), with the
    pixel mask OneFormer expects."""
    h = max(it[2].shape[1] for it in items)
    w = max(it[2].shape[2] for it in items)
    pv = torch.zeros(len(items), 3, h, w)
    pm = torch.zeros(len(items), h, w, dtype=torch.long)
    for j, it in enumerate(items):
        _, ih, iw = it[2].shape
        pv[j, :, :ih, :iw], pm[j, :ih, :iw] = it[2], 1
    return ([it[0] for it in items], [it[1] for it in items], pv, pm,
            torch.stack([it[3] for it in items]), [it[4] for it in items])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--scale", type=float, default=0.5, help="output size relative to the photo")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", default="0/1", help="i/n: label every n-th photo starting at i (parallel runs)")
    args = ap.parse_args()

    src, dst = data_root() / args.src, data_root() / args.dst
    dst.mkdir(parents=True, exist_ok=True)
    i, n = map(int, args.shard.split("/"))
    paths = [p for p in list_images(src)[i::n] if not (dst / f"{p.stem}.npz").exists()][: args.limit]
    print(f"{len(paths)} photos to label -> {dst}")
    dev = torch.device("cuda")

    proc = OneFormerProcessor.from_pretrained(SEG_MODEL)
    proc.image_processor.size = {"shortest_edge": 384, "longest_edge": 768}
    seg_model = OneFormerForUniversalSegmentation.from_pretrained(SEG_MODEL).to(dev).eval().half()
    depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to(dev).eval().half()
    groups = group_index(seg_model.config.id2label)
    for name, ids in zip(GROUPS, groups):
        print(f"  {name}: {[seg_model.config.id2label[i] for i in ids]}")

    loader = torch.utils.data.DataLoader(Photos(paths, proc), batch_size=args.batch, num_workers=args.workers,
                                         collate_fn=collate, prefetch_factor=4)
    bar = tqdm(total=len(paths), unit="img")
    for idx, sizes, pv, pixel_mask, task, dins in loader:
        inputs = {"pixel_values": pv.to(dev).half(), "pixel_mask": pixel_mask.to(dev), "task_inputs": task.to(dev)}
        probs = semantic_probs(seg_model, inputs)                       # (B, 150, h4, w4)
        grouped = torch.stack([probs[:, ids].sum(1) for ids in groups], 1)  # (B, G, h4, w4)
        ph, pw = pixel_mask.shape[-2:]
        gh, gw = grouped.shape[-2:]
        for j, i in enumerate(idx):
            vh = int(pixel_mask[j].any(1).sum())  # valid (unpadded) input region
            vw = int(pixel_mask[j].any(0).sum())
            g = grouped[j:j + 1, :, : round(vh * gh / ph), : round(vw * gw / pw)]
            w, h = sizes[j]
            size = (max(1, round(h * args.scale)), max(1, round(w * args.scale)))
            g = F.interpolate(g, size=size, mode="bilinear", align_corners=False)[0]
            d = depth_map(depth_model, dins[j].to(dev))
            d = F.interpolate(d[None, None], size=size, mode="bilinear", align_corners=False)[0]
            maps = (torch.cat([g, d]).clamp(0, 1) * 255).round().byte().cpu().numpy()
            tmp = dst / f"{paths[i].stem}.tmp.npz"  # write + rename: a killed run leaves no corrupt file
            np.savez_compressed(tmp, maps=maps)
            tmp.rename(dst / f"{paths[i].stem}.npz")
        bar.update(len(idx))
    bar.close()


if __name__ == "__main__":
    main()
