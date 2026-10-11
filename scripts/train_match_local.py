"""Train MatchNet-L (grademl.local.MatchLocalNet): content-aware reference
matching distilled from the DINOv2 teacher (grademl.semantic_match) into a
small on-device network.

Residual mode (model.residual / cfg "residual: true"): the student starts
from the global transfer's result (grademl.transfer.transfer_lut, analytic;
on device the existing match model's match head reproduces it) and only
learns the region-by-region correction on top, so the global transfer is
its floor instead of a bar to climb.

Pairs and teacher targets come from scripts/make_match_pairs.py. Validation
reports PSNR against the teacher for the student, the global transfer
(grademl.transfer.transfer_lut, what a LUT-only method gives) and no-op.

Usage:
  python scripts/train_match_local.py --config configs/match_local_v1.yaml
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_image
from torchvision.utils import make_grid

from grademl.data import data_root, load_rgb, resize
from grademl.local import MatchLocalNet, apply_local, grid_tv
from grademl.lut import apply_lut, lut_monotonicity, lut_smoothness
from grademl.metrics import psnr
from grademl.pro_looks import random_pro_look
from grademl.transfer import transfer_lut
from grademl.utils import seed_everything, worker_init

REPO = Path(__file__).resolve().parents[1]


class TeacherPairs(Dataset):
    """(photo, reference, teacher target), all 256px, as make_match_pairs.py built them."""

    def __init__(self, cfg: dict, split: str, limit: int | None = None):
        root = data_root()
        self.images = root / cfg["images"]
        self.out = root / cfg["teacher"]
        pairs = json.loads((self.out / "pairs.json").read_text())
        self.items = [(i, p) for i, p in enumerate(pairs) if p["split"] == split
                      and (self.out / "targets" / f"{i}.jpg").exists()][:limit]
        self.train = split == "train"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, n):
        i, p = self.items[n]
        photo = resize(load_rgb(self.images / p["photo"]))
        ref = resize(load_rgb(self.images / p["reference"]))
        if p["look"] >= 0:
            lut = random_pro_look(torch.Generator().manual_seed(p["look"]), 33)
            ref = apply_lut(ref[None], lut)[0].clamp(0, 1)
        tgt = read_image(str(self.out / "targets" / f"{i}.jpg")).float() / 255
        if self.train:
            if random.random() < 0.5:
                photo, tgt = photo.flip(-1), tgt.flip(-1)
            if random.random() < 0.5:
                ref = ref.flip(-1)
        return photo, ref, tgt


def run(model, photo, ref, residual: bool):
    """-> (output, luts, grid). Residual: the student grades the globally transferred photo."""
    base = apply_lut(photo, transfer_lut(photo, ref)).clamp(0, 1) if residual else photo
    luts, grid = model(base, ref)
    return apply_local(base, luts, grid), luts, grid


@torch.no_grad()
def evaluate(model, loader, device, residual: bool = False):
    model.eval()
    s = {"psnr": [], "global_transfer": [], "no_op": []}
    samples = []
    for photo, ref, tgt in loader:
        photo, ref, tgt = photo.to(device), ref.to(device), tgt.to(device)
        out = run(model, photo, ref, residual)[0]
        s["psnr"] += psnr(out, tgt).tolist()
        s["global_transfer"] += psnr(apply_lut(photo, transfer_lut(photo, ref)), tgt).tolist()
        s["no_op"] += psnr(photo, tgt).tolist()
        if len(samples) < 4 * 8:
            for j in range(min(2, len(photo))):
                samples += [photo[j], ref[j], out[j].clamp(0, 1), tgt[j]]
    model.train()
    return {k: float(np.mean(v)) for k, v in s.items()}, make_grid(samples, nrow=4, padding=4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO / "configs/match_local_v1.yaml")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--name")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--init", type=Path, help="start from these weights")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.name:
        cfg["name"] = args.name
    tc, dc = cfg["train"], cfg["data"]
    seed_everything(cfg["seed"])
    device = torch.device("cuda")
    run_dir = REPO / "runs" / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    writer = SummaryWriter(run_dir)

    train_set, val_set = TeacherPairs(dc, "train", args.limit), TeacherPairs(dc, "val", dc["val_samples"])
    train_loader = DataLoader(train_set, batch_size=dc["batch_size"], shuffle=True, drop_last=True,
                              num_workers=dc["workers"], pin_memory=True, persistent_workers=True,
                              worker_init_fn=worker_init, prefetch_factor=4)
    val_loader = DataLoader(val_set, batch_size=50, num_workers=8)
    model = MatchLocalNet(**cfg["model"]).to(device)
    residual = cfg.get("residual", False)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu")["model"])
        print(f"initialized from {args.init}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    total = tc["epochs"] * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc["lr"], total_steps=total,
                                                pct_start=min(0.3, tc["warmup_steps"] / total), anneal_strategy="cos")
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, {len(train_set)} train / "
          f"{len(val_set)} val pairs, {len(train_loader)} it/epoch", flush=True)

    best, step = float("-inf"), 0
    for epoch in range(1, tc["epochs"] + 1):
        t0, running = time.time(), 0.0
        for photo, ref, tgt in train_loader:
            photo, ref, tgt = (t.to(device, non_blocking=True) for t in (photo, ref, tgt))
            out, luts, grid = run(model, photo, ref, residual)
            l1 = F.l1_loss(out, tgt)
            loss = (l1 + tc["lambda_smooth"] * lut_smoothness(model.luts.basis)
                    + tc["lambda_mono"] * lut_monotonicity(luts, own_channel_only=True)
                    + tc["lambda_grid"] * grid_tv(grid))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += l1.item()
            step += 1
            if step % 50 == 0:
                writer.add_scalar("train/l1", l1.item(), step)
        msg = f"epoch {epoch:3d}  train l1 {running / len(train_loader):.4f}  {time.time() - t0:5.1f}s"
        if epoch % tc["eval_every"] == 0 or epoch == tc["epochs"]:
            scores, grid_img = evaluate(model, val_loader, device, residual)
            for k, v in scores.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            writer.add_image("val/photo_ref_pred_teacher", grid_img, epoch)
            msg += (f"  | val psnr {scores['psnr']:.2f} (global transfer {scores['global_transfer']:.2f}, "
                    f"no-op {scores['no_op']:.2f})")
            ckpt = {"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "arch": "match_local", "residual": residual, **scores}
            torch.save(ckpt, run_dir / "last.pt")
            if scores["psnr"] > best:
                best = scores["psnr"]
                torch.save(ckpt, run_dir / "best.pt")
                msg += "  *"
        print(msg, flush=True)
    writer.close()
    print(f"best val psnr {best:.2f} -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
