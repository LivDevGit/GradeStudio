"""Train GradeNet: per-photo correction + grading in a colorist direction.

Each sample: a finished Unsplash photo, a random technical fault and a
style; the input is the faulty photo, the target the finished photo in that
style (grademl.grade.make_grade_batch). Validation also reports what the
fixed preset gives on the faulty photo ("preset"): the gap is what the
per-photo model adds.

Usage:
  python scripts/train_grade.py --config configs/grade_v1.yaml
  python scripts/train_grade.py --config ... --epochs 1 --name smoke --limit 2000
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from grademl.data import data_root
from grademl.grade import STYLES, GradePairs, make_grade_batch, recipe_luts, style_onehot
from grademl.local import LocalGradeNet, apply_local, grid_tv
from grademl.local_looks import FinishBank, LocalGradePairs, make_local_batch
from grademl.lut import apply_lut, lut_monotonicity, lut_smoothness
from grademl.metrics import delta_e76, psnr
from grademl.model import GradeNet
from grademl.synthetic import list_images
from grademl.utils import seed_everything, worker_init

REPO = Path(__file__).resolve().parents[1]


def build(batch, device, recipes, bank):
    """Move a loader batch to the GPU -> (clean photo, input, target, style)."""
    batch = [t.to(device, non_blocking=True) for t in batch]
    if bank is None:
        photo, fault, style = batch
        return (photo, *make_grade_batch(photo, fault, style, recipes), style)
    photo, maps, fault, local, style = batch
    return (photo, *make_local_batch(photo, maps, fault, local, style, recipes, bank), style)


def predict(model, inp, style, device):
    """Graded input + (LUTs, grid or None) for the regularizers."""
    if isinstance(model, LocalGradeNet):
        luts, grid = model(inp, style_onehot(style, device))
        return apply_local(inp, luts, grid), luts, grid
    lut = model(inp, style_onehot(style, device))
    return apply_lut(inp, lut), lut, None


@torch.no_grad()
def evaluate(model, loader, device, recipes, bank=None, n_samples: int = 8):
    model.eval()
    scores = {k: [] for k in ["psnr", "delta_e", "psnr_none", "psnr_styled", "preset_styled", "no_op"]}
    samples = []
    for batch in loader:
        photo, inp, tgt, style = build(batch, device, recipes, bank)
        out = predict(model, inp, style, device)[0]
        p = psnr(out, tgt)
        # Faulty inputs only: on a fault-free photo the fixed preset is the
        # target itself (infinite PSNR), which would swamp the comparison.
        faulty = (inp - photo).abs().flatten(1).amax(1) > 1e-4
        styled = (style > 0) & faulty
        scores["psnr"] += p.tolist()
        scores["delta_e"] += delta_e76(out, tgt).tolist()
        scores["psnr_none"] += p[(style == 0) & faulty].tolist()
        scores["psnr_styled"] += p[styled].tolist()
        # A fixed filter: the recipe applied to the faulty photo as it is.
        scores["preset_styled"] += psnr(apply_lut(inp[styled], recipes[style[styled]]), tgt[styled]).tolist()
        scores["no_op"] += psnr(inp[faulty], tgt[faulty]).tolist()
        for i in range(min(n_samples - len(samples) // 3, len(inp))):
            samples += [inp[i], out[i].clamp(0, 1), tgt[i]]
    model.train()
    return {k: float(np.mean(v)) for k, v in scores.items() if v}, make_grid(samples, nrow=3, padding=4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO / "configs/grade_v1.yaml")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--name")
    ap.add_argument("--limit", type=int, help="use only N train photos (smoke tests)")
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

    photos = list_images(data_root() / dc["images"])
    val_photos = photos[:: dc["val_every_nth"]]
    held_out = set(val_photos)
    train_photos = [p for p in photos if p not in held_out][: args.limit]
    common = dict(p_clean=dc["p_clean"], p_none=dc["p_none"])
    arch = cfg.get("arch", "global")
    if arch == "local":
        maps = data_root() / dc["maps"]
        common["p_local"] = dc.get("p_local", 0.6)
        train_set = LocalGradePairs(train_photos, maps, train=True, **common)
        val_set = LocalGradePairs(val_photos, maps, train=False, length=dc["val_samples"], **common)
    else:
        train_set = GradePairs(train_photos, train=True, **common)
        val_set = GradePairs(val_photos, train=False, length=dc["val_samples"], **common)
    train_loader = DataLoader(train_set, batch_size=dc["batch_size"], shuffle=True, drop_last=True,
                              num_workers=dc["workers"], pin_memory=True, persistent_workers=True,
                              worker_init_fn=worker_init, prefetch_factor=4)
    val_loader = DataLoader(val_set, batch_size=50, num_workers=8)

    if arch == "local":
        model = LocalGradeNet(cond_dim=len(STYLES), **cfg["model"]).to(device)
        bank = FinishBank(STYLES, cfg["model"]["lut_size"], device)
    else:
        model = GradeNet(len(STYLES), **cfg["model"]).to(device)
        bank = None
    recipes = recipe_luts(cfg["model"]["lut_size"], device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    total = tc["epochs"] * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc["lr"], total_steps=total,
                                                pct_start=min(0.3, tc["warmup_steps"] / total),
                                                anneal_strategy="cos")
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, {len(STYLES)} styles, "
          f"{len(train_photos)} train / {len(val_photos)} val photos, {len(train_loader)} it/epoch", flush=True)

    best, step = float("-inf"), 0
    for epoch in range(1, tc["epochs"] + 1):
        t0, running = time.time(), 0.0
        for batch in train_loader:
            _, inp, tgt, style = build(batch, device, recipes, bank)
            out, lut, grid = predict(model, inp, style, device)
            l1 = F.l1_loss(out, tgt)
            loss = (l1 + tc["lambda_smooth"] * lut_smoothness(model.luts.basis)
                    + tc["lambda_mono"] * lut_monotonicity(lut, own_channel_only=True))
            if grid is not None:
                loss = loss + tc.get("lambda_grid", 1e-3) * grid_tv(grid)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += l1.item()
            step += 1
            if step % 50 == 0:
                writer.add_scalar("train/l1", l1.item(), step)
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/lr", sched.get_last_lr()[0], step)

        msg = f"epoch {epoch:3d}  train l1 {running / len(train_loader):.4f}  {time.time() - t0:5.1f}s"
        if epoch % tc["eval_every"] == 0 or epoch == tc["epochs"]:
            scores, grid = evaluate(model, val_loader, device, recipes, bank)
            for k, v in scores.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            writer.add_image("val/input_pred_target", grid, epoch)
            msg += (f"  | val psnr {scores['psnr']:.2f} dE {scores['delta_e']:.2f}  "
                    f"styled {scores['psnr_styled']:.2f} (fixed preset {scores['preset_styled']:.2f})  "
                    f"correction {scores['psnr_none']:.2f}  (no-op {scores['no_op']:.2f})")
            ckpt = {"model": model.state_dict(), "cfg": cfg, "epoch": epoch,
                    "arch": "grade_local" if arch == "local" else "grade",
                    "styles": list(STYLES), **scores}
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
