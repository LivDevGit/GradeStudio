"""Stage 2 training: predict the LUT that gives a photo the look of a reference.

Data: Unsplash Lite photos + synthetic looks (see grademl.synthetic). Each
sample is (content, reference graded with look L, content graded with L, L).

Look head loss: L1 on the graded image + L1 on the LUT itself (we know the
exact LUT, which supervises colors that are not in the photo too).
Match head (model.heads=2): same losses against grademl.transfer's classic
color-transfer LUT for (content, reference), computed on the GPU.
Plus LUT smoothness / own-channel monotonicity regularizers.

Usage:
  python scripts/train_match.py --config configs/match_v3.yaml
  python scripts/train_match.py --config ... --epochs 1 --name smoke --limit 2000
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
from grademl.lut import apply_lut, lut_monotonicity, lut_smoothness
from grademl.metrics import delta_e76, psnr
from grademl.model import MatchLUTNet
from grademl.synthetic import SyntheticMatchPairs, list_images, make_match_batch
from grademl.transfer import palette_transfer_lut, transfer_lut
from grademl.utils import seed_everything, worker_init

REPO = Path(__file__).resolve().parents[1]
TEACHERS = {"global": transfer_lut, "palette": palette_transfer_lut}
TEACHER = "global"  # set from cfg["train"]["teacher"] in main()


def as_heads(pred) -> tuple[torch.Tensor, torch.Tensor | None]:
    return (pred[0], pred[1]) if isinstance(pred, tuple) else (pred, None)


def lut_losses(pred_lut, true_lut, content, tc) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(image L1, LUT L1, total) for one head."""
    img_l1 = F.l1_loss(apply_lut(content, pred_lut), apply_lut(content, true_lut))
    lut_l1 = F.l1_loss(pred_lut, true_lut)
    total = (img_l1 + tc["lambda_lut"] * lut_l1
             + tc["lambda_mono"] * lut_monotonicity(pred_lut, own_channel_only=True))
    return img_l1, lut_l1, total


@torch.no_grad()
def evaluate(model, loader, device, n_samples: int = 6):
    model.eval()
    size = model.luts.basis.shape[-1]
    scores = {k: [] for k in ["psnr", "delta_e", "lut_l1", "psnr_identity", "match_psnr", "match_lut_l1"]}
    samples = []
    for content, ref_scene, lut in loader:
        content, ref_scene, lut = content.to(device), ref_scene.to(device), lut.to(device)
        reference, target, lut = make_match_batch(content, ref_scene, lut, size)
        look, match = as_heads(model(content, reference))
        out = apply_lut(content, look)
        scores["psnr"] += psnr(out, target).tolist()
        scores["delta_e"] += delta_e76(out, target).tolist()
        scores["lut_l1"] += (look - lut).abs().flatten(1).mean(1).tolist()
        scores["psnr_identity"] += psnr(content, target).tolist()
        row = [reference, content, out, target]
        if match is not None:
            teacher = TEACHERS[TEACHER](content, reference, size)
            m_out, t_out = apply_lut(content, match), apply_lut(content, teacher)
            scores["match_psnr"] += psnr(m_out, t_out).tolist()
            scores["match_lut_l1"] += (match - teacher).abs().flatten(1).mean(1).tolist()
            row += [m_out, t_out]
        if len(samples) < n_samples * len(row):
            for i in range(min(n_samples - len(samples) // len(row), len(content))):
                samples += [x[i].clamp(0, 1) for x in row]
    model.train()
    # Grid rows: reference | content | look pred | look target [| match pred | teacher]
    ncol = 6 if scores["match_psnr"] else 4
    return ({k: float(np.mean(v)) for k, v in scores.items() if v},
            make_grid(samples, nrow=ncol, padding=4))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO / "configs/match_v3.yaml")
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
    global TEACHER
    TEACHER = tc.get("teacher", "global")

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
    lut_size = cfg["model"]["lut_size"]
    common = dict(p_generic=dc["p_generic"], p_identity=dc.get("p_identity", 0.0), p_pro=dc.get("p_pro", 0.0))
    train_set = SyntheticMatchPairs(train_photos, train=True, p_same=dc.get("p_same", 0.0), **common)
    val_set = SyntheticMatchPairs(val_photos, train=False, length=dc["val_samples"], **common)
    train_loader = DataLoader(train_set, batch_size=dc["batch_size"], shuffle=True, drop_last=True,
                              num_workers=dc["workers"], pin_memory=True, persistent_workers=True,
                              worker_init_fn=worker_init, prefetch_factor=4)
    val_loader = DataLoader(val_set, batch_size=50, num_workers=8)

    model = MatchLUTNet(**cfg["model"])
    if tc.get("init_from"):
        # Warm start (e.g. a single-head checkpoint); the match head starts
        # as a copy of the look head.
        state = torch.load(REPO / tc["init_from"], map_location="cpu")["model"]
        missing, _ = model.load_state_dict(state, strict=False)
        if model.match_head is not None and any(k.startswith("match_head") for k in missing):
            model.match_head.load_state_dict(model.head.state_dict())
        print(f"warm start from {tc['init_from']} (new: {sorted({k.split('.')[0] for k in missing})})")
    model.to(device)
    two_heads = model.match_head is not None

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    total = tc["epochs"] * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc["lr"], total_steps=total,
                                                pct_start=min(0.3, tc["warmup_steps"] / total),
                                                anneal_strategy="cos")
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
          f"{len(train_photos)} train / {len(val_photos)} val photos, {len(train_loader)} it/epoch", flush=True)

    best, step = float("-inf"), 0
    for epoch in range(1, tc["epochs"] + 1):
        t0, running = time.time(), 0.0
        for content, ref_scene, lut in train_loader:
            content, ref_scene, lut = (t.to(device, non_blocking=True) for t in (content, ref_scene, lut))
            reference, _, lut = make_match_batch(content, ref_scene, lut, lut_size)
            look, match = as_heads(model(content, reference))
            img_l1, lut_l1, loss = lut_losses(look, lut, content, tc)
            loss = loss + tc["lambda_smooth"] * lut_smoothness(model.luts.basis)
            if two_heads:
                teacher = TEACHERS[TEACHER](content, reference, lut_size)
                m_img, m_lut, m_loss = lut_losses(match, teacher, content, tc)
                loss = loss + tc.get("lambda_match", 1.0) * m_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += img_l1.item()
            step += 1
            if step % 50 == 0:
                writer.add_scalar("train/img_l1", img_l1.item(), step)
                writer.add_scalar("train/lut_l1", lut_l1.item(), step)
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
                if two_heads:
                    writer.add_scalar("train/match_img_l1", m_img.item(), step)
                    writer.add_scalar("train/match_lut_l1", m_lut.item(), step)

        msg = f"epoch {epoch:3d}  train img_l1 {running / len(train_loader):.4f}  {time.time() - t0:5.1f}s"
        if epoch % tc["eval_every"] == 0 or epoch == tc["epochs"]:
            scores, grid = evaluate(model, val_loader, device)
            for k, v in scores.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            writer.add_image("val/samples", grid, epoch)
            msg += (f"  | val psnr {scores['psnr']:.2f} dE {scores['delta_e']:.2f} "
                    f"lut_l1 {scores['lut_l1']:.4f} (no-op {scores['psnr_identity']:.2f})")
            # Model selection: look PSNR, plus match PSNR when there is a match head.
            score = scores["psnr"]
            if two_heads:
                msg += f"  match psnr {scores['match_psnr']:.2f} lut_l1 {scores['match_lut_l1']:.4f}"
                score = (scores["psnr"] + scores["match_psnr"]) / 2
            ckpt = {"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "arch": "match", **scores}
            torch.save(ckpt, run_dir / "last.pt")
            if score > best:
                best = score
                torch.save(ckpt, run_dir / "best.pt")
                msg += "  *"
        print(msg, flush=True)

    writer.close()
    print(f"best val score {best:.2f} -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
