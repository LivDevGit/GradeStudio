"""Stage 1 training: image-adaptive 3D LUT on FiveK.

Usage:
  python scripts/train_adaptive.py --config configs/adaptive_fivek.yaml
  python scripts/train_adaptive.py --config ... --epochs 1 --name smoke   # quick check

Writes runs/<name>/{best.pt,last.pt,config.yaml} and TensorBoard logs:
  tensorboard --logdir runs
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

from grademl.data import FiveKPairs
from grademl.lut import apply_lut, lut_monotonicity, lut_smoothness
from grademl.metrics import delta_e76, psnr
from grademl.model import AdaptiveLUTNet
from grademl.utils import seed_everything, worker_init

REPO = Path(__file__).resolve().parents[1]


@torch.no_grad()
def evaluate(model, loader, device, n_samples: int = 4):
    """Predict the LUT from the 256px input, apply it at full resolution."""
    model.eval()
    scores = {"psnr": [], "delta_e": [], "psnr_identity": []}
    samples = []
    for i, (net_in, inp, tgt) in enumerate(loader):
        net_in, inp, tgt = net_in.to(device), inp.to(device), tgt.to(device)
        out = apply_lut(inp, model(net_in))
        scores["psnr"].append(psnr(out, tgt).item())
        scores["delta_e"].append(delta_e76(out, tgt).item())
        scores["psnr_identity"].append(psnr(inp, tgt).item())
        if i < n_samples:
            samples += [inp[0], out[0].clamp(0, 1), tgt[0]]
    model.train()
    # Grid rows: input | prediction | expert. Resize so differently shaped
    # images fit one grid.
    samples = [F.interpolate(s[None], size=(256, 384), mode="bilinear", antialias=True)[0] for s in samples]
    return {k: float(np.mean(v)) for k, v in scores.items()}, make_grid(samples, nrow=3, padding=4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO / "configs/adaptive_fivek.yaml")
    ap.add_argument("--epochs", type=int, help="override config")
    ap.add_argument("--name", help="override run name")
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

    train_set = FiveKPairs("train", target=dc["target"])
    test_set = FiveKPairs("test", target=dc["target"])
    train_loader = DataLoader(train_set, batch_size=dc["batch_size"], shuffle=True, drop_last=True,
                              num_workers=dc["workers"], pin_memory=True, persistent_workers=True,
                              worker_init_fn=worker_init)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=4)

    model = AdaptiveLUTNet(**cfg["model"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"] * len(train_loader))
    print(f"{sum(p.numel() for p in model.parameters()) / 1e3:.0f}K params, "
          f"{len(train_set)} train / {len(test_set)} test, {len(train_loader)} it/epoch")

    best, step = float("-inf"), 0
    for epoch in range(1, tc["epochs"] + 1):
        t0, running = time.time(), 0.0
        for inp, tgt in train_loader:
            inp, tgt = inp.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            out = apply_lut(inp, model(inp))
            mse = F.mse_loss(out, tgt)
            basis = model.luts.basis
            loss = mse + tc["lambda_smooth"] * lut_smoothness(basis) + tc["lambda_mono"] * lut_monotonicity(basis)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += mse.item()
            step += 1
            if step % 50 == 0:
                writer.add_scalar("train/mse", mse.item(), step)
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/lr", sched.get_last_lr()[0], step)

        train_psnr = 10 * np.log10(1 / (running / len(train_loader)))
        msg = f"epoch {epoch:3d}  train psnr {train_psnr:5.2f}  {time.time() - t0:4.1f}s"

        if epoch % tc["eval_every"] == 0 or epoch == tc["epochs"]:
            scores, grid = evaluate(model, test_loader, device)
            for k, v in scores.items():
                writer.add_scalar(f"test/{k}", v, epoch)
            writer.add_image("test/input_pred_target", grid, epoch)
            msg += (f"  | test psnr {scores['psnr']:.2f} dE {scores['delta_e']:.2f}"
                    f" (identity {scores['psnr_identity']:.2f})")
            ckpt = {"model": model.state_dict(), "cfg": cfg, "epoch": epoch, **scores}
            torch.save(ckpt, run_dir / "last.pt")
            if scores["psnr"] > best:
                best = scores["psnr"]
                torch.save(ckpt, run_dir / "best.pt")
                msg += "  *"
        print(msg, flush=True)

    writer.close()
    print(f"best test psnr {best:.2f} -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
