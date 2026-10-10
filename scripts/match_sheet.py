"""Visual report for a match checkpoint.

Sheet 1 (synthetic): one row per creative direction:
    reference (held-out photo + look) | content | prediction | ground truth
Sheet 2 (real references): ungraded held-out photos used as references,
    i.e. no synthetic look involved; rows: reference | content | prediction.

Usage: python scripts/match_sheet.py --ckpt runs/match_unsplash/best.pt
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.utils import make_grid

from grademl.data import data_root, load_rgb, resize
from grademl.lut import apply_lut
from grademl.metrics import psnr, rgb_to_lab
from grademl.model import MatchLUTNet
from grademl.pro_looks import DIRECTIONS, random_pro_look
from grademl.synthetic import list_images
from grademl.transfer import palette_transfer_lut, transfer_lut

TEACHERS = {'global': transfer_lut, 'palette': palette_transfer_lut}

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
TILE = (192, 288)


def tile(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x[None], size=TILE, mode="bilinear", antialias=True)[0].clamp(0, 1)


def save_sheet(rows: list[list[torch.Tensor]], labels: list[str], header: list[str], out: Path) -> None:
    grid = make_grid([t for r in rows for t in r], nrow=len(rows[0]), padding=3, pad_value=1.0)
    img = Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).round().astype("uint8"))
    canvas = Image.new("RGB", (img.width + 110, img.height + 22), "white")
    canvas.paste(img, (110, 22))
    d = ImageDraw.Draw(canvas)
    for j, h in enumerate(header):
        d.text((110 + 3 + j * (TILE[1] + 3) + 4, 5), h, fill="black")
    for i, lab in enumerate(labels):
        d.text((4, 22 + 3 + i * (TILE[0] + 3) + TILE[0] // 2), lab, fill="black")
    canvas.save(out, quality=92)
    print(f"saved {out}")


def colorfulness(img: torch.Tensor) -> float:
    lab = rgb_to_lab(F.interpolate(img[None], size=(64, 64), mode="area"))
    return lab[0, 1:].pow(2).sum(0).sqrt().mean().item()


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=REPO / "runs/match_unsplash/best.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu")
    model = MatchLUTNet(**ck["cfg"]["model"])
    model.load_state_dict(ck["model"])
    model.eval().to(dev)

    dc = ck["cfg"]["data"]
    val = list_images(data_root() / dc["images"])[:: dc["val_every_nth"]]
    g = torch.Generator().manual_seed(args.seed)
    pick = lambda: val[int(torch.randint(len(val), (), generator=g))]

    def predict(content: torch.Tensor, reference: torch.Tensor, head: int = 0) -> torch.Tensor:
        """head 0 = look, 1 = match (two-head models only)."""
        out = model(resize(content[None].to(dev)), resize(reference[None].to(dev)))
        lut = out[head] if isinstance(out, tuple) else out
        return apply_lut(content[None].to(dev), lut)[0].cpu()

    two_heads = model.match_head is not None

    # Sheet 1: synthetic looks per direction.
    rows, labels, scores = [], [], []
    for d in DIRECTIONS:
        look = random_pro_look(g, direction=d, strength=(1.0, 1.0))
        ref_scene, content = load_rgb(pick()), load_rgb(pick())
        reference = apply_lut(ref_scene[None], look)[0]
        target = apply_lut(content[None], look)[0]
        pred = predict(content, reference)
        scores.append(psnr(pred[None], target[None]).item())
        rows.append([tile(reference), tile(content), tile(pred), tile(target)])
        labels.append(f"{d}\n{scores[-1]:.1f} dB")
    out = REPO / "exports"
    out.mkdir(exist_ok=True)
    save_sheet(rows, labels, ["reference", "your photo", "model", "ground truth"], out / "match_synthetic.jpg")

    # Sheet 2: real photos as references (the most colorful held-out ones).
    pool = [val[int(i)] for i in torch.randperm(len(val), generator=g)[:150]]
    pool.sort(key=lambda p: -colorfulness(load_rgb(p)))
    # Vivid references onto ordinary photos (middle of the colorfulness range;
    # near-gray photos can only ever receive a tint).
    mid = len(pool) // 2
    refs, contents = pool[:6], pool[mid - 3: mid + 3]
    pairs = [(load_rgb(r), load_rgb(c), r.stem[:12]) for r, c in zip(refs, contents)]
    samples = REPO / "data" / "samples"
    if (samples / "arch_original.webp").exists():  # the user's own test pair first
        from preview_pro_looks import load_any
        pairs.insert(0, (load_any(samples / "ref_cinqueterre.jpg"), load_any(samples / "arch_original.webp"), "arch"))
    rows = []
    for reference, content, _ in pairs:
        teacher = TEACHERS[ck['cfg']['train'].get('teacher', 'global')]
        small = lambda x: F.interpolate(x[None], size=(256, 256), mode='area')
        classic = apply_lut(content[None], teacher(small(content), small(reference)))[0]
        row = [tile(reference), tile(content), tile(predict(content, reference))]
        if two_heads:
            row.append(tile(predict(content, reference, head=1)))
        rows.append(row + [tile(classic)])
    header = ["reference (real photo)", "your photo", "model: look"]
    header += ["model: match"] if two_heads else []
    save_sheet(rows, [name for *_, name in pairs], header + ["teacher"],
               out / "match_real.jpg")
    print(f"synthetic mean psnr {sum(scores) / len(scores):.2f} dB")


if __name__ == "__main__":
    main()
