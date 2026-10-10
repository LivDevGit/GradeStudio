"""Export a LUT network to Core ML (.mlpackage).

The model takes 256x256 RGB image(s) and outputs the LUT (1, 3, D, D, D),
layout lut[c, b, g, r]. Applying it to the full-resolution photo is done in
the app (Metal / CIColorCube), not in the model.

Also writes <name>.testvec.npz (inputs + PyTorch output) so the Mac side can
check that Core ML predictions match PyTorch.

Usage:
  python scripts/export_coreml.py --model match              # untrained, randomized head
  python scripts/export_coreml.py --model adaptive --ckpt runs/x/best.pt
"""
import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from grademl.model import INPUT_SIZE, AdaptiveLUTNet, MatchLUTNet

REPO = Path(__file__).resolve().parents[1]


def build(kind: str, ckpt: str | None) -> torch.nn.Module:
    cls = AdaptiveLUTNet if kind == "adaptive" else MatchLUTNet
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        model = cls(**state["cfg"]["model"])
        model.load_state_dict(state["model"])
    else:
        # Untrained nets output a constant identity LUT (zero bases), which
        # would make the parity check meaningless. Randomize the bases so the
        # output actually depends on the inputs.
        model = cls()
        torch.manual_seed(0)
        with torch.no_grad():
            model.luts.basis.normal_(std=0.05)
    return model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["adaptive", "match"], default="match")
    ap.add_argument("--ckpt")
    ap.add_argument("--out", type=Path, default=REPO / "exports")
    ap.add_argument("--fp32", action="store_true", help="keep float32 compute (default float16)")
    args = ap.parse_args()

    model = build(args.model, args.ckpt)
    names = ["content"] if args.model == "adaptive" else ["content", "reference"]

    # uint8 test images, exactly what the app feeds through the ImageType.
    rng = np.random.default_rng(1)
    images = [rng.integers(0, 256, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8) for _ in names]
    examples = tuple(torch.from_numpy(im).permute(2, 0, 1)[None].float() / 255 for im in images)
    with torch.no_grad():
        traced = torch.jit.trace(model, examples)
        out = model(*examples)
    # Two-head match models return (look, match).
    outs = out if isinstance(out, tuple) else (out,)
    out_names = ["lut_look", "lut_match"] if len(outs) == 2 else ["lut"]
    expected = {f"expected_{n}": o.numpy() for n, o in zip(out_names, outs)}

    # ImageType: the app passes a 256x256 CVPixelBuffer directly; Core ML
    # scales 0..255 -> 0..1 to match training.
    inputs = [
        ct.ImageType(name=n, shape=(1, 3, INPUT_SIZE, INPUT_SIZE), scale=1 / 255.0,
                     color_layout=ct.colorlayout.RGB)
        for n in names
    ]
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=[ct.TensorType(name=n) for n in out_names],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32 if args.fp32 else ct.precision.FLOAT16,
    )
    mlmodel.short_description = (
        "Predicts 3D LUT(s) (1,3,D,D,D), layout lut[c,b,g,r], from 256x256 RGB input(s). "
        "lut_look: the reference's grade only; lut_match: also matches its colors. "
        "Blend lut_look -> lut_match for a 'match strength' slider."
    )

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_lut"
    path = args.out / f"{stem}.mlpackage"
    mlmodel.save(str(path))

    np.savez_compressed(args.out / f"{stem}.testvec.npz", **expected, **dict(zip(names, images)))

    ops = sorted({op.op_type for f in mlmodel._mil_program.functions.values() for op in f.operations})
    n_params = sum(p.numel() for p in model.parameters())
    size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
    print(f"saved {path}  ({size_mb:.1f} MB, {n_params / 1e3:.0f}K params)")
    print(f"MIL ops: {', '.join(ops)}")


if __name__ == "__main__":
    main()
