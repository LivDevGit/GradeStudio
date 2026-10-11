"""Export a LUT network to Core ML (.mlpackage).

The model takes 256x256 RGB image(s) and outputs the LUT (1, 3, D, D, D),
layout lut[c, b, g, r]. Applying it to the full-resolution photo is done in
the app (Metal / CIColorCube), not in the model.

Also writes <name>.testvec.npz (inputs + PyTorch output) so the Mac side can
check that Core ML predictions match PyTorch.

GradeNet (--model grade) takes the photo plus a `style` one-hot (1, S) in
the order of grademl.grade.STYLES ("none" first); the order is also written
to <name>.styles.json for the app.

Usage:
  python scripts/export_coreml.py --model match              # untrained, randomized head
  python scripts/export_coreml.py --model adaptive --ckpt runs/x/best.pt
  python scripts/export_coreml.py --model grade --ckpt runs/grade_v1/best.pt
"""
import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from grademl.grade import STYLES
from grademl.model import INPUT_SIZE, AdaptiveLUTNet, GradeNet, MatchLUTNet

REPO = Path(__file__).resolve().parents[1]


def build(kind: str, ckpt: str | None) -> torch.nn.Module:
    if kind == "grade":
        cls = lambda **kw: GradeNet(len(STYLES), **kw)  # noqa: E731
    else:
        cls = AdaptiveLUTNet if kind == "adaptive" else MatchLUTNet
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        if kind == "grade" and state.get("styles") != list(STYLES):
            raise SystemExit(f"{ckpt} was trained with styles {state.get('styles')}, code has {list(STYLES)}")
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
    ap.add_argument("--model", choices=["adaptive", "match", "grade"], default="match")
    ap.add_argument("--ckpt")
    ap.add_argument("--out", type=Path, default=REPO / "exports")
    ap.add_argument("--fp32", action="store_true", help="keep float32 compute (default float16)")
    args = ap.parse_args()

    model = build(args.model, args.ckpt)
    names = ["content", "reference"] if args.model == "match" else ["content"]

    # uint8 test images, exactly what the app feeds through the ImageType.
    rng = np.random.default_rng(1)
    images = [rng.integers(0, 256, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8) for _ in names]
    examples = tuple(torch.from_numpy(im).permute(2, 0, 1)[None].float() / 255 for im in images)
    extra = {}
    if args.model == "grade":
        style = np.zeros((1, len(STYLES)), dtype=np.float32)
        style[0, STYLES.index("portra")] = 1
        extra["style"] = style
        examples = examples + (torch.from_numpy(style),)
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
    if args.model == "grade":
        inputs.append(ct.TensorType(name="style", shape=(1, len(STYLES))))
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=[ct.TensorType(name=n) for n in out_names],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32 if args.fp32 else ct.precision.FLOAT16,
    )
    if args.model == "grade":
        mlmodel.short_description = (
            "GradeNet: 3D LUT (1,3,D,D,D), layout lut[c,b,g,r], that corrects and grades this 256x256 "
            f"photo in the style given as a one-hot (1,{len(STYLES)}): {', '.join(STYLES)}."
        )
    else:
        mlmodel.short_description = (
            "Predicts 3D LUT(s) (1,3,D,D,D), layout lut[c,b,g,r], from 256x256 RGB input(s). "
            "lut_look: the reference's grade only; lut_match: also matches its colors. "
            "Blend lut_look -> lut_match for a 'match strength' slider."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_lut"
    path = args.out / f"{stem}.mlpackage"
    mlmodel.save(str(path))

    np.savez_compressed(args.out / f"{stem}.testvec.npz", **expected, **dict(zip(names, images)), **extra)
    if args.model == "grade":
        (args.out / f"{stem}.styles.json").write_text(json.dumps(list(STYLES)))

    ops = sorted({op.op_type for f in mlmodel._mil_program.functions.values() for op in f.operations})
    n_params = sum(p.numel() for p in model.parameters())
    size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
    print(f"saved {path}  ({size_mb:.1f} MB, {n_params / 1e3:.0f}K params)")
    print(f"MIL ops: {', '.join(ops)}")


if __name__ == "__main__":
    main()
