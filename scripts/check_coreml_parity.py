"""Run on a Mac: check the exported Core ML model against PyTorch.

Loads exports/match_lut.mlpackage and the test vector written next to it by
export_coreml.py (uint8 input images + the LUTs PyTorch produced for them),
runs Core ML on CPU, GPU and Neural Engine, and reports the max / mean
difference plus timing. FP16 compute should land within ~1e-2 max, ~1e-3 mean.

Usage (macOS, Python with coremltools + pillow):
  python scripts/check_coreml_parity.py [--model exports/match_lut.mlpackage]
"""
import argparse
import time
from pathlib import Path

import coremltools as ct
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=REPO / "exports" / "match_lut.mlpackage")
    args = ap.parse_args()
    vec = np.load(args.model.with_suffix("").with_suffix(".testvec.npz"))
    inputs = {k: Image.fromarray(vec[k]) for k in ("content", "reference") if k in vec}
    if "style" in vec:  # GradeNet: style one-hot
        inputs["style"] = vec["style"]
    expected = {k.removeprefix("expected_"): vec[k] for k in vec.files if k.startswith("expected_")}

    for unit in [ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_GPU, ct.ComputeUnit.ALL]:
        model = ct.models.MLModel(str(args.model), compute_units=unit)
        model.predict(inputs)  # warm-up
        t0 = time.perf_counter()
        for _ in range(10):
            out = model.predict(inputs)
        ms = (time.perf_counter() - t0) / 10 * 1000
        diffs = {k: np.abs(out[k] - v) for k, v in expected.items()}
        report = ", ".join(f"{k}: max {d.max():.4f} mean {d.mean():.5f}" for k, d in diffs.items())
        print(f"{unit.name:12s} {ms:6.1f} ms | {report}")


if __name__ == "__main__":
    main()
