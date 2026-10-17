"""Build the integration kit for the Grade Studio iOS app (private; not for GitHub).

Collects the Core ML models, their test vectors, the editor calibration, the
preset bank, the Swift / Metal reference code and the Python reference
implementations to port, and writes *golden outputs*: what Python produces
on real photos, so the iOS port can be checked end to end:

  golden/<photo>/input.png                        full-res input (long side 1080)
  golden/<photo>/grade_<style>_<t>.png            GradeNet-L edit at intensity t
  golden/<photo>/grade_<style>.npz                luts (K,3,D,D,D), grid (1,K,G,16,16),
                                                  packed grid (16, G*16, 4) as the app packs it
  golden/<photo>/top3.json                        the editor's top 3 + every candidate's score parts
  golden/match_<photo>_<ref>/...                  global transfer LUT, MatchNet-L outputs, final render

Usage: python scripts/make_integration_kit.py [--out exports/integration_kit] [--zip ~/Downloads/...]
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from grademl.data import load_rgb, resize
from grademl.editor import local_edit, local_top_versions
from grademl.grade import STYLES
from grademl.local import LocalGradeNet, MatchLocalNet, apply_local, pack_grid
from grademl.lut import apply_lut
from grademl.transfer import transfer_lut

REPO = Path(__file__).resolve().parents[1]
PHOTOS = REPO / "data/readme_photos"
GRADE = ["N_XU4dkhXWU", "T8-Y9IwPFxE", "zwdkxQZu0Ko"]          # portrait, landscape with sky, city at dusk
MATCH = [("NzT_APyJVUw", "0heMYir_u_E"), ("oUp9WYowj-c", "SjzuQoUE5Ig")]


def save_png(t: torch.Tensor, path: Path) -> None:
    a = (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    Image.fromarray(a).save(path)


def load_photo(pid: str, dev) -> torch.Tensor:
    x = load_rgb(PHOTOS / f"{pid}.jpg")
    s = 1080 / max(x.shape[1:])
    if s < 1:
        x = F.interpolate(x[None], scale_factor=s, mode="area")[0]
    return x.to(dev)


def load(path: str, cls, dev, **kw):
    ck = torch.load(REPO / path, map_location="cpu")
    net = cls(**kw, **ck["cfg"]["model"])
    net.load_state_dict(ck["model"])
    return net.to(dev).eval()


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "exports/integration_kit")
    ap.add_argument("--zip", type=Path)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "golden").mkdir(parents=True)

    # Models + test vectors (exports/ is produced by scripts/export_coreml.py and export_presets.py).
    ex = REPO / "exports"
    for name in ("grade_local_lut", "match_local_lut", "match_lut"):
        shutil.copytree(ex / f"{name}.mlpackage", out / "models" / f"{name}.mlpackage")
        shutil.copy(ex / f"{name}.testvec.npz", out / "models")
    shutil.copy(ex / "grade_local_lut.styles.json", out / "models")
    shutil.copytree(ex / "presets", out / "presets")
    (out / "config").mkdir()
    shutil.copy(REPO / "configs/editor_calibration.json", out / "config")
    shutil.copytree(REPO / "ios", out / "swift_reference")
    (out / "python_reference").mkdir()
    for f in ("local.py", "editor.py", "grade.py", "surprise.py", "transfer.py", "color.py", "lut.py", "grain.py"):
        shutil.copy(REPO / "src/grademl" / f, out / "python_reference")
    shutil.copy(REPO / "docs/ios_integration.md", out)
    shutil.copy(REPO / "scripts/check_coreml_parity.py", out)

    gl = load("runs/grade_local_v4/best.pt", LocalGradeNet, dev, cond_dim=len(STYLES))
    ml = load("runs/match_local_v2/best.pt", MatchLocalNet, dev)
    summary = {"grade": {}, "match": {}}

    for pid in GRADE:
        x = load_photo(pid, dev)
        d = out / "golden" / pid
        d.mkdir()
        save_png(x, d / "input.png")
        for style in ("none", "golden_hour", "teal_orange"):
            e = local_edit(gl, x, style)
            luts, grid = gl(resize(x)[None], torch.nn.functional.one_hot(
                torch.tensor([STYLES.index(style)], device=dev), len(STYLES)).float())
            packed = pack_grid(grid[0]).permute(1, 2, 0).cpu().numpy()      # (16, G*16, 4) RGBA rows, top first
            np.savez_compressed(d / f"grade_{style}.npz", luts=luts[0].cpu().numpy(), grid=grid.cpu().numpy(),
                                packed_rgba=packed)
            for t in ((1.0,) if style == "none" else (1.0, 0.5)):
                save_png(e.render(x, t), d / f"grade_{style}_{t:g}.png")
        edits = local_top_versions(x, gl, None, n=3)
        top = {"picks": [{"rank": e.rank, "style": e.style, "intensity": e.intensity, "score": e.best.score,
                          "parts": e.best.parts, "reasons": e.reasons} for e in edits],
               "all_candidates": [{"style": c.style, "intensity": c.intensity, "score": c.score, "parts": c.parts}
                                  for c in edits[0].shortlist],
               "note": "no person mask passed (skin part 0); calibration: config/editor_calibration.json"}
        (d / "top3.json").write_text(json.dumps(top, indent=1))
        summary["grade"][pid] = [(p["style"], p["intensity"]) for p in top["picks"]]

    for pid, rid in MATCH:
        x, r = load_photo(pid, dev), load_photo(rid, dev)
        d = out / "golden" / f"match_{pid}_{rid}"
        d.mkdir()
        save_png(x, d / "photo.png")
        save_png(r, d / "reference.png")
        small = F.interpolate(x[None], size=(256, 256), mode="area")
        rs = F.interpolate(r[None], size=(256, 256), mode="area")
        glut = transfer_lut(small, rs)                                     # (1, 3, 33, 33, 33)
        base_small = apply_lut(small, glut).clamp(0, 1)
        luts, grid = ml(base_small, rs)
        final = apply_local(apply_lut(x[None], glut).clamp(0, 1), luts, grid)[0]
        np.savez_compressed(d / "match.npz", global_lut=glut[0].cpu().numpy(), photo_256=small[0].cpu().numpy(),
                            reference_256=rs[0].cpu().numpy(), luts=luts[0].cpu().numpy(), grid=grid.cpu().numpy())
        save_png(apply_lut(x[None], glut)[0], d / "global_transfer.png")
        save_png(final, d / "matchnet_l.png")
        summary["match"][f"{pid}<-{rid}"] = "ok"

    (out / "golden" / "summary.json").write_text(json.dumps(summary, indent=1))
    shutil.copy(REPO / "docs/INTEGRATION.md", out / "INTEGRATION.md")
    if args.zip:
        shutil.make_archive(str(args.zip.with_suffix("")), "zip", out)
        print(f"-> {args.zip}")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"kit at {out} ({size:.0f} MB)")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
