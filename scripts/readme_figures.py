"""README figures from photos that are NOT training data.

The Unsplash dataset terms forbid publishing any part of the dataset, so the
figures use photos downloaded individually from unsplash.com under the
Unsplash License (id, photographer, image
URL, page in docs/images/photos.json; none of them is in the training set;
missing ones are downloaded into data/readme_photos). Writes docs/images/*.jpg
and docs/images/CREDITS.md.

  editor.jpg   original | the AI editor's top 3 (GradeNet-L v4)
  local.jpg    original | one LUT per photo (GradeNet v1) | GradeNet-L v4, same look
  match.jpg    photo | reference | global transfer | content-aware (DINOv2) | MatchNet-L

Usage: python scripts/readme_figures.py
"""
import json
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from grademl.data import load_rgb
from grademl.editor import local_edit, local_top_versions
from grademl.grade import STYLES, grade_luts
from grademl.local import LocalGradeNet, MatchLocalNet, apply_local
from grademl.lut import apply_lut
from grademl.mask import person_mask
from grademl.model import GradeNet
from grademl.semantic_match import semantic_match
from grademl.transfer import transfer_lut

REPO = Path(__file__).resolve().parents[1]
PHOTOS = REPO / "data/readme_photos"
OUT = REPO / "docs/images"

EDITOR = ["nq77bkITPIw", "NzT_APyJVUw", "T8-Y9IwPFxE", "zwdkxQZu0Ko", "tShk_bW2PtU", "0RuxwyVlnhE", "N_XU4dkhXWU"]
LOCAL = [("Ejs3dmfZEpo", "golden_hour"), ("1GH_dWF-4m0", "teal_orange"), ("oUp9WYowj-c", "cinestill_night"),
         ("N_XU4dkhXWU", "portra")]
MATCH = [("NzT_APyJVUw", "0heMYir_u_E"), ("T8-Y9IwPFxE", "CQIu3lBuDQs"), ("oUp9WYowj-c", "SjzuQoUE5Ig"),
         ("zwdkxQZu0Ko", "s1oMlP-RlqU")]


def font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default(size=size)


CELL = 340  # every image is fitted into a CELL x CELL box, so columns line up


def to_pil(t: torch.Tensor) -> Image.Image:
    a = (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    im = Image.fromarray(a)
    s = CELL / max(im.size)
    return im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)


def grid(rows: list[list[Image.Image]], header: list[str], captions: list[list[str]] | None = None) -> Image.Image:
    """Images centered in CELL x CELL boxes; column headers on top, optional caption under each."""
    gap, head, cap = 10, 44, 28 if captions else 0
    cols = len(header)
    width = gap + cols * (CELL + gap)
    height = head + len(rows) * (CELL + cap + gap)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    f, fc = font(20), font(16)
    for j, h in enumerate(header):
        draw.text((gap + j * (CELL + gap) + CELL / 2, head / 2), h, fill=(30, 30, 30), font=f, anchor="mm")
    for i, r in enumerate(rows):
        y0 = head + i * (CELL + cap + gap)
        for j, im in enumerate(r):
            x0 = gap + j * (CELL + gap)
            canvas.paste(im, (x0 + (CELL - im.width) // 2, y0 + (CELL - im.height) // 2))
            if captions and captions[i][j]:
                draw.text((x0 + CELL / 2, y0 + CELL + cap / 2), captions[i][j], fill=(90, 90, 90), font=fc,
                          anchor="mm")
    return canvas


def nice(style: str) -> str:
    return {"none": "natural"}.get(style, style.replace("_", " "))


def load(path: str, cls, dev, **kw):
    ck = torch.load(REPO / path, map_location="cpu")
    net = cls(**kw, **ck["cfg"]["model"])
    net.load_state_dict(ck["model"])
    return net.to(dev).eval()


@torch.no_grad()
def main() -> None:
    dev = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {c["id"]: c for c in json.loads((OUT / "photos.json").read_text())}
    PHOTOS.mkdir(parents=True, exist_ok=True)
    for pid, c in meta.items():
        if not (PHOTOS / f"{pid}.jpg").exists():
            urllib.request.urlretrieve(c["image"] + "?w=1080&q=90&fm=jpg", PHOTOS / f"{pid}.jpg")
    img = {pid: load_rgb(PHOTOS / f"{pid}.jpg").to(dev) for pid in meta}
    gl = load("runs/grade_local_v4/best.pt", LocalGradeNet, dev, cond_dim=len(STYLES))
    v1 = load("runs/grade_v1/best.pt", GradeNet, dev, n_styles=len(STYLES))
    ml = load("runs/match_local_v2/best.pt", MatchLocalNet, dev)
    used = set()

    rows, caps = [], []
    for pid in EDITOR:
        x = img[pid]
        edits = local_top_versions(x, gl, person_mask(x), n=3)
        rows.append([to_pil(x)] + [to_pil(e.render(x)) for e in edits])
        caps.append([""] + [f"#{e.rank} {nice(e.style)} {e.intensity:.0%}" for e in edits])
        used.add(pid)
    grid(rows, ["original", "AI editor #1", "#2", "#3"], caps).save(OUT / "editor.jpg", quality=88)

    rows, caps = [], []
    for pid, style in LOCAL:
        x = img[pid]
        _, looks = grade_luts(v1, x, [style])
        rows.append([to_pil(x), to_pil(apply_lut(x[None], looks[0])[0]), to_pil(local_edit(gl, x, style).render(x))])
        caps.append(["", nice(style), nice(style)])
        used.add(pid)
    grid(rows, ["original", "one LUT per photo (v1)", "GradeNet-L (local)"], caps).save(OUT / "local.jpg", quality=88)

    rows = []
    for pid, rid in MATCH:
        x, r = img[pid], img[rid]
        glob = apply_lut(x[None], transfer_lut(x[None], r[None]))[0].clamp(0, 1)
        sem = semantic_match(x, r).render(x)
        small = F.interpolate(x[None], size=(256, 256), mode="area")
        rs = F.interpolate(r[None], size=(256, 256), mode="area")
        base_lut = transfer_lut(small, rs)
        luts, grid_ = ml(apply_lut(small, base_lut).clamp(0, 1), rs)
        student = apply_local(apply_lut(x[None], base_lut).clamp(0, 1), luts, grid_)[0]
        rows.append([to_pil(x), to_pil(r), to_pil(glob), to_pil(sem), to_pil(student)])
        used |= {pid, rid}
    grid(rows, ["photo", "reference", "global transfer", "content-aware", "MatchNet-L"]
         ).save(OUT / "match.jpg", quality=88)

    lines = ["# Photo credits", "",
             "All photos in these figures are from [Unsplash](https://unsplash.com), used under the",
             "[Unsplash License](https://unsplash.com/license), downloaded individually (none of them is",
             "in the training data). Edited versions are produced by the models in this repository.", ""]
    for pid in sorted(used, key=lambda p: meta[p]["photographer"]):
        lines.append(f"- Photo by {meta[pid]['photographer']} on Unsplash: {meta[pid]['page']}")
    (OUT / "CREDITS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for f in sorted(OUT.glob("*.jpg")):
        print(f"{f.relative_to(REPO)}  {f.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
