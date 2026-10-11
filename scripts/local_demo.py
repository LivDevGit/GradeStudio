"""Local test UI for the v7 models (Gradio): GradeNet-L editor and content-aware matching.

  AI Edit          GradeNet-L edits the photo in every style, judges every
                   candidate and shows its top 3, with its reasons. Pick one,
                   set the intensity, add grain, download at full resolution.
  Pick a style     one direction of your choice, edited for this photo.
  Match a photo    give it a reference: content-aware matching (sky to sky,
                   skin to skin; grademl.semantic_match), or the global /
                   palette transfers for comparison.

Every edit is predicted from a 256px copy and applied to the full-resolution
photo (K LUTs + bilateral grid), the same split as on device.

Usage:
  python scripts/local_demo.py                                   # runs/grade_local_v1/best.pt
  python scripts/local_demo.py --ckpt runs/other/best.pt --port 7861
Then open http://localhost:7860.
"""
import argparse
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from grademl.editor import LocalEdit, local_edit, local_top_versions
from grademl.grade import STYLES
from grademl.grain import Grain, add_grain
from grademl.local import LocalGradeNet
from grademl.lut import apply_lut
from grademl.mask import person_mask
from grademl.pro_looks import DIRECTIONS
from grademl.semantic_match import semantic_match
from grademl.transfer import palette_transfer_lut, transfer_lut

REPO = Path(__file__).resolve().parents[1]
MAX_SIDE = 4096  # larger photos are downscaled first (GPU memory)


def to_tensor(img: np.ndarray, device) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(img[..., :3])).permute(2, 0, 1).float().to(device) / 255
    s = MAX_SIDE / max(t.shape[1:])
    if s < 1:
        t = torch.nn.functional.interpolate(t[None], scale_factor=s, mode="area")[0]
    return t


def to_image(t: torch.Tensor) -> np.ndarray:
    return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def save_png(img: np.ndarray, name: str) -> str:
    from PIL import Image

    path = Path(tempfile.gettempdir()) / f"{name}.png"
    Image.fromarray(img).save(path)
    return str(path)


def nice(style: str) -> str:
    return style.replace("_", " ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=REPO / "runs/grade_local_v1/best.pt")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--no-person", action="store_true", help="skip the person mask (skin check)")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu")
    net = LocalGradeNet(cond_dim=len(STYLES), **ck["cfg"]["model"])
    net.load_state_dict(ck["model"])
    net = net.to(dev).eval()

    def finish(edit: LocalEdit, img: torch.Tensor, intensity: float, grain: float, size: float) -> torch.Tensor:
        out = edit.render(img, intensity)
        # slider 0..1 -> grain std 0..0.08 (0.05 is clearly visible)
        return add_grain(out, Grain(amount=0.08 * grain, size=size)) if grain > 0 else out

    # ------------------------------------------------------------ AI Edit
    def ai_edit(photo, grain, size):
        if photo is None:
            raise gr.Error("Upload a photo first.")
        img = to_tensor(photo, dev)
        mask = None if args.no_person else person_mask(img)
        edits = local_top_versions(img, net, mask, n=3)
        gallery = [(to_image(finish(e, img, e.intensity, grain, size)),
                    f"#{e.rank}  {nice(e.style)} · {e.intensity:.0%}") for e in edits]
        first = edits[0]
        return (gallery, edits, 0, gr.update(value=first.intensity), "\n".join(f"- {r}" for r in first.reasons),
                (photo, gallery[0][0]))

    def ai_select(edits, evt: gr.SelectData):
        e = edits[evt.index]
        return evt.index, gr.update(value=e.intensity), "\n".join(f"- {r}" for r in e.reasons)

    def ai_render(photo, edits, idx, intensity, grain, size):
        if photo is None or not edits:
            return None, None
        img = to_tensor(photo, dev)
        out = to_image(finish(edits[idx], img, intensity, grain, size))
        return (photo, out), save_png(out, f"grademl_{edits[idx].style}")

    # ------------------------------------------------------------ Pick a style
    def style_edit(photo, style, intensity, grain, size):
        if photo is None:
            raise gr.Error("Upload a photo first.")
        img = to_tensor(photo, dev)
        name = "none" if style.startswith("none") else style.replace(" ", "_")
        out = to_image(finish(local_edit(net, img, name), img, intensity, grain, size))
        return (photo, out), save_png(out, f"grademl_{name}")

    # ------------------------------------------------------------ Match
    def match(photo, ref, method, strength):
        if photo is None or ref is None:
            raise gr.Error("Upload a photo and a reference.")
        img, r = to_tensor(photo, dev), to_tensor(ref, dev)
        if method.startswith("Content-aware"):
            m = semantic_match(img, r)
            out = m.render(img, strength)
            info = "Regions matched (share of photo → confidence): " + ", ".join(
                f"{float(c):.0%}→{float(q):.0%}" for c, q in zip(m.coverage, m.confidence))
        else:
            fn = transfer_lut if method.startswith("Global") else palette_transfer_lut
            lut = fn(img[None], r[None])[0]
            out = (img + strength * (apply_lut(img[None], lut)[0] - img)).clamp(0, 1)
            info = ""
        out = to_image(out)
        return (photo, out), save_png(out, "grademl_match"), info

    styles = ["none (correction only)"] + [nice(d) for d in DIRECTIONS]
    with gr.Blocks(title="GradeML local demo") as demo:
        gr.Markdown("# GradeML local demo\nAI color grading that edits *your* photo: correction, a look, and the "
                    "local work a colorist does with masks (sky, subject, depth, edges), learned end to end.")
        with gr.Row():
            grain = gr.Slider(0, 1, 0, step=0.05, label="Film grain")
            gsize = gr.Slider(0.5, 3, 1.2, step=0.1, label="Grain size")
        with gr.Tab("✨ AI Edit"):
            with gr.Row():
                with gr.Column(scale=1):
                    photo = gr.Image(label="Your photo", type="numpy")
                    go = gr.Button("Edit my photo", variant="primary")
                    intensity = gr.Slider(0, 1.5, 1.0, step=0.05, label="Intensity")
                    reasons = gr.Markdown()
                with gr.Column(scale=2):
                    gallery = gr.Gallery(label="Editor's top 3 (click to choose)", columns=3, height=260)
                    view = gr.ImageSlider(label="Before / after", type="numpy")
                    download = gr.File(label="Full resolution")
            edits, idx = gr.State([]), gr.State(0)
            go.click(ai_edit, [photo, grain, gsize], [gallery, edits, idx, intensity, reasons, view])
            gallery.select(ai_select, [edits], [idx, intensity, reasons]).then(
                ai_render, [photo, edits, idx, intensity, grain, gsize], [view, download])
            for c in (intensity, grain, gsize):
                c.release(ai_render, [photo, edits, idx, intensity, grain, gsize], [view, download])
        with gr.Tab("🎨 Pick a style"):
            with gr.Row():
                with gr.Column(scale=1):
                    photo2 = gr.Image(label="Your photo", type="numpy")
                    style = gr.Dropdown(styles, value="golden hour", label="Direction")
                    inten2 = gr.Slider(0, 1.5, 1.0, step=0.05, label="Intensity")
                    go2 = gr.Button("Edit", variant="primary")
                with gr.Column(scale=2):
                    view2 = gr.ImageSlider(label="Before / after", type="numpy")
                    download2 = gr.File(label="Full resolution")
            ins2 = [photo2, style, inten2, grain, gsize]
            go2.click(style_edit, ins2, [view2, download2])
            style.change(style_edit, ins2, [view2, download2])
            inten2.release(style_edit, ins2, [view2, download2])
        with gr.Tab("🖼️ Match a photo"):
            with gr.Row():
                with gr.Column(scale=1):
                    photo3 = gr.Image(label="Your photo", type="numpy")
                    ref = gr.Image(label="Reference (a film still, a photo you love)", type="numpy")
                    method = gr.Radio(["Content-aware (sky→sky, skin→skin)", "Global transfer", "Palette transfer"],
                                      value="Content-aware (sky→sky, skin→skin)", label="Method")
                    strength = gr.Slider(0, 1.5, 1.0, step=0.05, label="Strength")
                    go3 = gr.Button("Match", variant="primary")
                with gr.Column(scale=2):
                    view3 = gr.ImageSlider(label="Before / after", type="numpy")
                    info = gr.Markdown()
                    download3 = gr.File(label="Full resolution")
            ins3 = [photo3, ref, method, strength]
            go3.click(match, ins3, [view3, download3, info])
            strength.release(match, ins3, [view3, download3, info])
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
