"""Gradio test UI.

match (stage 2):    upload a photo, get 3 graded options side by side. Each
                    option takes its look from a reference image: either one
                    you upload (a film still, a photo you like) or a preset
                    creative direction (teal & orange, warm film, ...).
adaptive (stage 1): upload a photo -> auto-enhanced result.

The LUT is predicted from 256px copies and applied to the full-resolution
photo, the same split as on device. Intensity blends the predicted LUT with
the identity LUT (what a "strength" slider in the app would do).

Usage:
  python scripts/app.py --ckpt runs/match_unsplash/best.pt
  python scripts/app.py --ckpt runs/adaptive_fivek/best.pt
Then open http://localhost:7860 in a Windows browser.
"""
import argparse
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F

from grademl.data import data_root, load_rgb, resize
from grademl.lut import apply_lut, identity_lut, write_cube
from grademl.model import AdaptiveLUTNet, MatchLUTNet
from grademl.pro_looks import DIRECTIONS, random_pro_look
from grademl.synthetic import list_images

N_OPTIONS = 3
DEFAULT_DIRECTIONS = ["golden_hour", "teal_orange", "portra"]
CUSTOM = "custom (upload a reference)"


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img[..., :3])).permute(2, 0, 1).float() / 255


def to_image(t: torch.Tensor) -> np.ndarray:
    return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def load_model(path: Path, device):
    ckpt = torch.load(path, map_location="cpu")
    arch = ckpt.get("arch", "adaptive")
    model = (MatchLUTNet if arch == "match" else AdaptiveLUTNet)(**ckpt["cfg"]["model"])
    model.load_state_dict(ckpt["model"])
    return model.eval().to(device), arch, ckpt


def preset_canvas() -> torch.Tensor:
    """2x2 collage of varied held-out photos: preset looks are shown on it so
    the reference covers skin, sky, foliage, shadows and highlights."""
    photos = list_images(data_root() / "unsplash" / "images_384")
    if not photos:
        photos = list_images(data_root() / "fivek" / "expertC")
    picks = [photos[i] for i in np.linspace(0, len(photos) - 1, 4).astype(int)]
    tiles = [F.interpolate(load_rgb(p)[None], size=(192, 288), mode="bilinear", antialias=True)[0]
             for p in picks]
    return torch.cat([torch.cat(tiles[:2], 2), torch.cat(tiles[2:], 2)], 1)


def build_match_ui(model, ckpt, device, title):
    ident = identity_lut(model.luts.basis.shape[-1], device=device)
    canvas = preset_canvas()

    def preset_reference(direction: str, seed: float):
        if direction == CUSTOM:
            return gr.update()
        # Preset references show the recipe at full strength (no random intensity).
        look = random_pro_look(torch.Generator().manual_seed(int(seed)), direction=direction, strength=(1.0, 1.0))
        return to_image(apply_lut(canvas[None], look)[0])

    two_heads = model.match_head is not None

    size = model.luts.basis.shape[-1]

    @torch.no_grad()
    def generate(photo, strength, *per_option):
        if photo is None:
            raise gr.Error("Upload a photo first")
        n = N_OPTIONS
        refs, matches, directions, seeds = (per_option[i * n:(i + 1) * n] for i in range(4))
        x = to_tensor(photo).to(device)[None]
        net_photo = resize(x)
        gallery, cubes = [], []
        tmp = Path(tempfile.mkdtemp())
        for k, (ref, m, d, sd) in enumerate(zip(refs, matches, directions, seeds)):
            if d != CUSTOM:
                # A preset is a known recipe: apply its LUT directly. Asking
                # the model to re-infer it from a reference image only loses
                # strength (regression to the mean).
                lut = random_pro_look(torch.Generator().manual_seed(int(sd)), size=size,
                                      direction=d, strength=(1.0, 1.0)).to(device)
            elif ref is None:
                continue
            else:
                pred = model(net_photo, resize(to_tensor(ref).to(device)[None]))
                if two_heads:  # blend "grade only" -> "match colors too"
                    lut = pred[0][0] + m * (pred[1][0] - pred[0][0])
                else:
                    lut = pred[0]
            lut = ident + strength * (lut - ident)
            gallery.append((to_image(apply_lut(x, lut)[0]), f"Option {k + 1}"))
            cube = tmp / f"option_{k + 1}.cube"
            write_cube(lut, cube, title=f"grademl option {k + 1}")
            cubes.append(str(cube))
        return [(photo[..., :3], "Original")] + gallery, cubes

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"## {title}\nModel `{ckpt.get('cfg', {}).get('name')}` epoch {ckpt.get('epoch')}, "
                    f"val PSNR {ckpt.get('psnr', float('nan')):.2f} dB. Each option takes its look from the "
                    "reference above it. Presets (🎲 for another variation) apply a pro-style recipe "
                    "directly; for an uploaded reference the model predicts the LUT. *Match* = 0 copies only "
                    "the reference's grade (graded film stills); 1 also maps your photo's palette onto the "
                    "reference's (any photo you like).")
        with gr.Row():
            photo = gr.Image(label="Your photo", type="numpy", height=320)
        refs, matches, directions, seeds = [], [], [], []
        with gr.Row():
            for k in range(N_OPTIONS):
                with gr.Column():
                    direction = gr.Dropdown([CUSTOM, *DIRECTIONS], value=DEFAULT_DIRECTIONS[k],
                                            label=f"Option {k + 1}: look source")
                    seed = gr.Number(value=k, precision=0, visible=False)
                    ref = gr.Image(label="Reference", type="numpy", height=200,
                                   value=preset_reference(DEFAULT_DIRECTIONS[k], k))
                    dice = gr.Button("🎲 another variation", size="sm")
                    match = gr.Slider(0, 1, value=0.0, step=0.05, label="Match: grade only -> colors too",
                                      visible=two_heads)
                    direction.change(preset_reference, [direction, seed], ref)
                    direction.change(lambda d: 1.0 if d == CUSTOM else 0.0, direction, match)
                    dice.click(lambda s: s + 100, seed, seed).then(preset_reference, [direction, seed], ref)
                    ref.upload(lambda: CUSTOM, None, direction)
                    refs.append(ref)
                    matches.append(match)
                    directions.append(direction)
                    seeds.append(seed)
        with gr.Row():
            strength = gr.Slider(0, 1.5, value=1.0, step=0.05, label="Intensity")
            btn = gr.Button("Grade my photo", variant="primary")
        gallery = gr.Gallery(label="Results", columns=N_OPTIONS + 1, height=420, object_fit="contain")
        cubes = gr.Files(label="LUTs (.cube) for each option")
        inputs = [photo, strength, *refs, *matches, *directions, *seeds]
        btn.click(generate, inputs, [gallery, cubes])
        strength.release(generate, inputs, [gallery, cubes])
        for m in matches:
            m.release(generate, inputs, [gallery, cubes])
    return demo


def build_adaptive_ui(model, ckpt, device, title):
    ident = identity_lut(model.luts.basis.shape[-1], device=device)

    @torch.no_grad()
    def run(photo, strength):
        if photo is None:
            raise gr.Error("Upload a photo")
        x = to_tensor(photo).to(device)[None]
        lut = model(resize(x))[0]
        lut = ident + strength * (lut - ident)
        cube = Path(tempfile.mkdtemp()) / "grademl_auto.cube"
        write_cube(lut, cube)
        return (photo[..., :3], to_image(apply_lut(x, lut)[0])), str(cube)

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"## {title}\nEpoch {ckpt.get('epoch')}, test PSNR {ckpt.get('psnr', float('nan')):.2f} dB")
        with gr.Row():
            with gr.Column(scale=1):
                photo = gr.Image(label="Your photo", type="numpy")
                strength = gr.Slider(0, 1.5, value=1.0, step=0.05, label="Intensity")
                btn = gr.Button("Apply", variant="primary")
                cube = gr.File(label="Predicted LUT (.cube)")
            with gr.Column(scale=2):
                result = gr.ImageSlider(label="Before / after", type="numpy")
        btn.click(run, [photo, strength], [result, cube])
        strength.release(run, [photo, strength], [result, cube])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="public gradio.live link")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, arch, ckpt = load_model(args.ckpt, device)
    if arch == "match":
        demo = build_match_ui(model, ckpt, device, "Match the Look")
    else:
        demo = build_adaptive_ui(model, ckpt, device, "Adaptive LUT auto-enhance")
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
