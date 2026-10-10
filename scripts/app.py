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
from grademl.synthetic import DIRECTIONS, list_images, random_look

N_OPTIONS = 3
DEFAULT_DIRECTIONS = ["teal_orange", "warm_film", "cool_moody"]
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
        look = random_look(torch.Generator().manual_seed(int(seed)), direction=direction)
        return to_image(apply_lut(canvas[None], look)[0])

    @torch.no_grad()
    def generate(photo, strength, *refs):
        if photo is None:
            raise gr.Error("Upload a photo first")
        x = to_tensor(photo).to(device)[None]
        net_photo = resize(x)
        gallery, cubes = [], []
        tmp = Path(tempfile.mkdtemp())
        for k, ref in enumerate(refs):
            if ref is None:
                continue
            lut = model(net_photo, resize(to_tensor(ref).to(device)[None]))[0]
            lut = ident + strength * (lut - ident)
            gallery.append((to_image(apply_lut(x, lut)[0]), f"Option {k + 1}"))
            cube = tmp / f"option_{k + 1}.cube"
            write_cube(lut, cube, title=f"grademl option {k + 1}")
            cubes.append(str(cube))
        return [(photo[..., :3], "Original")] + gallery, cubes

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"## {title}\nModel `{ckpt.get('cfg', {}).get('name')}` epoch {ckpt.get('epoch')}, "
                    f"val PSNR {ckpt.get('psnr', float('nan')):.2f} dB. Each option takes its look from the "
                    "reference above it: pick a creative direction (🎲 for another variation) or upload "
                    "any image whose colors you like.")
        with gr.Row():
            photo = gr.Image(label="Your photo", type="numpy", height=320)
        refs = []
        with gr.Row():
            for k in range(N_OPTIONS):
                with gr.Column():
                    direction = gr.Dropdown([CUSTOM, *DIRECTIONS], value=DEFAULT_DIRECTIONS[k],
                                            label=f"Option {k + 1}: look source")
                    seed = gr.Number(value=k, precision=0, visible=False)
                    ref = gr.Image(label="Reference", type="numpy", height=200,
                                   value=preset_reference(DEFAULT_DIRECTIONS[k], k))
                    dice = gr.Button("🎲 another variation", size="sm")
                    direction.change(preset_reference, [direction, seed], ref)
                    dice.click(lambda s: s + 100, seed, seed).then(preset_reference, [direction, seed], ref)
                    ref.upload(lambda: CUSTOM, None, direction)
                    refs.append(ref)
        with gr.Row():
            strength = gr.Slider(0, 1.5, value=1.0, step=0.05, label="Intensity")
            btn = gr.Button("Grade my photo", variant="primary")
        gallery = gr.Gallery(label="Results", columns=N_OPTIONS + 1, height=420, object_fit="contain")
        cubes = gr.Files(label="LUTs (.cube) for each option")
        btn.click(generate, [photo, strength, *refs], [gallery, cubes])
        strength.release(generate, [photo, strength, *refs], [gallery, cubes])
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
