"""Gradio test UI.

match (stage 2):    upload a photo, get 3 graded options side by side. Each
                    option takes its look from a reference image: either one
                    you upload (a film still, a photo you like) or a preset
                    creative direction (teal & orange, warm film, ...).
                    "Subject-aware" grades the person and the background
                    differently (grademl.compose), with a person mask.
                    Styles run through GradeNet (grademl.grade, --grader):
                    each photo is corrected and graded in the direction by
                    the model; "Fixed recipes" shows the plain preset for
                    comparison. "Best versions" lets GradeNet grade the photo
                    in every style and shows the editor's top 3 choices. Film grain is free
                    (amount / size) on any option.
adaptive (stage 1): upload a photo -> auto-enhanced result.

The LUT is predicted from 256px copies and applied to the full-resolution
photo, the same split as on device. Intensity blends the predicted LUT with
the identity LUT (what a "strength" slider in the app would do).

Usage:
  python scripts/app.py --ckpt runs/match_v6/best.pt      # GradeNet from runs/grade_v1 if present
  python scripts/app.py --ckpt runs/match_v6/best.pt --grader runs/other/best.pt
  python scripts/app.py --ckpt runs/adaptive_fivek/best.pt
Then open http://localhost:7860 in a Windows browser.
"""
import argparse
import hashlib
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F

from grademl.compose import SubjectAware, compose_luts, scale_lut, split_luts, subject_aware_grade
from grademl.data import data_root, load_rgb, resize
from grademl.grade import grade_luts, top_versions
from grademl.grain import Grain, add_grain
from grademl.lut import apply_lut, identity_lut, write_cube
from grademl.mask import coverage, person_mask
from grademl.model import AdaptiveLUTNet, GradeNet, MatchLUTNet
from grademl.pro_looks import DIRECTIONS, random_pro_look
from grademl.surprise import surprise
from grademl.synthetic import list_images

REPO = Path(__file__).resolve().parents[1]
N_OPTIONS = 3
DEFAULT_DIRECTIONS = ["golden_hour", "teal_orange", "portra"]
CUSTOM = "custom (upload a reference)"
ML_ENGINE = "ML: GradeNet edits this photo"
FIXED_ENGINE = "Fixed recipes (for comparison)"


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img[..., :3])).permute(2, 0, 1).float() / 255


def to_image(t: torch.Tensor) -> np.ndarray:
    return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def load_model(path: Path, device):
    ckpt = torch.load(path, map_location="cpu")
    arch = ckpt.get("arch", "adaptive")
    if arch == "grade":
        model = GradeNet(len(ckpt["styles"]), **ckpt["cfg"]["model"])
    else:
        model = {"match": MatchLUTNet}.get(arch, AdaptiveLUTNet)(**ckpt["cfg"]["model"])
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


def build_match_ui(model, ckpt, device, title, grader=None):
    ident = identity_lut(model.luts.basis.shape[-1], device=device)
    size = model.luts.basis.shape[-1]
    canvas = preset_canvas()
    two_heads = model.match_head is not None
    masks: dict[str, torch.Tensor] = {}  # photo hash -> person mask, segment once per photo

    def mask_for(photo: np.ndarray, x: torch.Tensor) -> torch.Tensor:
        key = hashlib.sha1(np.ascontiguousarray(photo)).hexdigest()
        if key not in masks:
            masks.clear()
            masks[key] = person_mask(x[0])
        return masks[key]

    def preset_lut(direction: str, seed: float, boost: float) -> torch.Tensor:
        return random_pro_look(torch.Generator().manual_seed(int(seed)), size=size,
                               direction=direction, strength=(float(boost), float(boost)))

    def preset_reference(direction: str, seed: float, boost: float = 1.0):
        if direction == CUSTOM:
            return gr.update()
        return to_image(apply_lut(canvas[None], preset_lut(direction, seed, boost))[0])

    @torch.no_grad()
    def generate(photo, strength, engine, grain_amount, grain_size, split_on, subj_s, bg_s, bg_pop, *per_option):
        if photo is None:
            raise gr.Error("Upload a photo first")
        n = N_OPTIONS
        refs, matches, directions, seeds, boosts = (per_option[i * n:(i + 1) * n] for i in range(5))
        x = to_tensor(photo).to(device)[None]
        net_photo = resize(x)
        cfg = SubjectAware(subj_s, bg_s, bg_pop)
        mask = mask_for(photo, x) if split_on else None
        if mask is None:
            mask_view = None
        elif coverage(mask) < cfg.min_coverage:
            mask_view = gr.update(value=to_image(mask.expand(3, -1, -1)),
                                  label="No person found: options use the plain look")
        else:
            mask_view = gr.update(value=to_image(mask.expand(3, -1, -1)),
                                  label=f"Person mask ({coverage(mask):.0%} of the frame)")
        use_ml = grader is not None and engine == ML_ENGINE
        presets = [d for d in directions if d != CUSTOM]
        if use_ml and presets:
            correction, ml_luts = grade_luts(grader, net_photo, presets)
        gallery, cubes = [], []
        tmp = Path(tempfile.mkdtemp())
        for k, (ref, m, d, sd, b) in enumerate(zip(refs, matches, directions, seeds, boosts)):
            base, t = None, strength
            if d != CUSTOM and use_ml:
                # GradeNet edits this photo in the direction: correction +
                # grade, adapted to its content. Intensity scales the grade
                # on top of the correction.
                lut, base, t = ml_luts[presets.index(d)], correction, strength * float(b)
                label = f"Option {k + 1}: {d} (ML, this photo)"
            elif d != CUSTOM:
                # Fixed recipe, the same mapping for every photo (for comparison).
                lut = preset_lut(d, sd, b).to(device)
                label = f"Option {k + 1}: {d} (fixed recipe)"
            elif ref is None:
                continue
            else:
                pred = model(net_photo, resize(to_tensor(ref).to(device)[None]))
                if two_heads:  # blend "grade only" -> "match colors too"
                    lut = pred[0][0] + m * (pred[1][0] - pred[0][0])
                else:
                    lut = pred[0]
                label = f"Option {k + 1}: your reference"
            out = add_grain(subject_aware_grade(x, lut, mask, cfg, t, base=base),
                            Grain(grain_amount, grain_size), seed=k)
            gallery.append((to_image(out[0]), label))
            if mask is not None and coverage(mask) >= cfg.min_coverage:
                subj, bg, pop = split_luts(lut, cfg, t, base=base)
                for part, part_lut in [("subject", subj), ("background", compose_luts(bg, pop))]:
                    cube = tmp / f"option_{k + 1}_{part}.cube"
                    write_cube(part_lut, cube, title=f"grademl option {k + 1} {part}")
                    cubes.append(str(cube))
            else:
                cube = tmp / f"option_{k + 1}.cube"
                write_cube(scale_lut(lut, t, base), cube, title=f"grademl option {k + 1}")
                cubes.append(str(cube))
        return [(photo[..., :3], "Original")] + gallery, cubes, mask_view

    @torch.no_grad()
    def do_best_version(photo, grain_amount, grain_size):
        """GradeNet edits the photo in every direction; the editor's top 3 choices (grademl.grade)."""
        if photo is None:
            raise gr.Error("Upload a photo first")
        if grader is None:
            raise gr.Error("No GradeNet loaded: start the app with --grader runs/grade_v1/best.pt")
        x = to_tensor(photo).to(device)[None]
        mask = mask_for(photo, x)
        edits = top_versions(x[0], grader, mask, n=3)
        tmp = Path(tempfile.mkdtemp())
        gallery, notes, cubes = [(photo[..., :3], "Original")], [], []
        for e in edits:
            if e.subject is not None:
                out = subject_aware_grade(x, e.look, mask, e.subject, e.best.intensity, base=e.base)
                subj, bg, pop = split_luts(e.look, e.subject, e.best.intensity, base=e.base)
                parts = [("subject", subj), ("background", compose_luts(bg, pop))]
            else:
                out = apply_lut(x, e.lut)
                parts = [("look", e.lut)]
            out = add_grain(out, Grain(grain_amount, grain_size), seed=e.rank)
            name = e.best.style.replace("_", " ")
            gallery.append((to_image(out[0]), f"#{e.rank} {name} {e.best.intensity:.0%}"))
            notes.append(f"**#{e.rank} {name}**\n" + "\n".join(f"- {r}" for r in e.reasons))
            for part, lut in parts:
                cube = tmp / f"editor_{e.rank}_{e.best.style}_{part}.cube"
                write_cube(lut, cube, title=f"grademl editor #{e.rank} {e.best.style} {part}")
                cubes.append(str(cube))
        if grain_amount > 0:
            notes.append(f"Grain {grain_amount:.3f}, size {grain_size:.1f} on all versions "
                         "(your setting; not in the .cube files, LUTs cannot hold grain)")
        table = [[c.style, round(c.intensity, 2), round(c.score, 2),
                  *(round(c.parts[k], 2) for k in ("critic", "fit", "bold", "clip", "skin"))]
                 for c in edits[0].shortlist[:12]]
        return gallery, "\n\n".join(notes), table, cubes

    def do_surprise(photo, counter):
        """Fill the options with bold looks picked for this photo."""
        if photo is None:
            raise gr.Error("Upload a photo first")
        picks = surprise(to_tensor(photo), n=N_OPTIONS, seed=int(counter))
        dirs = [p.direction for p in picks]
        seeds_ = [p.seed for p in picks]
        boosts_ = [round(p.strength, 3) for p in picks]
        refs_ = [preset_reference(d, s, b) for d, s, b in zip(dirs, seeds_, boosts_)]
        return [*dirs, *seeds_, *boosts_, *refs_, counter + 1]

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"## {title}\nModel `{ckpt.get('cfg', {}).get('name')}` epoch {ckpt.get('epoch')}. "
                    "Each option takes its look from the reference above it. Presets (🎲 for another "
                    "variation) apply a pro-style recipe directly; for an uploaded reference the model "
                    "predicts the LUT. *Match* = 0 copies only the reference's grade (graded film stills); "
                    "1 also maps your photo's palette onto the reference's (any photo you like). "
                    "**✨ Surprise me** picks three bold, different looks suited to your photo.")
        with gr.Row():
            photo = gr.Image(label="Your photo", type="numpy", height=320)
        refs, matches, directions, seeds, boosts = [], [], [], [], []
        with gr.Row():
            for k in range(N_OPTIONS):
                with gr.Column():
                    direction = gr.Dropdown([CUSTOM, *DIRECTIONS], value=DEFAULT_DIRECTIONS[k],
                                            label=f"Option {k + 1}: look source")
                    seed = gr.Number(value=k, precision=0, visible=False)
                    boost = gr.Number(value=1.0, visible=False)
                    ref = gr.Image(label="Reference", type="numpy", height=200,
                                   value=preset_reference(DEFAULT_DIRECTIONS[k], k))
                    dice = gr.Button("🎲 another variation", size="sm")
                    match = gr.Slider(0, 1, value=0.0, step=0.05, label="Match: grade only -> colors too",
                                      visible=two_heads)
                    direction.input(preset_reference, [direction, seed, boost], ref)
                    direction.input(lambda d: 1.0 if d == CUSTOM else 0.0, direction, match)
                    dice.click(lambda s: s + 100, seed, seed).then(preset_reference, [direction, seed, boost], ref)
                    ref.upload(lambda: CUSTOM, None, direction)
                    refs.append(ref)
                    matches.append(match)
                    directions.append(direction)
                    seeds.append(seed)
                    boosts.append(boost)
        with gr.Row():
            strength = gr.Slider(0, 1.5, value=1.0, step=0.05, label="Intensity")
            engine = gr.Radio([ML_ENGINE, FIXED_ENGINE], value=ML_ENGINE if grader is not None else FIXED_ENGINE,
                              label="Styles", interactive=grader is not None)
            surprise_btn = gr.Button("✨ Surprise me")
            best_btn = gr.Button("🎬 Best versions (editor's top 3)")
            btn = gr.Button("Grade my photo", variant="primary")
        with gr.Row():
            grain_amount = gr.Slider(0, 0.12, value=0.0, step=0.005, label="Film grain: amount (any style)")
            grain_size = gr.Slider(0.5, 3, value=1.2, step=0.1, label="Film grain: size")
        with gr.Accordion("🎬 Best versions: GradeNet edits the photo in every style, "
                          "the editor ranks its top 3", open=True):
            with gr.Row():
                best_out = gr.Gallery(label="Original and the editor's choices #1-#3", columns=4, height=420,
                                      object_fit="contain")
                with gr.Column():
                    best_why = gr.Markdown()
                    best_cubes = gr.Files(label="Editor's LUTs (.cube)")
            best_table = gr.Dataframe(headers=["style", "intensity", "score", "critic", "fit",
                                               "bold", "clip", "skin"], label="Candidates (best first)",
                                      interactive=False)
        with gr.Accordion("Subject-aware: grade the person and the background differently", open=False):
            with gr.Row():
                split_on = gr.Checkbox(value=False, label="On")
                subj_s = gr.Slider(0, 1.5, value=SubjectAware.subject_strength, step=0.05,
                                   label="Look on the person")
                bg_s = gr.Slider(0, 1.5, value=SubjectAware.background_strength, step=0.05,
                                 label="Look on the background")
                bg_pop = gr.Slider(0, 1, value=SubjectAware.background_pop, step=0.05,
                                   label="Background pop (darker, calmer)")
            mask_view = gr.Image(label="Person mask", type="numpy", height=200, interactive=False)
        counter = gr.Number(value=0, precision=0, visible=False)
        gallery = gr.Gallery(label="Results", columns=N_OPTIONS + 1, height=420, object_fit="contain")
        cubes = gr.Files(label="LUTs (.cube) for each option (subject + background when subject-aware)")
        inputs = [photo, strength, engine, grain_amount, grain_size, split_on, subj_s, bg_s, bg_pop,
                  *refs, *matches, *directions, *seeds, *boosts]
        outputs = [gallery, cubes, mask_view]
        btn.click(generate, inputs, outputs)
        strength.release(generate, inputs, outputs)
        for c in [*matches, subj_s, bg_s, bg_pop]:
            c.release(generate, inputs, outputs)
        split_on.input(generate, inputs, outputs)
        engine.input(generate, inputs, outputs)
        for c in (grain_amount, grain_size):
            c.release(generate, inputs, outputs)
        best_btn.click(do_best_version, [photo, grain_amount, grain_size],
                       [best_out, best_why, best_table, best_cubes])
        surprise_btn.click(do_surprise, [photo, counter], [*directions, *seeds, *boosts, *refs, counter]) \
            .then(generate, inputs, outputs)
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
    ap.add_argument("--grader", type=Path, default=REPO / "runs" / "grade_v1" / "best.pt",
                    help="GradeNet checkpoint: ML styles + Best version (skipped if missing)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, arch, ckpt = load_model(args.ckpt, device)
    if arch == "match":
        grader = load_model(args.grader, device)[0] if args.grader.exists() else None
        if grader is None:
            print(f"no GradeNet checkpoint at {args.grader}: fixed recipes only, Best version disabled")
        demo = build_match_ui(model, ckpt, device, "Match the Look", grader)
    else:
        demo = build_adaptive_ui(model, ckpt, device, "Adaptive LUT auto-enhance")
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
