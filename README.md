# Match the Look — reference-based color grading

Give the model a **reference image** (a film still, a photo you like, or a preset creative direction) and **your photo**. It predicts a **3D LUT** that carries the reference's look over to your photo.

The prototype is built to run on-device. A small network predicts the LUT from downscaled (256px) inputs, and the LUT is then applied to the full-resolution image in Metal / Core Image. The goal is to ship it in [Grade Studio](#) once the results are good enough.

Starting point: *Learning Image-adaptive 3D Lookup Tables for High Performance Photo Enhancement in Real-time* (Zeng et al., TPAMI 2020).

## How it works

```
content 256px ─┐   CNN encoder + color statistics      shared MLP ─┬─ look head  ─┐
               ├─► (histograms, per-tone chroma,     ──►           │              ├─► weights × 64 basis LUTs
reference 256px┘    8³ joint RGB histogram)                        └─ match head ─┘     + identity = 33³ LUT(s)
                                                                                         │
full-res photo ──────────────────────────────────────────────── trilinear LUT lookup ◄──┘  (Metal / CIColorCube)
```

- **Look head**: copies only the *grade* applied to the reference (white balance, curves, split toning, saturation...). Trained on synthetic pairs where the exact LUT is known: `reference = photo B + look`, `target = photo A + same look`.
- **Match head**: additionally pulls the photo's colors toward the reference's, for ordinary (ungraded) photos as references. Trained to reproduce a damped classic color transfer (`grademl.transfer`: L* histogram matching with slope limits, per-tone-band a\*b\* shift, Monge-Kantorovich chroma map), baked into a LUT.
- The app blends the two LUTs with a "match" slider and scales the result with an intensity slider. The model never touches full-resolution pixels.

## Results

Held-out Unsplash photos, 1000 synthetic look pairs, reference from a *different* scene (PSNR of the graded photo vs. the exact target):

| Model | PSNR | ΔE76 | Notes |
|---|---|---|---|
| no-op (ungraded photo) | 22.99 dB | – | |
| v1: 24 LUTs, channel histograms | 26.06 dB | 9.31 | |
| v2: 64 LUTs, + 8³ joint RGB histogram | 26.75 dB | 8.58 | same-scene reference: 29.56 dB (v1: 26.89) |
| v3: v2 + match head (warm start, 20 epochs) | 26.41 dB | – | look head gives up 0.3 dB to the shared trunk |
| v4: v3 + fixed CNN encoder (warm start) | 26.59 dB | – | |
| **v5: v4 architecture trained from scratch** (40 epochs) | **26.91 dB** | – | current model |

v1–v3 were trained with a CNN encoder whose output was constant (see *Lessons learned*); all their adaptivity came from the color-statistics branch. v4 fixed the encoder on top of v3; v5 trains the fixed architecture from scratch and is best on both heads.

Match head vs. the classic-transfer teacher it imitates: **36.27 dB** (v5; v4: 35.86, v3: 34.97) with natural photos as references, 36.09 dB with graded ones (no-op: 16.9 dB). The look head alone stays far from the teacher (17.3 dB), so the two heads really do different things and the "match" slider spans a meaningful range.

`scripts/match_sheet.py` renders visual reports (`exports/match_synthetic.jpg`, `exports/match_real.jpg`).

## Pipeline

| Stage | What | Status |
|---|---|---|
| Environment | WSL2 + PyTorch 2.7 (cu128) on an RTX 5070, Python 3.12 | done |
| Core ML | Export to ML Program (iOS 17, fp16), only standard MIL ops (conv, instance_norm, linear, exp, reduce_mean, ...); ~19 MB | done (Mac parity check pending) |
| Data | Unsplash Lite (25k photos, commercial-OK for training) + 8 synthetic creative directions; FiveK for stage 1 | done |
| Stage 1 | Adaptive auto-enhance on FiveK (validation of the technique): 23.67 dB test PSNR vs 17.80 no-op (was 20.9 with the broken encoder; the paper reports 25.2 on its own FiveK variant) | done |
| Stage 2 | Reference-conditioned LUT prediction: look + match heads (v5) | done (first version) |
| Demo | Gradio UI: 3 side-by-side options, presets or custom reference, .cube export | done (local) |
| On-device | Core ML model + Metal LUT application in Grade Studio | todo |

## Lessons learned

- **Dead-gradient init.** A first version initialized both the basis LUTs (except identity) and the weight head to zero, so `grad(basis) ∝ weight` and `grad(weight) ∝ basis` stayed zero forever and the model collapsed to one global LUT. Fixed by a residual bank (`identity + Σ wᵢ·basisᵢ`, zero bases, random head).
- **An encoder that ignored its input.** Blocks were `conv -> LeakyReLU -> InstanceNorm`, followed by a global mean. The spatial mean of an InstanceNorm output is exactly its bias, so the encoder returned the same vector for every image. Stage 1 could therefore only learn one global LUT (stuck at ~20.7 dB on FiveK), and the match models leaned entirely on the color statistics. Found by checking that predicted weights had zero variance across images; fixed with `conv -> norm -> activation`, plus a regression test.
- **Instance norm erases color casts.** Even a working CNN encoder normalizes away per-channel means, i.e. the color casts that define a look; explicit color statistics fixed that, and a joint RGB histogram gave the biggest jump when reference and photo share a scene.
- **Test the Core ML export first.** Shape arithmetic in `reshape` (`h*w`) traces to an `int` op coremltools can't convert; `-1` works.
- **CPU-bound data loading.** Rendering LUTs at 17³ in workers and grading on the GPU gave a ~4x speed-up.

## Layout

```
src/grademl/   model, LUT ops, synthetic looks, color transfer, datasets, metrics
scripts/       setup, data download/preprocessing, training, export, Gradio app, reports
configs/       training configs (YAML)
tests/         unit tests (LUT ops, looks, transfer, model)
data/          local datasets (git-ignored)
exports/       Core ML artifacts and reports (git-ignored)
```

## Setup (WSL2 / Ubuntu)

```bash
bash scripts/setup_wsl.sh                # venv (Python 3.12), PyTorch cu128, deps, GPU check
source ~/.venvs/grademl/bin/activate
python scripts/download_unsplash.py      # 25k photos -> $GRADEML_DATA/unsplash/images
python scripts/resize_images.py unsplash/images unsplash/images_384 --size 384
python scripts/train_match.py --config configs/match_v5.yaml
python scripts/app.py --ckpt runs/match_v5/best.pt    # http://localhost:7860
python scripts/export_coreml.py --model match --ckpt runs/match_v5/best.pt
```

Datasets live at `$GRADEML_DATA` (default `~/grademl-data`, on the WSL filesystem for I/O speed).

## Data & licensing

- **Unsplash Lite** license allows training models for internal business use; check the terms (`$GRADEML_DATA/unsplash/TERMS.md`) before shipping.
- **MIT-Adobe FiveK** is research-only and is used only to validate the technique (stage 1), never for the shipped model.
- The looks themselves are synthetic (`grademl.synthetic`), so their exact LUTs are known.
