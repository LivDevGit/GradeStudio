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

### v6: pro-style looks and palette transfer

Feedback from testing on real photos: the grades were too timid to read as a creative, photographer-level look. Two changes:

- **Pro looks** (`grademl.pro_looks`): a colorist-style pipeline with hue-selective HSL (8 sectors), hue-qualified color wheels (e.g. warm highlights that stay off the sky, so warm/cool separation survives), filmic tone and 14 recipes (golden hour, teal & orange, Portra, Fuji Chrome, Cinestill 800T, bleach bypass, autumn, ...). Preview: `scripts/preview_pro_looks.py`.
- **Palette transfer** (`grademl.transfer.palette_transfer_lut`): k-means palettes of photo and reference, each photo cluster pulled toward its matched reference cluster, spread over the LUT with Gaussian kernels. The photo's sky takes the reference's blue, its warm surfaces take the reference's warm tones, instead of one global shift.

v6 (from scratch, 40 epochs): look head 28.16 dB on held-out pro looks (v5 on the same looks: 23.34 dB); match head 32.75 dB vs the palette teacher.

**Design decision:** presets are known recipes, so the app applies their LUTs directly. Routing a preset through the model (preset reference image -> predicted LUT) visibly weakened it: an L1-trained regressor hedges toward milder looks when unsure. The model is used where it is needed, for user-supplied references.

### Surprise me

`grademl.surprise` picks three bold looks for a photo: it profiles the image (sky, greenery, warm surfaces, darkness, colorfulness), scores the 14 directions, samples three from different families (warm / film / stylized / cool / soft / mono) at 1.0-1.3x preset intensity, and rejects variations that blow highlights or crush blacks. Rule-based for now; the interface allows swapping in a model that learns which options users keep.

### GradeNet: the AI edits each photo, "Best versions"

A preset applies the same color mapping to every photo. **GradeNet**
(`grademl.grade`) looks at the photo and predicts the LUT for *it*: one
network, the photo plus a style (`none` = correction only, or one of the 14
directions) in, a 33^3 LUT out.

Training (Unsplash only, so it can ship): the input is a finished photo
with random technical faults (exposure, white balance, contrast,
saturation, a flat camera profile); the target is the finished photo
graded in the requested direction, toned down where the full direction
would clip this photo. One pass therefore corrects the photo and grades it,
adapted to its content. What a direction *means* comes from the colorist
recipes (`grademl.pro_looks`); how to apply it to a given photo is learned.

Held-out photos with faults, 40 epochs (`configs/grade_v1.yaml`):

| | PSNR vs the finished, graded target |
|---|---|
| faulty photo as is | 23.16 dB |
| fixed preset on the faulty photo | 27.02 dB |
| **GradeNet, styled** | **28.31 dB** |
| GradeNet, correction only | 27.70 dB |

**🎬 Best versions** (`top_versions`): GradeNet grades the photo in all 14
directions at three intensities, then judges every result itself: with
style `none` it predicts how much a graded photo still needs correcting
(less = reads as finished). Plus a bonus for a visible grade, penalties for
clipping and for skin pushed off its natural color (person mask), and a
small scene-fit prior. The editor shows its top 3 choices, ranked (one
per style, each at its best intensity), each with subject-aware grading
when there is a person and the reasons for the choice.

**Film grain** (`grademl.grain`) is free on any grade (amount, size):
LUTs cannot hold spatial noise, so it is a procedural step after the LUT,
soft-light blended so it lives in the midtones like film grain.

### v7: local grading — GradeNet-L, people & skin, content-aware matching

Everything up to v6 applies **one color mapping to the whole photo**. It
cannot recover a blown sky without darkening the subject, protect skin
while the rest takes a strong look, or separate foreground from distance.
Colorists do that with masks; v7 learns it.

**GradeNet-L** (`grademl.local`): the network predicts K = 4 LUTs and a
bilateral grid (16 x 16 x 8 luma bins) of blend weights. Every pixel takes
a softmax mix of the 4 LUTs, read from the grid at its position *and its
own luminance* (HDRNet-style slicing), so the blend follows the photo's
edges and no segmentation runs on device. Applying it at full resolution
is 4 LUT passes plus one tiny kernel (`grademl.local.slice_packed` is the
reference; `ios/LocalGrade.ci.metal` the integration sketch).

**Local targets** (`grademl.local_looks`): the masks a colorist would draw
come from OneFormer (sky, people, plants, water, buildings; MIT) and Depth
Anything V2 (Apache-2.0), precomputed once (`scripts/label_scenes.py`), and
are used only to build training targets:

- sky: density and color in the look's own palette (pre-look, so B&W gets a red-filter sky)
- people: gentler look, light on the subject, calmer background
- skin: natural hue and chroma under every look (halfway back to the photo's skin, then toward the skin line), a little light on faces
- depth: aerial perspective (distant parts softer, in the look's "air" tint)
- vignette, held off the highlights
- input faults to fix: bright sky over dark ground, backlit subject, graduated light fall-off, color casts on the subject only (foliage bounce, open shade, mixed light)

People are ~10% of Unsplash, so v2/v3 oversample them, center half of their
crops on the person, and weight the loss toward people (x4) and skin (x6).

Held-out photos with faults, all models measured on the same v7 targets
(`scripts/eval_local.py`; PSNR in dB, skin hue error in degrees):

| | all photos | sky | people photos | person | skin | skin hue |
|---|---|---|---|---|---|---|
| fixed preset | 25.41 | | 25.31 | | | |
| GradeNet v1 (one LUT) | 25.77 | 22.03 | 25.61 | 26.43 | 22.44 | 9.2° |
| GradeNet-L v1 | **27.80** | 27.07 | 27.23 | 26.51 | 22.98 | 9.8° |
| GradeNet-L v2 (people x5) | 27.50 | 26.92 | **27.92** | **27.87** | **25.35** | **6.3°** |
| **GradeNet-L v3** (people x2.5, shipped) | 27.70 | **27.12** | 27.84 | 27.75 | 25.22 | 6.7° |

The sky gain (+5 dB over one LUT) is the clearest local effect; the people
work cut skin hue error by a third. The editor's top 3 (`grademl.editor`)
runs on GradeNet-L with the same scoring as v1, intensities capped at 100%.

On FiveK (real expert retouches, research only), the same local network
reaches 24.26 dB vs 24.13 dB for an identical network with one LUT and
23.67 dB for the previous model: most of that gain is the stronger
backbone, because FiveK's expert edits are mostly global.

**More and more varied people: Open Images** (v4). Unsplash has ~2.4k photos
with a person. `scripts/download_openimages_people.py` takes Open Images'
validation + test sets (human-verified person boxes, CC BY 2.0, attribution
for every photo in `$GRADEML_DATA/openimages/attribution.csv`): photos where a
person covers >= 10% of the frame, not a depiction, >= 640 px. 30,319
downloaded (resized to 384 px, originals not kept); `scripts/filter_quality.py`
keeps 22,699 (drops black and white and the photos the v3 critic finds far
from finished, beyond Unsplash's 75th percentile). v4 = v3 fine-tuned on
Unsplash + these, Open Images at sampling weight 0.4 (39% of samples have a
person).

Skin metrics are also split by skin tone (ITA angle of the photo's skin:
light > 41 deg, medium 10-41, dark < 10). On photos ITA measures *apparent*
tone, lighting included, so "dark" also holds lighter skin in dim light.

| Open Images held-out people (431 photos) | person | skin | skin hue | skin light | skin medium | skin dark |
|---|---|---|---|---|---|---|
| GradeNet v1 (one LUT) | 25.17 | 22.85 | 10.1° | 20.77 / 12.4° | 21.94 / 9.6° | 23.52 / 9.9° |
| GradeNet-L v3 | 26.42 | 25.09 | 7.3° | 23.23 / 8.6° | 23.87 / 7.4° | 25.85 / 7.0° |
| **GradeNet-L v4** | **27.75** | **26.71** | **5.9°** | **25.44 / 7.9°** | **25.88 / 6.0°** | **27.23 / 5.6°** |

On Unsplash people v4 also improves (skin 25.22 -> 25.81 dB, hue 6.7° ->
6.0°); on all Unsplash photos it gives up 0.1 dB (27.70 -> 27.60; sky 27.12
-> 26.97). v4 is the shipped GradeNet-L; the editor is recalibrated for it
(B&W / bleach bypass as #1: 2%, all 15 looks can win).

**Editor ranking fix** (`grademl.editor.score_candidates`, `scripts/editor_report.py`).
The top-3 editor leaned on a few dramatic looks: on 250 held-out photos
five looks took 91% of the #1 picks, B&W / bleach bypass 23%, and five looks
never won. The cause was per-look, not per-photo: golden hour always changes
a lot (a fixed "bold" bonus) and B&W leaves the critic nothing to correct (a
fixed critic advantage). Now (1) critic and bold are calibrated per look
against their mean over 250 *other* held-out photos
(`configs/editor_calibration.json`), (2) a "keep" penalty stops looks from
draining a colorful photo (B&W stays an option for muted scenes), and (3)
the top 3 come from three different look families.

| 250 held-out photos | before | after |
|---|---|---|
| B&W / bleach bypass as #1 | 23% | 4% |
| looks that ever win #1 | 12 of 15 | 15 of 15 |
| "natural" offered in the top 3 | 0% | 45% |

Re-run `python scripts/editor_report.py --calibrate --offset 250 --n 250`
after retraining GradeNet-L (the calibration belongs to a checkpoint).


**Content-aware matching** (`grademl.semantic_match`): DINOv2 (ViT-S/14,
Apache-2.0) patch features find, for each region of the photo, the same
*kind of thing* in the reference (sky to sky, skin to skin, foliage to
foliage); color transfers per region, tone stays mostly global (per-region
tone curves left seams). Global and palette transfer turn skin gray or
muddy a purple sky when the reference's colors sit elsewhere; the semantic
match does not (`scripts/semantic_sheet.py` -> `exports/semantic_match.jpg`).

**MatchNet-L** (`grademl.local.MatchLocalNet`) distills that matcher into a
7.5M-parameter on-device network: both images through the same conv stem,
then every photo cell *attends to* the reference cells (cross-attention on
features + cell colors) and the result drives the LUTs and the grid. It is
residual on the global transfer (which the existing match model already
reproduces), so it only has to learn the region-by-region part. Trained on
49k pairs (`scripts/make_match_pairs.py`: mostly content neighbours, the
reference through a random pro look), teacher precomputed:

| vs the DINOv2 teacher (1000 held-out pairs) | PSNR |
|---|---|
| no-op | 17.96 dB |
| global transfer | 33.28 dB |
| MatchNet-L v1 (non-residual) | 31.66 dB |
| **MatchNet-L v2 (residual)** | **34.71 dB** |

Visually MatchNet-L stays close to the global transfer
(`exports/match_local.jpg`): it learns part of the teacher's per-region
changes, which are themselves subtle. The DINOv2 matcher is the stronger
tool; MatchNet-L is its first on-device approximation. On device its
`content` input is the photo *after* the global transfer (the existing
match model's match LUT), see `scripts/train_match_local.py`.

Lessons from v7:
- **A student that must relearn the easy part plateaus below it.** Non-residual MatchNet-L stalled under the global transfer it was supposed to beat; making it residual put it above in 2 epochs.
- **Oversampling trades.** People x5 bought skin quality but cost 0.3 dB on everything else; x2.5 after it kept most of both.
- **Per-region tone curves seam.** In smooth dark areas region boundaries became visible bands; tone mostly global, color per region fixed it.
- Infrastructure: one GPU process with DataLoader workers labels 25k photos at ~18/s; parallel model copies overflowed 12 GB VRAM into shared memory and ran at <1/s.

### Subject-aware grading

The look can be graded differently on the person and on the background
(`grademl.compose`): gentler on the subject so skin stays natural, full
strength plus a "background pop" (a bit darker and calmer) behind, blended
through a soft person mask. It works with every look source: presets,
Surprise me and model predictions.

The mask is the only platform-specific part (`grademl.mask`):

| | Segmentation | Edges |
|---|---|---|
| Python (local) | torchvision DeepLabV3-ResNet101, COCO "person" (weights download once into the torch cache, ~240 MB; `--seg-model mobilenet` for a 42 MB, faster, blobbier one) | guided filter at 1024 px |
| iOS (Grade Studio) | Vision `VNGeneratePersonSegmentationRequest`, built into the OS | `CIGuidedFilter` |

With no person in the frame the photo gets the plain look. Try it without a checkpoint:

```
python scripts/subject_grade.py photo.jpg --look portra --cube
```

or in the Gradio app (accordion "Subject-aware"). A selfie segmenter
(MediaPipe) was considered and dropped: it is trained on faces close to the
camera and misses groups and people further away, and its package pulls an
OpenCV build that conflicts with the one this project uses.

### App integration

`scripts/export_presets.py` exports the presets as a LUT bank (84 x float16 33^3, 18 MB, plus the subject-aware background LUT and defaults); `ios/GradeML.swift` has the Core ML runner, LUT blending and CIColorCube application, the preset loader, a Swift port of Surprise me and subject-aware grading (Vision mask + CIBlendWithMask); `docs/ios_integration.md` lists the Mac/iPhone verification steps (`scripts/check_coreml_parity.py`).

`ios/GradeML.swift` also has `Grader` (GradeNet in Core ML) and `Editor.topVersions`, the Swift port of Best versions, plus `Grain`.

`scripts/match_sheet.py` renders visual reports (`exports/match_synthetic.jpg`, `exports/match_real.jpg`).

## Pipeline

| Stage | What | Status |
|---|---|---|
| Environment | WSL2 + PyTorch 2.7 (cu128) on an RTX 5070, Python 3.12 | done |
| Core ML | Export to ML Program (iOS 17, fp16), only standard MIL ops (conv, instance_norm, linear, exp, reduce_mean, ...); ~19 MB | done (Mac parity check pending) |
| Data | Unsplash Lite (25k photos, commercial-OK for training) + 8 synthetic creative directions; FiveK for stage 1 | done |
| Stage 1 | Adaptive auto-enhance on FiveK (validation of the technique): 23.67 dB test PSNR vs 17.80 no-op (was 20.9 with the broken encoder; the paper reports 25.2 on its own FiveK variant) | done |
| Stage 2 | Reference-conditioned LUT prediction: look + match heads (v6, pro looks + palette teacher) | done (second version) |
| Demo | Gradio UI: 3 side-by-side options, presets or custom reference, .cube export | done (local) |
| Subject-aware | Person mask (DeepLabV3 locally, Vision on iOS) + split grading of subject and background | done in Python; Swift written, untested on device |
| GradeNet | Per-photo correction + grading in 14 directions, Best version editor, free grain | done in Python; Core ML export done; Swift written, untested on device |
| v7 local | GradeNet-L v4 (local edits, people & skin, + Open Images people), semantic matching, MatchNet-L; Core ML exports | done in Python; app integration pending |
| On-device | Core ML model + CIColorCube LUT application, preset bank, Surprise me (Swift written, untested on device) | in progress |

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
python scripts/train_match.py --config configs/match_v6.yaml
python scripts/app.py --ckpt runs/match_v6/best.pt    # http://localhost:7860
python scripts/export_coreml.py --model match --ckpt runs/match_v6/best.pt
```

Datasets live at `$GRADEML_DATA` (default `~/grademl-data`, on the WSL filesystem for I/O speed).

## Data & licensing

- **Unsplash Lite** license allows training models for internal business use; check the terms (`$GRADEML_DATA/unsplash/TERMS.md`) before shipping.
- **Open Images** people photos (v4): CC BY 2.0, attribution per photo in `openimages/attribution.csv`; training data only, never shipped.
- **Label models** (training targets only, not shipped): OneFormer Swin-T ADE20K (MIT), Depth Anything V2 Small (Apache-2.0), DINOv2 ViT-S/14 (Apache-2.0).
- **MIT-Adobe FiveK** is research-only and is used only to validate the technique (stage 1), never for the shipped model.
- The looks themselves are synthetic (`grademl.synthetic`), so their exact LUTs are known.
