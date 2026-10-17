# Grade Studio ML — integration brief

This kit brings the on-device ML color grading into the Grade Studio iOS app.
It replaces the filter-based color grading (one LUT per style for every photo)
with **GradeNet-L**: per-photo correction + grading in 15 styles, done region
by region, plus the editor's **top 3** and **content-aware reference matching**.

Everything here was produced and validated in Python (PyTorch). **Nothing has
run on Apple hardware yet**: the first job of the integration is to prove the
iOS port matches the golden outputs in `golden/`.

## Kit contents

```
models/grade_local_lut.mlpackage      GradeNet-L v4 (the AI editor), 13 MB fp16
models/grade_local_lut.styles.json    style one-hot order: "none", then 14 looks
models/match_local_lut.mlpackage      MatchNet-L v2 (content-aware match, residual), 15 MB
models/match_lut.mlpackage            match model v6 (look / match heads; existing feature)
models/*.testvec.npz                  Core ML parity vectors (inputs + PyTorch outputs)
presets/                              fixed-preset LUT bank (existing feature, fallback)
config/editor_calibration.json        per-look calibration for the editor's scoring
swift_reference/                      Swift + Core Image Metal reference code (UNCOMPILED, see below)
python_reference/                     the Python source of truth for everything to port
golden/                               expected outputs on real photos (acceptance tests)
check_coreml_parity.py                Mac script: Core ML vs PyTorch outputs
ios_integration.md                    older integration notes (v1 models, subject-aware, grain)
```

## How GradeNet-L works (what the app must do)

1. Resize the photo to **256x256** (squash, ignore aspect ratio) and run
   `grade_local_lut` with inputs `content` (image) and `style` (one-hot
   `(1, 15)`, order from `styles.json`).
2. Outputs: `luts` `(4, 3, 33, 33, 33)`, layout `lut[c, b, g, r]`, values in
   sRGB-encoded 0..1; `grid` `(1, 4, 8, 16, 16)`: logits over
   (LUT k, luma bin, y, x), y = 0 is the **top** row of the photo.
3. At **full resolution**, for every pixel p (sRGB-encoded values):
   ```
   luma    = 0.2126 R + 0.7152 G + 0.0722 B
   logit_k = trilinear sample of grid[k] at (x, y, luma), align_corners=True
   w       = softmax_k(logit_k)
   out     = sum_k w_k * LUT_k(p)          (blend on sRGB-encoded values)
   ```
   Reference: `python_reference/local.py` (`slice_grid`, `apply_local`;
   `pack_grid` + `slice_packed` is exactly what the Metal kernel does).
4. In the app: 4 `CIColorCubeWithColorSpace` passes (sRGB color space) + one
   Core Image kernel that samples the packed grid and blends
   (`swift_reference/LocalGrade.ci.metal`, `LocalGrade.swift`).
   Core Image works in **linear** light: convert to sRGB before computing
   luma and blending, back to linear after (the reference kernel does).
5. **Intensity slider:** `out = none + t * (style - none)` in image space,
   where `none` is the same pipeline with style `"none"` (correction only).
   Run the model once per style needed; sliders only re-blend images.
6. **Film grain** (optional, existing): after the grade (`python_reference/grain.py`).

## The editor's top 3 (port of `python_reference/editor.py`)

`local_top_versions` / `score_candidates`:

- Grade the 256 px photo in **all 15 styles** (batch the 15 one-hots), render
  each at 256 px; candidates = "none" at 1.0 and each look at intensities
  0.6, 0.85, 1.0 (image blend toward "none").
- Score parts per candidate (see the code for exact formulas):
  `critic` (run GradeNet-L with "none" on the candidate: mean |change| in %,
  negative), `bold`, `fit` (`surprise.direction_scores` on the corrected
  photo), `clip`, `skin` (needs a person mask: Vision
  `VNGeneratePersonSegmentationRequest`), `keep` (chroma kept on colorful photos).
- Calibration: `config/editor_calibration.json` → for `critic` and `bold`,
  `part - mean[look][part] + mean over looks`.
- Top 3 = best candidates from 3 different families (`FAMILIES` in editor.py).
- Cost: 15 + 46 model runs on 256 px thumbnails; run off the main thread.

## Content-aware matching (MatchNet-L)

MatchNet-L is **residual on the classic global color transfer**
(`python_reference/transfer.py`, `transfer_lut` with default arguments),
**not** on `match_lut.mlpackage` (whose v6 match head imitates a different,
palette-based transfer). So:

1. Port `transfer_lut` to Swift (Lab conversion, luminance quantile curve,
   per-band a\*b\* means, 2x2 Monge-Kantorovich map; computed on 64x64
   thumbnails, evaluated on the 33^3 grid). Golden: `golden/match_*/match.npz`
   → `global_lut` for the given `photo_256` / `reference_256`.
2. `base = transfer_lut applied to the photo`; run `match_local_lut` with
   `content` = base at 256 px, `reference` = reference at 256 px.
3. Render like GradeNet-L, on top of the globally transferred full-res photo.

## Order of work and acceptance tests

1. **Parity (Mac):** `python check_coreml_parity.py --model models/<name>.mlpackage`
   for the three models. Expect max |diff| ~1e-2 (fp16), on CPU, GPU and ANE.
2. **LUT + grid render:** load `golden/<photo>/grade_<style>.npz` (skip the
   model), render `input.png` in Swift, compare with `grade_<style>_1.png`:
   **PSNR > 40 dB**. If the sky treatment lands on the ground, the grid row
   order is flipped; if colors swap, the LUT channel order is wrong.
   `packed_rgba` in the same file is the expected packed grid texture.
3. **Full pipeline:** model → render on the same photos; compare with the
   golden PNGs (PSNR > 35 dB; fp16 + resampling differences).
4. **Intensity:** `grade_<style>_0.5.png`.
5. **Editor:** `golden/<photo>/top3.json` — same 3 styles in the same order
   (scores within ~0.1; no person mask was used for these).
6. **Matching:** port `transfer_lut`, check `global_lut` (max |diff| < 2e-3),
   then the full MatchNet-L render vs `matchnet_l.png`.
7. **Speed on device:** target < 50 ms per model run, full-res render
   interactive on a 12-48 MP photo.

## Notes

- `swift_reference/` was written without a Swift toolchain: treat it as a
  detailed sketch, compile and fix it. `GradeML.swift` holds the existing
  v1 classes (ColorCube, Grader, PresetBank, SubjectAware, Grain).
- Models are trained on: Unsplash Lite (dataset terms: check the
  commercial-use question, section 2.A) and Open Images people photos
  (CC BY 2.0). Do not ship any training data; only the models.
- Source repo (code, training, results): https://github.com/LivDevGit/GradeStudio

## Prompt for Claude Code (in the app project)

> Integrate the ML color grading from the kit in `<path to kit>` into Grade
> Studio. Read `INTEGRATION.md` first and follow its order of work: verify
> Core ML parity, then build the GradeNet-L render (4 CIColorCube passes +
> the bilateral-grid Core Image kernel) and prove it against the golden
> outputs before wiring any UI. Then the intensity slider, the editor's top 3
> (port `editor.py` with the calibration file), and finally MatchNet-L with a
> Swift port of `transfer_lut`. Treat `python_reference/` as the source of
> truth and `swift_reference/` as an uncompiled sketch. Report each
> acceptance test's numbers as you go.
