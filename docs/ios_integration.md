# iOS integration (Grade Studio)

Status: the model, presets and Swift glue exist; nothing has run on Apple
hardware yet. Steps 1-3 below are the checklist before shipping.

## What ships

| Asset | Made by | Size |
|---|---|---|
| `match_lut.mlpackage` (Xcode compiles it to `.mlmodelc`) | `scripts/export_coreml.py --ckpt runs/match_v6/best.pt` | ~19 MB fp16 |
| `grade_lut.mlpackage` + `grade_lut.styles.json` (GradeNet) | `scripts/export_coreml.py --model grade --ckpt runs/grade_v1/best.pt` | ~13 MB fp16 |
| `presets/` (14 directions x 6 variations, `background_pop.lut16`, `manifest.json`) | `scripts/export_presets.py` | ~18 MB |
| `ios/GradeML.swift` | hand-written | – |

Model I/O: inputs `content`, `reference` (256x256 RGB images; Core ML scales
0..255 to 0..1). Outputs `lut_look`, `lut_match`: `(1, 3, 33, 33, 33)`,
layout `lut[c, b, g, r]`, values in sRGB-encoded 0..1.

GradeNet I/O: inputs `content` (256x256 RGB) and `style` (one-hot
`(1, 15)` in the order of `grade_lut.styles.json`: `"none"` = correction
only, then the 14 directions). Output `lut`, same layout. The LUT corrects
*and* grades this photo; it is computed per photo, not a fixed preset.

## Flow in the app

```
Style (ML, per photo): grader.grade(photo, styles: [style])         -> (correction, looks)
                       looks[style]!.scaled(intensity, from: correction)
Best versions:         Editor.topVersions(photo:, grader:, mask:, context:) -> [Edit] (#1-#3)
                       edits[k].render(fullResCIImage, mask: mask, pop: bank.backgroundPop())
Custom reference:      LookMatcher.predict(photo, reference)       -> (look, match)
                       look.blended(toward: match, amount: matchSlider).intensity(intensitySlider)
Fixed presets:         PresetBank.cube(...) / Surprise.pick(...)    (fallback, comparison)
Then:                  cube.apply(to: fullResCIImage), Grain.apply(to:amount:size:) if wanted
```

GradeNet runs once per photo for all styles (15 predictions, batched);
sliders only re-blend LUTs. Best versions adds ~45 critic predictions on
256 px thumbnails plus CPU scoring; run it off the main thread.

Grain is free on any grade: two sliders (amount 0-0.12, size 0.5-3),
applied after the LUT and the subject-aware blend. It cannot live in a
.cube, so exported LUTs never contain it.

The network runs once per (photo, reference) pair on 256px copies; slider
changes only re-blend two LUTs (cheap) and re-render through CIColorCube.

### Subject-aware grading (person vs background)

```
Once per photo:   mask = SubjectMask.person(in: image, context: ctx,
                                            minCoverage: bank.subjectAware.minCoverage)
                  pop  = bank.backgroundPop()
Every render:     bank.subjectAware.render(image, look: cube, intensity: s, mask: mask, pop: pop)
```

The mask comes from Vision (`VNGeneratePersonSegmentationRequest`, iOS 15+,
built into the OS: nothing extra ships), refined with `CIGuidedFilter` at
1024 px. `nil` (no person, or under 1% of the frame) renders the plain look.
Python does the same with DeepLabV3 (`grademl.mask`); the compositing rule
is shared (`grademl.compose`, defaults in `manifest.json` "subject_aware").
Expose the three settings as sliders: look on the person (0.75), look on
the background (1.0), background pop (0.6).

## Verification checklist (Mac / iPhone)

1. **Parity**: copy `exports/match_lut.mlpackage` and
   `exports/match_lut.testvec.npz` to the Mac, then
   `python scripts/check_coreml_parity.py`. Expect max |diff| around 1e-2 or
   less (fp16) on CPU, GPU and ANE, and note the timings.
2. **Compile `GradeML.swift`** in a test target (it was written without a
   Swift toolchain). Unit test: `ColorCube.identity(size: 33).apply(to:)`
   must return the input image; a preset cube applied in Swift must match
   the same `.lut16` applied in Python (`grademl.lut.apply_lut`) on a test image.
3. **LUT orientation**: apply `lut_look` from the parity test vector in Swift
   and in Python on the same image; a red/blue swap means the channel order
   in `LookMatcher.cube(from:)` is off.
4. Profile on device: Instruments, first prediction (model load) vs.
   steady state; target < 50 ms for the prediction on recent iPhones.
5. **Subject-aware parity**: save the Vision mask of a test photo as a
   grayscale PNG, then run
   `python scripts/subject_grade.py photo.jpg --look exports/presets/portra_0.cube --mask vision_mask.png`
   (presets exported with `--cube`) and compare `*_split.jpg` with the
   Swift render of `portra` variation 0 at intensity 1. Expect differences
   only at mask edges (both blend in linear light, but resample the mask
   differently).
   Also check `CIGuidedFilter` exists on the target iOS version (the code
   falls back to a plain upscale) and time `SubjectMask.person` with
   `.accurate` vs `.balanced`.

6. **GradeNet parity**: export with `--model grade`; the test vector holds
   a `style` one-hot, `check_coreml_parity.py` passes it. Then on device
   compare `Editor.topVersions` with `grademl.grade.top_versions` on the
   same photo: the ranked styles should agree, scores within ~0.1
   (thumbnail resampling differs). Check the mask orientation in
   `Editor.skinIndices` (it flips CIContext rows); a wrong flip shows as
   skin penalties on photos where skin did not change.
7. **Grain**: `Grain.apply` at amount 0.06, size 1.2 next to
   `grademl.grain.add_grain` with the same settings; the noise differs,
   strength and clump size should look the same.

## Keeping Python and Swift in sync

`Editor.topVersions` mirrors `top_versions()` in `src/grademl/grade.py`
(weights, intensities, skin and clipping rules); `Grader` uses the style
order from `grade_lut.styles.json`, written by the export.


`Surprise.scores` mirrors `direction_scores()` in `src/grademl/surprise.py`.
If the Python scoring changes, update the Swift copy (or move the weights
into `manifest.json`).

`SubjectAware.render` mirrors `subject_aware_grade()` in
`src/grademl/compose.py`, and `SubjectMask.refine` mirrors
`grademl.mask.refine_mask`. The background LUT and the default strengths
travel in the preset export, so changing them in Python only needs a
re-export.
