# iOS integration (Grade Studio)

Status: the model, presets and Swift glue exist; nothing has run on Apple
hardware yet. Steps 1-3 below are the checklist before shipping.

## What ships

| Asset | Made by | Size |
|---|---|---|
| `match_lut.mlpackage` (Xcode compiles it to `.mlmodelc`) | `scripts/export_coreml.py --ckpt runs/match_v6/best.pt` | ~19 MB fp16 |
| `presets/` (14 directions x 6 variations, `.lut16` + `manifest.json`) | `scripts/export_presets.py` | ~18 MB |
| `ios/GradeML.swift` | hand-written | – |

Model I/O: inputs `content`, `reference` (256x256 RGB images; Core ML scales
0..255 to 0..1). Outputs `lut_look`, `lut_match`: `(1, 3, 33, 33, 33)`,
layout `lut[c, b, g, r]`, values in sRGB-encoded 0..1.

## Flow in the app

```
Preset / Surprise me:  PresetBank.cube(...) or Surprise.pick(...)   -> ColorCube
Custom reference:      LookMatcher.predict(photo, reference)       -> (look, match)
                       look.blended(toward: match, amount: matchSlider)
Then:                  cube.intensity(intensitySlider).apply(to: fullResCIImage)
```

The network runs once per (photo, reference) pair on 256px copies; slider
changes only re-blend two LUTs (cheap) and re-render through CIColorCube.

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

## Keeping Python and Swift in sync

`Surprise.scores` mirrors `direction_scores()` in `src/grademl/surprise.py`.
If the Python scoring changes, update the Swift copy (or move the weights
into `manifest.json`).
