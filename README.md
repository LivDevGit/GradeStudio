# Match the Look — reference-based color grading

Give the model a **reference image** (e.g. a film still) and **your photo**. It predicts a **3D LUT** that carries the reference's look over to your photo.

The prototype is built to run on-device. A small network predicts the LUT from downscaled (~256px) inputs, and the LUT is then applied to the full-resolution image in Metal / Core Image. The goal is to ship it in [Grade Studio](#) once the results are good enough.

Based on *Learning Image-adaptive 3D Lookup Tables for High Performance Photo Enhancement in Real-time* (Zeng et al., TPAMI 2020).

## Pipeline

| Stage | What | Status |
|---|---|---|
| Environment | WSL2 + PyTorch 2.7 (cu128) on an RTX 5070, Python 3.12 | done |
| Core ML smoke test | Both nets export to ML Program (iOS 17, fp16); MIL ops: conv, instance_norm, leaky_relu, linear, reduce_mean, reshape | done (Mac parity check pending) |
| Data | MIT-Adobe FiveK (low-res sRGB) + synthetic pairs from random LUTs/presets | todo |
| Stage 1 | Adaptive auto-enhance on FiveK to validate the technique | todo |
| Stage 2 | Reference-conditioned LUT prediction ("match the look") | todo |
| Demo | Gradio UI, later Hugging Face Spaces | todo |
| On-device | Core ML model + Metal LUT application in Grade Studio | todo |

## Layout

```
src/grademl/   model, LUT ops, datasets, training
scripts/       setup, data download/preprocessing, export
configs/       training configs (YAML)
tests/         unit tests (LUT ops, export parity)
data/          local datasets (git-ignored)
exports/       Core ML / ONNX artifacts (git-ignored)
```

## Setup (WSL2 / Ubuntu)

```bash
bash scripts/setup_wsl.sh            # venv, PyTorch cu128, deps, GPU check
source ~/.venvs/grademl/bin/activate
```

Datasets live at `$GRADEML_DATA` (default `~/grademl-data`, on the WSL filesystem for I/O speed).

## Data & licensing

- **MIT-Adobe FiveK** is licensed for research use. Here it is only used to validate the technique (stage 1).
- The **"match the look" model** is trained mainly on **synthetic pairs**: an image, plus the same image with a random LUT/preset applied, plus the exact parameters of that edit. Those pairs make up the shippable training set.
