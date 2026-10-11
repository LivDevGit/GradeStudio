"""Film grain: a finishing step after the LUT, free on any style.

A LUT maps colors to colors and cannot make grain (spatial noise), so grain
is a separate, procedural step with two user controls (amount, size). It
works on top of any grade: styles, references, the editor's version.

The method is chosen to be reproducible in Core Image (ios/GradeML.swift,
Grain): uniform noise at a coarse grid, linearly upscaled (clumps of about
`size` px), centered on mid-gray and soft-light blended onto the photo. Soft
light leaves black and white almost untouched and is strongest in the
midtones, like real film grain. The noise itself differs between Python
and Swift (different generators); the look (strength, clump size, tonal
response) is the same.

`size` is relative: grain px = size * short side / 1500, so grain looks the
same on a preview and on the full-resolution export.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

UNIFORM_STD = 12 ** -0.5  # std of U(0, 1); amount is the std of the grain layer


@dataclass(frozen=True)
class Grain:
    amount: float = 0.0  # std of the gray grain layer around 0.5 (0.05 = clearly visible)
    size: float = 1.0    # clump size, relative (see module doc)


def soft_light(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    """W3C / PDF soft light (what CISoftLightBlendMode implements)."""
    d = torch.where(base <= 0.25, ((16 * base - 12) * base + 4) * base, base.clamp_min(0).sqrt())
    return torch.where(blend <= 0.5,
                       base - (1 - 2 * blend) * base * (1 - base),
                       base + (2 * blend - 1) * (d - base))


def add_grain(img: torch.Tensor, grain: Grain, seed: int = 0) -> torch.Tensor:
    """img (3, H, W) or (B, 3, H, W) in [0, 1] -> same shape with grain."""
    if grain.amount <= 0:
        return img
    batched = img.dim() == 4
    x = img if batched else img[None]
    b, _, h, w = x.shape
    px = max(1.0, grain.size * min(h, w) / 1500)
    gh, gw = max(2, round(h / px)), max(2, round(w / px))
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.rand(b, 1, gh, gw, generator=g).to(x)
    noise = F.interpolate(noise, size=(h, w), mode="bilinear", align_corners=False)
    layer = (0.5 + (grain.amount / UNIFORM_STD) * (noise - 0.5)).clamp(0, 1)
    out = soft_light(x, layer.expand_as(x)).clamp(0, 1)  # same noise on R, G, B: luminance grain
    return out if batched else out[0]
