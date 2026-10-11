"""Subject-aware grading: one look, graded differently on the person and
on the background, blended through a soft person mask (grademl.mask).

  subject     = look at subject_strength           (gentler: skin stays natural)
  background  = look at background_strength, then BACKGROUND_POP at
                background_pop                     (a bit darker and calmer,
                                                    so the subject stands out)
  out         = mask * subject + (1 - mask) * background   (in linear light)

Works with any LUT: presets, Surprise me picks and model predictions. The
same rule is implemented in ios/GradeML.swift (SubjectAware.render): two
CIColorCube passes on the background, one on the subject, CIBlendWithMask
in Core Image's linear working space. Keep the two in sync.

With no subject in the frame (mask coverage < MIN_COVERAGE) the photo gets
the plain look, otherwise a landscape would be darkened as "background".
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .color import linear_to_srgb, srgb_to_linear
from .lut import DEFAULT_SIZE, apply_lut, identity_lut
from .mask import MIN_COVERAGE, coverage
from .pro_looks import ProLook, render_pro_look

# Exported with the presets (background_pop.lut16) so the app uses the exact same LUT.
BACKGROUND_POP = ProLook(ev=-0.5, contrast=-0.08, highlights=-0.06, saturation=0.7, temp=-0.03)


@dataclass
class SubjectAware:
    subject_strength: float = 0.75
    background_strength: float = 1.0
    background_pop: float = 0.6
    min_coverage: float = MIN_COVERAGE


def background_pop_lut(size: int = DEFAULT_SIZE) -> torch.Tensor:
    return render_pro_look(BACKGROUND_POP, size)


def scale_lut(lut: torch.Tensor, s: float, base: torch.Tensor | None = None) -> torch.Tensor:
    """base + s * (lut - base), clamped: the app's intensity slider. base
    defaults to the identity; for GradeNet it is the photo's correction LUT,
    so intensity scales the style and keeps the correction."""
    if base is None:
        base = identity_lut(lut.shape[-1], device=lut.device, dtype=lut.dtype)
    return (base + s * (lut - base)).clamp(0, 1)


def compose_luts(first: torch.Tensor, then: torch.Tensor) -> torch.Tensor:
    """LUT equivalent to applying `first`, then `then` (for .cube export)."""
    d = first.shape[-1]
    return apply_lut(first.reshape(1, 3, d, d * d), then).reshape(3, d, d, d)


def blend(subject: torch.Tensor, background: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mask blend in linear light (what CIBlendWithMask does in Core Image's
    default working space; it also gives cleaner hair edges than sRGB)."""
    lin = mask * srgb_to_linear(subject) + (1 - mask) * srgb_to_linear(background)
    return linear_to_srgb(lin.clamp(0, 1))


def split_luts(look: torch.Tensor, cfg: SubjectAware, intensity: float = 1.0,
               pop: torch.Tensor | None = None, base: torch.Tensor | None = None,
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(subject LUT, background LUT, background pop LUT), all already scaled."""
    if pop is None:
        pop = background_pop_lut(look.shape[-1]).to(look)
    return (scale_lut(look, intensity * cfg.subject_strength, base),
            scale_lut(look, intensity * cfg.background_strength, base),
            scale_lut(pop, cfg.background_pop))


def subject_aware_grade(img: torch.Tensor, look: torch.Tensor, mask: torch.Tensor | None,
                        cfg: SubjectAware = SubjectAware(), intensity: float = 1.0,
                        pop: torch.Tensor | None = None, base: torch.Tensor | None = None) -> torch.Tensor:
    """img (B, 3, H, W) or (3, H, W); look (3, D, D, D) at strength 1;
    mask (1, H, W) or None; base: what intensity 0 means (default identity;
    GradeNet: the correction LUT). Returns the graded image, same shape as img."""
    batched = img.dim() == 4
    x = img if batched else img[None]
    if mask is None or coverage(mask) < cfg.min_coverage:
        out = apply_lut(x, scale_lut(look, intensity, base))
    else:
        subj, bg, pop_ = split_luts(look, cfg, intensity, pop, base)
        out = blend(apply_lut(x, subj), apply_lut(apply_lut(x, bg), pop_), mask.to(x))
    return out if batched else out[0]
