"""Pro-style color grades ("looks") rendered as 3D LUTs.

The first look generator (grademl.synthetic) only had global operations
(white balance, curves, saturation, luminance-band tints). What makes a
grade look professional is mostly *hue-selective* work and color
separation: push skies and greens toward teal while warm tones go golden,
deepen one hue, mute another, protect skin. This module follows a
colorist's pipeline, all as a color -> color function, so any look bakes
into a LUT:

  1. white balance + exposure (linear light)
  2. tone: filmic contrast around a pivot, shadow / highlight lift,
     black lift and white roll-off (L*)
  3. HSL: per hue sector (red, orange, yellow, green, aqua, blue, purple,
     magenta) a hue shift, saturation and luminance change, smoothly
     interpolated around the hue circle and faded out for near-neutrals
  4. saturation and vibrance
  5. optional monochrome (after HSL, so HSL luminance acts like color
     filters on B&W film)
  6. color wheels: a*b* tint for shadows, midtones and highlights
  7. film-like desaturation of deep shadows and bright highlights

`DIRECTIONS` holds recipes modeled on well-known looks; each call samples
a variation around the recipe, so no two looks are identical.
"""
from __future__ import annotations

import dataclasses
import math
import random
from dataclasses import dataclass, field

import torch

from .color import lab_to_rgb, rgb_to_lab, srgb_to_linear, linear_to_srgb
from .lut import DEFAULT_SIZE, identity_lut

# CIELAB hue angles (degrees) of the 8 HSL sectors. Sky blue sits around
# 250-270, skin around 50-60.
SECTORS = ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta")
SECTOR_HUES = torch.tensor([30.0, 60.0, 95.0, 135.0, 190.0, 260.0, 300.0, 340.0])
SECTOR_WIDTH = 22.0
LUMA = torch.tensor([0.2126, 0.7152, 0.0722])


def _zeros8() -> torch.Tensor:
    return torch.zeros(8)


def _zeros2() -> torch.Tensor:
    return torch.zeros(2)


@dataclass
class ProLook:
    """Parameters of a grade. Defaults are the identity."""
    # 1. input
    temp: float = 0.0          # + warm
    tint: float = 0.0          # + green / - magenta
    ev: float = 0.0
    # 2. tone (L* normalized to 0..1)
    contrast: float = 0.0      # + S-curve, - flatter (|c| <= 0.6)
    pivot: float = 0.45
    shadows: float = 0.0       # lift (+) / deepen (-) shadows, in L*/100
    highlights: float = 0.0    # lift (+) / roll off (-) highlights
    black_lift: float = 0.0    # matte blacks
    white_roll: float = 0.0    # lowered whites
    # 3. HSL per sector
    hue_shift: torch.Tensor = field(default_factory=_zeros8)  # degrees
    sat_shift: torch.Tensor = field(default_factory=_zeros8)  # fraction, -1..+
    lum_shift: torch.Tensor = field(default_factory=_zeros8)  # L* units
    # 4. global color
    saturation: float = 1.0
    vibrance: float = 0.0
    # 5. monochrome
    mono: bool = False
    # 6. color wheels (a*, b*) offsets
    shadow_tint: torch.Tensor = field(default_factory=_zeros2)
    mid_tint: torch.Tensor = field(default_factory=_zeros2)
    high_tint: torch.Tensor = field(default_factory=_zeros2)
    # How much the wheels apply per hue sector (a colorist's qualifier):
    # e.g. 0.2 on blue/aqua keeps a warm highlight tint off the sky, which
    # preserves warm/cool separation instead of tinting everything.
    wheel_mask: torch.Tensor = field(default_factory=lambda: torch.ones(8))
    # 7. film desaturation at the extremes
    shadow_desat: float = 0.0
    highlight_desat: float = 0.0

    def scaled(self, s: float) -> "ProLook":
        """Interpolate every parameter between identity (s=0) and this look (s=1); s>1 exaggerates."""
        neutral = ProLook()
        out = {}
        for f in dataclasses.fields(self):
            v, n = getattr(self, f.name), getattr(neutral, f.name)
            if f.name == "pivot":
                out[f.name] = v
            elif isinstance(v, bool):
                out[f.name] = v and s > 0
            else:
                out[f.name] = n + s * (v - n)
        out["contrast"] = max(-0.6, min(0.8, out["contrast"]))
        return ProLook(**out)


# ---------------------------------------------------------------- rendering

def _angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b + 180) % 360 - 180


def _hue_interp(h: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Smoothly interpolate per-sector values at hue angles h (degrees)."""
    w = torch.exp(-0.5 * (_angle_diff(h[..., None], SECTOR_HUES) / SECTOR_WIDTH) ** 2)
    return (w * values).sum(-1) / (w.sum(-1) + 1e-6)


def _tone(x: torch.Tensor, lk: ProLook) -> torch.Tensor:
    """Tone curve on L*/100. Monotone for the clamped parameter ranges."""
    c = lk.contrast
    if c:
        k = 7.0
        lo, hi = torch.sigmoid(torch.tensor(-k * lk.pivot)), torch.sigmoid(torch.tensor(k * (1 - lk.pivot)))
        s = (torch.sigmoid(k * (x - lk.pivot)) - lo) / (hi - lo)
        x = x + c * (s - x)
    if lk.shadows:
        x = x + lk.shadows * 6.75 * x * (1 - x) ** 2        # bump peaking at 1/3
    if lk.highlights:
        x = x + lk.highlights * 6.75 * x**2 * (1 - x)       # bump peaking at 2/3
    return lk.black_lift + (1 - lk.black_lift - lk.white_roll) * x.clamp(0, 1)


def render_pro_look(lk: ProLook, size: int = DEFAULT_SIZE) -> torch.Tensor:
    """Bake a ProLook into a (3, size, size, size) LUT."""
    rgb = identity_lut(size).reshape(3, -1).T.contiguous()  # (N, 3)

    # 1. White balance + exposure in linear light.
    if lk.temp or lk.tint or lk.ev:
        gains = torch.tensor([1 + lk.temp, 1 + lk.tint, 1 - lk.temp]).clamp_min(0.05)
        gains = gains / (gains @ LUMA)
        rgb = linear_to_srgb(srgb_to_linear(rgb) * gains * 2**lk.ev).clamp(0, 1)

    lab = rgb_to_lab(rgb)
    L, a, b = lab[:, 0] / 100, lab[:, 1], lab[:, 2]
    # 2. Tone.
    L = _tone(L, lk)

    # 3. HSL in LCh, gated by chroma so near-neutrals keep their hue.
    C = torch.sqrt(a**2 + b**2)
    h = torch.rad2deg(torch.atan2(b, a)) % 360
    gate = C**2 / (C**2 + 12.0**2)
    if lk.hue_shift.abs().sum():
        h = h + _hue_interp(h, lk.hue_shift) * gate
    if lk.sat_shift.abs().sum():
        C = C * (1 + _hue_interp(h, lk.sat_shift)).clamp_min(0)
    if lk.lum_shift.abs().sum():
        L = L + _hue_interp(h, lk.lum_shift) * gate * (C / 40).clamp(max=1) / 100

    # 4. Saturation / vibrance (vibrance boosts muted colors more).
    C = C * lk.saturation
    if lk.vibrance:
        C = C * (1 + lk.vibrance * (1 - C / 60).clamp(0, 1))
    # 5. Monochrome.
    if lk.mono:
        C = torch.zeros_like(C)
    a, b = C * torch.cos(torch.deg2rad(h)), C * torch.sin(torch.deg2rad(h))

    # 6. Color wheels by tonal range.
    x = L.clamp(0, 1)
    ws, wh = (1 - x) ** 2, x**2
    wm = (1 - ws - wh).clamp_min(0)
    tint = ws[:, None] * lk.shadow_tint + wm[:, None] * lk.mid_tint + wh[:, None] * lk.high_tint
    if (lk.wheel_mask != 1).any():
        # Masked by the original hue; neutrals (low chroma) always get the tint.
        m = _hue_interp(h, lk.wheel_mask)
        tint = tint * (gate * m + (1 - gate))[:, None]
    a, b = a + tint[:, 0], b + tint[:, 1]

    # 7. Film desaturation of the extremes.
    if lk.shadow_desat or lk.highlight_desat:
        k = (1 - lk.shadow_desat * (1 - x) ** 3 - lk.highlight_desat * x**3).clamp(0, 1)
        a, b = a * k, b * k

    out = lab_to_rgb(torch.stack([L * 100, a, b], -1))
    return out.T.reshape(3, size, size, size).contiguous()


# ---------------------------------------------------------------- recipes

def _v(**kw) -> torch.Tensor:
    """Per-sector vector from keyword args, e.g. _v(green=40, blue=-20)."""
    out = torch.zeros(8)
    for k, val in kw.items():
        out[SECTORS.index(k)] = val
    return out


def _ab(a: float, b: float) -> torch.Tensor:
    return torch.tensor([a, b], dtype=torch.float32)


# a*b* directions: +a magenta/red, -a green, +b yellow, -b blue.
# teal ~ (-10, -8), orange ~ (8, 14), gold ~ (3, 16), magenta ~ (12, -8).
RECIPES = {
    "teal_orange": ProLook(
        contrast=0.35, black_lift=0.02,
        hue_shift=_v(red=10, yellow=-22, green=45, aqua=10, blue=-30, purple=-35, magenta=-25),
        sat_shift=_v(red=0.15, orange=0.35, yellow=0.2, green=-0.3, aqua=0.25, blue=0.2, purple=-0.4, magenta=-0.4),
        lum_shift=_v(orange=5, yellow=3, green=-5, blue=-8),
        saturation=1.05, shadow_tint=_ab(-12, -10), mid_tint=_ab(3, 6), high_tint=_ab(8, 16),
        # warm tints stay off the teal side, teal shadows stay off skin / warm tones
        wheel_mask=_v(red=0.6, orange=0.5, yellow=0.7, green=1, aqua=0.3, blue=0.3, purple=0.5, magenta=0.6)),
    "golden_hour": ProLook(
        temp=0.08, ev=0.05, contrast=0.15, highlights=-0.04, black_lift=0.03, white_roll=0.02,
        hue_shift=_v(red=6, orange=-4, yellow=-18, green=-30, aqua=-12, blue=-6),
        sat_shift=_v(red=0.15, orange=0.35, yellow=0.3, green=-0.15, aqua=0.05, blue=0.12),
        lum_shift=_v(orange=6, yellow=5, blue=-5, aqua=-3),
        vibrance=0.15, shadow_tint=_ab(1, 6), mid_tint=_ab(5, 14), high_tint=_ab(7, 20),
        wheel_mask=_v(red=1, orange=1, yellow=1, green=0.8, aqua=0.15, blue=0.1, purple=0.4, magenta=0.8)),
    "portra": ProLook(
        temp=0.12, contrast=-0.15, black_lift=0.05, white_roll=0.04,
        hue_shift=_v(red=5, orange=-3, yellow=-10, green=-30, aqua=15, blue=-20),
        sat_shift=_v(orange=0.1, green=-0.35, blue=-0.25),
        saturation=0.9, shadow_tint=_ab(-2, 4), high_tint=_ab(4, 8), highlight_desat=0.2),
    "fuji_chrome": ProLook(
        temp=-0.05, contrast=0.3,
        hue_shift=_v(red=-5, yellow=-5, green=15, blue=-15),
        sat_shift=_v(red=0.1),
        saturation=0.75, shadow_tint=_ab(-5, -3), high_tint=_ab(2, 5), shadow_desat=0.3),
    "cinestill_night": ProLook(
        temp=-0.3, tint=-0.05, contrast=0.3, black_lift=0.02,
        hue_shift=_v(orange=10, red=-5),
        sat_shift=_v(red=0.3, orange=0.2, blue=0.1, green=-0.4),
        saturation=1.15, shadow_tint=_ab(-8, -14), high_tint=_ab(10, 6)),
    "moody_green": ProLook(
        ev=-0.3, contrast=0.25, black_lift=0.04,
        hue_shift=_v(green=25, yellow=-10),
        sat_shift=_v(green=-0.45, yellow=-0.3, blue=-0.3, orange=0.1),
        lum_shift=_v(green=-10),
        shadow_tint=_ab(-8, -2), high_tint=_ab(3, 6), shadow_desat=0.2),
    "matte_pastel": ProLook(
        ev=0.2, contrast=-0.3, black_lift=0.12, white_roll=0.05,
        hue_shift=_v(green=20, blue=-15, red=10),
        saturation=0.75, shadow_tint=_ab(-4, -6), high_tint=_ab(6, 3), highlight_desat=0.2),
    "bleach_bypass": ProLook(
        temp=-0.05, contrast=0.5, shadows=-0.05, saturation=0.45,
        shadow_tint=_ab(-3, -4), highlight_desat=0.3),
    "vibrant_travel": ProLook(
        contrast=0.25, vibrance=0.4, saturation=1.1,
        hue_shift=_v(green=-8, blue=5, aqua=10),
        sat_shift=_v(blue=0.25, aqua=0.3, green=0.1, orange=0.15),
        lum_shift=_v(blue=-8, aqua=-4), high_tint=_ab(2, 5)),
    "autumn": ProLook(
        temp=0.15, contrast=0.2,
        hue_shift=_v(green=-45, yellow=-20),
        sat_shift=_v(green=0.1, orange=0.3, yellow=0.2, blue=-0.2),
        lum_shift=_v(green=4), shadow_tint=_ab(2, 4), high_tint=_ab(6, 12)),
    "neon_night": ProLook(
        temp=-0.15, tint=-0.15, contrast=0.4,
        hue_shift=_v(yellow=-35, orange=-10, green=60, aqua=30, blue=30, purple=10),
        saturation=1.25, shadow_tint=_ab(12, -14), high_tint=_ab(-10, -6)),
    "bw_toned": ProLook(
        mono=True, contrast=0.3, lum_shift=_v(blue=-15, orange=8, red=5),
        shadow_tint=_ab(2, 6), high_tint=_ab(2, 8)),
    "vintage_70s": ProLook(
        temp=0.18, tint=0.04, contrast=-0.1, black_lift=0.08, white_roll=0.06,
        hue_shift=_v(blue=-30, green=-20, red=5),
        sat_shift=_v(blue=-0.4), saturation=0.8,
        shadow_tint=_ab(-4, 10), high_tint=_ab(3, 10)),
    "nordic_cool": ProLook(
        temp=-0.12, contrast=0.1, black_lift=0.03, saturation=0.85,
        hue_shift=_v(green=20, yellow=5), sat_shift=_v(orange=-0.1),
        shadow_tint=_ab(-4, -8), high_tint=_ab(-1, -2)),
}
DIRECTIONS = tuple(RECIPES)


class _Rand:
    def __init__(self, g: torch.Generator):
        self.g = g

    def u(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * torch.rand((), generator=self.g).item()

    def n(self, std: float, size=()):
        t = torch.randn(size, generator=self.g) * std
        return t.item() if size == () else t

    def p(self, prob: float) -> bool:
        return torch.rand((), generator=self.g).item() < prob


def _vary(lk: ProLook, r: _Rand, amount: float = 1.0) -> ProLook:
    """A variation around a recipe: every parameter jittered a little."""
    j = amount
    return dataclasses.replace(
        lk,
        temp=lk.temp + r.n(0.05 * j), tint=lk.tint + r.n(0.02 * j), ev=lk.ev + r.n(0.1 * j),
        contrast=lk.contrast + r.n(0.08 * j), shadows=lk.shadows + r.n(0.03 * j),
        highlights=lk.highlights + r.n(0.03 * j),
        black_lift=max(0.0, lk.black_lift + r.n(0.015 * j)),
        white_roll=max(0.0, lk.white_roll + r.n(0.01 * j)),
        hue_shift=lk.hue_shift + r.n(7 * j, 8),
        sat_shift=lk.sat_shift + r.n(0.08 * j, 8),
        lum_shift=lk.lum_shift + r.n(2 * j, 8),
        saturation=max(0.0, lk.saturation + r.n(0.06 * j)),
        vibrance=lk.vibrance + r.n(0.05 * j),
        shadow_tint=lk.shadow_tint + r.n(2.5 * j, 2),
        mid_tint=lk.mid_tint + r.n(1.5 * j, 2),
        high_tint=lk.high_tint + r.n(2.5 * j, 2),
    )


def _random_pro(r: _Rand) -> ProLook:
    """A random but coherent pro-style grade (not tied to a recipe)."""
    warm = r.u(-1, 1)
    return ProLook(
        temp=0.2 * warm + r.n(0.05), tint=r.n(0.04), ev=r.n(0.15),
        contrast=r.u(-0.3, 0.45), shadows=r.n(0.04), highlights=r.n(0.04),
        black_lift=max(0.0, r.n(0.05)), white_roll=max(0.0, r.n(0.03)),
        hue_shift=r.n(18, 8), sat_shift=r.n(0.2, 8), lum_shift=r.n(5, 8),
        saturation=max(0.3, 1 + r.n(0.2)), vibrance=r.n(0.15),
        mono=r.p(0.04),
        shadow_tint=r.n(8, 2), mid_tint=r.n(4, 2), high_tint=r.n(8, 2),
        shadow_desat=max(0.0, r.n(0.15)), highlight_desat=max(0.0, r.n(0.15)),
    )


def sample_pro_look(g: torch.Generator | None = None, direction: str | None = None,
                    strength: tuple[float, float] = (0.7, 1.3), p_random: float = 0.3) -> ProLook:
    """A variation of a named direction, or (direction=None) a random
    direction / with probability p_random a random pro grade. Intensity is
    drawn from `strength`."""
    r = _Rand(g or torch.Generator().manual_seed(random.getrandbits(63)))
    if direction is None:
        direction = "random" if r.p(p_random) else DIRECTIONS[int(torch.randint(len(DIRECTIONS), (), generator=r.g))]
    lk = _random_pro(r) if direction == "random" else _vary(RECIPES[direction], r)
    return lk.scaled(r.u(*strength))


def random_pro_look(g: torch.Generator | None = None, size: int = DEFAULT_SIZE, direction: str | None = None,
                    strength: tuple[float, float] = (0.7, 1.3)) -> torch.Tensor:
    return render_pro_look(sample_pro_look(g, direction, strength), size)
