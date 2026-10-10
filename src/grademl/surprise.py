"""'Surprise me': pick bold, photo-appropriate and mutually different looks.

No aesthetic labels exist for this yet, so the choice is rule-based and
transparent:
  1. profile the photo (sky, greenery, warm surfaces, darkness, colorfulness)
  2. score every pro-look direction for that profile
  3. sample n directions (softmax over scores), at most one per family
     (warm / cool / film / mono / stylized), so the options really differ
  4. push intensity a bit past the presets (bolder) and reject a variation
     if it clips noticeably more highlights / shadows than the original

A learned version (from which options users keep) can replace the scoring
later without changing the interface.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .color import rgb_to_lab
from .lut import DEFAULT_SIZE, apply_lut
from .pro_looks import DIRECTIONS, render_pro_look, sample_pro_look

FAMILY = {
    "golden_hour": "warm", "autumn": "warm", "portra": "film", "vintage_70s": "film",
    "fuji_chrome": "film", "teal_orange": "stylized", "neon_night": "stylized",
    "cinestill_night": "stylized", "moody_green": "cool", "nordic_cool": "cool",
    "matte_pastel": "soft", "vibrant_travel": "soft", "bleach_bypass": "mono", "bw_toned": "mono",
}


@dataclass
class Profile:
    brightness: float    # mean L*/100
    dark: float          # share of pixels with L* < 20
    sky: float           # share of bright-ish blue pixels
    green: float         # share of green foliage
    warm: float          # share of warm (red/orange/yellow) pixels
    colorfulness: float  # mean chroma / 40


def photo_profile(img: torch.Tensor) -> Profile:
    """img: (3, H, W) in [0, 1]."""
    small = F.interpolate(img[None], size=(96, 96), mode="area")[0]
    lab = rgb_to_lab(small.permute(1, 2, 0).reshape(-1, 3))
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    C = torch.sqrt(a**2 + b**2)
    h = torch.rad2deg(torch.atan2(b, a)) % 360
    colored = C > 10
    return Profile(
        brightness=float(L.mean() / 100),
        dark=float((L < 20).float().mean()),
        sky=float((colored & (h > 220) & (h < 300) & (L > 45)).float().mean()),
        green=float((colored & (h > 100) & (h < 170)).float().mean()),
        # Warm counts subtle chroma too: beige stone and skin are warm at C* ~ 6-12.
        warm=float(((C > 6) & ((h < 100) | (h > 340))).float().mean()),
        colorfulness=float(C.mean() / 40),
    )


def direction_scores(p: Profile) -> dict[str, float]:
    """Higher = better fit. Every direction keeps a base score, so any look
    can still come up; the profile tilts the odds."""
    night = p.brightness < 0.3 or p.dark > 0.4
    s = {d: 1.0 for d in DIRECTIONS}
    s["golden_hour"] += 2.0 * p.sky + 1.5 * p.warm + (0 if night else 0.5)
    s["teal_orange"] += 3.0 * min(p.sky + p.green, 1) + 1.5 * p.warm
    s["autumn"] += 3.0 * p.green
    s["moody_green"] += 2.5 * p.green + (0.5 if p.brightness < 0.45 else 0)
    s["cinestill_night"] += 2.5 * night + 1.0 * p.warm * night
    s["neon_night"] += 2.0 * night
    s["nordic_cool"] += 1.5 * p.sky + 0.5 * (p.colorfulness < 0.5)
    s["vibrant_travel"] += 1.5 * p.sky + 1.0 * p.green - 1.0 * (p.colorfulness > 0.9)
    s["portra"] += 1.0 * p.warm + 0.5
    s["fuji_chrome"] += 0.8
    s["vintage_70s"] += 0.8 * p.warm
    s["matte_pastel"] += 1.0 * (p.brightness > 0.55) - 1.0 * night
    # Typical photos sit at colorfulness ~0.25-0.5; only nearly colorless
    # ones lean monochrome.
    s["bw_toned"] += 1.2 * (p.colorfulness < 0.15)
    s["bleach_bypass"] += 0.8 * (p.colorfulness < 0.2)
    if not night:
        s["cinestill_night"] -= 0.5
        s["neon_night"] -= 0.7
    return {d: max(v, 0.05) for d, v in s.items()}


def _clipping(img: torch.Tensor) -> float:
    """Share of pixels that lost detail: blown to white or crushed to black.

    A single channel at 0 or 1 is normal for saturated colors (a deep teal
    has almost no red) and does not count; counting it rejected exactly the
    bold looks this feature is for.
    """
    blown = img.min(0).values > 0.97
    crushed = img.max(0).values < 0.02
    return float((blown | crushed).float().mean())


@dataclass
class Suggestion:
    direction: str
    seed: int
    strength: float
    lut: torch.Tensor  # (3, D, D, D)


@torch.no_grad()
def surprise(img: torch.Tensor, n: int = 3, seed: int = 0, temperature: float = 0.4,
             strength: tuple[float, float] = (1.0, 1.3), size: int = DEFAULT_SIZE,
             max_extra_clip: float = 0.04) -> list[Suggestion]:
    """n bold, distinct looks suited to img ((3, H, W) in [0, 1])."""
    g = torch.Generator().manual_seed(seed)
    scores = direction_scores(photo_profile(img))
    names = list(scores)
    logits = torch.tensor([scores[d] for d in names]).log() / temperature
    small = F.interpolate(img[None], size=(128, 128), mode="area")
    base_clip = _clipping(small[0])

    picks: list[Suggestion] = []
    used_families: set[str] = set()
    order = torch.multinomial(logits.softmax(0), len(names), replacement=False, generator=g).tolist()
    for i in order:
        d = names[i]
        if FAMILY[d] in used_families:
            continue
        for attempt in range(6):
            look_seed = int(torch.randint(2**31 - 1, (), generator=g))
            lo, hi = strength
            s = lo + (hi - lo) * torch.rand((), generator=g).item() * (0.85**attempt)  # back off on retries
            lut = render_pro_look(sample_pro_look(torch.Generator().manual_seed(look_seed), direction=d,
                                                  strength=(s, s)), size)
            if _clipping(apply_lut(small, lut)[0]) - base_clip <= max_extra_clip:
                picks.append(Suggestion(d, look_seed, s, lut))
                used_families.add(FAMILY[d])
                break
        if len(picks) == n:
            break
    return picks
