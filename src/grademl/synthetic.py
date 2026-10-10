"""Synthetic "looks" for training the reference-matching model.

A look is a color grade (white balance, exposure, tone curves, channel
mixing, saturation, split toning, fade...) described by a `Look` and baked
into a 3D LUT by `render_look`. Looks are sampled either fully at random or
from a named creative direction (`DIRECTIONS`: cinematic teal & orange, warm
film, moody, pastel fade, B&W, ...), each a distribution over parameters, so
every sample of a direction is a different variation of the same idea.

Because we generate the look, we know the exact LUT, so a training sample is:

    content   = photo A
    reference = photo B graded with the look      (different scene!)
    target    = photo A graded with the same look

The model has to infer the look from B and transfer it to A, which is
exactly the "match the look" task.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .data import load_rgb, random_crop_box, resize
from .lut import DEFAULT_SIZE, apply_lut, identity_lut

LUMA = torch.tensor([0.2126, 0.7152, 0.0722])  # Rec. 709
# YIQ chroma plane: hue angle 0 = orange, pi = teal/blue, pi/2 = magenta,
# -pi/2 = green, about -0.6 = yellow. Offsets in this plane leave luminance alone.
_YIQ2RGB = torch.tensor([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647], [1.0, -1.106, 1.703]])
_RGB2YIQ = torch.linalg.inv(_YIQ2RGB)

ORANGE, TEAL, MAGENTA, GREEN, YELLOW, BLUE, PINK, CYAN = 0.0, math.pi, math.pi / 2, -math.pi / 2, -0.6, 2.8, 1.0, -2.6


@dataclass
class Look:
    """Parameters of a color grade. Defaults are the identity."""
    temp: float = 0.0            # + warm / - cool white balance
    tint: float = 0.0            # + green / - magenta
    ev: float = 0.0              # exposure in stops
    mixer: torch.Tensor | None = None                                  # (3, 3) channel mixer
    channel_curves: torch.Tensor = field(default_factory=lambda: torch.zeros(3, 3))  # per-channel curve coeffs
    curve: torch.Tensor = field(default_factory=lambda: torch.zeros(3))  # master curve coeffs
    saturation: float = 1.0
    mono: bool = False
    hue_shift: float = 0.0       # radians
    shadow_tint: tuple[float, float] = (0.0, 0.0)     # (hue, amount)
    highlight_tint: tuple[float, float] = (0.0, 0.0)  # (hue, amount)
    lift: float = 0.0            # faded blacks
    crush: float = 0.0           # lowered whites


def _luma(x: torch.Tensor) -> torch.Tensor:
    return (x @ LUMA)[:, None]


def _tint(hue: float, amount: float) -> torch.Tensor:
    """RGB offset with the given hue and ~zero luminance."""
    return _YIQ2RGB[:, 1:] @ torch.tensor([math.cos(hue), math.sin(hue)]) * amount


_K = torch.arange(1, 4, dtype=torch.float32) * math.pi
_GRID = torch.linspace(0, 1, 256)


def _monotone(a: torch.Tensor, min_slope: float = 0.2) -> torch.Tensor:
    """Shrink curve coefficients until f(x) = x + sum a_k sin(k pi x) is monotone."""
    slope = 1 + (a * _K * torch.cos(_GRID[:, None] * _K)).sum(1)
    if slope.min() < min_slope:
        a = a * ((1 - min_slope) / (1 - slope.min()))
    return a


def _curve(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """f(x) = x + sum_k a_k sin(k pi x), f(0)=0, f(1)=1.
    a_1 > 0 brightens mids; a_2 < 0 adds contrast (S-curve), > 0 flattens."""
    return x + (_monotone(a) * torch.sin(x[..., None] * _K)).sum(-1)


def render_look(look: Look, size: int = DEFAULT_SIZE) -> torch.Tensor:
    """Bake a Look into a (3, size, size, size) LUT."""
    x = identity_lut(size).reshape(3, -1).T.contiguous()  # (N, 3) RGB

    # White balance + exposure in (approximately) linear light.
    if look.temp or look.tint or look.ev:
        gains = torch.tensor([1 + look.temp, 1 + look.tint, 1 - look.temp])
        gains = gains / (gains @ LUMA)  # keep overall brightness
        x = (x.clamp(0, 1) ** 2.2 * gains * 2**look.ev).clamp(0, 1) ** (1 / 2.2)

    # Channel mixer; rows sum to 1 so neutral grays stay neutral.
    if look.mixer is not None:
        m = look.mixer
        x = (x @ (m / m.sum(1, keepdim=True)).T).clamp(0, 1)

    if look.channel_curves.abs().sum() > 0:
        x = torch.stack([_curve(x[:, c], look.channel_curves[c]) for c in range(3)], 1).clamp(0, 1)
    if look.curve.abs().sum() > 0:
        x = _curve(x, look.curve).clamp(0, 1)

    l = _luma(x)
    if look.mono:
        x = l.expand_as(x).clone()
    elif look.saturation != 1.0:
        x = (l + (x - l) * look.saturation).clamp(0, 1)

    if look.hue_shift:
        c, s = math.cos(look.hue_shift), math.sin(look.hue_shift)
        rot = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]])
        x = (x @ _RGB2YIQ.T @ rot.T @ _YIQ2RGB.T).clamp(0, 1)

    if look.shadow_tint[1] or look.highlight_tint[1]:
        l = _luma(x)
        x = x + (1 - l) ** 2 * _tint(*look.shadow_tint) + l**2 * _tint(*look.highlight_tint)
        x = x.clamp(0, 1)

    if look.lift or look.crush:
        x = look.lift + (1 - look.lift - look.crush) * x

    return x.clamp(0, 1).T.reshape(3, size, size, size).contiguous()


# ---------------------------------------------------------------- sampling

class _Rand:
    """Every random draw goes through one torch.Generator (reproducible)."""

    def __init__(self, g: torch.Generator):
        self.g = g

    def u(self, a: float, b: float) -> float:
        return a + (b - a) * torch.rand((), generator=self.g).item()

    def n(self, std: float = 1.0, size=()) -> torch.Tensor | float:
        t = torch.randn(size, generator=self.g) * std
        return t.item() if size == () else t

    def p(self, prob: float) -> bool:
        return torch.rand((), generator=self.g).item() < prob

    def choice(self, seq):
        return seq[int(torch.randint(len(seq), (), generator=self.g))]


def _generic(r: _Rand) -> Look:
    """Anything goes: each adjustment on or off at random."""
    lk = Look()
    if r.p(0.8):
        lk.temp, lk.tint = r.u(-0.25, 0.25), r.u(-0.1, 0.1)
        lk.ev = r.n(0.35) if r.p(0.6) else 0.0
    if r.p(0.5):
        lk.mixer = torch.eye(3) + r.n(0.07, (3, 3))
    if r.p(0.7):
        lk.channel_curves = r.n(1.0, (3, 3)) * torch.tensor([0.05, 0.05, 0.025])
    if r.p(0.9):
        lk.curve = r.n(1.0, 3) * torch.tensor([0.08, 0.08, 0.04])
    if r.p(0.05):
        lk.mono = True
    elif r.p(0.8):
        lk.saturation = min(max(math.exp(r.n(0.3)), 0.4), 1.6)
    if r.p(0.3):
        lk.hue_shift = r.n(math.radians(12))
    if r.p(0.6):
        h = r.u(0, 2 * math.pi)
        h2 = h + math.pi + r.n(0.6) if r.p(0.6) else r.u(0, 2 * math.pi)
        lk.shadow_tint, lk.highlight_tint = (h, r.u(0, 0.1)), (h2, r.u(0, 0.08))
    if r.p(0.3):
        lk.lift, lk.crush = r.u(0, 0.1), r.u(0, 0.06)
    return lk


def _teal_orange(r: _Rand) -> Look:
    """Blockbuster look: teal shadows, warm skin / highlights, punchy contrast."""
    return Look(
        temp=r.u(0.0, 0.12), curve=torch.tensor([r.n(0.03), r.u(-0.12, -0.05), r.n(0.02)]),
        saturation=r.u(1.0, 1.25),
        shadow_tint=(TEAL + r.n(0.25), r.u(0.08, 0.16)),
        highlight_tint=(ORANGE + r.n(0.2), r.u(0.05, 0.12)),
        lift=r.u(0, 0.04),
    )


def _warm_film(r: _Rand) -> Look:
    """Kodak-ish negative film: warm, gentle contrast, lifted blacks, creamy highlights."""
    return Look(
        temp=r.u(0.08, 0.25), tint=r.u(-0.02, 0.05),
        mixer=torch.eye(3) + r.n(0.04, (3, 3)),
        channel_curves=torch.tensor([[r.u(0, 0.04), 0, 0], [r.n(0.02), 0, 0], [r.u(-0.05, 0), r.n(0.02), 0]]),
        curve=torch.tensor([r.u(0, 0.06), r.u(-0.06, 0.0), 0]),
        saturation=r.u(0.85, 1.05),
        shadow_tint=(YELLOW + r.n(0.3), r.u(0.0, 0.06)),
        lift=r.u(0.03, 0.09), crush=r.u(0.0, 0.05),
    )


def _cool_moody(r: _Rand) -> Look:
    """Dark, desaturated, cool; green-teal shadows, heavy contrast."""
    return Look(
        temp=r.u(-0.25, -0.08), tint=r.u(0.0, 0.06), ev=r.u(-0.6, -0.15),
        curve=torch.tensor([r.u(-0.06, 0.0), r.u(-0.12, -0.04), r.n(0.02)]),
        saturation=r.u(0.55, 0.85),
        shadow_tint=(r.choice([TEAL, GREEN, CYAN]) + r.n(0.2), r.u(0.04, 0.12)),
        lift=r.u(0, 0.05),
    )


def _pastel_fade(r: _Rand) -> Look:
    """Bright, airy, faded: low contrast, raised blacks, soft pink / cyan tints."""
    return Look(
        temp=r.n(0.08), ev=r.u(0.1, 0.45),
        curve=torch.tensor([r.u(0.02, 0.08), r.u(0.03, 0.09), 0]),
        saturation=r.u(0.6, 0.9),
        shadow_tint=(r.choice([CYAN, BLUE, TEAL]) + r.n(0.3), r.u(0.03, 0.09)),
        highlight_tint=(r.choice([PINK, ORANGE, MAGENTA]) + r.n(0.3), r.u(0.02, 0.07)),
        lift=r.u(0.08, 0.18), crush=r.u(0.0, 0.06),
    )


def _bw(r: _Rand) -> Look:
    """Black and white, plain or toned (sepia, selenium, cyanotype)."""
    lk = Look(mono=True, curve=torch.tensor([r.n(0.04), r.u(-0.14, 0.02), r.n(0.02)]),
              # Non-negative weights, like a color filter on B&W film
              # (red filter = dark skies); negative ones would invert tones.
              lift=r.u(0, 0.06), mixer=torch.eye(3) + r.n(0.2, (3, 3)).abs())
    if r.p(0.6):
        lk.shadow_tint = (r.choice([ORANGE, YELLOW, BLUE, TEAL]) + r.n(0.2), r.u(0.03, 0.1))
        lk.highlight_tint = (lk.shadow_tint[0] + r.n(0.3), r.u(0.0, 0.06))
    return lk


def _vibrant(r: _Rand) -> Look:
    """Punchy travel / social look: saturated, contrasty, clean whites."""
    return Look(
        temp=r.n(0.06), ev=r.u(0.0, 0.25),
        curve=torch.tensor([r.u(0, 0.05), r.u(-0.1, -0.03), r.n(0.02)]),
        saturation=r.u(1.2, 1.5),
        hue_shift=r.n(math.radians(5)),
    )


def _vintage(r: _Rand) -> Look:
    """Faded 70s print / cross-processed: yellow-green cast, crushed blues, low contrast."""
    return Look(
        temp=r.u(0.05, 0.2), tint=r.u(0.0, 0.08),
        mixer=torch.eye(3) + r.n(0.08, (3, 3)),
        channel_curves=torch.tensor([[r.u(0, 0.06), r.n(0.03), 0], [r.u(0, 0.05), 0, 0],
                                     [r.u(-0.08, -0.02), r.u(0.0, 0.06), 0]]),
        curve=torch.tensor([r.n(0.03), r.u(0.0, 0.06), 0]),
        saturation=r.u(0.7, 0.95),
        shadow_tint=(YELLOW + r.n(0.4), r.u(0.03, 0.1)),
        lift=r.u(0.05, 0.14), crush=r.u(0.02, 0.08),
    )


def _neon_night(r: _Rand) -> Look:
    """Cyberpunk / night city: magenta-cyan split, deep blacks, strong saturation."""
    flip = r.p(0.5)
    return Look(
        temp=r.u(-0.2, 0.0), tint=r.u(-0.12, -0.02), ev=r.u(-0.4, 0.0),
        curve=torch.tensor([r.n(0.03), r.u(-0.14, -0.06), r.n(0.02)]),
        saturation=r.u(1.15, 1.45),
        shadow_tint=((MAGENTA if flip else CYAN) + r.n(0.2), r.u(0.06, 0.14)),
        highlight_tint=((CYAN if flip else MAGENTA) + r.n(0.2), r.u(0.04, 0.1)),
    )


DIRECTIONS = {
    "teal_orange": _teal_orange,
    "warm_film": _warm_film,
    "cool_moody": _cool_moody,
    "pastel_fade": _pastel_fade,
    "bw": _bw,
    "vibrant": _vibrant,
    "vintage": _vintage,
    "neon_night": _neon_night,
}


def _scale(look: Look, s: float) -> Look:
    """Interpolate every parameter between identity (s=0) and the look (s=1)."""
    mix = None if look.mixer is None else torch.eye(3) + s * (look.mixer - torch.eye(3))
    return Look(
        temp=s * look.temp, tint=s * look.tint, ev=s * look.ev, mixer=mix,
        channel_curves=s * look.channel_curves, curve=s * look.curve,
        saturation=1 + s * (look.saturation - 1), mono=look.mono and s > 0,
        hue_shift=s * look.hue_shift,
        shadow_tint=(look.shadow_tint[0], s * look.shadow_tint[1]),
        highlight_tint=(look.highlight_tint[0], s * look.highlight_tint[1]),
        lift=s * look.lift, crush=s * look.crush,
    )


def sample_look(g: torch.Generator | None = None, direction: str | None = None,
                strength: float = 1.0, p_generic: float = 0.35) -> Look:
    """Sample a Look from a named direction, or (direction=None) from a random
    direction with probability 1 - p_generic, otherwise a generic random look.
    Each look draws its own intensity in [0.5, 1.2] (times `strength`)."""
    r = _Rand(g or torch.Generator().manual_seed(random.getrandbits(63)))
    if direction is None:
        direction = "generic" if r.p(p_generic) else r.choice(list(DIRECTIONS))
    look = _generic(r) if direction == "generic" else DIRECTIONS[direction](r)
    return _scale(look, strength * r.u(0.5, 1.2))


def random_look(g: torch.Generator | None = None, size: int = DEFAULT_SIZE,
                strength: float = 1.0, direction: str | None = None) -> torch.Tensor:
    """Sample a look and return it as a (3, size, size, size) LUT."""
    return render_look(sample_look(g, direction, strength), size)


# ---------------------------------------------------------------- dataset

def list_images(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in exts)


class SyntheticMatchPairs(Dataset):
    """Base photo pairs + random looks for the reference-matching task.

    Returns (content, ref_scene, lut): two different photos (256px) and a
    look LUT rendered at `render_size` (17^3 by default; looks are smooth, so
    `make_match_batch` upsamples it to the full LUT size on the GPU with
    negligible error). Grading also happens on the GPU: DataLoader workers
    are CPU-bound otherwise.

    train=True: random crops and a fresh look every time.
    train=False: fixed looks and pairings per index (seeded), for comparable validation.
    p_same: probability that the reference is another random crop of the
    *same* photo (train only). Easier pairs that teach the model to compare
    colors directly; the rest use a different scene.
    p_pro: probability of a pro-style look (grademl.pro_looks) instead of
    the original generator.
    p_identity: probability of no look at all, i.e. the reference is an
    ordinary photo. Teaches the look head that natural photos carry no
    grade, and gives the match head natural references to learn from.
    """

    def __init__(self, images: list[Path], train: bool = True, length: int | None = None,
                 strength: float = 1.0, render_size: int = 17, seed: int = 0,
                 p_generic: float = 0.35, p_same: float = 0.0, p_identity: float = 0.0,
                 p_pro: float = 0.0):
        if len(images) < 2:
            raise ValueError("need at least 2 base images")
        self.images, self.train = images, train
        self.length = length or len(images)
        self.strength, self.render_size, self.seed, self.p_generic = strength, render_size, seed, p_generic
        self.p_same = p_same if train else 0.0
        self.p_identity = p_identity
        self.p_pro = p_pro  # share of looks from grademl.pro_looks (HSL, qualifiers, film recipes)

    def __len__(self) -> int:
        return self.length

    def _load(self, path: Path, rng: random.Random) -> torch.Tensor:
        img = load_rgb(path)
        if self.train:
            t, l, h, w = random_crop_box(*img.shape[1:])
            img = img[:, t:t + h, l:l + w]
            if rng.random() < 0.5:
                img = img.flip(-1)
        return resize(img)

    def __getitem__(self, i: int):
        rng = random.Random() if self.train else random.Random(self.seed * 1_000_003 + i)
        n = len(self.images)
        a = i % n
        # Short-circuit so p_same=0 draws no random number: validation
        # pairings must not depend on this option.
        if self.p_same and rng.random() < self.p_same:
            b = a  # same photo, independent crop / flip
        else:
            b = (a + rng.randrange(1, n)) % n  # a different scene for the reference
        g = torch.Generator().manual_seed(rng.getrandbits(63))
        if self.p_identity and rng.random() < self.p_identity:
            lut = identity_lut(self.render_size)
        elif self.p_pro and rng.random() < self.p_pro:
            from .pro_looks import render_pro_look, sample_pro_look
            lut = render_pro_look(sample_pro_look(g), self.render_size)
        else:
            lut = render_look(sample_look(g, strength=self.strength, p_generic=self.p_generic), self.render_size)
        return self._load(self.images[a], rng), self._load(self.images[b], rng), lut


def upsample_lut(lut: torch.Tensor, size: int) -> torch.Tensor:
    """(B, 3, d, d, d) -> (B, 3, size, size, size), trilinear; grid points stay aligned."""
    if lut.shape[-1] == size:
        return lut
    return F.interpolate(lut, size=(size,) * 3, mode="trilinear", align_corners=True)


def make_match_batch(content: torch.Tensor, ref_scene: torch.Tensor, lut: torch.Tensor,
                     lut_size: int = DEFAULT_SIZE):
    """Grade a batch on the current device.
    Returns (reference, target, lut): ref_scene and content with the look,
    and the look upsampled to lut_size (the LUT supervision target)."""
    lut = upsample_lut(lut, lut_size)
    return apply_lut(ref_scene, lut), apply_lut(content, lut), lut
