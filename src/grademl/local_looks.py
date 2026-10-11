"""Local finishing: what a colorist does with masks after the global look.

GradeNet v1 learned global grades only, because its targets were global:
one LUT per (photo, direction). A professional edit is a global look plus
local work, and the local work is most of what separates "a filter" from
"an edit":

  sky       recover its density and enrich it in the look's own color
            (teal for teal & orange, peach for golden hour, a darker sky
            for black & white like a red filter on film)
  subject   people get the look at reduced strength (skin stays natural)
            and a little light on them; the background is calmed so they
            stand out
  depth     aerial perspective: distant parts lose a little contrast and
            take the look's "air" tint, near parts keep their punch
  vignette  a light edge burn, held off the highlights

and the input side gets the local faults a photographer fixes with masks:

  a bright sky over a dark foreground, a backlit subject, light falling
  off across the frame (a graduated imbalance)

The masks (sky, person, depth: scripts/label_scenes.py) are used only to
build these (input, target) pairs. LocalGradeNet (grademl.local) never sees
them: it learns from the photo itself where the sky and the subject are and
how to treat them, with a bilateral grid that follows the photo's edges.

Everything runs batched on the GPU; region looks are LUTs rendered once
per direction (FinishBank).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .color import linear_to_srgb, srgb_to_linear
from .lut import DEFAULT_SIZE, apply_lut, identity_lut
from .mask import guided_filter
from .pro_looks import DIRECTIONS, ProLook, _ab, _v, render_pro_look

# Scene map channels written by scripts/label_scenes.py.
SKY, PERSON, PLANT, WATER, BUILT, DEPTH = range(6)


@dataclass
class Finish:
    """Local work for one direction. Amounts are 0..1 blends of the region look."""
    sky_pre: ProLook = field(default_factory=lambda: ProLook(
        ev=-0.25, highlights=-0.1, contrast=0.1, sat_shift=_v(blue=0.2, aqua=0.15)))
    sky_post: ProLook = field(default_factory=ProLook)   # tint after the look, sky only
    sky: float = 0.8
    air: tuple[float, float] = (-1.0, -4.0)               # a*b* tint of distant haze
    haze: float = 0.5                                     # aerial perspective amount
    subject_strength: float = 0.7                         # look strength on people
    subject_ev: float = 0.15                              # light on the subject
    background_ev: float = -0.15                          # calm the background (people only)
    background_sat: float = 0.92
    vignette: float = 0.25                                # EV at the corners


def _sky(ev=-0.25, hl=-0.1, c=0.1, sat=None, hue=None, temp=0.0, saturation=1.0) -> ProLook:
    return ProLook(ev=ev, highlights=hl, contrast=c, temp=temp, saturation=saturation,
                   sat_shift=sat if sat is not None else _v(blue=0.2, aqua=0.15),
                   hue_shift=hue if hue is not None else torch.zeros(8))


FINISHES: dict[str, Finish] = {
    # "none": a natural pro edit: sky and subject balanced, nothing stylized.
    "none": Finish(sky_pre=_sky(ev=-0.2, c=0.05, sat=_v(blue=0.1, aqua=0.08)), sky=0.7, haze=0.0,
                   subject_strength=1.0, subject_ev=0.12, background_ev=-0.08, background_sat=0.97,
                   vignette=0.0),
    "teal_orange": Finish(sky_pre=_sky(ev=-0.3, sat=_v(blue=0.3, aqua=0.3), hue=_v(blue=-15)),
                          sky_post=ProLook(high_tint=_ab(-6, -4), mid_tint=_ab(-6, -6)),
                          air=(-4.0, -6.0), haze=0.6, vignette=0.3),
    "golden_hour": Finish(sky_pre=_sky(ev=-0.2, temp=0.08, sat=_v(orange=0.3, yellow=0.2, blue=0.05)),
                          sky_post=ProLook(high_tint=_ab(6, 12), mid_tint=_ab(4, 8)),
                          air=(3.0, 8.0), haze=0.7, vignette=0.25),
    "portra": Finish(sky_pre=_sky(ev=-0.12, c=0.0, sat=_v(blue=-0.1, aqua=0.05), hue=_v(blue=-10)),
                     air=(1.0, 3.0), haze=0.4, subject_strength=0.8, vignette=0.15),
    "fuji_chrome": Finish(sky_pre=_sky(ev=-0.35, c=0.15, sat=_v(blue=0.15, aqua=0.1)),
                          air=(-2.0, -5.0), haze=0.4, vignette=0.3),
    "cinestill_night": Finish(sky_pre=_sky(ev=-0.25, temp=-0.08, sat=_v(blue=0.15)),
                              sky_post=ProLook(mid_tint=_ab(-2, -8)),
                              air=(-3.0, -8.0), haze=0.3, vignette=0.35),
    "moody_green": Finish(sky_pre=_sky(ev=-0.4, c=0.15, saturation=0.7, sat=torch.zeros(8)),
                          air=(-3.0, -1.0), haze=0.6, background_ev=-0.25, vignette=0.45),
    "matte_pastel": Finish(sky_pre=_sky(ev=-0.05, hl=-0.08, c=-0.05, saturation=0.85, sat=torch.zeros(8)),
                           sky_post=ProLook(high_tint=_ab(5, 1)),
                           air=(2.0, -2.0), haze=0.5, subject_ev=0.1, vignette=0.1),
    "bleach_bypass": Finish(sky_pre=_sky(ev=-0.4, c=0.25, saturation=0.8, sat=torch.zeros(8)),
                            air=(-1.0, -2.0), haze=0.3, background_ev=-0.25, vignette=0.4),
    "vibrant_travel": Finish(sky_pre=_sky(ev=-0.35, c=0.15, sat=_v(blue=0.4, aqua=0.35)),
                             air=(-2.0, -5.0), haze=0.3, subject_strength=0.75, vignette=0.2),
    "autumn": Finish(sky_pre=_sky(ev=-0.2, temp=0.05, sat=_v(blue=0.1)),
                     sky_post=ProLook(high_tint=_ab(3, 6)), air=(3.0, 7.0), haze=0.6, vignette=0.3),
    "neon_night": Finish(sky_pre=_sky(ev=-0.25, sat=_v(blue=0.3, purple=0.3), hue=_v(blue=20)),
                         sky_post=ProLook(mid_tint=_ab(8, -10), high_tint=_ab(6, -8)),
                         air=(6.0, -8.0), haze=0.4, vignette=0.4),
    # Darkening blue *before* the mono conversion: a red filter on B&W film.
    "bw_toned": Finish(sky_pre=ProLook(ev=-0.3, contrast=0.15, lum_shift=_v(blue=-25, aqua=-15)),
                       air=(0.0, 0.0), haze=0.4, subject_strength=0.9, vignette=0.4),
    "vintage_70s": Finish(sky_pre=_sky(ev=-0.1, c=-0.05, saturation=0.75, sat=torch.zeros(8)),
                          sky_post=ProLook(high_tint=_ab(3, 10)), air=(3.0, 8.0), haze=0.6, vignette=0.3),
    "nordic_cool": Finish(sky_pre=_sky(ev=-0.2, temp=-0.05, sat=_v(blue=0.1)),
                          air=(-2.0, -6.0), haze=0.6, vignette=0.2),
}
assert set(FINISHES) == {"none", *DIRECTIONS}


class FinishBank:
    """Every direction's region looks as stacked LUTs, indexed by style
    (grademl.grade.STYLES order: "none", then DIRECTIONS)."""

    def __init__(self, styles: tuple[str, ...], size: int = DEFAULT_SIZE, device=None):
        fin = [FINISHES[s] for s in styles]
        self.sky_pre = torch.stack([render_pro_look(f.sky_pre, size) for f in fin]).to(device)
        self.sky_post = torch.stack([render_pro_look(f.sky_post, size) for f in fin]).to(device)
        haze = [ProLook(contrast=-0.15, black_lift=0.04, saturation=0.85, mid_tint=_ab(*f.air),
                        high_tint=_ab(*f.air)) for f in fin]
        self.haze = torch.stack([render_pro_look(h, size) for h in haze]).to(device)
        bg = [ProLook(ev=f.background_ev, saturation=f.background_sat, contrast=-0.05) for f in fin]
        self.background = torch.stack([render_pro_look(b, size) for b in bg]).to(device)

        def col(name):
            return torch.tensor([getattr(f, name) for f in fin], device=device, dtype=torch.float32)

        self.sky, self.haze_amt, self.vignette = col("sky"), col("haze"), col("vignette")
        self.subject_strength, self.subject_ev = col("subject_strength"), col("subject_ev")


# ---------------------------------------------------------------- masks


def smoothstep(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    t = ((x - lo) / (hi - lo)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def refine(mask: torch.Tensor, img: torch.Tensor, r: int = 4, eps: float = 1e-3) -> torch.Tensor:
    """Snap soft masks (B, C, H, W) to the photo's edges (gray guided filter)."""
    guide = (img * torch.tensor([0.2126, 0.7152, 0.0722], device=img.device).view(1, 3, 1, 1)).sum(1, keepdim=True)
    out = [guided_filter(mask[:, c:c + 1], guide, r, eps) for c in range(mask.shape[1])]
    return torch.cat(out, 1).clamp(0, 1)


def region_masks(maps: torch.Tensor, img: torch.Tensor) -> dict[str, torch.Tensor]:
    """Scene maps (B, 6, H, W) in [0, 1] at the photo's size -> soft masks
    (B, 1, H, W): sky, person, far (distant, not sky), and whether the photo
    has a person at all (B,)."""
    sky = smoothstep(maps[:, SKY:SKY + 1], 0.3, 0.7)
    person = smoothstep(maps[:, PERSON:PERSON + 1], 0.3, 0.7)
    sky, person = refine(torch.cat([sky, person], 1), img).split(1, 1)
    depth = maps[:, DEPTH:DEPTH + 1].clamp(0, 1)  # antialiased resizing can overshoot
    far = ((1 - depth) ** 1.5) * (1 - sky) * (1 - person)
    has_person = (person.mean((1, 2, 3)) > 0.03).float()
    return {"sky": sky, "person": person, "far": far, "has_person": has_person}


def vignette_map(b: int, h: int, w: int, device) -> torch.Tensor:
    """(B, 1, H, W): 0 in the middle, 1 in the corners, aspect-aware."""
    ys = torch.linspace(-1, 1, h, device=device)[:, None]
    xs = torch.linspace(-1, 1, w, device=device)[None]
    r2 = (xs**2 + ys**2) / 2
    return (r2**1.25).expand(b, 1, h, w)


# ---------------------------------------------------------------- operations


def _bl(x: torch.Tensor) -> torch.Tensor:
    return x.view(-1, 1, 1, 1)


def mix(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Mask blend in linear light (as compose.blend)."""
    return linear_to_srgb((srgb_to_linear(a) * (1 - m) + srgb_to_linear(b) * m).clamp(0, 1))


def gain(img: torch.Tensor, ev: torch.Tensor) -> torch.Tensor:
    """Exposure change ev (B, 1, H, W), in linear light."""
    return linear_to_srgb((srgb_to_linear(img) * torch.exp2(ev)).clamp(0, 1))


def finish(photo: torch.Tensor, look: torch.Tensor, style: torch.Tensor, maps: torch.Tensor,
           bank: FinishBank) -> torch.Tensor:
    """The finished photo: global look plus local work.

    photo (B, 3, H, W) clean; look (B, 3, D, D, D) the global look LUT
    (identity for "none"); style (B,) index into the bank; maps (B, 6, H, W).
    """
    m = region_masks(maps, photo)
    ident = identity_lut(look.shape[-1], device=photo.device)

    # 1. Before the look: sky density + color (pre-look so B&W gets a filter
    #    effect) and light on the subject.
    pre = mix(photo, apply_lut(photo, bank.sky_pre[style]), m["sky"] * _bl(bank.sky[style]))
    pre = gain(pre, m["person"] * _bl(bank.subject_ev[style]))

    # 2. The look, gentler on people.
    graded = apply_lut(pre, look)
    s = _bl(bank.subject_strength[style]).view(-1, 1, 1, 1, 1)
    soft_look = ident + s * (look - ident)
    graded = mix(graded, apply_lut(pre, soft_look), m["person"])

    # 3. After the look: sky tint, aerial perspective, calmer background
    #    behind people, vignette held off the highlights.
    graded = mix(graded, apply_lut(graded, bank.sky_post[style]), m["sky"] * _bl(bank.sky[style]))
    graded = mix(graded, apply_lut(graded, bank.haze[style]), m["far"] * _bl(bank.haze_amt[style]))
    bg = (1 - m["person"]) * _bl(m["has_person"])
    graded = mix(graded, apply_lut(graded, bank.background[style]), bg)
    lum = (graded * torch.tensor([0.2126, 0.7152, 0.0722], device=photo.device).view(1, 3, 1, 1)).sum(1, keepdim=True)
    v = vignette_map(*photo.shape[:1], *photo.shape[2:], photo.device) * (1 - 0.6 * lum**2)
    return gain(graded, -v * _bl(bank.vignette[style])).clamp(0, 1)


# ---------------------------------------------------------------- local faults


N_FAULT = 6  # sky_ev, ground_ev, subject_ev, grad_ev, grad_angle, grad_pos


def random_local_fault(rng: random.Random, p: float = 0.6) -> torch.Tensor:
    """Parameters of the local exposure faults a photographer fixes with
    masks (each present with some probability; p=0 for none):
      sky_ev     sky brighter than the scene (bright sky, dark ground)
      ground_ev  everything but the sky darker
      subject_ev a backlit, underexposed subject
      grad_*     light falling off across the frame (a graduated imbalance)
    """
    if rng.random() > p:
        return torch.zeros(N_FAULT)
    sky = rng.uniform(0.2, 0.8) if rng.random() < 0.5 else 0.0
    ground = -rng.uniform(0.2, 0.6) if rng.random() < 0.4 else 0.0
    subject = -rng.uniform(0.3, 0.8) if rng.random() < 0.4 else 0.0
    grad = rng.choice([-1, 1]) * rng.uniform(0.25, 0.7) if rng.random() < 0.35 else 0.0
    return torch.tensor([sky, ground, subject, grad, rng.uniform(0, 2 * math.pi), rng.uniform(-0.4, 0.4)])


def apply_local_fault(photo: torch.Tensor, params: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
    """photo (B, 3, H, W), params (B, N_FAULT), maps (B, 6, H, W)."""
    if not params.abs().any():
        return photo
    m = region_masks(maps, photo)
    b, _, h, w = photo.shape
    ys = torch.linspace(-1, 1, h, device=photo.device)[:, None]
    xs = torch.linspace(-1, 1, w, device=photo.device)[None]
    ang, pos = params[:, 4].view(-1, 1, 1), params[:, 5].view(-1, 1, 1)
    proj = xs * torch.cos(ang) + ys * torch.sin(ang)                  # (B, H, W)
    grad = torch.sigmoid((proj - pos) / 0.25)[:, None]
    ev = (m["sky"] * _bl(params[:, 0]) + (1 - m["sky"]) * _bl(params[:, 1])
          + m["person"] * _bl(params[:, 2]) + grad * _bl(params[:, 3]))
    return gain(photo, ev)


# ---------------------------------------------------------------- training data


def load_maps(path, size: tuple[int, int]) -> torch.Tensor:
    """Scene maps (.npz from label_scenes.py) as (6, H, W) float in [0, 1] at `size`."""
    import numpy as np

    maps = torch.from_numpy(np.load(path)["maps"]).float() / 255
    return F.interpolate(maps[None], size=size, mode="bilinear", align_corners=False)[0]


class LocalGradePairs(torch.utils.data.Dataset):
    """(photo 256px, scene maps 256px, global fault LUT, local fault params, style).

    Photos without scene maps are skipped. Faults, the look and the local
    finishing are applied on the GPU (make_local_batch).
    """

    def __init__(self, images: list, maps_dir, train: bool = True, length: int | None = None,
                 render_size: int = 17, p_clean: float = 0.15, p_none: float = 0.25,
                 p_local: float = 0.6, seed: int = 0):
        from pathlib import Path

        self.maps_dir = Path(maps_dir)
        self.images = [p for p in images if (self.maps_dir / f"{p.stem}.npz").exists()]
        self.train = train
        self.length = length or len(self.images)
        self.render_size, self.p_clean, self.p_none, self.p_local, self.seed = (
            render_size, p_clean, p_none, p_local, seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        from .data import load_rgb, random_crop_box, resize
        from .grade import STYLES, random_fault

        rng = random.Random() if self.train else random.Random(self.seed * 1_000_003 + i)
        path = self.images[i % len(self.images)]
        img = load_rgb(path)
        maps = load_maps(self.maps_dir / f"{path.stem}.npz", img.shape[1:])
        if self.train:
            t, l, h, w = random_crop_box(*img.shape[1:])
            img, maps = img[:, t:t + h, l:l + w], maps[:, t:t + h, l:l + w]
            if rng.random() < 0.5:
                img, maps = img.flip(-1), maps.flip(-1)
        clean = rng.random() < self.p_clean
        fault = identity_lut(self.render_size) if clean else render_pro_look(random_fault(rng), self.render_size)
        local = torch.zeros(N_FAULT) if clean else random_local_fault(rng, self.p_local)
        style = 0 if rng.random() < self.p_none else rng.randrange(1, len(STYLES))
        return resize(img), resize(maps), fault, local, style


def make_local_batch(photo, maps, fault, local, style, recipes, bank: FinishBank):
    """(input, target) on the GPU: the photo with local and global faults,
    and the finished photo (look + local finishing)."""
    from .grade import backed_off_looks

    size = recipes.shape[-1]
    fault = F.interpolate(fault, size=(size,) * 3, mode="trilinear", align_corners=True)
    inp = apply_lut(apply_local_fault(photo, local, maps), fault).clamp(0, 1)
    look = backed_off_looks(photo, style, recipes)
    return inp, finish(photo, look, style, maps, bank)
