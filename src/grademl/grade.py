"""Machine-learned grading: GradeNet edits each photo in a colorist direction.

A fixed preset applies the same color mapping to every photo. GradeNet
(grademl.model) instead looks at the photo and predicts the LUT for it:

  training input   a finished Unsplash photo with random technical faults
                   (exposure, white balance, contrast, saturation, a flat
                   camera profile)
  training target  the finished photo graded in the requested direction,
                   toned down where the full direction would clip
                   highlights or crush shadows on this photo

So one pass corrects the photo (a dark photo is lifted, a cast removed)
and grades it, adapted to what is in it. What each direction means
(golden hour: warm gold highlights that stay off the sky, ...) comes from
the colorist recipes in grademl.pro_looks; how to apply it to a given
photo is learned. Style "none" is correction only.

"Best version" (top_versions / best_version) asks GradeNet for every
direction at a few intensities and lets the same network judge the results:
with style "none" it predicts how much a graded photo still needs
correcting, and the less the better. Plus a bonus for a visible grade, penalties for clipping and
for skin pushed off its natural color, and a small prior for directions
that suit the scene. The editor ranks its choices: #1 is its pick, #2 and
#3 the next best, one per style (the same style at another intensity is not
a different choice).
ios/GradeML.swift (Grader, Editor) mirrors this.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .color import rgb_to_lab
from .compose import SubjectAware, scale_lut
from .data import load_rgb, random_crop_box, resize
from .lut import DEFAULT_SIZE, apply_lut, identity_lut
from .mask import MIN_COVERAGE, coverage
from .pro_looks import DIRECTIONS, RECIPES, ProLook, render_pro_look
from .surprise import FAMILY, direction_scores, photo_profile

STYLES = ("none", *DIRECTIONS)  # GradeNet's style one-hot order; exported to the app


def style_onehot(styles: list[int] | torch.Tensor, device=None) -> torch.Tensor:
    idx = torch.as_tensor(styles, device=device)
    return F.one_hot(idx, len(STYLES)).float()


# ---------------------------------------------------------------- training data


def random_fault(rng: random.Random) -> ProLook:
    """Technical faults a pro would fix, each present with p=0.6."""
    def on() -> bool:
        return rng.random() < 0.6

    if rng.random() < 0.25:  # flat camera profile (ungraded raw / log-ish)
        flat = dict(contrast=-rng.uniform(0.2, 0.45), saturation=rng.uniform(0.65, 0.85),
                    black_lift=rng.uniform(0.02, 0.07), white_roll=rng.uniform(0.0, 0.05))
    else:
        flat = {}
    return ProLook(
        temp=rng.gauss(0, 0.12) if on() else 0.0,
        tint=rng.gauss(0, 0.05) if on() else 0.0,
        ev=rng.gauss(0, 0.4) if on() else 0.0,
        contrast=flat.get("contrast", rng.uniform(-0.35, 0.3) if on() else 0.0),
        shadows=rng.gauss(0, 0.05) if on() else 0.0,
        highlights=rng.gauss(0, 0.05) if on() else 0.0,
        black_lift=flat.get("black_lift", max(0.0, rng.gauss(0, 0.03)) if on() else 0.0),
        white_roll=flat.get("white_roll", max(0.0, rng.gauss(0, 0.03)) if on() else 0.0),
        saturation=flat.get("saturation", math.exp(rng.gauss(0, 0.25)) if on() else 1.0),
        vibrance=rng.gauss(0, 0.1) if on() else 0.0,
    )


class GradePairs(Dataset):
    """(finished photo 256px, fault LUT at render_size, style index).

    Faults and grading are applied on the GPU (make_grade_batch).
    p_clean: no fault (the photo is already finished: leave it alone);
    p_none: style "none" (learn the correction on its own, used as critic).
    """

    def __init__(self, images: list[Path], train: bool = True, length: int | None = None,
                 render_size: int = 17, p_clean: float = 0.15, p_none: float = 0.25, seed: int = 0):
        self.images, self.train = images, train
        self.length = length or len(images)
        self.render_size, self.p_clean, self.p_none, self.seed = render_size, p_clean, p_none, seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        rng = random.Random() if self.train else random.Random(self.seed * 1_000_003 + i)
        img = load_rgb(self.images[i % len(self.images)])
        if self.train:
            t, l, h, w = random_crop_box(*img.shape[1:])
            img = img[:, t:t + h, l:l + w]
            if rng.random() < 0.5:
                img = img.flip(-1)
        fault = (identity_lut(self.render_size) if rng.random() < self.p_clean
                 else render_pro_look(random_fault(rng), self.render_size))
        style = 0 if rng.random() < self.p_none else rng.randrange(1, len(STYLES))
        return resize(img), fault, style


def recipe_luts(size: int = DEFAULT_SIZE, device=None) -> torch.Tensor:
    """(len(STYLES), 3, D, D, D): identity for "none", then each recipe at strength 1."""
    luts = [identity_lut(size)] + [render_pro_look(RECIPES[d], size) for d in DIRECTIONS]
    return torch.stack(luts).to(device)


def clipped(x: torch.Tensor) -> torch.Tensor:
    """Share of pixels blown to white or crushed to black, per image. (B,)
    (Same rule as grademl.surprise._clipping.)"""
    return ((x.amin(1) > 0.97) | (x.amax(1) < 0.02)).float().mean((1, 2))


def backed_off_looks(photo: torch.Tensor, style: torch.Tensor, recipes: torch.Tensor,
                     max_extra_clip: float = 0.03, backoff: float = 0.65) -> torch.Tensor:
    """(B, 3, D, D, D) look LUT per photo: the recipe, or the recipe at
    `backoff` strength where the full direction clips more than
    max_extra_clip of this photo (the net learns to hold back on photos that
    cannot take the full look)."""
    look = recipes[style]
    soft = (clipped(apply_lut(photo, look)) - clipped(photo)) > max_extra_clip
    if soft.any():
        ident = identity_lut(recipes.shape[-1], device=photo.device)
        look = torch.where(soft[:, None, None, None, None], ident + backoff * (look - ident), look)
    return look


def make_grade_batch(photo: torch.Tensor, fault: torch.Tensor, style: torch.Tensor, recipes: torch.Tensor,
                     max_extra_clip: float = 0.03, backoff: float = 0.65):
    """(input, target) on the GPU: the faulty photo, and the finished photo
    graded in the style (backed off where it would clip, backed_off_looks)."""
    size = recipes.shape[-1]
    fault = F.interpolate(fault, size=(size,) * 3, mode="trilinear", align_corners=True)
    inp = apply_lut(photo, fault).clamp(0, 1)
    return inp, apply_lut(photo, backed_off_looks(photo, style, recipes, max_extra_clip, backoff))


# ---------------------------------------------------------------- inference


@torch.no_grad()
def grade_luts(net: torch.nn.Module, img: torch.Tensor, styles: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """img (3, H, W) or (1, 3, H, W) -> (correction LUT, (S, 3, D, D, D) LUTs for `styles`).
    Intensity t for a style: scale_lut(lut, t, base=correction)."""
    dev = next(net.parameters()).device
    x = resize(img.to(dev) if img.dim() == 4 else img.to(dev)[None])
    idx = [0] + [STYLES.index(s) for s in styles]
    luts = net(x.expand(len(idx), -1, -1, -1), style_onehot(idx, dev))
    return luts[0], luts[1:]


@dataclass
class EditorWeights:
    critic: float = 1.0     # per % of mean color change the critic still wants
    fit: float = 0.5        # per unit of log scene-fit prior (grademl.surprise.direction_scores)
    bold: float = 0.35      # per % of visible change the grade makes (capped)
    bold_cap: float = 8.0
    clip: float = 0.5       # per % of extra clipped pixels
    skin: float = 1.0       # per unit of skin penalty


@dataclass
class Candidate:
    style: str              # "none" = correction only
    intensity: float
    score: float
    parts: dict[str, float]


@dataclass
class Edit:
    base: torch.Tensor            # correction LUT (3, D, D, D)
    look: torch.Tensor            # GradeNet LUT of the chosen style (correction included)
    best: Candidate               # this version's style and intensity
    shortlist: list[Candidate]    # every candidate, best first
    subject: SubjectAware | None  # settings when a person is in the frame
    reasons: list[str] = field(default_factory=list)
    rank: int = 1                 # the editor's ranking: 1 = its pick, 2, 3 = next choices

    @property
    def lut(self) -> torch.Tensor:
        """The edit as one LUT at the chosen intensity (subject-aware grading not included)."""
        return scale_lut(self.look, self.best.intensity, base=self.base)


def _skin(lab: torch.Tensor) -> torch.Tensor:
    """Boolean mask of plausible skin pixels: lab (N, 3)."""
    L, a, b = lab.unbind(-1)
    C = torch.sqrt(a**2 + b**2)
    h = torch.rad2deg(torch.atan2(b, a)) % 360
    return (h > 20) & (h < 80) & (C > 8) & (C < 50) & (L > 25) & (L < 92)


def skin_penalty(ref: torch.Tensor, cand: torch.Tensor, where: torch.Tensor) -> torch.Tensor:
    """How far candidates push skin off its corrected color. ref (3, h, w),
    cand (C, 3, h, w), where (h, w) bool = person & skin-colored. Hue drift
    beyond 6 degrees and chroma change beyond 25% count, in units of 10
    degrees / 25%. Returns (C,)."""
    if where.sum() < 20:
        return torch.zeros(len(cand), device=cand.device)
    r = rgb_to_lab(ref.permute(1, 2, 0)[where]).mean(0)
    q = rgb_to_lab(cand.permute(0, 2, 3, 1)[:, where]).mean(1)
    dh = torch.rad2deg(torch.atan2(q[:, 2], q[:, 1]) - torch.atan2(r[2], r[1])).abs()
    dh = torch.minimum(dh, 360 - dh)
    dc = (q[:, 1:].norm(dim=1) / r[1:].norm().clamp_min(1e-3) - 1).abs()
    return (dh - 6).clamp_min(0) / 10 + (dc - 0.25).clamp_min(0) / 0.25


@torch.no_grad()
def critic(net: torch.nn.Module, x: torch.Tensor, batch: int = 32) -> torch.Tensor:
    """Mean color change (in %) GradeNet ("none") would still apply to each image. (C,)"""
    out = []
    for i in range(0, len(x), batch):
        xb = x[i:i + batch]
        lut = net(xb, style_onehot([0] * len(xb), xb.device))
        out.append((apply_lut(xb, lut) - xb).abs().mean((1, 2, 3)) * 100)
    return torch.cat(out)


def pick_distinct(ranked: list[Candidate], n: int) -> list[Candidate]:
    """The editor's n best choices, by score, one per style (each at its
    best-scoring intensity)."""
    picks: list[Candidate] = []
    for c in ranked:
        if len(picks) == n:
            break
        if all(c.style != p.style for p in picks):
            picks.append(c)
    return picks


@torch.no_grad()
def top_versions(img: torch.Tensor, net: torch.nn.Module, mask: torch.Tensor | None = None, n: int = 3,
                 intensities: tuple[float, ...] = (0.8, 1.0, 1.25),
                 w: EditorWeights = EditorWeights()) -> list[Edit]:
    """The editor's n best versions of img ((3, H, W) in [0, 1]), ranked by its
    score, one per style. net: trained GradeNet; mask (1, H, W) or None."""
    dev = next(net.parameters()).device
    small = resize(img.to(dev))[None]
    base, looks = grade_luts(net, small, list(DIRECTIONS))
    corrected = apply_lut(small, base)

    specs = [("none", 1.0)] + [(d, s) for d in DIRECTIONS for s in intensities]
    luts = [base] + [scale_lut(looks[i], s, base) for i in range(len(DIRECTIONS)) for s in intensities]
    cands = apply_lut(small.expand(len(luts), -1, -1, -1), torch.stack(luts))

    crit = critic(net, cands)
    change = (cands - corrected).abs().mean((1, 2, 3)) * 100
    clip = (clipped(cands) - clipped(corrected)).clamp_min(0) * 100
    prior = direction_scores(photo_profile(corrected[0]))
    top = max(prior.values())
    person = mask is not None and coverage(mask) >= MIN_COVERAGE
    if person:
        m = F.interpolate(mask.to(dev)[None], size=small.shape[-2:], mode="bilinear")[0, 0] > 0.5
        lab = rgb_to_lab(corrected[0].permute(1, 2, 0))
        skin = skin_penalty(corrected[0], cands, m & _skin(lab.reshape(-1, 3)).reshape(m.shape))
    else:
        skin = torch.zeros(len(specs), device=dev)

    ranked = []
    for i, (d, s) in enumerate(specs):
        parts = {"critic": -w.critic * float(crit[i]),
                 "fit": 0.0 if d == "none" else w.fit * math.log(prior[d] / top),
                 "bold": w.bold * min(float(change[i]), w.bold_cap),
                 "clip": -w.clip * float(clip[i]), "skin": -w.skin * float(skin[i])}
        ranked.append(Candidate(d, s, sum(parts.values()), parts))
    ranked.sort(key=lambda c: c.score, reverse=True)

    edits = []
    for rank, pick in enumerate(pick_distinct(ranked, n), 1):
        look = base if pick.style == "none" else looks[DIRECTIONS.index(pick.style)]
        subject = None
        if person:
            subject = SubjectAware(subject_strength=0.6 if -pick.parts["skin"] > 0.5 * w.skin else 0.75)
        reasons = explain(small[0], corrected[0], pick, ranked[0], rank, subject)
        edits.append(Edit(base, look, pick, ranked, subject, reasons, rank))
    return edits


def best_version(img: torch.Tensor, net: torch.nn.Module, mask: torch.Tensor | None = None,
                 intensities: tuple[float, ...] = (0.8, 1.0, 1.25),
                 w: EditorWeights = EditorWeights()) -> Edit:
    """The editor's single pick (top_versions with n=1)."""
    return top_versions(img, net, mask, 1, intensities, w)[0]


def explain(before: torch.Tensor, after: torch.Tensor, pick: Candidate, best: Candidate, rank: int,
            subject: SubjectAware | None) -> list[str]:
    """Short reasons for the edit, in the order an editor would give them."""
    lb = rgb_to_lab(before.permute(1, 2, 0).reshape(-1, 3))
    la = rgb_to_lab(after.permute(1, 2, 0).reshape(-1, 3))
    dl = float(la[:, 0].mean() - lb[:, 0].mean())
    db = float(la[:, 2].mean() - lb[:, 2].mean())
    dsat = float(la[:, 1:].norm(dim=1).mean() / lb[:, 1:].norm(dim=1).mean().clamp_min(1e-3) - 1)
    fixes = []
    if abs(dl) > 2:
        fixes.append(f"{'brighter' if dl > 0 else 'darker'} ({dl:+.0f} L*)")
    if abs(db) > 1.5:
        fixes.append(f"{'warmer' if db > 0 else 'cooler'} white balance")
    if abs(dsat) > 0.08:
        fixes.append(f"{'more' if dsat > 0 else 'less'} saturation ({dsat:+.0%})")
    out = ["Correction: " + (", ".join(fixes) if fixes else "the photo was already well balanced")]
    if pick.style == "none":
        out.append("Grade: kept it clean; every style scored lower on this photo")
    else:
        out.append(f"Grade: {pick.style.replace('_', ' ')} at {pick.intensity:.0%}, "
                   f"edited by GradeNet for this photo ({FAMILY[pick.style]} family)")
    out.append(f"Editor's choice #{rank}" + ("" if rank == 1 else f" (score {pick.score:.2f} vs {best.score:.2f} for #1)"))
    if pick.parts["skin"] < -0.05:
        out.append("Skin: kept close to natural (candidates that shifted it were penalized)")
    if subject is not None:
        out.append(f"Person found: grade at {subject.subject_strength:.0%} on the subject, "
                   "background a bit darker and calmer")
    return out
