"""The AI editor on GradeNet-L (grademl.local): every style, judged, top 3.

Same idea as grademl.grade.top_versions, for the local model: GradeNet-L
edits the photo in every direction (correction + look + local finishing:
sky, subject, depth, vignette), at a few intensities; then the same network
with style "none" judges each candidate (how much correcting it would
still want: less = reads as finished), plus the bonus for a visible grade,
the penalties for clipping and skin pushed off its color, and the scene-fit
prior. The ranking and the reasons are shared with grade.py.

Intensity for a local edit is a blend in image space between the
correction-only edit ("none") and the styled edit, so 0 keeps the
correction and the local balancing, 1 is the full edit.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from .color import rgb_to_lab
from .data import resize
from .grade import (STYLES, Candidate, EditorWeights, _skin, clipped, explain, pick_distinct, skin_penalty,
                    style_onehot)
from .local import LocalGradeNet, apply_local
from .mask import MIN_COVERAGE, coverage
from .pro_looks import DIRECTIONS
from .surprise import FAMILY, direction_scores, photo_profile


@dataclass
class LocalEdit:
    """A GradeNet-L edit: LUTs + grids for "none" and the chosen style."""
    style: str
    intensity: float
    luts: torch.Tensor        # (K, 3, D, D, D) styled
    grid: torch.Tensor        # (K, G, gh, gw)
    base_luts: torch.Tensor   # correction only ("none")
    base_grid: torch.Tensor
    best: Candidate | None = None
    reasons: list[str] = field(default_factory=list)
    rank: int = 1
    shortlist: list[Candidate] = field(default_factory=list)   # every candidate, best first

    def render(self, img: torch.Tensor, intensity: float | None = None) -> torch.Tensor:
        """img (3, H, W) at any resolution -> edited image."""
        t = self.intensity if intensity is None else intensity
        x = img[None]
        styled = apply_local(x, self.luts[None], self.grid[None])
        if self.style == "none" or t == 1:
            return styled[0].clamp(0, 1)
        base = apply_local(x, self.base_luts[None], self.base_grid[None])
        return (base + t * (styled - base))[0].clamp(0, 1)


@torch.no_grad()
def local_grade(net: LocalGradeNet, img: torch.Tensor, styles: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """img (3, H, W) -> (luts (S, K, 3, D, D, D), grids (S, K, G, gh, gw)) for `styles`."""
    dev = next(net.parameters()).device
    x = resize(img.to(dev))[None].expand(len(styles), -1, -1, -1)
    return net(x, style_onehot([STYLES.index(s) for s in styles], dev))


@torch.no_grad()
def local_critic(net: LocalGradeNet, x: torch.Tensor, batch: int = 32) -> torch.Tensor:
    """Mean color change (in %) GradeNet-L ("none") would still apply. x (C, 3, 256, 256) -> (C,)"""
    out = []
    for i in range(0, len(x), batch):
        xb = x[i:i + batch]
        luts, grid = net(xb, style_onehot([0] * len(xb), xb.device))
        out.append((apply_local(xb, luts, grid) - xb).abs().mean((1, 2, 3)) * 100)
    return torch.cat(out)


@torch.no_grad()
def local_edit(net: LocalGradeNet, img: torch.Tensor, style: str, intensity: float = 1.0) -> LocalEdit:
    """A single edit in a given style."""
    luts, grids = local_grade(net, img, ["none", style])
    return LocalEdit(style, intensity, luts[1], grids[1], luts[0], grids[0])


CALIBRATION = Path(__file__).resolve().parents[2] / "configs" / "editor_calibration.json"
CALIBRATED = ("critic", "bold")   # parts whose typical level differs by look, not by photo
FAMILIES = {**FAMILY, "none": "natural"}


@dataclass
class EditorWeightsL(EditorWeights):
    """EditorWeights plus the color-keeping penalty of the local editor."""
    keep: float = 3.0          # per unit of chroma lost (1 - chroma ratio), on colorful photos only
    keep_from: float = 0.25    # colorfulness (mean C*/40) where the penalty starts
    keep_full: float = 0.5     # ... and where it reaches full weight


def load_calibration(calibration) -> dict | None:
    """"default" -> configs/editor_calibration.json if present, a dict as is, None -> raw scores."""
    if calibration == "default":
        return json.loads(CALIBRATION.read_text())["stats"] if CALIBRATION.exists() else None
    return calibration


@torch.no_grad()
def score_candidates(img: torch.Tensor, net: LocalGradeNet, mask: torch.Tensor | None = None,
                     intensities: tuple[float, ...] = (0.6, 0.85, 1.0),
                     w: EditorWeightsL = EditorWeightsL(), calibration="default"):
    """Every (look, intensity) candidate of img, scored and ranked (best first).

    Score parts:
      critic  GradeNet-L ("none") on the candidate: how much it would still
              change (less = reads as finished)
      bold    a visible grade is rewarded (capped)
      fit     scene-fit prior (grademl.surprise.direction_scores)
      clip    extra blown / crushed pixels
      skin    skin pushed off its color (with a person mask)
      keep    color drained from a colorful photo (B&W of a flower field)
    critic and bold are calibrated per look (calibration: per-look mean over
    held-out photos, scripts/editor_report.py --calibrate): a look that
    removes what the critic would correct (B&W has no color left to fix)
    or always changes a lot is otherwise favored on every photo; after
    calibration each look is judged by how this photo suits it compared
    with how it usually scores.

    Returns (ranked candidates, context dict for building edits).
    """
    dev = next(net.parameters()).device
    small = resize(img.to(dev))[None]
    names = ["none", *DIRECTIONS]
    luts, grids = local_grade(net, img, names)
    renders = apply_local(small.expand(len(names), -1, -1, -1), luts, grids)   # (S, 3, 256, 256)
    corrected = renders[:1]

    specs = [("none", 1.0)] + [(d, s) for d in DIRECTIONS for s in intensities]
    cands = [corrected[0]] + [(corrected[0] + s * (renders[1 + i] - corrected[0])).clamp(0, 1)
                              for i in range(len(DIRECTIONS)) for s in intensities]
    cands = torch.stack(cands)

    crit = local_critic(net, cands)
    change = (cands - corrected).abs().mean((1, 2, 3)) * 100
    clip = (clipped(cands) - clipped(corrected)).clamp_min(0) * 100
    profile = photo_profile(corrected[0])
    prior = direction_scores(profile)
    top = max(prior.values())
    chroma = rgb_to_lab(cands.permute(0, 2, 3, 1))[..., 1:].norm(dim=-1).mean((1, 2))
    ratio = (chroma / chroma[0].clamp_min(1e-3)).clamp(max=1)
    colorful = min(1.0, max(0.0, (profile.colorfulness - w.keep_from) / (w.keep_full - w.keep_from)))
    person = mask is not None and coverage(mask) >= MIN_COVERAGE
    if person:
        m = F.interpolate(mask.to(dev)[None], size=small.shape[-2:], mode="bilinear")[0, 0] > 0.5
        lab = rgb_to_lab(corrected[0].permute(1, 2, 0))
        skin = skin_penalty(corrected[0], cands, m & _skin(lab.reshape(-1, 3)).reshape(m.shape))
    else:
        skin = torch.zeros(len(specs), device=dev)

    cal = load_calibration(calibration)
    if cal:
        grand = {k: sum(cal[s][k][0] for s in cal) / len(cal) for k in CALIBRATED}
    ranked = []
    for i, (d, s) in enumerate(specs):
        parts = {"critic": -w.critic * float(crit[i]),
                 "fit": 0.0 if d == "none" else w.fit * math.log(prior[d] / top),
                 "bold": w.bold * min(float(change[i]), w.bold_cap),
                 "clip": -w.clip * float(clip[i]), "skin": -w.skin * float(skin[i]),
                 "keep": -w.keep * colorful * float(1 - ratio[i])}
        if cal and d in cal:
            for k in CALIBRATED:
                parts[k] = parts[k] - cal[d][k][0] + grand[k]
        ranked.append(Candidate(d, s, sum(parts.values()), parts))
    ranked.sort(key=lambda c: c.score, reverse=True)
    return ranked, {"names": names, "luts": luts, "grids": grids, "small": small, "corrected": corrected}


def pick_families(ranked: list[Candidate], n: int) -> list[Candidate]:
    """The n best candidates from n different look families (warm, film,
    cool, mono, stylized, soft, natural), so the alternatives really differ."""
    picks: list[Candidate] = []
    for c in ranked:
        if len(picks) == n:
            break
        if all(FAMILIES[c.style] != FAMILIES[p.style] for p in picks):
            picks.append(c)
    return picks


@torch.no_grad()
def local_top_versions(img: torch.Tensor, net: LocalGradeNet, mask: torch.Tensor | None = None, n: int = 3,
                       intensities: tuple[float, ...] = (0.6, 0.85, 1.0),
                       w: EditorWeightsL = EditorWeightsL(), calibration="default") -> list[LocalEdit]:
    """The editor's n best edits of img ((3, H, W) in [0, 1]), from n different
    look families. Intensities stop at 1: a GradeNet-L edit at 1 is the look as
    designed, finishing included; past it the "bold" bonus rewarded
    exaggeration (on real photos the v1 editor picked 125% most of the time)."""
    ranked, ctx = score_candidates(img, net, mask, intensities, w, calibration)
    names, luts, grids, small, corrected = (ctx[k] for k in ("names", "luts", "grids", "small", "corrected"))
    edits = []
    for rank, pick in enumerate(pick_families(ranked, n), 1):
        j = names.index(pick.style)
        reasons = explain(small[0], corrected[0], pick, ranked[0], rank, None)
        reasons[1] = reasons[1].replace("edited by GradeNet", "edited by GradeNet-L")
        reasons.append("Local work: sky, subject, depth and edges handled separately (learned, no masks)")
        if pick.parts["keep"] < -0.3:
            reasons.append("Color: this look drains some of the photo's color (kept lower in the ranking)")
        edits.append(LocalEdit(pick.style, pick.intensity, luts[j], grids[j], luts[0], grids[0],
                               pick, reasons, rank, ranked))
    return edits
