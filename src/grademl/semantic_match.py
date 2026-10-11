"""Content-aware reference matching: each part of the photo takes the colors
of the *same kind of thing* in the reference.

Global transfer (grademl.transfer) maps one color distribution onto
another: a reference with a big teal sky pulls the whole photo teal, and
the photo's skin may take the color of the reference's brick wall because
they share a brightness. A colorist matches like with like: skin to skin,
sky to sky, foliage to foliage, shadows of the street to shadows of the
street.

  1. DINOv2 (ViT-S/14, Apache-2.0) patch features of photo and reference.
     Self-supervised features put "sky", "skin", "leaves", "asphalt" close
     together across very different photos.
  2. The photo's patches are clustered in feature space into k regions
     (soft membership, snapped to the photo's edges with a guided filter).
  3. Every photo patch finds its counterparts in the reference (softmax over
     cosine similarity); a region's counterpart is the reference pixels its
     patches point to, weighted.
  4. Per region, a damped Lab transfer (luminance curve, per-band tint,
     chroma spread; grademl.transfer) from the region's pixels to its
     counterpart's pixels, baked into a LUT. Where the reference has no
     counterpart (low similarity: the photo has a sky, the reference does
     not) the region falls back to the global transfer.
  5. Output: k LUTs + per-pixel weights. `to_grid` fits the weights into
     the bilateral grid of grademl.local, so the app renders a semantic
     match with the same Metal kernel as GradeNet-L.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .color import lab_to_rgb, rgb_to_lab
from .local import blend_weights, grid_tv
from .lut import DEFAULT_SIZE, apply_lut
from .mask import guided_filter
from .transfer import _kmeans, transfer_lut, transfer_lut_lab

DINO = "facebook/dinov2-small"
_MODEL: dict = {}


def _dino(device):
    if device not in _MODEL:
        from transformers import AutoModel

        _MODEL[device] = AutoModel.from_pretrained(DINO).to(device).eval()
    return _MODEL[device]


@torch.no_grad()
def patch_features(img: torch.Tensor, long_side: int = 448) -> torch.Tensor:
    """(3, H, W) in [0, 1] -> (h, w, C) L2-normalized DINOv2 patch features."""
    H, W = img.shape[-2:]
    s = long_side / max(H, W)
    th, tw = max(14, round(H * s / 14) * 14), max(14, round(W * s / 14) * 14)
    x = F.interpolate(img[None], size=(th, tw), mode="bilinear", antialias=True, align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
    tokens = _dino(img.device)(pixel_values=(x - mean) / std).last_hidden_state[0, 1:]
    return F.normalize(tokens, dim=-1).reshape(th // 14, tw // 14, -1)


@dataclass
class SemanticMatch:
    luts: torch.Tensor        # (k, 3, D, D, D)
    weights: torch.Tensor     # (k, H, W) per-pixel, sums to 1
    confidence: torch.Tensor  # (k,) how well each region found a counterpart (0..1)
    coverage: torch.Tensor    # (k,) share of the photo in each region

    def render(self, img: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        """img (3, H, W) -> matched photo. strength scales away from the photo."""
        out = torch.zeros_like(img)
        w = self.weights
        if w.shape[-2:] != img.shape[-2:]:
            w = F.interpolate(w[None], size=img.shape[-2:], mode="bilinear", align_corners=False)[0]
        for k in range(len(self.luts)):
            out = out + w[k:k + 1] * apply_lut(img[None], self.luts[k])[0]
        return (img + strength * (out - img)).clamp(0, 1)


def _samples(lab: torch.Tensor, weight: torch.Tensor, n: int, g: torch.Generator) -> torch.Tensor:
    """n Lab samples (n, 3) drawn from pixels lab (N, 3) with probability ~ weight (N,)."""
    idx = torch.multinomial(weight.clamp_min(0) + 1e-8, n, replacement=True, generator=g)
    return lab[idx]


@torch.no_grad()
def semantic_match(photo: torch.Tensor, reference: torch.Tensor, k: int = 6, size: int = DEFAULT_SIZE,
                   temp: float = 0.05, cluster_temp: float = 0.08, n_samples: int = 4096,
                   min_sim: float = 0.25, full_sim: float = 0.5, region_tone: float = 0.3,
                   tone: float = 0.7, tint: float = 0.85, max_tint: float = 22.0, spread: float = 0.8,
                   global_kw: dict | None = None, seed: int = 0) -> SemanticMatch:
    """photo, reference: (3, H, W) in [0, 1] on the same device.

    temp:          softmax temperature of the patch correspondences
    cluster_temp:  softness of the region membership
    min_sim/full_sim: a region's mean best-match similarity below min_sim
                   uses the global transfer, above full_sim its own transfer
    region_tone:   how much of each region's own lightness curve to use over
                   the global one; tone stays mostly global (per-region
                   curves leave seams in smooth areas), color goes per region
    tone/tint/max_tint/spread: region transfer strength (grademl.transfer);
                   stronger than the global defaults, because matching like
                   with like is what makes a strong transfer safe
    """
    dev = photo.device
    g = torch.Generator(device=dev).manual_seed(seed)
    fp, fr = patch_features(photo), patch_features(reference)       # (hp, wp, C), (hr, wr, C)
    hp, wp, c = fp.shape
    fpf, frf = fp.reshape(-1, c), fr.reshape(-1, c)

    # 2. Regions of the photo in feature space.
    centers, mass = _kmeans(fpf[None], k)
    centers = F.normalize(centers[0], dim=-1)
    member = ((fpf @ centers.T) / cluster_temp).softmax(-1)          # (Np, k)

    # Pixel colors per patch cell (patch grid resolution x 2 for color detail).
    def cell_lab(img, h, w):
        small = F.interpolate(img[None], size=(2 * h, 2 * w), mode="area")[0]
        return rgb_to_lab(small.permute(1, 2, 0)), small         # (2h, 2w, 3)

    lab_p, _ = cell_lab(photo, hp, wp)
    lab_r, _ = cell_lab(reference, *fr.shape[:2])

    def per_pixel(v: torch.Tensor, h: int, w: int) -> torch.Tensor:  # (h*w, ...) patch values -> 2x2 pixels
        v = v.reshape(h, w, -1)
        return v.repeat_interleave(2, 0).repeat_interleave(2, 1).reshape(4 * h * w, -1)

    # 3. Correspondences: for each photo patch, a distribution over reference patches.
    sim = fpf @ frf.T                                                # (Np, Nr)
    corr = (sim / temp).softmax(-1)
    best = sim.max(-1).values                                        # (Np,)
    region_to_ref = member.T @ corr                                  # (k, Nr) reference weight per region
    conf = (member * best[:, None]).sum(0) / member.sum(0).clamp_min(1e-6)
    conf = ((conf - min_sim) / (full_sim - min_sim)).clamp(0, 1)     # (k,)

    # 4. Per-region transfer from the region's pixels to its counterpart's.
    lp = lab_p.reshape(-1, 3)
    lr = lab_r.reshape(-1, 3)
    w_p = per_pixel(member, hp, wp)                                  # (4Np, k)
    w_r = per_pixel(region_to_ref.T, *fr.shape[:2])                  # (4Nr, k)
    src = torch.stack([_samples(lp, w_p[:, j], n_samples, g) for j in range(k)])
    dst = torch.stack([_samples(lr, w_r[:, j], n_samples, g) for j in range(k)])
    region = transfer_lut_lab(src, dst, size, tone=tone, tint=tint, max_tint=max_tint, spread=spread)
    glob = transfer_lut(photo[None], reference[None], size, **(global_kw or {}))[0]
    region = _mix_tone(region, glob, region_tone)
    luts = glob + conf.view(-1, 1, 1, 1, 1) * (region - glob)

    # Region weights at full resolution, snapped to the photo's edges.
    H, W = photo.shape[-2:]
    m = member.T.reshape(1, k, hp, wp)
    m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
    guide = (photo * torch.tensor([0.2126, 0.7152, 0.0722], device=dev).view(3, 1, 1)).sum(0)[None, None]
    r = max(2, round(0.02 * max(H, W)))
    m = torch.cat([guided_filter(m[:, j:j + 1], guide, r, 1e-3) for j in range(k)], 1).clamp_min(0)
    weights = (m / m.sum(1, keepdim=True).clamp_min(1e-6))[0]
    return SemanticMatch(luts, weights, conf, weights.mean((1, 2)))


def _mix_tone(region: torch.Tensor, glob: torch.Tensor, t: float) -> torch.Tensor:
    """Region LUTs (k, 3, D, D, D) with L* = glob + t * (region - glob) and the region's a*b*."""
    k, _, d = region.shape[:3]
    lr = rgb_to_lab(region.reshape(k, 3, -1).transpose(1, 2))
    lg = rgb_to_lab(glob.reshape(3, -1).T)[None]
    lr[..., 0] = lg[..., 0] + t * (lr[..., 0] - lg[..., 0])
    return lab_to_rgb(lr).transpose(1, 2).reshape(region.shape).contiguous()


def to_grid(match: SemanticMatch, photo: torch.Tensor, grid_bins: int = 8, grid_size: int = 16,
            steps: int = 300, size: int = 256) -> torch.Tensor:
    """Fit a bilateral grid (k, G, gs, gs) of logits whose sliced softmax
    reproduces match.weights on this photo, so the app can render the match
    with the GradeNet-L kernel (grademl.local.apply_local)."""
    img = F.interpolate(photo[None], size=(size, size), mode="area")
    target = F.interpolate(match.weights[None], size=(size, size), mode="area")
    k = target.shape[1]
    init = F.adaptive_avg_pool2d(target, grid_size).clamp_min(1e-4).log()
    grid = init[:, :, None].repeat(1, 1, grid_bins, 1, 1).clone().requires_grad_(True)
    opt = torch.optim.Adam([grid], lr=0.1)
    with torch.enable_grad():
        for _ in range(steps):
            w = blend_weights(grid, img)
            loss = (w - target).abs().mean() + 1e-3 * grid_tv(grid)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return grid.detach()[0]


# ---------------------------------------------------------------- batched teacher


@torch.no_grad()
def patch_features_batch(img: torch.Tensor, side: int = 224) -> torch.Tensor:
    """(B, 3, H, W) -> (B, n, n, C) normalized DINOv2 features of square
    `side` copies (squashed, like the models' inputs)."""
    x = F.interpolate(img, size=(side, side), mode="bilinear", antialias=True, align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
    tokens = _dino(img.device)(pixel_values=(x - mean) / std).last_hidden_state[:, 1:]
    n = side // 14
    return F.normalize(tokens, dim=-1).reshape(img.shape[0], n, n, -1)


@torch.no_grad()
def semantic_match_batch(photo: torch.Tensor, reference: torch.Tensor, k: int = 6, size: int = DEFAULT_SIZE,
                         temp: float = 0.05, cluster_temp: float = 0.08, n_samples: int = 2048,
                         min_sim: float = 0.25, full_sim: float = 0.5, region_tone: float = 0.3,
                         tone: float = 0.7, tint: float = 0.85, max_tint: float = 22.0, spread: float = 0.8,
                         seed: int | None = None) -> torch.Tensor:
    """semantic_match for a batch of (photo, reference) pairs at the models'
    input size (B, 3, S, S), returning the matched photos (B, 3, S, S): the
    teacher for MatchNet-L. Same steps as semantic_match, batched."""
    b, _, s, _ = photo.shape
    dev = photo.device
    g = torch.Generator(device=dev)
    if seed is not None:
        g.manual_seed(seed)
    else:
        g.seed()
    fp, fr = patch_features_batch(photo), patch_features_batch(reference)  # (B, n, n, C)
    n, c = fp.shape[1], fp.shape[-1]
    fpf, frf = fp.reshape(b, -1, c), fr.reshape(b, -1, c)

    centers, _ = _kmeans(fpf, k)
    centers = F.normalize(centers, dim=-1)
    member = (fpf @ centers.transpose(1, 2) / cluster_temp).softmax(-1)          # (B, N, k)
    sim = fpf @ frf.transpose(1, 2)                                              # (B, N, N)
    corr = (sim / temp).softmax(-1)
    best = sim.max(-1).values
    region_to_ref = member.transpose(1, 2) @ corr                                # (B, k, N)
    conf = (member * best[..., None]).sum(1) / member.sum(1).clamp_min(1e-6)
    conf = ((conf - min_sim) / (full_sim - min_sim)).clamp(0, 1)                 # (B, k)

    def cells(img):  # (B, 4N, 3) Lab at 2x the patch grid
        small = F.interpolate(img, size=(2 * n, 2 * n), mode="area")
        return rgb_to_lab(small.permute(0, 2, 3, 1)).reshape(b, -1, 3)

    def per_pixel(v):  # (B, N, k) -> (B, 4N, k)
        v = v.reshape(b, n, n, -1)
        return v.repeat_interleave(2, 1).repeat_interleave(2, 2).reshape(b, 4 * n * n, -1)

    lab_p, lab_r = cells(photo), cells(reference)
    w_p = per_pixel(member).transpose(1, 2).reshape(b * k, -1)                   # (B*k, 4N)
    w_r = per_pixel(region_to_ref.transpose(1, 2)).transpose(1, 2).reshape(b * k, -1)
    ip = torch.multinomial(w_p.clamp_min(0) + 1e-8, n_samples, replacement=True, generator=g)
    ir = torch.multinomial(w_r.clamp_min(0) + 1e-8, n_samples, replacement=True, generator=g)
    src = lab_p.repeat_interleave(k, 0).gather(1, ip[..., None].expand(-1, -1, 3))
    dst = lab_r.repeat_interleave(k, 0).gather(1, ir[..., None].expand(-1, -1, 3))
    region = transfer_lut_lab(src, dst, size, tone=tone, tint=tint, max_tint=max_tint, spread=spread)
    region = region.reshape(b, k, 3, size, size, size)
    glob = transfer_lut(photo, reference, size)                                   # (B, 3, D, D, D)
    region = torch.stack([_mix_tone(region[i], glob[i], region_tone) for i in range(b)])
    luts = glob[:, None] + conf.view(b, k, 1, 1, 1, 1) * (region - glob[:, None])

    m = F.interpolate(member.transpose(1, 2).reshape(b, k, n, n), size=(s, s), mode="bilinear",
                      align_corners=False)
    guide = (photo * torch.tensor([0.2126, 0.7152, 0.0722], device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True)
    r = max(2, round(0.02 * s))
    m = torch.cat([guided_filter(m[:, j:j + 1], guide, r, 1e-3) for j in range(k)], 1).clamp_min(0)
    w = m / m.sum(1, keepdim=True).clamp_min(1e-6)
    out = torch.zeros_like(photo)
    for j in range(k):
        out = out + w[:, j:j + 1] * apply_lut(photo, luts[:, j])
    return out.clamp(0, 1)
