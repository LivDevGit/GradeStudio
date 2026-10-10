"""Classic reference color transfer, baked into a 3D LUT ("teacher").

Used as the training target for the model's "full match" head, i.e. what
"make my photo's colors look like this reference" means when the
reference is any photo (not a graded version of some known base).

Per (content, reference) pair, in CIELAB:
  1. Luminance: histogram (quantile) matching of L*, with the curve's slope
     clamped so it can never posterize or invert tones.
  2. Chroma: per tonal band (shadows / mids / highlights) mean a*b* shift,
     so split toning transfers (teal shadows, warm highlights)...
  3. ...plus a 2x2 Monge-Kantorovich linear map on the band-centered a*b*
     (Pitie & Kokaram 2007), which transfers saturation and hue spread.
     Its eigenvalues are clamped so near-gray inputs don't explode.
The resulting color -> color function is evaluated on the LUT grid, so it
runs on device as an ordinary LUT. Everything is batched (B pairs at once).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .color import lab_to_rgb, rgb_to_lab
from .lut import DEFAULT_SIZE, identity_lut

L_BINS = 101                       # luminance curve resolution (L* 0..100)
BAND_CENTERS = (15.0, 50.0, 85.0)  # shadows, mids, highlights (L*)
BAND_WIDTH = 20.0


def _pixels_lab(img: torch.Tensor, size: int = 64) -> torch.Tensor:
    """(B, 3, H, W) -> (B, size*size, 3) Lab of a downscaled copy."""
    small = F.interpolate(img, size=(size, size), mode="area")
    return rgb_to_lab(small.permute(0, 2, 3, 1).reshape(img.shape[0], -1, 3))


def luminance_curve(lc: torch.Tensor, lr: torch.Tensor, min_slope: float = 0.3,
                    max_slope: float = 3.0, smooth: int = 5) -> torch.Tensor:
    """Quantile-matching curve L_content -> L_reference sampled at L* = 0..100.

    lc, lr: (B, N) luminance samples. Returns (B, L_BINS).
    """
    grid = torch.linspace(0, 100, L_BINS, device=lc.device).expand(lc.shape[0], -1).contiguous()
    lc_sorted, lr_sorted = lc.sort(1).values, lr.sort(1).values
    n = lc.shape[1]
    # Percentile of each grid value within the content, then the reference
    # luminance at that same percentile (linear interpolation).
    pct = torch.searchsorted(lc_sorted.contiguous(), grid).float().clamp(0, n - 1) / (n - 1)
    pos = pct * (n - 1)
    lo = pos.floor().long().clamp(0, n - 1)
    hi = (lo + 1).clamp(max=n - 1)
    frac = pos - lo
    curve = lr_sorted.gather(1, lo) * (1 - frac) + lr_sorted.gather(1, hi) * frac

    # Outside the content's luminance range the quantile curve is flat (it
    # just repeats the reference min / max). Extrapolate with slope 1
    # instead, so tones the photo doesn't have yet stay separated.
    c_min, c_max = lc_sorted[:, :1], lc_sorted[:, -1:]
    curve = torch.where(grid < c_min, lr_sorted[:, :1] - (c_min - grid), curve)
    curve = torch.where(grid > c_max, lr_sorted[:, -1:] + (grid - c_max), curve)

    # Smooth, then rebuild from clamped slopes: monotone, no flat or
    # cliff-like segments (flat = posterization, cliffs = noise blow-up).
    k = torch.ones(1, 1, smooth, device=lc.device) / smooth
    smoothed = F.conv1d(F.pad(curve[:, None], (smooth // 2, smooth // 2), mode="replicate"), k)[:, 0]
    step = 100 / (L_BINS - 1)
    d = smoothed.diff(dim=1).clamp(min_slope * step, max_slope * step)
    rebuilt = torch.cat([torch.zeros_like(d[:, :1]), d.cumsum(1)], 1)
    # Anchor the rebuilt curve where the photo actually has pixels (its
    # luminance histogram), not at L*=0: otherwise clamping in an empty
    # range would shift every tone the photo does have.
    hist = torch.zeros_like(grid).scatter_add_(
        1, (lc.clamp(0, 100) / step).round().long(), torch.ones_like(lc))
    offset = ((smoothed - rebuilt) * hist).sum(1, keepdim=True) / hist.sum(1, keepdim=True)
    return (rebuilt + offset).clamp(0, 100)


def _band_weights(l: torch.Tensor) -> torch.Tensor:
    """(...,) L* -> (..., 3) normalized soft membership in shadows/mids/highlights."""
    c = torch.tensor(BAND_CENTERS, device=l.device)
    w = torch.exp(-0.5 * ((l[..., None] - c) / BAND_WIDTH) ** 2)
    return w / w.sum(-1, keepdim=True)


def _band_means(lab: torch.Tensor) -> torch.Tensor:
    """(B, N, 3) -> (B, 3 bands, 2) mean a*b* per tonal band."""
    w = _band_weights(lab[..., 0])                                  # (B, N, K)
    return torch.einsum("bnk,bnc->bkc", w, lab[..., 1:]) / (w.sum(1)[..., None] + 1e-6)


def _sqrtm_sym(m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched matrix square root and inverse square root of SPD (B, 2, 2)."""
    vals, vecs = torch.linalg.eigh(m)
    vals = vals.clamp_min(1e-6)
    s = vecs @ torch.diag_embed(vals.sqrt()) @ vecs.transpose(1, 2)
    si = vecs @ torch.diag_embed(vals.rsqrt()) @ vecs.transpose(1, 2)
    return s, si


def chroma_map(lab_c: torch.Tensor, lab_r: torch.Tensor, reg: float = 9.0,
               min_gain: float = 0.05, max_gain: float = 2.5):
    """Band means (B,3,2) for content/reference and the MK matrix (B,2,2)."""
    mc, mr = _band_means(lab_c), _band_means(lab_r)
    # Residual chroma after removing the per-band mean.
    rc = lab_c[..., 1:] - _band_weights(lab_c[..., 0]) @ mc
    rr = lab_r[..., 1:] - _band_weights(lab_r[..., 0]) @ mr
    eye = torch.eye(2, device=lab_c.device) * reg
    cov_c = rc.transpose(1, 2) @ rc / rc.shape[1] + eye
    cov_r = rr.transpose(1, 2) @ rr / rr.shape[1] + eye
    sc, sci = _sqrtm_sym(cov_c)
    mid, _ = _sqrtm_sym(sc @ cov_r @ sc)
    t = sci @ mid @ sci
    # T is symmetric PD; clamp its gains (eigenvalues).
    vals, vecs = torch.linalg.eigh(0.5 * (t + t.transpose(1, 2)))
    t = vecs @ torch.diag_embed(vals.clamp(min_gain, max_gain)) @ vecs.transpose(1, 2)
    return mc, mr, t


@torch.no_grad()
def transfer_lut(content: torch.Tensor, reference: torch.Tensor, size: int = DEFAULT_SIZE,
                 tone: float = 0.6, tint: float = 0.6, max_tint: float = 15.0,
                 spread: float = 0.7) -> torch.Tensor:
    """(B, 3, H, W) x2 in [0, 1] -> (B, 3, size, size, size) LUT that maps the
    content's colors toward the reference's color distribution.

    A full transfer copies the *scene's* colors too (a reference with a big
    cyan backdrop turns everything cyan), so each part is applied partially:
      tone:     fraction of the luminance curve
      tint:     fraction of the per-band mean a*b* shift, capped at max_tint
                (a*b* units) per band
      spread:   fraction of the chroma (saturation / hue spread) transform
    """
    b = content.shape[0]
    lab_c, lab_r = _pixels_lab(content), _pixels_lab(reference)
    curve = luminance_curve(lab_c[..., 0], lab_r[..., 0])          # (B, L_BINS)
    mc, mr, t = chroma_map(lab_c, lab_r)
    shift = (mr - mc) * tint                                        # (B, bands, 2)
    norm = shift.norm(dim=-1, keepdim=True)
    shift = shift * (max_tint / norm.clamp_min(max_tint))           # cap the magnitude
    eye = torch.eye(2, device=content.device)
    t = eye + spread * (t - eye)

    grid = identity_lut(size, device=content.device).reshape(3, -1).T   # (N, 3) RGB
    lab = rgb_to_lab(grid).expand(b, -1, -1)                            # (B, N, 3)
    l_in = lab[..., 0]
    # Piecewise-linear lookup of the luminance curve.
    pos = l_in.clamp(0, 100) / 100 * (L_BINS - 1)
    lo = pos.floor().long().clamp(max=L_BINS - 2)
    frac = pos - lo
    l_full = curve.gather(1, lo) * (1 - frac) + curve.gather(1, lo + 1) * frac
    l_out = l_in + tone * (l_full - l_in)

    w = _band_weights(l_in)
    content_tint = w @ mc
    ab_out = (lab[..., 1:] - content_tint) @ t.transpose(1, 2) + content_tint + w @ shift
    out = lab_to_rgb(torch.cat([l_out[..., None], ab_out], -1))  # (B, N, 3)
    return out.permute(0, 2, 1).reshape(b, 3, size, size, size).contiguous()
