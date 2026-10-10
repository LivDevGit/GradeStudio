"""LUT-predicting networks.

Both models follow Zeng et al. (2020): a small CNN looks at a downscaled
image and predicts weights that blend a set of learned basis LUTs into one
image-adaptive LUT. The network only outputs the LUT; applying it to the
full-resolution photo happens outside the model (apply_lut / Metal).

- AdaptiveLUTNet: content only (stage 1, FiveK auto-enhance).
- MatchLUTNet: content + reference (stage 2, "match the look").
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lut import DEFAULT_SIZE, identity_lut

INPUT_SIZE = 256  # models expect inputs already resized to this


def _block(cin: int, cout: int) -> list[nn.Module]:
    return [
        nn.Conv2d(cin, cout, 3, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.InstanceNorm2d(cout, affine=True),
    ]


class Encoder(nn.Module):
    """256x256 RGB -> feature vector. ~270K params."""

    def __init__(self, width: int = 16, out_dim: int = 128):
        super().__init__()
        w = width
        self.body = nn.Sequential(
            nn.Conv2d(3, w, 3, stride=2, padding=1), nn.LeakyReLU(0.2),  # 128
            *_block(w, 2 * w),          # 64
            *_block(2 * w, 4 * w),      # 32
            *_block(4 * w, 8 * w),      # 16
            *_block(8 * w, out_dim),    # 8
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global mean instead of AdaptiveAvgPool: fixed input size, and it
        # converts to a plain reduce_mean in Core ML.
        return self.body(x).mean(dim=(2, 3))


class LUTBank(nn.Module):
    """N learnable residual basis LUTs: output = identity + sum_i w_i * basis_i.

    The bases start at zero, so any untrained network returns the input
    image unchanged. The predicted weights must NOT start at zero too: the
    gradient of basis_i is proportional to w_i and the gradient of w_i to
    basis_i, so zero-initializing both would freeze them at zero forever
    (see _head).
    """

    def __init__(self, n_luts: int = 3, size: int = DEFAULT_SIZE):
        super().__init__()
        self.basis = nn.Parameter(torch.zeros(n_luts, 3, size, size, size))
        self.register_buffer("identity", identity_lut(size).flatten())

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        # (B, N) x (N, 3*D^3) -> (B, 3, D, D, D); matmul instead of einsum for
        # a simpler Core ML graph.
        n, c, d, h, w = self.basis.shape
        flat = self.identity + weights @ self.basis.reshape(n, -1)
        return flat.reshape(-1, c, d, h, w)


def _head(in_dim: int, n_luts: int) -> nn.Linear:
    # Default (random) weights so every basis LUT receives gradient from the
    # first step; zero bias. Output is still the identity LUT at init because
    # the bases are zero.
    head = nn.Linear(in_dim, n_luts)
    nn.init.zeros_(head.bias)
    return head


class AdaptiveLUTNet(nn.Module):
    """Stage 1: image -> LUT that enhances it."""

    def __init__(self, n_luts: int = 3, lut_size: int = DEFAULT_SIZE, width: int = 16):
        super().__init__()
        self.encoder = Encoder(width)
        self.dropout = nn.Dropout(0.5)
        self.head = _head(self.encoder.out_dim, n_luts)
        self.luts = LUTBank(n_luts, lut_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        weights = self.head(self.dropout(self.encoder(image)))
        return self.luts(weights)


class ColorStats(nn.Module):
    """Explicit global color statistics of an image.

    The CNN encoder uses instance norm, which strips per-channel means, i.e.
    exactly the color casts that define a look. This branch hands them to
    the head directly:
      - soft histograms of R, G, B and luminance (bins x 4)
      - mean chroma (RGB minus luminance) in luminance bands from shadows to
        highlights, plus how much of the image falls in each band. This is
        what captures split toning ("teal shadows, orange highlights").
      - optionally (bins3d > 0) a soft joint RGB histogram with bins3d^3
        cells: the full color distribution, so the head can compare which
        colors the reference has against which the photo has.
    Only elementwise ops, means and rank <= 4 tensors (Core ML's limit is 5
    for most ops), so it converts cleanly.
    """

    def __init__(self, bins: int = 16, bands: int = 5, bins3d: int = 0):
        super().__init__()
        self.register_buffer("hist_c", torch.linspace(0, 1, bins).view(1, 1, bins, 1, 1))
        self.register_buffer("band_c", torch.linspace(0, 1, bands).view(1, bands, 1, 1))
        self.register_buffer("luma_w", torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1))
        self.hist_k = 0.5 * (bins - 1) ** 2   # gaussian width ~ one bin
        self.band_k = 0.5 * (bands - 1) ** 2
        self.bins3d = bins3d
        if bins3d:
            self.register_buffer("c3d", torch.linspace(0, 1, bins3d).view(1, bins3d, 1))
            self.k3d = 0.5 * (bins3d - 1) ** 2
        self.out_dim = 4 * bins + bands * 4 + bins3d**3

    def joint_hist(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, h, w) -> (B, bins3d^3) soft joint RGB histogram, sqrt-scaled."""
        # Reshape with -1, not h*w: arithmetic on traced shapes becomes an
        # `int` op that coremltools cannot convert.
        b, k = x.shape[0], self.bins3d
        flat = x.reshape(b, 3, 1, -1)
        a = torch.exp(-self.k3d * (flat - self.c3d.unsqueeze(1)) ** 2)  # (B, 3, k, N) per-axis weights
        rg = (a[:, 0].unsqueeze(2) * a[:, 1].unsqueeze(1)).reshape(b, k * k, 1, -1)
        rgb = (rg * a[:, 2].unsqueeze(1)).reshape(b, k**3, -1)           # (B, k^3, N)
        hist = rgb.mean(2)
        hist = hist / (hist.sum(1, keepdim=True) + 1e-6)
        return torch.sqrt(hist * k**3 + 1e-6)  # ~1 for a uniform distribution

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.avg_pool2d(x, 4)  # 256 -> 64: plenty for global statistics
        luma = (x * self.luma_w).sum(1, keepdim=True)
        chans = torch.cat([x, luma], 1).unsqueeze(2)                     # (B, 4, 1, h, w)
        hist = torch.exp(-self.hist_k * (chans - self.hist_c) ** 2).mean((3, 4))
        hist = hist / (hist.sum(2, keepdim=True) + 1e-6)                 # (B, 4, bins)

        w = torch.exp(-self.band_k * (luma - self.band_c) ** 2)          # (B, bands, h, w)
        mass = w.mean((2, 3))                                            # (B, bands)
        chroma = x - luma                                                # (B, 3, h, w)
        band_chroma = (w.unsqueeze(2) * chroma.unsqueeze(1)).mean((3, 4)) / (mass.unsqueeze(2) + 1e-4)
        b = x.shape[0]
        # Chroma is small (~0.05); scale it up so it is not drowned out.
        feats = [hist.reshape(b, -1), mass, 10 * band_chroma.reshape(b, -1)]
        if self.bins3d:
            feats.append(self.joint_hist(F.avg_pool2d(x, 2)))  # 32x32 is enough for the 3D histogram
        return torch.cat(feats, 1)


class MatchLUTNet(nn.Module):
    """Stage 2: (content, reference) -> LUT that gives content the reference's look.

    A shared CNN encoder (scene content) and ColorStats (color distribution)
    embed both images; the head sees both embeddings and their difference,
    since the LUT has to map one color distribution onto the other.
    """

    def __init__(self, n_luts: int = 24, lut_size: int = DEFAULT_SIZE, width: int = 16, hidden: int = 512,
                 bins3d: int = 0, heads: int = 1):
        super().__init__()
        self.encoder = Encoder(width)
        self.stats = ColorStats(bins3d=bins3d)
        d = self.encoder.out_dim + self.stats.out_dim
        self.mlp = nn.Sequential(
            nn.Linear(3 * d, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2), nn.Dropout(0.1),
        )
        self.head = _head(hidden, n_luts)
        # Optional second head sharing trunk and LUT bases:
        #   head       -> "look": only the grade applied to the reference
        #   match_head -> "match": also pull the photo's colors toward the
        #                 reference's (trained on grademl.transfer targets)
        self.match_head = _head(hidden, n_luts) if heads == 2 else None
        self.luts = LUTBank(n_luts, lut_size)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.encoder(x), self.stats(x)], 1)

    def forward(self, content: torch.Tensor, reference: torch.Tensor):
        """Returns the look LUT, or (look LUT, match LUT) with heads=2."""
        fc, fr = self.embed(content), self.embed(reference)
        h = self.mlp(torch.cat([fc, fr, fr - fc], dim=1))
        look = self.luts(self.head(h))
        if self.match_head is None:
            return look
        return look, self.luts(self.match_head(h))
