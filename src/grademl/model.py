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
    """N learnable basis LUTs; output = sum_i w_i * basis_i.

    Basis 0 starts as identity and the rest as zero, so a network that
    predicts w = [1, 0, ..., 0] returns the input image unchanged.
    """

    def __init__(self, n_luts: int = 3, size: int = DEFAULT_SIZE):
        super().__init__()
        basis = torch.zeros(n_luts, 3, size, size, size)
        basis[0] = identity_lut(size)
        self.basis = nn.Parameter(basis)

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        # (B, N) x (N, 3*D^3) -> (B, 3, D, D, D); matmul instead of einsum for
        # a simpler Core ML graph.
        n, c, d, h, w = self.basis.shape
        flat = weights @ self.basis.reshape(n, -1)
        return flat.reshape(-1, c, d, h, w)


def _head(in_dim: int, n_luts: int) -> nn.Linear:
    head = nn.Linear(in_dim, n_luts)
    # Start at the identity LUT: w = [1, 0, ...] regardless of features.
    nn.init.zeros_(head.weight)
    with torch.no_grad():
        head.bias.zero_()
        head.bias[0] = 1.0
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


class MatchLUTNet(nn.Module):
    """Stage 2: (content, reference) -> LUT that gives content the reference's look.

    One shared encoder embeds both images; the head sees both embeddings and
    their difference, since the LUT has to map one color distribution onto
    the other.
    """

    def __init__(self, n_luts: int = 8, lut_size: int = DEFAULT_SIZE, width: int = 16):
        super().__init__()
        self.encoder = Encoder(width)
        d = self.encoder.out_dim
        self.mlp = nn.Sequential(nn.Linear(3 * d, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.head = _head(256, n_luts)
        self.luts = LUTBank(n_luts, lut_size)

    def forward(self, content: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        fc, fr = self.encoder(content), self.encoder(reference)
        weights = self.head(self.mlp(torch.cat([fc, fr, fr - fc], dim=1)))
        return self.luts(weights)
