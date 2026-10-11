"""Local grading: K image-adaptive LUTs blended per pixel by a bilateral grid.

A single 3D LUT applies one color mapping to the whole frame: it cannot
brighten a face without brightening a wall of the same color, hold a sky
while lifting the foreground, or grade the background apart from the
subject. Colorists do exactly that with masks and power windows.

LocalGradeNet predicts, from the 256px photo,

  K LUTs        each identity + w_k @ basis (one shared residual bank)
  a weight grid (K, G, gh, gw) logits: a bilateral grid over (x, y, luma)

and the full-resolution result is

  weights(x, y) = softmax_k( grid sampled at (x, y, luma(x, y)) )
  out(x, y)     = sum_k weights_k(x, y) * LUT_k(pixel)

Slicing the grid by the pixel's own luminance (HDRNet, Gharbi et al. 2017)
makes the blend follow edges: a bright sky and a dark ridge at the same
(x, y) cell land in different luma bins, so they can take different LUTs
without a halo, and no segmentation runs on the device. On iOS this is K
CIColorCube passes plus one Metal kernel that samples the grid as a 3D
texture and blends (ios/LocalGrade.metal).

The network is conditioned on an optional vector (a style one-hot for
GradeNet-L, a reference embedding for matching), so the same body serves
every task.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lut import DEFAULT_SIZE
from .model import ColorStats, LUTBank

LUMA = (0.2126, 0.7152, 0.0722)


def luma(img: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) -> (B, 1, H, W)."""
    w = torch.tensor(LUMA, device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
    return (img * w).sum(1, keepdim=True)


def slice_grid(grid: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    """Sample a bilateral grid at every pixel.

    grid: (B, K, G, gh, gw) - K channels over G luma bins and a gh x gw grid
    img:  (B, 3, H, W) in [0, 1], any resolution
    returns (B, K, H, W)
    """
    b, _, h, w = img.shape
    ys = torch.linspace(-1, 1, h, device=img.device, dtype=img.dtype)
    xs = torch.linspace(-1, 1, w, device=img.device, dtype=img.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gz = luma(img.clamp(0, 1))[:, 0] * 2 - 1                            # (B, H, W)
    coords = torch.stack([gx.expand(b, -1, -1), gy.expand(b, -1, -1), gz], -1)  # x, y, z=luma
    out = F.grid_sample(grid, coords[:, None], mode="bilinear", padding_mode="border", align_corners=True)
    return out[:, :, 0]


def blend_weights(grid: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    """(B, K, H, W) per-pixel LUT weights (softmax over K)."""
    return slice_grid(grid, img).softmax(1)


def apply_local(img: torch.Tensor, luts: torch.Tensor, grid: torch.Tensor,
                weights: torch.Tensor | None = None) -> torch.Tensor:
    """img (B, 3, H, W); luts (B, K, 3, D, D, D); grid (B, K, G, gh, gw).
    Returns the locally graded image (B, 3, H, W)."""
    from .lut import apply_lut

    if weights is None:
        weights = blend_weights(grid, img)
    out = torch.zeros_like(img)
    for k in range(luts.shape[1]):
        out = out + weights[:, k:k + 1] * apply_lut(img, luts[:, k])
    return out


def grid_tv(grid: torch.Tensor) -> torch.Tensor:
    """Smoothness of the weight grid along luma, y and x (squared differences)."""
    return sum(grid.diff(dim=d).pow(2).mean() for d in (-3, -2, -1))


def _conv(cin: int, cout: int, stride: int = 1) -> list[nn.Module]:
    return [nn.Conv2d(cin, cout, 3, stride=stride, padding=1), nn.InstanceNorm2d(cout, affine=True),
            nn.LeakyReLU(0.2)]


class LocalGradeNet(nn.Module):
    """Photo (+ optional condition) -> K LUTs + bilateral weight grid.

    256x256 input. Backbone: strided convs to a 16x16 local feature map and
    an 8x8 map that is pooled into the global vector together with explicit
    color statistics (instance norm strips color casts, see ColorStats) and
    the condition. Global -> K LUT weight vectors; global fused into the
    local features (HDRNet's local/global fusion) -> grid logits.

    The last grid layer starts at zero, so an untrained net blends the K
    LUTs uniformly, and the zero LUT bases make every LUT the identity: the
    untrained net leaves the photo unchanged.
    """

    def __init__(self, k: int = 4, n_luts: int = 24, lut_size: int = DEFAULT_SIZE, width: int = 24,
                 hidden: int = 512, bins3d: int = 8, grid_bins: int = 8, cond_dim: int = 0,
                 cond_embed: int = 32):
        super().__init__()
        w = width
        self.k, self.n, self.grid_bins = k, n_luts, grid_bins
        self.stem = nn.Sequential(
            nn.Conv2d(3, w, 3, stride=2, padding=1), nn.LeakyReLU(0.2),  # 128
            *_conv(w, 2 * w, 2),                                         # 64
            *_conv(2 * w, 4 * w, 2),                                     # 32
            *_conv(4 * w, 8 * w, 2),                                     # 16
        )
        c16 = 8 * w
        self.local = nn.Sequential(*_conv(c16, c16), *_conv(c16, c16))
        self.down = nn.Sequential(*_conv(c16, c16, 2), *_conv(c16, c16, 2))  # 8, 4
        self.stats = ColorStats(bins3d=bins3d)
        self.cond = nn.Linear(cond_dim, cond_embed, bias=False) if cond_dim else None
        d = c16 + self.stats.out_dim + (cond_embed if cond_dim else 0)
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2), nn.Dropout(0.1),
        )
        self.lut_head = nn.Linear(hidden, k * n_luts)
        nn.init.zeros_(self.lut_head.bias)
        self.fuse = nn.Linear(hidden, c16)
        self.grid_head = nn.Sequential(
            nn.Conv2d(c16, c16, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(c16, k * grid_bins, 1),
        )
        nn.init.zeros_(self.grid_head[-1].weight)
        nn.init.zeros_(self.grid_head[-1].bias)
        self.luts = LUTBank(n_luts, lut_size)

    def forward(self, image: torch.Tensor, cond: torch.Tensor | None = None, flat: bool = False):
        """image (B, 3, 256, 256); cond (B, cond_dim) or None.
        Returns (luts (B, K, 3, D, D, D), grid (B, K, G, 16, 16)).
        flat: luts as (B*K, 3, D, D, D) instead (Core ML export: rank <= 5)."""
        b = image.shape[0]
        f16 = self.stem(image)
        g = [self.down(f16).mean(dim=(2, 3)), self.stats(image)]
        if self.cond is not None:
            g.append(self.cond(cond))
        h = self.mlp(torch.cat(g, 1))
        luts = self.luts(self.lut_head(h).reshape(-1, self.n))          # (B*K, 3, D, D, D)
        if not flat:
            luts = luts.reshape(b, self.k, *luts.shape[1:])
        loc = self.local(f16) + self.fuse(h)[:, :, None, None]
        grid = self.grid_head(F.leaky_relu(loc, 0.2))                    # (B, K*G, 16, 16)
        grid = grid.reshape(b, self.k, self.grid_bins, *grid.shape[-2:])
        return luts, grid


# ---------------------------------------------------------------- app packing


def pack_grid(grid: torch.Tensor) -> torch.Tensor:
    """(K<=4, G, gh, gw) grid -> (4, gh, G*gw) image: luma slices side by
    side, one RGBA channel per LUT (unused channels at -1e4, i.e. weight 0).
    This is the float texture the app's Core Image kernel samples
    (ios/LocalGrade.ci.metal); slice_packed is its reference."""
    k, g, gh, gw = grid.shape
    out = torch.full((4, gh, g * gw), -1e4, dtype=grid.dtype, device=grid.device)
    out[:k] = grid.permute(0, 2, 1, 3).reshape(k, gh, g * gw)
    return out


def slice_packed(packed: torch.Tensor, img: torch.Tensor, g: int) -> torch.Tensor:
    """What the Metal kernel computes, in PyTorch: packed (4, gh, G*gw),
    img (3, H, W) -> per-pixel weights (4, H, W). Matches blend_weights."""
    _, gh, total = packed.shape
    gw = total // g
    h, w = img.shape[-2:]
    y = torch.linspace(0, gh - 1, h, device=img.device)[:, None].expand(h, w)
    x = torch.linspace(0, gw - 1, w, device=img.device)[None].expand(h, w)
    z = luma(img[None].clamp(0, 1))[0, 0] * (g - 1)
    z0 = z.floor().clamp(0, g - 2)
    fz = z - z0

    def bilinear(xx, yy):  # texel-center sampling with clamp-to-edge, as Core Image does
        x0, y0 = xx.floor().clamp(0, total - 2), yy.floor().clamp(0, gh - 2)
        fx, fy = (xx - x0).clamp(0, 1), (yy - y0).clamp(0, 1)
        x0, y0 = x0.long(), y0.long()

        def at(a, b):
            return packed[:, b, a]

        return (at(x0, y0) * (1 - fx) * (1 - fy) + at(x0 + 1, y0) * fx * (1 - fy)
                + at(x0, y0 + 1) * (1 - fx) * fy + at(x0 + 1, y0 + 1) * fx * fy)

    # Each slice is sampled inside its own tile; x never crosses into the
    # next tile because it stays in [0, gw - 1] and tiles are gw wide.
    lo = bilinear(x + z0 * gw, y)
    hi = bilinear(x + (z0 + 1) * gw, y)
    return (lo * (1 - fz) + hi * fz).softmax(0)


# ---------------------------------------------------------------- reference matching


class MatchLocalNet(nn.Module):
    """(photo, reference) -> K LUTs + bilateral grid: content-aware matching
    on device, distilled from grademl.semantic_match (DINOv2 + per-region
    transfer) into one small network.

    Both images go through the same conv stem to a 16x16 feature map. Every
    photo cell then attends to the reference cells (cross-attention, keys
    from reference features, values = reference features + the cell's mean
    color and color spread), so each part of the photo gathers the colors of
    the parts of the reference that look like it: sky reads the reference's
    sky, skin its skin. The attended context feeds the grid (where to apply
    which LUT) and, pooled together with both images' color statistics, the
    LUT weights (what the K LUTs do). Plain matmul / softmax: converts to
    Core ML like the rest.
    """

    def __init__(self, k: int = 4, n_luts: int = 32, lut_size: int = DEFAULT_SIZE, width: int = 24,
                 hidden: int = 640, bins3d: int = 8, grid_bins: int = 8, attn_dim: int = 96):
        super().__init__()
        w = width
        self.k, self.n, self.grid_bins = k, n_luts, grid_bins
        self.stem = nn.Sequential(
            nn.Conv2d(3, w, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            *_conv(w, 2 * w, 2), *_conv(2 * w, 4 * w, 2), *_conv(4 * w, 8 * w, 2),
        )
        c16 = 8 * w
        self.local = nn.Sequential(*_conv(c16, c16), *_conv(c16, c16))
        self.down = nn.Sequential(*_conv(c16, c16, 2), *_conv(c16, c16, 2))
        self.stats = ColorStats(bins3d=bins3d)
        # Cell colors: mean and std of RGB in each 16x16 cell (instance norm in
        # the stem removes exactly this information).
        self.q = nn.Conv2d(c16 + 6, attn_dim, 1)
        self.kk = nn.Conv2d(c16 + 6, attn_dim, 1)
        self.v = nn.Conv2d(c16 + 6, c16, 1)
        self.scale = attn_dim ** -0.5
        d = 2 * (c16 + self.stats.out_dim) + c16
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2), nn.Dropout(0.1),
        )
        self.lut_head = nn.Linear(hidden, k * n_luts)
        nn.init.zeros_(self.lut_head.bias)
        self.fuse = nn.Linear(hidden, c16)
        self.grid_head = nn.Sequential(
            nn.Conv2d(2 * c16 + 6, c16, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(c16, k * grid_bins, 1),
        )
        nn.init.zeros_(self.grid_head[-1].weight)
        nn.init.zeros_(self.grid_head[-1].bias)
        self.luts = LUTBank(n_luts, lut_size)

    @staticmethod
    def cell_colors(img: torch.Tensor) -> torch.Tensor:
        """(B, 3, 256, 256) -> (B, 6, 16, 16): mean and std of each 16x16 cell."""
        m = F.avg_pool2d(img, 16)
        sq = F.avg_pool2d(img * img, 16)
        return torch.cat([m, (sq - m * m).clamp_min(1e-6).sqrt() * 4], 1)

    def forward(self, photo: torch.Tensor, reference: torch.Tensor, flat: bool = False):
        b = photo.shape[0]
        fp, fr = self.stem(photo), self.stem(reference)                  # (B, C, 16, 16)
        cp, cr = self.cell_colors(photo), self.cell_colors(reference)
        xp, xr = torch.cat([fp, cp], 1), torch.cat([fr, cr], 1)
        q = self.q(xp).flatten(2).transpose(1, 2)                        # (B, 256, a)
        kk = self.kk(xr).flatten(2)                                      # (B, a, 256)
        v = self.v(xr).flatten(2).transpose(1, 2)                        # (B, 256, C)
        attn = (q @ kk * self.scale).softmax(-1)
        ctx = (attn @ v).transpose(1, 2).reshape(b, -1, *fp.shape[-2:])  # (B, C, 16, 16)

        g = torch.cat([self.down(fp).mean(dim=(2, 3)), self.stats(photo),
                       self.down(fr).mean(dim=(2, 3)), self.stats(reference), ctx.mean(dim=(2, 3))], 1)
        h = self.mlp(g)
        luts = self.luts(self.lut_head(h).reshape(-1, self.n))
        if not flat:
            luts = luts.reshape(b, self.k, *luts.shape[1:])
        loc = torch.cat([self.local(fp) + self.fuse(h)[:, :, None, None], ctx, cp], 1)
        grid = self.grid_head(F.leaky_relu(loc, 0.2))
        return luts, grid.reshape(b, self.k, self.grid_bins, *grid.shape[-2:])
