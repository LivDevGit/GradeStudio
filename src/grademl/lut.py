"""3D LUT primitives: identity, trilinear application, .cube I/O.

Layout convention (matches the .cube file order and Zeng et al.):
    lut[c, b, g, r]   shape (3, D, D, D), values in [0, 1]
i.e. the output channel c for the input color (r, g, b) sits at index
[c, b_idx, g_idx, r_idx]. With this layout `grid_sample` can treat the LUT as
a 3D volume of depth=b, height=g, width=r and look up (x=r, y=g, z=b).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_SIZE = 33  # 33^3 is the common "high quality" .cube size


def identity_lut(size: int = DEFAULT_SIZE, device=None, dtype=torch.float32) -> torch.Tensor:
    """LUT that maps every color to itself. Shape (3, size, size, size)."""
    ramp = torch.linspace(0, 1, size, device=device, dtype=dtype)
    b, g, r = torch.meshgrid(ramp, ramp, ramp, indexing="ij")
    return torch.stack([r, g, b])


def apply_lut(img: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Apply a 3D LUT with trilinear interpolation.

    img: (B, 3, H, W) RGB in [0, 1]
    lut: (B, 3, D, D, D) or (3, D, D, D), layout lut[c, b, g, r]
    returns (B, 3, H, W)

    Training-time reference implementation. On device the same lookup is a
    Metal / Core Image 3D texture sample, so it is not part of the Core ML model.
    """
    if lut.dim() == 4:
        lut = lut.unsqueeze(0).expand(img.shape[0], -1, -1, -1, -1)
    # grid_sample wants coords in [-1, 1], last dim ordered (x, y, z) = (r, g, b)
    grid = img.clamp(0, 1).permute(0, 2, 3, 1).unsqueeze(1) * 2 - 1  # (B, 1, H, W, 3)
    out = F.grid_sample(lut, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out.squeeze(2)  # (B, 3, 1, H, W) -> (B, 3, H, W)


def lut_smoothness(lut: torch.Tensor) -> torch.Tensor:
    """Total-variation regularizer (Zeng et al.): mean squared difference
    between neighboring LUT entries along each input axis. lut: (..., 3, D, D, D)."""
    return sum(d.pow(2).mean() for d in _axis_diffs(lut))


def lut_monotonicity(lut: torch.Tensor, own_channel_only: bool = False) -> torch.Tensor:
    """Penalize outputs that decrease as an input channel increases (Zeng et al.).
    Keeps the mapping order-preserving, which avoids banding and color inversions.

    own_channel_only: only require R_out to rise with R_in (G with G, B with B).
    Creative grades legitimately break cross-channel monotonicity (more
    saturation lowers R_out as G_in rises), so stage 2 uses this variant.
    """
    diffs = _axis_diffs(lut)
    if own_channel_only:  # diffs are along r, g, b -> pick channel c of diff c
        return sum(F.relu(-d[..., c, :, :, :]).mean() for c, d in enumerate(diffs))
    return sum(F.relu(-d).mean() for d in diffs)


def _axis_diffs(lut: torch.Tensor) -> list[torch.Tensor]:
    return [lut.diff(dim=-1), lut.diff(dim=-2), lut.diff(dim=-3)]  # along r, g, b


def read_cube(path: str | Path) -> torch.Tensor:
    """Read an Adobe/Resolve .cube 3D LUT into (3, D, D, D)."""
    size, rows = None, []
    domain_min, domain_max = np.zeros(3), np.ones(3)
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key = line.split()[0].upper()
        if key == "LUT_3D_SIZE":
            size = int(line.split()[1])
        elif key == "DOMAIN_MIN":
            domain_min = np.array(line.split()[1:4], dtype=np.float64)
        elif key == "DOMAIN_MAX":
            domain_max = np.array(line.split()[1:4], dtype=np.float64)
        elif key == "LUT_1D_SIZE":
            raise ValueError(f"{path}: 1D LUTs are not supported")
        elif key[0].isalpha():
            continue  # TITLE and other keywords we don't need
        else:
            rows.append(line.split()[:3])
    if size is None:
        raise ValueError(f"{path}: missing LUT_3D_SIZE")
    data = np.asarray(rows, dtype=np.float64)
    if data.shape != (size**3, 3):
        raise ValueError(f"{path}: expected {size**3} rows, got {data.shape[0]}")
    if not (np.allclose(domain_min, 0) and np.allclose(domain_max, 1)):
        raise ValueError(f"{path}: non-[0,1] domain not supported")
    # .cube order: red varies fastest, then green, then blue -> (b, g, r, c)
    lut = data.reshape(size, size, size, 3).transpose(3, 0, 1, 2)
    return torch.from_numpy(np.ascontiguousarray(lut)).float()


def write_cube(lut: torch.Tensor, path: str | Path, title: str = "grademl") -> None:
    """Write a (3, D, D, D) LUT as a .cube file."""
    lut = lut.detach().float().cpu().clamp(0, 1)
    size = lut.shape[-1]
    rows = lut.permute(1, 2, 3, 0).reshape(-1, 3).numpy()  # (b, g, r, c), red fastest
    lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0 0 0", "DOMAIN_MAX 1 1 1"]
    lines += [f"{r:.6f} {g:.6f} {b:.6f}" for r, g, b in rows]
    Path(path).write_text("\n".join(lines) + "\n")
