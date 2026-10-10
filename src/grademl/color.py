"""sRGB <-> CIELAB (D65) on channel-last tensors (..., 3)."""
import torch

_RGB2XYZ = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_XYZ2RGB = torch.linalg.inv(_RGB2XYZ)
_WHITE = torch.tensor([0.95047, 1.0, 1.08883])
_EPS, _KAPPA = 216 / 24389, 24389 / 27


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x.clamp_min(0) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(1e-12) ** (1 / 2.4) - 0.055)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """(..., 3) sRGB in [0, 1] -> Lab (L in 0..100)."""
    xyz = srgb_to_linear(rgb.clamp(0, 1)) @ _RGB2XYZ.to(rgb).T / _WHITE.to(rgb)
    f = torch.where(xyz > _EPS, xyz.clamp_min(1e-12) ** (1 / 3), (_KAPPA * xyz + 16) / 116)
    return torch.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """(..., 3) Lab -> sRGB, clamped to [0, 1]."""
    fy = (lab[..., 0] + 16) / 116
    f = torch.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = torch.where(f**3 > _EPS, f**3, (116 * f - 16) / _KAPPA) * _WHITE.to(lab)
    return linear_to_srgb(xyz @ _XYZ2RGB.to(lab).T).clamp(0, 1)
