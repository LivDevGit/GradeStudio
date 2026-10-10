"""Image quality metrics: PSNR and CIE76 color difference (the two numbers
FiveK papers report)."""
import torch

# sRGB (D65) -> XYZ
_RGB2XYZ = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_WHITE_D65 = torch.tensor([0.95047, 1.0, 1.08883])


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR in dB for (B, 3, H, W) images in [0, 1]."""
    mse = (pred.clamp(0, 1) - target).pow(2).flatten(1).mean(1)
    return 10 * torch.log10(1 / mse.clamp_min(1e-10))


def rgb_to_lab(img: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) sRGB in [0, 1] -> CIELAB (L in 0..100)."""
    img = img.clamp(0, 1)
    lin = torch.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
    xyz = torch.einsum("ij,bjhw->bihw", _RGB2XYZ.to(img), lin)
    xyz = xyz / _WHITE_D65.to(img).view(1, 3, 1, 1)
    eps = 216 / 24389
    f = torch.where(xyz > eps, xyz.clamp_min(1e-12) ** (1 / 3), (24389 / 27 * xyz + 16) / 116)
    l = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return torch.stack([l, a, b], dim=1)


def delta_e76(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-image mean CIE76 color difference."""
    d = rgb_to_lab(pred) - rgb_to_lab(target)
    return d.pow(2).sum(1).sqrt().flatten(1).mean(1)
