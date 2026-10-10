import torch

from grademl.color import lab_to_rgb, rgb_to_lab
from grademl.lut import apply_lut
from grademl.metrics import psnr
from grademl.transfer import transfer_lut

FULL = dict(tone=1.0, tint=1.0, max_tint=100.0, spread=1.0)  # no damping


def smooth_image(seed: int, b: int = 2) -> torch.Tensor:
    """Photo-like test image: smooth color gradients plus some texture."""
    g = torch.Generator().manual_seed(seed)
    coarse = torch.rand(b, 3, 4, 4, generator=g)
    img = torch.nn.functional.interpolate(coarse, size=(64, 64), mode="bicubic", align_corners=False)
    return (img + 0.03 * torch.randn(b, 3, 64, 64, generator=g)).clamp(0, 1)


def test_lab_roundtrip():
    rgb = torch.rand(1000, 3)
    assert torch.allclose(lab_to_rgb(rgb_to_lab(rgb)), rgb, atol=1e-4)


def test_lab_reference_values():
    lab = rgb_to_lab(torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    assert torch.allclose(lab[0], torch.tensor([100.0, 0, 0]), atol=0.05)
    assert torch.allclose(lab[1], torch.zeros(3), atol=0.05)
    assert torch.allclose(lab[2], torch.tensor([53.24, 80.09, 67.20]), atol=0.1)  # sRGB red


def test_self_transfer_is_near_identity():
    img = smooth_image(0)
    for kw in ({}, FULL):
        out = apply_lut(img, transfer_lut(img, img, size=17, **kw))
        assert psnr(out, img).min() > 33


def test_transfer_moves_colors_toward_reference():
    content = smooth_image(1)
    warm = (content * torch.tensor([1.15, 1.0, 0.75]).view(1, 3, 1, 1)).clamp(0, 1)
    ref = smooth_image(2) * torch.tensor([1.15, 1.0, 0.75]).view(1, 3, 1, 1)
    out = apply_lut(content, transfer_lut(content, ref.clamp(0, 1), size=17, **FULL))
    ab = lambda x: rgb_to_lab(x.permute(0, 2, 3, 1))[..., 1:].mean((1, 2))
    # The warm reference pushes b* (yellow) up, like warming the content directly would.
    assert (ab(out)[:, 1] > ab(content)[:, 1] + 3).all()
    assert (ab(warm)[:, 1] > ab(content)[:, 1]).all()


def test_gray_reference_gives_near_monochrome():
    content = smooth_image(3)
    gray = content.mean(1, keepdim=True).expand(-1, 3, -1, -1)
    gray_ref = smooth_image(4).mean(1, keepdim=True).expand(-1, 3, -1, -1)
    out = apply_lut(content, transfer_lut(content, gray_ref, 17, **FULL))
    chroma = lambda x: rgb_to_lab(x.permute(0, 2, 3, 1))[..., 1:].norm(dim=-1).mean()
    assert chroma(out) < 0.25 * chroma(content)
    assert chroma(gray) < 1.0


def test_luminance_stays_monotone():
    content, ref = smooth_image(5, b=1), smooth_image(6, b=1)
    lut = transfer_lut(content, ref, size=17)
    i = torch.arange(17)
    gray_l = rgb_to_lab(lut[0][:, i, i, i].T)[:, 0]
    assert (gray_l.diff() > -0.5).all()


def test_damping_caps_the_tint():
    content = smooth_image(7)
    cyan = torch.zeros_like(content)
    cyan[:, 1:] = 0.9  # flat cyan backdrop as "reference"
    full = apply_lut(content, transfer_lut(content, cyan, 17, **FULL))
    damped = apply_lut(content, transfer_lut(content, cyan, 17))
    shift = lambda x: (rgb_to_lab(x.permute(0, 2, 3, 1)) - rgb_to_lab(content.permute(0, 2, 3, 1)))[..., 1:].norm(dim=-1).mean()
    assert shift(damped) < 0.6 * shift(full)
