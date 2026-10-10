import torch

from grademl.color import rgb_to_lab
from grademl.lut import identity_lut
from grademl.pro_looks import DIRECTIONS, ProLook, render_pro_look, sample_pro_look


def test_identity_look_is_identity():
    assert torch.allclose(render_pro_look(ProLook(), 17), identity_lut(17), atol=2e-3)


def test_zero_strength_is_identity():
    for d in DIRECTIONS:
        lk = sample_pro_look(torch.Generator().manual_seed(0), direction=d, strength=(0.0, 0.0))
        assert torch.allclose(render_pro_look(lk, 17), identity_lut(17), atol=2e-3), d


def test_looks_valid_and_reproducible():
    for s in range(5):
        for d in [None, *DIRECTIONS]:
            a = render_pro_look(sample_pro_look(torch.Generator().manual_seed(s), direction=d), 17)
            b = render_pro_look(sample_pro_look(torch.Generator().manual_seed(s), direction=d), 17)
            assert torch.equal(a, b) and torch.isfinite(a).all()
            assert a.min() >= 0 and a.max() <= 1


def test_gray_axis_stays_ordered():
    # Whatever the colors do, a gray ramp must stay a ramp (no tone inversion).
    i = torch.arange(17)
    for s in range(5):
        for d in [None, *DIRECTIONS]:
            lut = render_pro_look(sample_pro_look(torch.Generator().manual_seed(s), direction=d), 17)
            L = rgb_to_lab(lut[:, i, i, i].T)[:, 0]
            assert L.diff().min() > -0.5, (s, d)


def test_hsl_moves_only_the_target_hue():
    # Shifting greens toward teal must leave a red patch (almost) alone.
    lk = ProLook(hue_shift=torch.tensor([0, 0, 0, 45.0, 0, 0, 0, 0]))
    lut = render_pro_look(lk, 33)
    green = lut[:, 0, 32, 8]   # (r=8/32, g=1, b=0)
    red = lut[:, 0, 0, 32]     # (r=1, g=0, b=0)
    assert (rgb_to_lab(green[None]) - rgb_to_lab(torch.tensor([[0.25, 1.0, 0.0]]))).norm() > 5
    assert (rgb_to_lab(red[None]) - rgb_to_lab(torch.tensor([[1.0, 0.0, 0.0]]))).norm() < 1.5
