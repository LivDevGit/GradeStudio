import torch

from grademl.lut import apply_lut, identity_lut, read_cube, write_cube
from grademl.model import AdaptiveLUTNet, MatchLUTNet


def rand_img(b=2, h=17, w=23):
    torch.manual_seed(0)
    return torch.rand(b, 3, h, w)


def test_identity_lut_is_noop():
    img = rand_img()
    out = apply_lut(img, identity_lut(33))
    assert torch.allclose(out, img, atol=1e-5)


def test_linear_lut_is_exact():
    # Any affine color map is reproduced exactly by trilinear interpolation,
    # so this pins down the channel / axis ordering.
    ident = identity_lut(9)
    r, g, b = ident
    swapped = torch.stack([b, r, 0.5 * g + 0.25])  # out = (B, R, 0.5G + 0.25)
    img = rand_img()
    out = apply_lut(img, swapped)
    expected = torch.stack([img[:, 2], img[:, 0], 0.5 * img[:, 1] + 0.25], dim=1)
    assert torch.allclose(out, expected, atol=1e-5)


def test_batched_luts():
    img = rand_img(b=2)
    luts = torch.stack([identity_lut(17), 1 - identity_lut(17)])
    out = apply_lut(img, luts)
    assert torch.allclose(out[0], img[0], atol=1e-5)
    assert torch.allclose(out[1], 1 - img[1], atol=1e-5)


def test_cube_roundtrip(tmp_path):
    lut = identity_lut(5) ** 2
    write_cube(lut, tmp_path / "t.cube")
    back = read_cube(tmp_path / "t.cube")
    assert torch.allclose(back, lut, atol=1e-6)


def test_cube_file_order(tmp_path):
    # In .cube files red varies fastest: the second data row is (r=1/(D-1), g=0, b=0).
    write_cube(identity_lut(3), tmp_path / "t.cube")
    rows = [l for l in (tmp_path / "t.cube").read_text().splitlines() if l[0].isdigit()]
    assert rows[1].split() == ["0.500000", "0.000000", "0.000000"]
    assert rows[3].split() == ["0.000000", "0.500000", "0.000000"]


def test_untrained_models_predict_identity():
    img = rand_img(b=2, h=256, w=256)
    for model, args in [(AdaptiveLUTNet(), (img,)), (MatchLUTNet(), (img, img.flip(0)))]:
        lut = model.eval()(*args)
        assert lut.shape == (2, 3, 33, 33, 33)
        assert torch.allclose(lut, identity_lut(33).expand_as(lut), atol=1e-6)
