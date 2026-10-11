import torch

from grademl.local import LocalGradeNet, apply_local, blend_weights, slice_grid
from grademl.lut import identity_lut


def test_untrained_net_is_identity():
    net = LocalGradeNet(k=3, n_luts=8, lut_size=17, width=8, hidden=64, cond_dim=5).eval()
    img = torch.rand(2, 3, 256, 256)
    with torch.no_grad():
        luts, grid = net(img, torch.eye(5)[:2])
    assert luts.shape == (2, 3, 3, 17, 17, 17) and grid.shape == (2, 3, 8, 16, 16)
    assert torch.allclose(luts, identity_lut(17).expand_as(luts), atol=1e-6)
    out = apply_local(torch.rand(2, 3, 100, 70), luts, grid)
    assert out.shape == (2, 3, 100, 70)


def test_grid_slices_by_luma():
    # Two luma bins: dark pixels -> LUT 0, bright pixels -> LUT 1, anywhere in the frame.
    grid = torch.zeros(1, 2, 2, 4, 4)
    grid[:, 0, 0], grid[:, 1, 1] = 10, 10
    img = torch.zeros(1, 3, 8, 8)
    img[..., :, 4:] = 1.0
    w = blend_weights(grid, img)
    assert w[0, 0, :, :4].min() > 0.99 and w[0, 1, :, 4:].min() > 0.99


def test_slice_is_resolution_independent():
    grid = torch.randn(1, 2, 4, 6, 6)
    img = torch.rand(1, 3, 64, 64)
    big = torch.nn.functional.interpolate(img, size=(128, 128), mode="nearest")
    a = slice_grid(grid, img)
    b = torch.nn.functional.interpolate(slice_grid(grid, big), size=(64, 64), mode="area")
    assert (a - b).abs().mean() < 0.05


def test_packed_grid_matches_slicing():
    # The app's packed-texture slicing (ios/LocalGrade.ci.metal) must equal slice_grid.
    from grademl.local import pack_grid, slice_packed
    torch.manual_seed(0)
    for k in (3, 4):
        grid = torch.randn(1, k, 8, 16, 16)
        img = torch.rand(1, 3, 37, 53)
        ref = blend_weights(grid, img)[0]
        got = slice_packed(pack_grid(grid[0]), img[0], 8)
        assert torch.allclose(got[:k], ref, atol=1e-4)
        assert k == 4 or got[k:].abs().max() < 1e-6


def test_match_net_shapes_identity_and_reference_dependence():
    from grademl.local import MatchLocalNet
    net = MatchLocalNet(k=3, n_luts=8, lut_size=17, width=8, hidden=64).eval()
    p, r = torch.rand(2, 3, 256, 256), torch.rand(2, 3, 256, 256)
    with torch.no_grad():
        luts, grid = net(p, r)
        assert luts.shape == (2, 3, 3, 17, 17, 17) and grid.shape == (2, 3, 8, 16, 16)
        assert torch.allclose(luts, identity_lut(17).expand_as(luts), atol=1e-6)
        net.luts.basis.normal_(std=0.05)
        a = net(p, r)[0]
        b = net(p, r.flip(1))[0]
    assert (a - b).abs().max() > 1e-4
