import torch

from grademl.model import ColorStats, MatchLUTNet


def test_joint_hist_is_a_distribution():
    stats = ColorStats(bins3d=8)
    x = torch.rand(2, 3, 32, 32)
    h = stats.joint_hist(x)
    assert h.shape == (2, 512)
    p = (h**2 - 1e-6) / 512  # undo sqrt scaling
    assert torch.allclose(p.sum(1), torch.ones(2), atol=1e-3)


def test_joint_hist_locates_colors():
    stats = ColorStats(bins3d=8)
    red = torch.zeros(1, 3, 16, 16)
    red[:, 0] = 1.0
    h = stats.joint_hist(red)[0].reshape(8, 8, 8)  # (r, g, b) bins
    assert h.argmax().item() == torch.tensor(7 * 64)  # r=7, g=0, b=0


def test_match_net_v2_shapes_and_identity_init():
    model = MatchLUTNet(n_luts=64, hidden=768, bins3d=8).eval()
    c, r = torch.rand(2, 3, 256, 256), torch.rand(2, 3, 256, 256)
    lut = model(c, r)
    assert lut.shape == (2, 3, 33, 33, 33)
    # Zero bases -> identity LUT before training, whatever the inputs.
    from grademl.lut import identity_lut
    assert torch.allclose(lut, identity_lut(33).expand_as(lut), atol=1e-6)


def test_encoder_output_depends_on_input():
    # Regression: an encoder ending in InstanceNorm + global mean outputs
    # the norm's bias for every image.
    from grademl.model import Encoder
    torch.manual_seed(0)
    enc = Encoder().eval()
    with torch.no_grad():
        a, b = enc(torch.rand(1, 3, 256, 256)), enc(torch.rand(1, 3, 256, 256) ** 3)
    assert (a - b).abs().max() > 1e-3
    legacy = Encoder(legacy=True).eval()
    with torch.no_grad():
        la, lb = legacy(torch.rand(1, 3, 256, 256)), legacy(torch.rand(1, 3, 256, 256) ** 3)
    assert torch.allclose(la, lb, atol=1e-5)  # documents the old bug
