import cv2
import numpy as np
import torch

from grademl.lut import identity_lut
from grademl.synthetic import (DIRECTIONS, SyntheticMatchPairs, make_match_batch, random_look,
                               render_look, sample_look, upsample_lut)


def test_zero_strength_is_identity():
    for seed in range(20):
        lut = random_look(torch.Generator().manual_seed(seed), size=17, strength=0.0)
        assert torch.allclose(lut, identity_lut(17), atol=1e-4), seed


def test_looks_are_valid_and_deterministic():
    for seed in range(50):
        a = random_look(torch.Generator().manual_seed(seed), size=17)
        b = random_look(torch.Generator().manual_seed(seed), size=17)
        assert a.shape == (3, 17, 17, 17)
        assert torch.equal(a, b)
        assert a.min() >= 0 and a.max() <= 1
        assert torch.isfinite(a).all()


def test_looks_differ_and_are_not_trivial():
    luts = [random_look(torch.Generator().manual_seed(s), size=17) for s in range(20)]
    dist = [(l - identity_lut(17)).abs().mean().item() for l in luts]
    assert np.median(dist) > 0.01  # most looks visibly change the image
    assert not torch.allclose(luts[0], luts[1])


def test_looks_preserve_tonal_order():
    # Cross-channel decreases are legitimate (more saturation lowers R as G
    # rises), but each channel must rise with itself and grays must stay
    # ordered by brightness, or the grade would invert tones.
    cases = [(s, d) for s in range(10) for d in [None, *DIRECTIONS]]
    for s, d in cases:
        lut = random_look(torch.Generator().manual_seed(s), size=17, direction=d)
        own = [lut[0].diff(dim=-1), lut[1].diff(dim=-2), lut[2].diff(dim=-3)]  # dR/dR, dG/dG, dB/dB
        assert min(x.min().item() for x in own) > -0.02, (s, d)
        i = torch.arange(17)
        gray_luma = torch.tensor([0.2126, 0.7152, 0.0722]) @ lut[:, i, i, i]
        assert gray_luma.diff().min().item() > -1e-4, (s, d)


def test_synthetic_pairs(tmp_path):
    rng = np.random.default_rng(0)
    paths = []
    for k in range(3):
        p = tmp_path / f"{k}.png"
        cv2.imwrite(str(p), rng.integers(0, 256, (40, 60, 3), dtype=np.uint8))
        paths.append(p)
    ds = SyntheticMatchPairs(paths, train=False, render_size=9)
    content, ref_scene, lut = ds[0]
    assert content.shape == ref_scene.shape == (3, 256, 256)
    assert lut.shape == (3, 9, 9, 9)
    assert not torch.equal(content, ref_scene)  # reference comes from another photo
    # Validation samples are reproducible.
    again = ds[0]
    assert all(torch.equal(x, y) for x, y in zip((content, ref_scene, lut), again))
    reference, target, full = make_match_batch(content[None], ref_scene[None], lut[None], lut_size=17)
    assert reference.shape == target.shape == (1, 3, 256, 256)
    assert full.shape == (1, 3, 17, 17, 17)


def test_upsampled_render_matches_full_render():
    # Training renders looks at 17^3 and upsamples on the GPU. Reference and
    # target use the same upsampled LUT, so training stays self-consistent;
    # this checks the looks keep their character. Larger errors only occur
    # at clamping kinks in saturated corners of the cube.
    errs = []
    for s in range(40):
        look = sample_look(torch.Generator().manual_seed(s))
        full = render_look(look, 33)
        up = upsample_lut(render_look(look, 17)[None], 33)[0]
        errs.append((full - up).abs().flatten())
    errs = torch.cat(errs)
    assert errs.mean() < 0.002, errs.mean()                  # ~0.5/255 on average
    assert errs.quantile(0.99) < 0.01, errs.quantile(0.99)   # < ~2.5/255 for 99% of entries
