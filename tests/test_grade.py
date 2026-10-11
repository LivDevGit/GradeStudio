import random

import torch

from grademl.compose import scale_lut
from grademl.grade import (STYLES, Candidate, best_version, clipped, grade_luts, make_grade_batch,
                           pick_distinct, random_fault, recipe_luts, style_onehot, top_versions)
from grademl.grain import Grain, add_grain, soft_light
from grademl.lut import apply_lut, identity_lut
from grademl.model import GradeNet
from grademl.pro_looks import DIRECTIONS, render_pro_look


def small_net(seed=0):
    torch.manual_seed(seed)
    net = GradeNet(len(STYLES), n_luts=4, lut_size=9, hidden=32).eval()
    with torch.no_grad():
        net.luts.basis.normal_(std=0.02)
    return net


def test_styles_start_with_none_then_directions():
    assert STYLES[0] == "none" and STYLES[1:] == DIRECTIONS
    assert style_onehot([0, 3]).shape == (2, len(STYLES))


def test_untrained_grade_net_is_identity():
    net = GradeNet(len(STYLES), n_luts=4, lut_size=9, hidden=32).eval()
    with torch.no_grad():
        lut = net(torch.rand(2, 3, 256, 256), style_onehot([0, 5]))
    assert torch.allclose(lut, identity_lut(9).expand(2, -1, -1, -1, -1), atol=1e-6)


def test_style_changes_the_prediction():
    net = small_net()
    corr, luts = grade_luts(net, torch.rand(3, 64, 64), ["portra", "teal_orange"])
    assert corr.shape == (3, 9, 9, 9) and luts.shape == (2, 3, 9, 9, 9)
    assert not torch.allclose(luts[0], luts[1])


def test_faults_are_valid_luts():
    rng = random.Random(0)
    for _ in range(20):
        lut = render_pro_look(random_fault(rng), 9)
        assert torch.isfinite(lut).all() and lut.min() >= 0 and lut.max() <= 1


def test_grade_batch_targets():
    torch.manual_seed(0)
    photo = torch.rand(3, 3, 32, 32) * 0.6 + 0.2
    fault = identity_lut(9).expand(3, -1, -1, -1, -1).contiguous()
    recipes = recipe_luts(9)
    inp, tgt = make_grade_batch(photo, fault, torch.tensor([0, 1, 2]), recipes)
    assert torch.allclose(inp, photo, atol=1e-5)
    assert torch.allclose(tgt[0], photo[0], atol=1e-5)               # "none": the finished photo
    assert torch.allclose(tgt[1], apply_lut(photo[1:2], recipes[1])[0], atol=1e-5)


def test_grade_batch_backs_off_when_the_look_clips():
    photo = torch.full((1, 3, 16, 16), 0.5)
    recipes = torch.stack([identity_lut(9), torch.ones(3, 9, 9, 9)])  # a "look" that blows everything out
    _, tgt = make_grade_batch(photo, identity_lut(9)[None], torch.tensor([1]), recipes)
    assert clipped(tgt)[0] == 0 and tgt.mean() < 1  # held back to backoff strength


def test_scale_lut_relative_to_base():
    base, look = torch.rand(3, 5, 5, 5), torch.rand(3, 5, 5, 5)
    assert torch.allclose(scale_lut(look, 0.0, base), base)
    assert torch.allclose(scale_lut(look, 1.0, base), look)


def test_best_version_ranks_and_explains():
    net = small_net()
    img = torch.rand(3, 64, 96)
    mask = torch.zeros(1, 64, 96)
    mask[:, 10:50, 30:70] = 1
    edit = best_version(img, net, mask, intensities=(1.0,))
    assert len(edit.shortlist) == 1 + len(DIRECTIONS)
    assert edit.shortlist[0] is edit.best
    assert all(a.score >= b.score for a, b in zip(edit.shortlist, edit.shortlist[1:]))
    assert edit.subject is not None and edit.reasons
    assert edit.lut.shape == (3, 9, 9, 9)


def test_grain_keeps_tone_and_spares_extremes():
    img = torch.full((3, 300, 400), 0.5)
    out = add_grain(img, Grain(0.06, 1.0), seed=1)
    assert out.std() > 0.02 and abs(out.mean() - 0.5) < 0.01
    assert torch.equal(out[0], out[1])  # luminance grain
    assert add_grain(torch.zeros(3, 50, 50), Grain(0.08, 1.0)).max() < 1e-6
    assert torch.equal(add_grain(img, Grain(0.0)), img)


def test_soft_light_neutral_gray_is_identity():
    x = torch.rand(3, 8, 8)
    assert torch.allclose(soft_light(x, torch.full_like(x, 0.5)), x, atol=1e-6)


def test_top_versions_are_the_editors_ranked_choices():
    net = small_net()
    edits = top_versions(torch.rand(3, 64, 96), net, None, n=3, intensities=(0.8, 1.2))
    assert [e.rank for e in edits] == [1, 2, 3]
    assert edits[0].best is edits[0].shortlist[0]
    assert len({e.best.style for e in edits}) == 3
    assert edits[0].best.score >= edits[1].best.score >= edits[2].best.score


def test_pick_distinct_one_per_style_in_score_order():
    c = lambda s, v: Candidate(s, 1.0, v, {})  # noqa: E731
    ranked = [c("golden_hour", 3), c("golden_hour", 2.9), c("autumn", 2.5), c("none", 2), c("portra", 1)]
    assert [p.style for p in pick_distinct(ranked, 3)] == ["golden_hour", "autumn", "none"]
