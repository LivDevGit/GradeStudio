import torch

from grademl.compose import (SubjectAware, background_pop_lut, compose_luts, scale_lut, split_luts,
                             subject_aware_grade)
from grademl.lut import apply_lut, identity_lut
from grademl.mask import guided_filter, refine_mask
from grademl.pro_looks import random_pro_look


def setup(h=24, w=32):
    torch.manual_seed(0)
    img = torch.rand(3, h, w)
    look = random_pro_look(torch.Generator().manual_seed(1), size=17, direction="teal_orange", strength=(1.0, 1.0))
    return img, look


def test_full_mask_is_subject_grade():
    img, look = setup()
    cfg = SubjectAware()
    out = subject_aware_grade(img, look, torch.ones(1, *img.shape[-2:]), cfg)
    expected = apply_lut(img[None], scale_lut(look, cfg.subject_strength))[0]
    assert torch.allclose(out, expected, atol=1e-4)


def test_empty_mask_falls_back_to_plain_look():
    # No person in the frame: the whole photo must not be treated as background.
    img, look = setup()
    plain = apply_lut(img[None], look)[0]
    assert torch.allclose(subject_aware_grade(img, look, torch.zeros(1, *img.shape[-2:])), plain, atol=1e-5)
    assert torch.allclose(subject_aware_grade(img, look, None), plain, atol=1e-5)


def test_background_side_gets_the_pop():
    img, look = setup()
    m = torch.zeros(1, *img.shape[-2:])
    m[:, :, :16] = 1  # left half is the person
    cfg = SubjectAware(background_pop=1.0)
    out = subject_aware_grade(img, look, m, cfg)
    _, bg, pop = split_luts(look, cfg)
    expected_bg = apply_lut(apply_lut(img[None], bg), pop)[0]
    assert torch.allclose(out[:, :, 16:], expected_bg[:, :, 16:], atol=1e-4)
    assert out[:, :, 16:].mean() < apply_lut(img[None], look)[0][:, :, 16:].mean()  # pop darkens


def test_neutral_settings_are_the_plain_look():
    img, look = setup()
    m = (torch.rand(1, *img.shape[-2:]) > 0.5).float()
    cfg = SubjectAware(subject_strength=1.0, background_strength=1.0, background_pop=0.0)
    assert torch.allclose(subject_aware_grade(img, look, m, cfg), apply_lut(img[None], look)[0], atol=1e-4)


def test_compose_luts_matches_sequential_apply():
    img, look = setup()
    pop = background_pop_lut(17)
    seq = apply_lut(apply_lut(img[None], look), pop)
    assert (apply_lut(img[None], compose_luts(look, pop)) - seq).abs().max() < 0.03  # 17^3 interpolation
    assert torch.allclose(compose_luts(identity_lut(17), pop), pop, atol=1e-5)


def test_guided_filter_keeps_constants_and_range():
    guide = torch.rand(1, 1, 40, 40)
    flat = torch.full((1, 1, 40, 40), 0.7)
    assert torch.allclose(guided_filter(flat, guide, 4, 1e-3), flat, atol=1e-4)
    prob = torch.zeros(1, 10, 10)
    prob[:, 3:7, 3:7] = 1
    m = refine_mask(prob, torch.rand(3, 50, 70))
    assert m.shape == (1, 50, 70) and m.min() >= 0 and m.max() <= 1


def test_refine_snaps_to_image_edges():
    # Coarse blob over a sharp bright square: refining must make the mask
    # step sharper exactly at the square's edge than plain upsampling does.
    img = torch.zeros(3, 64, 64)
    img[:, 16:48, 16:48] = 1
    prob = torch.nn.functional.interpolate(img[:1, None], size=(8, 8), mode="area")[0]
    prob = torch.nn.functional.avg_pool2d(prob[None], 3, 1, 1)[0]  # blurrier than the square
    m = refine_mask(prob, img, radius=0.06)[0, 32]
    plain = torch.nn.functional.interpolate(prob[None], size=(64, 64), mode="bilinear")[0, 0, 32]
    assert m[16] - m[15] > 2 * (plain[16] - plain[15])
