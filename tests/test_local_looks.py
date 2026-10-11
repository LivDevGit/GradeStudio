import random

import torch

from grademl.grade import STYLES
from grademl.local_looks import (FINISHES, N_FAULT, FinishBank, apply_local_fault, finish, random_local_fault,
                                 region_masks)
from grademl.lut import identity_lut


def _maps(sky_rows: int = 8, depth: float = 0.5) -> torch.Tensor:
    m = torch.zeros(1, 6, 32, 32)
    m[:, 0, :sky_rows] = 1.0      # sky on top
    m[:, 5] = depth
    return m


def test_every_style_has_a_finish():
    assert set(FINISHES) == set(STYLES)


def test_sky_is_deepened_only_where_the_sky_is():
    bank = FinishBank(STYLES, 17)
    photo = torch.full((1, 3, 32, 32), 0.7)
    out = finish(photo, identity_lut(17)[None], torch.tensor([0]), _maps(), bank)
    sky, ground = out[0, :, :6].mean(), out[0, :, 20:].mean()
    assert sky < 0.69                      # "none" still recovers the sky
    assert (ground - 0.7).abs() < 0.02     # and leaves the rest alone (no vignette for "none")


def test_no_fault_is_identity():
    photo = torch.rand(2, 3, 32, 32)
    out = apply_local_fault(photo, torch.zeros(2, N_FAULT), _maps().expand(2, -1, -1, -1))
    assert torch.equal(out, photo)


def test_sky_fault_brightens_sky():
    photo = torch.full((1, 3, 32, 32), 0.4)
    p = torch.zeros(1, N_FAULT)
    p[0, 0] = 0.6  # +0.6 EV in linear light: sRGB 0.4 -> 0.48
    out = apply_local_fault(photo, p, _maps())
    assert out[0, :, :6].mean() > 0.46 and (out[0, :, 20:].mean() - 0.4).abs() < 0.01


def test_random_local_fault_shape_and_probability():
    rng = random.Random(0)
    faults = torch.stack([random_local_fault(rng, p=0.6) for _ in range(500)])
    assert faults.shape == (500, N_FAULT)
    active = (faults[:, :4].abs().sum(1) > 0).float().mean()
    assert 0.3 < active < 0.65


def test_masks_handle_depth_overshoot():
    m = _maps(depth=1.02)  # antialiased resize can overshoot 1
    masks = region_masks(m, torch.rand(1, 3, 32, 32))
    assert all(torch.isfinite(v).all() for v in masks.values())


def test_skin_finish_pulls_a_green_face_back_and_keeps_bw_gray():
    from grademl.color import rgb_to_lab
    from grademl.local_looks import skin_finish
    photo = torch.tensor([0.80, 0.62, 0.52]).view(1, 3, 1, 1).expand(1, 3, 4, 4).contiguous()  # skin
    green = torch.tensor([0.62, 0.70, 0.50]).view(1, 3, 1, 1).expand(1, 3, 4, 4).contiguous()  # graded green
    out = skin_finish(green, photo, torch.tensor([0.25]), torch.tensor([0.0]))

    def hue(x):
        lab = rgb_to_lab(x[0, :, 0, 0])
        return torch.rad2deg(torch.atan2(lab[2], lab[1])).item()
    assert abs(hue(out) - hue(photo)) < abs(hue(green) - hue(photo)) / 2 + 1
    gray = torch.full((1, 3, 4, 4), 0.5)
    out = skin_finish(gray, photo, torch.tensor([0.25]), torch.tensor([1.0]))
    assert (out[0, 0] - out[0, 1]).abs().max() < 0.01  # B&W stays gray, only lighter
    assert out.mean() > 0.5


def test_subject_cast_only_on_the_person():
    m = torch.zeros(1, 6, 32, 32)
    m[:, 1, 8:24, 8:24] = 1.0
    photo = torch.full((1, 3, 32, 32), 0.5)
    p = torch.zeros(1, N_FAULT)
    p[0, 6:8] = torch.tensor([-9.0, 4.0])  # green bounce
    out = apply_local_fault(photo, p, m)
    assert out[0, 1, 16, 16] > out[0, 0, 16, 16] + 0.02       # person: green
    assert (out[0, :, 0, 0] - 0.5).abs().max() < 0.01           # background untouched


def test_person_crop_contains_the_person():
    from grademl.local_looks import LocalGradePairs
    ds = LocalGradePairs.__new__(LocalGradePairs)
    maps = torch.zeros(6, 300, 400)
    maps[1, 100:200, 250:300] = 1.0
    t, l, h, w = ds._person_box(maps, random.Random(0))
    assert t <= 100 and t + h >= 200 and l <= 250 and l + w >= 300
    assert 0.74 <= h / w <= 1.34
