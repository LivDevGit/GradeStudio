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
