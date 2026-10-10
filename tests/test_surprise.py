import torch

from grademl.surprise import FAMILY, Profile, direction_scores, photo_profile, surprise


def sky_photo() -> torch.Tensor:
    img = torch.zeros(3, 64, 64)
    img[:, :40] = torch.tensor([0.35, 0.55, 0.9]).view(3, 1, 1)   # blue sky
    img[:, 40:] = torch.tensor([0.75, 0.65, 0.5]).view(3, 1, 1)   # warm stone
    return img


def test_profile_sees_sky_and_warm():
    p = photo_profile(sky_photo())
    assert p.sky > 0.5 and p.warm > 0.2 and p.dark < 0.05


def test_night_photo_prefers_night_looks():
    night = Profile(brightness=0.15, dark=0.6, sky=0.0, green=0.0, warm=0.2, colorfulness=0.5)
    day = Profile(brightness=0.55, dark=0.05, sky=0.5, green=0.1, warm=0.3, colorfulness=0.6)
    assert direction_scores(night)["cinestill_night"] > direction_scores(day)["cinestill_night"]
    assert direction_scores(day)["golden_hour"] > direction_scores(night)["golden_hour"]


def test_surprise_gives_distinct_families_and_is_reproducible():
    img = sky_photo()
    a = surprise(img, n=3, seed=7, size=17)
    b = surprise(img, n=3, seed=7, size=17)
    assert len(a) == 3
    assert len({FAMILY[s.direction] for s in a}) == 3
    assert [s.direction for s in a] == [s.direction for s in b]
    assert all(torch.equal(x.lut, y.lut) for x, y in zip(a, b))
    assert all(1.0 * 0.85**5 <= s.strength <= 1.3 for s in a)


def test_surprise_varies_with_seed():
    img = sky_photo()
    runs = {tuple(s.direction for s in surprise(img, n=3, seed=k, size=9)) for k in range(6)}
    assert len(runs) > 1
