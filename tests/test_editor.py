import torch

from grademl.editor import local_edit, local_top_versions
from grademl.grade import STYLES
from grademl.local import LocalGradeNet


def _net():
    torch.manual_seed(0)
    net = LocalGradeNet(k=2, n_luts=4, lut_size=9, width=8, hidden=32, cond_dim=len(STYLES)).eval()
    with torch.no_grad():
        net.luts.basis.normal_(std=0.05)
    return net


def test_local_edit_intensity_zero_is_the_correction():
    net, img = _net(), torch.rand(3, 64, 48)
    e = local_edit(net, img, "portra")
    full, none = e.render(img, 1.0), e.render(img, 0.0)
    corr = local_edit(net, img, "none").render(img)
    assert full.shape == img.shape
    assert torch.allclose(none, corr, atol=1e-5)


def test_top_versions_distinct_styles():
    edits = local_top_versions(torch.rand(3, 80, 120), _net(), n=3)
    assert [e.rank for e in edits] == [1, 2, 3]
    assert len({e.style for e in edits}) == 3
    assert all(e.reasons for e in edits)


def test_top_versions_come_from_different_families():
    from grademl.editor import FAMILIES
    edits = local_top_versions(torch.rand(3, 80, 120), _net(), n=3, calibration=None)
    assert len({FAMILIES[e.style] for e in edits}) == 3
    assert edits[0].shortlist and edits[0].shortlist[0].score >= edits[-1].best.score


def test_keep_penalizes_draining_a_colorful_photo():
    from grademl.editor import score_candidates
    img = torch.zeros(3, 64, 64)
    img[0, :, :32], img[1, :, 32:] = 0.9, 0.8  # strongly colored
    ranked, _ = score_candidates(img, _net(), calibration=None)
    keep = {(c.style, c.intensity): c.parts["keep"] for c in ranked}
    assert all(v <= 0 for v in keep.values()) and keep[("none", 1.0)] == 0


def test_calibration_levels_the_parts():
    from grademl.editor import score_candidates
    net, img = _net(), torch.rand(3, 64, 64)
    raw, _ = score_candidates(img, net, calibration=None)
    cal = {s: {"critic": [-5.0 if s == "bw_toned" else -1.0, 1.0], "bold": [1.0, 1.0]}
           for s in {c.style for c in raw}}
    calibrated, _ = score_candidates(img, net, calibration=cal)
    r = {(c.style, c.intensity): c.parts["critic"] for c in raw}
    c = {(x.style, x.intensity): x.parts["critic"] for x in calibrated}
    k = ("bw_toned", 1.0)
    # bw usually scores -5 on the critic, the others -1: its critic part rises by 4 relative to them
    assert abs((c[k] - r[k]) - (c[("portra", 1.0)] - r[("portra", 1.0)]) - 4.0) < 1e-4
