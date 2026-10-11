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
