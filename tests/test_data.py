import cv2
import numpy as np
import torch

from grademl.data import load_rgb
from grademl.metrics import delta_e76, psnr


def test_load_rgb_8_and_16_bit(tmp_path):
    rgb8 = np.random.default_rng(0).integers(0, 256, (4, 5, 3), dtype=np.uint8)
    rgb16 = rgb8.astype(np.uint16) * 257  # same colors at 16-bit
    cv2.imwrite(str(tmp_path / "a.png"), rgb8[..., ::-1])
    cv2.imwrite(str(tmp_path / "b.png"), rgb16[..., ::-1])
    a, b = load_rgb(tmp_path / "a.png"), load_rgb(tmp_path / "b.png")
    assert a.shape == b.shape == (3, 4, 5)
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(a, torch.from_numpy(rgb8).permute(2, 0, 1).float() / 255)


def test_metrics_sanity():
    x = torch.rand(2, 3, 8, 8)
    assert torch.all(psnr(x, x) > 90)
    assert torch.allclose(delta_e76(x, x), torch.zeros(2), atol=1e-3)
    white, black = torch.ones(1, 3, 2, 2), torch.zeros(1, 3, 2, 2)
    assert abs(delta_e76(white, black).item() - 100) < 0.1  # L* 100 vs 0
