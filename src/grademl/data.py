"""Datasets."""
from __future__ import annotations

import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image

from .model import INPUT_SIZE


def data_root() -> Path:
    return Path(os.environ.get("GRADEML_DATA", Path.home() / "grademl-data"))


def load_rgb(path: Path) -> torch.Tensor:
    """(3, H, W) float in [0, 1]. Handles 8- and 16-bit PNGs (FiveK is 16-bit)."""
    img = read_image(str(path), ImageReadMode.RGB)
    if img.dtype == torch.uint16:
        return img.to(torch.int32).float() / 65535  # few ops support uint16 directly
    return img.float() / 255


def resize(img: torch.Tensor, size: int = INPUT_SIZE) -> torch.Tensor:
    """(3, H, W) or (B, 3, H, W) -> square size x size, antialiased.
    Squashing the aspect ratio is fine: the network only needs the color
    statistics, and train and test resize the same way."""
    batched = img.dim() == 4
    out = F.interpolate(img if batched else img[None], size=(size, size),
                        mode="bilinear", antialias=True, align_corners=False)
    return out if batched else out[0]


def random_crop_box(h: int, w: int, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)) -> tuple[int, int, int, int]:
    """RandomResizedCrop-style box (top, left, height, width)."""
    area = h * w
    for _ in range(10):
        a = area * random.uniform(*scale)
        r = random.uniform(*ratio)
        cw, ch = round((a * r) ** 0.5), round((a / r) ** 0.5)
        if 0 < cw <= w and 0 < ch <= h:
            return random.randint(0, h - ch), random.randint(0, w - cw), ch, cw
    return 0, 0, h, w


def jitter_input(img: torch.Tensor) -> torch.Tensor:
    """Brightness / saturation jitter on the input only (as in Zeng et al.):
    the target stays fixed, so the net learns to correct exposure and color
    variation instead of memorizing one mapping."""
    img = img * random.uniform(0.8, 1.2)
    gray = (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2])[None]
    img = gray + (img - gray) * random.uniform(0.8, 1.2)
    return img.clamp(0, 1)


class FiveKPairs(Dataset):
    """MIT-Adobe FiveK input / Expert C pairs.

    train: random crop + flip + input jitter, both resized to INPUT_SIZE,
           returns (input, target) for batching.
    test:  full-resolution pair, plus `net_input` (resized) for the model,
           returns (net_input, input, target). Use batch_size=1 (sizes vary).
    """

    def __init__(self, split: str, root: Path | None = None, target: str = "expertC"):
        self.root = (root or data_root()) / "fivek"
        self.names = (self.root / f"{split}.txt").read_text().split()
        self.train = split == "train"
        self.target = target

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int):
        name = self.names[i]
        inp = load_rgb(self.root / "input" / name)
        tgt = load_rgb(self.root / self.target / name)
        if not self.train:
            return resize(inp), inp, tgt
        t, l, h, w = random_crop_box(*inp.shape[1:])
        inp, tgt = inp[:, t:t + h, l:l + w], tgt[:, t:t + h, l:l + w]
        if random.random() < 0.5:
            inp, tgt = inp.flip(-1), tgt.flip(-1)
        return jitter_input(resize(inp)), resize(tgt)
