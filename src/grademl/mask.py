"""Person masks: separate the subject from the background.

The LUT model stays global; subject-aware grading is a separate step that
needs a soft mask (1 = person, 0 = background). Where the mask comes from
depends on the platform, the compositing rule does not (grademl.compose):

  Python (this module)  torchvision DeepLabV3 (COCO, VOC "person" class),
                        weights downloaded once into the torch hub cache,
                        never committed
  iOS (Grade Studio)    Vision VNGeneratePersonSegmentationRequest, built
                        into the OS (ios/GradeML.swift, SubjectMask)

Both give a coarse mask; `refine_mask` snaps its edges to the photo with a
guided filter (He et al. 2010), the same filter iOS has as CIGuidedFilter.

DeepLabV3 instead of a selfie segmenter: photos have groups and people far
from the camera, where selfie models (trained on faces close to the lens)
miss. It also adds no dependency (torchvision is already installed).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.io import ImageReadMode, read_image

PERSON = 15             # index of "person" in the 21 VOC classes torchvision's DeepLab predicts
SEG_SIZE = 520          # short side the torchvision weights were trained at
MIN_COVERAGE = 0.01     # below this the photo has no subject: grade it uniformly
MODELS = ("resnet101", "mobilenet")  # first = default

_cache: dict[tuple[str, str], torch.nn.Module] = {}


def _load(model: str, device: torch.device) -> torch.nn.Module:
    key = (model, str(device))
    if key not in _cache:
        from torchvision.models import segmentation as seg
        if model == "resnet101":    # ~240 MB, clearly cleaner edges (hair, hands, arms)
            net = seg.deeplabv3_resnet101(weights=seg.DeepLabV3_ResNet101_Weights.DEFAULT)
        elif model == "mobilenet":  # ~42 MB, faster on CPU, blobbier edges
            net = seg.deeplabv3_mobilenet_v3_large(weights=seg.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
        else:
            raise ValueError(f"unknown model {model!r}, expected one of {MODELS}")
        _cache[key] = net.eval().to(device)
    return _cache[key]


@torch.no_grad()
def person_probability(img: torch.Tensor, model: str = MODELS[0]) -> torch.Tensor:
    """img: (3, H, W) RGB in [0, 1] -> (1, h, w) person probability at the
    segmentation resolution (short side SEG_SIZE). Coarse; see refine_mask."""
    device = img.device
    h, w = img.shape[-2:]
    s = SEG_SIZE / min(h, w)
    x = F.interpolate(img[None], size=(max(1, round(h * s)), max(1, round(w * s))),
                      mode="bilinear", antialias=True, align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    logits = _load(model, device)((x - mean) / std)["out"]
    return logits.softmax(1)[:, PERSON]


def _box(x: torch.Tensor, r: int) -> torch.Tensor:
    """Mean over a (2r+1)^2 window, edges handled by renormalizing."""
    return F.avg_pool2d(x, 2 * r + 1, 1, r, count_include_pad=False) if r > 0 else x


def guided_filter(src: torch.Tensor, guide: torch.Tensor, r: int, eps: float) -> torch.Tensor:
    """Gray-guide guided filter. src, guide: (B, 1, H, W). The output follows
    the guide's edges where src changes, and is smooth elsewhere."""
    mean_i, mean_p = _box(guide, r), _box(src, r)
    cov = _box(guide * src, r) - mean_i * mean_p
    var = _box(guide * guide, r) - mean_i**2
    a = cov / (var + eps)
    b = mean_p - a * mean_i
    return _box(a, r) * guide + _box(b, r)


def refine_mask(prob: torch.Tensor, img: torch.Tensor, work: int = 1024,
                radius: float = 0.03, eps: float = 1e-4) -> torch.Tensor:
    """Upsample a coarse mask to the photo and align its edges to it.

    prob: (1, h, w) in [0, 1]; img: (3, H, W). The guided filter runs at
    `work` px (long side) with a radius relative to that size, so the result
    looks the same for any photo resolution; then bilinear to full size.
    Returns (1, H, W) in [0, 1].
    """
    H, W = img.shape[-2:]
    s = min(1.0, work / max(H, W))
    wh, ww = max(1, round(H * s)), max(1, round(W * s))
    small = F.interpolate(img[None], size=(wh, ww), mode="bilinear", antialias=True, align_corners=False)
    guide = (small * torch.tensor([0.2126, 0.7152, 0.0722], device=img.device)[:, None, None]).sum(1, keepdim=True)
    p = F.interpolate(prob[None], size=(wh, ww), mode="bilinear", align_corners=False)
    r = max(1, round(radius * max(wh, ww)))
    p = guided_filter(p, guide, r, eps).clamp(0, 1)
    return F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)[0].clamp(0, 1)


def person_mask(img: torch.Tensor, model: str = MODELS[0]) -> torch.Tensor:
    """(3, H, W) photo -> (1, H, W) soft person mask in [0, 1]."""
    return refine_mask(person_probability(img, model), img)


def coverage(mask: torch.Tensor) -> float:
    """Share of the frame covered by the subject."""
    return float(mask.mean())


def load_mask(path: str | Path, size: tuple[int, int] | None = None) -> torch.Tensor:
    """Read a grayscale mask image (e.g. one saved by the iOS app) as (1, H, W)."""
    m = read_image(str(path), ImageReadMode.GRAY).float() / 255
    if size is not None and tuple(m.shape[-2:]) != tuple(size):
        m = F.interpolate(m[None], size=size, mode="bilinear", align_corners=False)[0]
    return m


def mask_to_image(mask: torch.Tensor) -> torch.Tensor:
    """(1, H, W) -> uint8 (1, H, W), for torchvision.io.write_png."""
    return (mask.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
