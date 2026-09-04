"""Small from-scratch CNN and binary segmentation math for the vessel masks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SmallSegCNN(nn.Module):
    """Compact U-Net-like binary segmenter initialized only by PyTorch defaults."""

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = ConvBlock(3, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(64, 128)
        self.dec3 = ConvBlock(128 + 64, 64)
        self.dec2 = ConvBlock(64 + 32, 32)
        self.dec1 = ConvBlock(32 + 16, 16)
        self.output = nn.Conv2d(16, 1, kernel_size=1)

    @staticmethod
    def _upsample(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, size=reference.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bottleneck = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat((self._upsample(bottleneck, e3), e3), dim=1))
        d2 = self.dec2(torch.cat((self._upsample(d3, e2), e2), dim=1))
        d1 = self.dec1(torch.cat((self._upsample(d2, e1), e1), dim=1))
        return self.output(d1)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    dims = tuple(range(1, probabilities.ndim))
    intersection = torch.sum(probabilities * target, dim=dims)
    denominator = torch.sum(probabilities, dim=dims) + torch.sum(target, dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target) + soft_dice_loss(logits, target)


@dataclass(frozen=True)
class SegmentationMetrics:
    dice: float
    iou: float
    precision: float
    recall: float
    pixel_accuracy: float
    foreground_fraction: float


def binary_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> SegmentationMetrics:
    """Calculate foreground metrics from binary 0/1 tensors."""
    pred = prediction.detach().bool().reshape(-1)
    truth = target.detach().bool().reshape(-1)
    if pred.numel() != truth.numel():
        raise ValueError("prediction and target must have the same number of pixels")

    tp = float(torch.sum(pred & truth).item())
    fp = float(torch.sum(pred & ~truth).item())
    fn = float(torch.sum(~pred & truth).item())
    tn = float(torch.sum(~pred & ~truth).item())

    dice_den = 2.0 * tp + fp + fn
    iou_den = tp + fp + fn
    precision_den = tp + fp
    recall_den = tp + fn
    dice = 1.0 if dice_den <= eps else (2.0 * tp) / dice_den
    iou = 1.0 if iou_den <= eps else tp / iou_den
    precision = 1.0 if precision_den <= eps else tp / precision_den
    recall = 1.0 if recall_den <= eps else tp / recall_den
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    foreground_fraction = float(torch.mean(pred.float()).item())
    return SegmentationMetrics(
        dice=dice,
        iou=iou,
        precision=precision,
        recall=recall,
        pixel_accuracy=accuracy,
        foreground_fraction=foreground_fraction,
    )


def logits_to_binary(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between zero and one")
    return (torch.sigmoid(logits) >= threshold).to(torch.uint8)
