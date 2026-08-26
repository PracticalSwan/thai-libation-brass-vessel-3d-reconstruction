"""Image-quality metrics and explainable ACCEPT/WARN/REJECT decisions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Sequence

import cv2
import numpy as np


DEFAULT_GEOMETRY_EXCLUSIONS = frozenset(range(289, 298))


@dataclass(frozen=True)
class QualityRecord:
    index: int
    filename: str
    readable: bool
    width: int = 0
    height: int = 0
    brightness: float = 0.0
    contrast: float = 0.0
    blur_score: float = 0.0
    dark_percent: float = 0.0
    bright_percent: float = 0.0
    feature_count: int = 0
    error: str = ""

    @classmethod
    def unreadable(cls, *, index: int, filename: str) -> "QualityRecord":
        return cls(
            index=index,
            filename=filename,
            readable=False,
            error="unreadable image",
        )


@dataclass(frozen=True)
class QualityThresholds:
    blur_warn: float
    blur_reject: float
    brightness_low_warn: float
    brightness_high_warn: float
    contrast_warn: float
    dark_percent_warn: float
    bright_percent_warn: float
    feature_count_warn: int
    feature_count_reject: int


@dataclass(frozen=True)
class QualityDecision:
    label: str
    reasons: tuple[str, ...]


def _analysis_image(image: np.ndarray, analysis_width: int) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected a BGR image with three channels")
    if image.dtype != np.uint8:
        raise ValueError("expected a uint8 image")
    height, width = image.shape[:2]
    if width <= analysis_width:
        return image
    scale = analysis_width / width
    return cv2.resize(
        image,
        (analysis_width, round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def analyze_image_array(
    image: np.ndarray,
    *,
    index: int,
    filename: str,
    analysis_width: int = 800,
) -> QualityRecord:
    """Measure one BGR image on a standardized analysis scale."""
    height, width = image.shape[:2]
    analysis = _analysis_image(image, analysis_width)
    gray = cv2.cvtColor(analysis, cv2.COLOR_BGR2GRAY)
    sift = getattr(cv2, "SIFT_create")(nfeatures=5000)
    keypoints = sift.detect(gray, None)

    return QualityRecord(
        index=index,
        filename=filename,
        readable=True,
        width=width,
        height=height,
        brightness=float(np.mean(gray)),
        contrast=float(np.std(gray)),
        blur_score=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        dark_percent=float(np.mean(gray <= 5) * 100.0),
        bright_percent=float(np.mean(gray >= 250) * 100.0),
        feature_count=len(keypoints),
    )


def analyze_image(
    path: Path,
    *,
    index: int,
    analysis_width: int = 800,
) -> QualityRecord:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return QualityRecord.unreadable(index=index, filename=path.name)
    return analyze_image_array(
        image,
        index=index,
        filename=path.name,
        analysis_width=analysis_width,
    )


def calibrate_thresholds(
    records: Sequence[QualityRecord],
    *,
    excluded_indices: Collection[int] = DEFAULT_GEOMETRY_EXCLUSIONS,
) -> QualityThresholds:
    """Derive warning and severe-failure cutoffs from eligible captures."""
    eligible = [
        record
        for record in records
        if record.readable and record.index not in excluded_indices
    ]
    if not eligible:
        raise ValueError("cannot calibrate thresholds without readable images")

    def percentile(field: str, value: float) -> float:
        values = [float(getattr(record, field)) for record in eligible]
        return float(np.percentile(values, value))

    median_blur = percentile("blur_score", 50)
    median_features = percentile("feature_count", 50)
    return QualityThresholds(
        blur_warn=percentile("blur_score", 5),
        blur_reject=min(percentile("blur_score", 1), median_blur * 0.25),
        brightness_low_warn=percentile("brightness", 5),
        brightness_high_warn=percentile("brightness", 95),
        contrast_warn=percentile("contrast", 5),
        dark_percent_warn=percentile("dark_percent", 95),
        bright_percent_warn=percentile("bright_percent", 95),
        feature_count_warn=round(percentile("feature_count", 5)),
        feature_count_reject=round(median_features * 0.25),
    )


def decide_quality(
    record: QualityRecord,
    thresholds: QualityThresholds,
    *,
    excluded_indices: Collection[int] = DEFAULT_GEOMETRY_EXCLUSIONS,
) -> QualityDecision:
    """Turn metrics plus known capture geometry into an explainable decision."""
    if not record.readable:
        return QualityDecision("REJECT", (record.error or "unreadable image",))
    if record.index in excluded_indices:
        return QualityDecision(
            "REJECT", ("separate hand-held/flipped sequence",)
        )
    if (
        record.blur_score < thresholds.blur_reject
        and record.feature_count < thresholds.feature_count_reject
    ):
        return QualityDecision(
            "REJECT", ("severe blur and too few local features",)
        )

    reasons: list[str] = []
    if record.blur_score < thresholds.blur_warn:
        reasons.append("low sharpness")
    if record.brightness < thresholds.brightness_low_warn:
        reasons.append("low brightness")
    elif record.brightness > thresholds.brightness_high_warn:
        reasons.append("high brightness")
    if record.contrast < thresholds.contrast_warn:
        reasons.append("low contrast")
    if record.dark_percent > thresholds.dark_percent_warn:
        reasons.append("dark clipping outlier")
    if record.bright_percent > thresholds.bright_percent_warn:
        reasons.append("bright clipping outlier")
    if record.feature_count < thresholds.feature_count_warn:
        reasons.append("low local-feature count")

    if reasons:
        return QualityDecision("WARN", tuple(reasons))
    return QualityDecision("ACCEPT", ())
