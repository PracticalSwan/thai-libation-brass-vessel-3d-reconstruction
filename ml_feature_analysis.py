"""Step 8: classify the existing Step 6 SIFT features by CNN-predicted masks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from geometry_detection import SiftFeatures, extract_sift


@dataclass(frozen=True)
class FeatureMaskResult:
    total_keypoints: int
    vessel_keypoints: int
    background_keypoints: int
    vessel_feature_fraction: float
    background_feature_fraction: float
    mask_foreground_fraction: float
    vessel_indices: tuple[int, ...]
    background_indices: tuple[int, ...]


def classify_keypoints_by_mask(
    features: SiftFeatures,
    full_resolution_mask: np.ndarray,
) -> FeatureMaskResult:
    """Map analysis-pixel SIFT points through explicit Step 6 scale metadata."""
    if full_resolution_mask.ndim != 2:
        raise ValueError("feature mask must be a 2D binary array")
    original_width, original_height = features.scale.original_size
    if full_resolution_mask.shape != (original_height, original_width):
        raise ValueError(
            "feature mask size does not match SIFT original size: "
            f"mask={full_resolution_mask.shape[::-1]}, "
            f"expected={features.scale.original_size}"
        )
    values = set(int(value) for value in np.unique(full_resolution_mask))
    if not values.issubset({0, 255}):
        raise ValueError("feature mask must contain only 0 and 255")
    if features.descriptors is not None and len(features.descriptors) != len(features.keypoints):
        raise ValueError("SIFT descriptor/keypoint count mismatch")

    vessel: list[int] = []
    background: list[int] = []
    for index, keypoint in enumerate(features.keypoints):
        x = int(round(keypoint.pt[0] * features.scale.scale_x_to_original))
        y = int(round(keypoint.pt[1] * features.scale.scale_y_to_original))
        x = min(max(x, 0), original_width - 1)
        y = min(max(y, 0), original_height - 1)
        if full_resolution_mask[y, x] > 0:
            vessel.append(index)
        else:
            background.append(index)

    total = len(features.keypoints)
    vessel_count = len(vessel)
    background_count = len(background)
    vessel_fraction = vessel_count / total if total else 0.0
    background_fraction = background_count / total if total else 0.0
    return FeatureMaskResult(
        total_keypoints=total,
        vessel_keypoints=vessel_count,
        background_keypoints=background_count,
        vessel_feature_fraction=vessel_fraction,
        background_feature_fraction=background_fraction,
        mask_foreground_fraction=float(np.count_nonzero(full_resolution_mask))
        / float(full_resolution_mask.size),
        vessel_indices=tuple(vessel),
        background_indices=tuple(background),
    )


def analyze_image_features(
    image: np.ndarray,
    full_resolution_prediction: np.ndarray,
) -> tuple[SiftFeatures, FeatureMaskResult]:
    features = extract_sift(image)
    return features, classify_keypoints_by_mask(features, full_resolution_prediction)


def draw_keypoint_classes(
    features: SiftFeatures,
    result: FeatureMaskResult,
    *,
    max_each: int = 500,
) -> np.ndarray:
    """Draw vessel/background SIFT classes on the Step 6 analysis image."""
    canvas = features.analysis_image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    vessel = result.vessel_indices[:max_each]
    background = result.background_indices[:max_each]
    for index in background:
        x, y = features.keypoints[index].pt
        cv2.circle(canvas, (round(x), round(y)), 2, (255, 120, 40), 1, cv2.LINE_AA)
    for index in vessel:
        x, y = features.keypoints[index].pt
        cv2.circle(canvas, (round(x), round(y)), 2, (40, 220, 40), 1, cv2.LINE_AA)
    return canvas
