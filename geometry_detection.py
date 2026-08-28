"""Reusable two-view feature and epipolar geometry for Step 6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageScale:
    """Explicit mapping between analysis pixels and original-image pixels."""

    original_size: tuple[int, int]
    analysis_size: tuple[int, int]
    scale_x_to_original: float
    scale_y_to_original: float

    def points_to_original(self, points: np.ndarray) -> np.ndarray:
        converted = np.asarray(points, dtype=np.float64).copy()
        converted[..., 0] *= self.scale_x_to_original
        converted[..., 1] *= self.scale_y_to_original
        return converted

    def points_to_analysis(self, points: np.ndarray) -> np.ndarray:
        converted = np.asarray(points, dtype=np.float64).copy()
        converted[..., 0] /= self.scale_x_to_original
        converted[..., 1] /= self.scale_y_to_original
        return converted


@dataclass(frozen=True)
class SiftConfig:
    maximum_width: int = 1200
    nfeatures: int = 8000
    ratio_threshold: float = 0.75
    minimum_correspondences: int = 8
    ransac_threshold: float = 1.5
    confidence: float = 0.99
    rng_seed: int = 4213


@dataclass(frozen=True)
class SiftFeatures:
    analysis_image: np.ndarray
    keypoints: tuple[cv2.KeyPoint, ...]
    descriptors: np.ndarray | None
    scale: ImageScale
    status: str

    @property
    def keypoint_count(self) -> int:
        return len(self.keypoints)


@dataclass(frozen=True)
class GeometryMatchResult:
    fundamental_matrix: np.ndarray | None
    candidate_matches: tuple[cv2.DMatch, ...]
    inlier_mask: np.ndarray
    inlier_matches: tuple[cv2.DMatch, ...]
    points_a: np.ndarray
    points_b: np.ndarray
    status: str

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_matches)

    @property
    def inlier_count(self) -> int:
        return len(self.inlier_matches)

    @property
    def inlier_ratio(self) -> float:
        if not self.candidate_matches:
            return 0.0
        return self.inlier_count / self.candidate_count


def _validate_image(image: np.ndarray) -> None:
    if image.ndim not in {2, 3} or image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError("image must be a non-empty grayscale or color array")


def prepare_analysis_image(
    image: np.ndarray, maximum_width: int = 1200
) -> tuple[np.ndarray, ImageScale]:
    """Create an aspect-preserving in-memory copy and explicit scale metadata."""
    _validate_image(image)
    if maximum_width < 1:
        raise ValueError("maximum_width must be positive")
    original_height, original_width = image.shape[:2]
    if original_width <= maximum_width:
        analysis = image.copy()
    else:
        resize_scale = maximum_width / original_width
        analysis = cv2.resize(
            image,
            (maximum_width, round(original_height * resize_scale)),
            interpolation=cv2.INTER_AREA,
        )
    analysis_height, analysis_width = analysis.shape[:2]
    scale = ImageScale(
        original_size=(original_width, original_height),
        analysis_size=(analysis_width, analysis_height),
        scale_x_to_original=original_width / analysis_width,
        scale_y_to_original=original_height / analysis_height,
    )
    return analysis, scale


def extract_sift(
    image: np.ndarray, config: SiftConfig = SiftConfig()
) -> SiftFeatures:
    """Extract SIFT features at the documented analysis scale."""
    analysis, scale = prepare_analysis_image(image, config.maximum_width)
    if analysis.ndim == 2:
        gray = analysis
    else:
        gray = cv2.cvtColor(analysis, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=config.nfeatures)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    keypoint_tuple = tuple(keypoints or ())
    if descriptors is None or not keypoint_tuple:
        return SiftFeatures(
            analysis_image=analysis,
            keypoints=keypoint_tuple,
            descriptors=None,
            scale=scale,
            status="descriptors_unavailable",
        )
    return SiftFeatures(
        analysis_image=analysis,
        keypoints=keypoint_tuple,
        descriptors=np.asarray(descriptors, dtype=np.float32),
        scale=scale,
        status="ok",
    )


def match_sift(
    first: SiftFeatures,
    second: SiftFeatures,
    ratio_threshold: float = 0.75,
) -> tuple[cv2.DMatch, ...]:
    """Run BF-L2 two-neighbor matching and the Lowe ratio filter."""
    if not 0.0 < ratio_threshold < 1.0:
        raise ValueError("ratio_threshold must be between zero and one")
    if first.descriptors is None or second.descriptors is None:
        return ()
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    neighbors = matcher.knnMatch(first.descriptors, second.descriptors, k=2)
    matches = tuple(
        pair[0]
        for pair in neighbors
        if len(pair) == 2 and pair[0].distance < ratio_threshold * pair[1].distance
    )
    for match in matches:
        if not 0 <= match.queryIdx < len(first.keypoints):
            raise ValueError("SIFT match query index is out of range")
        if not 0 <= match.trainIdx < len(second.keypoints):
            raise ValueError("SIFT match train index is out of range")
    return matches


def _candidate_points(
    first: SiftFeatures,
    second: SiftFeatures,
    matches: Sequence[cv2.DMatch],
) -> tuple[np.ndarray, np.ndarray]:
    first_points: list[tuple[float, float]] = []
    second_points: list[tuple[float, float]] = []
    for match in matches:
        if not 0 <= match.queryIdx < len(first.keypoints):
            raise ValueError("SIFT match query index is out of range")
        if not 0 <= match.trainIdx < len(second.keypoints):
            raise ValueError("SIFT match train index is out of range")
        first_points.append(first.keypoints[match.queryIdx].pt)
        second_points.append(second.keypoints[match.trainIdx].pt)
    return (
        np.asarray(first_points, dtype=np.float32).reshape(-1, 2),
        np.asarray(second_points, dtype=np.float32).reshape(-1, 2),
    )


def _failed_geometry(
    matches: tuple[cv2.DMatch, ...],
    points_a: np.ndarray,
    points_b: np.ndarray,
    status: str,
) -> GeometryMatchResult:
    return GeometryMatchResult(
        fundamental_matrix=None,
        candidate_matches=matches,
        inlier_mask=np.zeros(len(matches), dtype=bool),
        inlier_matches=(),
        points_a=points_a,
        points_b=points_b,
        status=status,
    )


def estimate_fundamental_geometry(
    first: SiftFeatures,
    second: SiftFeatures,
    matches: Sequence[cv2.DMatch],
    *,
    ransac_threshold: float = 1.5,
    confidence: float = 0.99,
    rng_seed: int = 4213,
    minimum_correspondences: int = 8,
) -> GeometryMatchResult:
    """Estimate one Fundamental Matrix and retain its exact RANSAC inlier set."""
    candidate_matches = tuple(matches)
    points_a, points_b = _candidate_points(first, second, candidate_matches)
    if len(candidate_matches) < minimum_correspondences:
        return _failed_geometry(
            candidate_matches, points_a, points_b, "insufficient_geometry"
        )

    cv2.setRNGSeed(rng_seed)
    try:
        fundamental, mask = cv2.findFundamentalMat(
            points_a,
            points_b,
            cv2.FM_RANSAC,
            ransac_threshold,
            confidence,
        )
    except cv2.error:
        fundamental, mask = None, None
    if (
        fundamental is None
        or np.asarray(fundamental).shape != (3, 3)
        or not np.isfinite(fundamental).all()
        or mask is None
        or np.asarray(mask).size != len(candidate_matches)
    ):
        return _failed_geometry(
            candidate_matches,
            points_a,
            points_b,
            "degenerate_fundamental_matrix",
        )

    inlier_mask = np.asarray(mask).reshape(-1).astype(bool)
    inlier_matches = tuple(
        match
        for match, is_inlier in zip(candidate_matches, inlier_mask, strict=True)
        if is_inlier
    )
    if not inlier_matches:
        return _failed_geometry(
            candidate_matches, points_a, points_b, "no_ransac_inliers"
        )
    return GeometryMatchResult(
        fundamental_matrix=np.asarray(fundamental, dtype=np.float64),
        candidate_matches=candidate_matches,
        inlier_mask=inlier_mask,
        inlier_matches=inlier_matches,
        points_a=points_a,
        points_b=points_b,
        status="ok",
    )


def _valid_fundamental_matrix(fundamental: np.ndarray | None) -> np.ndarray:
    if (
        fundamental is None
        or np.asarray(fundamental).shape != (3, 3)
        or not np.isfinite(fundamental).all()
    ):
        raise ValueError("a finite 3x3 fundamental matrix is required")
    return np.asarray(fundamental, dtype=np.float64)


def _point_array(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError("points must be a finite Nx2 array")
    return array


def compute_epilines(
    fundamental: np.ndarray | None,
    points: np.ndarray,
    *,
    which_image: int,
) -> np.ndarray:
    """Compute epilines in the opposite image from one verified `F`."""
    matrix = _valid_fundamental_matrix(fundamental)
    point_array = _point_array(points)
    if which_image not in {1, 2}:
        raise ValueError("which_image must be 1 or 2")
    lines = cv2.computeCorrespondEpilines(
        point_array.astype(np.float32).reshape(-1, 1, 2),
        which_image,
        matrix,
    )
    line_array = np.asarray(lines, dtype=np.float64).reshape(-1, 3)
    if len(line_array) != len(point_array) or not np.isfinite(line_array).all():
        raise ValueError("OpenCV returned invalid epipolar lines")
    return line_array


def sampson_errors(
    fundamental: np.ndarray | None,
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> np.ndarray:
    """Return first-order geometric residuals in analysis-pixel units squared."""
    matrix = _valid_fundamental_matrix(fundamental)
    first = _point_array(points_a)
    second = _point_array(points_b)
    if len(first) != len(second):
        raise ValueError("Sampson point arrays must have matching lengths")
    first_h = np.column_stack((first, np.ones(len(first))))
    second_h = np.column_stack((second, np.ones(len(second))))
    first_lines = (matrix @ first_h.T).T
    second_lines = (matrix.T @ second_h.T).T
    numerators = np.sum(second_h * first_lines, axis=1) ** 2
    denominators = (
        first_lines[:, 0] ** 2
        + first_lines[:, 1] ** 2
        + second_lines[:, 0] ** 2
        + second_lines[:, 1] ** 2
    )
    errors = np.full(len(first), np.inf, dtype=np.float64)
    valid = denominators > np.finfo(np.float64).eps
    errors[valid] = numerators[valid] / denominators[valid]
    return errors


def clip_epiline_to_image(
    line: np.ndarray, image_size: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Clip `ax + by + c = 0` to the inclusive image border."""
    coefficients = np.asarray(line, dtype=np.float64).reshape(-1)
    width, height = image_size
    if (
        coefficients.size != 3
        or not np.isfinite(coefficients).all()
        or width < 1
        or height < 1
    ):
        return None
    a, b, c = coefficients
    epsilon = np.finfo(np.float64).eps * 100
    intersections: list[tuple[int, int]] = []

    def add_point(x: float, y: float) -> None:
        if -1e-7 <= x <= width - 1 + 1e-7 and -1e-7 <= y <= height - 1 + 1e-7:
            point = (
                int(np.clip(round(x), 0, width - 1)),
                int(np.clip(round(y), 0, height - 1)),
            )
            if point not in intersections:
                intersections.append(point)

    if abs(b) > epsilon:
        add_point(0.0, -c / b)
        add_point(float(width - 1), -(a * (width - 1) + c) / b)
    if abs(a) > epsilon:
        add_point(-c / a, 0.0)
        add_point(-(b * (height - 1) + c) / a, float(height - 1))
    if len(intersections) < 2:
        return None
    return intersections[0], intersections[1]


def _spatial_sample_indices(
    points_a: np.ndarray,
    points_b: np.ndarray,
    eligible_indices: np.ndarray,
    max_points: int,
) -> np.ndarray:
    if max_points < 1:
        raise ValueError("max_points must be positive")
    if len(eligible_indices) <= max_points:
        return eligible_indices.copy()
    combined = np.column_stack((points_a[eligible_indices], points_b[eligible_indices]))
    minimum = combined.min(axis=0)
    span = combined.max(axis=0) - minimum
    span[span == 0.0] = 1.0
    normalized = (combined - minimum) / span
    centroid = normalized.mean(axis=0)
    selected_local = [int(np.argmax(np.sum((normalized - centroid) ** 2, axis=1)))]
    minimum_distances = np.sum(
        (normalized - normalized[selected_local[0]]) ** 2, axis=1
    )
    while len(selected_local) < max_points:
        minimum_distances[selected_local] = -1.0
        next_local = int(np.argmax(minimum_distances))
        selected_local.append(next_local)
        new_distances = np.sum(
            (normalized - normalized[next_local]) ** 2, axis=1
        )
        minimum_distances = np.minimum(minimum_distances, new_distances)
    return eligible_indices[np.asarray(selected_local, dtype=np.int64)]


def select_spatial_inliers(
    points_a: np.ndarray,
    points_b: np.ndarray,
    inlier_mask: np.ndarray,
    *,
    max_points: int = 10,
) -> np.ndarray:
    """Select deterministic, spatially spread indices from an inlier mask."""
    first = _point_array(points_a)
    second = _point_array(points_b)
    mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    if len(first) != len(second) or len(first) != len(mask):
        raise ValueError("points and inlier mask must have matching lengths")
    eligible = np.flatnonzero(mask)
    return _spatial_sample_indices(first, second, eligible, max_points)


def select_display_matches(
    first: SiftFeatures,
    second: SiftFeatures,
    matches: Sequence[cv2.DMatch],
    *,
    max_matches: int = 60,
) -> tuple[cv2.DMatch, ...]:
    """Return a deterministic spatial subset for drawing only."""
    measured = tuple(matches)
    points_a, points_b = _candidate_points(first, second, measured)
    indices = _spatial_sample_indices(
        points_a,
        points_b,
        np.arange(len(measured), dtype=np.int64),
        max_matches,
    )
    return tuple(measured[int(index)] for index in indices)
