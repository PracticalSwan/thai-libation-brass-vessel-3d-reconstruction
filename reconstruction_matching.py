"""Step 9B/9C reconstruction-readiness feature matching and connectivity logic."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import median
from typing import Sequence

import cv2
import numpy as np

from analysis_common import SelectedImageRecord
from geometry_detection import (
    SiftConfig,
    SiftFeatures,
    estimate_fundamental_geometry,
    match_sift,
    sampson_errors,
)


BENCHMARK_PAIRS: tuple[tuple[int, int], ...] = (
    (15, 16),
    (45, 46),
    (75, 76),
    (105, 106),
    (135, 136),
    (165, 166),
    (195, 196),
    (225, 226),
    (255, 256),
    (280, 281),
    (15, 17),
    (45, 47),
    (75, 77),
    (105, 107),
    (135, 137),
    (165, 167),
    (195, 197),
    (225, 227),
    (255, 257),
    (280, 282),
)

FEATURE_MODES = ("unmasked", "raw_cnn", "reconstruction_mask")


@dataclass(frozen=True)
class PairGeometryMetrics:
    pair_a: int
    pair_b: int
    mode: str
    keypoints_a: int
    keypoints_b: int
    candidate_matches: int
    inliers: int
    inlier_ratio: float
    median_sampson_error: float
    p90_sampson_error: float
    grid_coverage: float
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SubsetDecision:
    selected_index: int
    filename: str
    include: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_source_mask(features: SiftFeatures, mask: np.ndarray) -> None:
    if mask.ndim != 2:
        raise ValueError("feature mask must be a 2D array")
    original_width, original_height = features.scale.original_size
    if mask.shape != (original_height, original_width):
        raise ValueError(
            "feature mask geometry mismatch: "
            f"expected {(original_width, original_height)}, found {mask.shape[::-1]}"
        )
    values = set(int(value) for value in np.unique(mask))
    if not values.issubset({0, 255}):
        raise ValueError("feature mask must contain only 0 and 255")
    if features.descriptors is not None and len(features.descriptors) != len(features.keypoints):
        raise ValueError("SIFT descriptor/keypoint count mismatch")


def filter_sift_features(features: SiftFeatures, mask: np.ndarray) -> SiftFeatures:
    """Retain existing Step 6 SIFT features whose original-scale pixels are foreground."""
    _validate_source_mask(features, mask)
    original_width, original_height = features.scale.original_size
    keep: list[int] = []
    for index, keypoint in enumerate(features.keypoints):
        x = int(round(keypoint.pt[0] * features.scale.scale_x_to_original))
        y = int(round(keypoint.pt[1] * features.scale.scale_y_to_original))
        x = min(max(x, 0), original_width - 1)
        y = min(max(y, 0), original_height - 1)
        if mask[y, x] > 0:
            keep.append(index)
    keypoints = tuple(features.keypoints[index] for index in keep)
    if features.descriptors is None or not keep:
        descriptors = None
    else:
        descriptors = np.asarray(features.descriptors[keep], dtype=np.float32)
    return SiftFeatures(
        analysis_image=features.analysis_image,
        keypoints=keypoints,
        descriptors=descriptors,
        scale=features.scale,
        status="ok" if descriptors is not None and keypoints else "descriptors_unavailable",
    )


def grid_coverage(
    points: np.ndarray, image_size: tuple[int, int], *, grid_size: int = 4
) -> float:
    """Return fraction of grid cells containing at least one point."""
    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    width, height = image_size
    if width < 1 or height < 1:
        raise ValueError("image_size must be positive")
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not len(array):
        return 0.0
    if not np.isfinite(array).all():
        raise ValueError("grid coverage points must be finite")
    x_cells = np.floor(np.clip(array[:, 0] / width, 0.0, 1.0 - 1e-12) * grid_size).astype(int)
    y_cells = np.floor(np.clip(array[:, 1] / height, 0.0, 1.0 - 1e-12) * grid_size).astype(int)
    occupied = set(zip(x_cells.tolist(), y_cells.tolist(), strict=True))
    return len(occupied) / float(grid_size * grid_size)


def _finite_summary(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return math.nan, math.nan
    return float(np.median(finite)), float(np.percentile(finite, 90))


def measure_pair(
    first: SiftFeatures,
    second: SiftFeatures,
    *,
    pair: tuple[int, int],
    mode: str,
    config: SiftConfig = SiftConfig(),
) -> PairGeometryMetrics:
    if mode not in FEATURE_MODES:
        raise ValueError(f"unknown feature mode: {mode}")
    matches = match_sift(first, second, config.ratio_threshold)
    geometry = estimate_fundamental_geometry(
        first,
        second,
        matches,
        ransac_threshold=config.ransac_threshold,
        confidence=config.confidence,
        rng_seed=config.rng_seed,
        minimum_correspondences=config.minimum_correspondences,
    )
    median_error = math.nan
    p90_error = math.nan
    coverage = 0.0
    if geometry.status == "ok" and geometry.fundamental_matrix is not None:
        inlier_a = geometry.points_a[geometry.inlier_mask]
        inlier_b = geometry.points_b[geometry.inlier_mask]
        errors = sampson_errors(geometry.fundamental_matrix, inlier_a, inlier_b)
        median_error, p90_error = _finite_summary(errors)
        coverage = 0.5 * (
            grid_coverage(inlier_a, first.scale.analysis_size)
            + grid_coverage(inlier_b, second.scale.analysis_size)
        )
    return PairGeometryMetrics(
        pair_a=pair[0],
        pair_b=pair[1],
        mode=mode,
        keypoints_a=first.keypoint_count,
        keypoints_b=second.keypoint_count,
        candidate_matches=geometry.candidate_count,
        inliers=geometry.inlier_count,
        inlier_ratio=geometry.inlier_ratio,
        median_sampson_error=median_error,
        p90_sampson_error=p90_error,
        grid_coverage=coverage,
        status=geometry.status,
    )


def _median_finite(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(median(finite)) if finite else math.nan


def summarize_benchmark(
    rows: Sequence[PairGeometryMetrics],
) -> dict[str, object]:
    if not rows:
        raise ValueError("benchmark rows must not be empty")
    grouped: dict[str, list[PairGeometryMetrics]] = {mode: [] for mode in FEATURE_MODES}
    for row in rows:
        if row.mode not in grouped:
            raise ValueError(f"unknown benchmark mode: {row.mode}")
        grouped[row.mode].append(row)
    if not grouped["unmasked"]:
        raise ValueError("benchmark requires unmasked baseline rows")

    modes: dict[str, dict[str, object]] = {}
    for mode, mode_rows in grouped.items():
        if not mode_rows:
            continue
        modes[mode] = {
            "pair_count": len(mode_rows),
            "total_candidates": sum(row.candidate_matches for row in mode_rows),
            "total_inliers": sum(row.inliers for row in mode_rows),
            "median_inlier_ratio": float(median(row.inlier_ratio for row in mode_rows)),
            "median_sampson_error": _median_finite(
                [row.median_sampson_error for row in mode_rows]
            ),
            "median_grid_coverage": float(median(row.grid_coverage for row in mode_rows)),
            "ok_pair_count": sum(row.status == "ok" for row in mode_rows),
        }

    baseline = modes["unmasked"]
    baseline["qualified"] = True
    for mode in ("raw_cnn", "reconstruction_mask"):
        if mode not in modes:
            continue
        current = modes[mode]
        base_error = float(baseline["median_sampson_error"])
        current_error = float(current["median_sampson_error"])
        error_ok = (
            (not math.isfinite(base_error) and not math.isfinite(current_error))
            or (
                math.isfinite(base_error)
                and math.isfinite(current_error)
                and current_error <= 1.10 * base_error
            )
        )
        current["qualified"] = bool(
            int(current["total_inliers"]) >= 0.95 * int(baseline["total_inliers"])
            and float(current["median_inlier_ratio"])
            >= float(baseline["median_inlier_ratio"])
            and error_ok
        )
    summary: dict[str, object] = {"modes": modes}
    summary["chosen_mode"] = choose_feature_mode(summary)
    return summary


def choose_feature_mode(summary: dict[str, object]) -> str:
    modes = summary.get("modes")
    if not isinstance(modes, dict) or "unmasked" not in modes:
        raise ValueError("benchmark summary is missing unmasked mode")
    qualified: list[tuple[str, dict[str, object]]] = []
    for mode in ("raw_cnn", "reconstruction_mask"):
        entry = modes.get(mode)
        if isinstance(entry, dict) and entry.get("qualified") is True:
            qualified.append((mode, entry))
    if not qualified:
        return "unmasked"
    return max(
        qualified,
        key=lambda item: (
            float(item[1]["median_inlier_ratio"]),
            int(item[1]["total_inliers"]),
        ),
    )[0]


def is_strong_edge(metrics: PairGeometryMetrics) -> bool:
    return bool(
        metrics.status == "ok"
        and metrics.inliers >= 15
        and metrics.inlier_ratio >= 0.15
    )


def adjacent_pairs(indices: Sequence[int]) -> tuple[tuple[int, int], ...]:
    values = tuple(int(value) for value in indices)
    if len(values) < 2:
        return ()
    if len(set(values)) != len(values):
        raise ValueError("connectivity indices must be unique")
    return tuple(zip(values[:-1], values[1:], strict=True))


def bridge_pairs_for_weak_edges(
    indices: Sequence[int], edge_rows: Sequence[PairGeometryMetrics]
) -> tuple[tuple[int, int], ...]:
    values = tuple(int(value) for value in indices)
    position = {value: index for index, value in enumerate(values)}
    bridges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for row in edge_rows:
        if is_strong_edge(row):
            continue
        if row.pair_a not in position or row.pair_b not in position:
            raise ValueError("connectivity edge references unknown selected index")
        first_position = position[row.pair_a]
        second_position = position[row.pair_b]
        if second_position != first_position + 1:
            raise ValueError("connectivity edge is not adjacent in selected sequence")
        if first_position == 0:
            continue
        pair = (values[first_position - 1], values[second_position])
        if pair not in seen:
            seen.add(pair)
            bridges.append(pair)
    return tuple(bridges)


def choose_reconstruction_subset(
    records: Sequence[SelectedImageRecord],
    adjacent_rows: Sequence[PairGeometryMetrics],
    bridge_rows: Sequence[PairGeometryMetrics],
) -> tuple[SubsetDecision, ...]:
    if not records:
        raise ValueError("subset selection requires selected records")
    indices = tuple(record.index for record in records)
    adjacent_map = {(row.pair_a, row.pair_b): row for row in adjacent_rows}
    bridge_map = {(row.pair_a, row.pair_b): row for row in bridge_rows}
    expected_adjacent = adjacent_pairs(indices)
    missing = [pair for pair in expected_adjacent if pair not in adjacent_map]
    if missing:
        raise ValueError(f"missing adjacent connectivity edge: {missing[0]}")

    decisions: list[SubsetDecision] = []
    for position, record in enumerate(records):
        if position == 0:
            right = adjacent_map[expected_adjacent[0]] if expected_adjacent else None
            reason = "keep_endpoint_connected" if right and is_strong_edge(right) else "keep_endpoint_weak"
            decisions.append(
                SubsetDecision(record.index, record.filename, True, reason)
            )
            continue
        if position == len(records) - 1:
            left = adjacent_map[expected_adjacent[-1]] if expected_adjacent else None
            reason = "keep_endpoint_connected" if left and is_strong_edge(left) else "keep_endpoint_weak"
            decisions.append(
                SubsetDecision(record.index, record.filename, True, reason)
            )
            continue

        left_pair = (indices[position - 1], record.index)
        right_pair = (record.index, indices[position + 1])
        left = adjacent_map[left_pair]
        right = adjacent_map[right_pair]
        if is_strong_edge(left) or is_strong_edge(right):
            decisions.append(
                SubsetDecision(record.index, record.filename, True, "keep_connected")
            )
            continue

        bridge_pair = (indices[position - 1], indices[position + 1])
        bridge = bridge_map.get(bridge_pair)
        if bridge is not None and is_strong_edge(bridge):
            decisions.append(
                SubsetDecision(
                    record.index,
                    record.filename,
                    False,
                    "exclude_weak_bridged",
                )
            )
        else:
            decisions.append(
                SubsetDecision(
                    record.index,
                    record.filename,
                    True,
                    "keep_weak_bridge_needed",
                )
            )
    return tuple(decisions)
