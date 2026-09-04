import cv2
import numpy as np
import pytest

from geometry_detection import ImageScale, SiftFeatures
from reconstruction_matching import (
    BENCHMARK_PAIRS,
    PairGeometryMetrics,
    choose_feature_mode,
    filter_sift_features,
    grid_coverage,
    summarize_benchmark,
)


def _features() -> SiftFeatures:
    keypoints = (
        cv2.KeyPoint(2.0, 2.0, 1.0),
        cv2.KeyPoint(7.0, 7.0, 1.0),
    )
    descriptors = np.vstack(
        (np.ones((1, 128), np.float32), np.full((1, 128), 2.0, np.float32))
    )
    return SiftFeatures(
        analysis_image=np.zeros((10, 10, 3), np.uint8),
        keypoints=keypoints,
        descriptors=descriptors,
        scale=ImageScale(
            original_size=(20, 20),
            analysis_size=(10, 10),
            scale_x_to_original=2.0,
            scale_y_to_original=2.0,
        ),
        status="ok",
    )


def _metric(
    mode: str,
    pair: tuple[int, int],
    *,
    inliers: int,
    ratio: float,
    sampson: float,
) -> PairGeometryMetrics:
    return PairGeometryMetrics(
        pair_a=pair[0],
        pair_b=pair[1],
        mode=mode,
        keypoints_a=100,
        keypoints_b=100,
        candidate_matches=max(inliers, 1),
        inliers=inliers,
        inlier_ratio=ratio,
        median_sampson_error=sampson,
        p90_sampson_error=sampson,
        grid_coverage=0.5,
        status="ok",
    )


def test_filter_sift_features_keeps_only_keypoints_inside_mask():
    features = _features()
    mask = np.zeros((20, 20), np.uint8)
    mask[4, 4] = 255

    filtered = filter_sift_features(features, mask)

    assert len(filtered.keypoints) == 1
    assert filtered.keypoints[0].pt == features.keypoints[0].pt
    assert filtered.descriptors is not None
    assert filtered.descriptors.shape == (1, 128)
    assert np.array_equal(filtered.descriptors[0], features.descriptors[0])


def test_filter_sift_features_rejects_wrong_geometry_or_non_binary_mask():
    features = _features()
    with pytest.raises(ValueError):
        filter_sift_features(features, np.zeros((10, 10), np.uint8))
    bad = np.zeros((20, 20), np.uint8)
    bad[0, 0] = 127
    with pytest.raises(ValueError):
        filter_sift_features(features, bad)


def test_grid_coverage_counts_occupied_cells():
    points = np.array([[1.0, 1.0], [9.0, 9.0]], dtype=np.float32)
    assert grid_coverage(points, (16, 16), grid_size=4) == pytest.approx(2 / 16)
    assert grid_coverage(np.empty((0, 2), np.float32), (16, 16), grid_size=4) == 0.0


def test_benchmark_pairs_are_frozen_twenty_pairs():
    assert len(BENCHMARK_PAIRS) == 20
    assert BENCHMARK_PAIRS[:2] == ((15, 16), (45, 46))
    assert BENCHMARK_PAIRS[-1] == (280, 282)


def test_choose_feature_mode_requires_95_percent_inliers_and_nonworse_ratio():
    rows = []
    for pair in ((15, 16), (45, 46)):
        rows.append(_metric("unmasked", pair, inliers=50, ratio=0.30, sampson=0.50))
        rows.append(_metric("raw_cnn", pair, inliers=48, ratio=0.32, sampson=0.52))
        rows.append(
            _metric("reconstruction_mask", pair, inliers=47, ratio=0.40, sampson=0.40)
        )
    summary = summarize_benchmark(rows)

    assert summary["modes"]["raw_cnn"]["qualified"] is True
    assert summary["modes"]["reconstruction_mask"]["qualified"] is False
    assert choose_feature_mode(summary) == "raw_cnn"


def test_choose_feature_mode_falls_back_to_unmasked_when_masks_do_not_qualify():
    rows = []
    for pair in ((15, 16), (45, 46)):
        rows.append(_metric("unmasked", pair, inliers=50, ratio=0.30, sampson=0.50))
        rows.append(_metric("raw_cnn", pair, inliers=40, ratio=0.50, sampson=0.40))
        rows.append(
            _metric("reconstruction_mask", pair, inliers=49, ratio=0.29, sampson=0.40)
        )
    summary = summarize_benchmark(rows)

    assert choose_feature_mode(summary) == "unmasked"
