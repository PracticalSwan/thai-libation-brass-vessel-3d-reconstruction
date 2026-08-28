from __future__ import annotations

import cv2
import numpy as np
import pytest

from geometry_detection import (
    ImageScale,
    SiftConfig,
    SiftFeatures,
    clip_epiline_to_image,
    compute_epilines,
    estimate_fundamental_geometry,
    extract_sift,
    match_sift,
    sampson_errors,
    select_display_matches,
    select_spatial_inliers,
)


def _features_from_points(points: np.ndarray) -> SiftFeatures:
    keypoints = tuple(
        cv2.KeyPoint(float(point[0]), float(point[1]), 1.0) for point in points
    )
    descriptors = np.zeros((len(points), 128), dtype=np.float32)
    for index in range(len(points)):
        descriptors[index, index % 128] = 1.0
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    return SiftFeatures(
        analysis_image=image,
        keypoints=keypoints,
        descriptors=descriptors,
        scale=ImageScale(
            original_size=(640, 480),
            analysis_size=(640, 480),
            scale_x_to_original=1.0,
            scale_y_to_original=1.0,
        ),
        status="ok",
    )


def _matches(count: int) -> tuple[cv2.DMatch, ...]:
    return tuple(
        cv2.DMatch(_queryIdx=index, _trainIdx=index, _distance=0.1)
        for index in range(count)
    )


def _projected_correspondences(count: int = 40) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4213)
    points_3d = np.column_stack(
        (
            rng.uniform(-1.0, 1.0, count),
            rng.uniform(-0.8, 0.8, count),
            rng.uniform(4.0, 8.0, count),
        )
    )
    focal = 600.0
    center = np.array([320.0, 240.0])
    first = focal * points_3d[:, :2] / points_3d[:, 2, None] + center
    translated = points_3d.copy()
    translated[:, 0] -= 0.45
    second = focal * translated[:, :2] / translated[:, 2, None] + center
    return first.astype(np.float32), second.astype(np.float32)


def test_extract_sift_preserves_explicit_coordinate_mapping() -> None:
    """Catches ambiguous or inverted scale metadata after analysis downscaling."""
    rng = np.random.default_rng(4213)
    image = rng.integers(0, 256, size=(1200, 1600, 3), dtype=np.uint8)
    original = image.copy()

    features = extract_sift(image, SiftConfig(maximum_width=1200, nfeatures=300))

    assert features.analysis_image.shape[:2] == (900, 1200)
    assert features.scale.original_size == (1600, 1200)
    assert features.scale.analysis_size == (1200, 900)
    assert features.scale.scale_x_to_original == pytest.approx(4 / 3)
    assert features.scale.scale_y_to_original == pytest.approx(4 / 3)
    converted = features.scale.points_to_original(
        np.array([[0.0, 0.0], [1200.0, 900.0]], dtype=np.float64)
    )
    np.testing.assert_allclose(converted, [[0.0, 0.0], [1600.0, 1200.0]])
    np.testing.assert_allclose(
        features.scale.points_to_analysis(converted),
        [[0.0, 0.0], [1200.0, 900.0]],
    )
    np.testing.assert_array_equal(image, original)


def test_blank_image_returns_descriptor_unavailable_status() -> None:
    """Catches descriptor-less images being represented as successful SIFT runs."""
    features = extract_sift(np.zeros((120, 160, 3), dtype=np.uint8))

    assert features.status == "descriptors_unavailable"
    assert features.descriptors is None
    assert features.keypoints == ()


def test_match_sift_returns_only_valid_ratio_test_indices() -> None:
    """Catches malformed match indices or bypassing the Lowe ratio filter."""
    first_image = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(
        first_image,
        "CSX4213",
        (30, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    for x in range(20, 301, 35):
        cv2.circle(first_image, (x, 190), 8, (180, 180, 180), -1)
    second_image = np.roll(first_image, shift=4, axis=1)
    first = extract_sift(first_image)
    second = extract_sift(second_image)

    matches = match_sift(first, second)

    assert matches
    assert all(0 <= match.queryIdx < len(first.keypoints) for match in matches)
    assert all(0 <= match.trainIdx < len(second.keypoints) for match in matches)


def test_fundamental_geometry_reports_insufficient_correspondences() -> None:
    """Catches fewer than eight points reaching OpenCV as apparent success."""
    points = np.array([[20.0 + i * 10.0, 30.0 + i * 4.0] for i in range(7)])
    first = _features_from_points(points)
    second = _features_from_points(points + np.array([4.0, 0.0]))

    result = estimate_fundamental_geometry(first, second, _matches(7))

    assert result.status == "insufficient_geometry"
    assert result.fundamental_matrix is None
    assert result.candidate_count == 7
    assert result.inlier_count == 0
    assert result.inlier_mask.shape == (7,)


def test_malformed_fundamental_matrix_is_a_structured_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches malformed OpenCV output being reported as usable geometry."""
    first_points, second_points = _projected_correspondences(12)
    first = _features_from_points(first_points)
    second = _features_from_points(second_points)

    monkeypatch.setattr(
        cv2,
        "findFundamentalMat",
        lambda *args: (
            np.ones((2, 2), dtype=np.float64),
            np.ones((12, 1), dtype=np.uint8),
        ),
    )

    result = estimate_fundamental_geometry(first, second, _matches(12))

    assert result.status == "degenerate_fundamental_matrix"
    assert result.fundamental_matrix is None
    assert result.inlier_count == 0


def test_controlled_camera_correspondences_produce_verified_geometry() -> None:
    """Catches an estimator that cannot recover a valid real two-view relation."""
    first_points, second_points = _projected_correspondences()
    result = estimate_fundamental_geometry(
        _features_from_points(first_points),
        _features_from_points(second_points),
        _matches(len(first_points)),
    )

    assert result.status == "ok"
    assert result.fundamental_matrix is not None
    assert result.fundamental_matrix.shape == (3, 3)
    assert np.isfinite(result.fundamental_matrix).all()
    assert result.inlier_count >= 30
    assert result.inlier_mask.shape == (len(first_points),)
    assert len(result.inlier_matches) == result.inlier_count


def test_epipolar_lines_and_sampson_errors_use_matching_correspondences() -> None:
    """Catches line direction mistakes and residuals unrelated to x'Fx."""
    fundamental = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    first = np.array([[10.0, 20.0], [40.0, 60.0], [80.0, 100.0]])
    second = np.array([[15.0, 20.0], [45.0, 60.0], [85.0, 100.0]])

    lines = compute_epilines(fundamental, first, which_image=1)
    clean = sampson_errors(fundamental, first, second)
    corrupted = sampson_errors(
        fundamental, first, second + np.array([0.0, 25.0])
    )

    assert lines.shape == (3, 3)
    assert np.isfinite(lines).all()
    assert np.median(clean) == pytest.approx(0.0)
    assert np.median(corrupted) > np.median(clean)
    with pytest.raises(ValueError, match="finite 3x3 fundamental matrix"):
        compute_epilines(None, first, which_image=1)  # type: ignore[arg-type]


def test_epiline_clipping_returns_only_image_border_points() -> None:
    """Catches drawing epipolar lines outside or across the wrong image bounds."""
    clipped = clip_epiline_to_image(
        np.array([1.0, -1.0, 0.0]), image_size=(100, 80)
    )
    invalid = clip_epiline_to_image(
        np.array([0.0, 0.0, 1.0]), image_size=(100, 80)
    )

    assert clipped == ((0, 0), (79, 79))
    assert invalid is None
    assert all(0 <= x < 100 and 0 <= y < 80 for x, y in clipped)


def test_spatial_sampling_is_deterministic_and_keeps_only_inliers() -> None:
    """Catches random or non-inlier presentation samples masquerading as evidence."""
    grid = np.array(
        [(float(x), float(y)) for y in range(0, 100, 10) for x in range(0, 100, 10)]
    )
    shifted = grid + np.array([4.0, 1.0])
    mask = np.array([(index % 3) != 0 for index in range(len(grid))])

    first = select_spatial_inliers(grid, shifted, mask, max_points=10)
    second = select_spatial_inliers(grid, shifted, mask, max_points=10)

    np.testing.assert_array_equal(first, second)
    assert len(first) == 10
    assert mask[first].all()


def test_display_match_sampling_changes_only_the_drawn_subset() -> None:
    """Catches dense figures or presentation sampling that alters measured matches."""
    grid = np.array(
        [(float(x), float(y)) for y in range(0, 100, 10) for x in range(0, 100, 10)]
    )
    first = _features_from_points(grid)
    second = _features_from_points(grid + np.array([3.0, 2.0]))
    measured = _matches(len(grid))

    displayed_once = select_display_matches(first, second, measured, max_matches=12)
    displayed_twice = select_display_matches(first, second, measured, max_matches=12)

    assert len(measured) == 100
    assert len(displayed_once) == 12
    assert [match.queryIdx for match in displayed_once] == [
        match.queryIdx for match in displayed_twice
    ]
    assert set(displayed_once).issubset(set(measured))
