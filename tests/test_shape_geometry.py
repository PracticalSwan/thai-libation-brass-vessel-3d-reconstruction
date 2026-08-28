from __future__ import annotations

import cv2
import numpy as np
import pytest

from shape_geometry import (
    ShapeConfig,
    analyze_shape,
    detect_edges,
    measure_contour_geometry,
)


def _ellipse_image(
    *,
    width: int = 800,
    height: int = 600,
    axes: tuple[int, int] = (180, 90),
    angle: float = 25.0,
) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.ellipse(
        image,
        (width // 2, height // 2),
        axes,
        angle,
        0,
        360,
        (255, 255, 255),
        -1,
    )
    return image


def _axis_distance(first: float, second: float) -> float:
    return abs((first - second + 90.0) % 180.0 - 90.0)


def test_detect_edges_is_binary_and_preserves_source_pixels() -> None:
    """Catches non-binary Canny output or in-place source modification."""
    image = _ellipse_image(width=1600, height=1200)
    original = image.copy()

    edge_result = detect_edges(image, ShapeConfig(maximum_width=1200))

    assert edge_result.analysis_image.shape[:2] == (900, 1200)
    assert edge_result.edges.shape == (900, 1200)
    assert edge_result.edges.dtype == np.uint8
    assert set(np.unique(edge_result.edges)) <= {0, 255}
    np.testing.assert_array_equal(image, original)


def test_known_ellipse_returns_centroid_box_axis_and_ellipse() -> None:
    """Catches contour selection or measurements that lose known ellipse geometry."""
    image = _ellipse_image()

    result = analyze_shape(image)

    assert result.status == "ok"
    assert result.selected_candidate is not None
    assert result.geometry is not None
    assert result.geometry.ellipse is not None
    assert result.geometry.centroid[0] == pytest.approx(400.0, abs=3.0)
    assert result.geometry.centroid[1] == pytest.approx(300.0, abs=3.0)
    assert min(result.geometry.ellipse.axes) == pytest.approx(180.0, rel=0.12)
    assert max(result.geometry.ellipse.axes) == pytest.approx(360.0, rel=0.12)
    assert result.geometry.ellipse_fit_median_residual < 0.03
    assert result.geometry.ellipse_fit_p90_residual < 0.08
    x, y, width, height = result.geometry.bounding_box
    assert x < 400 < x + width
    assert y < 300 < y + height


def test_principal_axis_follows_rotated_silhouette() -> None:
    """Catches swapped coordinates or an axis unrelated to the contour moments."""
    image = _ellipse_image(axes=(170, 45), angle=32.0)

    result = analyze_shape(image)

    assert result.geometry is not None
    assert _axis_distance(result.geometry.principal_axis_angle_deg, 32.0) < 5.0
    first, second = result.geometry.principal_axis_endpoints
    assert first != second


def test_blank_image_reports_no_reliable_contour() -> None:
    """Catches forced shape success on an image with no usable geometry."""
    result = analyze_shape(np.zeros((300, 400, 3), dtype=np.uint8))

    assert result.status == "no_reliable_contour"
    assert result.selected_candidate is None
    assert result.geometry is None
    assert result.candidates == ()


def test_fewer_than_five_contour_points_disables_ellipse() -> None:
    """Catches calling fitEllipse outside its documented point-count boundary."""
    contour = np.array([[[10, 10]], [[100, 15]], [[45, 120]], [[10, 10]]], dtype=np.int32)

    geometry = measure_contour_geometry(contour)

    assert geometry.ellipse is None
    assert geometry.status == "ellipse_unavailable"
    assert geometry.contour_area > 0


def test_non_elliptical_vessel_contour_does_not_force_ellipse() -> None:
    """Catches a mathematically fitted but visually misleading global ellipse."""
    image = np.full((640, 600, 3), 125, dtype=np.uint8)
    polygon = np.array(
        [
            [275, 45],
            [325, 45],
            [340, 180],
            [390, 240],
            [500, 320],
            [470, 520],
            [360, 590],
            [300, 470],
            [240, 590],
            [130, 520],
            [100, 320],
            [210, 240],
            [260, 180],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [polygon], (0, 180, 255))

    result = analyze_shape(image)

    assert result.geometry is not None
    assert result.geometry.ellipse is None
    assert result.geometry.status == "ellipse_unavailable"
    assert result.geometry.ellipse_fit_median_residual > 0.08


def test_tiny_shape_is_retained_as_candidate_but_not_forced_as_vessel() -> None:
    """Catches an arbitrary speck being promoted to the vessel contour."""
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.circle(image, (400, 300), 4, (255, 255, 255), -1)

    result = analyze_shape(image)

    assert result.candidates
    assert result.status == "no_reliable_contour"
    assert result.selected_candidate is None
    assert result.geometry is None


def test_gold_supported_contour_ignores_gray_background_edges() -> None:
    """Catches classroom/background edges merging into the brass vessel contour."""
    image = np.full((600, 800, 3), 145, dtype=np.uint8)
    cv2.line(image, (0, 80), (799, 80), (20, 20, 20), 8)
    cv2.line(image, (690, 0), (690, 599), (20, 20, 20), 8)
    cv2.ellipse(
        image,
        (400, 330),
        (150, 230),
        0,
        0,
        360,
        (0, 180, 255),
        -1,
    )

    result = analyze_shape(image)

    assert result.status == "ok"
    assert result.edge_result.contour_source == "hsv_saturation_mask"
    assert result.geometry is not None
    x, y, width, height = result.geometry.bounding_box
    assert 240 <= x <= 260
    assert 90 <= y <= 110
    assert 295 <= width <= 305
    assert 455 <= height <= 465


def test_shape_configuration_rejects_invalid_canny_thresholds() -> None:
    """Catches a configuration whose edge-stage meaning is undefined."""
    with pytest.raises(ValueError, match="Canny"):
        detect_edges(
            _ellipse_image(),
            ShapeConfig(canny_low=180, canny_high=100),
        )
