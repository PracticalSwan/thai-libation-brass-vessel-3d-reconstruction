"""Explainable classical 2D vessel-shape measurements for Step 6."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees

import cv2
import numpy as np

from geometry_detection import ImageScale, prepare_analysis_image


@dataclass(frozen=True)
class ShapeConfig:
    maximum_width: int = 1200
    gaussian_kernel: int = 5
    canny_low: int = 45
    canny_high: int = 135
    morphology_kernel: int = 11
    morphology_iterations: int = 2
    gold_hue_low: int = 5
    gold_hue_high: int = 40
    gold_saturation_minimum: int = 55
    gold_value_minimum: int = 35
    minimum_gold_fraction: float = 0.002
    minimum_area_fraction: float = 0.002
    minimum_score: float = 0.30
    weak_score: float = 0.45
    border_margin: int = 2
    ellipse_median_residual_maximum: float = 0.08
    ellipse_p90_residual_maximum: float = 0.25


@dataclass(frozen=True)
class EdgeResult:
    analysis_image: np.ndarray
    grayscale: np.ndarray
    edges: np.ndarray
    contour_edges: np.ndarray
    contour_source: str
    scale: ImageScale


@dataclass(frozen=True)
class ContourCandidate:
    contour: np.ndarray
    area: float
    area_fraction: float
    bounding_box: tuple[int, int, int, int]
    centroid: tuple[float, float]
    solidity: float
    border_contacts: int
    score: float


@dataclass(frozen=True)
class EllipseGeometry:
    center: tuple[float, float]
    axes: tuple[float, float]
    angle_deg: float


@dataclass(frozen=True)
class ShapeGeometryResult:
    contour_area: float
    bounding_box: tuple[int, int, int, int]
    centroid: tuple[float, float]
    principal_axis_angle_deg: float
    principal_axis_endpoints: tuple[tuple[int, int], tuple[int, int]]
    ellipse: EllipseGeometry | None
    ellipse_fit_median_residual: float | None
    ellipse_fit_p90_residual: float | None
    status: str


@dataclass(frozen=True)
class ShapeAnalysisResult:
    edge_result: EdgeResult
    candidates: tuple[ContourCandidate, ...]
    selected_candidate: ContourCandidate | None
    geometry: ShapeGeometryResult | None
    status: str
    notes: tuple[str, ...]


def _validate_config(config: ShapeConfig) -> None:
    if config.maximum_width < 1:
        raise ValueError("maximum_width must be positive")
    if config.gaussian_kernel < 1 or config.gaussian_kernel % 2 == 0:
        raise ValueError("gaussian_kernel must be a positive odd number")
    if not 0 <= config.canny_low < config.canny_high <= 255:
        raise ValueError("Canny thresholds must satisfy 0 <= low < high <= 255")
    if config.morphology_kernel < 1 or config.morphology_kernel % 2 == 0:
        raise ValueError("morphology_kernel must be a positive odd number")
    if config.morphology_iterations < 0:
        raise ValueError("morphology_iterations must not be negative")
    if not 0 <= config.gold_hue_low < config.gold_hue_high <= 179:
        raise ValueError("gold hue thresholds must be ordered within 0..179")
    if not 0 <= config.gold_saturation_minimum <= 255:
        raise ValueError("gold_saturation_minimum must be within 0..255")
    if not 0 <= config.gold_value_minimum <= 255:
        raise ValueError("gold_value_minimum must be within 0..255")
    if not 0.0 <= config.minimum_gold_fraction < 1.0:
        raise ValueError("minimum_gold_fraction must be between zero and one")
    if not 0.0 <= config.minimum_area_fraction < 1.0:
        raise ValueError("minimum_area_fraction must be between zero and one")
    if not 0.0 <= config.minimum_score <= config.weak_score <= 1.0:
        raise ValueError("shape score thresholds must be ordered within zero and one")
    if not 0.0 <= config.ellipse_median_residual_maximum <= 1.0:
        raise ValueError("ellipse median residual threshold must be within zero and one")
    if not 0.0 <= config.ellipse_p90_residual_maximum <= 1.0:
        raise ValueError("ellipse p90 residual threshold must be within zero and one")


def detect_edges(
    image: np.ndarray, config: ShapeConfig = ShapeConfig()
) -> EdgeResult:
    """Create binary Canny edges and a lightly closed contour-working copy."""
    _validate_config(config)
    analysis, scale = prepare_analysis_image(image, config.maximum_width)
    if analysis.ndim == 2:
        grayscale = analysis.copy()
    else:
        grayscale = cv2.cvtColor(analysis, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(
        grayscale, (config.gaussian_kernel, config.gaussian_kernel), 0
    )
    edges = cv2.Canny(blurred, config.canny_low, config.canny_high)
    contour_source = "canny_edges"
    contour_input = edges
    if analysis.ndim == 3:
        hsv = cv2.cvtColor(analysis, cv2.COLOR_BGR2HSV)
        gold_mask = cv2.inRange(
            hsv,
            np.array(
                [
                    config.gold_hue_low,
                    config.gold_saturation_minimum,
                    config.gold_value_minimum,
                ],
                dtype=np.uint8,
            ),
            np.array([config.gold_hue_high, 255, 255], dtype=np.uint8),
        )
        gold_fraction = float(np.count_nonzero(gold_mask)) / gold_mask.size
        if gold_fraction >= config.minimum_gold_fraction:
            contour_input = gold_mask
            contour_source = "hsv_saturation_mask"
    if config.morphology_iterations:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.morphology_kernel, config.morphology_kernel),
        )
        contour_edges = cv2.morphologyEx(
            contour_input,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=config.morphology_iterations,
        )
    else:
        contour_edges = contour_input.copy()
    return EdgeResult(
        analysis_image=analysis,
        grayscale=grayscale,
        edges=edges,
        contour_edges=contour_edges,
        contour_source=contour_source,
        scale=scale,
    )


def find_contour_candidates(edges: np.ndarray) -> tuple[np.ndarray, ...]:
    """Extract all nonzero-area contours for explainable later scoring."""
    if edges.ndim != 2 or edges.dtype != np.uint8:
        raise ValueError("contour edge map must be a uint8 single-channel image")
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return tuple(
        contour
        for contour in contours
        if len(contour) >= 3 and abs(float(cv2.contourArea(contour))) > 0.0
    )


def _contour_centroid(contour: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(contour)
    if abs(moments["m00"]) > np.finfo(float).eps:
        return (
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        )
    points = contour.reshape(-1, 2).astype(np.float64)
    center = points.mean(axis=0)
    return float(center[0]), float(center[1])


def score_contour_candidates(
    contours: tuple[np.ndarray, ...] | list[np.ndarray],
    image_shape: tuple[int, int],
    config: ShapeConfig = ShapeConfig(),
) -> tuple[ContourCandidate, ...]:
    """Score candidates by area, centrality, extent, solidity, and borders."""
    _validate_config(config)
    height, width = image_shape
    if width < 1 or height < 1:
        raise ValueError("image_shape must be positive (height, width)")
    image_area = float(width * height)
    image_center = np.array([width / 2.0, height / 2.0])
    half_diagonal = max(float(np.linalg.norm(image_center)), 1.0)
    candidates: list[ContourCandidate] = []
    for contour in contours:
        area = abs(float(cv2.contourArea(contour)))
        x, y, box_width, box_height = cv2.boundingRect(contour)
        centroid = _contour_centroid(contour)
        center_distance = float(
            np.linalg.norm(np.asarray(centroid) - image_center) / half_diagonal
        )
        centrality = max(0.0, 1.0 - center_distance)
        area_fraction = area / image_area
        area_score = min(area_fraction / 0.20, 1.0)
        extent_fraction = (box_width * box_height) / image_area
        extent_score = min(extent_fraction / 0.25, 1.0)
        hull_area = abs(float(cv2.contourArea(cv2.convexHull(contour))))
        solidity = area / hull_area if hull_area > 0 else 0.0
        border_contacts = sum(
            (
                x <= config.border_margin,
                y <= config.border_margin,
                x + box_width >= width - config.border_margin,
                y + box_height >= height - config.border_margin,
            )
        )
        score = (
            0.45 * area_score
            + 0.25 * centrality
            + 0.15 * extent_score
            + 0.15 * min(solidity, 1.0)
            - 0.20 * border_contacts
        )
        candidates.append(
            ContourCandidate(
                contour=contour,
                area=area,
                area_fraction=area_fraction,
                bounding_box=(x, y, box_width, box_height),
                centroid=centroid,
                solidity=solidity,
                border_contacts=border_contacts,
                score=float(score),
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.score,
                candidate.area,
                -candidate.bounding_box[1],
                -candidate.bounding_box[0],
            ),
            reverse=True,
        )
    )


def select_vessel_contour(
    candidates: tuple[ContourCandidate, ...] | list[ContourCandidate],
    config: ShapeConfig = ShapeConfig(),
) -> ContourCandidate | None:
    """Choose the highest explainable candidate or report no reliable contour."""
    _validate_config(config)
    for candidate in candidates:
        if (
            candidate.area_fraction >= config.minimum_area_fraction
            and candidate.score >= config.minimum_score
            and candidate.border_contacts < 3
        ):
            return candidate
    return None


def measure_contour_geometry(
    contour: np.ndarray, config: ShapeConfig = ShapeConfig()
) -> ShapeGeometryResult:
    """Measure centroid, box, PCA axis, and an optional fitted ellipse."""
    _validate_config(config)
    points = np.asarray(contour).reshape(-1, 2)
    if len(points) < 3:
        raise ValueError("contour needs at least three points")
    area = abs(float(cv2.contourArea(points.reshape(-1, 1, 2))))
    bounding_box = tuple(int(value) for value in cv2.boundingRect(points))
    centroid = _contour_centroid(points.reshape(-1, 1, 2))

    mean, eigenvectors, _ = cv2.PCACompute2(
        points.astype(np.float64), mean=np.empty((0), dtype=np.float64)
    )
    direction = eigenvectors[0]
    angle = degrees(atan2(float(direction[1]), float(direction[0]))) % 180.0
    _, _, box_width, box_height = bounding_box
    half_length = 0.5 * max(box_width, box_height)
    center = np.asarray(centroid, dtype=np.float64)
    delta = direction * half_length
    first_endpoint = tuple(int(round(value)) for value in center - delta)
    second_endpoint = tuple(int(round(value)) for value in center + delta)

    ellipse: EllipseGeometry | None = None
    ellipse_fit_median_residual: float | None = None
    ellipse_fit_p90_residual: float | None = None
    if len(points) >= 5:
        try:
            ellipse_raw = cv2.fitEllipse(points.astype(np.float32))
        except cv2.error:
            ellipse_raw = None
        if ellipse_raw is not None:
            (center_x, center_y), (axis_a, axis_b), ellipse_angle = ellipse_raw
            ellipse_values = np.array(
                [center_x, center_y, axis_a, axis_b, ellipse_angle], dtype=np.float64
            )
            if np.isfinite(ellipse_values).all() and axis_a > 0 and axis_b > 0:
                fitted_ellipse = EllipseGeometry(
                    center=(float(center_x), float(center_y)),
                    axes=(float(axis_a), float(axis_b)),
                    angle_deg=float(ellipse_angle),
                )
                angle_radians = np.deg2rad(fitted_ellipse.angle_deg)
                cosine = np.cos(angle_radians)
                sine = np.sin(angle_radians)
                centered = points.astype(np.float64) - np.asarray(
                    fitted_ellipse.center, dtype=np.float64
                )
                local_x = cosine * centered[:, 0] + sine * centered[:, 1]
                local_y = -sine * centered[:, 0] + cosine * centered[:, 1]
                semi_axis_x = fitted_ellipse.axes[0] / 2.0
                semi_axis_y = fitted_ellipse.axes[1] / 2.0
                normalized_radius = np.sqrt(
                    (local_x / semi_axis_x) ** 2
                    + (local_y / semi_axis_y) ** 2
                )
                residuals = np.abs(normalized_radius - 1.0)
                ellipse_fit_median_residual = float(np.median(residuals))
                ellipse_fit_p90_residual = float(np.percentile(residuals, 90))
                if (
                    ellipse_fit_median_residual
                    <= config.ellipse_median_residual_maximum
                    and ellipse_fit_p90_residual
                    <= config.ellipse_p90_residual_maximum
                ):
                    ellipse = fitted_ellipse
    return ShapeGeometryResult(
        contour_area=area,
        bounding_box=bounding_box,
        centroid=centroid,
        principal_axis_angle_deg=float(angle),
        principal_axis_endpoints=(first_endpoint, second_endpoint),
        ellipse=ellipse,
        ellipse_fit_median_residual=ellipse_fit_median_residual,
        ellipse_fit_p90_residual=ellipse_fit_p90_residual,
        status="ok" if ellipse is not None else "ellipse_unavailable",
    )


def analyze_shape(
    image: np.ndarray, config: ShapeConfig = ShapeConfig()
) -> ShapeAnalysisResult:
    """Run deterministic classical shape analysis with honest weak states."""
    edge_result = detect_edges(image, config)
    contours = find_contour_candidates(edge_result.contour_edges)
    candidates = score_contour_candidates(
        contours, edge_result.edges.shape, config
    )
    selected = select_vessel_contour(candidates, config)
    if selected is None:
        return ShapeAnalysisResult(
            edge_result=edge_result,
            candidates=candidates,
            selected_candidate=None,
            geometry=None,
            status="no_reliable_contour",
            notes=("No candidate passed the documented area and score thresholds.",),
        )
    geometry = measure_contour_geometry(selected.contour, config)
    if selected.score < config.weak_score:
        status = "weak_contour"
        notes = (
            "The selected candidate passed the minimum threshold but has a weak score.",
        )
    else:
        status = geometry.status
        notes = ()
    return ShapeAnalysisResult(
        edge_result=edge_result,
        candidates=candidates,
        selected_candidate=selected,
        geometry=geometry,
        status=status,
        notes=notes,
    )
