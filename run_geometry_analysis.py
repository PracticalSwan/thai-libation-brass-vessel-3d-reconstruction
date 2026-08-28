"""Run Step 6 geometry analysis and generate measured presentation evidence."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from analysis_common import (
    SelectedImageRecord,
    path_for_index,
    verify_selected_images,
)
from geometry_detection import (
    GeometryMatchResult,
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
from shape_geometry import ShapeAnalysisResult, ShapeConfig, analyze_shape


@dataclass(frozen=True)
class GeometryRunConfig:
    images_dir: Path
    selection_manifest: Path
    output_root: Path
    expected_selected_count: int = 288
    pairs: tuple[tuple[int, int], ...] = ((165, 166), (255, 256))
    epipolar_pair: tuple[int, int] = (165, 166)
    shape_indices: tuple[int, ...] = (165, 255)
    sift: SiftConfig = SiftConfig()
    shape: ShapeConfig = ShapeConfig()
    raw_dir: Path | None = None


@dataclass(frozen=True)
class PairAnalysis:
    index_a: int
    index_b: int
    filename_a: str
    filename_b: str
    features_a: SiftFeatures
    features_b: SiftFeatures
    geometry: GeometryMatchResult


@dataclass(frozen=True)
class ShapeAnalysis:
    index: int
    filename: str
    result: ShapeAnalysisResult


@dataclass(frozen=True)
class GeometryAnalysisSummary:
    complete: bool
    artifacts: tuple[str, ...]
    pair_metrics: tuple[dict[str, Any], ...]
    shape_metrics: tuple[dict[str, Any], ...]


def _paths_overlap(first: Path, second: Path) -> bool:
    first_resolved = first.resolve()
    second_resolved = second.resolve()
    return (
        first_resolved == second_resolved
        or first_resolved in second_resolved.parents
        or second_resolved in first_resolved.parents
    )


def _validate_output_boundary(config: GeometryRunConfig) -> None:
    protected = [config.images_dir, config.selection_manifest]
    if config.raw_dir is not None:
        protected.append(config.raw_dir)
    for path in protected:
        if _paths_overlap(config.output_root, path):
            raise ValueError(
                f"analysis output must not overlap protected source path: {path}"
            )


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unreadable analysis image: {path}")
    return image


def analyze_pair(
    images_dir: Path,
    records: Sequence[SelectedImageRecord],
    pair: tuple[int, int],
    sift_config: SiftConfig = SiftConfig(),
) -> PairAnalysis:
    """Measure one selected pair with the reusable Step 6 SIFT pipeline."""
    index_a, index_b = pair
    path_a = path_for_index(records, images_dir, index_a)
    path_b = path_for_index(records, images_dir, index_b)
    features_a = extract_sift(_read_image(path_a), sift_config)
    features_b = extract_sift(_read_image(path_b), sift_config)
    candidates = match_sift(
        features_a, features_b, ratio_threshold=sift_config.ratio_threshold
    )
    geometry = estimate_fundamental_geometry(
        features_a,
        features_b,
        candidates,
        ransac_threshold=sift_config.ransac_threshold,
        confidence=sift_config.confidence,
        rng_seed=sift_config.rng_seed,
        minimum_correspondences=sift_config.minimum_correspondences,
    )
    return PairAnalysis(
        index_a=index_a,
        index_b=index_b,
        filename_a=records[index_a - 1].filename,
        filename_b=records[index_b - 1].filename,
        features_a=features_a,
        features_b=features_b,
        geometry=geometry,
    )


def analyze_shape_record(
    images_dir: Path,
    records: Sequence[SelectedImageRecord],
    index: int,
    shape_config: ShapeConfig = ShapeConfig(),
) -> ShapeAnalysis:
    path = path_for_index(records, images_dir, index)
    return ShapeAnalysis(
        index=index,
        filename=records[index - 1].filename,
        result=analyze_shape(_read_image(path), shape_config),
    )


def _resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] == width:
        return image.copy()
    scale = width / image.shape[1]
    return cv2.resize(
        image,
        (width, max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )


def _header(image: np.ndarray, lines: Sequence[str], height: int = 110) -> np.ndarray:
    bar = np.full((height, image.shape[1], 3), 247, dtype=np.uint8)
    font_scale = max(0.58, min(1.0, image.shape[1] / 1900.0))
    line_height = max(28, height // max(len(lines), 1))
    for line_number, line in enumerate(lines):
        cv2.putText(
            bar,
            line,
            (22, min(height - 12, 28 + line_number * line_height)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )
    return np.vstack((bar, image))


def _labeled_panel(image: np.ndarray, label: str, width: int = 520) -> np.ndarray:
    resized = _resize_to_width(image, width)
    bar = np.full((58, width, 3), 35, dtype=np.uint8)
    cv2.putText(
        bar,
        label,
        (14, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.vstack((bar, resized))


def _stack_rows(rows: Sequence[np.ndarray], width: int = 1800) -> np.ndarray:
    resized = [_resize_to_width(row, width) for row in rows]
    return np.vstack(resized)


def render_match_figure(pair: PairAnalysis) -> np.ndarray:
    """Render candidate SIFT matches above exact RANSAC inlier matches."""
    candidates = select_display_matches(
        pair.features_a,
        pair.features_b,
        pair.geometry.candidate_matches,
        max_matches=60,
    )
    inliers = select_display_matches(
        pair.features_a,
        pair.features_b,
        pair.geometry.inlier_matches,
        max_matches=60,
    )
    candidate_row = cv2.drawMatches(
        pair.features_a.analysis_image,
        list(pair.features_a.keypoints),
        pair.features_b.analysis_image,
        list(pair.features_b.keypoints),
        list(candidates),
        None,
        matchColor=(0, 150, 255),
        singlePointColor=(180, 180, 180),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    inlier_row = cv2.drawMatches(
        pair.features_a.analysis_image,
        list(pair.features_a.keypoints),
        pair.features_b.analysis_image,
        list(pair.features_b.keypoints),
        list(inliers),
        None,
        matchColor=(40, 205, 40),
        singlePointColor=(180, 180, 180),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    candidate_row = _header(
        candidate_row,
        (
            "Candidate SIFT matches after Lowe ratio test",
            f"Displayed {len(candidates)} of {pair.geometry.candidate_count} measured candidates",
        ),
        height=100,
    )
    inlier_row = _header(
        inlier_row,
        (
            "Fundamental-Matrix RANSAC inliers",
            f"Displayed {len(inliers)} of {pair.geometry.inlier_count} verified inliers",
        ),
        height=100,
    )
    figure = _stack_rows((candidate_row, inlier_row))
    return _header(
        figure,
        (
            f"Selected pair {pair.index_a}-{pair.index_b}: {pair.filename_a} | {pair.filename_b}",
            f"Keypoints {pair.features_a.keypoint_count} / {pair.features_b.keypoint_count}; "
            f"candidates {pair.geometry.candidate_count}; inliers {pair.geometry.inlier_count}; "
            f"ratio {pair.geometry.inlier_ratio:.3f}; status {pair.geometry.status}",
        ),
        height=120,
    )


def _inlier_residuals(pair: PairAnalysis) -> np.ndarray:
    if pair.geometry.status != "ok" or pair.geometry.fundamental_matrix is None:
        raise ValueError(
            f"pair {pair.index_a}-{pair.index_b} has no valid epipolar geometry: "
            f"{pair.geometry.status}"
        )
    return sampson_errors(
        pair.geometry.fundamental_matrix,
        pair.geometry.points_a[pair.geometry.inlier_mask],
        pair.geometry.points_b[pair.geometry.inlier_mask],
    )


def render_epipolar_figure(pair: PairAnalysis) -> np.ndarray:
    """Render corresponding points and lines from the pair's exact `F`/inliers."""
    residuals = _inlier_residuals(pair)
    selected = select_spatial_inliers(
        pair.geometry.points_a,
        pair.geometry.points_b,
        pair.geometry.inlier_mask,
        max_points=10,
    )
    points_a = pair.geometry.points_a[selected]
    points_b = pair.geometry.points_b[selected]
    lines_a = compute_epilines(
        pair.geometry.fundamental_matrix, points_b, which_image=2
    )
    lines_b = compute_epilines(
        pair.geometry.fundamental_matrix, points_a, which_image=1
    )
    image_a = pair.features_a.analysis_image.copy()
    image_b = pair.features_b.analysis_image.copy()
    palette = (
        (54, 67, 244),
        (65, 176, 65),
        (255, 165, 0),
        (180, 105, 255),
        (0, 215, 255),
        (255, 90, 90),
        (203, 192, 255),
        (128, 0, 255),
        (0, 128, 255),
        (255, 128, 0),
    )
    for item_index, (point_a, point_b, line_a, line_b) in enumerate(
        zip(points_a, points_b, lines_a, lines_b, strict=True)
    ):
        color = palette[item_index % len(palette)]
        clipped_a = clip_epiline_to_image(
            line_a, pair.features_a.scale.analysis_size
        )
        clipped_b = clip_epiline_to_image(
            line_b, pair.features_b.scale.analysis_size
        )
        if clipped_a is not None:
            cv2.line(image_a, clipped_a[0], clipped_a[1], color, 3, cv2.LINE_AA)
        if clipped_b is not None:
            cv2.line(image_b, clipped_b[0], clipped_b[1], color, 3, cv2.LINE_AA)
        cv2.circle(
            image_a,
            tuple(int(round(value)) for value in point_a),
            8,
            color,
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(
            image_b,
            tuple(int(round(value)) for value in point_b),
            8,
            color,
            -1,
            cv2.LINE_AA,
        )
    panel_a = _labeled_panel(image_a, f"Image {pair.index_a}: points + epilines", 900)
    panel_b = _labeled_panel(image_b, f"Image {pair.index_b}: points + epilines", 900)
    figure = np.hstack((panel_a, panel_b))
    return _header(
        figure,
        (
            f"Epipolar geometry: same F and {pair.geometry.inlier_count} RANSAC inliers",
            f"{len(selected)} spatial correspondences | median Sampson "
            f"{float(np.median(residuals)):.4f} analysis-px^2 | projective constraint only",
            "No camera pose, triangulation, or 3D reconstruction was computed",
        ),
        height=150,
    )


def render_shape_figure(shape: ShapeAnalysis) -> np.ndarray:
    """Render original, Canny, contour evidence, and measured shape overlay."""
    result = shape.result
    original = result.edge_result.analysis_image.copy()
    edge_panel = cv2.cvtColor(result.edge_result.edges, cv2.COLOR_GRAY2BGR)
    candidates_panel = original.copy()
    for candidate in result.candidates[:12]:
        cv2.drawContours(
            candidates_panel, [candidate.contour], -1, (170, 170, 170), 2
        )
    if result.selected_candidate is not None:
        cv2.drawContours(
            candidates_panel,
            [result.selected_candidate.contour],
            -1,
            (40, 220, 40),
            5,
        )
    overlay = original.copy()
    if result.selected_candidate is not None:
        cv2.drawContours(
            overlay,
            [result.selected_candidate.contour],
            -1,
            (40, 220, 40),
            4,
        )
    if result.geometry is not None:
        geometry = result.geometry
        x, y, width, height = geometry.bounding_box
        cv2.rectangle(
            overlay,
            (x, y),
            (x + width - 1, y + height - 1),
            (255, 0, 255),
            4,
        )
        cv2.circle(
            overlay,
            tuple(int(round(value)) for value in geometry.centroid),
            9,
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )
        cv2.line(
            overlay,
            geometry.principal_axis_endpoints[0],
            geometry.principal_axis_endpoints[1],
            (255, 255, 0),
            4,
            cv2.LINE_AA,
        )
        if geometry.ellipse is not None:
            ellipse = geometry.ellipse
            cv2.ellipse(
                overlay,
                (ellipse.center, ellipse.axes, ellipse.angle_deg),
                (0, 165, 255),
                4,
                cv2.LINE_AA,
            )
    panels = (
        _labeled_panel(original, "1. Original selected image"),
        _labeled_panel(edge_panel, "2. Grayscale Canny edges"),
        _labeled_panel(
            candidates_panel,
            "3. Contours (HSV gold mask)"
            if result.edge_result.contour_source == "hsv_saturation_mask"
            else "3. Contours (Canny fallback)",
        ),
        _labeled_panel(overlay, "4. Box + center + PCA + optional ellipse"),
    )
    figure = np.hstack(panels)
    if result.geometry is None or result.selected_candidate is None:
        details = f"status {result.status}; no geometric overlay forced"
    else:
        geometry = result.geometry
        ellipse_text = (
            "omitted: weak global fit"
            if geometry.ellipse is None
            else f"{geometry.ellipse.axes[0]:.1f} x {geometry.ellipse.axes[1]:.1f} px"
        )
        details = (
            f"status {result.status}; contour area {geometry.contour_area:.1f} px^2; "
            f"axis {geometry.principal_axis_angle_deg:.1f} deg; ellipse {ellipse_text}"
        )
    return _header(
        figure,
        (
            f"Classical vessel geometry - selected index {shape.index}: {shape.filename}",
            details,
        ),
        height=120,
    )


def _fit_to_box(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def render_geometry_summary(
    match_figure: np.ndarray,
    epipolar_figure: np.ndarray,
    shape_figure: np.ndarray,
) -> np.ndarray:
    """Combine the three real Step 6 demonstrations into one compact page."""
    cells = (
        _fit_to_box(match_figure, 600, 680),
        _fit_to_box(epipolar_figure, 600, 680),
        _fit_to_box(shape_figure, 600, 680),
    )
    body = np.hstack(cells)
    body = _header(
        body,
        (
            "1. SIFT candidates + RANSAC     ->     2. Epipolar constraints     ->     3. Classical vessel shape",
        ),
        height=80,
    )
    return _header(
        body,
        (
            "STEP 6 - GEOMETRY DETECTION AND ANALYSIS",
            "Real selected photographs; measured 2D/two-view geometry only; no camera pose or 3D reconstruction",
        ),
        height=120,
    )


def pair_metrics(pair: PairAnalysis) -> dict[str, Any]:
    residuals = (
        _inlier_residuals(pair)
        if pair.geometry.status == "ok"
        else np.asarray([], dtype=np.float64)
    )
    return {
        "index_1": pair.index_a,
        "index_2": pair.index_b,
        "filename_1": pair.filename_a,
        "filename_2": pair.filename_b,
        "keypoints_1": pair.features_a.keypoint_count,
        "keypoints_2": pair.features_b.keypoint_count,
        "candidate_matches": pair.geometry.candidate_count,
        "ransac_inliers": pair.geometry.inlier_count,
        "inlier_ratio": pair.geometry.inlier_ratio,
        "status": pair.geometry.status,
        "median_sampson_error": (
            float(np.median(residuals)) if len(residuals) else None
        ),
        "p90_sampson_error": (
            float(np.percentile(residuals, 90)) if len(residuals) else None
        ),
        "analysis_width_1": pair.features_a.scale.analysis_size[0],
        "analysis_height_1": pair.features_a.scale.analysis_size[1],
        "analysis_width_2": pair.features_b.scale.analysis_size[0],
        "analysis_height_2": pair.features_b.scale.analysis_size[1],
    }


def shape_metrics(shape: ShapeAnalysis) -> dict[str, Any]:
    result = shape.result
    row: dict[str, Any] = {
        "index": shape.index,
        "filename": shape.filename,
        "status": result.status,
        "contour_source": result.edge_result.contour_source,
        "candidate_count": len(result.candidates),
        "original_width": result.edge_result.scale.original_size[0],
        "original_height": result.edge_result.scale.original_size[1],
        "analysis_width": result.edge_result.scale.analysis_size[0],
        "analysis_height": result.edge_result.scale.analysis_size[1],
        "scale_x_to_original": result.edge_result.scale.scale_x_to_original,
        "scale_y_to_original": result.edge_result.scale.scale_y_to_original,
        "selected_score": None,
        "contour_area": None,
        "contour_area_fraction": None,
        "bounding_box_x": None,
        "bounding_box_y": None,
        "bounding_box_width": None,
        "bounding_box_height": None,
        "centroid_x": None,
        "centroid_y": None,
        "principal_axis_angle_deg": None,
        "ellipse_center_x": None,
        "ellipse_center_y": None,
        "ellipse_axis_1": None,
        "ellipse_axis_2": None,
        "ellipse_angle_deg": None,
        "ellipse_fit_median_residual": None,
        "ellipse_fit_p90_residual": None,
    }
    if result.selected_candidate is not None:
        row["selected_score"] = result.selected_candidate.score
        row["contour_area_fraction"] = result.selected_candidate.area_fraction
    if result.geometry is not None:
        geometry = result.geometry
        row.update(
            {
                "contour_area": geometry.contour_area,
                "bounding_box_x": geometry.bounding_box[0],
                "bounding_box_y": geometry.bounding_box[1],
                "bounding_box_width": geometry.bounding_box[2],
                "bounding_box_height": geometry.bounding_box[3],
                "centroid_x": geometry.centroid[0],
                "centroid_y": geometry.centroid[1],
                "principal_axis_angle_deg": geometry.principal_axis_angle_deg,
                "ellipse_fit_median_residual": geometry.ellipse_fit_median_residual,
                "ellipse_fit_p90_residual": geometry.ellipse_fit_p90_residual,
            }
        )
        if geometry.ellipse is not None:
            row.update(
                {
                    "ellipse_center_x": geometry.ellipse.center[0],
                    "ellipse_center_y": geometry.ellipse.center[1],
                    "ellipse_axis_1": geometry.ellipse.axes[0],
                    "ellipse_axis_2": geometry.ellipse.axes[1],
                    "ellipse_angle_deg": geometry.ellipse.angle_deg,
                }
            )
    return row


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty report: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_image(path: Path, image: np.ndarray) -> None:
    if image.ndim != 3 or image.size == 0 or not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write presentation image: {path}")


def run_geometry_analysis(config: GeometryRunConfig) -> GeometryAnalysisSummary:
    """Verify all selected inputs, measure Step 6, and write final evidence."""
    _validate_output_boundary(config)
    verified = verify_selected_images(
        config.images_dir,
        config.selection_manifest,
        expected_count=config.expected_selected_count,
    )

    geometry_dir = config.output_root / "geometry"
    reports_dir = config.output_root / "reports"
    presentation_dir = config.output_root / "previews" / "presentation"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    presentation_dir.mkdir(parents=True, exist_ok=True)
    final_summary_path = reports_dir / "geometry_summary.json"
    if final_summary_path.exists():
        final_summary_path.unlink()

    all_pairs = list(dict.fromkeys((*config.pairs, config.epipolar_pair)))
    pair_results: dict[tuple[int, int], PairAnalysis] = {}
    match_figures: dict[tuple[int, int], np.ndarray] = {}
    artifacts: list[str] = []
    for pair in all_pairs:
        analysis = analyze_pair(config.images_dir, verified.records, pair, config.sift)
        pair_results[pair] = analysis
        if pair in config.pairs:
            figure = render_match_figure(analysis)
            match_figures[pair] = figure
            path = presentation_dir / f"geometry_01_matches_{pair[0]}_{pair[1]}.png"
            _write_image(path, figure)
            artifacts.append(path.relative_to(config.output_root).as_posix())

    primary = pair_results[config.epipolar_pair]
    epipolar_figure = render_epipolar_figure(primary)
    epipolar_path = presentation_dir / (
        f"geometry_02_epipolar_{primary.index_a}_{primary.index_b}.png"
    )
    _write_image(epipolar_path, epipolar_figure)
    artifacts.append(epipolar_path.relative_to(config.output_root).as_posix())

    shape_results: list[ShapeAnalysis] = []
    shape_figures: dict[int, np.ndarray] = {}
    for index in config.shape_indices:
        analysis = analyze_shape_record(
            config.images_dir, verified.records, index, config.shape
        )
        shape_results.append(analysis)
        figure = render_shape_figure(analysis)
        shape_figures[index] = figure
        path = presentation_dir / f"geometry_03_shape_{index}.png"
        _write_image(path, figure)
        artifacts.append(path.relative_to(config.output_root).as_posix())

    if not config.pairs or not config.shape_indices:
        raise ValueError("at least one pair and one shape index are required")
    summary_figure = render_geometry_summary(
        match_figures[config.pairs[0]],
        epipolar_figure,
        shape_figures[config.shape_indices[0]],
    )
    summary_figure_path = presentation_dir / "geometry_04_summary.png"
    _write_image(summary_figure_path, summary_figure)
    artifacts.append(summary_figure_path.relative_to(config.output_root).as_posix())

    pair_rows = tuple(pair_metrics(pair_results[pair]) for pair in all_pairs)
    shape_rows = tuple(shape_metrics(result) for result in shape_results)
    pair_report = geometry_dir / "pair_metrics.csv"
    shape_report = geometry_dir / "shape_metrics.csv"
    epipolar_report = geometry_dir / "epipolar_metrics.json"
    input_report = reports_dir / "input_verification.json"
    _write_csv(pair_report, pair_rows)
    _write_csv(shape_report, shape_rows)
    primary_residuals = _inlier_residuals(primary)
    _write_json(
        epipolar_report,
        {
            "pair": [primary.index_a, primary.index_b],
            "filenames": [primary.filename_a, primary.filename_b],
            "fundamental_matrix": primary.geometry.fundamental_matrix.tolist(),
            "ransac_inlier_count": primary.geometry.inlier_count,
            "sampson_error_units": "analysis pixels squared",
            "median_sampson_error": float(np.median(primary_residuals)),
            "p90_sampson_error": float(np.percentile(primary_residuals, 90)),
            "maximum_sampson_error": float(np.max(primary_residuals)),
            "interpretation": (
                "Two-view projective constraint only; no camera pose, triangulation, "
                "or reconstruction was computed."
            ),
        },
    )
    _write_json(
        input_report,
        {
            "verified": True,
            "verified_selected_count": len(verified.records),
            "selection_manifest": str(config.selection_manifest.resolve()),
            "selection_manifest_sha256": verified.manifest_sha256,
            "selected_images_directory": str(verified.images_dir),
            "checks": [
                "manifest schema and deterministic row order",
                "missing and extra entries",
                "readability, dimensions, size, and SHA-256 for every selected image",
            ],
        },
    )
    for report in (pair_report, epipolar_report, shape_report, input_report):
        artifacts.append(report.relative_to(config.output_root).as_posix())
    artifacts.append(final_summary_path.relative_to(config.output_root).as_posix())
    artifacts = sorted(artifacts)

    summary_payload = {
        "step": 6,
        "complete": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "verified_selected_count": len(verified.records),
            "selected_images_directory": str(verified.images_dir),
            "selection_manifest": str(config.selection_manifest.resolve()),
            "selection_manifest_sha256": verified.manifest_sha256,
        },
        "configuration": {
            "sift": asdict(config.sift),
            "shape": asdict(config.shape),
            "pairs": [list(pair) for pair in config.pairs],
            "epipolar_pair": list(config.epipolar_pair),
            "shape_indices": list(config.shape_indices),
        },
        "runtime": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "pair_metrics": list(pair_rows),
        "shape_metrics": list(shape_rows),
        "artifacts": artifacts,
        "scope_exclusions": [
            "Step 7/8 ML and SAM",
            "pyCOLMAP and reconstruction",
            "meshing, texturing, and Blender",
        ],
    }
    _write_json(final_summary_path, summary_payload)
    return GeometryAnalysisSummary(
        complete=True,
        artifacts=tuple(artifacts),
        pair_metrics=pair_rows,
        shape_metrics=shape_rows,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("preprocessing/pycolmap_input/images"),
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=Path("preprocessing/reports/selection_manifest.csv"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("analysis"))
    parser.add_argument("--expected-count", type=int, default=288)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = GeometryRunConfig(
        images_dir=args.images_dir,
        selection_manifest=args.selection_manifest,
        output_root=args.output_root,
        expected_selected_count=args.expected_count,
        raw_dir=Path("IMG20260826122949"),
    )
    print("Verifying selected manifest and images...", flush=True)
    summary = run_geometry_analysis(config)
    print(f"Step 6 complete: {len(summary.artifacts)} artifacts", flush=True)
    for row in summary.pair_metrics:
        print(
            f"Pair {row['index_1']}-{row['index_2']}: "
            f"{row['candidate_matches']} candidates, {row['ransac_inliers']} inliers, "
            f"ratio {row['inlier_ratio']:.3f}",
            flush=True,
        )
    for row in summary.shape_metrics:
        # User-approved reviewer hardening: honest failure states may have no PCA axis.
        axis = row["principal_axis_angle_deg"]
        axis_text = "unavailable" if axis is None else f"{axis:.2f} deg"
        print(
            f"Shape {row['index']}: {row['status']}, source {row['contour_source']}, "
            f"axis {axis_text}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
