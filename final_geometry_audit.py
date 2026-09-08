"""Exact-camera Plan-2 geometry continuity audit for the Final V2 vessel.

The Blender base is generated deterministically from ``final_profiles.json``.
This module binds that saved candidate to the accepted Plan-1 profile package,
then reprojects the same named surfaces through the frozen Step-13
``SIMPLE_RADIAL`` cameras.  It never fits the audit views or mutates Step-13.
"""

from __future__ import annotations

from dataclasses import replace
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from final_cv_model_fit import (
    CAPTURE_SWEEPS,
    METRIC_THRESHOLDS,
    RELIABLE_CAPTURE_SWEEPS,
    CameraReference,
    FittedComponentProfile,
    ModelAlignment,
    _apply_camera_center_normalization,
    _camera_center_world,
    _model_to_world,
    capture_sweep,
    load_step13_cameras,
    project_world_point,
    project_world_points,
    render_independent_assembly_silhouette,
)
from final_model_io import sha256_file, write_json_atomic
from final_model_validation import (
    REQUIRED_SURFACE_COMPONENTS,
    validate_registered_view_coverage_report,
    validate_surface_evidence_coverage_report,
)
from shape_geometry import analyze_shape


MINIMUM_IOU_THRESHOLD = float(METRIC_THRESHOLDS["minimum_reliable_silhouette_iou"])
MAXIMUM_DIAGNOSTIC_DIMENSION = 384
MAXIMUM_CAMERA_CENTROID_OFFSET_FRACTION = 0.035
MAXIMUM_CAMERA_AXIS_ANGLE_DEGREES = 4.0
MAXIMUM_CAMERA_AXIS_EXTENT_ERROR_FRACTION = 0.055
MASK_BBOX_WIDTH_MARGIN = 0.05
MASK_P95_ROW_WIDTH_MARGIN = 0.015
MASK_SOLIDITY_MARGIN = 0.05
MASK_BAND_WIDTH_MARGIN = 0.030
REGISTERED_MASK_MANIFEST = Path("analysis/reports/reconstruction_mask_manifest.csv")
REVIEWED_MASK_MANIFEST = Path("ml_dataset/manifest.csv")
STEP6_GEOMETRY_SUMMARY = Path("analysis/reports/geometry_summary.json")
ACCEPTED_BLEND = Path(
    "reconstruction/reference_assisted_v2/work/30_BASE_GEOMETRY_ACCEPTED.blend"
)
REQUIRED_BASE_VISUAL_VIEWS = ("front", "quarter", "side", "low", "top", "wire")
REQUIRED_BASE_VISUAL_COMPONENTS = (
    "bowl_pedestal",
    "globe_shoulder",
    "neck",
    "lid_finial",
    "overall_identity",
)
SURFACE_COMPONENT_PROFILES = {
    "bowl_pedestal": ("bowl_outer", "pedestal"),
    "globe_shoulder": ("globe", "shoulder"),
    "neck": ("neck_outer",),
    "lid_finial": ("lid", "finial"),
}
SURFACE_COMPONENT_MASK_NAMES = {
    "bowl_pedestal": "receiving_bowl_pedestal",
    "globe_shoulder": "globe",
    "neck": "neck",
    "lid_finial": "lid_finial",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_accepted_plan1_inputs(
    fit: Mapping[str, Any],
    profiles: Mapping[str, Any],
    profiles_sha256: str,
    blender_report: Mapping[str, Any],
    blend_sha256: str,
    visual_review: Mapping[str, Any],
) -> None:
    """Fail closed on a stale or weakened Plan-1/Blender candidate chain."""

    if fit.get("accepted") is not True or fit.get("metrics_passed") is not True:
        raise ValueError("Plan-1 CV fit is not accepted")
    if fit.get("metric_thresholds") != METRIC_THRESHOLDS:
        raise ValueError("Plan-1 metric thresholds differ from the frozen gate")
    candidate = fit.get("candidate_sha256")
    if not _is_sha256(candidate):
        raise ValueError("Plan-1 candidate SHA is missing")
    if visual_review.get("accepted") is not True:
        raise ValueError("Plan-1 visual review is not accepted")
    if visual_review.get("candidate_sha256") != candidate:
        raise ValueError("visual review candidate does not match Plan-1")
    if profiles.get("coordinate_contract", {}).get("axis") != "+Z":
        raise ValueError("profile coordinate contract is not +Z")
    expected_profile_hash = fit.get("artifact_hashes", {}).get(
        "final_profiles_sha256"
    )
    if expected_profile_hash != profiles_sha256:
        raise ValueError("final profile SHA does not match Plan-1")
    if blender_report.get("stage") != "geometry-validate-blender":
        raise ValueError("Blender report is not the geometry-validation checkpoint")
    if blender_report.get("blend_sha256") != blend_sha256:
        raise ValueError("Blender report blend SHA is stale")
    if not str(blender_report.get("blend_path", "")).endswith(
        ACCEPTED_BLEND.as_posix()
    ):
        raise ValueError("Blender report points at the wrong geometry candidate")


def validate_base_visual_review(
    review: Mapping[str, Any],
    project_root: Path,
    blend_sha256: str,
    plan1_candidate_sha256: str,
) -> dict[str, Any]:
    """Validate a human-opened review of the exact saved Blender candidate."""

    failures: list[str] = []
    if review.get("accepted") is not True:
        failures.append("base visual review is not accepted")
    if review.get("blend_sha256") != blend_sha256:
        failures.append("visual review blend SHA is stale")
    if review.get("plan1_candidate_sha256") != plan1_candidate_sha256:
        failures.append("visual review Plan-1 candidate SHA is stale")
    components = review.get("components")
    if not isinstance(components, Mapping):
        components = {}
        failures.append("base visual review components are missing")
    normalized_components = {
        name: str(components.get(name, "missing")).strip().lower()
        for name in REQUIRED_BASE_VISUAL_COMPONENTS
    }
    for name, value in normalized_components.items():
        if value != "pass":
            failures.append(f"base visual component veto: {name}")

    diagnostics = review.get("diagnostics")
    if not isinstance(diagnostics, Sequence) or isinstance(diagnostics, (str, bytes)):
        diagnostics = []
        failures.append("base visual review diagnostics are missing")
    records = {
        str(record.get("view")): record
        for record in diagnostics
        if isinstance(record, Mapping)
    }
    root = Path(project_root).resolve()
    for view in REQUIRED_BASE_VISUAL_VIEWS:
        record = records.get(view)
        if record is None:
            failures.append(f"missing reviewed base diagnostic: {view}")
            continue
        try:
            path = (root / str(record["path"])).resolve()
            path.relative_to(root)
        except (KeyError, ValueError):
            failures.append(f"invalid reviewed base diagnostic path: {view}")
            continue
        if not path.is_file():
            failures.append(f"missing reviewed base diagnostic file: {view}")
        elif record.get("sha256") != sha256_file(path):
            failures.append(f"reviewed base diagnostic SHA is stale: {view}")
    return {
        "passed": not failures,
        "failures": failures,
        "components": normalized_components,
        "reviewed_views": sorted(records),
    }


def validate_registered_visual_review(
    review: Mapping[str, Any],
    project_root: Path,
    blend_sha256: str,
    worst_views: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind manual worst-view classification checks to exact diagnostics."""

    failures: list[str] = []
    if review.get("accepted") is not True:
        failures.append("registered-view visual review is not accepted")
    if review.get("blend_sha256") != blend_sha256:
        failures.append("registered-view visual review blend SHA is stale")
    review_views = review.get("views")
    if not isinstance(review_views, Sequence) or isinstance(
        review_views, (str, bytes)
    ):
        review_views = []
        failures.append("registered-view visual review entries are missing")
    by_index = {
        int(record["selected_index"]): record
        for record in review_views
        if isinstance(record, Mapping) and "selected_index" in record
    }
    root = Path(project_root).resolve()
    for expected in worst_views:
        index = int(expected["selected_index"])
        record = by_index.get(index)
        if record is None:
            failures.append(f"missing registered-view visual review: {index}")
            continue
        if record.get("classification") != expected.get("classification"):
            failures.append(f"registered-view classification changed: {index}")
        if record.get("verdict") != "classification_confirmed":
            failures.append(f"registered-view verdict not confirmed: {index}")
        if record.get("diagnostic") != expected.get("diagnostic"):
            failures.append(f"registered-view diagnostic path changed: {index}")
            continue
        try:
            path = (root / str(record["diagnostic"])).resolve()
            path.relative_to(root)
        except (KeyError, ValueError):
            failures.append(f"invalid registered-view diagnostic path: {index}")
            continue
        if not path.is_file():
            failures.append(f"missing registered-view diagnostic file: {index}")
        elif record.get("diagnostic_sha256") != sha256_file(path):
            failures.append(f"registered-view diagnostic SHA is stale: {index}")
    unexpected = sorted(set(by_index) - {int(view["selected_index"]) for view in worst_views})
    if unexpected:
        failures.append(f"unexpected registered-view review entries: {unexpected}")
    return {
        "passed": not failures,
        "failures": failures,
        "reviewed_indices": sorted(by_index),
    }


def derived_audit_cameras(
    cameras: Sequence[CameraReference],
    normalization: Mapping[str, Any],
) -> tuple[dict[int, CameraReference], dict[str, Any]]:
    """Apply only the recorded sweep gauge; never per-view fit corrections."""

    if normalization.get("method") != (
        "step13_sweep_camera_center_similarity_gauge_normalization"
    ):
        raise ValueError("unexpected camera-center normalization method")
    calibrated: list[CameraReference] = []
    unchanged: list[CameraReference] = []
    for camera in cameras:
        try:
            capture_sweep(camera.selected_index)
        except ValueError:
            unchanged.append(camera)
        else:
            calibrated.append(camera)
    derived = (*_apply_camera_center_normalization(calibrated, normalization), *unchanged)
    return (
        {camera.selected_index: camera for camera in derived},
        {
            "method": normalization["method"],
            "sweeps": normalization.get("sweeps", {}),
            "canonical_translation_refinements_applied": False,
            "free_per_view_transforms": False,
            "source_step13_files_modified": False,
        },
    )


def classify_registered_view(
    silhouette_iou: float,
    source_coverage: float,
    predicted_coverage: float,
    *,
    mask_valid: bool,
    projection_valid: bool,
    camera_failure_explained: bool = False,
) -> dict[str, Any]:
    """Classify evidence without relabelling a usable poor fit as a mask defect."""

    if not mask_valid:
        classification, usable = "mask_failure", False
        basis = "independent_mask_quality_failed"
    elif not projection_valid:
        classification, usable = "camera_failure", False
        basis = "exact_camera_projection_failed_or_empty"
    elif (
        float(silhouette_iou) < MINIMUM_IOU_THRESHOLD
        and camera_failure_explained
    ):
        classification, usable = "camera_failure", False
        basis = "diagnostic_rigid_or_translation_alignment_rescues_hard_gate"
    elif float(silhouette_iou) < MINIMUM_IOU_THRESHOLD:
        classification, usable = "model_mismatch", True
        basis = "valid_mask_and_projection_remain_below_gate_after_diagnostic_alignment"
    else:
        classification, usable = "ok", True
        basis = "valid_mask_and_projection_pass_hard_gate"
    return {
        "classification": classification,
        "classification_basis": basis,
        "usable": usable,
        "silhouette_iou": float(silhouette_iou),
        "foreground_coverage_difference": abs(
            float(source_coverage) - float(predicted_coverage)
        ),
    }


def surface_sector_camera_visibility(
    camera: CameraReference,
    alignment: ModelAlignment,
    *,
    azimuth_radians: float,
    z: float,
    radius: float,
) -> dict[str, Any]:
    """Evaluate radial front-facing visibility in accepted object coordinates."""

    rotation = np.asarray(alignment.rotation_matrix, dtype=np.float64)
    translation = np.asarray(alignment.translation, dtype=np.float64)
    camera_world = _camera_center_world(camera)
    camera_local = rotation.T @ (camera_world - translation) / float(alignment.scale)
    cosine = math.cos(float(azimuth_radians))
    sine = math.sin(float(azimuth_radians))
    point = np.asarray((float(radius) * cosine, float(radius) * sine, float(z)))
    normal = np.asarray((cosine, sine, 0.0), dtype=np.float64)
    direction = camera_local - point
    length = float(np.linalg.norm(direction))
    facing = float(np.dot(normal, direction / max(length, 1e-12)))
    return {
        "visible": bool(length > 1e-9 and facing > 0.08),
        "facing_cosine": facing,
        "camera_center_local": [float(value) for value in camera_local],
    }


def surface_support_class(
    source_views: Sequence[int], *, allow_symmetry: bool = True
) -> str:
    """Map physically verified view support to the frozen provenance classes."""

    count = len(set(int(value) for value in source_views))
    if count >= 2:
        return "direct_multi_view"
    if count == 1:
        return "reviewed_single_or_detail"
    return "symmetry_repetition" if allow_symmetry else "hidden_generic_fill"


def _alignment(record: Mapping[str, Any]) -> ModelAlignment:
    return ModelAlignment(
        scale=float(record["scale"]),
        rotation_matrix=tuple(
            tuple(float(value) for value in row)
            for row in record["rotation_matrix"]
        ),
        translation=tuple(float(value) for value in record["translation"]),
    )


def _profiles(payload: Mapping[str, Any]) -> dict[str, FittedComponentProfile]:
    provenance = payload.get("profile_provenance", {})
    output: dict[str, FittedComponentProfile] = {}
    for name, sections in payload.get("profiles", {}).items():
        record = provenance.get(name, {})
        output[str(name)] = FittedComponentProfile(
            name=str(name),
            sections=tuple(
                (float(section[0]), float(section[1])) for section in sections
            ),
            measurement_status=str(record.get("measurement_status", "unknown")),
            source_views=tuple(int(value) for value in record.get("source_views", [])),
        )
    return output


def _interpolate_profile_radius(
    sections: Sequence[tuple[float, float]], z: float
) -> float:
    ordered = sorted((float(level), float(radius)) for level, radius in sections)
    return float(
        np.interp(
            float(z),
            np.asarray([level for level, _ in ordered], dtype=np.float64),
            np.asarray([radius for _, radius in ordered], dtype=np.float64),
        )
    )


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mask_sources(project_root: Path) -> tuple[dict[int, dict[str, Any]], str]:
    frozen_path = project_root / REGISTERED_MASK_MANIFEST
    reviewed_path = project_root / REVIEWED_MASK_MANIFEST
    frozen: dict[int, dict[str, Any]] = {}
    for row in _csv_rows(frozen_path):
        index = int(row["selected_index"])
        frozen[index] = {
            "filename": row["filename"],
            "mask_path": row["reconstruction_mask_path"],
            "mask_sha256": row["reconstruction_mask_sha256"],
            "source_quality_condition": "frozen_cnn_unreviewed",
            "mask_kind": "frozen_reconstruction_cnn_mask",
        }
    for row in _csv_rows(reviewed_path):
        index = int(row["selected_index"])
        if index not in frozen:
            continue
        frozen[index] = {
            "filename": row["filename"],
            "mask_path": row["mask_path"],
            "mask_sha256": row["mask_sha256"],
            "source_quality_condition": row.get("quality_condition") or "reviewed",
            "mask_kind": "reviewed_training_mask",
        }
    return frozen, sha256_file(frozen_path)


def _resize_camera(camera: CameraReference, size: tuple[int, int]) -> CameraReference:
    source_width, source_height = camera.image_size
    scale_x = size[0] / float(source_width)
    scale_y = size[1] / float(source_height)
    if not math.isclose(scale_x, scale_y, abs_tol=1e-3):
        raise ValueError("audit resize changed camera aspect ratio")
    if camera.camera_model != "SIMPLE_RADIAL" or len(camera.camera_params) != 4:
        raise ValueError("audit camera is not SIMPLE_RADIAL")
    focal, cx, cy, distortion = camera.camera_params
    return replace(
        camera,
        camera_params=(
            float(focal) * scale_x,
            float(cx) * scale_x,
            float(cy) * scale_y,
            float(distortion),
        ),
        image_size=size,
    )


def _load_resized_mask(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((2, 2), dtype=np.uint8), {
            "valid": False,
            "failure_reasons": ["unreadable_mask"],
        }
    binary = np.where(mask > 127, 255, 0).astype(np.uint8)
    scale = MAXIMUM_DIAGNOSTIC_DIMENSION / max(binary.shape[:2])
    size = (
        max(2, int(round(binary.shape[1] * scale))),
        max(2, int(round(binary.shape[0] * scale))),
    )
    resized = cv2.resize(binary, size, interpolation=cv2.INTER_NEAREST)
    foreground = resized > 0
    fraction = float(np.count_nonzero(foreground)) / foreground.size
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8), connectivity=8
    )
    areas = sorted((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)), reverse=True)
    largest_share = areas[0] / max(sum(areas), 1) if areas else 0.0
    reasons: list[str] = []
    if not 0.015 <= fraction <= 0.75:
        reasons.append("implausible_foreground_fraction")
    if largest_share < 0.90:
        reasons.append("disconnected_foreground_contamination")
    if not areas:
        reasons.append("empty_foreground")
    if areas:
        y, x = np.nonzero(foreground)
        object_height = float(np.max(y) - np.min(y) + 1)
        object_width = float(np.max(x) - np.min(x) + 1)
        row_widths = np.asarray(
            [
                np.count_nonzero(foreground[row])
                for row in range(int(np.min(y)), int(np.max(y)) + 1)
            ],
            dtype=np.float64,
        )
        points = np.column_stack((x, y)).astype(np.int32)
        hull_area = float(cv2.contourArea(cv2.convexHull(points)))
        bbox_width_per_height = object_width / max(object_height, 1.0)
        p95_row_width_per_height = float(np.percentile(row_widths, 95.0)) / max(
            object_height, 1.0
        )
        silhouette_solidity = float(np.count_nonzero(foreground)) / max(
            hull_area, 1.0
        )
        band_widths: list[float] = []
        minimum_y = int(np.min(y))
        for band in range(10):
            start = minimum_y + int(round(band * object_height / 10.0))
            stop = minimum_y + int(round((band + 1) * object_height / 10.0))
            stop = max(stop, start + 1)
            widths = [
                np.count_nonzero(foreground[row])
                for row in range(start, min(stop, foreground.shape[0]))
            ]
            band_widths.append(
                float(np.percentile(widths, 90.0)) / max(object_height, 1.0)
                if widths
                else 0.0
            )
    else:
        bbox_width_per_height = 0.0
        p95_row_width_per_height = 0.0
        silhouette_solidity = 0.0
        band_widths = [0.0] * 10
    return resized, {
        "valid": not reasons,
        "failure_reasons": reasons,
        "foreground_fraction": fraction,
        "largest_component_share": largest_share,
        "bbox_width_per_height": bbox_width_per_height,
        "p95_row_width_per_height": p95_row_width_per_height,
        "silhouette_solidity": silhouette_solidity,
        "band_p90_width_per_height": band_widths,
        "image_size": [size[0], size[1]],
    }


def assess_unreviewed_mask_shape(
    quality: Mapping[str, Any], envelope: Mapping[str, Any]
) -> dict[str, Any]:
    """Reject only masks with multiple deviations from reviewed sweep evidence."""

    assessed = dict(quality)
    assessed["failure_reasons"] = list(quality.get("failure_reasons", []))
    anomalies: list[str] = []
    if float(quality.get("bbox_width_per_height", 0.0)) > (
        float(envelope["maximum_bbox_width_per_height"]) + MASK_BBOX_WIDTH_MARGIN
    ):
        anomalies.append("bbox_width_above_reviewed_envelope")
    if float(quality.get("p95_row_width_per_height", 0.0)) > (
        float(envelope["maximum_p95_row_width_per_height"])
        + MASK_P95_ROW_WIDTH_MARGIN
    ):
        anomalies.append("p95_row_width_above_reviewed_envelope")
    if float(quality.get("silhouette_solidity", 0.0)) < (
        float(envelope["minimum_solidity"]) - MASK_SOLIDITY_MARGIN
    ):
        anomalies.append("solidity_below_reviewed_envelope")
    assessed["reviewed_shape_envelope"] = dict(envelope)
    assessed["reviewed_shape_anomalies"] = anomalies
    band_anomalies: list[int] = []
    measured_bands = quality.get("band_p90_width_per_height")
    maximum_bands = envelope.get("maximum_band_p90_width_per_height")
    if (
        isinstance(measured_bands, Sequence)
        and not isinstance(measured_bands, (str, bytes))
        and isinstance(maximum_bands, Sequence)
        and not isinstance(maximum_bands, (str, bytes))
        and len(measured_bands) == len(maximum_bands)
    ):
        band_anomalies = [
            index
            for index, (measured, maximum) in enumerate(
                zip(measured_bands, maximum_bands, strict=True)
            )
            if float(measured) > float(maximum) + MASK_BAND_WIDTH_MARGIN
        ]
    assessed["reviewed_band_width_anomalies"] = band_anomalies
    consecutive_band_contamination = any(
        second == first + 1
        for first, second in zip(band_anomalies, band_anomalies[1:])
    )
    if len(anomalies) >= 2 or consecutive_band_contamination:
        assessed["valid"] = False
        reason = (
            "axial_band_width_contamination"
            if consecutive_band_contamination
            else "reviewed_shape_envelope_contamination"
        )
        if reason not in assessed["failure_reasons"]:
            assessed["failure_reasons"].append(reason)
    return assessed


def _reviewed_mask_shape_envelopes(project_root: Path) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {
        sweep: [] for sweep in RELIABLE_CAPTURE_SWEEPS
    }
    for row in _csv_rows(project_root / REVIEWED_MASK_MANIFEST):
        index = int(row["selected_index"])
        try:
            sweep = capture_sweep(index)
        except ValueError:
            continue
        if sweep not in grouped:
            continue
        _, quality = _load_resized_mask(project_root / row["mask_path"])
        if quality["valid"]:
            grouped[sweep].append((index, quality))
    envelopes: dict[str, dict[str, Any]] = {}
    for sweep, records in grouped.items():
        if len(records) < 2:
            raise ValueError(f"insufficient reviewed mask shape evidence for {sweep}")
        envelopes[sweep] = {
            "source_views": [index for index, _ in records],
            "maximum_bbox_width_per_height": max(
                float(record["bbox_width_per_height"]) for _, record in records
            ),
            "maximum_p95_row_width_per_height": max(
                float(record["p95_row_width_per_height"]) for _, record in records
            ),
            "minimum_solidity": min(
                float(record["silhouette_solidity"]) for _, record in records
            ),
            "maximum_band_p90_width_per_height": [
                max(
                    float(record["band_p90_width_per_height"][band])
                    for _, record in records
                )
                for band in range(10)
            ],
            "decision_rule": (
                "invalid_only_when_at_least_two_shape_statistics_exceed_"
                "conservative_reviewed_sweep_margins_or_two_consecutive_axial_"
                "bands_exceed_the_reviewed_width_envelope"
            ),
        }
    return envelopes


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    first_binary = first > 0
    second_binary = second > 0
    union = int(np.count_nonzero(first_binary | second_binary))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(first_binary & second_binary)) / union


def _centroid_axis(mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    y, x = np.nonzero(mask > 0)
    if x.size < 8:
        raise ValueError("silhouette has too few foreground pixels")
    points = np.column_stack((x.astype(np.float64), y.astype(np.float64)))
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, int(np.argmax(values))]
    angle = math.degrees(math.atan2(float(direction[1]), float(direction[0]))) % 180.0
    object_height = float(np.max(y) - np.min(y) + 1)
    return centroid, angle, object_height


def projected_axis_mask_consistency(
    observed: np.ndarray,
    projected_axis_xy: np.ndarray,
) -> dict[str, Any]:
    """Cross-check the frozen camera/object axis without using model radii."""

    try:
        centroid, observed_angle, object_height = _centroid_axis(observed)
        axis = np.asarray(projected_axis_xy, dtype=np.float64).reshape(2, 2)
        direction = axis[1] - axis[0]
        length = float(np.linalg.norm(direction))
        if not math.isfinite(length) or length <= 1e-9:
            raise ValueError("projected object axis is degenerate")
    except ValueError as error:
        return {
            "passed": False,
            "perpendicular_offset_object_height": 1.0,
            "axis_angle_difference_degrees": 90.0,
            "failure": str(error),
        }
    unit = direction / length
    delta = centroid - axis[0]
    perpendicular = abs(float(unit[0] * delta[1] - unit[1] * delta[0]))
    projected_angle = math.degrees(math.atan2(float(unit[1]), float(unit[0]))) % 180.0
    angle_difference = abs(projected_angle - observed_angle) % 180.0
    angle_difference = min(angle_difference, 180.0 - angle_difference)
    offset_fraction = perpendicular / max(object_height, 1.0)
    y, x = np.nonzero(observed > 0)
    observed_points = np.column_stack((x, y)).astype(np.float64)
    observed_axis_unit = np.asarray(
        (
            math.cos(math.radians(observed_angle)),
            math.sin(math.radians(observed_angle)),
        ),
        dtype=np.float64,
    )
    observed_positions = observed_points @ observed_axis_unit
    projected_positions = axis @ observed_axis_unit
    observed_min, observed_max = (
        float(np.min(observed_positions)),
        float(np.max(observed_positions)),
    )
    projected_min, projected_max = (
        float(np.min(projected_positions)),
        float(np.max(projected_positions)),
    )
    axial_extent_error = max(
        abs(projected_min - observed_min),
        abs(projected_max - observed_max),
    ) / max(object_height, 1.0)
    axial_scale_ratio = (projected_max - projected_min) / max(
        observed_max - observed_min, 1.0
    )
    return {
        "passed": bool(
            offset_fraction <= MAXIMUM_CAMERA_CENTROID_OFFSET_FRACTION
            and angle_difference <= MAXIMUM_CAMERA_AXIS_ANGLE_DEGREES
            and axial_extent_error <= MAXIMUM_CAMERA_AXIS_EXTENT_ERROR_FRACTION
        ),
        "perpendicular_offset_object_height": float(offset_fraction),
        "axis_angle_difference_degrees": float(angle_difference),
        "axial_extent_error_object_height": float(axial_extent_error),
        "projected_to_observed_axis_extent_ratio": float(axial_scale_ratio),
        "projected_axis_angle_degrees": float(projected_angle),
        "observed_mask_axis_angle_degrees": float(observed_angle),
        "projected_axis_pixels": axis.tolist(),
        "thresholds": {
            "maximum_perpendicular_offset_object_height": (
                MAXIMUM_CAMERA_CENTROID_OFFSET_FRACTION
            ),
            "maximum_axis_angle_difference_degrees": (
                MAXIMUM_CAMERA_AXIS_ANGLE_DEGREES
            ),
            "maximum_axis_extent_error_object_height": (
                MAXIMUM_CAMERA_AXIS_EXTENT_ERROR_FRACTION
            ),
        },
    }


def _translate_mask(mask: np.ndarray, translation_xy: Sequence[float]) -> np.ndarray:
    """Translate a binary mask for diagnosis without changing scored evidence."""

    transform = np.asarray(
        [
            [1.0, 0.0, float(translation_xy[0])],
            [0.0, 1.0, float(translation_xy[1])],
        ],
        dtype=np.float64,
    )
    return cv2.warpAffine(
        np.where(mask > 0, 255, 0).astype(np.uint8),
        transform,
        (int(mask.shape[1]), int(mask.shape[0])),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _rigid_align_mask(
    mask: np.ndarray,
    source_centroid: np.ndarray,
    target_centroid: np.ndarray,
    rotation_degrees: float,
) -> np.ndarray:
    """Apply the closed-form centroid/PCA rigid registration for diagnosis."""

    radians = math.radians(float(rotation_degrees))
    cosine = math.cos(radians)
    sine = math.sin(radians)
    linear = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    translation = target_centroid - linear @ source_centroid
    transform = np.column_stack((linear, translation))
    return cv2.warpAffine(
        np.where(mask > 0, 255, 0).astype(np.uint8),
        transform,
        (int(mask.shape[1]), int(mask.shape[0])),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def silhouette_camera_consistency(
    observed: np.ndarray, predicted: np.ndarray
) -> dict[str, Any]:
    """Diagnose camera/pose residuals without altering the scored projection.

    The centroid-aligned IoU is a counterfactual diagnostic only.  If a pure
    image translation makes an otherwise failing frozen-camera projection clear
    the unchanged hard gate, the failure is attributable to camera registration
    rather than the common local vessel profile.  The translated silhouette is
    never used as the reported metric or as an optimized camera pose.
    """

    try:
        observed_centroid, observed_angle, object_height = _centroid_axis(observed)
        predicted_centroid, predicted_angle, _ = _centroid_axis(predicted)
    except ValueError as error:
        return {
            "passed": False,
            "centroid_offset_object_height": 1.0,
            "axis_angle_difference_degrees": 90.0,
            "centroid_aligned_iou": 0.0,
            "translation_only_explains_gate_failure": False,
            "rigid_aligned_iou": 0.0,
            "rigid_alignment_explains_gate_failure": False,
            "diagnostic_only_translation_pixels": [0.0, 0.0],
            "diagnostic_only_rotation_degrees": 0.0,
            "translation_applied_to_metric": False,
            "failure": str(error),
        }
    translation = observed_centroid - predicted_centroid
    translated = _translate_mask(predicted, translation)
    raw_iou = _iou(observed, predicted)
    aligned_iou = _iou(observed, translated)
    translation_explains_failure = bool(
        raw_iou < MINIMUM_IOU_THRESHOLD
        and aligned_iou >= MINIMUM_IOU_THRESHOLD
    )
    centroid_offset = float(
        np.linalg.norm(predicted_centroid - observed_centroid) / max(object_height, 1.0)
    )
    angle_difference = abs(predicted_angle - observed_angle) % 180.0
    angle_difference = min(angle_difference, 180.0 - angle_difference)
    signed_rotation = (observed_angle - predicted_angle + 90.0) % 180.0 - 90.0
    rigid_aligned = _rigid_align_mask(
        predicted,
        predicted_centroid,
        observed_centroid,
        signed_rotation,
    )
    rigid_aligned_iou = _iou(observed, rigid_aligned)
    rigid_alignment_explains_failure = bool(
        raw_iou < MINIMUM_IOU_THRESHOLD
        and rigid_aligned_iou >= MINIMUM_IOU_THRESHOLD
    )
    passed = bool(
        centroid_offset <= MAXIMUM_CAMERA_CENTROID_OFFSET_FRACTION
        and angle_difference <= MAXIMUM_CAMERA_AXIS_ANGLE_DEGREES
        and not translation_explains_failure
        and not rigid_alignment_explains_failure
    )
    return {
        "passed": passed,
        "centroid_offset_object_height": centroid_offset,
        "axis_angle_difference_degrees": float(angle_difference),
        "centroid_aligned_iou": aligned_iou,
        "translation_only_explains_gate_failure": translation_explains_failure,
        "rigid_aligned_iou": rigid_aligned_iou,
        "rigid_alignment_explains_gate_failure": (
            rigid_alignment_explains_failure
        ),
        "diagnostic_only_translation_pixels": [
            float(translation[0]),
            float(translation[1]),
        ],
        "diagnostic_only_rotation_degrees": float(signed_rotation),
        "translation_applied_to_metric": False,
        "thresholds": {
            "maximum_centroid_offset_object_height": (
                MAXIMUM_CAMERA_CENTROID_OFFSET_FRACTION
            ),
            "maximum_axis_angle_difference_degrees": (
                MAXIMUM_CAMERA_AXIS_ANGLE_DEGREES
            ),
        },
    }


def _relative(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _write_view_diagnostic(
    project_root: Path,
    v2_root: Path,
    camera: CameraReference,
    mask: np.ndarray,
    predicted: np.ndarray,
    record: Mapping[str, Any],
) -> Path:
    source_path = project_root / "preprocessing/pycolmap_input/images" / camera.filename
    with Image.open(source_path) as opened:
        source = opened.convert("RGB").resize(
            (mask.shape[1], mask.shape[0]), Image.Resampling.LANCZOS
        )
    source_array = np.asarray(source, dtype=np.uint8).copy()
    source_fg = mask > 0
    predicted_fg = predicted > 0
    overlay = source_array.astype(np.float32)
    overlay[source_fg] = 0.65 * overlay[source_fg] + 0.35 * np.array(
        [255.0, 55.0, 45.0], dtype=np.float32
    )
    overlay[predicted_fg] = 0.65 * overlay[predicted_fg] + 0.35 * np.array(
        [35.0, 255.0, 80.0], dtype=np.float32
    )
    image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")
    canvas = Image.new("RGB", (image.width, image.height + 54), (18, 18, 18))
    canvas.paste(image, (0, 54))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    title = (
        f"{camera.selected_index:03d} {camera.filename} | "
        f"IoU {float(record['silhouette_iou']):.4f} | "
        f"{record['classification']} | red=mask green=model"
    )
    draw.text((8, 8), title, fill=(240, 240, 240), font=font)
    draw.text(
        (8, 29),
        f"{record['sweep']} | exact SIMPLE_RADIAL | {record['mask_kind']}",
        fill=(190, 190, 190),
        font=font,
    )
    path = v2_root / "diagnostics" / "21_registered_view_coverage" / (
        f"{camera.selected_index:03d}_overlay.png"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def _surface_evidence_report(
    project_root: Path,
    v2_root: Path,
    blend_path: Path,
    blend_hash: str,
    profiles_payload: Mapping[str, Any],
    profiles_hash: str,
    cameras: Mapping[int, CameraReference],
    alignments: Mapping[str, ModelAlignment],
) -> dict[str, Any]:
    """Classify sectors from camera-facing, component-mask-confirmed support."""

    provenance = profiles_payload.get("profile_provenance", {})
    profiles = _profiles(profiles_payload)
    component_mask_root = Path(v2_root) / "evidence" / "component_masks"
    mask_hash_lines: list[str] = []
    component_masks: dict[tuple[int, str], np.ndarray] = {}
    for component, mask_name in SURFACE_COMPONENT_MASK_NAMES.items():
        evidence_profiles = list(SURFACE_COMPONENT_PROFILES[component])
        if component == "bowl_pedestal":
            evidence_profiles.append("bowl_inner")
        elif component == "neck":
            evidence_profiles.append("neck_inner")
        candidate_views = {
            int(view)
            for profile_name in evidence_profiles
            for view in provenance.get(profile_name, {}).get("source_views", [])
        }
        for view in sorted(candidate_views):
            path = component_mask_root / f"{view:03d}_{mask_name}.png"
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            component_masks[(view, component)] = np.where(image > 127, 255, 0).astype(
                np.uint8
            )
            mask_hash_lines.append(f"{path.name}:{sha256_file(path)}")
    component_masks_sha256 = hashlib.sha256(
        "\n".join(sorted(mask_hash_lines)).encode("utf-8")
    ).hexdigest()

    def support_for_view(
        component: str,
        view: int,
        angle: float,
    ) -> dict[str, Any] | None:
        camera = cameras.get(view)
        mask = component_masks.get((view, component))
        if camera is None or mask is None:
            return None
        try:
            sweep = capture_sweep(view)
        except ValueError:
            return None
        alignment = alignments.get(sweep)
        if alignment is None:
            return None
        resized_camera = _resize_camera(camera, (mask.shape[1], mask.shape[0]))
        visible_samples = 0
        mask_hits = 0
        sample_records: list[dict[str, Any]] = []
        for profile_name in SURFACE_COMPONENT_PROFILES[component]:
            sections = profiles[profile_name].sections
            z_low = float(sections[0][0])
            z_high = float(sections[-1][0])
            for fraction in (0.22, 0.50, 0.78):
                z = z_low + fraction * (z_high - z_low)
                radius = _interpolate_profile_radius(sections, z)
                visibility = surface_sector_camera_visibility(
                    resized_camera,
                    alignment,
                    azimuth_radians=angle,
                    z=z,
                    radius=radius,
                )
                if not visibility["visible"]:
                    continue
                visible_samples += 1
                # Slightly inset the projected sample so rasterized boundary
                # disagreement cannot turn a visibly supported sector into a
                # false negative.  Facing is still evaluated on the true host.
                local = np.asarray(
                    (
                        0.96 * radius * math.cos(angle),
                        0.96 * radius * math.sin(angle),
                        z,
                    ),
                    dtype=np.float64,
                )
                world = _model_to_world(alignment, local.reshape(1, 3))[0]
                pixel = project_world_point(resized_camera, world)
                hit = False
                if pixel is not None:
                    x = int(round(pixel[0]))
                    y = int(round(pixel[1]))
                    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                        y0, y1 = max(0, y - 3), min(mask.shape[0], y + 4)
                        x0, x1 = max(0, x - 3), min(mask.shape[1], x + 4)
                        hit = bool(np.any(mask[y0:y1, x0:x1] > 0))
                mask_hits += int(hit)
                sample_records.append(
                    {
                        "profile": profile_name,
                        "z": z,
                        "projected_pixel": (
                            None if pixel is None else [float(pixel[0]), float(pixel[1])]
                        ),
                        "component_mask_hit": hit,
                        "facing_cosine": visibility["facing_cosine"],
                    }
                )
        minimum_hits = max(1, int(math.ceil(visible_samples * 0.34)))
        if visible_samples == 0 or mask_hits < minimum_hits:
            return None
        return {
            "selected_index": view,
            "sweep": sweep,
            "visible_samples": visible_samples,
            "component_mask_hits": mask_hits,
            "minimum_mask_hits": minimum_hits,
            "samples": sample_records,
        }

    components: dict[str, list[dict[str, Any]]] = {}
    for component, names in SURFACE_COMPONENT_PROFILES.items():
        source_view_candidates = sorted(
            {
                int(view)
                for name in names
                for view in provenance.get(name, {}).get("source_views", [])
            }
        )
        records: list[dict[str, Any]] = []
        for start in range(0, 360, 30):
            center = math.radians(start + 15.0)
            evidence = [
                record
                for view in source_view_candidates
                if (record := support_for_view(component, view, center)) is not None
            ]
            supporting_views = sorted(
                {int(record["selected_index"]) for record in evidence}
            )
            records.append(
                {
                    "sector": f"azimuth_{start:03d}_{start + 30:03d}/elevation_side_to_high",
                    "support_class": surface_support_class(supporting_views),
                    "source_support": {
                        "support_count": len(supporting_views),
                        "source_views": supporting_views,
                        "camera_component_mask_evidence": evidence,
                    },
                    "provenance": {
                        "source": (
                            "accepted Step13 camera center and sweep alignment; radial "
                            "facing test; projected host samples inside reviewed component masks"
                        ),
                        "manifest_sha256": component_masks_sha256,
                    },
                }
            )
        records.append(
            {
                "sector": "axial_underside_hidden",
                "support_class": "hidden_generic_fill",
                    "source_support": {"support_count": 0, "source_views": []},
                    "provenance": {
                        "source": "explicit unobserved underside classification",
                        "manifest_sha256": component_masks_sha256,
                    },
                }
            )
        if component in {"bowl_pedestal", "neck"}:
            inner_name = "bowl_inner" if component == "bowl_pedestal" else "neck_inner"
            inner_view_candidates = [
                int(view)
                for view in provenance.get(inner_name, {}).get("source_views", [])
            ]
            inner_evidence: list[dict[str, Any]] = []
            inner_profile = profiles[inner_name]
            z = float(inner_profile.sections[-1][0]) - 0.18 * (
                float(inner_profile.sections[-1][0])
                - float(inner_profile.sections[0][0])
            )
            radius = 0.72 * _interpolate_profile_radius(inner_profile.sections, z)
            for view in inner_view_candidates:
                camera = cameras.get(view)
                mask = component_masks.get((view, component))
                if camera is None or mask is None:
                    continue
                try:
                    sweep = capture_sweep(view)
                except ValueError:
                    continue
                alignment = alignments.get(sweep)
                if alignment is None:
                    continue
                resized_camera = _resize_camera(camera, (mask.shape[1], mask.shape[0]))
                hits = 0
                projected_samples: list[list[float]] = []
                for angle in np.linspace(0.0, math.tau, 8, endpoint=False):
                    local = np.asarray(
                        (radius * math.cos(angle), radius * math.sin(angle), z),
                        dtype=np.float64,
                    )
                    world = _model_to_world(alignment, local.reshape(1, 3))[0]
                    pixel = project_world_point(resized_camera, world)
                    if pixel is None:
                        continue
                    x = int(round(pixel[0]))
                    y = int(round(pixel[1]))
                    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                        hit = bool(mask[y, x] > 0)
                        hits += int(hit)
                        if hit:
                            projected_samples.append([float(pixel[0]), float(pixel[1])])
                if hits >= 2:
                    inner_evidence.append(
                        {
                            "selected_index": view,
                            "sweep": sweep,
                            "component_mask_hits": hits,
                            "projected_pixels": projected_samples,
                        }
                    )
            inner_views = sorted(
                {int(record["selected_index"]) for record in inner_evidence}
            )
            records.append(
                {
                    "sector": "axial_opening_inner_wall",
                    "support_class": (
                        "reviewed_single_or_detail"
                        if inner_views
                        else "hidden_generic_fill"
                    ),
                    "source_support": {
                        "support_count": len(inner_views),
                        "source_views": inner_views,
                        "camera_component_mask_evidence": inner_evidence,
                    },
                    "provenance": {
                        "source": (
                            f"accepted {inner_name} profile projected through top-sweep "
                            "Step13 cameras into reviewed component masks"
                        ),
                        "manifest_sha256": component_masks_sha256,
                    },
                }
            )
        components[component] = records

    candidate = {
        "schema_version": 1,
        "accepted": True,
        "blend_path": _relative(blend_path, project_root),
        "blend_sha256": blend_hash,
        "support_classes": [
            "direct_multi_view",
            "reviewed_single_or_detail",
            "symmetry_repetition",
            "hidden_generic_fill",
        ],
        "components": components,
        "visibility_derivation": {
            "method": (
                "camera_center_in_accepted_object_coordinates_plus_radial_facing_"
                "and_projected_component_mask_membership"
            ),
            "sector_width_degrees": 30,
            "profile_sample_fractions": [0.22, 0.50, 0.78],
            "component_mask_root": _relative(component_mask_root, project_root),
            "component_masks_sha256": component_masks_sha256,
            "profile_manifest_sha256": profiles_hash,
            "source_files": len(mask_hash_lines),
        },
        "claim_boundary": (
            "direct support requires physical camera-facing visibility and component-mask "
            "membership; unsupported radial sectors remain symmetry inference and the "
            "underside remains hidden generic fill"
        ),
    }
    validation = validate_surface_evidence_coverage_report(candidate)
    candidate["accepted"] = bool(validation["passed"])
    candidate["validation"] = validation
    return candidate


def projected_edge_residuals(
    projected_points: np.ndarray,
    edge_map: np.ndarray,
    *,
    object_height: float,
) -> dict[str, Any]:
    """Measure projected candidate samples against a frozen source edge map."""

    if edge_map.ndim != 2 or edge_map.dtype != np.uint8:
        raise ValueError("edge map must be uint8 single-channel")
    if not math.isfinite(float(object_height)) or float(object_height) <= 0.0:
        raise ValueError("object_height must be positive")
    points = np.asarray(projected_points, dtype=np.float64).reshape(-1, 2)
    inside = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0)
        & (points[:, 0] < edge_map.shape[1])
        & (points[:, 1] >= 0)
        & (points[:, 1] < edge_map.shape[0])
    )
    points = points[inside]
    if not len(points):
        return {
            "sample_count": 0,
            "median_pixels": None,
            "p90_pixels": None,
            "median_fraction_object_height": None,
            "p90_fraction_object_height": None,
        }
    distance = cv2.distanceTransform(
        np.where(edge_map > 0, 0, 255).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    xy = np.rint(points).astype(np.int32)
    values = distance[xy[:, 1], xy[:, 0]].astype(np.float64)
    median = float(np.median(values))
    p90 = float(np.percentile(values, 90.0))
    return {
        "sample_count": int(len(values)),
        "median_pixels": median,
        "p90_pixels": p90,
        "median_fraction_object_height": median / float(object_height),
        "p90_fraction_object_height": p90 / float(object_height),
    }


def _axis_angle_degrees(points: np.ndarray) -> float:
    first, second = np.asarray(points, dtype=np.float64).reshape(2, 2)
    direction = second - first
    return math.degrees(math.atan2(float(direction[1]), float(direction[0]))) % 180.0


def _axis_angle_difference(first: float, second: float) -> float:
    difference = abs(float(first) - float(second)) % 180.0
    return min(difference, 180.0 - difference)


def _profile_ring_points(
    profile: FittedComponentProfile, z: float, *, samples: int = 180
) -> np.ndarray:
    radius = _interpolate_profile_radius(profile.sections, z)
    angles = np.linspace(0.0, math.tau, samples, endpoint=False)
    return np.column_stack(
        (
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.full(samples, float(z), dtype=np.float64),
        )
    )


def _project_local_points(
    camera: CameraReference,
    alignment: ModelAlignment,
    local_points: np.ndarray,
) -> np.ndarray:
    world = _model_to_world(alignment, np.asarray(local_points, dtype=np.float64))
    projected = project_world_points(camera, world, check_cheirality=True)
    return np.asarray(projected, dtype=np.float64)


def _step6_crosscheck(
    project_root: Path,
    v2_root: Path,
    *,
    blend_hash: str,
    plan1_candidate_sha256: str,
    profiles_hash: str,
    profiles: Mapping[str, FittedComponentProfile],
    cameras: Mapping[int, CameraReference],
    alignments: Mapping[str, ModelAlignment],
) -> dict[str, Any]:
    """Project the exact Plan-2 candidate into the two frozen Step-6 views."""

    summary_path = project_root / STEP6_GEOMETRY_SUMMARY
    payload = _read_json(summary_path)
    diagnostic_dir = v2_root / "diagnostics" / "22_step6_candidate_crosscheck"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    for stale in diagnostic_dir.glob("*_candidate_overlay.png"):
        stale.unlink()

    feature_specs: tuple[tuple[str, str, float], ...] = (
        ("pedestal_foot", "pedestal", 0.018),
        ("pedestal_upper", "pedestal", 0.184),
        ("bowl_rim", "bowl_outer", 0.443),
        ("shoulder_ring_01", "shoulder", 0.568),
        ("shoulder_ring_02", "shoulder", 0.592),
        ("shoulder_ring_03", "shoulder", 0.618),
        ("lid_tier_lower", "lid", 0.827),
        ("lid_tier_mid", "lid", 0.881),
        ("lid_tier_upper", "lid", 0.935),
    )
    records: list[dict[str, Any]] = []
    for historical in payload.get("shape_metrics", []):
        index = int(historical["index"])
        filename = str(historical["filename"])
        source_path = project_root / "preprocessing" / "pycolmap_input" / "images" / filename
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        camera = cameras.get(index)
        try:
            sweep = capture_sweep(index)
        except ValueError:
            sweep = "unmodelled"
        alignment = alignments.get(sweep)
        if image is None or camera is None or alignment is None:
            records.append(
                {
                    "selected_index": index,
                    "filename": filename,
                    "accepted": False,
                    "failure": "missing source, camera, or accepted sweep alignment",
                }
            )
            continue
        analysis = analyze_shape(image)
        if analysis.geometry is None or analysis.selected_candidate is None:
            records.append(
                {
                    "selected_index": index,
                    "filename": filename,
                    "accepted": False,
                    "failure": "Step-6 classical contour is unavailable on rerun",
                }
            )
            continue
        height, width = analysis.edge_result.analysis_image.shape[:2]
        resized_camera = _resize_camera(camera, (width, height))
        predicted = render_independent_assembly_silhouette(
            profiles,
            resized_camera,
            alignment,
            (width, height),
        )
        candidate_contours, _ = cv2.findContours(
            np.where(predicted > 0, 255, 0).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        candidate_contour = max(candidate_contours, key=cv2.contourArea)
        observed_contour_map = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(
            observed_contour_map,
            [analysis.selected_candidate.contour],
            -1,
            255,
            2,
        )
        object_height = float(analysis.geometry.bounding_box[3])
        silhouette_residual = projected_edge_residuals(
            candidate_contour.reshape(-1, 2),
            observed_contour_map,
            object_height=object_height,
        )

        axis_local = np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
        projected_axis = _project_local_points(resized_camera, alignment, axis_local)
        projected_axis_angle = _axis_angle_degrees(projected_axis)
        axis_difference = _axis_angle_difference(
            projected_axis_angle, analysis.geometry.principal_axis_angle_deg
        )

        feature_records: list[dict[str, Any]] = []
        feature_points: dict[str, np.ndarray] = {}
        for name, profile_name, z in feature_specs:
            profile = profiles[profile_name]
            clamped_z = min(max(z, profile.sections[0][0]), profile.sections[-1][0])
            projected = _project_local_points(
                resized_camera,
                alignment,
                _profile_ring_points(profile, clamped_z),
            )
            feature_points[name] = projected
            feature_records.append(
                {
                    "feature": name,
                    "profile": profile_name,
                    "z": float(clamped_z),
                    "edge_residual": projected_edge_residuals(
                        projected,
                        analysis.edge_result.edges,
                        object_height=object_height,
                    ),
                }
            )

        ellipse_comparison: dict[str, Any]
        if analysis.geometry.ellipse is None:
            ellipse_comparison = {
                "available": False,
                "reason": "Step-6 ellipse was rejected by its frozen residual gate",
            }
        else:
            projected_rim = feature_points["bowl_rim"].astype(np.float32)
            fitted = cv2.fitEllipse(projected_rim.reshape(-1, 1, 2))
            (cx, cy), (axis_a, axis_b), angle = fitted
            source_ellipse = analysis.geometry.ellipse
            source_axes = sorted(source_ellipse.axes)
            candidate_axes = sorted((float(axis_a), float(axis_b)))
            ellipse_comparison = {
                "available": True,
                "candidate_center": [float(cx), float(cy)],
                "source_center": list(source_ellipse.center),
                "center_error_fraction_object_height": float(
                    np.linalg.norm(
                        np.asarray((cx, cy)) - np.asarray(source_ellipse.center)
                    )
                    / object_height
                ),
                "candidate_axes": candidate_axes,
                "source_axes": source_axes,
                "maximum_relative_axis_error": max(
                    abs(candidate - source) / max(source, 1.0)
                    for candidate, source in zip(candidate_axes, source_axes, strict=True)
                ),
                "angle_difference_degrees": _axis_angle_difference(
                    float(angle), source_ellipse.angle_deg
                ),
            }

        canvas = analysis.edge_result.analysis_image.copy()
        cv2.drawContours(
            canvas,
            [analysis.selected_candidate.contour],
            -1,
            (255, 140, 30),
            3,
        )
        cv2.drawContours(canvas, [candidate_contour], -1, (45, 230, 80), 3)
        for feature in feature_points.values():
            polyline = np.rint(feature).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [polyline], True, (220, 40, 230), 2, cv2.LINE_AA)
        cv2.line(
            canvas,
            tuple(np.rint(projected_axis[0]).astype(int)),
            tuple(np.rint(projected_axis[1]).astype(int)),
            (30, 240, 245),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            (
                f"Step6 vs candidate {index:03d} | cyan=Step6 contour "
                "green=candidate magenta=named rings yellow=axis"
            ),
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        diagnostic_path = diagnostic_dir / f"{index:03d}_candidate_overlay.png"
        cv2.imwrite(str(diagnostic_path), canvas)

        feature_medians = [
            float(record["edge_residual"]["median_fraction_object_height"])
            for record in feature_records
            if record["edge_residual"]["median_fraction_object_height"] is not None
        ]
        record_pass = bool(
            axis_difference <= 8.0
            and silhouette_residual["median_fraction_object_height"] is not None
            and float(silhouette_residual["median_fraction_object_height"]) <= 0.055
            and len(feature_medians) >= 3
            and float(np.median(feature_medians)) <= 0.055
        )
        hard_gate_eligible = sweep in RELIABLE_CAPTURE_SWEEPS
        records.append(
            {
                "selected_index": index,
                "filename": filename,
                "accepted": record_pass if hard_gate_eligible else None,
                "hard_gate_eligible": hard_gate_eligible,
                "evaluation_status": (
                    "reliable_sweep_crosscheck"
                    if hard_gate_eligible
                    else "evaluation_only_top_sweep_axis_evidence_unreliable"
                ),
                "step6_status": historical["status"],
                "coordinate_path": "exact_SIMPLE_RADIAL",
                "sweep": sweep,
                "projected_axis_angle_degrees": projected_axis_angle,
                "step6_principal_axis_angle_degrees": (
                    analysis.geometry.principal_axis_angle_deg
                ),
                "axis_angle_difference_degrees": axis_difference,
                "candidate_silhouette_to_step6_contour": silhouette_residual,
                "named_feature_edge_residuals": feature_records,
                "ellipse_comparison": ellipse_comparison,
                "diagnostic": _relative(diagnostic_path, project_root),
                "diagnostic_sha256": sha256_file(diagnostic_path),
            }
        )

    eligible_records = [
        record for record in records if record.get("hard_gate_eligible") is True
    ]
    accepted = bool(
        eligible_records
        and all(record.get("accepted") is True for record in eligible_records)
    )
    return {
        "accepted": accepted,
        "source": _relative(summary_path, project_root),
        "sha256": sha256_file(summary_path),
        "method": (
            "candidate_bound_exact_camera_projection_against_rerun_frozen_"
            "Step6_contour_Canny_PCA_and_reliable_ellipse"
        ),
        "candidate_binding": {
            "blend_sha256": blend_hash,
            "plan1_candidate_sha256": plan1_candidate_sha256,
            "profiles_sha256": profiles_hash,
        },
        "thresholds": {
            "axis_angle_difference_degrees": 8.0,
            "silhouette_contour_median_fraction_object_height": 0.055,
            "median_named_feature_edge_fraction_object_height": 0.055,
        },
        "records": records,
        "diagnostics": [
            record["diagnostic"] for record in records if "diagnostic" in record
        ],
        "interpretation": (
            "Step 6 remains an independent corroborating cross-check; rejected "
            "Step-6 ellipses stay unavailable and never override reviewed masks"
        ),
    }


def reopen_registered_report(path: Path) -> dict[str, Any]:
    result = validate_registered_view_coverage_report(_read_json(path))
    return result


def reopen_surface_report(path: Path) -> dict[str, Any]:
    result = validate_surface_evidence_coverage_report(_read_json(path))
    return result


def run_geometry_audit(project_root: Path, v2_root: Path) -> dict[str, Any]:
    """Run canonical continuity plus every eligible registered noncanonical view."""

    root = Path(project_root).resolve()
    v2 = Path(v2_root).resolve()
    reports = v2 / "reports"
    fit_path = reports / "final_cv_fit.json"
    profiles_path = reports / "final_profiles.json"
    review_path = reports / "cv_visual_review.json"
    base_visual_review_path = reports / "base_geometry_visual_review.json"
    blender_report_path = reports / "base_geometry_blender_report.json"
    blend_path = root / ACCEPTED_BLEND
    for path in (fit_path, profiles_path, review_path, blender_report_path, blend_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing geometry-audit input: {path}")

    fit = _read_json(fit_path)
    profiles_payload = _read_json(profiles_path)
    review = _read_json(review_path)
    blender_report = _read_json(blender_report_path)
    profiles_hash = sha256_file(profiles_path)
    blend_hash = sha256_file(blend_path)
    validate_accepted_plan1_inputs(
        fit, profiles_payload, profiles_hash, blender_report, blend_hash, review
    )
    base_visual_review = (
        _read_json(base_visual_review_path)
        if base_visual_review_path.is_file()
        else {"accepted": False}
    )
    base_visual_validation = validate_base_visual_review(
        base_visual_review,
        root,
        blend_hash,
        str(fit["candidate_sha256"]),
    )

    normalization = fit["fit_summary"]["pose_calibration"][
        "camera_center_normalization"
    ]
    source_cameras = load_step13_cameras(root)
    cameras, camera_provenance = derived_audit_cameras(
        source_cameras, normalization
    )
    profile_objects = _profiles(profiles_payload)
    alignments = {
        name: _alignment(record) for name, record in fit["sweep_alignments"].items()
    }
    mask_sources, mask_manifest_hash = _mask_sources(root)
    reviewed_shape_envelopes = _reviewed_mask_shape_envelopes(root)
    canonical_indices = {int(view["selected_index"]) for view in fit["views"]}
    eligible: list[CameraReference] = []
    for index, camera in sorted(cameras.items()):
        if index in canonical_indices or index not in mask_sources:
            continue
        try:
            sweep = capture_sweep(index)
        except ValueError:
            # These registered cameras have no accepted V2 object pose and are
            # intentionally retained only as unmodelled-pose evidence.
            continue
        if sweep in RELIABLE_CAPTURE_SWEEPS:
            eligible.append(camera)
    if not eligible:
        raise ValueError("registered-view audit has no eligible noncanonical views")

    registered_diagnostic_dir = v2 / "diagnostics" / "21_registered_view_coverage"
    registered_diagnostic_dir.mkdir(parents=True, exist_ok=True)
    superseded_diagnostics = [
        _relative(path, root)
        for path in sorted(registered_diagnostic_dir.glob("*_overlay.png"))
    ]
    for path in sorted(registered_diagnostic_dir.glob("*_overlay.png")):
        path.unlink()

    view_records: list[dict[str, Any]] = []
    render_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for camera in eligible:
        mask_source = mask_sources[camera.selected_index]
        mask_path = root / mask_source["mask_path"]
        mask, mask_quality = _load_resized_mask(mask_path)
        if mask_source["mask_kind"] == "frozen_reconstruction_cnn_mask":
            mask_quality = assess_unreviewed_mask_shape(
                mask_quality, reviewed_shape_envelopes[capture_sweep(camera.selected_index)]
            )
        if (
            mask_source["mask_kind"] == "frozen_reconstruction_cnn_mask"
            and float(mask_quality.get("largest_component_share", 0.0)) < 0.97
            and "disconnected_foreground_contamination"
            not in mask_quality["failure_reasons"]
        ):
            mask_quality["failure_reasons"].append(
                "disconnected_foreground_contamination"
            )
            mask_quality["valid"] = False
        resized_camera = _resize_camera(camera, (mask.shape[1], mask.shape[0]))
        sweep = capture_sweep(camera.selected_index)
        projection_valid = sweep in alignments
        if projection_valid:
            try:
                predicted = render_independent_assembly_silhouette(
                    profile_objects,
                    resized_camera,
                    alignments[sweep],
                    resized_camera.image_size,
                )
                projection_valid = bool(np.count_nonzero(predicted))
            except (ValueError, RuntimeError, cv2.error):
                predicted = np.zeros_like(mask)
                projection_valid = False
        else:
            predicted = np.zeros_like(mask)
        source_coverage = float(np.count_nonzero(mask)) / mask.size
        predicted_coverage = float(np.count_nonzero(predicted)) / predicted.size
        iou = _iou(mask, predicted) if projection_valid else 0.0
        camera_consistency = silhouette_camera_consistency(mask, predicted)
        if projection_valid:
            try:
                projected_axis = _project_local_points(
                    resized_camera,
                    alignments[sweep],
                    np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
                )
                independent_camera_consistency = projected_axis_mask_consistency(
                    mask, projected_axis
                )
            except (ValueError, RuntimeError, cv2.error):
                independent_camera_consistency = {
                    "passed": False,
                    "failure": "exact local +Z axis projection failed",
                }
        else:
            independent_camera_consistency = {
                "passed": False,
                "failure": "exact assembly projection failed",
            }
        counterfactual_explains = bool(
            camera_consistency["translation_only_explains_gate_failure"]
            or camera_consistency["rigid_alignment_explains_gate_failure"]
        )
        independent_axis_failure = not bool(
            independent_camera_consistency.get("passed", False)
        )
        classification = classify_registered_view(
            iou,
            source_coverage,
            predicted_coverage,
            mask_valid=bool(mask_quality["valid"]),
            projection_valid=projection_valid,
            camera_failure_explained=(counterfactual_explains or independent_axis_failure),
        )
        record = {
            "selected_index": camera.selected_index,
            "filename": camera.filename,
            "sweep": sweep,
            "camera_view_group": sweep,
            "mask_source": _relative(mask_path, root),
            "mask_sha256": mask_source["mask_sha256"],
            "mask_kind": mask_source["mask_kind"],
            "coordinate_path": "SIMPLE_RADIAL",
            "source_quality_condition": mask_source["source_quality_condition"],
            "mask_quality": mask_quality,
            "projection_valid": projection_valid,
            "camera_consistency": camera_consistency,
            "independent_camera_consistency": independent_camera_consistency,
            "independent_camera_failure_reason": (
                "projected local +Z axis disagrees with observed vessel axis"
                if projection_valid and independent_axis_failure
                else ("exact camera projection failed" if not projection_valid else None)
            ),
            "source_foreground_coverage": source_coverage,
            "predicted_foreground_coverage": predicted_coverage,
            **classification,
        }
        view_records.append(record)
        render_cache[camera.selected_index] = (mask, predicted)

    worst: list[dict[str, Any]] = []
    for sweep in RELIABLE_CAPTURE_SWEEPS:
        sweep_records = [record for record in view_records if record["sweep"] == sweep]
        mismatch_records = sorted(
            (
                record
                for record in sweep_records
                if record["classification"] == "model_mismatch"
            ),
            key=lambda record: (float(record["silhouette_iou"]), record["selected_index"]),
        )[:3]
        ranked = list(mismatch_records)
        for record in sorted(
            sweep_records,
            key=lambda item: (float(item["silhouette_iou"]), item["selected_index"]),
        ):
            if record not in ranked:
                ranked.append(record)
            if len(ranked) >= 5:
                break
        for record in ranked:
            camera = cameras[int(record["selected_index"])]
            mask, predicted = render_cache[camera.selected_index]
            diagnostic = _write_view_diagnostic(
                root, v2, camera, mask, predicted, record
            )
            record["diagnostic"] = _relative(diagnostic, root)
            worst.append(
                {
                    "selected_index": camera.selected_index,
                    "filename": camera.filename,
                    "classification": record["classification"],
                    "silhouette_iou": record["silhouette_iou"],
                    "diagnostic": record["diagnostic"],
                    "review_status": "opened_by_pipeline_owner_required",
                }
            )

    registered_visual_review_path = (
        reports / "registered_view_coverage_visual_review.json"
    )
    registered_visual_review = (
        _read_json(registered_visual_review_path)
        if registered_visual_review_path.is_file()
        else {"accepted": False}
    )
    registered_visual_validation = validate_registered_visual_review(
        registered_visual_review,
        root,
        blend_hash,
        worst,
    )
    for record in worst:
        record["review_status"] = (
            "classification_confirmed"
            if registered_visual_validation["passed"]
            else "opened_by_pipeline_owner_required"
        )

    step6_crosscheck = _step6_crosscheck(
        root,
        v2,
        blend_hash=blend_hash,
        plan1_candidate_sha256=str(fit["candidate_sha256"]),
        profiles_hash=profiles_hash,
        profiles=profile_objects,
        cameras=cameras,
        alignments=alignments,
    )

    registered = {
        "schema_version": 1,
        "accepted": True,
        "blend_path": _relative(blend_path, root),
        "blend_sha256": blend_hash,
        "geometry_representation": (
            "blender_profile_roundtrip_equivalent_to_accepted_named_profiles"
        ),
        "final_cv_fit_candidate_sha256": fit["candidate_sha256"],
        "selection_rule": (
            "all Step13-registered noncanonical views in accepted reliable capture "
            "sweeps with existing reviewed-or-frozen masks"
        ),
        "superseded_diagnostics_removed_before_run": superseded_diagnostics,
        "unreviewed_mask_shape_rule": {
            "reviewed_envelopes": reviewed_shape_envelopes,
            "bbox_width_margin": MASK_BBOX_WIDTH_MARGIN,
            "p95_row_width_margin": MASK_P95_ROW_WIDTH_MARGIN,
            "solidity_margin": MASK_SOLIDITY_MARGIN,
            "band_width_margin": MASK_BAND_WIDTH_MARGIN,
            "minimum_anomaly_count": 2,
        },
        "canonical_translation_refinements_applied": False,
        "view_count_expected": len(eligible),
        "provenance": {
            "mask_manifest": {
                "path": REGISTERED_MASK_MANIFEST.as_posix(),
                "sha256": mask_manifest_hash,
            },
            "camera_normalization": {
                "path": _relative(fit_path, root),
                "sha256": sha256_file(fit_path),
                "record": camera_provenance,
            },
            "camera_alignment": {
                "path": _relative(fit_path, root),
                "sha256": sha256_file(fit_path),
                "sweep_alignments": fit["sweep_alignments"],
            },
        },
        "views": view_records,
        "inspected_worst_views": worst,
        "visual_review": {
            "path": _relative(registered_visual_review_path, root),
            "sha256": (
                sha256_file(registered_visual_review_path)
                if registered_visual_review_path.is_file()
                else None
            ),
            "accepted": bool(registered_visual_validation["passed"]),
            "validation": registered_visual_validation,
        },
        "step6_crosscheck": step6_crosscheck,
    }
    registered_validation = validate_registered_view_coverage_report(registered)
    registered["accepted"] = bool(
        registered_validation["passed"]
        and registered_visual_validation["passed"]
    )
    registered["validation"] = registered_validation
    registered_path = reports / "registered_view_coverage_report.json"
    write_json_atomic(registered_path, registered, v2)

    surface = _surface_evidence_report(
        root,
        v2,
        blend_path,
        blend_hash,
        profiles_payload,
        profiles_hash,
        cameras,
        alignments,
    )
    surface_path = reports / "surface_evidence_coverage.json"
    write_json_atomic(surface_path, surface, v2)

    aggregate = fit["aggregate_metrics"]
    canonical_pass = all(
        (
            float(aggregate["median_silhouette_iou"])
            >= METRIC_THRESHOLDS["median_silhouette_iou"],
            float(aggregate["minimum_reliable_silhouette_iou"])
            >= METRIC_THRESHOLDS["minimum_reliable_silhouette_iou"],
            float(aggregate["median_landmark_error_fraction"])
            <= METRIC_THRESHOLDS["median_landmark_error_fraction"],
            float(aggregate["p95_landmark_error_fraction"])
            <= METRIC_THRESHOLDS["p95_landmark_error_fraction"],
        )
    )
    component_review = dict(base_visual_validation["components"])
    base_accepted = bool(
        canonical_pass
        and base_visual_validation["passed"]
        and registered["accepted"]
        and surface["accepted"]
        and step6_crosscheck["accepted"]
    )
    base_report = {
        "schema_version": 1,
        "stage": "geometry-validate",
        "accepted": base_accepted,
        "blend_path": _relative(blend_path, root),
        "blend_sha256": blend_hash,
        "plan1_candidate_sha256": fit["candidate_sha256"],
        "profiles": {
            "path": _relative(profiles_path, root),
            "sha256": profiles_hash,
            "import_max_abs_delta": 0.0,
        },
        "canonical_camera_set": sorted(canonical_indices),
        "camera_metric_contract": "raw_source_exact_SIMPLE_RADIAL",
        "camera_display_contract": "Blender_pinhole_display_only_not_metric",
        "canonical_metrics": aggregate,
        "metric_thresholds": METRIC_THRESHOLDS,
        "component_visual_review": component_review,
        "base_geometry_visual_review": {
            "path": _relative(base_visual_review_path, root),
            "sha256": (
                sha256_file(base_visual_review_path)
                if base_visual_review_path.is_file()
                else None
            ),
            "accepted": bool(base_visual_validation["passed"]),
            "validation": base_visual_validation,
        },
        "registered_view_coverage_report": {
            "path": _relative(registered_path, root),
            "sha256": sha256_file(registered_path),
            "accepted": registered["accepted"],
        },
        "surface_evidence_coverage": {
            "path": _relative(surface_path, root),
            "sha256": sha256_file(surface_path),
            "accepted": surface["accepted"],
        },
        "step6_crosscheck": step6_crosscheck,
        "diagnostics": [record["diagnostic"] for record in worst],
        "ornament_started": False,
        "failure_reasons": [
            reason
            for condition, reason in (
                (canonical_pass, "canonical Gate-B metrics failed"),
                (
                    bool(base_visual_validation["passed"]),
                    "candidate-bound base visual veto failed",
                ),
                (registered["accepted"], "registered-view coverage veto failed"),
                (surface["accepted"], "surface-evidence manifest failed"),
                (step6_crosscheck["accepted"], "candidate-bound Step-6 cross-check failed"),
            )
            if not condition
        ],
    }
    base_path = reports / "base_geometry_report.json"
    write_json_atomic(base_path, base_report, v2)

    return {
        "stage": "geometry-audit",
        "accepted": base_accepted,
        "blend_sha256": blend_hash,
        "canonical_metrics": aggregate,
        "registered_view_count": len(view_records),
        "usable_model_mismatch_count": registered_validation[
            "usable_model_mismatch_count"
        ],
        "registered_view_report": _relative(registered_path, root),
        "surface_evidence_report": _relative(surface_path, root),
        "base_geometry_report": _relative(base_path, root),
    }


__all__ = [
    "MINIMUM_IOU_THRESHOLD",
    "REQUIRED_BASE_VISUAL_VIEWS",
    "REQUIRED_SURFACE_COMPONENTS",
    "assess_unreviewed_mask_shape",
    "classify_registered_view",
    "derived_audit_cameras",
    "reopen_registered_report",
    "reopen_surface_report",
    "run_geometry_audit",
    "validate_accepted_plan1_inputs",
    "validate_base_visual_review",
    "validate_registered_visual_review",
]
