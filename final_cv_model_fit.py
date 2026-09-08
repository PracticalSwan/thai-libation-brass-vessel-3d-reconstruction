"""Fit and gate the final V2 vessel from frozen multi-view CV evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pycolmap
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

from analysis_common import load_selected_manifest
from final_model_io import (
    V2_RELATIVE_ROOT,
    ensure_under_v2_root,
    sha256_file,
    write_json_atomic,
)


SELECTION_MANIFEST = Path("preprocessing/reports/selection_manifest.csv")
SELECTED_IMAGES = Path("preprocessing/pycolmap_input/images")
STEP13_BEST_MODEL = Path("reconstruction/external_learned_recovery/best")
STEP13_SUMMARY = Path(
    "reconstruction/external_learned_recovery/reports/step13_summary.json"
)
LANDMARK_EVIDENCE = Path(
    "reconstruction/reference_assisted_v2/evidence/annotations/landmarks.json"
)

METRIC_THRESHOLDS = {
    "median_silhouette_iou": 0.90,
    "minimum_reliable_silhouette_iou": 0.84,
    "median_landmark_error_fraction": 0.020,
    "p95_landmark_error_fraction": 0.040,
}
VISUAL_REVIEW_COMPONENTS = ("bowl", "globe", "neck", "lid", "finial")
PROFILE_RANGES = {
    # Local +Z runs from the bottom of the complete set (0) to the finial (1).
    # Image-space annotation bands are measured top-to-bottom, so invert those
    # bands before exposing the profiles to Blender.
    "pedestal": (0.00, 0.22),
    "bowl_outer": (0.21, 0.30),
    "globe": (0.30, 0.55),
    "shoulder": (0.50, 0.58),
    "neck_outer": (0.55, 0.81),
    "lid": (0.81, 0.94),
    "finial": (0.93, 1.00),
}
REQUIRED_PROFILES = (
    "pedestal",
    "bowl_outer",
    "bowl_inner",
    "globe",
    "shoulder",
    "neck_outer",
    "neck_inner",
    "lid",
    "finial",
)
OUTER_SILHOUETTE_PROFILES = (
    "pedestal",
    "bowl_outer",
    "globe",
    "shoulder",
    "neck_outer",
    "lid",
    "finial",
)
CAPTURE_SWEEPS = {
    "side_003_072": (3, 72),
    "low_090_142": (90, 142),
    "elevated_148_200": (148, 200),
    "top_206_255": (206, 255),
}
RELIABLE_CAPTURE_SWEEPS = (
    "side_003_072",
    "low_090_142",
    "elevated_148_200",
)
TOP_CAPTURE_SWEEP = "top_206_255"
GATE_CAMERA_SOURCES = (
    "step13",
    "step13_v2_sweep_gauge_normalized",
    "step13_v2_bounded_translation_refined",
)


@dataclass(frozen=True)
class CameraReference:
    selected_index: int
    filename: str
    image_id: int
    camera_model: str
    camera_params: tuple[float, ...]
    cam_from_world_rotation_xyzw: tuple[float, float, float, float]
    cam_from_world_translation: tuple[float, float, float]
    image_size: tuple[int, int]


@dataclass(frozen=True)
class ModelAlignment:
    scale: float
    rotation_matrix: tuple[tuple[float, float, float], ...]
    translation: tuple[float, float, float]


@dataclass(frozen=True)
class FittedComponentProfile:
    name: str
    sections: tuple[tuple[float, float], ...]
    measurement_status: str
    source_views: tuple[int, ...] = ()


@dataclass(frozen=True)
class FitObservation:
    selected_index: int
    filename: str
    view_category: str
    mask: np.ndarray
    axis_top_xy: tuple[float, float]
    axis_bottom_xy: tuple[float, float]
    landmarks: tuple[Mapping[str, float | str], ...]
    source_path: Path
    reviewed_mask_path: Path


@dataclass(frozen=True)
class LandmarkFitRecord:
    name: str
    observed_x: float
    observed_y: float
    predicted_x: float
    predicted_y: float
    error_fraction: float
    confidence: str = "medium"
    gate_included: bool = True


@dataclass(frozen=True)
class ViewFitMetrics:
    selected_index: int
    filename: str
    view_category: str
    silhouette_iou: float
    landmark_median_error_fraction: float
    landmark_max_error_fraction: float
    component_ious: Mapping[str, float]
    camera_source: str
    fit_included: bool
    inclusion_reason: str
    source_path: Path | None = None
    reviewed_mask_path: Path | None = None
    landmark_records: tuple[LandmarkFitRecord, ...] = ()
    predicted_mask: np.ndarray | None = field(default=None, repr=False)


@dataclass(frozen=True)
class FitResult:
    alignment: ModelAlignment
    profiles: tuple[FittedComponentProfile, ...]
    metrics: tuple[ViewFitMetrics, ...]
    metrics_passed: bool
    visual_component_review: Mapping[str, str]
    accepted: bool
    aggregate_metrics: Mapping[str, float]
    fit_summary: Mapping[str, Any] = field(default_factory=dict)
    sweep_alignments: Mapping[str, ModelAlignment] = field(default_factory=dict)


def _tuple(value: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(item) for item in value)


def _rigid3d(camera: CameraReference) -> pycolmap.Rigid3d:
    return pycolmap.Rigid3d(
        pycolmap.Rotation3d(
            xyzw=np.asarray(camera.cam_from_world_rotation_xyzw, dtype=np.float64)
        ),
        np.asarray(camera.cam_from_world_translation, dtype=np.float64),
    )


def _pycolmap_camera(camera: CameraReference) -> pycolmap.Camera:
    width, height = camera.image_size
    return pycolmap.Camera(
        model=camera.camera_model,
        width=width,
        height=height,
        params=np.asarray(camera.camera_params, dtype=np.float64),
    )


def load_step13_cameras(
    project_root: Path,
    *,
    summary_path: Path | None = None,
    reconstruction: Any | None = None,
) -> tuple[CameraReference, ...]:
    """Load only cameras that agree with the authoritative Step 13 report."""

    root = project_root.resolve()
    selected = load_selected_manifest(root / SELECTION_MANIFEST)
    selected_by_filename = {record.filename: record for record in selected}
    if len(selected_by_filename) != len(selected):
        raise ValueError("selected manifest contains duplicate filenames")

    with (summary_path or root / STEP13_SUMMARY).open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    try:
        report = summary["attempt_result"]["attempt"]["best_model"]
        report_names = set(report["registered_image_names"])
        report_registered_count = int(report["registered_images"])
        report_camera_count = int(report["camera_count"])
        report_camera_model = str(report["camera_model"])
        report_camera_params = _tuple(report["camera_params"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("incomplete authoritative Step 13 report") from error

    model = reconstruction or pycolmap.Reconstruction(str(root / STEP13_BEST_MODEL))
    image_ids = tuple(model.reg_image_ids())
    cameras: list[CameraReference] = []
    model_names: set[str] = set()
    if len(model.cameras) != report_camera_count:
        raise ValueError(
            "camera count disagrees with the authoritative Step 13 report"
        )
    if len(image_ids) != report_registered_count or len(report_names) != len(image_ids):
        raise ValueError(
            "registered image count disagrees with the authoritative Step 13 report"
        )

    for image_id in image_ids:
        image = model.image(image_id)
        name = str(image.name)
        model_names.add(name)
        selected = selected_by_filename.get(name)
        if selected is None:
            raise ValueError(f"Step 13 image is absent from selected manifest: {name}")
        camera = image.camera
        camera_model = str(camera.model_name)
        camera_params = _tuple(camera.params)
        image_size = (int(camera.width), int(camera.height))
        if camera_model != report_camera_model or camera_params != report_camera_params:
            raise ValueError(
                "camera intrinsics disagree with the authoritative Step 13 report"
            )
        if image_size != (selected.width, selected.height):
            raise ValueError(f"Step 13 image size disagrees with manifest: {name}")
        pose = image.cam_from_world()
        cameras.append(
            CameraReference(
                selected_index=selected.index,
                filename=name,
                image_id=int(image.image_id),
                camera_model=camera_model,
                camera_params=camera_params,
                cam_from_world_rotation_xyzw=_tuple(pose.rotation.quat),
                cam_from_world_translation=_tuple(pose.translation),
                image_size=image_size,
            )
        )

    if model_names != report_names:
        raise ValueError(
            "registered filenames disagree with the authoritative Step 13 report"
        )
    cameras.sort(key=lambda item: item.selected_index)
    if len({item.selected_index for item in cameras}) != len(cameras):
        raise ValueError("duplicate selected index in Step 13 cameras")
    return tuple(cameras)


def project_world_points(
    camera: CameraReference,
    points: np.ndarray,
    *,
    check_cheirality: bool = True,
) -> np.ndarray | None:
    """Project world points through pyCOLMAP, including its distortion model."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        raise ValueError("world points must have shape [n, 3]")
    camera_points = _rigid3d(camera) * values
    projected = _pycolmap_camera(camera).img_from_cam(
        camera_points, check_cheirality=check_cheirality
    )
    projected = np.asarray(projected, dtype=np.float64)
    if projected.size == 0 or not np.all(np.isfinite(projected)):
        return None
    return projected


def project_world_point(
    camera: CameraReference,
    point: np.ndarray,
) -> tuple[float, float] | None:
    """Project one world point and return Python pixel coordinates."""

    values = np.asarray(point, dtype=np.float64).reshape(1, 3)
    try:
        projected = project_world_points(camera, values, check_cheirality=True)
    except RuntimeError:
        return None
    if projected is None:
        return None
    return float(projected[0, 0]), float(projected[0, 1])


def _quaternion_matrix(
    quaternion: Sequence[float],
) -> np.ndarray:
    return np.asarray(
        pycolmap.Rotation3d(
            xyzw=np.asarray(quaternion, dtype=np.float64)
        ).matrix(),
        dtype=np.float64,
    )


def _world_ray(
    camera: CameraReference,
    image_xy: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    camera_ray = _pycolmap_camera(camera).cam_ray_from_img(
        np.asarray([[float(image_xy[0]), float(image_xy[1])]], dtype=np.float64)
    )[0]
    cam_from_world = _quaternion_matrix(camera.cam_from_world_rotation_xyzw)
    world_from_camera = cam_from_world.T
    direction = world_from_camera @ camera_ray
    length = float(np.linalg.norm(direction))
    if not math.isfinite(length) or length <= 1e-12:
        raise ValueError("camera produced a degenerate image ray")
    origin = world_from_camera @ (
        -np.asarray(camera.cam_from_world_translation, dtype=np.float64)
    )
    return origin, direction / length


def _triangulate_rays(
    rays: Sequence[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    if len(rays) < 2:
        raise ValueError("at least two rays are required for triangulation")
    a_matrix = np.zeros((3, 3), dtype=np.float64)
    b_vector = np.zeros(3, dtype=np.float64)
    for origin, direction in rays:
        perpendicular = np.identity(3) - np.outer(direction, direction)
        a_matrix += perpendicular
        b_vector += perpendicular @ origin
    if np.linalg.matrix_rank(a_matrix, tol=1e-9) < 3:
        raise ValueError("rays do not span a 3D point")
    return np.linalg.solve(a_matrix, b_vector)


def _model_to_world(
    alignment: ModelAlignment,
    local_points: np.ndarray,
) -> np.ndarray:
    rotation = np.asarray(alignment.rotation_matrix, dtype=np.float64)
    translation = np.asarray(alignment.translation, dtype=np.float64)
    return alignment.scale * (np.asarray(local_points) @ rotation.T) + translation


AXIAL_LANDMARK_KEYS = (
    "finial_top",
    "finial_bottom",
    "lid_max",
    "lid_lower",
    "neck_top",
    "neck_base",
    "globe_top_transition",
    "globe_max",
    "globe_bottom_transition",
    "bowl_rim",
    "bowl_bottom_transition",
    "pedestal_waist",
    "foot",
)


def _confidence_rank(value: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(value, 0)


def _axial_landmark_center(
    observation: FitObservation,
    key: str,
) -> tuple[tuple[float, float], str] | None:
    """Return the image-space center of one repeatable axial landmark level."""

    by_name = {str(item["name"]): item for item in observation.landmarks}
    direct = by_name.get(key)
    if direct is not None:
        return (
            (float(direct["x"]), float(direct["y"])),
            str(direct.get("confidence", "medium")),
        )
    left = by_name.get(f"{key}_left")
    right = by_name.get(f"{key}_right")
    if left is None or right is None:
        return None
    confidence = min(
        (str(left.get("confidence", "medium")), str(right.get("confidence", "medium"))),
        key=_confidence_rank,
    )
    return (
        (
            (float(left["x"]) + float(right["x"])) * 0.5,
            (float(left["y"]) + float(right["y"])) * 0.5,
        ),
        confidence,
    )


def _robust_triangulate_rays(
    rays: Sequence[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Triangulate a landmark while rejecting gross inconsistent image rays."""

    active = list(rays)
    if len(active) < 2:
        raise ValueError("at least two rays are required for robust triangulation")
    for _ in range(4):
        point = _triangulate_rays(active)
        if len(active) <= 4:
            return point
        distances = np.asarray(
            [
                np.linalg.norm(
                    (np.identity(3) - np.outer(direction, direction))
                    @ (point - origin)
                )
                for origin, direction in active
            ],
            dtype=np.float64,
        )
        median = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median)))
        cutoff = median + max(3.5 * 1.4826 * mad, max(median, 1e-9) * 0.35)
        keep = [ray for ray, distance in zip(active, distances, strict=True) if distance <= cutoff]
        if len(keep) < 4 or len(keep) == len(active):
            return point
        active = keep
    return _triangulate_rays(active)


def _common_landmark_levels(
    observations: Sequence[FitObservation],
) -> dict[str, float]:
    """Estimate one shared normalized +Z level per named physical landmark."""

    levels: dict[str, float] = {"axis_top": 1.0, "axis_bottom": 0.0}
    for key in AXIAL_LANDMARK_KEYS:
        values: list[float] = []
        for observation in observations:
            if observation.view_category == "top_down_rim":
                continue
            centered = _axial_landmark_center(observation, key)
            if centered is None or centered[1] == "low":
                continue
            values.append(
                _observed_z_from_landmark(
                    centered[0], observation.axis_top_xy, observation.axis_bottom_xy
                )
            )
        if values:
            levels[key] = float(np.median(np.asarray(values, dtype=np.float64)))
    return levels


def _alignment_from_axis(translation: np.ndarray, axis: np.ndarray) -> ModelAlignment:
    scale = float(np.linalg.norm(axis))
    if not math.isfinite(scale) or scale <= 1e-9:
        raise ValueError("triangulated model axis is degenerate")
    unit_axis = axis / scale
    helper = np.asarray((1.0, 0.0, 0.0))
    if abs(float(np.dot(helper, unit_axis))) > 0.9:
        helper = np.asarray((0.0, 1.0, 0.0))
    local_y = np.cross(unit_axis, helper)
    local_y /= np.linalg.norm(local_y)
    local_x = np.cross(local_y, unit_axis)
    rotation = np.column_stack((local_x, local_y, unit_axis))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 1] *= -1.0
    return ModelAlignment(
        scale=scale,
        rotation_matrix=tuple(tuple(float(value) for value in row) for row in rotation),
        translation=tuple(float(value) for value in translation),
    )


def _unpack_shared_scale_sweep_poses(
    parameters: np.ndarray,
    names: Sequence[str],
) -> dict[str, ModelAlignment]:
    """Build sweep poses with one exact physical scale for all reliable sweeps."""

    pose_parameter_count = len(names) * 6
    if len(parameters) < pose_parameter_count + 2:
        raise ValueError("shared-scale sweep parameters are incomplete")
    shared_scale = float(math.exp(parameters[pose_parameter_count]))
    top_scale = float(math.exp(parameters[pose_parameter_count + 1]))
    poses: dict[str, ModelAlignment] = {}
    for index, name in enumerate(names):
        start = index * 6
        translation = np.asarray(parameters[start:start + 3], dtype=np.float64)
        direction = np.asarray(parameters[start + 3:start + 6], dtype=np.float64)
        direction_norm = float(np.linalg.norm(direction))
        if not math.isfinite(direction_norm) or direction_norm <= 1e-9:
            raise ValueError(f"degenerate sweep axis direction for {name}")
        scale = top_scale if name == TOP_CAPTURE_SWEEP else shared_scale
        poses[name] = _alignment_from_axis(
            translation,
            direction / direction_norm * scale,
        )
    return poses


def _camera_center_world(camera: CameraReference) -> np.ndarray:
    """Return the world-space optical center for one cam_from_world reference."""

    rotation = _quaternion_matrix(camera.cam_from_world_rotation_xyzw)
    translation = np.asarray(camera.cam_from_world_translation, dtype=np.float64)
    return -(rotation.T @ translation)


def _scale_camera_center_about_anchor(
    camera: CameraReference,
    anchor_world: Sequence[float],
    factor: float,
) -> CameraReference:
    """Normalize recovered sweep scale without mutating Step-13 source cameras.

    Scaling every camera center about the independently fitted object anchor by
    ``shared_scale / recovered_sweep_scale`` is projectively equivalent to the
    original sweep-specific object scale. It therefore preserves the recovered
    source rays while expressing the same local object with one V2 physical scale.
    """

    value = float(factor)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("camera-center normalization factor must be positive")
    anchor = np.asarray(anchor_world, dtype=np.float64)
    if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
        raise ValueError("camera-center normalization anchor must be a finite 3-vector")
    rotation = _quaternion_matrix(camera.cam_from_world_rotation_xyzw)
    center = _camera_center_world(camera)
    normalized_center = anchor + value * (center - anchor)
    normalized_translation = -(rotation @ normalized_center)
    return replace(
        camera,
        cam_from_world_translation=tuple(float(item) for item in normalized_translation),
    )


def _apply_camera_center_normalization(
    cameras: Sequence[CameraReference],
    normalization: Mapping[str, Any],
) -> tuple[CameraReference, ...]:
    """Build derived V2 camera references from recorded per-sweep gauge factors."""

    sweep_records = normalization.get("sweeps", {})
    output: list[CameraReference] = []
    for camera in cameras:
        record = sweep_records.get(capture_sweep(camera.selected_index))
        if record is None:
            output.append(camera)
            continue
        output.append(
            _scale_camera_center_about_anchor(
                camera,
                record["anchor_world"],
                float(record["factor"]),
            )
        )
    return tuple(output)


def _refine_alignment_image_space(
    initial: ModelAlignment,
    observations: Sequence[FitObservation],
    cameras: Sequence[CameraReference],
    common_levels: Mapping[str, float],
) -> ModelAlignment:
    """Bundle-calibrate object translation/axis against fixed Step 13 source cameras."""

    camera_by_index = {camera.selected_index: camera for camera in cameras}
    rotation = np.asarray(initial.rotation_matrix, dtype=np.float64)
    initial_axis = rotation[:, 2] * float(initial.scale)
    initial_parameters = np.concatenate(
        (np.asarray(initial.translation, dtype=np.float64), initial_axis)
    )

    observations_for_fit = [
        observation
        for observation in observations
        if observation.view_category != "top_down_rim"
    ]

    def residual(parameters: np.ndarray) -> np.ndarray:
        translation = parameters[:3]
        axis = parameters[3:]
        if float(np.linalg.norm(axis)) <= 1e-9:
            return np.full(2 * len(observations_for_fit), 1e3, dtype=np.float64)
        values: list[float] = []
        for observation in observations_for_fit:
            camera = camera_by_index[observation.selected_index]
            normalizer = _object_image_height(observation.mask)
            for key in AXIAL_LANDMARK_KEYS:
                centered = _axial_landmark_center(observation, key)
                level = common_levels.get(key)
                if centered is None or level is None or centered[1] == "low":
                    continue
                world = translation + axis * float(level)
                projected = project_world_point(camera, world)
                if projected is None:
                    values.extend((2.0, 2.0))
                    continue
                confidence_weight = 1.25 if centered[1] == "high" else 1.0
                values.extend(
                    (
                        confidence_weight * (float(projected[0]) - centered[0][0]) / normalizer,
                        confidence_weight * (float(projected[1]) - centered[0][1]) / normalizer,
                    )
                )
        return np.asarray(values, dtype=np.float64)

    initial_residual = residual(initial_parameters)
    if initial_residual.size < 16:
        return initial
    solved = least_squares(
        residual,
        initial_parameters,
        loss="soft_l1",
        f_scale=0.0125,
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
        max_nfev=500,
    )
    if not solved.success and solved.cost >= float(np.dot(initial_residual, initial_residual)) * 0.5:
        return initial
    return _alignment_from_axis(solved.x[:3], solved.x[3:])


def fit_model_alignment(
    observations: Sequence[FitObservation],
    cameras: Sequence[CameraReference],
) -> ModelAlignment:
    """Calibrate a common object pose from repeatable landmarks and fixed Step 13 cameras.

    The capture sweeps frequently crop the pedestal at the image boundary.  That
    boundary remains useful as a per-view visible-axis normalization endpoint, but
    it is not a repeatable physical 3D point and must not be triangulated as one.
    Instead, triangulate repeatable component-center landmarks, assign each a
    shared normalized Z level, and robustly regress the common world-space axis.
    """

    camera_by_index = {camera.selected_index: camera for camera in cameras}
    observation_by_index = {
        observation.selected_index: observation for observation in observations
    }
    if len(camera_by_index) != len(cameras) or len(observation_by_index) != len(
        observations
    ):
        raise ValueError("camera and observation indices must be unique")

    for selected_index, observation in observation_by_index.items():
        camera = camera_by_index.get(selected_index)
        if camera is None:
            raise ValueError(f"missing camera for observation {selected_index}")
        if camera.image_size != observation.mask.shape[::-1]:
            raise ValueError(f"mask/camera size mismatch: {observation.filename}")

    common_levels = _common_landmark_levels(observations)
    fitted_levels: list[float] = []
    fitted_points: list[np.ndarray] = []
    for key in AXIAL_LANDMARK_KEYS:
        rays: list[tuple[np.ndarray, np.ndarray]] = []
        for observation in observations:
            if observation.view_category == "top_down_rim":
                continue
            centered = _axial_landmark_center(observation, key)
            if centered is None or centered[1] == "low":
                continue
            rays.append(_world_ray(camera_by_index[observation.selected_index], centered[0]))
        if len(rays) >= 4 and key in common_levels:
            fitted_points.append(_robust_triangulate_rays(rays))
            fitted_levels.append(common_levels[key])

    if len(fitted_points) >= 4 and np.ptp(np.asarray(fitted_levels)) >= 0.35:
        z_values = np.asarray(fitted_levels, dtype=np.float64)
        points = np.asarray(fitted_points, dtype=np.float64)
        design = np.column_stack((np.ones(len(z_values)), z_values))
        coefficients, *_ = np.linalg.lstsq(design, points, rcond=None)
        initial_translation = coefficients[0]
        initial_axis = coefficients[1]
        rough_scale = max(float(np.linalg.norm(initial_axis)), 1e-6)

        def residual(parameters: np.ndarray) -> np.ndarray:
            translation = parameters[:3]
            axis = parameters[3:]
            predicted = translation[None, :] + z_values[:, None] * axis[None, :]
            return ((predicted - points) / rough_scale).reshape(-1)

        solved = least_squares(
            residual,
            np.concatenate((initial_translation, initial_axis)),
            loss="soft_l1",
            f_scale=0.015,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
            max_nfev=300,
        )
        initial = _alignment_from_axis(solved.x[:3], solved.x[3:])
        return _refine_alignment_image_space(
            initial, observations, cameras, common_levels
        )

    # Synthetic/minimal fixtures may provide only explicit axis endpoints.  Keep
    # this bounded fallback for tests and future sparse evidence, but real V2 data
    # must use the repeatable-landmark calibration above.
    top_rays: list[tuple[np.ndarray, np.ndarray]] = []
    bottom_rays: list[tuple[np.ndarray, np.ndarray]] = []
    for selected_index, observation in observation_by_index.items():
        if observation.view_category == "top_down_rim":
            continue
        camera = camera_by_index[selected_index]
        top_rays.append(_world_ray(camera, observation.axis_top_xy))
        bottom_rays.append(_world_ray(camera, observation.axis_bottom_xy))
    if len(top_rays) < 2:
        raise ValueError("fewer than two reliable axis views")
    top = _triangulate_rays(top_rays)
    bottom = _triangulate_rays(bottom_rays)
    return _alignment_from_axis(bottom, top - bottom)


def capture_sweep(selected_index: int) -> str:
    """Frozen capture intervals, never a fitted per-view grouping."""
    for name, (first, last) in CAPTURE_SWEEPS.items():
        if first <= selected_index <= last:
            return name
    raise ValueError(f"view {selected_index} outside calibrated capture intervals")


def mask_axis_plane(observation: FitObservation, camera: CameraReference) -> tuple[np.ndarray, float, dict]:
    """Backproject a robust silhouette midline to a world plane containing its axis."""
    rows = np.flatnonzero(np.any(observation.mask > 0, axis=1))
    centers = []
    used_rows = []
    for row in rows[::max(1, len(rows)//100)]:
        columns = np.flatnonzero(observation.mask[row] > 0)
        if len(columns) >= 4:
            centers.append((columns[0]+columns[-1])*0.5)
            used_rows.append(float(row))
    if len(centers) < 8:
        raise ValueError("insufficient silhouette axis evidence")
    y = np.asarray(used_rows)
    x = np.asarray(centers)
    center_y = float(np.mean(y))
    design = np.column_stack((y-center_y, np.ones(len(y))))
    seed = np.linalg.lstsq(design, x, rcond=None)[0]
    solution = least_squares(lambda p: design @ p-x, seed, loss="soft_l1", f_scale=1.0)
    endpoints = [(float(solution.x[0]*(value-center_y)+solution.x[1]), float(value))
                 for value in (y.min(), y.max())]
    origin, ray_first = _world_ray(camera, endpoints[0])
    _, ray_second = _world_ray(camera, endpoints[1])
    normal = np.cross(ray_first, ray_second)
    normal /= np.linalg.norm(normal)
    return normal, float(normal @ origin), {
        "selected_index": observation.selected_index,
        "axis_endpoints_xy": endpoints,
        "median_midline_residual_px": float(np.median(np.abs(design @ solution.x-x))),
    }


def audit_capture_axes(observations: Sequence[FitObservation], cameras: Sequence[CameraReference]) -> dict:
    """Label-independent check of whether a common world axis fits all sweeps."""
    by_camera = {camera.selected_index: camera for camera in cameras}
    groups = {}
    for name in CAPTURE_SWEEPS:
        selected = [obs for obs in observations if capture_sweep(obs.selected_index) == name]
        if len(selected) < 3:
            raise ValueError(f"at least three cameras required for sweep {name}")
        planes = [mask_axis_plane(obs, by_camera[obs.selected_index]) for obs in selected]
        normals = np.asarray([plane[0] for plane in planes])
        _, singular, vectors = np.linalg.svd(normals)
        axis = vectors[-1]
        residual = np.degrees(np.arcsin(np.clip(np.abs(normals @ axis), 0, 1)))
        groups[name] = {
            "axis_world": axis.tolist(), "plane_singular_values": singular.tolist(),
            "maximum_axis_plane_error_degrees": float(residual.max()),
            "views": [plane[2] for plane in planes],
        }
    comparisons = []
    names = list(groups)
    for i, first in enumerate(names):
        for second in names[i+1:]:
            dot = abs(float(np.dot(groups[first]["axis_world"], groups[second]["axis_world"])))
            comparisons.append({"first": first, "second": second,
                                "angle_degrees": float(np.degrees(np.arccos(np.clip(dot, 0, 1))))})
    return {"method": "mask_midline_backprojection_planes_fixed_step13_cameras",
            "groups": groups, "between_sweep_angles": comparisons,
            "interpretation": "object pose or recovered frame inconsistency; not proof of physical object motion"}


def _camera_axis_consistency_audit(
    observations: Sequence[FitObservation],
    cameras: Sequence[CameraReference],
    alignments: Mapping[str, ModelAlignment],
) -> dict[str, Any]:
    """Detect dual angle/offset outliers without consulting Gate-B fit metrics."""

    by_camera = {camera.selected_index: camera for camera in cameras}
    groups: dict[str, Any] = {}
    excluded: list[int] = []
    for sweep_name in RELIABLE_CAPTURE_SWEEPS:
        rows: list[dict[str, float | int]] = []
        pose = alignments[sweep_name]
        rotation = np.asarray(pose.rotation_matrix, dtype=np.float64)
        unit_axis = rotation[:, 2]
        translation = np.asarray(pose.translation, dtype=np.float64)
        for obs in observations:
            if capture_sweep(obs.selected_index) != sweep_name:
                continue
            normal, offset, _ = mask_axis_plane(obs, by_camera[obs.selected_index])
            angle = float(
                np.degrees(
                    np.arcsin(np.clip(abs(float(normal @ unit_axis)), 0.0, 1.0))
                )
            )
            normalized_offset = abs(float((normal @ translation - offset) / pose.scale))
            rows.append(
                {
                    "selected_index": obs.selected_index,
                    "axis_angle_error_degrees": angle,
                    "axis_offset_error_fraction": normalized_offset,
                }
            )
        angles = np.asarray(
            [float(row["axis_angle_error_degrees"]) for row in rows],
            dtype=np.float64,
        )
        offsets = np.asarray(
            [float(row["axis_offset_error_fraction"]) for row in rows],
            dtype=np.float64,
        )
        angle_median = float(np.median(angles))
        offset_median = float(np.median(offsets))
        angle_mad = float(np.median(np.abs(angles - angle_median)))
        offset_mad = float(np.median(np.abs(offsets - offset_median)))
        # Both constraints must fail: a 1.5-degree direction discrepancy alone can
        # come from mask stair-stepping, while a three-percent line-offset alone can
        # come from an imperfect coarse translation. Their conjunction means the
        # frozen camera cannot support the shared sweep axis closely enough to steer
        # physical geometry. These geometric tolerances never consult Gate-B IoU.
        angle_threshold = 1.5
        offset_threshold = 0.03
        for row in rows:
            is_outlier = bool(
                float(row["axis_angle_error_degrees"]) > angle_threshold
                and float(row["axis_offset_error_fraction"]) > offset_threshold
            )
            row["dual_robust_outlier"] = is_outlier
            if is_outlier:
                excluded.append(int(row["selected_index"]))
        groups[sweep_name] = {
            "angle_median_degrees": angle_median,
            "angle_mad_degrees": angle_mad,
            "offset_median_fraction": offset_median,
            "offset_mad_fraction": offset_mad,
            "angle_threshold_degrees": angle_threshold,
            "offset_threshold_fraction": offset_threshold,
            "views": rows,
        }
    return {
        "method": "initial_sweep_pose_dual_axis_plane_physical_consistency_bounds",
        "uses_gate_metrics": False,
        "maximum_axis_angle_error_degrees": 1.5,
        "maximum_axis_offset_error_fraction": 0.03,
        "excluded_selected_indices": sorted(excluded),
        "groups": groups,
    }


def fit_sweep_alignments(observations: Sequence[FitObservation], cameras: Sequence[CameraReference]) -> tuple[dict[str, ModelAlignment], dict[str, float], dict]:
    """Fit four shared sweep poses and common ring levels with all cameras frozen."""
    by_camera = {camera.selected_index: camera for camera in cameras}
    initial_levels = _common_landmark_levels([obs for obs in observations if obs.view_category == "normal_side"])
    keys = (
        "finial_top",
        "finial_bottom",
        "lid_max",
        "neck_top",
        "neck_base",
        "globe_top_transition",
        "globe_max",
        "globe_bottom_transition",
        "bowl_rim",
        "bowl_bottom_transition",
        "pedestal_waist",
        "foot",
    )
    alignments = {}
    for name in CAPTURE_SWEEPS:
        selected = [obs for obs in observations if capture_sweep(obs.selected_index) == name]
        world_points, levels = [], []
        for key in keys:
            rays = []
            for obs in selected:
                centered = _axial_landmark_center(obs, key)
                if centered and centered[1] != "low":
                    rays.append(_world_ray(by_camera[obs.selected_index], centered[0]))
            if len(rays) >= 3 and key in initial_levels:
                world_points.append(_triangulate_rays(rays))
                levels.append(initial_levels[key])
        if len(levels) < 3:
            raise ValueError(f"insufficient repeatable ring evidence in {name}")
        design = np.column_stack((np.ones(len(levels)), levels))
        coefficients = np.linalg.lstsq(design, np.asarray(world_points), rcond=None)[0]
        alignments[name] = _alignment_from_axis(coefficients[0], coefficients[1])

    camera_axis_consistency = _camera_axis_consistency_audit(
        observations, cameras, alignments
    )
    excluded_camera_views = set(
        camera_axis_consistency["excluded_selected_indices"]
    )
    # The three reliable sweeps reconstruct the same physical object at different
    # recovered world scales.  Because absolute SfM scale is a gauge, normalize the
    # *derived V2 camera centers* instead of changing the object's physical scale.
    # For a sweep with independent scale s and chosen common scale s*, moving every
    # camera center about the independently fitted object anchor by s*/s preserves
    # all original projective rays exactly.  Step-13 files remain untouched.
    names = list(RELIABLE_CAPTURE_SWEEPS)
    reliable_initial_scales = np.asarray(
        [alignments[name].scale for name in RELIABLE_CAPTURE_SWEEPS],
        dtype=np.float64,
    )
    shared_initial_scale = float(np.median(reliable_initial_scales))
    initialized_alignments = dict(alignments)
    normalized_by_camera = dict(by_camera)
    camera_center_normalization: dict[str, Any] = {
        "method": "step13_sweep_camera_center_similarity_gauge_normalization",
        "source_step13_files_modified": False,
        "shared_physical_scale": shared_initial_scale,
        "sweeps": {},
    }
    for name in RELIABLE_CAPTURE_SWEEPS:
        alignment = alignments[name]
        factor = shared_initial_scale / alignment.scale
        anchor = np.asarray(alignment.translation, dtype=np.float64)
        initialized_alignments[name] = ModelAlignment(
            scale=shared_initial_scale,
            rotation_matrix=alignment.rotation_matrix,
            translation=alignment.translation,
        )
        camera_center_normalization["sweeps"][name] = {
            "independent_recovered_scale": alignment.scale,
            "shared_physical_scale": shared_initial_scale,
            "factor": factor,
            "anchor_world": anchor.tolist(),
        }
        for obs in observations:
            if capture_sweep(obs.selected_index) == name:
                normalized_by_camera[obs.selected_index] = _scale_camera_center_about_anchor(
                    by_camera[obs.selected_index], anchor, factor
                )

    axis_planes = {
        obs.selected_index: mask_axis_plane(
            obs, normalized_by_camera[obs.selected_index]
        )[:2]
        for obs in observations
        if capture_sweep(obs.selected_index) != TOP_CAPTURE_SWEEP
    }
    mask_boundaries = {
        obs.selected_index: _mask_boundary_points(obs.mask) for obs in observations
    }
    object_heights = {
        obs.selected_index: _object_image_height(obs.mask) for obs in observations
    }
    # Fix two separated levels solely to remove the common similarity gauge.
    # All remaining ring levels are shared unknowns, not per-image height fractions.
    free_keys = [
        key
        for key in keys
        if key not in ("finial_top", "lid_max", "neck_base")
    ]
    pose_parameters = np.concatenate([
        np.concatenate(
            (
                initialized_alignments[name].translation,
                np.asarray(
                    initialized_alignments[name].rotation_matrix,
                    dtype=np.float64,
                )[:, 2],
            )
        )
        for name in names
    ])
    level_parameter_offset = len(pose_parameters)
    initial = np.concatenate(
        (
            pose_parameters,
            [initial_levels[key] for key in free_keys],
        )
    )

    def unpack(parameters):
        poses: dict[str, ModelAlignment] = {
            TOP_CAPTURE_SWEEP: alignments[TOP_CAPTURE_SWEEP]
        }
        for index, name in enumerate(names):
            start = index * 6
            direction = np.asarray(parameters[start + 3:start + 6], dtype=np.float64)
            direction_norm = float(np.linalg.norm(direction))
            if not math.isfinite(direction_norm) or direction_norm <= 1e-9:
                raise ValueError(f"degenerate shared-scale sweep axis for {name}")
            poses[name] = _alignment_from_axis(
                np.asarray(parameters[start:start + 3], dtype=np.float64),
                direction / direction_norm * shared_initial_scale,
            )
        shared = {
            **initial_levels,
            **dict(zip(free_keys, parameters[level_parameter_offset:])),
        }
        # V2's normalized local coordinate contract is exact: bottom=0 and the
        # physical finial top=1.  It is not another optimizable landmark level.
        shared["finial_top"] = 1.0
        shared["lid_lower"] = shared["lid_max"]
        return poses, shared

    def residual(parameters):
        poses, shared = unpack(parameters)
        landmark_values: list[float] = []
        silhouette_values: list[float] = []
        axis_values: list[float] = []
        for obs in observations:
            sweep_name = capture_sweep(obs.selected_index)
            if (
                sweep_name == TOP_CAPTURE_SWEEP
                or obs.selected_index in excluded_camera_views
            ):
                continue
            pose = poses[sweep_name]
            camera = normalized_by_camera[obs.selected_index]
            normalizer = object_heights[obs.selected_index]
            for key in keys:
                point = _axial_landmark_center(obs, key)
                if not point or point[1] == "low":
                    continue
                projected = project_world_point(
                    camera,
                    _model_to_world(pose, np.array([[0, 0, shared[key]]])),
                )
                landmark_values.extend(
                    (np.asarray(projected if projected is not None else (-10000, -10000)) - point[0])
                    / normalizer
                )

            # Sample the observed mask midline at several model heights. This is a
            # true silhouette residual and constrains lateral/vertical sweep pose
            # even where a manually reviewed named landmark is sparse. Every sample
            # contributes exactly two scalars so least_squares sees a fixed-size
            # residual vector even when a trial pose leaves the valid projection.
            for level in np.linspace(0.08, 0.92, 9):
                basis = _projected_axis_basis(camera, pose, float(level))
                endpoints = (
                    _lateral_endpoints_from_points(
                        mask_boundaries[obs.selected_index],
                        basis[0],
                        basis[1],
                        basis[2],
                        normalizer,
                    )
                    if basis is not None
                    else None
                )
                if basis is None or endpoints is None:
                    silhouette_values.extend((1.0, 1.0))
                    continue
                midpoint = (endpoints[0] + endpoints[1]) * 0.5
                silhouette_values.extend((np.asarray(basis[0]) - midpoint) / normalizer)

            # Task 10/11 require the projected model axis to agree with the observed
            # silhouette axis, not merely with component-center landmarks.
            normal, offset = axis_planes[obs.selected_index]
            rotation = np.asarray(pose.rotation_matrix, dtype=np.float64)
            axis_vector = rotation[:, 2] * pose.scale
            axis_scale = max(float(np.linalg.norm(axis_vector)), 1e-9)
            unit_axis = axis_vector / axis_scale
            translation = np.asarray(pose.translation, dtype=np.float64)
            axis_values.append(float(normal @ unit_axis))
            axis_values.append(float((normal @ translation - offset) / axis_scale))

        # Direction magnitude is not a physical degree of freedom because unpack()
        # normalizes it. Keep the four raw direction vectors well conditioned while
        # their orientations remain free and the three reliable sweeps share scale.
        for index, _name in enumerate(names):
            direction = parameters[index * 6 + 3:index * 6 + 6]
            axis_values.append(float(np.linalg.norm(direction) - 1.0))

        # The plan specifies group weights, not per-scalar weights. Normalize each
        # residual group by its sample count so the many landmark scalars do not
        # drown out silhouette/axis evidence, while preserving landmark scale for
        # the robust-loss threshold.
        landmark = np.asarray(landmark_values, dtype=np.float64)
        reference_count = max(len(landmark_values), 1)

        def weighted(values: list[float], group_weight: float) -> np.ndarray:
            if not values:
                return np.empty(0, dtype=np.float64)
            multiplier = (
                (group_weight / 1.5)
                * math.sqrt(reference_count / len(values))
            )
            return np.asarray(values, dtype=np.float64) * multiplier

        return np.concatenate(
            (
                landmark,
                weighted(silhouette_values, 1.0),
                weighted(axis_values, 0.5),
            )
        )

    solution = least_squares(residual, initial, loss="soft_l1", f_scale=0.0125,
                             max_nfev=250, xtol=1e-9, ftol=1e-9, gtol=1e-9)
    poses, shared = unpack(solution.x)
    axial_center_errors: list[float] = []
    for obs in observations:
        sweep_name = capture_sweep(obs.selected_index)
        if (
            sweep_name == TOP_CAPTURE_SWEEP
            or obs.selected_index in excluded_camera_views
        ):
            continue
        pose = poses[sweep_name]
        camera = normalized_by_camera[obs.selected_index]
        normalizer = _object_image_height(obs.mask)
        for key in keys:
            point = _axial_landmark_center(obs, key)
            if not point or point[1] == "low":
                continue
            projected = project_world_point(
                camera,
                _model_to_world(pose, np.array([[0, 0, shared[key]]])),
            )
            if projected is None:
                axial_center_errors.append(float("inf"))
            else:
                axial_center_errors.append(
                    float(np.linalg.norm(np.asarray(projected) - point[0]) / normalizer)
                )
    finite_errors = np.asarray(
        [value for value in axial_center_errors if math.isfinite(value)], dtype=np.float64
    )
    return poses, shared, {
        "method": "three_capture_sweep_poses_shared_physical_scale_with_step13_sweep_gauge_normalization",
        "free_per_view_transforms": False,
        "cameras_refitted": False,
        "source_step13_files_modified": False,
        "camera_center_normalization": camera_center_normalization,
        "reliable_capture_sweeps": list(RELIABLE_CAPTURE_SWEEPS),
        "fit_view_count": sum(
            capture_sweep(obs.selected_index) != TOP_CAPTURE_SWEEP
            and obs.selected_index not in excluded_camera_views
            for obs in observations
        ),
        "evaluation_only_sweep": TOP_CAPTURE_SWEEP,
        "camera_axis_consistency": camera_axis_consistency,
        "excluded_camera_geometry_views": sorted(excluded_camera_views),
        "shared_scale_projective_initialization": camera_center_normalization["sweeps"],
        "shared_reliable_physical_scale": poses[RELIABLE_CAPTURE_SWEEPS[0]].scale,
        "evaluation_only_top_scale": poses[TOP_CAPTURE_SWEEP].scale,
        "physical_scale_invariant": bool(
            max(poses[name].scale for name in RELIABLE_CAPTURE_SWEEPS)
            - min(poses[name].scale for name in RELIABLE_CAPTURE_SWEEPS)
            <= 1e-12
        ),
        "converged": bool(solution.success), "nfev": solution.nfev,
        "axial_center_error_median": (
            float(np.median(finite_errors)) if finite_errors.size else float("inf")
        ),
        "axial_center_error_p95": (
            float(np.percentile(finite_errors, 95)) if finite_errors.size else float("inf")
        ),
        "gauge_fixed_levels": {
            "finial_top": 1.0,
            **{
                key: initial_levels[key]
                for key in ("lid_max", "neck_base")
            },
        },
    }


def _profile_interpolator(
    sections: Sequence[tuple[float, float]],
) -> PchipInterpolator:
    levels = np.asarray([level for level, _ in sections], dtype=np.float64)
    radii = np.asarray([radius for _, radius in sections], dtype=np.float64)
    if levels.ndim != 1 or len(levels) < 2 or np.any(np.diff(levels) <= 0.0):
        raise ValueError("profile control levels must be strictly increasing")
    if np.any(radii < 0.0) or not np.all(np.isfinite(radii)):
        raise ValueError("profile radii must be finite and nonnegative")
    return PchipInterpolator(levels, radii, extrapolate=False)


def _circle_world_points(
    radius: float,
    level: float,
    angular_samples: int,
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, angular_samples, endpoint=False)
    return np.column_stack(
        (
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.full(angular_samples, level),
        )
    )


def render_profile_silhouette(
    sections: Sequence[tuple[float, float]],
    camera: CameraReference,
    alignment: ModelAlignment,
    image_size: tuple[int, int],
    *,
    angular_samples: int = 40,
    height_samples: int = 49,
) -> np.ndarray:
    """Rasterize a surface of revolution using actual pyCOLMAP projection."""

    width, height = image_size
    if width < 2 or height < 2:
        raise ValueError("invalid silhouette image size")
    interpolator = _profile_interpolator(sections)
    levels = np.unique(np.concatenate((
        np.linspace(sections[0][0], sections[-1][0], height_samples),
        np.asarray(sections)[:, 0],
    )))
    radii = np.clip(interpolator(levels), 0.0, None)
    rings = [
        _model_to_world(
            alignment,
            _circle_world_points(float(radius), float(level), angular_samples),
        )
        for level, radius in zip(levels, radii, strict=True)
    ]
    projected_rings: list[np.ndarray] = []
    for ring in rings:
        projected = project_world_points(camera, ring)
        if projected is None:
            return np.zeros((height, width), dtype=np.uint8)
        projected_rings.append(projected)

    # Each adjacent ring pair bounds a convex frustum. Union their projections,
    # including the end caps. Two lateral points per ring omit the visible arcs
    # in elevated views and degenerate entirely for an axial camera. A single
    # fillPoly call on overlapping faces is also wrong: its even/odd fill rule
    # cancels overlapping interiors. Independent convex fills give a solid union.
    canvas = np.zeros((height, width), dtype=np.uint8)
    for first, second in zip(projected_rings[:-1], projected_rings[1:]):
        points = np.rint(np.concatenate((first, second))).astype(np.int32)
        cv2.fillConvexPoly(canvas, cv2.convexHull(points), 255)
    return canvas


def _nearest_row_extents(
    binary: np.ndarray,
    requested_y: float,
) -> tuple[int, int] | None:
    foreground_rows = np.flatnonzero(np.any(binary > 0, axis=1))
    if foreground_rows.size == 0:
        return None
    row = int(foreground_rows[np.argmin(np.abs(foreground_rows - requested_y))])
    columns = np.flatnonzero(binary[row] > 0)
    if columns.size == 0:
        return None
    return int(columns[0]), int(columns[-1])


def _projected_half_width(
    camera: CameraReference,
    alignment: ModelAlignment,
    level: float,
    radius: float,
    center_xy: tuple[float, float],
    lateral_xy: np.ndarray,
    angular_samples: int = 48,
) -> float | None:
    points = _model_to_world(
        alignment,
        _circle_world_points(radius, level, angular_samples),
    )
    projected = project_world_points(camera, points)
    if projected is None:
        return None
    lateral = np.asarray(lateral_xy, dtype=np.float64)
    length = float(np.linalg.norm(lateral))
    if not math.isfinite(length) or length <= 1e-9:
        return None
    lateral /= length
    offsets = (projected - np.asarray(center_xy, dtype=np.float64)) @ lateral
    return float((np.max(offsets) - np.min(offsets)) * 0.5)


def _projected_axis_basis(
    camera: CameraReference,
    alignment: ModelAlignment,
    level: float,
) -> tuple[tuple[float, float], np.ndarray, np.ndarray] | None:
    """Return projected center, axis direction, and image-plane lateral direction."""

    center = project_world_point(
        camera,
        _model_to_world(alignment, np.asarray(((0.0, 0.0, float(level)),))),
    )
    if center is None:
        return None
    delta = 0.0125
    low = max(0.0, float(level) - delta)
    high = min(1.0, float(level) + delta)
    if high - low <= 1e-9:
        return None
    axis_points = _model_to_world(
        alignment,
        np.asarray(((0.0, 0.0, low), (0.0, 0.0, high)), dtype=np.float64),
    )
    projected_axis = project_world_points(camera, axis_points)
    if projected_axis is None:
        return None
    axis = projected_axis[1] - projected_axis[0]
    axis_length = float(np.linalg.norm(axis))
    if not math.isfinite(axis_length) or axis_length <= 1e-9:
        return None
    axis /= axis_length
    lateral = np.asarray((axis[1], -axis[0]), dtype=np.float64)
    return (float(center[0]), float(center[1])), axis, lateral


def _projected_profile_axial_endpoint(
    camera: CameraReference,
    alignment: ModelAlignment,
    level: float,
    radius: float,
    *,
    toward_top: bool,
    angular_samples: int = 96,
) -> tuple[float, float] | None:
    """Project the visible axial extreme of a finite-radius profile end ring.

    Reviewed ``axis_top``/``axis_bottom`` points are silhouette extrema. For an
    elevated camera, a horizontal end ring projects above or below its 3D axis
    center, so comparing those observations to the projected center introduces
    a systematic image-space translation error.
    """

    basis = _projected_axis_basis(camera, alignment, float(level))
    if basis is None:
        return None
    center, axis, _ = basis
    ring = project_world_points(
        camera,
        _model_to_world(
            alignment,
            _circle_world_points(
                max(float(radius), 0.0),
                float(level),
                angular_samples=angular_samples,
            ),
        ),
    )
    if ring is None:
        return None
    offsets = (ring - np.asarray(center, dtype=np.float64)) @ np.asarray(
        axis, dtype=np.float64
    )
    index = int(np.argmax(offsets) if toward_top else np.argmin(offsets))
    return float(ring[index, 0]), float(ring[index, 1])


def _projected_profile_lateral_endpoints(
    camera: CameraReference,
    alignment: ModelAlignment,
    level: float,
    radius: float,
    *,
    angular_samples: int = 96,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Project image-left/right tangent endpoints for one profile ring."""

    basis = _projected_axis_basis(camera, alignment, float(level))
    if basis is None:
        return None
    center, _, lateral = basis
    ring = project_world_points(
        camera,
        _model_to_world(
            alignment,
            _circle_world_points(
                max(float(radius), 0.0),
                float(level),
                angular_samples=angular_samples,
            ),
        ),
    )
    if ring is None:
        return None
    offsets = (ring - np.asarray(center, dtype=np.float64)) @ np.asarray(
        lateral, dtype=np.float64
    )
    endpoints = (
        np.asarray(ring[int(np.argmin(offsets))], dtype=np.float64),
        np.asarray(ring[int(np.argmax(offsets))], dtype=np.float64),
    )
    return endpoints if endpoints[0][0] <= endpoints[1][0] else endpoints[::-1]


def _paired_landmark_points(
    observation: FitObservation,
    key: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    by_name = {str(item["name"]): item for item in observation.landmarks}
    left = by_name.get(f"{key}_left")
    right = by_name.get(f"{key}_right")
    if left is None or right is None:
        return None
    return (
        np.asarray((float(left["x"]), float(left["y"])), dtype=np.float64),
        np.asarray((float(right["x"]), float(right["y"])), dtype=np.float64),
    )


def _mask_boundary_points(binary: np.ndarray) -> np.ndarray:
    """Return contour pixels sufficient for repeated silhouette cross-section queries."""

    mask = np.where(np.asarray(binary) > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    usable = [contour.reshape(-1, 2) for contour in contours if len(contour) >= 2]
    if not usable:
        return np.empty((0, 2), dtype=np.float64)
    return np.concatenate(usable, axis=0).astype(np.float64)


def _mask_boundary_distance_field(binary: np.ndarray) -> np.ndarray:
    """Return pixel distance to the nearest reviewed outer silhouette boundary."""

    mask = np.where(np.asarray(binary) > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("silhouette distance field requires foreground")
    boundary = np.zeros_like(mask)
    cv2.drawContours(boundary, contours, -1, 255, 1)
    inverse = np.where(boundary > 0, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(inverse, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def _sample_distance_field(field: np.ndarray, point_xy: Sequence[float]) -> float:
    """Bilinearly sample a 2D distance field; outside-image points fail closed."""

    x, y = (float(point_xy[0]), float(point_xy[1]))
    height, width = field.shape
    if not (math.isfinite(x) and math.isfinite(y)):
        return float("inf")
    if x < 0.0 or y < 0.0 or x > width - 1 or y > height - 1:
        return float("inf")
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    tx = x - x0
    ty = y - y0
    top = float(field[y0, x0]) * (1.0 - tx) + float(field[y0, x1]) * tx
    bottom = float(field[y1, x0]) * (1.0 - tx) + float(field[y1, x1]) * tx
    return top * (1.0 - ty) + bottom * ty


def _lateral_endpoints_from_points(
    points: np.ndarray,
    center_xy: tuple[float, float],
    axis_xy: np.ndarray,
    lateral_xy: np.ndarray,
    object_height: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return image-left/right boundary endpoints in a slab normal to the model axis."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (2,) or len(values) < 2:
        return None
    centered = values - np.asarray(center_xy, dtype=np.float64)
    axis_offsets = centered @ np.asarray(axis_xy, dtype=np.float64)
    slab = max(0.75, min(2.5, float(object_height) * 0.002))
    selected = np.abs(axis_offsets) <= slab
    if int(np.count_nonzero(selected)) < 2:
        selected = np.abs(axis_offsets) <= max(1.5, slab * 2.0)
    if int(np.count_nonzero(selected)) < 2:
        return None
    slab_points = values[selected]
    lateral_offsets = centered[selected] @ np.asarray(lateral_xy, dtype=np.float64)
    first = slab_points[int(np.argmin(lateral_offsets))]
    second = slab_points[int(np.argmax(lateral_offsets))]
    return (first, second) if first[0] <= second[0] else (second, first)


def _mask_lateral_endpoints(
    binary: np.ndarray,
    center_xy: tuple[float, float],
    axis_xy: np.ndarray,
    lateral_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return image-left/right silhouette endpoints in a slab normal to the model axis."""

    return _lateral_endpoints_from_points(
        _mask_boundary_points(binary),
        center_xy,
        axis_xy,
        lateral_xy,
        _object_image_height(binary),
    )


def _mask_half_width_perpendicular_to_axis(
    binary: np.ndarray,
    center_xy: tuple[float, float],
    axis_xy: np.ndarray,
    lateral_xy: np.ndarray,
) -> float | None:
    """Measure silhouette width in a narrow slab normal to the projected model axis."""

    endpoints = _mask_lateral_endpoints(binary, center_xy, axis_xy, lateral_xy)
    if endpoints is None:
        return None
    lateral = np.asarray(lateral_xy, dtype=np.float64)
    length = float(np.linalg.norm(lateral))
    if not math.isfinite(length) or length <= 1e-9:
        return None
    lateral /= length
    width = abs(float((endpoints[1] - endpoints[0]) @ lateral)) * 0.5
    return width if math.isfinite(width) and width > 0.0 else None


def _observed_landmark_span_basis(
    observation: FitObservation,
    top_key: str,
    bottom_key: str,
    fraction_from_bottom: float,
) -> tuple[tuple[float, float], np.ndarray, np.ndarray] | None:
    """Return an observed component-local center/axis/lateral sampling basis."""

    top = _axial_landmark_center(observation, top_key)
    bottom = _axial_landmark_center(observation, bottom_key)
    if top is None or bottom is None or top[1] == "low" or bottom[1] == "low":
        return None
    fraction = min(max(float(fraction_from_bottom), 0.0), 1.0)
    top_point = np.asarray(top[0], dtype=np.float64)
    bottom_point = np.asarray(bottom[0], dtype=np.float64)
    axis = top_point - bottom_point
    length = float(np.linalg.norm(axis))
    if not math.isfinite(length) or length <= 1e-9:
        return None
    axis /= length
    lateral = np.asarray((axis[1], -axis[0]), dtype=np.float64)
    center = bottom_point + fraction * (top_point - bottom_point)
    return (float(center[0]), float(center[1])), axis, lateral


def fit_profiles_from_observations(
    observations: Sequence[FitObservation],
    cameras: Sequence[CameraReference],
    alignment: ModelAlignment,
    *,
    control_levels: Sequence[float] | None = None,
    sweep_alignments: Mapping[str, ModelAlignment] | None = None,
) -> dict[str, FittedComponentProfile]:
    """Fit independent outer-radius controls from reviewed projected silhouettes."""

    camera_by_index = {camera.selected_index: camera for camera in cameras}
    pairs: list[tuple[FitObservation, CameraReference]] = []
    source_views: list[int] = []
    for observation in observations:
        camera = camera_by_index.get(observation.selected_index)
        if camera is None:
            raise ValueError(f"missing camera for observation {observation.selected_index}")
        if camera.image_size != observation.mask.shape[::-1]:
            raise ValueError(f"observation/camera size mismatch: {observation.filename}")
        if observation.view_category == "top_down_rim":
            continue
        if not np.any(observation.mask > 0):
            continue
        pairs.append((observation, camera))
        source_views.append(observation.selected_index)
    if len(pairs) < 2:
        raise ValueError("profile fitting requires at least two reliable side views")

    levels = (
        np.asarray(control_levels, dtype=np.float64)
        if control_levels is not None
        else np.linspace(0.0, 1.0, 33)
    )
    if levels.ndim != 1 or len(levels) < 2 or np.any(np.diff(levels) <= 0.0):
        raise ValueError("control levels must be a strictly increasing sequence")
    if levels[0] != 0.0 or levels[-1] != 1.0:
        levels = np.concatenate(((0.0,), levels[(levels > 0.0) & (levels < 1.0)], (1.0,)))

    trial_radius = 0.15
    fitted_radii = np.empty(len(levels), dtype=np.float64)
    for level_index, level in enumerate(levels):
        estimates: list[float] = []
        objectives: list[
            tuple[
                CameraReference,
                ModelAlignment,
                float,
                tuple[float, float],
                np.ndarray,
            ]
        ] = []
        for observation, camera in pairs:
            pose = sweep_alignments[capture_sweep(observation.selected_index)] if sweep_alignments else alignment
            basis = _projected_axis_basis(camera, pose, float(level))
            if basis is None:
                continue
            center, axis, lateral = basis
            observed_half_width = _mask_half_width_perpendicular_to_axis(
                observation.mask, center, axis, lateral
            )
            if observed_half_width is None:
                continue
            trial_width = _projected_half_width(
                camera, pose, float(level), trial_radius, center, lateral
            )
            if trial_width is None or trial_width <= 1e-9 or observed_half_width <= 0.0:
                continue
            estimate = trial_radius * observed_half_width / trial_width
            if np.isfinite(estimate) and estimate > 0.0:
                estimates.append(estimate)
                objectives.append((camera, pose, observed_half_width, center, lateral))
        if not estimates:
            raise ValueError(f"no silhouette evidence for profile level {level:.4f}")
        initial = float(np.median(estimates))

        if len(objectives) >= 2:
            def residual(parameter: np.ndarray) -> np.ndarray:
                values: list[float] = []
                for camera, pose, observed, center, lateral in objectives:
                    predicted = _projected_half_width(
                        camera,
                        pose,
                        float(level),
                        float(parameter[0]),
                        center,
                        lateral,
                    )
                    values.append((predicted if predicted is not None else 0.0) - observed)
                return np.asarray(values, dtype=np.float64)

            solved = least_squares(
                residual,
                np.asarray((initial,)),
                bounds=(0.002, 0.75),
                loss="soft_l1",
                f_scale=2.0,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
                max_nfev=80,
            )
            initial = float(solved.x[0])
        fitted_radii[level_index] = min(max(initial, 0.002), 0.75)

    sections = tuple(
        (float(level), float(radius))
        for level, radius in zip(levels, fitted_radii, strict=True)
    )
    profiles: dict[str, FittedComponentProfile] = {
        "outer": FittedComponentProfile(
            name="outer",
            sections=sections,
            measurement_status="projected_silhouette_fit",
            source_views=tuple(sorted(set(source_views))),
        )
    }
    by_level = dict(sections)
    for name, (low, high) in PROFILE_RANGES.items():
        component_sections = tuple(
            (level, radius)
            for level, radius in sections
            if low - 1e-12 <= level <= high + 1e-12
        )
        if len(component_sections) < 2:
            selected_levels = np.asarray((low, high), dtype=np.float64)
            component_sections = tuple(
                (float(level), float(np.interp(level, levels, fitted_radii)))
                for level in selected_levels
            )
        profiles[name] = FittedComponentProfile(
            name=name,
            sections=component_sections,
            measurement_status="projected_silhouette_fit",
            source_views=tuple(sorted(set(source_views))),
        )

    top_down_views = tuple(
        observation.selected_index
        for observation in observations
        if observation.view_category == "top_down_rim"
    )
    for name, outer_name, factor in (
        ("bowl_inner", "bowl_outer", 0.88),
        ("neck_inner", "neck_outer", 0.78),
    ):
        profiles[name] = FittedComponentProfile(
            name=name,
            sections=tuple(
                (level, radius * factor)
                for level, radius in profiles[outer_name].sections
            ),
            measurement_status="bounded_opening_inference",
            source_views=top_down_views,
        )
    return profiles


def _fit_component_profile_sections(
    observations: Sequence[FitObservation],
    cameras: Sequence[CameraReference],
    sweep_alignments: Mapping[str, ModelAlignment],
    component_masks: Mapping[int, np.ndarray],
    level_range: tuple[float, float],
    *,
    control_count: int = 9,
    axial_landmark_span: tuple[str, str] | None = None,
) -> tuple[tuple[tuple[float, float], ...], tuple[int, ...]]:
    """Fit one radial component from its own reviewed V2 component masks."""

    low, high = (float(level_range[0]), float(level_range[1]))
    if not (0.0 <= low < high <= 1.0):
        raise ValueError("component profile level range must satisfy 0 <= low < high <= 1")
    if control_count < 3:
        raise ValueError("component profile requires at least three controls")
    camera_by_index = {camera.selected_index: camera for camera in cameras}
    levels = np.linspace(low, high, control_count)
    boundaries = {
        index: _mask_boundary_points(mask)
        for index, mask in component_masks.items()
        if np.any(mask > 0)
    }
    object_heights = {
        index: _object_image_height(mask)
        for index, mask in component_masks.items()
        if np.any(mask > 0)
    }
    candidate_source_views = tuple(sorted(boundaries))
    if len(candidate_source_views) < 2:
        raise ValueError("component profile requires at least two non-empty reviewed views")
    used_source_views: set[int] = set()

    trial_radius = 0.15
    raw_radii = np.full(len(levels), np.nan, dtype=np.float64)
    for level_index, level in enumerate(levels):
        estimates: list[float] = []
        objectives: list[
            tuple[CameraReference, ModelAlignment, float, tuple[float, float], np.ndarray]
        ] = []
        for observation in observations:
            if observation.view_category == "top_down_rim":
                continue
            selected_index = observation.selected_index
            if selected_index not in boundaries:
                continue
            camera = camera_by_index[selected_index]
            pose = sweep_alignments[capture_sweep(selected_index)]
            basis = _projected_axis_basis(camera, pose, float(level))
            if basis is None:
                continue
            center, axis, lateral = basis
            observed_basis = (
                _observed_landmark_span_basis(
                    observation,
                    axial_landmark_span[0],
                    axial_landmark_span[1],
                    (float(level) - low) / (high - low),
                )
                if axial_landmark_span is not None
                else None
            )
            if axial_landmark_span is not None and observed_basis is None:
                continue
            observed_center, observed_axis, observed_lateral = (
                observed_basis if observed_basis is not None else basis
            )
            endpoints = _lateral_endpoints_from_points(
                boundaries[selected_index],
                observed_center,
                observed_axis,
                observed_lateral,
                object_heights[selected_index],
            )
            if endpoints is None:
                continue
            observed_half_width = abs(
                float((endpoints[1] - endpoints[0]) @ observed_lateral)
            ) * 0.5
            trial_width = _projected_half_width(
                camera, pose, float(level), trial_radius, center, lateral
            )
            if (
                trial_width is None
                or trial_width <= 1e-9
                or observed_half_width <= 0.0
            ):
                continue
            estimate = trial_radius * observed_half_width / trial_width
            if np.isfinite(estimate) and estimate > 0.0:
                estimates.append(estimate)
                objectives.append((camera, pose, observed_half_width, center, lateral))
                used_source_views.add(selected_index)
        if not estimates:
            continue
        radius = float(np.median(estimates))
        if len(objectives) >= 2:
            def residual(parameter: np.ndarray) -> np.ndarray:
                values: list[float] = []
                for camera, pose, observed_width, center, lateral in objectives:
                    predicted_width = _projected_half_width(
                        camera,
                        pose,
                        float(level),
                        float(parameter[0]),
                        center,
                        lateral,
                    )
                    values.append(
                        (predicted_width if predicted_width is not None else 0.0)
                        - observed_width
                    )
                return np.asarray(values, dtype=np.float64)

            solution = least_squares(
                residual,
                np.asarray((radius,)),
                bounds=(0.002, 0.75),
                loss="soft_l1",
                f_scale=2.0,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
                max_nfev=80,
            )
            radius = float(solution.x[0])
        raw_radii[level_index] = min(max(radius, 0.002), 0.75)

    valid = np.isfinite(raw_radii)
    if int(np.count_nonzero(valid)) < 3:
        raise ValueError("insufficient projected component-width evidence")
    fitted_radii = np.interp(levels, levels[valid], raw_radii[valid])
    sections = tuple(
        (float(level), float(radius))
        for level, radius in zip(levels, fitted_radii, strict=True)
    )
    source_views = tuple(sorted(used_source_views))
    if len(source_views) < 2:
        raise ValueError("component profile requires usable evidence from at least two views")
    return sections, source_views


def fit_independent_component_profiles(
    project_root: Path,
    observations: Sequence[FitObservation],
    cameras: Sequence[CameraReference],
    sweep_alignments: Mapping[str, ModelAlignment],
    common_levels: Mapping[str, float],
) -> dict[str, FittedComponentProfile]:
    """Fit the physically overlapping V2 assemblies/components independently."""

    required_levels = (
        "bowl_bottom_transition",
        "bowl_rim",
        "globe_bottom_transition",
        "globe_max",
        "globe_top_transition",
        "neck_base",
        "neck_top",
        "finial_bottom",
    )
    missing = [name for name in required_levels if name not in common_levels]
    if missing:
        raise ValueError("missing shared component levels: " + ", ".join(missing))

    ranges = {
        "pedestal": (0.0, float(common_levels["bowl_bottom_transition"])),
        "bowl_outer": (
            float(common_levels["bowl_bottom_transition"]),
            float(common_levels["bowl_rim"]),
        ),
        "globe": (
            float(common_levels["globe_bottom_transition"]),
            float(common_levels["globe_top_transition"]),
        ),
        "shoulder": (
            float(common_levels["globe_max"]),
            float(common_levels["neck_base"]),
        ),
        "neck_outer": (
            float(common_levels["neck_base"]),
            float(common_levels["neck_top"]),
        ),
        "lid": (
            float(common_levels["neck_top"]),
            float(common_levels["finial_bottom"]),
        ),
        "finial": (float(common_levels["finial_bottom"]), 1.0),
        "receiving_assembly": (0.0, float(common_levels["bowl_rim"])),
        "main_assembly": (float(common_levels["globe_bottom_transition"]), 1.0),
    }
    mask_source = {
        "pedestal": "receiving_bowl_pedestal",
        "bowl_outer": "receiving_bowl_pedestal",
        "globe": "globe",
        "shoulder": "main_vessel",
        "neck_outer": "neck",
        "lid": "lid_finial",
        "finial": "lid_finial",
        "receiving_assembly": "receiving_bowl_pedestal",
        "main_assembly": "main_vessel",
    }
    root = project_root.resolve()
    target_size_by_index = {
        observation.selected_index: observation.mask.shape[::-1] for observation in observations
    }
    mask_cache: dict[str, dict[int, np.ndarray]] = {}

    def masks_for(component_name: str) -> dict[int, np.ndarray]:
        cached = mask_cache.get(component_name)
        if cached is not None:
            return cached
        records: dict[int, np.ndarray] = {}
        for selected_index, target_size in target_size_by_index.items():
            path = (
                root
                / V2_RELATIVE_ROOT
                / "evidence"
                / "component_masks"
                / f"{selected_index:03d}_{component_name}.png"
            )
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            records[selected_index] = cv2.resize(
                mask, target_size, interpolation=cv2.INTER_NEAREST
            )
        mask_cache[component_name] = records
        return records

    profiles: dict[str, FittedComponentProfile] = {}
    for name, level_range in ranges.items():
        sections, source_views = _fit_component_profile_sections(
            observations,
            cameras,
            sweep_alignments,
            masks_for(mask_source[name]),
            level_range,
            control_count=13 if name.endswith("_assembly") else 9,
            axial_landmark_span=("finial_top", "finial_bottom")
            if name == "finial"
            else None,
        )
        profiles[name] = FittedComponentProfile(
            name=name,
            sections=sections,
            measurement_status="independent_projected_component_silhouette_fit",
            source_views=source_views,
        )

    top_down_views = tuple(
        observation.selected_index
        for observation in observations
        if observation.view_category == "top_down_rim"
    )
    for name, outer_name, factor in (
        ("bowl_inner", "bowl_outer", 0.88),
        ("neck_inner", "neck_outer", 0.78),
    ):
        profiles[name] = FittedComponentProfile(
            name=name,
            sections=tuple(
                (level, radius * factor)
                for level, radius in profiles[outer_name].sections
            ),
            measurement_status="bounded_opening_inference_from_top_down_evidence",
            source_views=top_down_views,
        )
    return profiles


def refine_component_profile_radii(
    project_root: Path,
    observations: Sequence[FitObservation],
    cameras: Sequence[CameraReference],
    sweep_alignments: Mapping[str, ModelAlignment],
    profiles: Mapping[str, FittedComponentProfile],
) -> tuple[dict[str, FittedComponentProfile], dict[str, Any]]:
    """Jointly refine all named outer-profile radii within tight measured bounds."""

    missing = [name for name in OUTER_SILHOUETTE_PROFILES if name not in profiles]
    if missing:
        raise ValueError("missing profiles for joint radius refinement: " + ", ".join(missing))
    camera_by_index = {camera.selected_index: camera for camera in cameras}
    reliable = tuple(obs for obs in observations if obs.view_category != "top_down_rim")
    if len(reliable) < 8:
        raise ValueError("joint radius refinement requires reliable multi-view evidence")

    whole_heights = {
        obs.selected_index: _object_image_height(obs.mask) for obs in reliable
    }
    component_source = {
        "pedestal": "receiving_bowl_pedestal",
        "bowl_outer": "receiving_bowl_pedestal",
        "globe": "globe",
        "shoulder": "main_vessel",
        "neck_outer": "neck",
        "lid": "lid_finial",
        "finial": "lid_finial",
    }
    root = project_root.resolve()
    component_boundaries: dict[tuple[str, int], np.ndarray] = {}
    component_distance_fields: dict[tuple[str, int], np.ndarray] = {}
    for name in OUTER_SILHOUETTE_PROFILES:
        source_name = component_source[name]
        for obs in reliable:
            path = (
                root
                / V2_RELATIVE_ROOT
                / "evidence"
                / "component_masks"
                / f"{obs.selected_index:03d}_{source_name}.png"
            )
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"missing component mask for joint refinement: {path.name}")
            resized = cv2.resize(mask, obs.mask.shape[::-1], interpolation=cv2.INTER_NEAREST)
            component_boundaries[(name, obs.selected_index)] = _mask_boundary_points(resized)
            component_distance_fields[(name, obs.selected_index)] = (
                _mask_boundary_distance_field(resized)
            )

    initial_values: list[float] = []
    levels_by_name: dict[str, tuple[float, ...]] = {}
    slices: dict[str, slice] = {}
    lower_values: list[float] = []
    upper_values: list[float] = []
    cursor = 0
    for name in OUTER_SILHOUETTE_PROFILES:
        sections = profiles[name].sections
        levels_by_name[name] = tuple(float(level) for level, _ in sections)
        values = [float(radius) for _, radius in sections]
        initial_values.extend(values)
        for radius in values:
            delta = max(0.006, radius * 0.12)
            lower_values.append(max(0.002, radius - delta))
            upper_values.append(min(0.75, radius + delta))
        slices[name] = slice(cursor, cursor + len(values))
        cursor += len(values)

    initial = np.asarray(initial_values, dtype=np.float64)
    lower = np.asarray(lower_values, dtype=np.float64)
    upper = np.asarray(upper_values, dtype=np.float64)
    residuals_per_parameter = len(reliable) * 3 + 1
    jac_sparsity = np.zeros(
        (len(initial) * residuals_per_parameter, len(initial)), dtype=bool
    )
    for parameter_index in range(len(initial)):
        start = parameter_index * residuals_per_parameter
        jac_sparsity[start : start + residuals_per_parameter, parameter_index] = True

    def residual(parameters: np.ndarray) -> np.ndarray:
        values: list[float] = []
        parameter_index = 0
        for name in OUTER_SILHOUETTE_PROFILES:
            for level in levels_by_name[name]:
                radius = float(parameters[parameter_index])
                for obs in reliable:
                    camera = camera_by_index[obs.selected_index]
                    pose = sweep_alignments[capture_sweep(obs.selected_index)]
                    normalizer = whole_heights[obs.selected_index]
                    basis = _projected_axis_basis(camera, pose, level)
                    if basis is None:
                        values.extend((1.0, 1.0, 1.0))
                        continue
                    center, axis, lateral = basis
                    projected_ring = project_world_points(
                        camera,
                        _model_to_world(
                            pose,
                            _circle_world_points(radius, level, angular_samples=40),
                        ),
                    )
                    if projected_ring is None:
                        values.extend((1.0, 1.0, 1.0))
                        continue
                    offsets = (
                        projected_ring - np.asarray(center, dtype=np.float64)
                    ) @ np.asarray(lateral, dtype=np.float64)
                    predicted_half_width = float((np.max(offsets) - np.min(offsets)) * 0.5)
                    predicted_endpoints = (
                        projected_ring[int(np.argmin(offsets))],
                        projected_ring[int(np.argmax(offsets))],
                    )
                    for endpoint in predicted_endpoints:
                        distance = _sample_distance_field(
                            component_distance_fields[(name, obs.selected_index)], endpoint
                        )
                        values.append(
                            min(distance / normalizer, 1.0)
                            if math.isfinite(distance)
                            else 1.0
                        )
                    observed_endpoints = _lateral_endpoints_from_points(
                        component_boundaries[(name, obs.selected_index)],
                        center,
                        axis,
                        lateral,
                        normalizer,
                    )
                    if observed_endpoints is None:
                        values.append(1.0)
                    else:
                        observed_half_width = abs(
                            float((observed_endpoints[1] - observed_endpoints[0]) @ lateral)
                        ) * 0.5
                        values.append(
                            (predicted_half_width - observed_half_width) / normalizer
                        )
                span = max(float(upper[parameter_index] - lower[parameter_index]), 1e-9)
                values.append(
                    0.12 * float(parameters[parameter_index] - initial[parameter_index]) / span
                )
                parameter_index += 1
        return np.asarray(values, dtype=np.float64)

    initial_residual = residual(initial)
    solution = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        jac_sparsity=jac_sparsity,
        loss="soft_l1",
        f_scale=0.0125,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
        max_nfev=100,
    )
    final_residual = residual(solution.x)
    initial_rms = float(np.sqrt(np.mean(np.square(initial_residual))))
    final_rms = float(np.sqrt(np.mean(np.square(final_residual))))
    use_solution = bool(np.isfinite(final_rms) and final_rms <= initial_rms + 1e-12)
    selected = solution.x if use_solution else initial

    refined = dict(profiles)
    for name in OUTER_SILHOUETTE_PROFILES:
        section_slice = slices[name]
        refined[name] = FittedComponentProfile(
            name=name,
            sections=tuple(
                (level, float(radius))
                for level, radius in zip(
                    levels_by_name[name], selected[section_slice], strict=True
                )
            ),
            measurement_status=(
                "joint_bounded_multi_view_profile_refinement"
                if use_solution
                else profiles[name].measurement_status
            ),
            source_views=profiles[name].source_views,
        )
    for inner_name, outer_name, factor in (
        ("bowl_inner", "bowl_outer", 0.88),
        ("neck_inner", "neck_outer", 0.78),
    ):
        refined[inner_name] = FittedComponentProfile(
            name=inner_name,
            sections=tuple(
                (level, radius * factor) for level, radius in refined[outer_name].sections
            ),
            measurement_status="bounded_opening_inference_from_top_down_evidence",
            source_views=profiles[inner_name].source_views,
        )
    return refined, {
        "method": "joint_bounded_named_profile_radius_refinement",
        "converged": bool(solution.success),
        "used_solution": use_solution,
        "nfev": int(solution.nfev),
        "parameter_count": len(initial),
        "radius_bound_fraction": 0.12,
        "minimum_absolute_radius_bound": 0.006,
        "initial_weighted_rms": initial_rms,
        "final_weighted_rms": final_rms,
        "free_per_view_transforms": False,
        "cameras_refitted": False,
    }


def render_independent_assembly_silhouette(
    profiles: Mapping[str, FittedComponentProfile],
    camera: CameraReference,
    alignment: ModelAlignment,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Render the union of independently fitted named outer components."""

    missing = [name for name in OUTER_SILHOUETTE_PROFILES if name not in profiles]
    if missing:
        raise ValueError("missing outer silhouette profiles: " + ", ".join(missing))
    canvas = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    for name in OUTER_SILHOUETTE_PROFILES:
        component = render_profile_silhouette(
            profiles[name].sections, camera, alignment, image_size
        )
        canvas = cv2.bitwise_or(canvas, component)
    return canvas


def refine_sweep_alignments_to_profile(
    observations: Sequence[FitObservation],
    cameras: Sequence[CameraReference],
    sweep_alignments: Mapping[str, ModelAlignment],
    profile_sections: Sequence[tuple[float, float]],
    common_levels: Mapping[str, float],
) -> tuple[dict[str, ModelAlignment], dict[str, Any]]:
    """Jointly refine each shared capture-sweep pose against the common profile.

    Step-13 cameras stay frozen and every observation in one capture sweep shares
    exactly one similarity transform. The residual groups implement the V2 plan's
    silhouette, landmark, axis, and component-width evidence without introducing
    per-view corrections.
    """

    camera_by_index = {camera.selected_index: camera for camera in cameras}
    profile = _profile_interpolator(profile_sections)
    sample_levels = tuple(float(value) for value in np.linspace(0.08, 0.92, 13))
    landmark_keys = (
        "finial_top",
        "finial_bottom",
        "lid_max",
        "neck_top",
        "neck_base",
        "globe_top_transition",
        "globe_max",
        "globe_bottom_transition",
        "bowl_rim",
        "bowl_bottom_transition",
        "pedestal_waist",
        "foot",
    )
    boundaries = {
        obs.selected_index: _mask_boundary_points(obs.mask) for obs in observations
    }
    boundary_distance_fields = {
        obs.selected_index: _mask_boundary_distance_field(obs.mask) for obs in observations
    }
    object_heights = {
        obs.selected_index: _object_image_height(obs.mask) for obs in observations
    }
    axis_planes = {
        obs.selected_index: mask_axis_plane(obs, camera_by_index[obs.selected_index])[:2]
        for obs in observations
    }
    refined = dict(sweep_alignments)
    summaries: dict[str, Any] = {}

    for sweep_name in CAPTURE_SWEEPS:
        sweep_observations = tuple(
            obs
            for obs in observations
            if capture_sweep(obs.selected_index) == sweep_name
            and obs.view_category != "top_down_rim"
        )
        if not sweep_observations:
            summaries[sweep_name] = {
                "status": "not_refined_evaluation_only_top_down",
                "view_count": 0,
            }
            continue

        initial_pose = refined[sweep_name]
        initial_rotation = np.asarray(initial_pose.rotation_matrix, dtype=np.float64)
        initial_axis_direction = initial_rotation[:, 2]
        initial_parameters = np.concatenate(
            (
                np.asarray(initial_pose.translation, dtype=np.float64),
                initial_axis_direction,
            )
        )
        scale = max(float(initial_pose.scale), 1e-6)
        translation_delta = max(scale * 0.15, 1e-5)
        axis_direction_delta = 0.20
        lower = initial_parameters - np.asarray(
            (translation_delta, translation_delta, translation_delta,
             axis_direction_delta, axis_direction_delta, axis_direction_delta),
            dtype=np.float64,
        )
        upper = initial_parameters + np.asarray(
            (translation_delta, translation_delta, translation_delta,
             axis_direction_delta, axis_direction_delta, axis_direction_delta),
            dtype=np.float64,
        )

        def pose_from(parameters: np.ndarray) -> ModelAlignment:
            direction = np.asarray(parameters[3:6], dtype=np.float64)
            direction_norm = float(np.linalg.norm(direction))
            if not math.isfinite(direction_norm) or direction_norm <= 1e-9:
                raise ValueError(f"degenerate refined axis direction for {sweep_name}")
            return _alignment_from_axis(
                parameters[:3], direction / direction_norm * scale
            )

        def residual(parameters: np.ndarray) -> np.ndarray:
            pose = pose_from(parameters)
            landmark_values: list[float] = []
            silhouette_values: list[float] = []
            width_values: list[float] = []
            axis_values: list[float] = []
            for obs in sweep_observations:
                camera = camera_by_index[obs.selected_index]
                normalizer = object_heights[obs.selected_index]
                for key in landmark_keys:
                    point = _axial_landmark_center(obs, key)
                    level = common_levels.get(key)
                    if not point or point[1] == "low" or level is None:
                        continue
                    observed_pair = _paired_landmark_points(obs, key)
                    if observed_pair is not None:
                        predicted_pair = _projected_profile_lateral_endpoints(
                            camera,
                            pose,
                            float(level),
                            float(profile(level)),
                        )
                        if predicted_pair is None:
                            landmark_values.extend((1.0, 1.0, 1.0, 1.0))
                        else:
                            for predicted, observed in zip(
                                predicted_pair, observed_pair, strict=True
                            ):
                                landmark_values.extend(
                                    (predicted - observed) / normalizer
                                )
                    else:
                        projected = project_world_point(
                            camera,
                            _model_to_world(
                                pose,
                                np.asarray(((0.0, 0.0, float(level)),)),
                            ),
                        )
                        if projected is None:
                            landmark_values.extend((1.0, 1.0))
                        else:
                            landmark_values.extend(
                                (np.asarray(projected) - np.asarray(point[0]))
                                / normalizer
                            )

                visible_axis_targets = (
                    (
                        obs.axis_top_xy[1] > 1.0,
                        _projected_profile_axial_endpoint(
                            camera,
                            pose,
                            1.0,
                            float(profile(1.0)),
                            toward_top=True,
                        ),
                        obs.axis_top_xy,
                    ),
                    (
                        obs.axis_bottom_xy[1] < camera.image_size[1] - 2.0,
                        _projected_profile_axial_endpoint(
                            camera,
                            pose,
                            0.0,
                            float(profile(0.0)),
                            toward_top=False,
                        ),
                        obs.axis_bottom_xy,
                    ),
                )
                for included, predicted_endpoint, target in visible_axis_targets:
                    if not included:
                        continue
                    if predicted_endpoint is None:
                        silhouette_values.extend((1.0, 1.0))
                    else:
                        silhouette_values.extend(
                            (np.asarray(predicted_endpoint) - np.asarray(target))
                            / normalizer
                        )

                for level in sample_levels:
                    basis = _projected_axis_basis(camera, pose, level)
                    endpoints = (
                        _lateral_endpoints_from_points(
                            boundaries[obs.selected_index],
                            basis[0],
                            basis[1],
                            basis[2],
                            normalizer,
                        )
                        if basis is not None
                        else None
                    )
                    if basis is None or endpoints is None:
                        silhouette_values.extend((1.0, 1.0, 1.0, 1.0))
                        width_values.append(1.0)
                        continue
                    center, _, lateral = basis
                    midpoint = (endpoints[0] + endpoints[1]) * 0.5
                    silhouette_values.extend((np.asarray(center) - midpoint) / normalizer)
                    observed_width = abs(float((endpoints[1] - endpoints[0]) @ lateral)) * 0.5
                    radius = float(profile(level))
                    predicted_width = _projected_half_width(
                        camera, pose, level, radius, center, lateral
                    )
                    width_values.append(
                        (
                            (predicted_width - observed_width) / normalizer
                            if predicted_width is not None
                            else 1.0
                        )
                    )
                    projected_ring = project_world_points(
                        camera,
                        _model_to_world(
                            pose,
                            _circle_world_points(radius, level, angular_samples=32),
                        ),
                    )
                    if projected_ring is None:
                        silhouette_values.extend((1.0, 1.0))
                    else:
                        offsets = (
                            projected_ring - np.asarray(center, dtype=np.float64)
                        ) @ np.asarray(lateral, dtype=np.float64)
                        predicted_endpoints = (
                            projected_ring[int(np.argmin(offsets))],
                            projected_ring[int(np.argmax(offsets))],
                        )
                        for predicted_endpoint in predicted_endpoints:
                            boundary_distance = _sample_distance_field(
                                boundary_distance_fields[obs.selected_index],
                                predicted_endpoint,
                            )
                            silhouette_values.append(
                                min(boundary_distance / normalizer, 1.0)
                                if math.isfinite(boundary_distance)
                                else 1.0
                            )

                normal, offset = axis_planes[obs.selected_index]
                rotation = np.asarray(pose.rotation_matrix, dtype=np.float64)
                axis_vector = rotation[:, 2] * pose.scale
                axis_scale = max(float(np.linalg.norm(axis_vector)), 1e-9)
                axis_values.append(float(normal @ (axis_vector / axis_scale)))
                axis_values.append(
                    float((normal @ np.asarray(pose.translation) - offset) / axis_scale)
                )

            groups = (
                (landmark_values, 1.5),
                (silhouette_values, 1.0),
                (width_values, 1.0),
                (
                    axis_values
                    + [float(np.linalg.norm(parameters[3:6]) - 1.0)],
                    0.5,
                ),
            )
            reference_count = max(len(landmark_values), 1)
            weighted_groups: list[np.ndarray] = []
            for values, weight in groups:
                if not values:
                    continue
                multiplier = (weight / 1.5) * math.sqrt(reference_count / len(values))
                weighted_groups.append(np.asarray(values, dtype=np.float64) * multiplier)
            return np.concatenate(weighted_groups)

        initial_residual = residual(initial_parameters)
        solution = least_squares(
            residual,
            initial_parameters,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.0125,
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
            max_nfev=120,
        )
        final_residual = residual(solution.x)
        initial_rms = float(np.sqrt(np.mean(np.square(initial_residual))))
        final_rms = float(np.sqrt(np.mean(np.square(final_residual))))
        improved = bool(np.isfinite(final_rms) and final_rms <= initial_rms + 1e-12)
        if improved:
            refined[sweep_name] = pose_from(solution.x)
        summaries[sweep_name] = {
            "status": "refined" if improved else "kept_initial",
            "view_count": len(sweep_observations),
            "converged": bool(solution.success),
            "nfev": int(solution.nfev),
            "initial_weighted_rms": initial_rms,
            "final_weighted_rms": final_rms,
            "bounded_translation_fraction": 0.15,
            "bounded_axis_direction_component_absolute": 0.20,
            "physical_scale_fixed": scale,
            "free_per_view_transforms": False,
            "cameras_refitted": False,
        }

    reliable_scales = [refined[name].scale for name in RELIABLE_CAPTURE_SWEEPS]
    return refined, {
        "method": "one_pass_profile_conditioned_shared_sweep_pose_refinement_fixed_scale",
        "sample_levels": list(sample_levels),
        "sweeps": summaries,
        "shared_reliable_physical_scale": reliable_scales[0],
        "physical_scale_invariant": bool(
            max(reliable_scales) - min(reliable_scales) <= 1e-12
        ),
        "free_per_view_transforms": False,
        "cameras_refitted": False,
    }


def refine_v2_camera_translations(
    observations: Sequence[FitObservation],
    cameras: Sequence[CameraReference],
    sweep_alignments: Mapping[str, ModelAlignment],
    profile_sections: Sequence[tuple[float, float]],
    common_levels: Mapping[str, float],
) -> tuple[tuple[CameraReference, ...], dict[str, Any]]:
    """Boundedly refine V2 camera translations while preserving Step-13 rotations/intrinsics."""

    by_index = {camera.selected_index: camera for camera in cameras}
    profile = _profile_interpolator(profile_sections)
    sample_levels = tuple(float(value) for value in np.linspace(0.08, 0.92, 11))
    landmark_keys = (
        "finial_top", "finial_bottom", "lid_max", "neck_top", "neck_base",
        "globe_top_transition", "globe_max", "globe_bottom_transition",
        "bowl_rim", "bowl_bottom_transition", "pedestal_waist", "foot",
    )
    refined = dict(by_index)
    summaries: dict[str, Any] = {}

    for obs in observations:
        sweep_name = capture_sweep(obs.selected_index)
        if sweep_name not in RELIABLE_CAPTURE_SWEEPS:
            continue
        camera = by_index[obs.selected_index]
        pose = sweep_alignments[sweep_name]
        boundary = _mask_boundary_points(obs.mask)
        distance_field = _mask_boundary_distance_field(obs.mask)
        normalizer = _object_image_height(obs.mask)
        initial = np.asarray(camera.cam_from_world_translation, dtype=np.float64)
        delta = max(float(pose.scale) * 0.08, 1e-5)
        lower = initial - delta
        upper = initial + delta

        def camera_from(parameters: np.ndarray) -> CameraReference:
            return replace(
                camera,
                cam_from_world_translation=tuple(float(value) for value in parameters),
            )

        def residual(parameters: np.ndarray) -> np.ndarray:
            trial = camera_from(parameters)
            landmark_values: list[float] = []
            silhouette_values: list[float] = []
            width_values: list[float] = []

            for key in landmark_keys:
                point = _axial_landmark_center(obs, key)
                level = common_levels.get(key)
                if point is None or point[1] == "low" or level is None:
                    continue
                observed_pair = _paired_landmark_points(obs, key)
                if observed_pair is not None:
                    predicted_pair = _projected_profile_lateral_endpoints(
                        trial,
                        pose,
                        float(level),
                        float(profile(level)),
                    )
                    if predicted_pair is None:
                        landmark_values.extend((1.0, 1.0, 1.0, 1.0))
                    else:
                        for predicted, observed in zip(
                            predicted_pair, observed_pair, strict=True
                        ):
                            landmark_values.extend(
                                (predicted - observed) / normalizer
                            )
                else:
                    projected = project_world_point(
                        trial,
                        _model_to_world(
                            pose,
                            np.asarray(((0.0, 0.0, float(level)),)),
                        ),
                    )
                    if projected is None:
                        landmark_values.extend((1.0, 1.0))
                    else:
                        landmark_values.extend(
                            (np.asarray(projected) - np.asarray(point[0]))
                            / normalizer
                        )

            for included, predicted_endpoint, target in (
                (
                    obs.axis_top_xy[1] > 1.0,
                    _projected_profile_axial_endpoint(
                        trial,
                        pose,
                        1.0,
                        float(profile(1.0)),
                        toward_top=True,
                    ),
                    obs.axis_top_xy,
                ),
                (
                    obs.axis_bottom_xy[1] < trial.image_size[1] - 2.0,
                    _projected_profile_axial_endpoint(
                        trial,
                        pose,
                        0.0,
                        float(profile(0.0)),
                        toward_top=False,
                    ),
                    obs.axis_bottom_xy,
                ),
            ):
                if not included:
                    continue
                if predicted_endpoint is None:
                    silhouette_values.extend((1.0, 1.0))
                else:
                    silhouette_values.extend(
                        (np.asarray(predicted_endpoint) - np.asarray(target))
                        / normalizer
                    )

            for level in sample_levels:
                basis = _projected_axis_basis(trial, pose, level)
                if basis is None:
                    silhouette_values.extend((1.0, 1.0, 1.0, 1.0))
                    width_values.append(1.0)
                    continue
                center, axis, lateral = basis
                observed_endpoints = _lateral_endpoints_from_points(
                    boundary, center, axis, lateral, normalizer
                )
                if observed_endpoints is None:
                    silhouette_values.extend((1.0, 1.0, 1.0, 1.0))
                    width_values.append(1.0)
                    continue
                midpoint = (observed_endpoints[0] + observed_endpoints[1]) * 0.5
                silhouette_values.extend((np.asarray(center) - midpoint) / normalizer)
                observed_half_width = abs(
                    float((observed_endpoints[1] - observed_endpoints[0]) @ lateral)
                ) * 0.5
                radius = float(profile(level))
                predicted_half_width = _projected_half_width(
                    trial, pose, level, radius, center, lateral
                )
                width_values.append(
                    (predicted_half_width - observed_half_width) / normalizer
                    if predicted_half_width is not None
                    else 1.0
                )
                ring = project_world_points(
                    trial,
                    _model_to_world(
                        pose, _circle_world_points(radius, level, angular_samples=32)
                    ),
                )
                if ring is None:
                    silhouette_values.extend((1.0, 1.0))
                else:
                    offsets = (ring - np.asarray(center)) @ np.asarray(lateral)
                    for endpoint in (
                        ring[int(np.argmin(offsets))],
                        ring[int(np.argmax(offsets))],
                    ):
                        distance = _sample_distance_field(distance_field, endpoint)
                        silhouette_values.append(
                            min(distance / normalizer, 1.0)
                            if math.isfinite(distance)
                            else 1.0
                        )

            groups = (
                (landmark_values, 1.5),
                (silhouette_values, 1.0),
                (width_values, 1.0),
            )
            reference_count = max(len(landmark_values), 1)
            weighted_groups: list[np.ndarray] = []
            for values, weight in groups:
                if values:
                    multiplier = (weight / 1.5) * math.sqrt(reference_count / len(values))
                    weighted_groups.append(np.asarray(values, dtype=np.float64) * multiplier)
            weighted_groups.append(0.20 * (parameters - initial) / delta)
            return np.concatenate(weighted_groups)

        initial_residual = residual(initial)
        solution = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.0125,
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
            max_nfev=80,
        )
        final_residual = residual(solution.x)
        initial_rms = float(np.sqrt(np.mean(np.square(initial_residual))))
        final_rms = float(np.sqrt(np.mean(np.square(final_residual))))
        use_solution = bool(np.isfinite(final_rms) and final_rms <= initial_rms + 1e-12)
        selected = solution.x if use_solution else initial
        refined[obs.selected_index] = camera_from(selected)
        translation_delta = float(np.linalg.norm(selected - initial))
        summaries[str(obs.selected_index)] = {
            "status": "refined" if use_solution else "kept_initial",
            "converged": bool(solution.success),
            "nfev": int(solution.nfev),
            "initial_weighted_rms": initial_rms,
            "final_weighted_rms": final_rms,
            "translation_delta_world": translation_delta,
            "translation_delta_fraction_of_object_height": translation_delta / max(pose.scale, 1e-9),
            "translation_component_bound_fraction_of_object_height": 0.08,
            "rotation_refined": False,
            "intrinsics_refined": False,
        }

    ordered = tuple(refined[camera.selected_index] for camera in cameras)
    return ordered, {
        "method": "bounded_per_camera_translation_refinement_against_shared_v2_geometry",
        "source_step13_files_modified": False,
        "rotation_refined": False,
        "intrinsics_refined": False,
        "free_per_view_object_transforms": False,
        "views": summaries,
    }


def refine_sweep_alignments_and_shared_scale_to_profile(
    observations: Sequence[FitObservation],
    cameras: Sequence[CameraReference],
    sweep_alignments: Mapping[str, ModelAlignment],
    profile_sections: Sequence[tuple[float, float]],
    common_levels: Mapping[str, float],
) -> tuple[dict[str, ModelAlignment], dict[str, Any]]:
    """Jointly refine three sweep poses and their one exact physical scale."""

    reliable = tuple(
        obs for obs in observations if obs.view_category != "top_down_rim"
    )
    if len(reliable) < 8:
        raise ValueError("joint shared-scale pose refinement requires eight reliable views")
    camera_by_index = {camera.selected_index: camera for camera in cameras}
    profile = _profile_interpolator(profile_sections)
    sample_levels = tuple(float(value) for value in np.linspace(0.08, 0.92, 13))
    landmark_keys = (
        "finial_top",
        "finial_bottom",
        "lid_max",
        "neck_top",
        "neck_base",
        "globe_top_transition",
        "globe_max",
        "globe_bottom_transition",
        "bowl_rim",
        "bowl_bottom_transition",
        "pedestal_waist",
        "foot",
    )
    boundaries = {
        obs.selected_index: _mask_boundary_points(obs.mask) for obs in reliable
    }
    boundary_distance_fields = {
        obs.selected_index: _mask_boundary_distance_field(obs.mask) for obs in reliable
    }
    object_heights = {
        obs.selected_index: _object_image_height(obs.mask) for obs in reliable
    }
    axis_planes = {
        obs.selected_index: mask_axis_plane(obs, camera_by_index[obs.selected_index])[:2]
        for obs in reliable
    }
    names = tuple(
        name
        for name in RELIABLE_CAPTURE_SWEEPS
        if any(capture_sweep(obs.selected_index) == name for obs in reliable)
    )
    if names != RELIABLE_CAPTURE_SWEEPS:
        raise ValueError("all three reliable capture sweeps are required")
    initial_scale = float(np.median([sweep_alignments[name].scale for name in names]))
    initial_values: list[float] = []
    lower_values: list[float] = []
    upper_values: list[float] = []
    for name in names:
        pose = sweep_alignments[name]
        direction = np.asarray(pose.rotation_matrix, dtype=np.float64)[:, 2]
        translation = np.asarray(pose.translation, dtype=np.float64)
        translation_delta = max(initial_scale * 0.20, 1e-5)
        initial_values.extend((*translation, *direction))
        lower_values.extend(
            (*(
                translation - translation_delta
            ), *(direction - 0.20))
        )
        upper_values.extend(
            (*(
                translation + translation_delta
            ), *(direction + 0.20))
        )
    initial_values.append(math.log(initial_scale))
    lower_values.append(math.log(initial_scale * 0.75))
    upper_values.append(math.log(initial_scale * 1.25))
    initial = np.asarray(initial_values, dtype=np.float64)
    lower = np.asarray(lower_values, dtype=np.float64)
    upper = np.asarray(upper_values, dtype=np.float64)

    def unpack(parameters: np.ndarray) -> dict[str, ModelAlignment]:
        shared_scale = float(math.exp(parameters[-1]))
        poses = dict(sweep_alignments)
        for index, name in enumerate(names):
            start = index * 6
            direction = np.asarray(parameters[start + 3:start + 6], dtype=np.float64)
            direction_norm = float(np.linalg.norm(direction))
            if not math.isfinite(direction_norm) or direction_norm <= 1e-9:
                raise ValueError(f"degenerate joint axis direction for {name}")
            poses[name] = _alignment_from_axis(
                parameters[start:start + 3],
                direction / direction_norm * shared_scale,
            )
        return poses

    def residual(parameters: np.ndarray) -> np.ndarray:
        poses = unpack(parameters)
        landmark_values: list[float] = []
        silhouette_values: list[float] = []
        width_values: list[float] = []
        axis_values: list[float] = []
        for obs in reliable:
            camera = camera_by_index[obs.selected_index]
            pose = poses[capture_sweep(obs.selected_index)]
            normalizer = object_heights[obs.selected_index]
            for key in landmark_keys:
                point = _axial_landmark_center(obs, key)
                level = common_levels.get(key)
                if not point or point[1] == "low" or level is None:
                    continue
                observed_pair = _paired_landmark_points(obs, key)
                if observed_pair is not None:
                    predicted_pair = _projected_profile_lateral_endpoints(
                        camera,
                        pose,
                        float(level),
                        float(profile(level)),
                    )
                    if predicted_pair is None:
                        landmark_values.extend((1.0, 1.0, 1.0, 1.0))
                    else:
                        for predicted, observed in zip(
                            predicted_pair, observed_pair, strict=True
                        ):
                            landmark_values.extend(
                                (predicted - observed) / normalizer
                            )
                else:
                    projected = project_world_point(
                        camera,
                        _model_to_world(
                            pose,
                            np.asarray(((0.0, 0.0, float(level)),)),
                        ),
                    )
                    landmark_values.extend(
                        (
                            np.asarray(
                                projected
                                if projected is not None
                                else (-10000, -10000)
                            )
                            - point[0]
                        )
                        / normalizer
                    )

            visible_axis_targets = (
                (
                    obs.axis_top_xy[1] > 1.0,
                    _projected_profile_axial_endpoint(
                        camera,
                        pose,
                        1.0,
                        float(profile(1.0)),
                        toward_top=True,
                    ),
                    obs.axis_top_xy,
                ),
                (
                    obs.axis_bottom_xy[1] < camera.image_size[1] - 2.0,
                    _projected_profile_axial_endpoint(
                        camera,
                        pose,
                        0.0,
                        float(profile(0.0)),
                        toward_top=False,
                    ),
                    obs.axis_bottom_xy,
                ),
            )
            for included, predicted_endpoint, target in visible_axis_targets:
                if not included:
                    continue
                if predicted_endpoint is None:
                    silhouette_values.extend((1.0, 1.0))
                else:
                    silhouette_values.extend(
                        (np.asarray(predicted_endpoint) - np.asarray(target))
                        / normalizer
                    )

            for level in sample_levels:
                basis = _projected_axis_basis(camera, pose, level)
                endpoints = (
                    _lateral_endpoints_from_points(
                        boundaries[obs.selected_index],
                        basis[0],
                        basis[1],
                        basis[2],
                        normalizer,
                    )
                    if basis is not None
                    else None
                )
                if basis is None or endpoints is None:
                    silhouette_values.extend((1.0, 1.0, 1.0, 1.0))
                    width_values.append(1.0)
                    continue
                center, _, lateral = basis
                midpoint = (endpoints[0] + endpoints[1]) * 0.5
                silhouette_values.extend((np.asarray(center) - midpoint) / normalizer)
                observed_width = abs(float((endpoints[1] - endpoints[0]) @ lateral)) * 0.5
                radius = float(profile(level))
                predicted_width = _projected_half_width(
                    camera, pose, level, radius, center, lateral
                )
                width_values.append(
                    (predicted_width - observed_width) / normalizer
                    if predicted_width is not None
                    else 1.0
                )
                projected_ring = project_world_points(
                    camera,
                    _model_to_world(
                        pose, _circle_world_points(radius, level, angular_samples=32)
                    ),
                )
                if projected_ring is None:
                    silhouette_values.extend((1.0, 1.0))
                else:
                    offsets = (projected_ring - np.asarray(center)) @ np.asarray(lateral)
                    for endpoint in (
                        projected_ring[int(np.argmin(offsets))],
                        projected_ring[int(np.argmax(offsets))],
                    ):
                        distance = _sample_distance_field(
                            boundary_distance_fields[obs.selected_index], endpoint
                        )
                        silhouette_values.append(
                            min(distance / normalizer, 1.0)
                            if math.isfinite(distance)
                            else 1.0
                        )

            normal, offset = axis_planes[obs.selected_index]
            rotation = np.asarray(pose.rotation_matrix, dtype=np.float64)
            unit_axis = rotation[:, 2]
            axis_values.append(float(normal @ unit_axis))
            axis_values.append(
                float((normal @ np.asarray(pose.translation) - offset) / pose.scale)
            )

        for index, _name in enumerate(names):
            direction = parameters[index * 6 + 3:index * 6 + 6]
            axis_values.append(float(np.linalg.norm(direction) - 1.0))
        axis_values.append(0.05 * float(parameters[-1] - math.log(initial_scale)))
        groups = (
            (landmark_values, 1.5),
            (silhouette_values, 1.0),
            (width_values, 1.0),
            (axis_values, 0.5),
        )
        reference_count = max(len(landmark_values), 1)
        weighted_groups: list[np.ndarray] = []
        for values, weight in groups:
            if values:
                multiplier = (weight / 1.5) * math.sqrt(reference_count / len(values))
                weighted_groups.append(np.asarray(values, dtype=np.float64) * multiplier)
        return np.concatenate(weighted_groups)

    initial_residual = residual(initial)
    solution = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.0125,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
        max_nfev=180,
    )
    final_residual = residual(solution.x)
    initial_rms = float(np.sqrt(np.mean(np.square(initial_residual))))
    final_rms = float(np.sqrt(np.mean(np.square(final_residual))))
    use_solution = bool(np.isfinite(final_rms) and final_rms <= initial_rms + 1e-12)
    selected = solution.x if use_solution else initial
    refined = unpack(selected)
    reliable_scales = [refined[name].scale for name in names]
    return refined, {
        "method": "joint_three_sweep_pose_and_one_shared_physical_scale_profile_refinement",
        "converged": bool(solution.success),
        "nfev": int(solution.nfev),
        "initial_weighted_rms": initial_rms,
        "final_weighted_rms": final_rms,
        "used_solution": use_solution,
        "initial_shared_physical_scale": initial_scale,
        "final_shared_physical_scale": reliable_scales[0],
        "physical_scale_invariant": bool(max(reliable_scales) - min(reliable_scales) <= 1e-12),
        "scale_bound_fraction": 0.25,
        "translation_bound_fraction": 0.20,
        "axis_direction_component_bound": 0.20,
        "free_per_view_transforms": False,
        "cameras_refitted": False,
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return float("inf")
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def fit_metric_gate(
    metrics: Sequence[ViewFitMetrics],
) -> bool:
    """Apply the frozen V2 whole-object geometry gate without visual veto."""

    reliable = tuple(
        metric for metric in metrics if metric.fit_included and metric.camera_source in GATE_CAMERA_SOURCES
    )
    if len(reliable) < 8:
        return False
    aggregate = _aggregate_metrics(metrics)
    return (
        aggregate["median_silhouette_iou"] >= METRIC_THRESHOLDS["median_silhouette_iou"]
        and aggregate["minimum_reliable_silhouette_iou"]
        >= METRIC_THRESHOLDS["minimum_reliable_silhouette_iou"]
        and aggregate["median_landmark_error_fraction"]
        <= METRIC_THRESHOLDS["median_landmark_error_fraction"]
        and aggregate["p95_landmark_error_fraction"]
        <= METRIC_THRESHOLDS["p95_landmark_error_fraction"]
    )


def _normalize_visual_review(
    review: Mapping[str, str] | None,
) -> Mapping[str, str]:
    values = dict(review or {})
    unknown = set(values) - set(VISUAL_REVIEW_COMPONENTS)
    invalid = {
        name: status for name, status in values.items() if status not in {"pass", "fail", "pending"}
    }
    if unknown or invalid:
        raise ValueError("invalid visual component review")
    return {
        component: values.get(component, "pending")
        for component in VISUAL_REVIEW_COMPONENTS
    }


def acceptance_decision(
    metrics_passed: bool,
    visual_component_review: Mapping[str, str] | None,
) -> bool:
    """Require both numeric metrics and every human visual review to pass."""

    review = _normalize_visual_review(visual_component_review)
    return bool(metrics_passed and all(review[name] == "pass" for name in review))


def _observed_z_from_landmark(
    point: Sequence[float],
    axis_top: Sequence[float],
    axis_bottom: Sequence[float],
) -> float:
    top = np.asarray(axis_top, dtype=np.float64)
    bottom = np.asarray(axis_bottom, dtype=np.float64)
    direction = bottom - top
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-12:
        return 0.5
    fraction = float(np.dot(np.asarray(point) - top, direction) / denominator)
    return 1.0 - min(max(fraction, 0.0), 1.0)


def _landmark_level_key(name: str) -> str:
    for suffix in ("_left", "_right"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _object_image_height(mask: np.ndarray) -> float:
    rows = np.flatnonzero(np.any(mask > 0, axis=1))
    if rows.size < 2:
        return 1.0
    return float(rows[-1] - rows[0] + 1)


def _landmark_records(
    observation: FitObservation,
    camera: CameraReference,
    alignment: ModelAlignment,
    predicted_mask: np.ndarray,
    common_levels: Mapping[str, float],
    profiles: Mapping[str, FittedComponentProfile] | None = None,
) -> tuple[LandmarkFitRecord, ...]:
    """Reproject named physical landmarks and normalize by object height.

    Lateral landmarks are semantic component measurements.  Measuring a globe
    landmark against the whole assembly can incorrectly select the wider bowl
    rim where the two projected parts overlap, so use the corresponding rendered
    component whenever fitted profiles are available.
    """

    normalizer = _object_image_height(observation.mask)
    records: list[LandmarkFitRecord] = []
    component_profiles = {
        "lid_max": ("lid",),
        "lid_lower": ("lid",),
        "neck_top": ("neck_outer",),
        "neck_base": ("neck_outer",),
        "globe_max": ("globe",),
        "bowl_rim": ("bowl_outer",),
        "bowl_bottom_transition": ("pedestal", "bowl_outer"),
        "pedestal_waist": ("pedestal",),
        "foot": ("pedestal",),
    }
    component_mask_cache: dict[tuple[str, ...], np.ndarray] = {}

    def lateral_mask(level_key: str) -> np.ndarray:
        profile_names = component_profiles.get(level_key)
        if profiles is None or profile_names is None:
            return predicted_mask
        missing = [name for name in profile_names if name not in profiles]
        if missing:
            return predicted_mask
        cached = component_mask_cache.get(profile_names)
        if cached is not None:
            return cached
        canvas = np.zeros_like(predicted_mask)
        for profile_name in profile_names:
            component = render_profile_silhouette(
                profiles[profile_name].sections,
                camera,
                alignment,
                camera.image_size,
            )
            canvas = cv2.bitwise_or(canvas, component)
        component_mask_cache[profile_names] = canvas
        return canvas

    for landmark in observation.landmarks:
        name = str(landmark["name"])
        confidence = str(landmark.get("confidence", "medium"))
        observed = np.asarray((float(landmark["x"]), float(landmark["y"])))
        level_key = _landmark_level_key(name)
        level = common_levels.get(level_key)
        if level is None:
            level = _observed_z_from_landmark(
                observed, observation.axis_top_xy, observation.axis_bottom_xy
            )
        projected_center = project_world_point(
            camera,
            _model_to_world(alignment, np.asarray(((0.0, 0.0, float(level)),))),
        )
        if projected_center is None:
            predicted = np.asarray((observation.axis_top_xy[0], observed[1]))
        elif name.endswith("_left") or name.endswith("_right"):
            basis = _projected_axis_basis(camera, alignment, float(level))
            endpoints = (
                _mask_lateral_endpoints(
                    lateral_mask(level_key), basis[0], basis[1], basis[2]
                )
                if basis is not None
                else None
            )
            if endpoints is None:
                predicted = np.asarray(projected_center)
            else:
                predicted = np.asarray(
                    endpoints[0] if name.endswith("_left") else endpoints[1],
                    dtype=np.float64,
                )
        else:
            predicted = np.asarray(projected_center)
        error = float(np.linalg.norm(predicted - observed) / normalizer)
        records.append(
            LandmarkFitRecord(
                name=name,
                observed_x=float(observed[0]),
                observed_y=float(observed[1]),
                predicted_x=float(predicted[0]),
                predicted_y=float(predicted[1]),
                error_fraction=error,
                confidence=confidence,
                gate_included=confidence != "low",
            )
        )
    return tuple(records)


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        raise ValueError("IoU masks must have matching shapes")
    left = first > 0
    right = second > 0
    union = int(np.count_nonzero(left | right))
    return float(np.count_nonzero(left & right) / union) if union else 0.0


def _component_ious_for_view(
    project_root: Path,
    observation: FitObservation,
    camera: CameraReference,
    alignment: ModelAlignment,
    profiles: Mapping[str, FittedComponentProfile],
) -> dict[str, float]:
    """Compare actual projected V2 components to reviewed component annotations.

    Component diagnostics must follow the projected model axis.  Horizontal-row
    slicing is invalid for tilted cameras, so render the named radial components
    directly and compose the same physical groups used by the reviewed masks.
    """

    component_names = (
        "main_vessel",
        "receiving_bowl_pedestal",
        "lid_finial",
        "neck",
        "globe",
    )
    target_size = camera.image_size
    component_dir = project_root.resolve() / V2_RELATIVE_ROOT / "evidence" / "component_masks"
    observed: dict[str, np.ndarray] = {}
    for name in component_names:
        path = component_dir / f"{observation.selected_index:03d}_{name}.png"
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        observed[name] = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    if not observed:
        return {}

    required = ("pedestal", "bowl_outer", "globe", "shoulder", "neck_outer", "lid", "finial")
    missing = [name for name in required if name not in profiles]
    if missing:
        raise ValueError("missing profiles for component IoU: " + ", ".join(missing))

    rendered = {
        name: render_profile_silhouette(
            profiles[name].sections, camera, alignment, target_size
        )
        for name in required
    }

    def union(*names: str) -> np.ndarray:
        canvas = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
        for name in names:
            canvas = cv2.bitwise_or(canvas, rendered[name])
        return canvas

    predicted_components = {
        "receiving_bowl_pedestal": union("pedestal", "bowl_outer"),
        "main_vessel": union("globe", "shoulder", "neck_outer", "lid", "finial"),
        "lid_finial": union("lid", "finial"),
        "neck": rendered["neck_outer"],
        "globe": rendered["globe"],
    }
    return {
        name: _iou(observed[name], predicted)
        for name, predicted in predicted_components.items()
        if name in observed
    }


def _aggregate_metrics(
    metrics: Sequence[ViewFitMetrics],
) -> Mapping[str, float]:
    reliable = [
        metric
        for metric in metrics
        if metric.fit_included and metric.camera_source in GATE_CAMERA_SOURCES
    ]
    landmark_errors = [
        record.error_fraction
        for metric in reliable
        for record in metric.landmark_records
        if record.gate_included
    ]
    if not landmark_errors:
        # Preserve compatibility with synthetic metric-only fixtures.
        landmark_errors = [metric.landmark_median_error_fraction for metric in reliable]
    return {
        "median_silhouette_iou": float(
            np.median([metric.silhouette_iou for metric in reliable])
        )
        if reliable
        else float("nan"),
        "minimum_reliable_silhouette_iou": min(
            (metric.silhouette_iou for metric in reliable), default=float("nan")
        ),
        "median_landmark_error_fraction": (
            float(np.median(landmark_errors)) if landmark_errors else float("nan")
        ),
        "p95_landmark_error_fraction": _percentile(landmark_errors, 95.0),
        "reliable_view_count": float(len(reliable)),
        "gate_landmark_count": float(len(landmark_errors)),
    }


def _load_real_observations(
    project_root: Path,
) -> tuple[tuple[FitObservation, ...], tuple[CameraReference, ...]]:
    from final_reference_evidence import (
        load_reviewed_reference_views,
        select_canonical_geometry_views,
    )

    root = project_root.resolve()
    views = load_reviewed_reference_views(root)
    canonical = select_canonical_geometry_views(views)
    cameras = load_step13_cameras(root)
    cameras_by_index = {camera.selected_index: camera for camera in cameras}
    with (root / LANDMARK_EVIDENCE).open("r", encoding="utf-8") as handle:
        landmark_payload = json.load(handle)
    landmark_views = {
        int(record["selected_index"]): record
        for record in landmark_payload["views"]
    }
    if set(landmark_views) != {int(row["selected_index"]) for row in canonical}:
        raise ValueError("canonical landmarks and camera views disagree")

    observations: list[FitObservation] = []
    fitted_cameras: list[CameraReference] = []
    maximum_dimension = 384
    for row in canonical:
        selected_index = int(row["selected_index"])
        view = next(item for item in views if item.selected_index == selected_index)
        camera = cameras_by_index[selected_index]
        record = landmark_views[selected_index]
        geometry_mask_value = record.get("geometry_mask_path")
        mask_path = (
            root / str(geometry_mask_value)
            if geometry_mask_value
            else (
                view.reviewed_mask_path
                if view.reviewed_mask_path.is_absolute()
                else root / view.reviewed_mask_path
            )
        )
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"unreadable reviewed mask: {view.filename}")
        scale = maximum_dimension / max(mask.shape[:2])
        target_size = (
            max(2, int(round(mask.shape[1] * scale))),
            max(2, int(round(mask.shape[0] * scale))),
        )
        resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        record = landmark_views[selected_index]
        landmarks = tuple(
            {
                "name": str(landmark["name"]),
                "x": float(landmark["x"]) * target_size[0] / float(record["source_image_size"][0]),
                "y": float(landmark["y"]) * target_size[1] / float(record["source_image_size"][1]),
                "confidence": str(landmark.get("confidence", "medium")),
            }
            for landmark in record["landmarks"]
        )
        by_name = {str(landmark["name"]): landmark for landmark in landmarks}
        source_width, source_height = (
            int(record["source_image_size"][0]),
            int(record["source_image_size"][1]),
        )
        scale_x = target_size[0] / float(source_width)
        scale_y = target_size[1] / float(source_height)
        if not math.isclose(scale_x, scale_y, rel_tol=0.0, abs_tol=1e-3):
            raise ValueError("diagnostic resize changed camera aspect ratio")
        if camera.camera_model != "SIMPLE_RADIAL" or len(camera.camera_params) != 4:
            raise ValueError("unsupported resized Step 13 camera model")
        focal, principal_x, principal_y, distortion = camera.camera_params
        resized_camera = CameraReference(
            selected_index=camera.selected_index,
            filename=camera.filename,
            image_id=camera.image_id,
            camera_model=camera.camera_model,
            camera_params=(
                focal * scale_x,
                principal_x * scale_x,
                principal_y * scale_y,
                distortion,
            ),
            cam_from_world_rotation_xyzw=camera.cam_from_world_rotation_xyzw,
            cam_from_world_translation=camera.cam_from_world_translation,
            image_size=target_size,
        )
        fitted_cameras.append(resized_camera)
        observations.append(
            FitObservation(
                selected_index=selected_index,
                filename=view.filename,
                view_category=str(row["view_category"]),
                mask=resized,
                axis_top_xy=(
                    float(by_name["axis_top"]["x"]),
                    float(by_name["axis_top"]["y"]),
                ),
                axis_bottom_xy=(
                    float(by_name["axis_bottom"]["x"]),
                    float(by_name["axis_bottom"]["y"]),
                ),
                landmarks=landmarks,
                source_path=root / SELECTED_IMAGES / view.filename,
                reviewed_mask_path=mask_path,
            )
        )
    return tuple(observations), tuple(fitted_cameras)


def fit_vessel_model(
    project_root: Path,
    *,
    visual_component_review: Mapping[str, str] | None = None,
) -> FitResult:
    """Fit all canonical views and return an honest fail-closed V2 result."""

    observations, cameras = _load_real_observations(project_root)
    sweep_alignments, common_levels, pose_summary = fit_sweep_alignments(observations, cameras)
    fit_cameras = _apply_camera_center_normalization(
        cameras, pose_summary["camera_center_normalization"]
    )
    excluded_camera_views = set(pose_summary["excluded_camera_geometry_views"])
    fit_observations = tuple(
        observation
        for observation in observations
        if observation.selected_index not in excluded_camera_views
    )
    alignment = sweep_alignments["side_003_072"]
    profiles = fit_profiles_from_observations(
        fit_observations, fit_cameras, alignment, sweep_alignments=sweep_alignments
    )
    joint_passes: list[dict[str, Any]] = []
    for _ in range(2):
        sweep_alignments, joint_pass = refine_sweep_alignments_to_profile(
            fit_observations,
            fit_cameras,
            sweep_alignments,
            profiles["outer"].sections,
            common_levels,
        )
        joint_passes.append(joint_pass)
        alignment = sweep_alignments["side_003_072"]
        profiles = fit_profiles_from_observations(
            fit_observations,
            fit_cameras,
            alignment,
            sweep_alignments=sweep_alignments,
        )
    reliable_scales = [
        sweep_alignments[name].scale for name in RELIABLE_CAPTURE_SWEEPS
    ]
    joint_refinement = {
        "method": "two_pass_profile_conditioned_sweep_pose_refinement_after_step13_gauge_normalization",
        "pass_count": len(joint_passes),
        "passes": joint_passes,
        "physical_scale_invariant": bool(
            max(reliable_scales) - min(reliable_scales) <= 1e-12
        ),
        "shared_reliable_physical_scale": reliable_scales[0],
        "free_per_view_transforms": False,
        "cameras_refitted": False,
        "source_step13_files_modified": False,
    }
    fit_cameras, camera_refinement = refine_v2_camera_translations(
        fit_observations,
        fit_cameras,
        sweep_alignments,
        profiles["outer"].sections,
        common_levels,
    )
    profiles = fit_profiles_from_observations(
        fit_observations,
        fit_cameras,
        alignment,
        sweep_alignments=sweep_alignments,
    )
    sweep_alignments, post_camera_pose_refinement = refine_sweep_alignments_to_profile(
        fit_observations,
        fit_cameras,
        sweep_alignments,
        profiles["outer"].sections,
        common_levels,
    )
    alignment = sweep_alignments["side_003_072"]
    profiles = fit_profiles_from_observations(
        fit_observations,
        fit_cameras,
        alignment,
        sweep_alignments=sweep_alignments,
    )
    joint_refinement["post_camera_refinement_pose_pass"] = post_camera_pose_refinement
    outer = profiles["outer"]
    component_profiles = fit_independent_component_profiles(
        project_root,
        fit_observations,
        fit_cameras,
        sweep_alignments,
        common_levels,
    )
    component_profiles, profile_refinement = refine_component_profile_radii(
        project_root,
        fit_observations,
        fit_cameras,
        sweep_alignments,
        component_profiles,
    )
    metrics: list[ViewFitMetrics] = []
    camera_by_index = {camera.selected_index: camera for camera in fit_cameras}
    for observation in observations:
        camera = camera_by_index[observation.selected_index]
        pose = sweep_alignments[capture_sweep(observation.selected_index)]
        predicted = render_independent_assembly_silhouette(
            component_profiles, camera, pose, camera.image_size
        )
        records = _landmark_records(
            observation,
            camera,
            pose,
            predicted,
            common_levels,
            component_profiles,
        )
        errors = [record.error_fraction for record in records if record.gate_included]
        reliable_sweep = capture_sweep(observation.selected_index) in RELIABLE_CAPTURE_SWEEPS
        reliable = (
            reliable_sweep
            and observation.selected_index not in excluded_camera_views
        )
        camera_source = (
            "step13_v2_bounded_translation_refined" if reliable_sweep else "step13"
        )
        metrics.append(
            ViewFitMetrics(
                selected_index=observation.selected_index,
                filename=observation.filename,
                view_category=observation.view_category,
                silhouette_iou=_iou(observation.mask, predicted),
                landmark_median_error_fraction=float(np.median(errors)) if errors else float("inf"),
                landmark_max_error_fraction=max(errors, default=float("inf")),
                component_ious=_component_ious_for_view(
                    project_root,
                    observation,
                    camera,
                    pose,
                    component_profiles,
                ),
                camera_source=camera_source,
                fit_included=reliable,
                inclusion_reason=(
                    "reliable_step13_camera_v2_bounded_translation_refined"
                    if reliable
                    else (
                        "evaluation_only_camera_axis_dual_robust_outlier"
                        if observation.selected_index in excluded_camera_views
                        else "evaluation_only_top_down_axis_unreliable"
                    )
                ),
                source_path=observation.source_path,
                reviewed_mask_path=observation.reviewed_mask_path,
                landmark_records=records,
                predicted_mask=predicted,
            )
        )
    aggregate = _aggregate_metrics(tuple(metrics))
    metrics_passed = fit_metric_gate(tuple(metrics))
    review = _normalize_visual_review(visual_component_review)
    accepted = acceptance_decision(metrics_passed, review)
    return FitResult(
        alignment=alignment,
        profiles=tuple(component_profiles[name] for name in REQUIRED_PROFILES) + (outer,),
        metrics=tuple(metrics),
        metrics_passed=metrics_passed,
        visual_component_review=review,
        accepted=accepted,
        aggregate_metrics=aggregate,
        fit_summary={
            "alignment_method": "shared_vessel_geometry_three_reliable_sweep_poses_plus_evaluation_only_top",
            "pose_calibration": pose_summary,
            "camera_refinement": camera_refinement,
            "profile_conditioned_pose_refinement": joint_refinement,
            "joint_profile_radius_refinement": profile_refinement,
            "capture_axis_audit": audit_capture_axes(observations, cameras),
            "shared_landmark_levels": common_levels,
            "profile_method": "independent_named_component_profiles_with_joint_bounded_radius_refinement",
            "camera_source": "protected_step13_plus_v2_sweep_gauge_and_bounded_translation_refinement",
            "source_step13_files_modified": False,
            "outer_control_level_count": len(outer.sections),
            "independent_profile_names": list(REQUIRED_PROFILES),
            "component_mask_source": "v2_reviewed_component_annotations",
            "component_mask_method": "reviewed_whole_mask_plus_cv_seeded_landmark_split",
        },
        sweep_alignments=sweep_alignments,
    )


def _relative(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def fit_report_payload(
    result: FitResult,
    *,
    project_root: Path,
) -> dict[str, Any]:
    root = project_root.resolve()
    payload = {
        "schema_version": 1,
        "camera_source": result.fit_summary.get("camera_source", "step13"),
        "metrics_passed": result.metrics_passed,
        "aggregate_metrics": dict(result.aggregate_metrics),
        "metric_thresholds": dict(METRIC_THRESHOLDS),
        "visual_component_review": dict(result.visual_component_review),
        "accepted": result.accepted,
        "alignment": {
            "scale": result.alignment.scale,
            "rotation_matrix": result.alignment.rotation_matrix,
            "translation": result.alignment.translation,
        },
        "fit_summary": dict(result.fit_summary),
        "sweep_alignments": {name: asdict(alignment) for name, alignment in result.sweep_alignments.items()},
        "capture_sweep_intervals": CAPTURE_SWEEPS,
        "views": [
            {
                **{
                    key: value
                    for key, value in asdict(metric).items()
                    if key != "predicted_mask"
                },
                "source_path": _relative(metric.source_path, root),
                "reviewed_mask_path": _relative(metric.reviewed_mask_path, root),
                "diagnostic_directory": (
                    f"{V2_RELATIVE_ROOT.as_posix()}/diagnostics/"
                    f"10_cv_reference_fit/{metric.selected_index:03d}"
                ),
            }
            for metric in result.metrics
        ],
    }
    # Review only this exact candidate: a stale all-pass sidecar must not approve
    # new geometry, calibration, masks or residuals after a subsequent correction.
    identity = {key: payload[key] for key in ("alignment", "sweep_alignments", "views", "fit_summary")}
    identity["profiles"] = profiles_payload(result)
    payload["candidate_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return payload


def profiles_payload(result: FitResult) -> dict[str, Any]:
    by_name = {profile.name: profile for profile in result.profiles}
    missing = [name for name in REQUIRED_PROFILES if name not in by_name]
    if missing:
        raise ValueError("fit result is missing required profiles: " + ", ".join(missing))
    return {
        "coordinate_contract": {
            "axis": "+Z",
            "normalized_set_height": 1.0,
            "scale_status": "relative_no_physical_measurement",
        },
        "profiles": {
            name: [list(section) for section in by_name[name].sections]
            for name in REQUIRED_PROFILES
        },
        "profile_provenance": {
            name: {
                "measurement_status": by_name[name].measurement_status,
                "source_views": list(by_name[name].source_views),
            }
            for name in REQUIRED_PROFILES
        },
        "source_views": [
            metric.selected_index
            for metric in result.metrics
            if metric.camera_source == "step13"
        ],
        "fit_metrics": dict(result.aggregate_metrics),
    }


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write CV-fit diagnostic: {path}")


def write_fit_outputs(
    project_root: Path,
    result: FitResult,
) -> tuple[Path, Path]:
    """Write reports and six canonical diagnostic panels per fitted view."""

    root = project_root.resolve()
    v2_root = ensure_under_v2_root(root / V2_RELATIVE_ROOT, root / V2_RELATIVE_ROOT)
    outer = next(profile for profile in result.profiles if profile.name == "outer")
    camera_size_by_index = {
        metric.selected_index: metric.reviewed_mask_path for metric in result.metrics
    }
    if len(camera_size_by_index) != len(result.metrics):
        raise ValueError("duplicate diagnostic view index")
    for metric in result.metrics:
        if metric.reviewed_mask_path is None or metric.source_path is None:
            raise ValueError(f"missing diagnostic source path: {metric.selected_index}")
        source = (
            metric.source_path
            if metric.source_path.is_absolute()
            else root / metric.source_path
        )
        mask_path = (
            metric.reviewed_mask_path
            if metric.reviewed_mask_path.is_absolute()
            else root / metric.reviewed_mask_path
        )
        source_image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        observed = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if source_image is None or observed is None:
            raise ValueError(f"unreadable diagnostic source for {metric.filename}")
        size = (max(2, observed.shape[1] // 8), max(2, observed.shape[0] // 8))
        source_resized = cv2.resize(source_image, size, interpolation=cv2.INTER_AREA)
        observed_resized = cv2.resize(observed, size, interpolation=cv2.INTER_NEAREST)
        if metric.predicted_mask is None:
            raise ValueError(f"missing fitted silhouette for {metric.filename}")
        predicted = cv2.resize(
            metric.predicted_mask, size, interpolation=cv2.INTER_NEAREST
        )
        overlay = source_resized.copy()
        overlay[observed_resized > 0] = (0, 220, 0)
        overlay[predicted > 0] = (220, 0, 220)
        difference = cv2.absdiff(observed_resized, predicted)
        landmark_panel = source_resized.copy()
        for record in metric.landmark_records:
            observed_point = (
                int(round(record.observed_x * size[0] / max(observed.shape[1] - 1, 1))),
                int(round(record.observed_y * size[1] / max(observed.shape[0] - 1, 1))),
            )
            cv2.circle(landmark_panel, observed_point, 3, (80, 220, 120), -1)
        diagnostic_root = ensure_under_v2_root(
            v2_root
            / "diagnostics"
            / "10_cv_reference_fit"
            / f"{metric.selected_index:03d}",
            v2_root,
        )
        _write_image(diagnostic_root / "source.png", source_resized)
        _write_image(diagnostic_root / "mask.png", observed_resized)
        _write_image(diagnostic_root / "fit_silhouette.png", predicted)
        _write_image(diagnostic_root / "overlay.png", overlay)
        _write_image(diagnostic_root / "difference.png", difference)
        _write_image(diagnostic_root / "landmarks.png", landmark_panel)

    report_path = v2_root / "reports" / "final_cv_fit.json"
    profiles_path = v2_root / "reports" / "final_profiles.json"
    write_json_atomic(profiles_path, profiles_payload(result), v2_root)
    report = fit_report_payload(result, project_root=root)
    report["artifact_hashes"] = {
        "final_profiles_sha256": sha256_file(profiles_path)
    }
    write_json_atomic(report_path, report, v2_root)
    return report_path, profiles_path


def _parse_visual_review(value: str) -> dict[str, str]:
    if not value:
        return {}
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("visual review must be a JSON object")
    return {str(key): str(item) for key, item in payload.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--write-outputs", action="store_true")
    parser.add_argument(
        "--visual-review",
        default="",
        help='JSON object such as {"bowl":"pass","globe":"pass"}',
    )
    arguments = parser.parse_args()
    result = fit_vessel_model(
        arguments.project_root,
        visual_component_review=_parse_visual_review(arguments.visual_review),
    )
    if arguments.write_outputs:
        report_path, profiles_path = write_fit_outputs(arguments.project_root, result)
        print(json.dumps({
            "accepted": result.accepted,
            "metrics_passed": result.metrics_passed,
            "report": str(report_path),
            "profiles": str(profiles_path),
        }, sort_keys=True))
    else:
        print(json.dumps({
            "accepted": result.accepted,
            "metrics_passed": result.metrics_passed,
            "aggregate_metrics": result.aggregate_metrics,
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
