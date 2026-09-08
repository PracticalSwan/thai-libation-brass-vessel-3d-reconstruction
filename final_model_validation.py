"""Fail-closed validation helpers for the final V2 vessel workflow."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np


QA_VERDICTS = frozenset({"SHIP", "SHIP WITH NOTES", "NO-SHIP"})
VISUAL_RATINGS = frozenset({"Match", "Close", "Miss"})
REQUIRED_COMPONENTS = (
    "bowl_pedestal",
    "globe_shoulder",
    "neck",
    "lid_finial",
    "overall_identity",
)
REQUIRED_VISUAL_RATINGS = (
    "camera",
    "silhouette_proportions",
    "depth_construction",
    "hero_ornament",
    "materials",
)
TOPOLOGY_ZERO_COUNTS = (
    "non_manifold_edges",
    "loose_vertices",
    "loose_edges",
    "loose_faces",
    "inverted_faces",
    "self_intersections",
)
DEFAULT_BLENDER_NAME = re.compile(r"^(?:Cube|Sphere|Torus)(?:\.\d+)?$", re.IGNORECASE)
LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
REGISTERED_VIEW_COORDINATE_PATHS = frozenset(
    {"SIMPLE_RADIAL", "paired_undistorted_pinhole"}
)
REGISTERED_VIEW_CLASSIFICATIONS = frozenset(
    {"ok", "mask_failure", "camera_failure", "model_mismatch"}
)
SURFACE_EVIDENCE_CLASSES = frozenset(
    {
        "direct_multi_view",
        "reviewed_single_or_detail",
        "symmetry_repetition",
        "hidden_generic_fill",
    }
)
REQUIRED_SURFACE_COMPONENTS = (
    "bowl_pedestal",
    "globe_shoulder",
    "neck",
    "lid_finial",
)

MEDIAN_IOU_THRESHOLD = 0.90
MINIMUM_IOU_THRESHOLD = 0.84
MEDIAN_LANDMARK_THRESHOLD = 0.020
P95_LANDMARK_THRESHOLD = 0.040
EXPORT_IOU_THRESHOLD = 0.995
EXPORT_MEAN_DIFFERENCE_THRESHOLD = 0.030


@dataclass(frozen=True)
class FinalValidationResult:
    geometry_passed: bool
    ornament_passed: bool
    topology_passed: bool
    uv_bake_passed: bool
    texture_passed: bool
    visual_identity_passed: bool
    export_reimport_passed: bool
    qa_verdict: str
    accepted: bool


def aggregate_final_validation(
    *,
    geometry: bool,
    ornament: bool,
    topology: bool,
    uv_bake: bool,
    texture: bool,
    visual_identity: bool,
    export_reimport: bool,
    qa_verdict: str,
) -> FinalValidationResult:
    """Aggregate final gates; canonical promotion requires exact ``SHIP``."""

    if qa_verdict not in QA_VERDICTS:
        raise ValueError(
            f"unsupported QA verdict {qa_verdict!r}; expected one of {sorted(QA_VERDICTS)}"
        )

    result = FinalValidationResult(
        geometry_passed=geometry,
        ornament_passed=ornament,
        topology_passed=topology,
        uv_bake_passed=uv_bake,
        texture_passed=texture,
        visual_identity_passed=visual_identity,
        export_reimport_passed=export_reimport,
        qa_verdict=qa_verdict,
        accepted=False,
    )
    gate_fields = (
        "geometry_passed",
        "ornament_passed",
        "topology_passed",
        "uv_bake_passed",
        "texture_passed",
        "visual_identity_passed",
        "export_reimport_passed",
    )
    all_gates_passed = all(getattr(result, field) is True for field in gate_fields)
    return replace(
        result, accepted=all_gates_passed and qa_verdict == "SHIP"
    )


def compare_render_images(reference_path: Path, candidate_path: Path) -> dict[str, object]:
    """Compare two renders after normalizing size and color space.

    RGB differences are foreground-normalized.  Silhouettes and coverage use alpha when
    either render is transparent; opaque comparison renders use the full canvas as their
    foreground.  This is export-equivalence evidence, not an artistic photo metric.
    """

    with Image.open(reference_path) as opened_reference, Image.open(candidate_path) as opened_candidate:
        reference_image = opened_reference.convert("RGBA")
        candidate_image = opened_candidate.convert("RGBA")
        reference_size = opened_reference.size
        candidate_size = opened_candidate.size

    if candidate_image.size != reference_image.size:
        # Resize premultiplied RGBA. Pillow otherwise bleeds the RGB values of
        # transparent pixels into the foreground while normalizing size.
        candidate_array = np.asarray(candidate_image, dtype=np.float32)
        alpha = candidate_array[:, :, 3:4] / 255.0
        premultiplied = np.concatenate(
            (candidate_array[:, :, :3] * alpha, candidate_array[:, :, 3:4]), axis=2
        ).astype(np.uint8)
        resized = np.asarray(
            Image.fromarray(premultiplied, mode="RGBA").resize(
                reference_image.size, resample=Image.Resampling.LANCZOS
            ),
            dtype=np.float32,
        )
        resized_alpha = np.clip(resized[:, :, 3:4], 0.0, 255.0)
        restored_rgb = np.divide(
            resized[:, :, :3],
            resized_alpha / 255.0,
            out=np.zeros_like(resized[:, :, :3]),
            where=resized_alpha > 0,
        )
        normalized = np.concatenate(
            (np.clip(restored_rgb, 0.0, 255.0), resized_alpha), axis=2
        ).astype(np.uint8)
        candidate_image = Image.fromarray(normalized, mode="RGBA")

    reference = np.asarray(reference_image, dtype=np.uint8)
    candidate = np.asarray(candidate_image, dtype=np.uint8)
    transparent = bool(np.any(reference[:, :, 3] < 255) or np.any(candidate[:, :, 3] < 255))
    reference_mask = reference[:, :, 3] > 0
    candidate_mask = candidate[:, :, 3] > 0
    foreground = reference_mask | candidate_mask

    intersection = int(np.count_nonzero(reference_mask & candidate_mask))
    union = int(np.count_nonzero(reference_mask | candidate_mask))
    silhouette_iou = intersection / union if union else 1.0
    reference_coverage = float(np.count_nonzero(reference_mask)) / reference_mask.size
    candidate_coverage = float(np.count_nonzero(candidate_mask)) / candidate_mask.size
    coverage_difference = abs(reference_coverage - candidate_coverage)

    if np.any(foreground):
        rgb_difference = np.abs(
            reference[:, :, :3][foreground].astype(np.float32)
            - candidate[:, :, :3][foreground].astype(np.float32)
        ) / 255.0
        mean_difference = float(np.mean(rgb_difference))
        percentile_95 = float(np.percentile(rgb_difference, 95))
    else:
        mean_difference = 0.0
        percentile_95 = 0.0

    return {
        "reference_size": tuple(reference_size),
        "candidate_size": tuple(candidate_size),
        "compared_size": reference_image.size,
        "color_space": "RGBA8",
        "transparent": transparent,
        "mean_absolute_rgb_difference": mean_difference,
        "percentile_95_absolute_difference": percentile_95,
        "silhouette_iou": float(silhouette_iou),
        "foreground_coverage_difference": float(coverage_difference),
    }


def _finite_metric(value: object, name: str, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be finite in [{minimum}, {maximum}]: {value!r}")
    return number


def _component_review(review: Mapping[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for component in REQUIRED_COMPONENTS:
        if component not in review:
            continue
        value = review[component]
        normalized[component] = value.strip().lower() if isinstance(value, str) else value
    return normalized


def evaluate_geometry_gate(
    *,
    per_view_iou: Mapping[object, object],
    per_view_landmark_error: Mapping[object, object],
    component_review: Mapping[str, object],
    reliable_views: Sequence[object] | None = None,
) -> dict[str, object]:
    """Evaluate the final silhouette, landmark, and component-veto gate."""

    view_ids = tuple(per_view_iou) if reliable_views is None else tuple(reliable_views)
    if not view_ids:
        raise ValueError("geometry validation requires at least one reliable view")

    failures: list[str] = []
    ious: dict[object, float] = {}
    errors: dict[object, float] = {}
    for view_id in view_ids:
        if view_id not in per_view_iou or view_id not in per_view_landmark_error:
            failures.append(f"missing metric for reliable view {view_id}")
            continue
        ious[view_id] = _finite_metric(per_view_iou[view_id], f"IoU view {view_id}", 0.0, 1.0)
        errors[view_id] = _finite_metric(
            per_view_landmark_error[view_id],
            f"landmark error view {view_id}",
            0.0,
            1.0,
        )

    if ious and errors:
        iou_values = np.asarray(list(ious.values()), dtype=np.float64)
        error_values = np.asarray(list(errors.values()), dtype=np.float64)
        median_iou = float(np.median(iou_values))
        minimum_iou = float(np.min(iou_values))
        median_error = float(np.median(error_values))
        p95_error = float(np.percentile(error_values, 95))

        if median_iou < MEDIAN_IOU_THRESHOLD:
            failures.append(f"median silhouette IoU {median_iou:.4f} < {MEDIAN_IOU_THRESHOLD:.4f}")
        if minimum_iou < MINIMUM_IOU_THRESHOLD:
            failures.append(
                f"minimum reliable-view silhouette IoU {minimum_iou:.4f} < {MINIMUM_IOU_THRESHOLD:.4f}"
            )
        if median_error > MEDIAN_LANDMARK_THRESHOLD:
            failures.append(
                f"median landmark error {median_error:.4f} > {MEDIAN_LANDMARK_THRESHOLD:.4f}"
            )
        if p95_error > P95_LANDMARK_THRESHOLD:
            failures.append(
                f"95th-percentile landmark error {p95_error:.4f} > {P95_LANDMARK_THRESHOLD:.4f}"
            )
    else:
        median_iou = minimum_iou = median_error = p95_error = float("nan")

    normalized_review = _component_review(component_review)
    missing_components = [
        component for component in REQUIRED_COMPONENTS if component not in normalized_review
    ]
    for component in missing_components:
        failures.append(f"missing component review: {component}")
    failed_components = [
        component
        for component in REQUIRED_COMPONENTS
        if component in normalized_review and normalized_review[component] != "pass"
    ]
    for component in failed_components:
        failures.append(f"component visual veto: {component}")

    return {
        "passed": not failures,
        "failures": failures,
        "metrics": {
            "median_iou": median_iou,
            "minimum_iou": minimum_iou,
            "median_landmark_error": median_error,
            "p95_landmark_error": p95_error,
        },
        "reliable_views": view_ids,
        "per_view_iou": ious,
        "per_view_landmark_error": errors,
        "component_review": normalized_review,
        "thresholds": {
            "median_iou": MEDIAN_IOU_THRESHOLD,
            "minimum_iou": MINIMUM_IOU_THRESHOLD,
            "median_landmark_error": MEDIAN_LANDMARK_THRESHOLD,
            "p95_landmark_error": P95_LANDMARK_THRESHOLD,
        },
    }


def evaluate_visual_identity_gate(
    category_ratings: Mapping[str, object],
    component_review: Mapping[str, object],
) -> dict[str, object]:
    """Aggregate observed visual ratings and component vetoes without inference."""

    failures: list[str] = []
    missing = [name for name in REQUIRED_VISUAL_RATINGS if name not in category_ratings]
    for name in missing:
        failures.append(f"missing visual rating: {name}")

    for name in REQUIRED_VISUAL_RATINGS:
        if name not in category_ratings:
            continue
        rating = category_ratings[name]
        if rating not in VISUAL_RATINGS:
            failures.append(f"invalid visual rating {rating!r} for {name}")
        elif rating == "Miss":
            failures.append(f"visual rating Miss for {name}")

    normalized_components = _component_review(component_review)
    for component in REQUIRED_COMPONENTS:
        if component not in normalized_components:
            failures.append(f"missing component review: {component}")
        elif normalized_components[component] != "pass":
            failures.append(f"component visual veto: {component}")

    return {
        "passed": not failures,
        "failures": failures,
        "category_ratings": dict(category_ratings),
        "component_review": normalized_components,
    }


def validate_topology_report(report: Mapping[str, object]) -> dict[str, object]:
    """Independently validate clean-topology evidence in a Blender report."""

    failures: list[str] = []
    if report.get("accepted") is not True:
        failures.append("topology report is not accepted")

    counts = report.get("counts")
    if not isinstance(counts, Mapping):
        failures.append("topology counts are missing")
        counts = {}
    for field in TOPOLOGY_ZERO_COUNTS:
        if field not in counts:
            failures.append(f"missing topology count: {field}")
            continue
        value = counts[field]
        if not isinstance(value, (int, np.integer)) or bool(value):
            failures.append(f"non-zero {field}: {value}")

    objects = report.get("objects")
    if not isinstance(objects, list) or not objects:
        failures.append("topology object audit is empty")
        objects = []
    for entry in objects:
        name = entry.get("name") if isinstance(entry, Mapping) else None
        if not isinstance(name, str) or not name.strip():
            failures.append("topology audit contains an unnamed object")
        elif DEFAULT_BLENDER_NAME.fullmatch(name.strip()):
            failures.append(f"default Blender object name: {name.strip()}")

    if report.get("v1_geometry_mixed") is True:
        failures.append("V1 geometry is mixed into the final export")
    if report.get("orphan_test_objects"):
        failures.append("orphan test objects remain in the final export")
    if report.get("unintended_hidden_geometry"):
        failures.append("unintended hidden geometry remains in the final export")

    return {"passed": not failures, "failures": failures, "counts": dict(counts)}


def _minimum_texture_resolution(report: Mapping[str, object]) -> int:
    value = report.get("texture_resolution")
    if isinstance(value, Mapping):
        values = [resolution for resolution in value.values() if isinstance(resolution, int)]
        return min(values) if values else 0
    return value if isinstance(value, int) else 0


def validate_uv_bake_report(report: Mapping[str, object]) -> dict[str, object]:
    """Validate UV layout, bake, resolution, and overlap evidence."""

    failures: list[str] = []
    if report.get("accepted") is not True:
        failures.append("UV/bake report is not accepted")
    if report.get("bake_maps_accepted") is not True:
        failures.append("baked maps were not accepted")
    if report.get("normal_orientation_verified") is not True:
        failures.append("normal-map orientation was not verified")
    if report.get("hero_bake_artifacts"):
        failures.append("hero-view bake artifact remains")
    if _minimum_texture_resolution(report) < 4096:
        failures.append("minimum final texture resolution is below 4096")
    if report.get("unintended_overlap") is True:
        failures.append("UV contains unintended unique-detail overlap")

    pack_efficiency = report.get("pack_efficiency")
    pack_exception = report.get("pack_efficiency_exception")
    if not isinstance(pack_efficiency, (int, float)) or not 0.0 <= float(pack_efficiency) <= 1.0:
        failures.append("pack efficiency is missing or invalid")
    elif float(pack_efficiency) < 0.75 and not pack_exception:
        failures.append("UV pack efficiency is below 0.75 without a documented exception")

    meshes = report.get("meshes")
    if not isinstance(meshes, list) or not meshes:
        failures.append("UV mesh audit is empty")
        meshes = []
    for mesh in meshes:
        if not isinstance(mesh, Mapping) or mesh.get("uv_map") != "UV_Main":
            mesh_name = mesh.get("name", "<unnamed>") if isinstance(mesh, Mapping) else "<unnamed>"
            failures.append(f"visible mesh does not use UV_Main: {mesh_name}")

    return {"passed": not failures, "failures": failures}


def validate_texture_report(report: Mapping[str, object]) -> dict[str, object]:
    """Validate final texture files and material visual evidence."""

    failures: list[str] = []
    if report.get("accepted") is not True:
        failures.append("texture/lookdev report is not accepted")

    files = report.get("files")
    if not isinstance(files, list) or not files:
        failures.append("texture file audit is empty")
        files = []
    for texture in files:
        if not isinstance(texture, Mapping):
            failures.append("texture file entry is invalid")
            continue
        if not texture.get("path") or not texture.get("hash"):
            failures.append(f"texture file evidence is incomplete: {texture.get('path', '<missing>')}")
        size = texture.get("size")
        if not isinstance(size, (list, tuple)) or len(size) != 2 or min(size) < 1:
            failures.append(f"texture dimensions are invalid: {texture.get('path')}")

    for path in report.get("missing_images", []):
        failures.append(f"missing texture image: {path}")
    if report.get("missing_uv_references"):
        failures.append("texture report contains missing UV references")

    conditions = report.get("visual_conditions")
    if not isinstance(conditions, Mapping):
        failures.append("material visual-condition audit is missing")
        conditions = {}
    required_conditions = (
        "bright_polished_brass_match",
        "large_regions_not_dark",
        "engraving_readable",
        "no_baked_moving_highlights",
        "no_severe_uv_seams",
        "no_unsupported_hero_synthesis",
    )
    for condition in required_conditions:
        if conditions.get(condition) is not True:
            failures.append(f"material condition not passed: {condition}")

    if report.get("visual_match") not in {"Match", "Close"}:
        failures.append(f"texture visual rating is not Match/Close: {report.get('visual_match')!r}")
    inferred = report.get("inferred_region_percentage")
    if not isinstance(inferred, (int, float)) or not 0.0 <= float(inferred) <= 100.0:
        failures.append("inferred texture-region percentage is missing or invalid")

    return {"passed": not failures, "failures": failures}


def validate_export_reimport_report(report: Mapping[str, object]) -> dict[str, object]:
    """Validate fresh GLB re-import counts, dependencies, and comparison render."""

    failures: list[str] = []
    if report.get("accepted") is not True:
        failures.append("export/re-import report is not accepted")
    for count_name in ("mesh_count", "material_count", "image_count"):
        count = report.get(count_name)
        if not isinstance(count, (int, np.integer)) or count < 1:
            failures.append(f"export {count_name} must be positive")
    for path in report.get("missing_external_files", []):
        failures.append(f"missing external export file: {path}")
    if report.get("orientation_verified") is not True:
        failures.append("export orientation was not verified")
    if report.get("normals_verified") is not True:
        failures.append("export normals were not verified")
    if report.get("visual_inspection_passed") is not True:
        failures.append("export visual inspection did not pass")

    comparison = report.get("render_comparison")
    if not isinstance(comparison, Mapping):
        failures.append("master/re-import render comparison is missing")
        comparison = {}
    silhouette_iou = comparison.get("silhouette_iou")
    if not isinstance(silhouette_iou, (int, float)) or not 0.0 <= float(silhouette_iou) <= 1.0:
        failures.append("export silhouette IoU is missing or invalid")
    elif float(silhouette_iou) < EXPORT_IOU_THRESHOLD:
        failures.append(
            f"export silhouette IoU {float(silhouette_iou):.4f} < {EXPORT_IOU_THRESHOLD:.4f}"
        )

    mean_difference = comparison.get("mean_absolute_rgb_difference")
    if not isinstance(mean_difference, (int, float)) or not 0.0 <= float(mean_difference) <= 1.0:
        failures.append("export mean RGB difference is missing or invalid")
    elif float(mean_difference) > EXPORT_MEAN_DIFFERENCE_THRESHOLD:
        failures.append(
            "export mean absolute RGB difference "
            f"{float(mean_difference):.4f} > {EXPORT_MEAN_DIFFERENCE_THRESHOLD:.4f}"
        )

    return {"passed": not failures, "failures": failures}


def _audit_metric(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    return _finite_metric(value, name, 0.0, 1.0)


def _require_sha256(value: object, name: str, failures: list[str]) -> None:
    if not isinstance(value, str) or not LOWER_SHA256.fullmatch(value):
        failures.append(f"{name} must be a lowercase 64-character SHA-256")


def _require_provenance_entry(
    provenance: Mapping[str, object], name: str, failures: list[str]
) -> None:
    entry = provenance.get(name)
    if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str) or not entry.get("path").strip():
        failures.append(f"missing provenance source: {name}")
        return
    _require_sha256(entry.get("sha256"), f"provenance {name} sha256", failures)


def _require_surface_provenance(
    provenance: object, component: str, record_index: int, failures: list[str]
) -> None:
    label = f"surface-evidence provenance {component}[{record_index}]"
    if not isinstance(provenance, Mapping) or not isinstance(
        provenance.get("source"), str
    ) or not provenance.get("source", "").strip():
        failures.append(f"{label} source is missing")
        return
    _require_sha256(provenance.get("manifest_sha256"), f"{label} manifest_sha256", failures)


def validate_registered_view_coverage_report(
    report: Mapping[str, object],
) -> dict[str, object]:
    """Validate candidate-bound non-canonical registered-view silhouette evidence."""

    failures: list[str] = []
    if report.get("accepted") is not True:
        failures.append("registered-view coverage report is not accepted")
    if not isinstance(report.get("blend_path"), str) or not report.get("blend_path").strip():
        failures.append("evaluated blend path is missing")
    _require_sha256(report.get("blend_sha256"), "blend_sha256", failures)
    _require_sha256(
        report.get("final_cv_fit_candidate_sha256"),
        "final_cv_fit_candidate_sha256",
        failures,
    )

    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        failures.append("registered-view provenance is missing")
    else:
        for name in ("mask_manifest", "camera_normalization", "camera_alignment"):
            _require_provenance_entry(provenance, name, failures)

    views = report.get("views")
    if not isinstance(views, list) or not views:
        failures.append("registered-view audit is empty")
        views = []

    usable_model_mismatches = 0
    inspected_indices: set[object] = set()
    selected_indices: set[object] = set()
    for index, view in enumerate(views):
        if not isinstance(view, Mapping):
            failures.append(f"view {index} entry is invalid")
            continue
        label = f"view {view.get('selected_index', index)}"

        selected_index = view.get("selected_index")
        if not isinstance(selected_index, int) or isinstance(selected_index, bool) or selected_index < 0:
            failures.append(f"{label} selected_index must be a non-negative integer")
        elif selected_index in selected_indices:
            failures.append(f"duplicate registered-view selected_index: {selected_index}")
        else:
            selected_indices.add(selected_index)
        for field in ("filename", "sweep", "mask_source", "source_quality_condition"):
            value = view.get(field)
            if not isinstance(value, str) or not value.strip():
                failures.append(f"{label} {field} is missing")
        try:
            _audit_metric(view.get("silhouette_iou"), f"{label} silhouette_iou")
        except ValueError as error:
            failures.append(str(error))
        try:
            _audit_metric(
                view.get("foreground_coverage_difference"),
                f"{label} foreground_coverage_difference",
            )
        except ValueError as error:
            failures.append(str(error))
        if view.get("coordinate_path") not in REGISTERED_VIEW_COORDINATE_PATHS:
            failures.append(
                f"{label} coordinate_path must be one of "
                f"{sorted(REGISTERED_VIEW_COORDINATE_PATHS)}"
            )
        if not isinstance(view.get("usable"), bool):
            failures.append(f"{label} usable must be true or false")
        projection_valid = view.get("projection_valid")
        if not isinstance(projection_valid, bool):
            failures.append(f"{label} projection_valid must be true or false")
        classification = view.get("classification")
        if classification not in REGISTERED_VIEW_CLASSIFICATIONS:
            failures.append(
                f"{label} classification must be one of "
                f"{sorted(REGISTERED_VIEW_CLASSIFICATIONS)}"
            )
        elif classification == "model_mismatch" and view.get("usable") is True:
            usable_model_mismatches += 1
        elif classification == "camera_failure":
            consistency = view.get("camera_consistency")
            counterfactual_explains = bool(
                isinstance(consistency, Mapping)
                and (
                    consistency.get("translation_only_explains_gate_failure") is True
                    or consistency.get("rigid_alignment_explains_gate_failure") is True
                )
            )
            independent_reason = view.get("independent_camera_failure_reason")
            independently_explained = bool(
                projection_valid is False
                or counterfactual_explains
                or (isinstance(independent_reason, str) and independent_reason.strip())
            )
            if not independently_explained:
                failures.append(f"{label} camera_failure is not independently explained")

    inspected = report.get("inspected_worst_views")
    if not isinstance(inspected, list) or not inspected:
        failures.append("inspected worst-view evidence is empty")
        inspected = []
    view_indices = {
        view.get("selected_index")
        for view in views
        if isinstance(view, Mapping) and "selected_index" in view
    }
    view_by_index = {
        view.get("selected_index"): view
        for view in views
        if isinstance(view, Mapping) and "selected_index" in view
    }
    for inspected_view in inspected:
        if not isinstance(inspected_view, Mapping):
            failures.append("inspected worst-view entry is invalid")
            continue
        selected_index = inspected_view.get("selected_index")
        inspected_indices.add(selected_index)
        if selected_index not in view_indices:
            failures.append(
                f"inspected worst view {selected_index!r} is absent from the audit views"
            )
            continue
        audited_view = view_by_index[selected_index]
        if inspected_view.get("classification") not in REGISTERED_VIEW_CLASSIFICATIONS:
            failures.append(
                f"inspected worst view {selected_index!r} classification is invalid"
            )
        elif inspected_view.get("classification") != audited_view.get("classification"):
            failures.append(
                f"inspected worst view {selected_index!r} classification differs from the audit view"
            )
        filename = inspected_view.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            failures.append(
                f"inspected worst view {selected_index!r} filename is missing"
            )
        elif filename != audited_view.get("filename"):
            failures.append(
                f"inspected worst view {selected_index!r} filename differs from the audit view"
            )

    if usable_model_mismatches >= 2:
        failures.append(
            "repeated usable model_mismatch requires geometry correction "
            f"({usable_model_mismatches} views)"
        )

    return {
        "passed": not failures,
        "failures": failures,
        "view_count": len(views),
        "usable_model_mismatch_count": usable_model_mismatches,
        "inspected_worst_view_indices": sorted(inspected_indices, key=repr),
    }


def validate_surface_evidence_coverage_report(
    report: Mapping[str, object],
) -> dict[str, object]:
    """Validate component/sector evidence classes without promoting inference."""

    failures: list[str] = []
    if report.get("accepted") is not True:
        failures.append("surface-evidence coverage report is not accepted")
    if not isinstance(report.get("blend_path"), str) or not report.get("blend_path").strip():
        failures.append("evaluated blend path is missing")
    _require_sha256(report.get("blend_sha256"), "blend_sha256", failures)

    components = report.get("components")
    if not isinstance(components, Mapping):
        failures.append("surface-evidence components are missing")
        components = {}
    for component in REQUIRED_SURFACE_COMPONENTS:
        records = components.get(component)
        if not isinstance(records, list) or not records:
            failures.append(f"surface-evidence component has no sectors: {component}")
            continue
        sectors: set[str] = set()
        for record_index, record in enumerate(records):
            if not isinstance(record, Mapping):
                failures.append(
                    f"surface-evidence record {component}[{record_index}] is invalid"
                )
                continue
            sector = record.get("sector")
            if not isinstance(sector, str) or not sector.strip():
                failures.append(
                    f"surface-evidence sector is missing: {component}[{record_index}]"
                )
            elif sector in sectors:
                failures.append(f"duplicate surface-evidence sector: {component}/{sector}")
            else:
                sectors.add(sector)
            if record.get("support_class") not in SURFACE_EVIDENCE_CLASSES:
                failures.append(
                    f"surface-evidence support_class must be one of "
                    f"{sorted(SURFACE_EVIDENCE_CLASSES)}: {component}[{record_index}]"
                )
            source_support = record.get("source_support")
            if not isinstance(source_support, Mapping) or not source_support:
                failures.append(
                    f"surface-evidence source_support is missing: {component}[{record_index}]"
                )
                continue
            support_count = source_support.get("support_count")
            if (
                not isinstance(support_count, int)
                or isinstance(support_count, bool)
                or support_count < 0
            ):
                failures.append(
                    f"surface-evidence support_count must be a non-negative integer: "
                    f"{component}[{record_index}]"
                )
            source_views = source_support.get("source_views")
            if not isinstance(source_views, list) or len(source_views) != support_count:
                failures.append(
                    f"surface-evidence source_views must match support_count: "
                    f"{component}[{record_index}]"
                )
                source_views = []
            support_class = record.get("support_class")
            expected_counts = {
                "direct_multi_view": lambda count: count >= 2,
                "reviewed_single_or_detail": lambda count: count >= 1,
                "symmetry_repetition": lambda count: count == 0,
                "hidden_generic_fill": lambda count: count == 0,
            }
            if (
                isinstance(support_count, int)
                and not isinstance(support_count, bool)
                and support_class in expected_counts
                and not expected_counts[support_class](support_count)
            ):
                failures.append(
                    f"surface-evidence support_count contradicts support_class: "
                    f"{component}[{record_index}]"
                )
            if support_class in {"direct_multi_view", "reviewed_single_or_detail"}:
                visibility = source_support.get("camera_component_mask_evidence")
                visibility_indices = {
                    item.get("selected_index")
                    for item in visibility
                    if isinstance(item, Mapping)
                } if isinstance(visibility, list) else set()
                if set(source_views) != visibility_indices:
                    failures.append(
                        "surface-evidence camera/component visibility evidence must "
                        f"exactly support source_views: {component}[{record_index}]"
                    )
            _require_surface_provenance(
                record.get("provenance"), component, record_index, failures
            )

    sector_count = sum(
        len(records)
        for records in components.values()
        if isinstance(records, list)
    )
    return {
        "passed": not failures,
        "failures": failures,
        "sector_count": sector_count,
    }


def _panel_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as opened:
        image = ImageOps.contain(opened.convert("RGB"), size)
        x = (size[0] - image.width) // 2
        y = (size[1] - image.height) // 2
        canvas = Image.new("RGB", size, (12, 12, 12))
        canvas.paste(image, (x, y))
        return canvas


def build_comparison_panel(
    output_path: Path,
    *,
    source: Path,
    final_render: Path,
    silhouette_overlay: Path,
    difference: Path | None = None,
    source_label: str = "SOURCE",
) -> Path:
    """Create a deterministic four-up geometry comparison panel."""

    image_size = (160, 180)
    border = 4
    top_height = 24
    bottom_height = 24
    canvas_width = image_size[0] * 4 + border * 3
    canvas_height = top_height + image_size[1] + bottom_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), (12, 12, 12))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    cells = (
        (source, source_label),
        (final_render, "FINAL RENDER"),
        (silhouette_overlay, "SILHOUETTE"),
        (difference, "DIFFERENCE/LANDMARKS"),
    )
    for index, (path, label) in enumerate(cells):
        x = index * (image_size[0] + border)
        draw.text((x + 4, 4), label, fill=(240, 240, 240), font=font)
        if path is not None:
            canvas.paste(_panel_image(path, image_size), (x, top_height))
        else:
            draw.rectangle(
                (x, top_height, x + image_size[0] - 1, top_height + image_size[1] - 1),
                outline=(100, 100, 100),
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)
    return output_path
