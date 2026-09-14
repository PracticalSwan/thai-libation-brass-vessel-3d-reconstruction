"""Audit-only v6-center/v43-rotation composition for V4 sparse diagnosis.

This script deliberately never writes a COLMAP model and never opens the
frozen canonical SQLite snapshot.  It composes the v43 calibrated rotation
field with the v6 camera-center field only in memory, evaluates the existing
independent constraints against a disposable copy of the v43 exact graph,
and records the result as diagnostic evidence.  A future mapper may use the
result only as an initializer if every relevant constraint is positive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
import pycolmap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from evaluate_v4_sparse_candidate import _bind_calibrated_audit_evidence
from v4_repair import (
    SparseIntegrityConfig,
    canonical_pair,
    collect_sparse_pair_evidence,
    load_ring_metadata,
    sparse_camera_center_translation_evidence,
    sparse_mask_projection_evidence,
    stable_directory_sha256,
)


DEFAULT_V43_MODEL = (
    Path(r"D:\Side Projects\CSX4213_V4_Dense_Work\workspace_v4repair_sparse_v1")
    / "sparse_candidate_v43_match_conflict_free_outlier_cap15_direct_calibrated_global"
    / "0"
)
DEFAULT_V6_MODEL = (
    Path(r"D:\Side Projects\CSX4213_V4_Dense_Work\workspace_v4repair_sparse_v1")
    / "sparse_candidate_v6_glomap_global"
    / "0"
)
DEFAULT_V43_DATABASE = (
    Path(r"D:\Side Projects\CSX4213_V4_Dense_Work\workspace_v4repair_sparse_v1")
    / "database_v43_match_conflict_free_outlier_cap15_direct_calibrated_exact1530_disposable.db"
)
REPAIR_ROOT = PROJECT_ROOT / "reconstruction" / "v4" / "repair" / "sparse_v1"
DEFAULT_RING_METADATA = REPAIR_ROOT / "selected_ring_metadata_v1.json"
DEFAULT_AUDIT_MANIFEST = REPAIR_ROOT / "independent_audit_pairs_v1.json"
DEFAULT_RAW_AUDIT = REPAIR_ROOT / "independent_audit_geometry_v8_calibrated_units.json"
DEFAULT_CLASSIFICATION = REPAIR_ROOT / "calibrated_pair_classification_v3_calibrated_units.json"
DEFAULT_EXCLUSIONS = REPAIR_ROOT / "geo_g10_edge_disposition_v1.json"
DEFAULT_CANONICAL_MANIFEST = REPAIR_ROOT / "canonical_sqlite_snapshot_v3.json"
DEFAULT_OUTPUT = REPAIR_ROOT / "v48_v6_centers_v43_rotations_diagnostic.json"
DEFAULT_MASK_ROOT = PROJECT_ROOT / "capture_v4" / "derived" / "masks"


class _CombinedImage:
    """Read-only image proxy with v43 rotation and v6 center."""

    def __init__(self, base: Any, center: np.ndarray) -> None:
        self._base = base
        self._center = np.asarray(center, dtype=np.float64).reshape(3)
        rotation = np.asarray(base.cam_from_world().rotation.matrix(), dtype=np.float64)
        translation = -rotation @ self._center
        self._pose = pycolmap.Rigid3d(pycolmap.Rotation3d(matrix=rotation), translation)

    def cam_from_world(self) -> Any:
        return self._pose

    def projection_center(self) -> np.ndarray:
        return self._center.copy()

    def viewing_direction(self) -> Any:
        return self._base.viewing_direction()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


class _CombinedReconstruction:
    """Delegate all read-only model data while replacing image poses in memory."""

    def __init__(self, base: Any, centers_by_name: Mapping[str, np.ndarray]) -> None:
        self._base = base
        self._centers = {str(name): np.asarray(value, dtype=np.float64).reshape(3) for name, value in centers_by_name.items()}
        self._images = {
            int(image_id): _CombinedImage(base.image(int(image_id)), self._centers[str(base.image(int(image_id)).name)])
            for image_id in base.reg_image_ids()
        }
        self._by_name = {str(image.name): image for image in self._images.values()}

    def image(self, image_id: int) -> _CombinedImage:
        return self._images[int(image_id)]

    def find_image_with_name(self, name: str) -> _CombinedImage:
        return self._by_name[str(name)]

    def reg_image_ids(self) -> list[int]:
        return [int(value) for value in self._base.reg_image_ids()]

    def camera(self, camera_id: int) -> Any:
        return self._base.camera(int(camera_id))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_translation_records(raw_audit: Mapping[str, Any], classification: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_pair = {
        int(item["pair_id"]): item
        for item in classification.get("classification", [])
        if isinstance(item, Mapping) and "pair_id" in item
    }
    records: list[dict[str, Any]] = []
    for item in raw_audit.get("records", []):
        if not isinstance(item, Mapping) or "pair_id" not in item:
            continue
        record = dict(item)
        record["well_conditioned_calibrated"] = bool(
            by_pair.get(int(item["pair_id"]), {}).get("well_conditioned_calibrated", False)
        )
        records.append(record)
    return records


def _frozen_canonical_evidence(manifest_path: Path, source_db: Path) -> dict[str, Any]:
    """Check frozen snapshot bytes without opening the canonical path as SQLite."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical_path = Path(str(manifest["canonical_path"]))
    recorded_sha = str(manifest["canonical_sha256"])
    current_sha = _sha256(canonical_path)
    sidecars = {
        suffix: canonical_path.with_name(canonical_path.name + suffix).exists()
        for suffix in ("-wal", "-shm")
    }
    return {
        "manifest_path": str(manifest_path.resolve()),
        "canonical_path": str(canonical_path.resolve()),
        "recorded_sha256": recorded_sha,
        "current_sha256": current_sha,
        "raw_sha_equal": current_sha == recorded_sha,
        "creation_logical_sha256": str(manifest.get("creation_logical_sha256", "")),
        "sidecars_present": sidecars,
        "canonical_sqlite_opened": False,
        "source_database_is_disposable": source_db.resolve() != canonical_path.resolve(),
    }


def _intrinsic_evidence(reconstruction: Any, expected_names: set[str]) -> dict[str, Any]:
    cameras = []
    valid = True
    for camera in reconstruction.cameras.values():
        params = np.asarray(camera.params, dtype=np.float64).reshape(-1)
        item = {
            "camera_id": int(camera.camera_id),
            "model": str(camera.model_name),
            "finite": bool(params.size >= 3 and np.isfinite(params).all()),
            "focal_positive": bool(params.size >= 1 and float(params[0]) > 0.0),
        }
        valid = valid and item["finite"] and item["focal_positive"]
        cameras.append(item)
    registered = {str(reconstruction.image(int(image_id)).name) for image_id in reconstruction.reg_image_ids()}
    return {
        "camera_count": len(cameras),
        "cameras": cameras,
        "all_finite_and_focal_positive": bool(valid and cameras),
        "registered_count": len(registered),
        "registered_set_exact": registered == expected_names,
    }


def _pose_field_evidence(
    v43: Any,
    v6: Any,
    combined: Any,
    expected_names: set[str],
) -> dict[str, Any]:
    v43_by_name = {str(v43.image(int(i)).name): v43.image(int(i)) for i in v43.reg_image_ids()}
    v6_by_name = {str(v6.image(int(i)).name): v6.image(int(i)) for i in v6.reg_image_ids()}
    combined_by_name = {
        str(combined.image(int(i)).name): combined.image(int(i))
        for i in combined.reg_image_ids()
    }
    names = set(v43_by_name) & set(v6_by_name)
    rotation_errors: list[float] = []
    center_errors: list[float] = []
    pose_translation_errors: list[float] = []
    rotation_orthogonality_errors: list[float] = []
    rotation_determinants: list[float] = []
    for name in sorted(names):
        source_rotation = np.asarray(
            v43_by_name[name].cam_from_world().rotation.matrix(),
            dtype=np.float64,
        )
        combined_pose = combined_by_name[name].cam_from_world()
        combined_rotation = np.asarray(combined_pose.rotation.matrix(), dtype=np.float64)
        product = combined_rotation @ source_rotation.T
        cosine = float(np.clip((np.trace(product) - 1.0) / 2.0, -1.0, 1.0))
        rotation_errors.append(float(np.degrees(np.arccos(cosine))))
        combined_center = np.asarray(combined_by_name[name].projection_center(), dtype=np.float64)
        source_center = np.asarray(v6_by_name[name].projection_center(), dtype=np.float64)
        center_errors.append(float(np.linalg.norm(combined_center - source_center)))
        expected_translation = -combined_rotation @ combined_center
        observed_translation = np.asarray(combined_pose.translation, dtype=np.float64)
        pose_translation_errors.append(float(np.linalg.norm(observed_translation - expected_translation)))
        rotation_orthogonality_errors.append(
            float(np.max(np.abs(combined_rotation @ combined_rotation.T - np.eye(3))))
        )
        rotation_determinants.append(float(np.linalg.det(combined_rotation)))
    return {
        "v43_registered_count": len(v43_by_name),
        "v6_registered_count": len(v6_by_name),
        "combined_registered_count": len(combined_by_name),
        "common_count": len(names),
        "common_set_is_exact_372": names == expected_names and len(names) == 372,
        "rotation_source": "v43 calibrated direct-pose GLOMAP field",
        "center_source": "v6 image-derived camera-center field",
        "combined_rotation_geodesic_max_deg_vs_v43": max(rotation_errors, default=float("inf")),
        "combined_center_max_delta_m_vs_v6": max(center_errors, default=float("inf")),
        "combined_pose_translation_round_trip_max_m": max(pose_translation_errors, default=float("inf")),
        "combined_rotation_orthogonality_max_abs": max(
            rotation_orthogonality_errors,
            default=float("inf"),
        ),
        "combined_rotation_determinant_min": min(rotation_determinants, default=float("nan")),
        "spliced_components_are_not_a_promotable_model": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    ring_metadata = load_ring_metadata(args.ring_metadata)
    raw_audit = json.loads(args.raw_audit.read_text(encoding="utf-8"))
    classification = json.loads(args.classification.read_text(encoding="utf-8"))
    audit_manifest = json.loads(args.audit_manifest.read_text(encoding="utf-8"))
    exclusions = json.loads(args.exclusions.read_text(encoding="utf-8"))
    expected_names = {
        str(name)
        for name, item in ring_metadata.items()
        if bool(item.get("selected", True))
    }
    fixed_pair_ids = [int(item["pair_id"]) for item in audit_manifest.get("pairs", [])]
    v43 = pycolmap.Reconstruction(str(args.v43_model))
    v6 = pycolmap.Reconstruction(str(args.v6_model))
    centers = {
        str(v6.image(int(image_id)).name): np.asarray(v6.image(int(image_id)).projection_center(), dtype=np.float64)
        for image_id in v6.reg_image_ids()
    }
    combined = _CombinedReconstruction(v43, centers)
    config = SparseIntegrityConfig()
    translation_records = _load_translation_records(raw_audit, classification)
    source_db_sha = _sha256(args.v43_database)

    with tempfile.TemporaryDirectory(prefix="v4_combined_pose_audit_") as temporary:
        working_db_path = Path(temporary) / "v43_exact_graph_working.db"
        shutil.copy2(args.v43_database, working_db_path)
        database = pycolmap.Database.open(str(working_db_path))
        try:
            evidence = collect_sparse_pair_evidence(
                database,
                combined,
                ring_metadata,
                pair_ids=fixed_pair_ids,
                config=config,
            )
        finally:
            database.close()

    bound_evidence = _bind_calibrated_audit_evidence(evidence, raw_audit, classification)
    excluded_pairs = {
        canonical_pair(*str(key).split("|", 1))
        for key in (exclusions.get("mapping_exclusions", {}) or {})
        if "|" in str(key)
    }
    rotation_failures = [
        item
        for item in bound_evidence
        if item.get("same_ring")
        and item.get("evaluable")
        and item.get("calibrated_conditioned")
        and canonical_pair(str(item["first"]), str(item["second"])) not in excluded_pairs
        and (
            item.get("rotation_disagreement_deg") is None
            or float(item["rotation_disagreement_deg"]) > config.maximum_rotation_disagreement_deg
        )
    ]
    trajectory_failures = [
        item
        for item in bound_evidence
        if item.get("same_ring")
        and item.get("trajectory_outlier")
        and canonical_pair(str(item["first"]), str(item["second"])) not in excluded_pairs
    ]
    translation = sparse_camera_center_translation_evidence(
        combined,
        translation_records,
        config=config,
    )
    translation_edges = list(translation.get("edges", []))
    if translation_edges:
        translation["direction_error_p90_deg"] = float(
            np.quantile(
                [float(item["direction_error_deg"]) for item in translation_edges],
                0.90,
            )
        )
        translation["orthogonal_residual_p90"] = float(
            np.quantile(
                [float(item["orthogonal_residual_fraction"]) for item in translation_edges],
                0.90,
            )
        )
    intrinsic = _intrinsic_evidence(v43, expected_names)
    pose_field = _pose_field_evidence(v43, v6, combined, expected_names)
    mask_projection = sparse_mask_projection_evidence(
        combined,
        args.mask_root,
        ring_metadata,
        config=config,
    )
    graph_report = json.loads(args.graph_report.read_text(encoding="utf-8"))
    exact_graph = graph_report.get("exact_mapper_graph_preflight", {})
    canonical = _frozen_canonical_evidence(args.canonical_manifest, args.v43_database)
    non_mask_checks = {
        "exact_372_view_set": bool(pose_field["common_set_is_exact_372"]),
        # pycolmap's Rotation3d serialization introduces only a few microdegrees
        # of round-trip noise; this is a representation check, not a relaxed
        # acceptance threshold for calibrated pair disagreement.
        "rotation_field_is_v43_calibrated_field": pose_field["combined_rotation_geodesic_max_deg_vs_v43"] <= 1e-5,
        "center_field_is_v6_image_derived_field": pose_field["combined_center_max_delta_m_vs_v6"] <= 1e-9,
        "combined_pose_round_trip": pose_field["combined_pose_translation_round_trip_max_m"] <= 1e-9,
        "combined_rotations_are_so3": bool(
            pose_field["combined_rotation_orthogonality_max_abs"] <= 1e-9
            and 0.999999 <= pose_field["combined_rotation_determinant_min"] <= 1.000001
        ),
        "calibrated_so3": not rotation_failures,
        "trajectory_continuity": not trajectory_failures,
        "calibrated_translation_direction": bool(translation.get("passed")),
        "intrinsics_integrity": bool(intrinsic["all_finite_and_focal_positive"]),
        "exact_graph_pair_ids": bool(
            exact_graph.get("passed")
            and exact_graph.get("expected_pair_count") == 1530
            and exact_graph.get("nonempty_two_view_geometry_count") == 1530
            and exact_graph.get("raw_match_pair_count") == 1530
        ),
        "immutable_canonical_snapshot": bool(
            canonical["raw_sha_equal"]
            and not any(canonical["sidecars_present"].values())
            and canonical["canonical_sqlite_opened"] is False
        ),
        "disposable_working_database": bool(canonical["source_database_is_disposable"]),
    }
    return {
        "schema_version": 1,
        "status": "diagnostic_only",
        "method": "in-memory v6 camera centers plus v43 calibrated rotations; disposable exact-graph audit; no model write or pose promotion",
        "v43_model": str(args.v43_model.resolve()),
        "v43_model_sha256": stable_directory_sha256(args.v43_model),
        "v6_model": str(args.v6_model.resolve()),
        "v6_model_sha256": stable_directory_sha256(args.v6_model),
        "source_disposable_database": str(args.v43_database.resolve()),
        "source_disposable_database_sha256": source_db_sha,
        "frozen_canonical_snapshot": canonical,
        "fixed_audit_pair_count": len(fixed_pair_ids),
        "fixed_audit_pair_ids_sha256": hashlib.sha256(json.dumps(fixed_pair_ids, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "classification_sha256": hashlib.sha256(args.classification.read_bytes()).hexdigest(),
        "raw_audit_sha256": hashlib.sha256(args.raw_audit.read_bytes()).hexdigest(),
        "excluded_pair_count": len(excluded_pairs),
        "pose_field": pose_field,
        "intrinsics": intrinsic,
        "exact_graph_preflight": exact_graph,
        "calibrated_so3": {
            "evaluated_pair_count": len(bound_evidence),
            "failure_count": len(rotation_failures),
            "failures": rotation_failures,
            "unchanged_v43_rotation_field": True,
        },
        "trajectory": {
            "failure_count": len(trajectory_failures),
            "failures": trajectory_failures,
        },
        "translation_direction": translation,
        "mask_projection_diagnostic": mask_projection,
        "non_mask_constraint_checks": non_mask_checks,
        "non_mask_constraints_passed": bool(all(non_mask_checks.values())),
        "full_sparse_candidate_eligible": bool(
            all(non_mask_checks.values()) and mask_projection.get("passed", False)
        ),
        "promotion_allowed": False,
        "reconstruction_path_required_if_reused": "Only a fresh pyCOLMAP reconstruction/optimization may consume these fields; this report is not a camera model and cannot be promoted.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v43-model", type=Path, default=DEFAULT_V43_MODEL)
    parser.add_argument("--v6-model", type=Path, default=DEFAULT_V6_MODEL)
    parser.add_argument("--v43-database", type=Path, default=DEFAULT_V43_DATABASE)
    parser.add_argument("--ring-metadata", type=Path, default=DEFAULT_RING_METADATA)
    parser.add_argument("--audit-manifest", type=Path, default=DEFAULT_AUDIT_MANIFEST)
    parser.add_argument("--raw-audit", type=Path, default=DEFAULT_RAW_AUDIT)
    parser.add_argument("--classification", type=Path, default=DEFAULT_CLASSIFICATION)
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--canonical-manifest", type=Path, default=DEFAULT_CANONICAL_MANIFEST)
    parser.add_argument("--graph-report", type=Path, default=REPAIR_ROOT / "match_conflict_free_v39_outlier_cap15_exact1530_preparation_report.json")
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "non_mask_constraints_passed": payload["non_mask_constraints_passed"],
        "full_sparse_candidate_eligible": payload["full_sparse_candidate_eligible"],
        "rotation_failures": payload["calibrated_so3"]["failure_count"],
        "trajectory_failures": payload["trajectory"]["failure_count"],
        "translation_passed": payload["translation_direction"].get("passed"),
        "mask_passed": payload["mask_projection_diagnostic"].get("passed"),
        "report": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
