"""Create a versioned calibrated two-view-geometry remap from a frozen DB copy.

The canonical SQLite snapshot is never opened here.  The input graph database
is copied byte-for-byte to a disposable output first; only that output is
opened and rewritten.  Existing imported keypoints and raw ALIKED/LightGlue
match rows are the only correspondence source.  The calibrated estimator is
run with identity row matches so its returned inlier indices can be mapped
back to the original keypoint indices without inventing correspondences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v4_repair import (
    colmap_pair_id,
    copy_frozen_sqlite_snapshot,
    load_canonical_sqlite_snapshot_manifest,
    validate_exact_mapper_graph,
)


# Must match the independent audit so a remap cannot silently select a
# different calibrated model from the same raw ALIKED/LightGlue rows.
CALIBRATED_RANSAC_SEED = 4201


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def radians_to_degrees(value: float | None) -> float | None:
    """Convert a finite pyCOLMAP Rotation3d.angle() value explicitly."""

    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return float(math.degrees(value))


def _normalize_translation(value: np.ndarray) -> np.ndarray:
    translation = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(translation))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("calibrated audit has no usable translation direction")
    return translation / norm


def _projective_matrix_error(actual: np.ndarray, expected: np.ndarray) -> float:
    """Return the residual after the best scalar alignment of E or F."""

    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != (3, 3) or expected.shape != (3, 3) or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return math.inf
    denominator = float(np.sum(expected * expected))
    if denominator <= 0.0:
        return math.inf
    scale = float(np.sum(actual * expected) / denominator)
    return float(np.max(np.abs(actual - scale * expected)))


def _pose_override_payload(
    estimated: Any,
    payload: dict[str, Any],
    first_camera: Any,
    second_camera: Any,
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
) -> tuple[Any, float, dict[str, Any]]:
    """Persist a previously audited rotation without an E-decomposition swap.

    COLMAP databases persist qvec/tvec as well as E/F.  Reconstructing a pose
    from E on a later read can choose a different essential decomposition,
    especially for low-parallax pairs.  R and t are therefore taken from the
    same deterministic calibrated audit record, and E/F are rebuilt from both.
    The newly constructed TwoViewGeometry is immediately round-trip checked so
    q/t, E and F cannot drift apart silently.
    """

    import pycolmap

    matrix = np.asarray(rotation_matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("pose override rotation must be a finite 3x3 matrix")
    determinant = float(np.linalg.det(matrix))
    if abs(determinant - 1.0) > 1e-4:
        raise ValueError(f"pose override rotation is not SO(3): det={determinant}")
    translation = _normalize_translation(translation_vector)
    essential = _skew(translation) @ matrix
    k1 = np.asarray(first_camera.calibration_matrix(), dtype=np.float64)
    k2 = np.asarray(second_camera.calibration_matrix(), dtype=np.float64)
    fundamental = np.linalg.inv(k2).T @ essential @ np.linalg.inv(k1)
    payload = dict(payload)
    payload["E"] = essential
    payload["F"] = fundamental
    payload["cam2_from_cam1"] = pycolmap.Rigid3d(pycolmap.Rotation3d(matrix), translation)
    rewritten = pycolmap.TwoViewGeometry(payload)
    roundtrip_rotation = np.asarray(rewritten.cam2_from_cam1.rotation.matrix(), dtype=np.float64)
    roundtrip_translation = _normalize_translation(
        np.asarray(rewritten.cam2_from_cam1.translation, dtype=np.float64)
    )
    roundtrip_essential = np.asarray(rewritten.E, dtype=np.float64)
    roundtrip_fundamental = np.asarray(rewritten.F, dtype=np.float64)
    rotation_error = float(np.max(np.abs(roundtrip_rotation - matrix)))
    translation_error = float(np.max(np.abs(roundtrip_translation - translation)))
    essential_error = _projective_matrix_error(roundtrip_essential, essential)
    fundamental_error = _projective_matrix_error(roundtrip_fundamental, fundamental)
    details = {
        "rotation_max_abs_error": rotation_error,
        "translation_max_abs_error": translation_error,
        "essential_projective_max_abs_error": essential_error,
        "fundamental_projective_max_abs_error": fundamental_error,
        "rotation_match": bool(rotation_error <= 1e-6),
        "translation_match": bool(translation_error <= 1e-6),
        "essential_match": bool(essential_error <= 1e-6),
        "fundamental_match": bool(fundamental_error <= 1e-6),
    }
    details["passed"] = bool(all(details[key] for key in ("rotation_match", "translation_match", "essential_match", "fundamental_match")))
    if not details["passed"]:
        raise RuntimeError(f"pose override R/t/E/F round-trip failed: {details}")
    runtime_angle = radians_to_degrees(float(rewritten.cam2_from_cam1.rotation.angle()))
    return rewritten, float(runtime_angle), details


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_rows(raw_matches: np.ndarray, first_count: int, second_count: int) -> np.ndarray:
    rows = np.asarray(raw_matches, dtype=np.int64).reshape(-1, 2)
    if not len(rows):
        return rows
    valid = (
        (rows[:, 0] >= 0)
        & (rows[:, 0] < first_count)
        & (rows[:, 1] >= 0)
        & (rows[:, 1] < second_count)
    )
    return rows[valid]


def _pair_key(first: str, second: str) -> str:
    return f"{first}|{second}"


def _reestimate_pair(
    database: Any,
    first_name: str,
    second_name: str,
    pycolmap: Any,
    disposition_records: Mapping[str, Mapping[str, Any]] | None = None,
    require_pose_override: bool = False,
) -> dict[str, Any]:
    first = database.read_image_with_name(first_name)
    second = database.read_image_with_name(second_name)
    first_id, second_id = int(first.image_id), int(second.image_id)
    first_keypoints = np.asarray(database.read_keypoints(first_id), dtype=np.float64)[:, :2]
    second_keypoints = np.asarray(database.read_keypoints(second_id), dtype=np.float64)[:, :2]
    raw_matches = _valid_rows(database.read_matches(first_id, second_id), len(first_keypoints), len(second_keypoints))
    result: dict[str, Any] = {
        "first": first_name,
        "second": second_name,
        "pair_key": _pair_key(first_name, second_name),
        "first_image_id": first_id,
        "second_image_id": second_id,
        "raw_match_count": int(len(raw_matches)),
        "status": "pending",
    }
    if len(raw_matches) < 5:
        result["status"] = "insufficient_raw_matches"
        return result

    points_first = first_keypoints[raw_matches[:, 0]]
    points_second = second_keypoints[raw_matches[:, 1]]
    identity = np.arange(len(raw_matches), dtype=np.uint32)
    identity_matches = np.column_stack((identity, identity))
    options = pycolmap.TwoViewGeometryOptions()
    options.compute_relative_pose = True
    options.ransac.max_error = 4.0
    options.ransac.random_seed = CALIBRATED_RANSAC_SEED
    estimated = pycolmap.estimate_calibrated_two_view_geometry(
        database.read_camera(int(first.camera_id)),
        points_first,
        database.read_camera(int(second.camera_id)),
        points_second,
        identity_matches,
        options,
    )
    estimated_inlier_rows = np.asarray(estimated.inlier_matches, dtype=np.int64).reshape(-1, 2)
    # With identity matches, each estimator row index is a raw-match row.  Do
    # not trust or persist any out-of-range row returned by the runtime.
    valid_estimated = (
        (estimated_inlier_rows[:, 0] >= 0)
        & (estimated_inlier_rows[:, 0] < len(raw_matches))
        & (estimated_inlier_rows[:, 1] >= 0)
        & (estimated_inlier_rows[:, 1] < len(raw_matches))
    ) if len(estimated_inlier_rows) else np.empty(0, dtype=bool)
    if len(estimated_inlier_rows) and not np.all(estimated_inlier_rows[:, 0] == estimated_inlier_rows[:, 1]):
        result["status"] = "non_identity_estimator_rows"
        result["returned_inlier_rows"] = int(len(estimated_inlier_rows))
        return result
    row_indices = estimated_inlier_rows[valid_estimated, 0] if len(estimated_inlier_rows) else np.empty(0, dtype=np.int64)
    mapped_inliers = raw_matches[row_indices].astype(np.uint32, copy=False)
    payload = estimated.todict(recursive=False)
    payload["inlier_matches"] = mapped_inliers
    override = (disposition_records or {}).get(_pair_key(first_name, second_name))
    override_record = override.get("independent_audit_record") if isinstance(override, Mapping) else None
    if require_pose_override and not isinstance(override_record, Mapping):
        raise ValueError(
            f"required calibrated pose override missing for {first_name} -> {second_name}"
        )
    override_matrix = (
        override_record.get("calibrated_reestimate_rotation_matrix_first_to_second")
        if isinstance(override_record, Mapping)
        else None
    )
    override_translation = (
        override_record.get("calibrated_reestimate_translation_first_to_second")
        if isinstance(override_record, Mapping)
        else None
    )
    if override_matrix is not None and override_translation is None:
        raise ValueError(
            f"pose override for {first_name} -> {second_name} lacks the paired calibrated translation"
        )
    if require_pose_override and override_matrix is None:
        raise ValueError(
            f"required calibrated rotation override missing for {first_name} -> {second_name}"
        )
    runtime_angle = radians_to_degrees(float(estimated.cam2_from_cam1.rotation.angle()))
    pose_override_round_trip = None
    if override_matrix is not None:
        rewritten, runtime_angle, pose_override_round_trip = _pose_override_payload(
            estimated,
            payload,
            database.read_camera(int(first.camera_id)),
            database.read_camera(int(second.camera_id)),
            np.asarray(override_matrix, dtype=np.float64),
            np.asarray(override_translation, dtype=np.float64),
        )
    else:
        rewritten = pycolmap.TwoViewGeometry(payload)
    database.update_two_view_geometry(first_id, second_id, rewritten)
    result.update(
        {
            "status": "rewritten",
            "config": int(estimated.config),
            "returned_inlier_rows": int(len(estimated_inlier_rows)),
            "mapped_inlier_count": int(len(mapped_inliers)),
            "tri_angle_rad": float(estimated.tri_angle),
            "tri_angle_deg": float(math.degrees(estimated.tri_angle)) if float(estimated.tri_angle) >= 0.0 else None,
            "rotation_angle_rad": float(rewritten.cam2_from_cam1.rotation.angle()),
            "rotation_angle_deg": radians_to_degrees(float(rewritten.cam2_from_cam1.rotation.angle())),
            "mapped_inlier_sha256": hashlib.sha256(np.asarray(mapped_inliers, dtype=np.uint32).tobytes()).hexdigest(),
            "runtime_estimate_rotation_angle_deg": runtime_angle,
            "pose_override_applied": bool(override_matrix is not None),
        }
    )
    if override_matrix is not None:
        result["pose_override_rotation_matrix_first_to_second"] = np.asarray(override_matrix, dtype=np.float64).tolist()
        result["pose_override_translation_first_to_second"] = _normalize_translation(
            np.asarray(override_translation, dtype=np.float64)
        ).tolist()
        result["pose_override_round_trip"] = pose_override_round_trip
        result["pose_override_source"] = "calibrated_pair_classification_v3.mapping_records.independent_audit_record.R_and_t"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Disposable graph DB source; never canonical.")
    parser.add_argument("--output", required=True, type=Path, help="New disposable remapped DB.")
    parser.add_argument("--snapshot-manifest", required=True, type=Path, help="Manifest for the immutable v3 source lineage.")
    parser.add_argument("--pairs", required=True, type=Path, help="JSON object containing a pairs list.")
    parser.add_argument("--retained-only", action="store_true", help="For a disposition report, rewrite retained_for_mapping records only.")
    parser.add_argument("--disposition", type=Path, help="Optional evidence-bound disposition JSON providing recorded pose matrices.")
    parser.add_argument(
        "--raw-audit-report",
        type=Path,
        help="Raw calibrated audit report used to verify the disposition hash contract.",
    )
    parser.add_argument(
        "--require-pose-overrides",
        action="store_true",
        help="Fail closed if any requested pair lacks its same-solution calibrated R and t override.",
    )
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    snapshot_manifest = load_canonical_sqlite_snapshot_manifest(args.snapshot_manifest)
    pair_payload = json.loads(args.pairs.read_text(encoding="utf-8"))
    pairs = (
        pair_payload.get("pairs", pair_payload.get("records", pair_payload.get("mapping_records")))
        if isinstance(pair_payload, dict)
        else pair_payload
    )
    if not isinstance(pairs, list):
        raise ValueError("pairs JSON must be a list or object containing pairs")
    disposition_records: dict[str, Mapping[str, Any]] = {}
    disposition_payload: Mapping[str, Any] | None = None
    if args.disposition is not None:
        disposition_payload = json.loads(args.disposition.read_text(encoding="utf-8"))
        disposition_items = disposition_payload.get(
            "records", disposition_payload.get("mapping_records", [])
        )
        for record in disposition_items:
            if isinstance(record, Mapping) and record.get("first") and record.get("second"):
                disposition_records[_pair_key(str(record["first"]), str(record["second"]))] = record
    if args.retained_only:
        pairs = [pair for pair in pairs if isinstance(pair, dict) and pair.get("disposition") == "retained_for_mapping"]
    normalized_pairs: list[tuple[str, str]] = []
    declared_pair_ids: list[int | None] = []
    for pair in pairs:
        if isinstance(pair, dict):
            normalized_pairs.append((str(pair["first"]), str(pair["second"])))
            declared_pair_ids.append(int(pair["pair_id"]) if pair.get("pair_id") is not None else None)
        else:
            normalized_pairs.append((str(pair[0]), str(pair[1])))
            declared_pair_ids.append(None)
    if len(set(normalized_pairs)) != len(normalized_pairs):
        raise ValueError("pairs JSON contains duplicate image pairs")
    if args.require_pose_overrides and args.disposition is None:
        raise ValueError("--require-pose-overrides requires --disposition")
    if args.require_pose_overrides and args.raw_audit_report is None:
        raise ValueError("--require-pose-overrides requires --raw-audit-report")
    if args.raw_audit_report is not None:
        raw_audit_sha = _sha256(args.raw_audit_report)
        if disposition_payload is None:
            raise ValueError("--raw-audit-report requires --disposition")
        if disposition_payload.get("raw_audit_report_sha256") != raw_audit_sha:
            raise ValueError("disposition is not bound to the supplied raw calibrated audit report")
        if disposition_payload.get("snapshot_manifest_sha256") != _sha256(args.snapshot_manifest):
            raise ValueError("disposition is not bound to the supplied snapshot manifest")
        if disposition_payload.get("canonical_snapshot_sha256") != snapshot_manifest["canonical_sha256"]:
            raise ValueError("disposition canonical snapshot hash does not match the manifest")
        if disposition_payload.get("fixed_pair_count") != len(json.loads(args.raw_audit_report.read_text(encoding="utf-8")).get("records", [])):
            raise ValueError("disposition fixed-pair count does not match the raw audit")
    source = args.input.resolve()
    destination = args.output.resolve()
    if source == Path(snapshot_manifest["canonical_path"]).resolve():
        raise ValueError("--input may not be the canonical SQLite snapshot; use a disposable graph copy")
    canonical_sha_before = _sha256(Path(snapshot_manifest["canonical_path"]))
    if source == destination:
        raise ValueError("input and output must be distinct disposable paths")
    if not source.is_file():
        raise FileNotFoundError(source)
    import sqlite3

    source_connection = sqlite3.connect(f"file:{source.as_posix()}?mode=ro&immutable=1", uri=True)
    try:
        image_ids = {
            str(name): int(image_id)
            for image_id, name in source_connection.execute("SELECT image_id, name FROM images")
        }
    finally:
        source_connection.close()
    expected_pair_ids: set[int] = set()
    for index, (first_name, second_name) in enumerate(normalized_pairs):
        if first_name not in image_ids or second_name not in image_ids:
            raise ValueError(f"requested pair references an image absent from the disposable DB: {first_name}, {second_name}")
        computed_pair_id = colmap_pair_id(image_ids[first_name], image_ids[second_name])
        declared_pair_id = declared_pair_ids[index]
        if declared_pair_id is not None and declared_pair_id != computed_pair_id:
            raise ValueError(
                f"pair_id mismatch for {first_name} -> {second_name}: "
                f"declared {declared_pair_id}, computed {computed_pair_id}"
            )
        expected_pair_ids.add(computed_pair_id)
    input_graph_preflight = validate_exact_mapper_graph(
        source,
        expected_pair_ids,
        canonical_path=snapshot_manifest["canonical_path"],
        require_raw_matches=True,
    )
    if not input_graph_preflight["passed"]:
        raise RuntimeError(
            "input disposable DB is not an exact retained mapper graph; "
            f"physically prune it before remapping: {input_graph_preflight}"
        )
    copy_info = copy_frozen_sqlite_snapshot(source, destination)
    source_sha = _sha256(source)
    import pycolmap

    database = pycolmap.Database.open(str(destination))
    records: list[dict[str, Any]] = []
    try:
        for first_name, second_name in normalized_pairs:
            records.append(
                _reestimate_pair(
                    database,
                    first_name,
                    second_name,
                    pycolmap,
                    disposition_records,
                    require_pose_override=args.require_pose_overrides,
                )
            )
    finally:
        database.close()
    output_graph_preflight = validate_exact_mapper_graph(
        destination,
        expected_pair_ids,
        canonical_path=snapshot_manifest["canonical_path"],
        require_raw_matches=True,
    )
    if not output_graph_preflight["passed"]:
        raise RuntimeError(f"remapped output is not an exact retained mapper graph: {output_graph_preflight}")
    destination_sha = _sha256(destination)
    canonical_sha_after = _sha256(Path(snapshot_manifest["canonical_path"]))
    if canonical_sha_before != canonical_sha_after or canonical_sha_after != snapshot_manifest["canonical_sha256"]:
        raise RuntimeError("canonical SQLite snapshot changed during two-view remap")
    rewritten = sum(record.get("status") == "rewritten" for record in records)
    failed = [record for record in records if record.get("status") != "rewritten"]
    payload = {
        "method": "pycolmap calibrated two-view re-estimation from existing raw matches; identity row mapping; deterministic RANSAC seed",
        "calibrated_ransac_seed": CALIBRATED_RANSAC_SEED,
        "pose_override_source": str(args.disposition.resolve()) if args.disposition is not None else None,
        "pose_override_source_sha256": _sha256(args.disposition) if args.disposition is not None else None,
        "classification_path": str(args.disposition.resolve()) if args.disposition is not None else None,
        "classification_sha256": _sha256(args.disposition) if args.disposition is not None else None,
        "raw_audit_report": str(args.raw_audit_report.resolve()) if args.raw_audit_report is not None else None,
        "raw_audit_report_sha256": _sha256(args.raw_audit_report) if args.raw_audit_report is not None else None,
        "require_pose_overrides": bool(args.require_pose_overrides),
        "pose_override_angle_units": {
            "stored_tri_angle_rad": "radians",
            "stored_rotation_angle_rad": "radians",
            "report_tri_angle": "degrees",
            "report_rotation_angle": "degrees",
        },
        "input_path": str(source),
        "input_sha256_before": source_sha,
        "output_path": str(destination),
        "snapshot_manifest": str(args.snapshot_manifest.resolve()),
        "snapshot_manifest_sha256": snapshot_manifest["manifest_sha256"],
        "canonical_snapshot_sha256": snapshot_manifest["canonical_sha256"],
        "canonical_creation_logical_sha256": snapshot_manifest["creation_logical_sha256"],
        "canonical_snapshot_sha256_before": canonical_sha_before,
        "canonical_snapshot_sha256_after": canonical_sha_after,
        "canonical_snapshot_hash_stable": True,
        "output_sha256_after": destination_sha,
        "byte_copy_verified": bool(copy_info.get("sha256") == source_sha),
        "input_exact_mapper_graph_preflight": input_graph_preflight,
        "output_exact_mapper_graph_preflight": output_graph_preflight,
        "track_loading_path": output_graph_preflight["track_loading_path"],
        "pair_count": len(records),
        "rewritten_pair_count": rewritten,
        "failed_pair_count": len(failed),
        "pose_override_applied_count": sum(bool(record.get("pose_override_applied")) for record in records),
        "pose_override_round_trip_pass_count": sum(
            bool(record.get("pose_override_round_trip", {}).get("passed"))
            for record in records
        ),
        "pose_override_classification_snapshot_manifest_sha256": (
            disposition_payload.get("snapshot_manifest_sha256")
            if disposition_payload is not None
            else None
        ),
        "pose_override_classification_audit_manifest_sha256": (
            disposition_payload.get("audit_manifest_sha256")
            if disposition_payload is not None
            else None
        ),
        "pose_override_classification_fixed_pair_ids_sha256": (
            disposition_payload.get("fixed_pair_ids_sha256")
            if disposition_payload is not None
            else None
        ),
        "failed_pairs": failed,
        "pairs": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
