"""Independent diagnostics for acquisition-local COLMAP pair geometry.

This audit intentionally reads a SQLite-consistent source snapshot, not the
mapper's pruned database.  It reports correspondence and essential-matrix
health for a fixed pair list so mapper edge selection cannot erase acceptance
evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v4_repair import create_disposable_sqlite_from_manifest, load_canonical_sqlite_snapshot_manifest


# Keep the independent geometry audit reproducible across disposable working
# copies.  The evidence is still computed from the immutable source rows; the
# seed only removes RANSAC's otherwise process-dependent model selection.
CALIBRATED_RANSAC_SEED = 4201

# pyCOLMAP exposes both TwoViewGeometry.tri_angle and
# Rotation3d.angle() in radians.  The persisted audit contract is degrees so
# that thresholds and SO(3) diagnostics are unambiguous to downstream tools.
PYCOLMAP_ANGLE_UNITS = {
    "TwoViewGeometry.tri_angle": "radians",
    "Rotation3d.angle": "radians",
}
AUDIT_ANGLE_UNITS = {
    "triangulation_angle": "degrees",
    "rotation_angle": "degrees",
}


def radians_to_degrees(value: float | None) -> float | None:
    """Convert a finite pyCOLMAP angle without silently changing its unit."""

    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return float(math.degrees(value))


def _optional_tri_angle_degrees(value: float | None) -> float | None:
    """Convert COLMAP's non-negative triangulation angle sentinel to degrees."""

    if value is None or not math.isfinite(float(value)) or float(value) < 0.0:
        return None
    return radians_to_degrees(float(value))


def _camera_matrix(camera: Any) -> tuple[np.ndarray, np.ndarray]:
    params = np.asarray(camera.params, dtype=np.float64).reshape(-1)
    model = str(camera.model_name)
    if model == "SIMPLE_RADIAL":
        if len(params) != 4:
            raise ValueError(f"unexpected SIMPLE_RADIAL parameter count: {len(params)}")
        focal, cx, cy, distortion = params
        return (
            np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64),
            np.array([distortion, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )
    if model in {"PINHOLE", "OPENCV", "FULL_OPENCV"}:
        focal_x, focal_y, cx, cy = params[:4]
        distortion = params[4:] if len(params) > 4 else np.empty(0, dtype=np.float64)
        return (
            np.array([[focal_x, 0.0, cx], [0.0, focal_y, cy], [0.0, 0.0, 1.0]], dtype=np.float64),
            np.asarray(distortion, dtype=np.float64),
        )
    raise ValueError(f"unsupported camera model for independent audit: {model}")


def _normalized_points(points: np.ndarray, camera: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import cv2

    matrix, distortion = _camera_matrix(camera)
    pixels = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    normalized = cv2.undistortPoints(pixels.reshape(-1, 1, 2), matrix, distortion).reshape(-1, 2)
    return pixels, normalized, matrix


def _rotation_from_essential(
    essential: np.ndarray, normalized_first: np.ndarray, normalized_second: np.ndarray
) -> tuple[np.ndarray | None, int, str | None]:
    import cv2

    try:
        count, rotation, _translation, mask = cv2.recoverPose(
            np.asarray(essential, dtype=np.float64),
            normalized_first,
            normalized_second,
            np.eye(3, dtype=np.float64),
            mask=None,
        )
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            return None, int(count), "non_finite_rotation"
        return np.asarray(rotation, dtype=np.float64), int(count), None
    except Exception as error:  # pragma: no cover - OpenCV/runtime-specific
        return None, 0, f"recover_pose_error:{type(error).__name__}"


def _sampson_residual(essential: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_h = np.column_stack([first, np.ones(len(first), dtype=np.float64)])
    second_h = np.column_stack([second, np.ones(len(second), dtype=np.float64)])
    essential = np.asarray(essential, dtype=np.float64)
    values = np.sum(second_h * (first_h @ essential.T), axis=1)
    ef = first_h @ essential.T
    et = second_h @ essential
    denominator = ef[:, 0] ** 2 + ef[:, 1] ** 2 + et[:, 0] ** 2 + et[:, 1] ** 2
    denominator = np.maximum(denominator, np.finfo(np.float64).eps)
    return (values**2) / denominator


def _matrix_angle(rotation: np.ndarray | None) -> float | None:
    if rotation is None:
        return None
    cosine = np.clip((float(np.trace(rotation)) - 1.0) / 2.0, -1.0, 1.0)
    return float(math.degrees(math.acos(float(cosine))))


def audit_pair(database: Any, first_name: str, second_name: str) -> dict[str, Any]:
    import cv2
    import pycolmap

    first = database.read_image_with_name(first_name)
    second = database.read_image_with_name(second_name)
    first_id, second_id = int(first.image_id), int(second.image_id)
    geometry = database.read_two_view_geometry(first_id, second_id)
    if geometry is None:
        geometry = database.read_two_view_geometry(second_id, first_id)
    if geometry is None:
        return {"first": first_name, "second": second_name, "status": "missing_two_view_geometry"}
    keypoints_first = np.asarray(database.read_keypoints(first_id), dtype=np.float64)[:, :2]
    keypoints_second = np.asarray(database.read_keypoints(second_id), dtype=np.float64)[:, :2]
    matches = np.asarray(geometry.inlier_matches, dtype=np.int64).reshape(-1, 2)
    valid = (
        (matches[:, 0] >= 0)
        & (matches[:, 0] < len(keypoints_first))
        & (matches[:, 1] >= 0)
        & (matches[:, 1] < len(keypoints_second))
    ) if len(matches) else np.empty(0, dtype=bool)
    matches = matches[valid]
    pixels_first = keypoints_first[matches[:, 0]] if len(matches) else np.empty((0, 2))
    pixels_second = keypoints_second[matches[:, 1]] if len(matches) else np.empty((0, 2))
    camera = database.read_camera(int(first.camera_id))
    pixels_first, normalized_first, _matrix = _normalized_points(pixels_first, camera)
    pixels_second, normalized_second, _matrix = _normalized_points(pixels_second, camera)
    essential = np.asarray(geometry.E, dtype=np.float64)
    singular_values = np.linalg.svd(essential, compute_uv=False) if essential.shape == (3, 3) else np.empty(0)
    rotation, cheirality_count, rotation_reason = _rotation_from_essential(
        essential, normalized_first, normalized_second
    ) if len(matches) >= 5 else (None, 0, "insufficient_inlier_matches")
    reestimated_rotation = None
    reestimated_inliers = 0
    reestimate_reason = None
    calibrated_rotation = None
    calibrated_translation = None
    calibrated_inliers = 0
    calibrated_config = None
    calibrated_tri_angle_rad = None
    calibrated_cheirality_inliers = 0
    calibrated_cheirality_fraction = 0.0
    calibrated_essential_rank_ratio = None
    calibrated_reason = None
    try:
        # Re-estimate from the raw imported match rows, not the mapper's
        # selected/pruned graph.  This is an independent check of the cached
        # correspondence geometry.
        raw_matches = np.asarray(database.read_matches(first_id, second_id), dtype=np.int64).reshape(-1, 2)
        raw_valid = (
            (raw_matches[:, 0] >= 0)
            & (raw_matches[:, 0] < len(keypoints_first))
            & (raw_matches[:, 1] >= 0)
            & (raw_matches[:, 1] < len(keypoints_second))
        ) if len(raw_matches) else np.empty(0, dtype=bool)
        raw_matches = raw_matches[raw_valid]
        raw_first = keypoints_first[raw_matches[:, 0]] if len(raw_matches) else np.empty((0, 2))
        raw_second = keypoints_second[raw_matches[:, 1]] if len(raw_matches) else np.empty((0, 2))
        identity = np.arange(len(raw_matches), dtype=np.uint32)
        identity_matches = np.column_stack((identity, identity))
        calibrated_options = pycolmap.TwoViewGeometryOptions()
        calibrated_options.compute_relative_pose = True
        calibrated_options.ransac.max_error = 4.0
        calibrated_options.ransac.random_seed = CALIBRATED_RANSAC_SEED
        calibrated = (
            pycolmap.estimate_calibrated_two_view_geometry(
                camera,
                raw_first,
                camera,
                raw_second,
                identity_matches,
                calibrated_options,
            )
            if len(raw_matches) >= 5
            else None
        )
        if calibrated is not None:
            calibrated_config = int(calibrated.config)
            calibrated_inliers = int(len(np.asarray(calibrated.inlier_matches)))
            calibrated_tri_angle_rad = float(calibrated.tri_angle)
            calibrated_essential = np.asarray(calibrated.E, dtype=np.float64)
            if calibrated_essential.shape == (3, 3) and np.isfinite(calibrated_essential).all():
                calibrated_singular_values = np.linalg.svd(calibrated_essential, compute_uv=False)
                if len(calibrated_singular_values) and calibrated_singular_values[0] > 0:
                    calibrated_essential_rank_ratio = float(
                        calibrated_singular_values[-1] / calibrated_singular_values[0]
                    )
            if calibrated.cam2_from_cam1 is not None:
                calibrated_rotation = np.asarray(calibrated.cam2_from_cam1.rotation.matrix(), dtype=np.float64)
                calibrated_translation = np.asarray(
                    calibrated.cam2_from_cam1.translation, dtype=np.float64
                ).reshape(3)
            calibrated_rows = np.asarray(calibrated.inlier_matches, dtype=np.int64).reshape(-1, 2)
            valid_calibrated_rows = (
                (calibrated_rows[:, 0] >= 0)
                & (calibrated_rows[:, 0] < len(raw_matches))
                & (calibrated_rows[:, 1] >= 0)
                & (calibrated_rows[:, 1] < len(raw_matches))
            ) if len(calibrated_rows) else np.empty(0, dtype=bool)
            if len(calibrated_rows) and np.all(calibrated_rows[:, 0] == calibrated_rows[:, 1]):
                calibrated_raw_rows = calibrated_rows[valid_calibrated_rows, 0]
                _, calibrated_normalized_first, _ = _normalized_points(
                    raw_first[calibrated_raw_rows], camera
                )
                _, calibrated_normalized_second, _ = _normalized_points(
                    raw_second[calibrated_raw_rows], camera
                )
                _, calibrated_cheirality_inliers, _ = _rotation_from_essential(
                    np.asarray(calibrated.E, dtype=np.float64),
                    calibrated_normalized_first,
                    calibrated_normalized_second,
                )
                calibrated_cheirality_fraction = float(
                    calibrated_cheirality_inliers / len(calibrated_raw_rows)
                ) if len(calibrated_raw_rows) else 0.0
        options = pycolmap.RANSACOptions()
        options.max_error = 4.0
        options.min_num_trials = 1000
        options.max_num_trials = 100000
        options.random_seed = CALIBRATED_RANSAC_SEED
        estimated = pycolmap.estimate_relative_pose(camera, raw_first, camera, raw_second, options) if len(raw_matches) >= 5 else None
        if estimated is None:
            reestimate_reason = "no_relative_pose"
        else:
            reestimated_rotation = np.asarray(estimated["cam2_from_cam1"].rotation.matrix(), dtype=np.float64)
            reestimated_inliers = int(estimated.get("num_inliers", 0))
    except Exception as error:  # pragma: no cover - pyCOLMAP/runtime-specific
        reestimate_reason = f"relative_pose_error:{type(error).__name__}"
    sampson = _sampson_residual(essential, normalized_first, normalized_second) if len(matches) and essential.shape == (3, 3) else np.empty(0)
    homography_inliers = 0
    homography_status = "not_run"
    if len(matches) >= 8:
        try:
            _homography, mask = cv2.findHomography(normalized_first, normalized_second, cv2.RANSAC, 0.003)
            homography_inliers = int(np.asarray(mask, dtype=np.uint8).reshape(-1).sum()) if mask is not None else 0
            homography_status = "ok"
        except Exception as error:  # pragma: no cover
            homography_status = f"error:{type(error).__name__}"
    return {
        "first": first_name,
        "second": second_name,
        "first_image_id": first_id,
        "second_image_id": second_id,
        "pair_id": int(min(first_id, second_id) * (2**31 - 1) + max(first_id, second_id)),
        "status": "complete",
        "geometry_config": int(geometry.config),
        "verified_inliers": int(len(matches)),
        "unique_first_keypoints": int(len(np.unique(matches[:, 0]))) if len(matches) else 0,
        "unique_second_keypoints": int(len(np.unique(matches[:, 1]))) if len(matches) else 0,
        "essential_finite": bool(essential.shape == (3, 3) and np.isfinite(essential).all()),
        "essential_singular_values": [float(value) for value in singular_values],
        "essential_rank3_to_rank1": float(singular_values[-1] / singular_values[0]) if len(singular_values) and singular_values[0] > 0 else None,
        "two_view_rotation_matrix_first_to_second": rotation.tolist() if rotation is not None else None,
        "two_view_rotation_angle_deg": _matrix_angle(rotation),
        "recover_pose_cheirality_inliers": cheirality_count,
        "recover_pose_inlier_fraction": float(cheirality_count / len(matches)) if len(matches) else 0.0,
        "recover_pose_reason": rotation_reason,
        "independent_reestimated_rotation_matrix_first_to_second": reestimated_rotation.tolist() if reestimated_rotation is not None else None,
        "independent_reestimated_rotation_angle_deg": _matrix_angle(reestimated_rotation),
        "independent_reestimated_inliers": reestimated_inliers,
        "independent_reestimate_vs_stored_geodesic_deg": _matrix_angle(reestimated_rotation @ rotation.T) if reestimated_rotation is not None and rotation is not None else None,
        "independent_reestimate_reason": reestimate_reason,
        "calibrated_reestimate_config": calibrated_config,
        "calibrated_reestimate_inliers": calibrated_inliers,
        "calibrated_reestimate_tri_angle_rad": calibrated_tri_angle_rad,
        "calibrated_reestimate_tri_angle_deg": _optional_tri_angle_degrees(calibrated_tri_angle_rad),
        "calibrated_reestimate_cheirality_inliers": calibrated_cheirality_inliers,
        "calibrated_reestimate_cheirality_fraction": calibrated_cheirality_fraction,
        "calibrated_essential_rank3_to_rank1": calibrated_essential_rank_ratio,
        "calibrated_reestimate_rotation_matrix_first_to_second": calibrated_rotation.tolist() if calibrated_rotation is not None else None,
        "calibrated_reestimate_rotation_angle_deg": _matrix_angle(calibrated_rotation),
        "calibrated_reestimate_translation_first_to_second": calibrated_translation.tolist() if calibrated_translation is not None else None,
        "calibrated_reestimate_vs_stored_geodesic_deg": _matrix_angle(calibrated_rotation @ rotation.T) if calibrated_rotation is not None and rotation is not None else None,
        "calibrated_reestimate_reason": calibrated_reason,
        "sampson_median": float(np.median(sampson)) if len(sampson) else None,
        "sampson_p95": float(np.percentile(sampson, 95)) if len(sampson) else None,
        "homography_inliers": homography_inliers,
        "homography_inlier_fraction": float(homography_inliers / len(matches)) if len(matches) else 0.0,
        "homography_status": homography_status,
        "tri_angle_rad": float(geometry.tri_angle) if math.isfinite(float(geometry.tri_angle)) else None,
        "tri_angle_deg": _optional_tri_angle_degrees(float(geometry.tri_angle)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, help="Deprecated compatibility check; never opened.")
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path, help="JSON list of {first,second} pairs")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    pair_payload = json.loads(args.pairs.read_text(encoding="utf-8"))
    pairs = pair_payload.get("pairs") if isinstance(pair_payload, Mapping) else pair_payload
    if not isinstance(pairs, Sequence):
        raise ValueError("pairs JSON must be a list or an object with a pairs list")
    snapshot_manifest = load_canonical_sqlite_snapshot_manifest(args.snapshot_manifest)
    canonical_path = Path(snapshot_manifest["canonical_path"]).resolve()
    if args.database is not None and args.database.resolve() != canonical_path:
        raise ValueError("--database does not match the canonical_path in --snapshot-manifest")
    canonical_sha_before = hashlib.sha256(canonical_path.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="v4_sparse_pair_audit_") as temporary:
        working_path = Path(temporary) / "working.db"
        disposable = create_disposable_sqlite_from_manifest(args.snapshot_manifest, working_path)
        working_copy = disposable["working"]
        import pycolmap

        database = pycolmap.Database.open(str(working_path))
        try:
            records = [audit_pair(database, str(pair["first"]), str(pair["second"])) for pair in pairs]
        finally:
            database.close()
    canonical_sha_after = hashlib.sha256(canonical_path.read_bytes()).hexdigest()
    payload = {
        "schema_version": 2,
        "method": "independent frozen SQLite snapshot correspondence and essential-matrix audit via disposable working copy; deterministic RANSAC seed",
        "calibrated_ransac_seed": CALIBRATED_RANSAC_SEED,
        "angle_units": AUDIT_ANGLE_UNITS,
        "pycolmap_angle_source_units": PYCOLMAP_ANGLE_UNITS,
        "database": str(canonical_path),
        "snapshot_manifest": str(args.snapshot_manifest.resolve()),
        "snapshot_manifest_sha256": snapshot_manifest["manifest_sha256"],
        "database_sha256": canonical_sha_after,
        "working_database_sha256": working_copy["sha256"],
        "canonical_snapshot_hash_before": canonical_sha_before,
        "canonical_snapshot_hash_after": canonical_sha_after,
        "canonical_snapshot_hash_stable": canonical_sha_before == canonical_sha_after,
        "pair_source_sha256": hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
        "pair_count": len(records),
        "records": records,
    }
    if canonical_sha_before != canonical_sha_after:
        raise RuntimeError("frozen canonical SQLite snapshot changed during independent audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "pair_count": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
