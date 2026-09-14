"""Evidence-bound sparse repair and downstream provenance gates for V4.

This module deliberately stays beside the historical V4 runner.  It reads the
already verified ALIKED/LightGlue-derived COLMAP database and historical model,
produces versioned diagnostics/constraint graphs, and refuses to overwrite any
historical artifact.  Expensive reconstruction is kept outside the helpers so
that each candidate can be audited before mapping or dense compute starts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from v4_config import sha256_file, write_json


PAIR_BASE = 2**31 - 1
_DUPLICATE_SUFFIX = re.compile(r"_(\d+)$")


@dataclass(frozen=True)
class SparseIntegrityConfig:
    """Bounded, evidence-oriented sparse integrity thresholds."""

    minimum_verified_inliers: int = 100
    minimum_shared_tracks: int = 20
    maximum_rotation_disagreement_deg: float = 30.0
    same_ring_window: int = 4
    minimum_cross_ring_inliers: int = 50
    trajectory_mad_scale: float = 8.0
    minimum_ring_registration_fraction: float = 0.95
    maximum_mean_reprojection_error: float = 3.0
    minimum_sparse_points: int = 100
    minimum_sparse_observations: int = 1000

    def validate(self) -> "SparseIntegrityConfig":
        if self.minimum_verified_inliers < 1 or self.minimum_shared_tracks < 0:
            raise ValueError("sparse integrity count thresholds must be positive/nonnegative")
        if not math.isfinite(self.maximum_rotation_disagreement_deg) or self.maximum_rotation_disagreement_deg <= 0:
            raise ValueError("maximum_rotation_disagreement_deg must be finite and positive")
        if self.same_ring_window < 1:
            raise ValueError("same_ring_window must be positive")
        if self.minimum_cross_ring_inliers < 1:
            raise ValueError("minimum_cross_ring_inliers must be positive")
        if not math.isfinite(self.trajectory_mad_scale) or self.trajectory_mad_scale <= 0:
            raise ValueError("trajectory_mad_scale must be finite and positive")
        if not math.isfinite(self.minimum_ring_registration_fraction) or not 0.0 < self.minimum_ring_registration_fraction <= 1.0:
            raise ValueError("minimum_ring_registration_fraction must be in (0, 1]")
        if not math.isfinite(self.maximum_mean_reprojection_error) or self.maximum_mean_reprojection_error <= 0:
            raise ValueError("maximum_mean_reprojection_error must be finite and positive")
        if self.minimum_sparse_points < 1 or self.minimum_sparse_observations < 1:
            raise ValueError("sparse point/observation thresholds must be positive")
        return self


def colmap_pair_id(first_image_id: int, second_image_id: int) -> int:
    """Return COLMAP's deterministic 64-bit pair identifier."""

    first, second = int(first_image_id), int(second_image_id)
    if first == second or first < 1 or second < 1:
        raise ValueError("COLMAP pair ids require two distinct positive image ids")
    if first > second:
        first, second = second, first
    return first * PAIR_BASE + second


def decode_colmap_pair_id(pair_id: int) -> tuple[int, int]:
    value = int(pair_id)
    if value < 1:
        raise ValueError("COLMAP pair id must be positive")
    first, second = divmod(value, PAIR_BASE)
    if first < 1 or second < 1 or second >= PAIR_BASE:
        raise ValueError(f"invalid COLMAP pair id: {pair_id}")
    return first, second


def canonical_pair(first: str, second: str) -> tuple[str, str]:
    first_name, second_name = Path(str(first)).name, Path(str(second)).name
    if not first_name or not second_name or first_name == second_name:
        raise ValueError("a sparse pair requires two distinct image names")
    return (first_name, second_name) if first_name < second_name else (second_name, first_name)


def duplicate_stem(name: str) -> str:
    """Collapse capture-system ``_01`` suffixes for duplicate-frame checks."""

    stem = Path(str(name)).stem
    return _DUPLICATE_SUFFIX.sub("", stem)


def is_near_duplicate(first: str, second: str) -> bool:
    return duplicate_stem(first) == duplicate_stem(second)


def load_ring_metadata(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("ring metadata must contain a records list")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("ring metadata record is not an object")
        name = Path(str(record.get("relative_path") or record.get("filename") or "")).name
        ring = str(record.get("logical_ring_id") or "")
        if not name or not ring:
            raise ValueError("ring metadata requires relative_path and logical_ring_id")
        result[name] = {
            "ring": ring,
            "frame_index": int(record.get("frame_index_within_logical_ring", 0)),
            "phase_01": float(record.get("phase_01") or 0.0),
            "selected": bool(record.get("selected_for_geometry", True)),
        }
    if len(result) != len(records):
        raise ValueError("ring metadata contains duplicate image names")
    return result


def stable_directory_sha256(path: str | Path) -> str:
    """Hash a COLMAP model directory by sorted relative path and file bytes."""

    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError(f"model directory is missing: {root}")
    files = sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.relative_to(root).as_posix())
    if not files:
        raise ValueError(f"model directory is empty: {root}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()


def _sqlite_logical_digest_connection(connection: sqlite3.Connection) -> dict[str, Any]:
    """Digest an already-open SQLite connection without reopening its file.

    BLOB values are represented by their length and SHA-256, avoiding a large
    JSON materialization while still binding keypoints, matches, and verified
    two-view rows to the snapshot lineage.
    """

    tables = ("cameras", "images", "keypoints", "matches", "two_view_geometries")
    digest = hashlib.sha256()
    counts: dict[str, int] = {}
    for table in tables:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if exists is None:
            counts[table] = 0
            digest.update(f"{table}:missing\n".encode("utf-8"))
            continue
        columns = [str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})").fetchall()]
        # The COLMAP tables are rowid tables, and the backup preserves their
        # rowids.  This is a logical digest, not a raw-byte claim.
        rows = connection.execute(f"SELECT * FROM {table} ORDER BY rowid")
        count = 0
        digest.update(f"{table}:{','.join(columns)}\n".encode("utf-8"))
        for row in rows:
            count += 1
            digest.update(f"{table}:{count}:".encode("utf-8"))
            for value in row:
                if isinstance(value, bytes):
                    digest.update(b"B")
                    digest.update(len(value).to_bytes(8, "big"))
                    digest.update(hashlib.sha256(value).digest())
                else:
                    encoded = repr(value).encode("utf-8")
                    digest.update(b"V")
                    digest.update(len(encoded).to_bytes(8, "big"))
                    digest.update(encoded)
            digest.update(b"\n")
        counts[table] = count
    return {"table_counts": counts, "logical_sha256": digest.hexdigest()}


def sqlite_logical_digest(path: str | Path) -> dict[str, Any]:
    """Return deterministic table counts and a digest for a working copy."""

    database_path = Path(path).resolve()
    if not database_path.is_file():
        raise ValueError(f"SQLite database is missing: {database_path}")
    with sqlite3.connect(str(database_path)) as connection:
        connection.execute("PRAGMA query_only=ON")
        payload = _sqlite_logical_digest_connection(connection)
    return {"path": str(database_path), **payload}


def create_sqlite_snapshot(source: str | Path, destination: str | Path) -> dict[str, Any]:
    """Create a consistent SQLite backup and digest it during creation only.

    The destination is never reopened after backup.  Callers must treat the
    returned destination as frozen bytes and use :func:`copy_frozen_sqlite_snapshot`
    before any SQLite/pyCOLMAP read.
    """

    source_path, destination_path = Path(source).resolve(), Path(destination).resolve()
    if not source_path.is_file():
        raise ValueError(f"source SQLite database is missing: {source_path}")
    if destination_path.exists():
        raise FileExistsError(f"destination SQLite snapshot already exists: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(source_path)) as source_connection:
        source_connection.execute("PRAGMA query_only=ON")
        logical = _sqlite_logical_digest_connection(source_connection)
        with sqlite3.connect(str(destination_path)) as destination_connection:
            source_connection.backup(destination_connection)
            destination_connection.commit()
    return {
        "path": str(destination_path),
        "sha256": sha256_file(destination_path),
        **logical,
    }


def load_canonical_sqlite_snapshot_manifest(path: str | Path) -> dict[str, Any]:
    """Validate the immutable v3 snapshot manifest without opening SQLite.

    The manifest is the only accepted source-of-truth handle for repair/audit
    commands.  Callers receive the canonical path for byte hashing/copying
    only; SQLite and pyCOLMAP must be opened on a disposable copy returned by
    :func:`create_disposable_sqlite_from_manifest`.
    """

    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("canonical SQLite snapshot manifest must be an object")
    if int(payload.get("schema_version", 0)) < 3:
        raise ValueError("canonical SQLite snapshot manifest must use schema_version >= 3")
    canonical_raw = str(payload.get("canonical_path") or payload.get("path") or "")
    canonical_path = Path(canonical_raw).resolve()
    expected_sha = str(payload.get("canonical_sha256") or payload.get("sha256") or "").lower()
    expected_logical = str(
        payload.get("creation_logical_sha256") or payload.get("logical_sha256") or ""
    ).lower()
    if not canonical_raw or len(expected_sha) != 64 or len(expected_logical) != 64:
        raise ValueError("canonical snapshot manifest lacks raw/logical hashes")
    if not canonical_path.is_file():
        raise FileNotFoundError(canonical_path)
    for suffix in ("-wal", "-shm"):
        if canonical_path.with_name(canonical_path.name + suffix).exists():
            raise ValueError(f"canonical snapshot has journal sidecar: {canonical_path}{suffix}")
    observed_sha = sha256_file(canonical_path).lower()
    if observed_sha != expected_sha:
        raise ValueError(
            f"canonical snapshot raw SHA does not match manifest: {observed_sha} != {expected_sha}"
        )
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "canonical_path": str(canonical_path),
        "canonical_sha256": expected_sha,
        "creation_logical_sha256": expected_logical,
        "table_counts": dict(payload.get("table_counts") or payload.get("counts") or {}),
        "payload": dict(payload),
    }


def create_disposable_sqlite_from_manifest(
    manifest_path: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    """Copy the manifest-bound canonical bytes without opening the source."""

    manifest = load_canonical_sqlite_snapshot_manifest(manifest_path)
    working = copy_frozen_sqlite_snapshot(manifest["canonical_path"], destination)
    return {"manifest": manifest, "working": working}


def copy_frozen_sqlite_snapshot(source: str | Path, destination: str | Path) -> dict[str, Any]:
    """Copy frozen SQLite bytes for disposable reads without opening ``source``."""

    source_path, destination_path = Path(source).resolve(), Path(destination).resolve()
    if not source_path.is_file():
        raise ValueError(f"frozen canonical SQLite snapshot is missing: {source_path}")
    if destination_path.exists():
        raise FileExistsError(f"working SQLite copy already exists: {destination_path}")
    for suffix in ("-wal", "-shm"):
        if source_path.with_name(source_path.name + suffix).exists():
            raise ValueError(f"canonical SQLite snapshot has journal sidecar: {source_path}{suffix}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination_path)
    source_sha = sha256_file(source_path)
    destination_sha = sha256_file(destination_path)
    if source_sha != destination_sha:
        raise IOError("frozen SQLite snapshot changed while copied")
    return {"source_path": str(source_path), "working_path": str(destination_path), "sha256": destination_sha}


def _camera_matrix(camera: Any) -> np.ndarray:
    params = np.asarray(camera.params, dtype=np.float64).reshape(-1)
    if len(params) < 3:
        raise ValueError("camera has too few parameters for a pinhole calibration")
    return np.array([[params[0], 0.0, params[1]], [0.0, params[0], params[2]], [0.0, 0.0, 1.0]], dtype=np.float64)


def _camera_distortion(camera: Any) -> np.ndarray:
    """Return COLMAP distortion parameters in OpenCV's calibrated form."""

    model = str(getattr(camera, "model_name", ""))
    params = np.asarray(camera.params, dtype=np.float64).reshape(-1)
    if model == "SIMPLE_RADIAL":
        if len(params) != 4:
            raise ValueError(f"unexpected SIMPLE_RADIAL parameter count: {len(params)}")
        return np.asarray([params[3], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if model == "PINHOLE":
        return np.empty(0, dtype=np.float64)
    if model == "OPENCV":
        return np.asarray(params[4:], dtype=np.float64)
    raise ValueError(f"unsupported camera model for two-view audit: {model}")


def _rotation_angle_deg(rotation: np.ndarray | None) -> float | None:
    if rotation is None:
        return None
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        return None
    cosine = np.clip((float(np.trace(value)) - 1.0) / 2.0, -1.0, 1.0)
    return float(math.degrees(math.acos(float(cosine))))


def _rotation_geodesic_deg(first: np.ndarray | None, second: np.ndarray | None) -> float | None:
    """Return the SO(3) geodesic between two first-to-second rotations."""

    if first is None or second is None:
        return None
    first_value = np.asarray(first, dtype=np.float64)
    second_value = np.asarray(second, dtype=np.float64)
    if first_value.shape != (3, 3) or second_value.shape != (3, 3):
        return None
    if not np.isfinite(first_value).all() or not np.isfinite(second_value).all():
        return None
    relative = first_value @ second_value.T
    if not np.isfinite(relative).all():
        return None
    cosine = np.clip((float(np.trace(relative)) - 1.0) / 2.0, -1.0, 1.0)
    return float(math.degrees(math.acos(float(cosine))))


def _recover_two_view_rotation(
    geometry: Any,
    keypoints_first: np.ndarray,
    keypoints_second: np.ndarray,
    camera: Any,
) -> tuple[np.ndarray | None, str | None]:
    """Recover the full calibrated first-camera-to-second-camera rotation.

    COLMAP stores the essential matrix with the usual ``x2.T E x1`` convention.
    OpenCV's calibrated ``recoverPose`` returns the corresponding transform from
    first-camera coordinates into second-camera coordinates.  Points are
    undistorted before decomposition so the comparison is not an angle-only or
    distorted-pixel approximation.
    """

    matches = np.asarray(getattr(geometry, "inlier_matches", np.empty((0, 2))))
    if matches.ndim != 2 or matches.shape[1] != 2 or len(matches) < 5:
        return None, "insufficient_inlier_matches"
    try:
        import cv2

        points_first = np.asarray(keypoints_first)[matches[:, 0].astype(np.int64), :2]
        points_second = np.asarray(keypoints_second)[matches[:, 1].astype(np.int64), :2]
        matrix = _camera_matrix(camera)
        distortion = _camera_distortion(camera)
        normalized_first = cv2.undistortPoints(
            points_first.reshape(-1, 1, 2), matrix, distortion
        ).reshape(-1, 2)
        normalized_second = cv2.undistortPoints(
            points_second.reshape(-1, 1, 2), matrix, distortion
        ).reshape(-1, 2)
        _count, rotation, _translation, _mask = cv2.recoverPose(
            np.asarray(geometry.E, dtype=np.float64),
            normalized_first,
            normalized_second,
            np.eye(3, dtype=np.float64),
        )
        value = np.asarray(rotation, dtype=np.float64)
        if value.shape != (3, 3) or not np.isfinite(value).all():
            return None, "non_finite_rotation"
        return value, None
    except Exception as error:  # pragma: no cover - dependency/runtime-specific
        return None, f"two_view_rotation_error:{type(error).__name__}"


def _angle_between(first: Any, second: Any) -> float | None:
    a, b = np.asarray(first, dtype=np.float64).reshape(-1), np.asarray(second, dtype=np.float64).reshape(-1)
    if len(a) != 3 or len(b) != 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    norms = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norms <= 1e-12:
        return None
    return float(math.degrees(math.acos(float(np.clip(np.dot(a, b) / norms, -1.0, 1.0)))))


def _final_pose_metrics(
    first: Any, second: Any
) -> tuple[float | None, float | None, float | None, np.ndarray | None]:
    try:
        first_pose, second_pose = first.cam_from_world(), second.cam_from_world()
        first_rotation = np.asarray(first_pose.rotation.matrix(), dtype=np.float64)
        second_rotation = np.asarray(second_pose.rotation.matrix(), dtype=np.float64)
        # X_cam2 = R_cam2_world @ R_cam1_world.T @ X_cam1.
        relative_rotation = second_rotation @ first_rotation.T
        rotation = _rotation_angle_deg(relative_rotation)
        center_first = np.asarray(first.projection_center(), dtype=np.float64)
        center_second = np.asarray(second.projection_center(), dtype=np.float64)
        center_jump = float(np.linalg.norm(center_second - center_first))
        ray_jump = _angle_between(first.viewing_direction(), second.viewing_direction())
        return rotation, center_jump, ray_jump, relative_rotation
    except Exception:
        return None, None, None, None


def _shared_track_count(first: Any, second: Any) -> int:
    try:
        first_tracks = {int(point.point3D_id) for point in first.points2D if point.has_point3D()}
        second_tracks = {int(point.point3D_id) for point in second.points2D if point.has_point3D()}
        return len(first_tracks & second_tracks)
    except Exception:
        return 0


def build_independent_sparse_audit_set(
    database: Any,
    ring_metadata: Mapping[str, Mapping[str, Any]],
    *,
    config: SparseIntegrityConfig = SparseIntegrityConfig(),
) -> dict[str, Any]:
    """Freeze a mapper-independent set of verified acquisition-local pairs.

    The set is generated from the SQLite-consistent source snapshot before any
    graph pruning.  It contains every non-duplicate local pair in the bounded
    acquisition window plus every verified cross-ring pair above the configured
    bridge floor.  A mapper may omit a pair for a candidate, but it cannot omit
    that pair from this acceptance audit.
    """

    config.validate()
    images = {int(image.image_id): image for image in database.read_all_images()}
    ids, geometries = database.read_two_view_geometries()
    selected: list[dict[str, Any]] = []
    for pair_id, geometry in zip(ids, geometries):
        try:
            first_id, second_id = decode_colmap_pair_id(int(pair_id))
        except ValueError:
            continue
        first_image, second_image = images.get(first_id), images.get(second_id)
        if first_image is None or second_image is None:
            continue
        first_name, second_name = str(first_image.name), str(second_image.name)
        first_meta, second_meta = ring_metadata.get(first_name), ring_metadata.get(second_name)
        if first_meta is None or second_meta is None:
            continue
        inliers = int(len(np.asarray(getattr(geometry, "inlier_matches", np.empty((0, 2))))))
        same_ring = str(first_meta["ring"]) == str(second_meta["ring"])
        frame_distance = (
            abs(int(first_meta.get("frame_index", 0)) - int(second_meta.get("frame_index", 0)))
            if same_ring
            else None
        )
        keep = (
            (same_ring and not is_near_duplicate(first_name, second_name) and frame_distance is not None
             and frame_distance <= config.same_ring_window and inliers >= config.minimum_verified_inliers)
            or (not same_ring and inliers >= config.minimum_cross_ring_inliers)
        )
        if not keep:
            continue
        selected.append(
            {
                "pair_id": int(pair_id),
                "first": first_name,
                "second": second_name,
                "first_ring": str(first_meta["ring"]),
                "second_ring": str(second_meta["ring"]),
                "same_ring": bool(same_ring),
                "frame_distance": frame_distance,
                "verified_inliers": inliers,
            }
        )
    selected.sort(key=lambda item: (int(item["pair_id"]), str(item["first"]), str(item["second"])))
    pair_ids = [int(item["pair_id"]) for item in selected]
    pair_digest = hashlib.sha256(
        json.dumps(pair_ids, separators=(",", ":"), sort_keys=False).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "method": "all non-duplicate verified local pairs plus all verified cross-ring bridges from the SQLite snapshot",
        "pair_count": len(selected),
        "same_ring_count": sum(1 for item in selected if item["same_ring"]),
        "cross_ring_count": sum(1 for item in selected if not item["same_ring"]),
        "pair_ids_sha256": pair_digest,
        "pairs": selected,
        "config": asdict(config),
    }


def _image_has_pose(image: Any) -> bool:
    """Support both pyCOLMAP's property and older callable pose APIs."""

    try:
        value = getattr(image, "has_pose")
        return bool(value() if callable(value) else value)
    except Exception:
        return False


def _mad(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    median = float(np.median(np.asarray(values, dtype=np.float64)))
    return float(np.median(np.abs(np.asarray(values, dtype=np.float64) - median)))


def _trajectory_outliers(
    records: Sequence[Mapping[str, Any]], config: SparseIntegrityConfig
) -> tuple[set[tuple[str, str]], dict[str, Any]]:
    by_ring: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        # Trajectory statistics are meaningful only for same-ring acquisition
        # neighbors.  Cross-ring bridges can have large, legitimate baselines
        # and must not distort the robust local-jump threshold.
        if record.get("same_ring", True) and record.get("ring") and record.get("center_jump") is not None:
            by_ring[str(record["ring"])].append(record)
    outliers: set[tuple[str, str]] = set()
    summaries: dict[str, Any] = {}
    for ring, values in sorted(by_ring.items()):
        jumps = [float(item["center_jump"]) for item in values if math.isfinite(float(item["center_jump"]))]
        rays = [float(item["ray_jump_deg"]) for item in values if item.get("ray_jump_deg") is not None]
        median = float(np.median(jumps)) if jumps else 0.0
        threshold = median + config.trajectory_mad_scale * max(_mad(jumps), 1e-9) if jumps else float("inf")
        ray_median = float(np.median(rays)) if rays else 0.0
        ray_threshold = ray_median + config.trajectory_mad_scale * max(_mad(rays), 1e-9) if rays else float("inf")
        flagged = 0
        for item in values:
            center_jump = item.get("center_jump")
            ray_jump = item.get("ray_jump_deg")
            is_outlier = (
                center_jump is not None and float(center_jump) > threshold
            ) or (ray_jump is not None and float(ray_jump) > ray_threshold)
            if is_outlier:
                outliers.add(canonical_pair(str(item["first"]), str(item["second"])))
                flagged += 1
        summaries[ring] = {
            "evaluated_edges": len(values),
            "center_jump_median": median,
            "center_jump_mad": _mad(jumps),
            "center_jump_threshold": threshold,
            "ray_jump_median_deg": ray_median,
            "ray_jump_mad_deg": _mad(rays),
            "ray_jump_threshold_deg": ray_threshold,
            "outlier_edges": flagged,
        }
    return outliers, summaries


def collect_sparse_pair_evidence(
    database: Any,
    reconstruction: Any,
    ring_metadata: Mapping[str, Mapping[str, Any]],
    *,
    config: SparseIntegrityConfig = SparseIntegrityConfig(),
    pair_ids: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    """Measure verified two-view vs final-pose evidence for local and bridge edges.

    Same-ring records are restricted to the bounded acquisition-local window
    and receive the strict final-pose/shared-track checks.  Strong cross-ring
    records are retained as connectivity evidence, but are deliberately not
    judged by the same local rotation threshold because their legitimate
    baseline can be much larger.
    """

    config.validate()
    images = {int(image.image_id): image for image in database.read_all_images()}
    names_to_ids = {str(image.name): image_id for image_id, image in images.items()}
    ids, geometries = database.read_two_view_geometries()
    allowed = {int(value) for value in pair_ids} if pair_ids is not None else None
    keypoints: dict[int, np.ndarray] = {}
    evidence: list[dict[str, Any]] = []
    for pair_id, geometry in zip(ids, geometries):
        if allowed is not None and int(pair_id) not in allowed:
            continue
        try:
            first_id, second_id = decode_colmap_pair_id(int(pair_id))
        except ValueError:
            continue
        first_image, second_image = images.get(first_id), images.get(second_id)
        if first_image is None or second_image is None:
            continue
        first_name, second_name = str(first_image.name), str(second_image.name)
        first_meta, second_meta = ring_metadata.get(first_name), ring_metadata.get(second_name)
        if first_meta is None or second_meta is None:
            continue
        same_ring = first_meta.get("ring") == second_meta.get("ring")
        frame_distance = (
            abs(int(first_meta.get("frame_index", 0)) - int(second_meta.get("frame_index", 0)))
            if same_ring
            else None
        )
        # Keep only local same-ring edges and sufficiently strong cross-ring
        # bridges.  The latter threshold is lower than the local graph's
        # selection threshold only when the verified evidence explicitly meets
        # the configured bridge floor.
        inliers = int(len(np.asarray(getattr(geometry, "inlier_matches", np.empty((0, 2))))))
        if same_ring and frame_distance is not None and frame_distance > config.same_ring_window:
            continue
        if not same_ring and inliers < config.minimum_cross_ring_inliers:
            continue
        duplicate = is_near_duplicate(first_name, second_name)
        try:
            if first_id not in keypoints:
                keypoints[first_id] = np.asarray(database.read_keypoints(first_id))
            if second_id not in keypoints:
                keypoints[second_id] = np.asarray(database.read_keypoints(second_id))
            camera = database.read_camera(int(first_image.camera_id))
            two_rotation_matrix, rotation_reason = _recover_two_view_rotation(
                geometry, keypoints[first_id], keypoints[second_id], camera
            )
        except Exception as error:
            two_rotation_matrix, rotation_reason = None, f"two_view_inputs_error:{type(error).__name__}"
        try:
            final_first = reconstruction.find_image_with_name(first_name)
            final_second = reconstruction.find_image_with_name(second_name)
            final_rotation, center_jump, ray_jump, final_rotation_matrix = _final_pose_metrics(
                final_first, final_second
            )
            shared_tracks = _shared_track_count(final_first, final_second)
            registered = bool(_image_has_pose(final_first) and _image_has_pose(final_second))
        except Exception:
            final_rotation, center_jump, ray_jump, final_rotation_matrix = None, None, None, None
            shared_tracks, registered = 0, False
        two_rotation = _rotation_angle_deg(two_rotation_matrix)
        disagreement = _rotation_geodesic_deg(final_rotation_matrix, two_rotation_matrix)
        evidence.append(
            {
                "pair_id": int(pair_id),
                "first": first_name,
                "second": second_name,
                "ring": str(first_meta["ring"]),
                "second_ring": str(second_meta["ring"]),
                "same_ring": bool(same_ring),
                "first_frame_index": int(first_meta.get("frame_index", 0)),
                "second_frame_index": int(second_meta.get("frame_index", 0)),
                "frame_distance": frame_distance,
                "verified_inliers": inliers,
                "two_view_rotation_deg": two_rotation,
                "two_view_rotation_matrix_first_to_second": (
                    two_rotation_matrix.tolist() if two_rotation_matrix is not None else None
                ),
                "two_view_rotation_reason": rotation_reason,
                "final_rotation_deg": final_rotation,
                "final_rotation_matrix_first_to_second": (
                    final_rotation_matrix.tolist() if final_rotation_matrix is not None else None
                ),
                "rotation_disagreement_deg": disagreement,
                "shared_final_tracks": shared_tracks,
                "camera_center_jump": center_jump,
                "ray_jump_deg": ray_jump,
                "registered": registered,
                "near_duplicate": duplicate,
                "evaluable": bool(
                    same_ring
                    and not duplicate
                    and inliers >= config.minimum_verified_inliers
                    and two_rotation_matrix is not None
                    and final_rotation is not None
                ),
            }
        )
    outliers, trajectory = _trajectory_outliers(evidence, config)
    for item in evidence:
        item["trajectory_outlier"] = canonical_pair(item["first"], item["second"]) in outliers
        reasons: list[str] = []
        if item["near_duplicate"]:
            reasons.append("near_duplicate_frame")
        if item["verified_inliers"] < config.minimum_verified_inliers:
            reasons.append("weak_verified_support")
        if item.get("same_ring", True) and item["rotation_disagreement_deg"] is not None and item["rotation_disagreement_deg"] > config.maximum_rotation_disagreement_deg:
            reasons.append("rotation_disagreement")
        if item.get("same_ring", True) and item["evaluable"] and item["shared_final_tracks"] < config.minimum_shared_tracks:
            reasons.append("shared_track_collapse")
        if item.get("same_ring", True) and item["trajectory_outlier"]:
            reasons.append("trajectory_outlier")
        item["failure_reasons"] = reasons
        item["passes_integrity"] = bool(item["evaluable"] and not reasons)
    # Keep the trajectory summaries available to callers without duplicating a
    # second traversal.  The list itself remains JSON-friendly.
    if evidence:
        evidence[0]["_trajectory_summaries"] = trajectory
    return evidence


def _ring_graph_connectivity(
    rings: Sequence[str],
    pair_evidence: Sequence[Mapping[str, Any]],
    ring_metadata: Mapping[str, Mapping[str, Any]],
    registered: set[str],
    *,
    minimum_cross_ring_inliers: int,
) -> dict[str, Any]:
    """Evaluate actual undirected ring-graph connectivity, not edge count."""

    normalized_rings = {str(value) for value in rings}
    adjacency: dict[str, set[str]] = {ring: set() for ring in normalized_rings}
    edges: set[tuple[str, str]] = set()
    for item in pair_evidence:
        first, second = str(item.get("first", "")), str(item.get("second", ""))
        first_ring = str(ring_metadata.get(first, {}).get("ring", item.get("ring", "")))
        second_ring = str(ring_metadata.get(second, {}).get("ring", item.get("second_ring", "")))
        if not first_ring or not second_ring or first_ring == second_ring:
            continue
        if first not in registered or second not in registered:
            continue
        if int(item.get("verified_inliers", 0)) < minimum_cross_ring_inliers:
            continue
        if first_ring not in adjacency or second_ring not in adjacency:
            continue
        edge = tuple(sorted((first_ring, second_ring)))
        edges.add(edge)
        adjacency[first_ring].add(second_ring)
        adjacency[second_ring].add(first_ring)
    components: list[list[str]] = []
    unseen = set(normalized_rings)
    while unseen:
        start = min(unseen)
        stack = [start]
        component: set[str] = set()
        while stack:
            ring = stack.pop()
            if ring in component:
                continue
            component.add(ring)
            unseen.discard(ring)
            stack.extend(sorted(adjacency[ring] - component))
        components.append(sorted(component))
    components.sort(key=lambda value: value[0] if value else "")
    return {
        "connected": bool(normalized_rings) and len(components) == 1,
        "rings": sorted(normalized_rings),
        "edges": [list(value) for value in sorted(edges)],
        "components": components,
    }


def sparse_integrity_gate(
    reconstruction: Any,
    ring_metadata: Mapping[str, Mapping[str, Any]],
    pair_evidence: Sequence[Mapping[str, Any]],
    *,
    config: SparseIntegrityConfig = SparseIntegrityConfig(),
    independent_audit_evidence: Sequence[Mapping[str, Any]] | None = None,
    expected_image_names: Sequence[str] | None = None,
    model_root: str | Path | None = None,
    database_lineage: Mapping[str, Any] | None = None,
    expected_model_sha256: str | None = None,
    audit_exclusions: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply strict sparse acceptance against a fixed, independent audit set.

    ``pair_evidence`` remains accepted for backwards-compatible diagnostics,
    but production acceptance must pass ``independent_audit_evidence`` built
    from the unpruned SQLite snapshot.  This prevents graph pruning from
    deleting contradictory edges from the evidence used by the gate.
    """

    config.validate()
    audit_evidence = list(independent_audit_evidence) if independent_audit_evidence is not None else list(pair_evidence)
    independent_bound = independent_audit_evidence is not None
    registered: set[str] = set()
    try:
        registered = {str(reconstruction.image(int(image_id)).name) for image_id in reconstruction.reg_image_ids()}
    except Exception:
        registered = set()
    rings = sorted({str(value.get("ring")) for value in ring_metadata.values() if value.get("selected", True)})
    ring_counts = {
        ring: {
            "selected": sum(1 for value in ring_metadata.values() if value.get("selected", True) and value.get("ring") == ring),
            "registered": sum(1 for name in registered if ring_metadata.get(name, {}).get("ring") == ring),
        }
        for ring in rings
    }
    for value in ring_counts.values():
        value["registration_fraction"] = value["registered"] / value["selected"] if value["selected"] else 0.0
    expected = {Path(str(name)).name for name in expected_image_names} if expected_image_names is not None else None
    model_count: int | None = None
    model_sha256: str | None = None
    if model_root is not None:
        root = Path(model_root).resolve()
        if root.is_dir() and all((root / filename).is_file() for filename in ("cameras.bin", "images.bin", "points3D.bin")):
            model_count = 1
            model_sha256 = stable_directory_sha256(root)
        elif root.is_dir():
            candidates = [
                child for child in sorted(root.iterdir(), key=lambda value: value.name)
                if child.is_dir() and all((child / filename).is_file() for filename in ("cameras.bin", "images.bin", "points3D.bin"))
            ]
            model_count = len(candidates)
            if model_count == 1:
                model_sha256 = stable_directory_sha256(candidates[0])
    model_hash_valid = bool(model_sha256 and (expected_model_sha256 is None or model_sha256 == str(expected_model_sha256).lower()))

    # Local pose/track checks apply only to same-ring evidence.  Cross-ring
    # records are consumed by the actual ring-graph connectivity check below.
    exclusion_keys = {
        canonical_pair(*str(key).split("|", 1))
        for key in (audit_exclusions or {})
        if "|" in str(key)
    }
    relevant = [
        item
        for item in audit_evidence
        if item.get("same_ring", True)
        and not item.get("near_duplicate")
        # The fixed audit remains complete and hash-bound, but local pose and
        # track acceptance must use only rotations that the independent raw
        # correspondence classifier marked well-conditioned.  Older callers
        # do not carry this field and retain the historical behavior.
        and bool(item.get("calibrated_conditioned", True))
        and int(item.get("verified_inliers", 0)) >= config.minimum_verified_inliers
        and canonical_pair(str(item.get("first")), str(item.get("second"))) not in exclusion_keys
    ]
    excluded_audit_items = [
        item for item in audit_evidence
        if canonical_pair(str(item.get("first")), str(item.get("second"))) in exclusion_keys
    ]
    audited_keys = {
        canonical_pair(str(item.get("first")), str(item.get("second")))
        for item in audit_evidence
    }
    exclusion_justifications_valid = not audit_exclusions or all(
        isinstance(item, Mapping)
        and str(item.get("status", "")) in {"independently_degenerate", "independently_erroneous"}
        and str(item.get("evidence", "")).strip()
        for item in (audit_exclusions or {}).values()
    ) and exclusion_keys.issubset(audited_keys)
    disagreement_failures = [item for item in relevant if item.get("rotation_disagreement_deg") is None or float(item["rotation_disagreement_deg"]) > config.maximum_rotation_disagreement_deg]
    track_failures = [item for item in relevant if item.get("rotation_disagreement_deg") is not None and int(item.get("shared_final_tracks", 0)) < config.minimum_shared_tracks]
    trajectory_failures = [item for item in relevant if bool(item.get("trajectory_outlier"))]
    trajectory_summaries = {}
    for item in audit_evidence:
        if isinstance(item.get("_trajectory_summaries"), Mapping):
            trajectory_summaries = dict(item["_trajectory_summaries"])
            break
    ring_connectivity = _ring_graph_connectivity(
        rings,
        audit_evidence,
        ring_metadata,
        registered,
        minimum_cross_ring_inliers=config.minimum_cross_ring_inliers,
    )
    database_lineage_valid = bool(
        isinstance(database_lineage, Mapping)
        and re.fullmatch(r"[0-9a-fA-F]{64}", str(database_lineage.get("snapshot_sha256", "")))
        and re.fullmatch(r"[0-9a-fA-F]{64}", str(database_lineage.get("logical_sha256", "")))
    )
    camera_integrity = False
    camera_summary: list[dict[str, Any]] = []
    try:
        camera_values = list(getattr(reconstruction, "cameras", {}).values())
        camera_integrity = bool(camera_values)
        for camera in camera_values:
            params = np.asarray(camera.params, dtype=np.float64).reshape(-1)
            finite = bool(params.size >= 3 and np.isfinite(params).all())
            focal_positive = bool(params.size >= 1 and float(params[0]) > 0.0)
            camera_integrity = camera_integrity and finite and focal_positive
            camera_summary.append({
                "camera_id": int(getattr(camera, "camera_id", -1)),
                "model": str(getattr(camera, "model_name", "")),
                "params": [float(value) for value in params],
                "finite": finite,
                "focal_positive": focal_positive,
            })
    except Exception:
        camera_integrity = False
    point_count = 0
    observation_count = 0
    mean_reprojection_error = float("inf")
    try:
        point_count = int(reconstruction.num_points3D())
        observation_count = int(reconstruction.compute_num_observations())
        mean_reprojection_error = float(reconstruction.compute_mean_reprojection_error())
    except Exception:
        pass
    checks = {
        "independent_audit_evidence_bound": independent_bound and bool(audit_evidence),
        "audit_exclusion_justifications_bound": exclusion_justifications_valid,
        "all_selected_rings_registered": bool(ring_counts) and all(
            value["registration_fraction"] >= config.minimum_ring_registration_fraction
            for value in ring_counts.values()
        ),
        "registered_image_set_exact": expected is not None and registered == expected,
        "same_ring_pose_consistency": not disagreement_failures and exclusion_justifications_valid,
        "shared_track_continuity": not track_failures and exclusion_justifications_valid,
        "trajectory_continuity": not trajectory_failures and exclusion_justifications_valid,
        "cross_ring_graph_connected": bool(ring_connectivity["connected"]),
        "single_coherent_model": model_count == 1 and bool(expected) and registered == expected and int(reconstruction.num_reg_images()) == len(expected),
        "stable_model_hash_bound": model_hash_valid,
        "sqlite_snapshot_lineage_bound": database_lineage_valid,
        "camera_intrinsics_integrity": camera_integrity,
        "sparse_point_health": point_count >= config.minimum_sparse_points,
        "observation_health": observation_count >= config.minimum_sparse_observations,
        "mean_reprojection_health": math.isfinite(mean_reprojection_error)
        and mean_reprojection_error <= config.maximum_mean_reprojection_error,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "ring_coverage": ring_counts,
        "registered_images": len(registered),
        "camera_summary": camera_summary,
        "sparse_points": point_count,
        "observations": observation_count,
        "mean_reprojection_error": mean_reprojection_error,
        "pair_count": len(pair_evidence),
        "independent_audit_pair_count": len(audit_evidence),
        "excluded_audit_pair_count": len(excluded_audit_items),
        "audit_exclusions": {str(key): dict(value) for key, value in (audit_exclusions or {}).items()},
        "relevant_pair_count": len(relevant),
        "disagreement_failure_count": len(disagreement_failures),
        "shared_track_failure_count": len(track_failures),
        "trajectory_failure_count": len(trajectory_failures),
        "cross_ring_connections": len(ring_connectivity["edges"]),
        "cross_ring_pairs": ring_connectivity["edges"],
        "ring_graph_connectivity": ring_connectivity,
        "model_count": model_count,
        "model_sha256": model_sha256,
        "database_lineage": dict(database_lineage or {}),
        "audit_evidence_source": "independent_sqlite_snapshot" if independent_bound else "caller_pair_evidence_diagnostic_only",
        "trajectory_summaries": trajectory_summaries,
        "failures": {
            "rotation": [dict(item) for item in disagreement_failures],
            "shared_tracks": [dict(item) for item in track_failures],
            "trajectory": [dict(item) for item in trajectory_failures],
        },
        "config": asdict(config),
    }


def build_corrected_sparse_graph(
    database: Any,
    ring_metadata: Mapping[str, Mapping[str, Any]],
    *,
    ring_order: Sequence[str] = (),
    config: SparseIntegrityConfig = SparseIntegrityConfig(),
    same_ring_inliers: int = 80,
    cross_ring_inliers: int | None = None,
    max_cross_edges_per_image: int = 2,
) -> dict[str, Any]:
    """Select a deterministic repair graph from verified cached geometry.

    Same-ring edges are local and exclude duplicate-frame degeneracies.  Cross-
    ring edges are ranked by verified support, with a per-image cap to avoid
    letting a repeated texture cluster dominate a ring bridge.
    """

    config.validate()
    if same_ring_inliers < 1 or max_cross_edges_per_image < 1:
        raise ValueError("graph thresholds must be positive")
    cross_threshold = int(cross_ring_inliers or config.minimum_cross_ring_inliers)
    images = {int(image.image_id): image for image in database.read_all_images()}
    ids, geometries = database.read_two_view_geometries()
    candidates: list[dict[str, Any]] = []
    for pair_id, geometry in zip(ids, geometries):
        try:
            first_id, second_id = decode_colmap_pair_id(int(pair_id))
        except ValueError:
            continue
        first_image, second_image = images.get(first_id), images.get(second_id)
        if first_image is None or second_image is None:
            continue
        first, second = str(first_image.name), str(second_image.name)
        first_meta, second_meta = ring_metadata.get(first), ring_metadata.get(second)
        if first_meta is None or second_meta is None:
            continue
        inliers = int(len(np.asarray(getattr(geometry, "inlier_matches", np.empty((0, 2))))))
        same_ring = first_meta["ring"] == second_meta["ring"]
        frame_distance = abs(int(first_meta["frame_index"]) - int(second_meta["frame_index"])) if same_ring else None
        if same_ring:
            selected = bool(
                not is_near_duplicate(first, second)
                and frame_distance is not None
                and frame_distance <= config.same_ring_window
                and inliers >= same_ring_inliers
            )
            reason = "same_ring_local_verified" if selected else "same_ring_duplicate_or_weak_or_nonlocal"
        else:
            selected = bool(inliers >= cross_threshold)
            reason = "cross_ring_verified_support_pending_cap" if selected else "cross_ring_weak_verified_support"
        candidates.append({
            "pair_id": int(pair_id),
            "first": first,
            "second": second,
            "first_ring": str(first_meta["ring"]),
            "second_ring": str(second_meta["ring"]),
            "same_ring": same_ring,
            "frame_distance": frame_distance,
            "verified_inliers": inliers,
            "selected": selected,
            "reason": reason,
        })
    # Cap cross-ring edges per image deterministically, while retaining at
    # least the strongest edge for every adjacent ring boundary.
    selected = [item for item in candidates if item["selected"] and item["same_ring"]]
    cross = [item for item in candidates if item["selected"] and not item["same_ring"]]
    cap_count: Counter[str] = Counter()
    boundary_order = {str(ring): index for index, ring in enumerate(ring_order)}
    cross.sort(key=lambda item: (-int(item["verified_inliers"]), item["first"], item["second"]))
    for item in cross:
        if cap_count[item["first"]] >= max_cross_edges_per_image or cap_count[item["second"]] >= max_cross_edges_per_image:
            item["selected"] = False
            item["reason"] = "cross_ring_per_image_cap"
            continue
        if boundary_order:
            left, right = sorted((item["first_ring"], item["second_ring"]), key=lambda value: boundary_order.get(value, 10**6))
            if abs(boundary_order.get(right, 10**6) - boundary_order.get(left, -10**6)) != 1:
                item["selected"] = False
                item["reason"] = "non_adjacent_ring_edge"
                continue
        cap_count[item["first"]] += 1
        cap_count[item["second"]] += 1
        selected.append(item)
    # Ensure each adjacent ring boundary has a strong bridge when evidence
    # exists, even if the per-image cap consumed the early candidates.
    if boundary_order:
        for left, right in zip(tuple(ring_order[:-1]), tuple(ring_order[1:])):
            boundary = [item for item in candidates if not item["same_ring"] and {item["first_ring"], item["second_ring"]} == {str(left), str(right)} and int(item["verified_inliers"]) >= cross_threshold]
            if not any({item["first_ring"], item["second_ring"]} == {str(left), str(right)} for item in selected):
                if boundary:
                    rescue = max(boundary, key=lambda item: (int(item["verified_inliers"]), item["first"], item["second"]))
                    if rescue not in selected:
                        rescue["selected"] = True
                        rescue["reason"] = "boundary_bridge_rescue"
                        selected.append(rescue)
    selected.sort(key=lambda item: (str(item["first"]), str(item["second"])))
    pairs = [list(canonical_pair(str(item["first"]), str(item["second"]))) for item in selected]
    return {
        "schema_version": 1,
        "status": "complete",
        "method": "verified ALIKED/LightGlue COLMAP two-view graph; acquisition-local same-ring plus capped adjacent-ring bridges",
        "pair_count": len(pairs),
        "pairs": pairs,
        "selected_edges": selected,
        "candidate_count": len(candidates),
        "same_ring_selected": sum(1 for item in selected if item["same_ring"]),
        "cross_ring_selected": sum(1 for item in selected if not item["same_ring"]),
        "ring_order": [str(value) for value in ring_order],
        "config": {
            "same_ring_inliers": same_ring_inliers,
            "cross_ring_inliers": cross_threshold,
            "max_cross_edges_per_image": max_cross_edges_per_image,
            "same_ring_window": config.same_ring_window,
        },
        "graph_sha256": hashlib.sha256(json.dumps(pairs, sort_keys=True).encode("utf-8")).hexdigest(),
    }


def write_sparse_graph_report(payload: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"sparse graph report already exists; preserve it: {target}")
    return write_json(target, dict(payload))


def prune_colmap_database(
    source: str | Path,
    destination: str | Path,
    selected_pairs: Sequence[Sequence[str]],
    *,
    image_ids_by_name: Mapping[str, int],
    snapshot_manifest: str | Path,
) -> Path:
    """Snapshot a COLMAP DB and retain only the selected verified pair rows.

    ``source`` must be the manifest-bound frozen canonical snapshot.  It is
    copied as bytes to a disposable working database before SQLite opens it,
    so the canonical file cannot acquire WAL/SHM state during pruning.
    """

    manifest = load_canonical_sqlite_snapshot_manifest(snapshot_manifest)
    source_path, destination_path = Path(source).resolve(), Path(destination).resolve()
    if source_path != Path(manifest["canonical_path"]).resolve():
        raise ValueError("pruning source must equal the canonical_path recorded in snapshot_manifest")
    if not source_path.is_file():
        raise ValueError(f"source COLMAP database is missing: {source_path}")
    if destination_path.exists():
        raise FileExistsError(f"destination database already exists; preserve it: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    copy_frozen_sqlite_snapshot(source_path, destination_path)
    pair_ids = [
        colmap_pair_id(image_ids_by_name[str(pair[0])], image_ids_by_name[str(pair[1])])
        for pair in selected_pairs
    ]
    with sqlite3.connect(str(destination_path)) as connection:
        connection.execute("CREATE TEMP TABLE selected_repair_pairs(pair_id INTEGER PRIMARY KEY)")
        connection.executemany("INSERT INTO selected_repair_pairs(pair_id) VALUES (?)", ((int(value),) for value in sorted(set(pair_ids))))
        connection.execute("DELETE FROM matches WHERE pair_id NOT IN (SELECT pair_id FROM selected_repair_pairs)")
        connection.execute("DELETE FROM two_view_geometries WHERE pair_id NOT IN (SELECT pair_id FROM selected_repair_pairs)")
        connection.commit()
    return destination_path


def parse_patchmatch_reference_config(path: str | Path) -> list[str]:
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if len(lines) % 2:
        raise ValueError(f"PatchMatch config has an incomplete reference/source pair: {path}")
    return [Path(lines[index]).name for index in range(0, len(lines), 2)]


def validate_one_reference_write(config_paths: Sequence[str | Path], expected_references: Sequence[str]) -> dict[str, Any]:
    """Prove exact one-reference-one-write provenance across complete tiles."""

    expected = {Path(str(name)).name for name in expected_references}
    counts: Counter[str] = Counter()
    per_config: dict[str, int] = {}
    for path in config_paths:
        refs = parse_patchmatch_reference_config(path)
        per_config[str(Path(path))] = len(refs)
        counts.update(refs)
    duplicates = {name: count for name, count in sorted(counts.items()) if count != 1}
    missing = sorted(expected - set(counts))
    unexpected = sorted(set(counts) - expected)
    return {
        "passed": not duplicates and not missing and not unexpected and set(counts) == expected,
        "expected_reference_count": len(expected),
        "reference_entry_count": int(sum(counts.values())),
        "unique_reference_count": len(counts),
        "duplicate_or_non_unit_references": duplicates,
        "missing_references": missing,
        "unexpected_references": unexpected,
        "per_config_reference_counts": per_config,
    }


def raw_component_gate(metrics: Mapping[str, Any], *, dominant_min: float = 0.95, second_max: float = 0.02) -> dict[str, Any]:
    """Hard continuity gate for Poisson output; missing evidence fails closed."""

    components = metrics.get("components") if isinstance(metrics.get("components"), Mapping) else metrics
    fractions = components.get("face_fractions") or components.get("face_fraction")
    if fractions is None:
        counts = components.get("face_counts")
        total = float(sum(counts)) if isinstance(counts, Sequence) and counts else 0.0
        fractions = [float(value) / total for value in counts] if total else []
    fractions = sorted((float(value) for value in fractions if math.isfinite(float(value))), reverse=True)
    dominant = fractions[0] if fractions else 0.0
    second = fractions[1] if len(fractions) > 1 else 0.0
    checks = {
        "component_evidence_available": bool(fractions),
        "dominant_face_fraction": dominant >= dominant_min,
        "second_largest_face_fraction": second <= second_max,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "component_count": len(fractions),
        "dominant_face_fraction": dominant,
        "second_largest_face_fraction": second,
        "thresholds": {"dominant_min": dominant_min, "second_max": second_max},
    }


__all__ = [
    "SparseIntegrityConfig",
    "build_corrected_sparse_graph",
    "build_independent_sparse_audit_set",
    "canonical_pair",
    "copy_frozen_sqlite_snapshot",
    "create_disposable_sqlite_from_manifest",
    "create_sqlite_snapshot",
    "collect_sparse_pair_evidence",
    "colmap_pair_id",
    "decode_colmap_pair_id",
    "duplicate_stem",
    "is_near_duplicate",
    "load_ring_metadata",
    "load_canonical_sqlite_snapshot_manifest",
    "parse_patchmatch_reference_config",
    "prune_colmap_database",
    "raw_component_gate",
    "sparse_integrity_gate",
    "sqlite_logical_digest",
    "stable_directory_sha256",
    "validate_one_reference_write",
    "write_sparse_graph_report",
]
