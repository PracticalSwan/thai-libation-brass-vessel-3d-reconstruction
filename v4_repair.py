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
    minimum_multi_view_track_fraction: float = 0.50
    minimum_track_length_p50: float = 3.0
    minimum_track_length_p90: float = 4.0
    minimum_view_mask_precision_p10: float = 0.90
    minimum_view_mask_recall_p10: float = 0.60
    minimum_view_mask_iou_p10: float = 0.60
    minimum_ring_mask_precision_mean: float = 0.90
    minimum_ring_mask_recall_mean: float = 0.60
    minimum_ring_mask_iou_mean: float = 0.60
    maximum_translation_direction_p90_deg: float = 35.0
    minimum_translation_positive_scale_fraction: float = 0.95
    maximum_translation_orthogonal_residual_p90: float = 0.60

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
        if not 0.0 < self.minimum_multi_view_track_fraction <= 1.0:
            raise ValueError("minimum_multi_view_track_fraction must be in (0, 1]")
        if self.minimum_track_length_p50 < 2.0 or self.minimum_track_length_p90 < 2.0:
            raise ValueError("track-length quantile thresholds must be at least two views")
        for name in (
            "minimum_view_mask_precision_p10",
            "minimum_view_mask_recall_p10",
            "minimum_view_mask_iou_p10",
            "minimum_ring_mask_precision_mean",
            "minimum_ring_mask_recall_mean",
            "minimum_ring_mask_iou_mean",
            "minimum_translation_positive_scale_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not math.isfinite(self.maximum_translation_direction_p90_deg) or not 0.0 < self.maximum_translation_direction_p90_deg <= 180.0:
            raise ValueError("maximum_translation_direction_p90_deg must be in (0, 180]")
        if not math.isfinite(self.maximum_translation_orthogonal_residual_p90) or self.maximum_translation_orthogonal_residual_p90 < 0.0:
            raise ValueError("maximum_translation_orthogonal_residual_p90 must be nonnegative")
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


def validate_exact_mapper_graph(
    database: str | Path,
    expected_pair_ids: Iterable[int],
    *,
    canonical_path: str | Path | None = None,
    require_raw_matches: bool = True,
) -> dict[str, Any]:
    """Fail closed unless a disposable mapper DB has exactly the expected graph.

    COLMAP reconstruction consumes verified inlier correspondences from
    ``two_view_geometries``.  Raw ``matches`` are pruned as a defensive
    lineage boundary so a future runtime cannot reintroduce a non-retained
    pair through feature-matching input.  This helper only opens ``database``
    and rejects the manifest-bound canonical path when it is supplied.
    """

    database_path = Path(database).resolve()
    if not database_path.is_file():
        raise ValueError(f"mapper database is missing: {database_path}")
    if canonical_path is not None and database_path == Path(canonical_path).resolve():
        raise ValueError("mapper preflight may only open a disposable database copy")
    expected = {int(pair_id) for pair_id in expected_pair_ids}
    readonly_uri = f"file:{database_path.as_posix()}?mode=ro&immutable=1"
    connection = sqlite3.connect(readonly_uri, uri=True)
    try:
        table_names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('matches', 'two_view_geometries')"
            )
        }
        if "two_view_geometries" not in table_names:
            raise ValueError("mapper database is missing two_view_geometries")
        verified = {
            int(row[0])
            for row in connection.execute(
                "SELECT pair_id FROM two_view_geometries WHERE rows > 0"
            )
        }
        verified_inlier_rows = int(
            connection.execute(
                "SELECT COALESCE(SUM(rows), 0) FROM two_view_geometries WHERE rows > 0"
            ).fetchone()[0]
            or 0
        )
        raw_matches = (
            {
                int(row[0])
                for row in connection.execute("SELECT pair_id FROM matches WHERE rows > 0")
            }
            if "matches" in table_names
            else set()
        )
        raw_match_rows = (
            int(connection.execute("SELECT COALESCE(SUM(rows), 0) FROM matches WHERE rows > 0").fetchone()[0] or 0)
            if "matches" in table_names
            else 0
        )
    finally:
        connection.close()
    extra_verified = sorted(verified - expected)
    missing_verified = sorted(expected - verified)
    extra_raw = sorted(raw_matches - expected)
    missing_raw = sorted(expected - raw_matches)
    verified_exact = verified == expected
    raw_exact = raw_matches == expected
    expected_pair_ids_sha256 = hashlib.sha256(
        json.dumps(sorted(expected), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    verified_pair_ids_sha256 = hashlib.sha256(
        json.dumps(sorted(verified), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    raw_match_pair_ids_sha256 = hashlib.sha256(
        json.dumps(sorted(raw_matches), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    checks = {
        "two_view_geometry_pair_ids_exact": verified_exact,
        "raw_match_pair_ids_exact": raw_exact if require_raw_matches else True,
        "two_view_geometry_count_exact": len(verified) == len(expected),
        "raw_match_pair_count_exact": len(raw_matches) == len(expected) if require_raw_matches else True,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "two_view_geometry_pair_ids_exact": verified_exact,
        "raw_match_pair_ids_exact": raw_exact,
        "database_path": str(database_path),
        "expected_pair_count": len(expected),
        "expected_pair_ids_sha256": expected_pair_ids_sha256,
        "nonempty_two_view_geometry_count": len(verified),
        "nonempty_two_view_geometry_inlier_rows": verified_inlier_rows,
        "actual_two_view_geometry_pair_ids_sha256": verified_pair_ids_sha256,
        "extra_two_view_geometry_pair_ids": extra_verified,
        "missing_two_view_geometry_pair_ids": missing_verified,
        "raw_match_pair_count": len(raw_matches),
        "raw_match_rows": raw_match_rows,
        "actual_raw_match_pair_ids_sha256": raw_match_pair_ids_sha256,
        "extra_raw_match_pair_ids": extra_raw,
        "missing_raw_match_pair_ids": missing_raw,
        "raw_matches_required_for_track_establishment": False,
        "track_loading_path": (
            "pycolmap 4.2 global_mapping -> DatabaseCache/CorrespondenceGraph "
            "from two_view_geometries.inlier_matches; raw matches retained only "
            "as defensive source lineage"
        ),
    }


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


def sparse_track_distribution_evidence(
    reconstruction: Any,
    *,
    config: SparseIntegrityConfig = SparseIntegrityConfig(),
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure genuine reconstruction track lengths independently of pair evidence.

    A mapper report that allocates one disjoint two-view point per selected pair
    can make a minimum shared-track check look healthy without establishing a
    globally connected SfM model.  This measurement reads the model's actual
    track elements and rejects that construction explicitly.  The provenance
    binding is intentionally required by the production gate so callers cannot
    relabel pair-local tracks as global continuity evidence.
    """

    config.validate()
    lengths: list[int] = []
    duplicate_image_tracks = 0
    try:
        points = list(getattr(reconstruction, "points3D", {}).values())
    except Exception:
        points = []
    for point in points:
        elements = list(getattr(getattr(point, "track", None), "elements", ()) or ())
        image_ids = [int(getattr(element, "image_id", -1)) for element in elements]
        if len(image_ids) != len(set(image_ids)):
            duplicate_image_tracks += 1
        if len(image_ids) >= 2:
            lengths.append(len(image_ids))
    values = np.asarray(lengths, dtype=np.float64)
    provenance_payload = dict(provenance or {})
    provenance_bound = bool(
        provenance_payload
        and str(provenance_payload.get("source", "")).strip()
        and str(provenance_payload.get("independent_graph_sha256", ""))
        and re.fullmatch(r"[0-9a-fA-F]{64}", str(provenance_payload.get("independent_graph_sha256", "")))
        and bool(provenance_payload.get("independent_graph", False))
    )
    constructed_pairwise = bool(provenance_payload.get("constructed_pairwise_tracks", False))
    if len(values):
        quantiles = {
            "p10": float(np.quantile(values, 0.10)),
            "p50": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
            "p99": float(np.quantile(values, 0.99)),
        }
        mean = float(np.mean(values))
        maximum = int(np.max(values))
        fraction_ge3 = float(np.mean(values >= 3.0))
        fraction_ge4 = float(np.mean(values >= 4.0))
        fraction_eq2 = float(np.mean(values == 2.0))
    else:
        quantiles = {key: 0.0 for key in ("p10", "p50", "p90", "p99")}
        mean = 0.0
        maximum = 0
        fraction_ge3 = fraction_ge4 = fraction_eq2 = 0.0
    checks = {
        "track_evidence_available": bool(len(values)),
        "no_duplicate_image_observations": duplicate_image_tracks == 0,
        "independent_track_provenance_bound": provenance_bound,
        "pair_local_track_partition_rejected": not constructed_pairwise,
        "multi_view_track_fraction": fraction_ge3 >= config.minimum_multi_view_track_fraction,
        "track_length_p50": quantiles["p50"] >= config.minimum_track_length_p50,
        "track_length_p90": quantiles["p90"] >= config.minimum_track_length_p90,
    }
    return {
        "schema_version": 1,
        "method": "actual COLMAP reconstruction track elements; independent graph provenance required",
        "point_count_with_tracks": int(len(values)),
        "mean_track_length": mean,
        "maximum_track_length": maximum,
        "track_length_quantiles": quantiles,
        "fraction_track_length_ge3": fraction_ge3,
        "fraction_track_length_ge4": fraction_ge4,
        "fraction_track_length_eq2": fraction_eq2,
        "duplicate_image_track_count": int(duplicate_image_tracks),
        "provenance": provenance_payload,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "thresholds": {
            "minimum_multi_view_track_fraction": config.minimum_multi_view_track_fraction,
            "minimum_track_length_p50": config.minimum_track_length_p50,
            "minimum_track_length_p90": config.minimum_track_length_p90,
        },
    }


def _resolve_sparse_mask_path(mask_root: Path, image_name: str) -> Path | None:
    candidates = (
        mask_root / image_name,
        mask_root / f"{image_name}.png",
        mask_root / f"{Path(image_name).stem}.png",
        mask_root / f"{Path(image_name).stem}.jpg",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _summary_quantiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "min": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "min": float(np.min(array)),
    }


def sparse_mask_projection_evidence(
    reconstruction: Any,
    mask_root: str | Path | None,
    ring_metadata: Mapping[str, Mapping[str, Any]],
    *,
    config: SparseIntegrityConfig = SparseIntegrityConfig(),
    dilation_px: int = 5,
) -> dict[str, Any]:
    """Compare the complete sparse cloud projection with immutable vessel masks.

    Every finite sparse point is projected into every registered view.  Point
    precision is the in-mask fraction of those projections.  Recall and IoU
    compare a fixed, documented 11x11-pixel dilated projection footprint with
    the binary vessel mask.  This deliberately measures global cross-view
    consistency, not only the matched observations that created a point.
    """

    config.validate()
    if mask_root is None:
        return {
            "schema_version": 1,
            "method": "all finite sparse points projected into every registered view against exact immutable masks",
            "status": "missing_mask_root",
            "passed": False,
            "checks": {"mask_root_available": False},
            "views": [],
            "rings": {},
        }
    import cv2

    root = Path(mask_root).resolve()
    if not root.is_dir():
        return {
            "schema_version": 1,
            "method": "all finite sparse points projected into every registered view against exact immutable masks",
            "status": "missing_mask_root",
            "passed": False,
            "checks": {"mask_root_available": False},
            "mask_root": str(root),
            "views": [],
            "rings": {},
        }
    points = []
    for point in list(getattr(reconstruction, "points3D", {}).values()):
        xyz = np.asarray(getattr(point, "xyz", ()), dtype=np.float64).reshape(-1)
        if xyz.shape == (3,) and np.isfinite(xyz).all():
            points.append(xyz)
    xyz_world = np.asarray(points, dtype=np.float64).reshape((-1, 3)) if points else np.empty((0, 3), dtype=np.float64)
    registered_names: list[str] = []
    try:
        registered_names = sorted(
            str(reconstruction.image(int(image_id)).name)
            for image_id in reconstruction.reg_image_ids()
        )
    except Exception:
        registered_names = []
    kernel_size = max(1, int(dilation_px) * 2 + 1)
    view_rows: list[dict[str, Any]] = []
    missing_masks: list[str] = []
    shape_mismatches: list[str] = []
    for name in registered_names:
        image = reconstruction.find_image_with_name(name)
        camera = reconstruction.camera(int(image.camera_id))
        mask_path = _resolve_sparse_mask_path(root, name)
        if mask_path is None:
            missing_masks.append(name)
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        expected_shape = (int(camera.height), int(camera.width))
        if mask is None or mask.shape != expected_shape:
            shape_mismatches.append(name)
            continue
        mask_binary = mask > 0
        pose = image.cam_from_world()
        rotation = np.asarray(pose.rotation.matrix(), dtype=np.float64)
        translation = np.asarray(pose.translation, dtype=np.float64).reshape(3)
        camera_points = xyz_world @ rotation.T + translation if len(xyz_world) else np.empty((0, 3), dtype=np.float64)
        finite = np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > 0.0)
        camera_points = camera_points[finite]
        projected = np.asarray(camera.img_from_cam(camera_points), dtype=np.float64) if len(camera_points) else np.empty((0, 2), dtype=np.float64)
        valid = np.isfinite(projected).all(axis=1)
        projected = projected[valid]
        valid = (
            (projected[:, 0] >= 0.0)
            & (projected[:, 0] < float(camera.width))
            & (projected[:, 1] >= 0.0)
            & (projected[:, 1] < float(camera.height))
        ) if len(projected) else np.empty(0, dtype=bool)
        projected = projected[valid]
        pixels = np.rint(projected).astype(np.int32) if len(projected) else np.empty((0, 2), dtype=np.int32)
        if len(pixels):
            valid_pixels = (
                (pixels[:, 0] >= 0)
                & (pixels[:, 0] < int(camera.width))
                & (pixels[:, 1] >= 0)
                & (pixels[:, 1] < int(camera.height))
            )
            pixels = pixels[valid_pixels]
        in_mask = mask_binary[pixels[:, 1], pixels[:, 0]] if len(pixels) else np.empty(0, dtype=bool)
        precision = float(np.mean(in_mask)) if len(in_mask) else 0.0
        footprint = np.zeros(mask_binary.shape, dtype=np.uint8)
        if len(pixels):
            footprint[pixels[:, 1], pixels[:, 0]] = 1
            footprint = cv2.dilate(footprint, np.ones((kernel_size, kernel_size), dtype=np.uint8), iterations=1)
        intersection = int(np.logical_and(footprint > 0, mask_binary).sum())
        union = int(np.logical_or(footprint > 0, mask_binary).sum())
        mask_area = int(mask_binary.sum())
        recall = float(intersection / max(mask_area, 1))
        iou = float(intersection / max(union, 1))
        view_rows.append(
            {
                "image_name": name,
                "ring": str(ring_metadata.get(name, {}).get("ring", "unknown")),
                "projected_point_count": int(len(pixels)),
                "in_mask_point_count": int(np.sum(in_mask)),
                "mask_area_pixels": mask_area,
                "precision": precision,
                "recall": recall,
                "iou": iou,
            }
        )
    ring_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in view_rows:
        ring_rows[str(row["ring"])].append(row)
    ring_summary = {
        ring: {
            "view_count": len(rows),
            "precision": _summary_quantiles([float(row["precision"]) for row in rows]),
            "recall": _summary_quantiles([float(row["recall"]) for row in rows]),
            "iou": _summary_quantiles([float(row["iou"]) for row in rows]),
        }
        for ring, rows in sorted(ring_rows.items())
    }
    expected_rings = sorted(
        {str(value.get("ring")) for value in ring_metadata.values() if value.get("selected", True)}
    )
    view_precision = _summary_quantiles([float(row["precision"]) for row in view_rows])
    view_recall = _summary_quantiles([float(row["recall"]) for row in view_rows])
    view_iou = _summary_quantiles([float(row["iou"]) for row in view_rows])
    checks = {
        "mask_root_available": True,
        "all_registered_views_measured": len(view_rows) == len(registered_names) and not missing_masks and not shape_mismatches,
        "view_precision_p10": view_precision["p10"] >= config.minimum_view_mask_precision_p10,
        "view_recall_p10": view_recall["p10"] >= config.minimum_view_mask_recall_p10,
        "view_iou_p10": view_iou["p10"] >= config.minimum_view_mask_iou_p10,
        "all_selected_rings_measured": bool(expected_rings) and all(
            ring in ring_summary and ring_summary[ring]["view_count"] > 0 for ring in expected_rings
        ),
        "ring_precision_mean": bool(expected_rings) and all(
            ring_summary[ring]["precision"]["mean"] >= config.minimum_ring_mask_precision_mean
            for ring in expected_rings if ring in ring_summary
        ),
        "ring_recall_mean": bool(expected_rings) and all(
            ring_summary[ring]["recall"]["mean"] >= config.minimum_ring_mask_recall_mean
            for ring in expected_rings if ring in ring_summary
        ),
        "ring_iou_mean": bool(expected_rings) and all(
            ring_summary[ring]["iou"]["mean"] >= config.minimum_ring_mask_iou_mean
            for ring in expected_rings if ring in ring_summary
        ),
    }
    return {
        "schema_version": 1,
        "method": "all finite sparse points projected into every registered view against exact immutable masks",
        "projection_footprint": {
            "dilation_px": int(dilation_px),
            "kernel_size": kernel_size,
            "recall_definition": "mask pixels intersecting the fixed dilated projected-point footprint divided by mask foreground pixels",
            "iou_definition": "intersection over union of the same footprint and mask foreground",
        },
        "mask_root": str(root),
        "registered_view_count": len(registered_names),
        "measured_view_count": len(view_rows),
        "missing_masks": missing_masks,
        "shape_mismatches": shape_mismatches,
        "view_summary": {"precision": view_precision, "recall": view_recall, "iou": view_iou},
        "rings": ring_summary,
        "views": view_rows,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "thresholds": {
            "minimum_view_mask_precision_p10": config.minimum_view_mask_precision_p10,
            "minimum_view_mask_recall_p10": config.minimum_view_mask_recall_p10,
            "minimum_view_mask_iou_p10": config.minimum_view_mask_iou_p10,
            "minimum_ring_mask_precision_mean": config.minimum_ring_mask_precision_mean,
            "minimum_ring_mask_recall_mean": config.minimum_ring_mask_recall_mean,
            "minimum_ring_mask_iou_mean": config.minimum_ring_mask_iou_mean,
        },
    }


def sparse_camera_center_translation_evidence(
    reconstruction: Any,
    translation_records: Sequence[Mapping[str, Any]] | None,
    *,
    config: SparseIntegrityConfig = SparseIntegrityConfig(),
) -> dict[str, Any]:
    """Validate one camera-center field against calibrated image translations.

    Calibrated two-view translation is a direction-only quantity.  The gate
    therefore checks the common candidate camera centers for finite, positive
    projections on those directions and bounded orthogonal residuals.  It does
    not invent metric scale or use capture phase as a substitute for evidence.
    """

    config.validate()
    records = list(translation_records or [])
    centers: dict[str, np.ndarray] = {}
    rotations: dict[str, np.ndarray] = {}
    try:
        for image_id in reconstruction.reg_image_ids():
            image = reconstruction.image(int(image_id))
            pose = image.cam_from_world()
            center = np.asarray(image.projection_center(), dtype=np.float64).reshape(3)
            rotation = np.asarray(pose.rotation.matrix(), dtype=np.float64)
            if np.isfinite(center).all() and rotation.shape == (3, 3) and np.isfinite(rotation).all():
                centers[str(image.name)] = center
                rotations[str(image.name)] = rotation
    except Exception:
        centers = {}
        rotations = {}
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("well_conditioned_calibrated") is False:
            continue
        first, second = str(record.get("first", "")), str(record.get("second", ""))
        vector = record.get("calibrated_translation_first_to_second")
        if vector is None:
            vector = record.get("calibrated_reestimate_translation_first_to_second")
        if first not in centers or second not in centers or vector is None:
            continue
        direction_camera = np.asarray(vector, dtype=np.float64).reshape(-1)
        if direction_camera.shape != (3,) or not np.isfinite(direction_camera).all():
            continue
        direction_norm = float(np.linalg.norm(direction_camera))
        if direction_norm <= 1e-9:
            continue
        direction_world = -rotations[second].T @ (direction_camera / direction_norm)
        direction_world /= max(float(np.linalg.norm(direction_world)), 1e-12)
        delta = centers[second] - centers[first]
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm <= 1e-9 or not np.isfinite(delta).all():
            continue
        scale = float(np.dot(delta, direction_world))
        residual = float(np.linalg.norm(delta - scale * direction_world) / delta_norm)
        cosine = float(np.clip(scale / delta_norm, -1.0, 1.0))
        rows.append(
            {
                "pair_id": int(record.get("pair_id", 0)),
                "first": first,
                "second": second,
                "same_ring": bool(record.get("same_ring", False)),
                "center_distance": delta_norm,
                "projected_scale": scale,
                "orthogonal_residual_fraction": residual,
                "direction_error_deg": float(math.degrees(math.acos(cosine))),
            }
        )
    direction_errors = [float(row["direction_error_deg"]) for row in rows]
    residuals = [float(row["orthogonal_residual_fraction"]) for row in rows]
    scales = [float(row["projected_scale"]) for row in rows]
    positive_fraction = float(np.mean(np.asarray(scales) > 0.0)) if scales else 0.0
    checks = {
        "translation_evidence_available": bool(rows),
        "finite_camera_centers": bool(centers),
        "direction_error_p90": bool(direction_errors) and float(np.quantile(direction_errors, 0.90)) <= config.maximum_translation_direction_p90_deg,
        "positive_projected_scale_fraction": positive_fraction >= config.minimum_translation_positive_scale_fraction,
        "orthogonal_residual_p90": bool(residuals) and float(np.quantile(residuals, 0.90)) <= config.maximum_translation_orthogonal_residual_p90,
    }
    return {
        "schema_version": 1,
        "method": "calibrated two-view translation directions compared with candidate camera centers",
        "candidate_center_count": len(centers),
        "evaluated_edge_count": len(rows),
        "direction_error_deg": _summary_quantiles(direction_errors),
        "orthogonal_residual_fraction": _summary_quantiles(residuals),
        "projected_scale": _summary_quantiles(scales),
        "positive_projected_scale_fraction": positive_fraction,
        "edges": rows,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "thresholds": {
            "maximum_translation_direction_p90_deg": config.maximum_translation_direction_p90_deg,
            "minimum_translation_positive_scale_fraction": config.minimum_translation_positive_scale_fraction,
            "maximum_translation_orthogonal_residual_p90": config.maximum_translation_orthogonal_residual_p90,
        },
    }


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
    track_provenance: Mapping[str, Any] | None = None,
    mask_projection_evidence: Mapping[str, Any] | None = None,
    camera_center_evidence: Mapping[str, Any] | None = None,
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
    track_distribution = sparse_track_distribution_evidence(
        reconstruction,
        config=config,
        provenance=track_provenance,
    )
    mask_projection = dict(mask_projection_evidence or {})
    camera_center_translation = dict(camera_center_evidence or {})
    mask_checks = dict(mask_projection.get("checks") or {})
    camera_checks = dict(camera_center_translation.get("checks") or {})
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
        "independent_multiview_track_distribution": bool(track_distribution.get("passed")),
        "mask_projection_per_view": bool(
            mask_checks.get("all_registered_views_measured")
            and mask_checks.get("view_precision_p10")
            and mask_checks.get("view_recall_p10")
            and mask_checks.get("view_iou_p10")
        ),
        "mask_projection_per_ring": bool(
            mask_checks.get("all_selected_rings_measured")
            and mask_checks.get("ring_precision_mean")
            and mask_checks.get("ring_recall_mean")
            and mask_checks.get("ring_iou_mean")
        ),
        "camera_center_translation_scale_consistency": bool(camera_center_translation.get("passed"))
        and bool(
            camera_checks.get("direction_error_p90")
            and camera_checks.get("positive_projected_scale_fraction")
            and camera_checks.get("orthogonal_residual_p90")
        ),
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
        "track_distribution": track_distribution,
        "mask_projection": mask_projection,
        "camera_center_translation": camera_center_translation,
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
    """Copy a frozen source DB and retain only the selected verified pairs.

    The source may be the manifest-bound canonical snapshot or a disposable
    byte copy created from it.  In either case it is copied before SQLite
    opens the destination, and the canonical path is never opened.  Both the
    verified ``two_view_geometries`` rows and raw ``matches`` rows are pruned
    because the exact graph is a provenance boundary for the mapper.
    """

    manifest = load_canonical_sqlite_snapshot_manifest(snapshot_manifest)
    source_path, destination_path = Path(source).resolve(), Path(destination).resolve()
    canonical_path = Path(manifest["canonical_path"]).resolve()
    if destination_path == canonical_path:
        raise ValueError("pruning destination may not be the canonical SQLite snapshot")
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
    connection = sqlite3.connect(str(destination_path))
    try:
        connection.execute("CREATE TEMP TABLE selected_repair_pairs(pair_id INTEGER PRIMARY KEY)")
        connection.executemany("INSERT INTO selected_repair_pairs(pair_id) VALUES (?)", ((int(value),) for value in sorted(set(pair_ids))))
        connection.execute("DELETE FROM matches WHERE pair_id NOT IN (SELECT pair_id FROM selected_repair_pairs)")
        connection.execute("DELETE FROM two_view_geometries WHERE pair_id NOT IN (SELECT pair_id FROM selected_repair_pairs)")
        connection.commit()
    finally:
        connection.close()
    preflight = validate_exact_mapper_graph(
        destination_path,
        pair_ids,
        canonical_path=canonical_path,
        require_raw_matches=True,
    )
    if not preflight["passed"]:
        raise RuntimeError(f"exact mapper graph pruning failed closed: {preflight}")
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
    "sparse_camera_center_translation_evidence",
    "sparse_integrity_gate",
    "sparse_mask_projection_evidence",
    "sparse_track_distribution_evidence",
    "sqlite_logical_digest",
    "stable_directory_sha256",
    "validate_exact_mapper_graph",
    "validate_one_reference_write",
    "write_sparse_graph_report",
]
