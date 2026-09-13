"""V4 LightGlue pairing and COLMAP database construction.

The source capture has unequal, phase-offset turntable rings.  This module
keeps the learned match stage explicit: feature caches are loaded only when
their recorded identities are intact, cross-ring offsets are estimated from
mask-restricted LightGlue support, and only the resulting deterministic pair
schedule is imported into COLMAP for geometric verification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import sqlite3
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from v4_config import (
    RECONSTRUCTION_V4_ROOT,
    assert_output_path,
    fingerprint,
    sha256_file,
    write_json,
)
from v4_features import V4FeatureConfig, V4Frontend, V4ImageFeatures, load_frontend
from v4_matching import (
    PhaseFrame,
    build_pair_schedule,
    circular_distance,
    estimate_circular_phase_offset,
    normalize_phase,
    schedule_hash,
)
from v4_sparse import V4SparseConfig


DEFAULT_RING_ORDER = ("g7_r1", "geo_g8", "geo_g9", "geo_g10", "geo_g11", "geo_g12")


@dataclass(frozen=True)
class V4PairingConfig:
    """One bounded, acquisition-aware LightGlue schedule configuration."""

    ring_order: tuple[str, ...] = DEFAULT_RING_ORDER
    candidate_offsets: tuple[float, ...] = tuple(index / 20.0 for index in range(20))
    anchor_count: int = 8
    same_ring_neighbors: int = 1
    wider_phase_neighbors: int = 2
    cross_ring_neighbors: int = 1
    minimum_offset_margin: float = 2.0

    def validate(self) -> "V4PairingConfig":
        if len(self.ring_order) < 2 or len(set(self.ring_order)) != len(self.ring_order):
            raise ValueError("V4 pairing ring_order must contain unique adjacent rings")
        if self.anchor_count < 1:
            raise ValueError("anchor_count must be positive")
        if self.same_ring_neighbors < 1 or self.wider_phase_neighbors < 0 or self.cross_ring_neighbors < 0:
            raise ValueError("pair neighborhoods require same_ring >= 1 and others >= 0")
        if self.minimum_offset_margin < 0 or not math.isfinite(float(self.minimum_offset_margin)):
            raise ValueError("minimum_offset_margin must be finite and nonnegative")
        offsets = tuple(normalize_phase(float(value)) for value in self.candidate_offsets)
        if not offsets or len(set(offsets)) != len(offsets):
            raise ValueError("candidate_offsets must contain unique finite phases")
        return self


def _normal_pair(first: str, second: str) -> tuple[str, str]:
    if not first or not second or first == second:
        raise ValueError("image pair requires two distinct filenames")
    return (first, second) if first < second else (second, first)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def selected_phase_frames(
    records: Sequence[Mapping[str, Any]],
    *,
    ring_order: Sequence[str] = DEFAULT_RING_ORDER,
) -> dict[str, tuple[PhaseFrame, ...]]:
    """Build selected geometry frames from the authoritative isolation index."""

    result: dict[str, list[PhaseFrame]] = {str(ring): [] for ring in ring_order}
    for record in records:
        if str(record.get("source_role", "geometry")) != "geometry":
            continue
        if not bool(record.get("selected_for_geometry", True)):
            continue
        ring = str(record.get("logical_ring_id") or "")
        if not ring:
            raise ValueError(f"selected geometry record has no logical ring: {record}")
        if ring not in result:
            result[ring] = []
        result[ring].append(
            PhaseFrame(
                filename=str(record["relative_path"]),
                ring_id=ring,
                phase_01=float(record.get("phase_01") or 0.0),
                selected=True,
            )
        )
    ordered: dict[str, tuple[PhaseFrame, ...]] = {}
    for ring in tuple(str(value) for value in ring_order) + tuple(sorted(set(result) - set(ring_order))):
        frames = tuple(sorted(result.get(ring, []), key=lambda item: (item.phase_01, item.filename)))
        if not frames:
            raise ValueError(f"selected geometry ring is empty: {ring}")
        ordered[ring] = frames
    return ordered


def load_v4_feature_cache(
    marker_path: str | Path = RECONSTRUCTION_V4_ROOT / "work" / "features_manifest.json",
) -> tuple[dict[str, V4ImageFeatures], dict[str, Any]]:
    """Load an identity-checked cache without importing the learned runtime."""

    marker = _load_json(marker_path)
    if marker.get("status") != "complete":
        raise ValueError("V4 feature cache marker is not complete")
    items = marker.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("V4 feature cache marker has no items")
    loaded: dict[str, V4ImageFeatures] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("V4 feature cache item is not an object")
        name = str(item.get("relative_path") or "")
        path = Path(str(item.get("path") or ""))
        if not name or not path.is_file():
            raise ValueError(f"V4 feature cache item is missing: {name}")
        if sha256_file(path) != str(item.get("sha256") or ""):
            raise ValueError(f"V4 feature cache checksum mismatch: {name}")
        with np.load(path, allow_pickle=False) as values:
            for key in ("keypoints", "descriptors", "scores", "image_size", "cache_identity"):
                if key not in values:
                    raise ValueError(f"V4 feature cache entry lacks {key}: {name}")
            cache_identity = str(values["cache_identity"].item())
            if cache_identity != str(item.get("cache_identity") or ""):
                raise ValueError(f"V4 feature cache identity mismatch: {name}")
            features = V4ImageFeatures(
                keypoints=np.asarray(values["keypoints"], dtype=np.float32),
                descriptors=np.asarray(values["descriptors"], dtype=np.float32),
                scores=np.asarray(values["scores"], dtype=np.float32),
                image_size=np.asarray(values["image_size"], dtype=np.float32),
            ).validate()
        if features.count != int(item.get("mask_supported_keypoints", -1)):
            raise ValueError(f"V4 feature cache count mismatch: {name}")
        loaded[name] = features
    if len(loaded) != len(items):
        raise ValueError("V4 feature cache contains duplicate image names")
    return loaded, marker


def _feature_payload(features: V4ImageFeatures, device: str) -> dict[str, Any]:
    import torch

    features.validate()
    return {
        "keypoints": torch.from_numpy(np.asarray(features.keypoints, dtype=np.float32)).to(device)[None],
        "descriptors": torch.from_numpy(np.asarray(features.descriptors, dtype=np.float32)).to(device)[None],
        "keypoint_scores": torch.from_numpy(np.asarray(features.scores, dtype=np.float32)).to(device)[None],
        "image_size": torch.from_numpy(np.asarray(features.image_size, dtype=np.float32)).to(device)[None],
    }


def match_feature_pair(
    first: V4ImageFeatures,
    second: V4ImageFeatures,
    frontend: V4Frontend,
    *,
    payload_cache: Mapping[int, Mapping[str, Any]] | None = None,
) -> np.ndarray:
    """Run the installed LightGlue matcher on two cached ALIKED tensors."""

    import torch
    from lightglue.utils import rbd

    first_payload = payload_cache.get(id(first)) if payload_cache is not None else None
    second_payload = payload_cache.get(id(second)) if payload_cache is not None else None
    if first_payload is None:
        first_payload = _feature_payload(first, frontend.device)
    if second_payload is None:
        second_payload = _feature_payload(second, frontend.device)
    with torch.inference_mode():
        prediction = rbd(
            frontend.matcher(
                {
                    "image0": first_payload,
                    "image1": second_payload,
                }
            )
        )
    try:
        values = np.asarray(prediction["matches"].detach().cpu().numpy())
    except (KeyError, AttributeError, TypeError) as error:
        raise RuntimeError("LightGlue returned an unsupported match structure") from error
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("LightGlue matches must have shape N x 2")
    if values.size == 0:
        return np.empty((0, 2), dtype=np.uint32)
    if not np.issubdtype(values.dtype, np.integer):
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError("LightGlue match indices must be integer-valued")
    indices = values.astype(np.int64, copy=False)
    if np.any(indices < 0) or np.any(indices[:, 0] >= first.count) or np.any(indices[:, 1] >= second.count):
        raise ValueError("LightGlue match index is outside the cached feature rows")
    return indices.astype(np.uint32, copy=False)


def build_feature_payload_cache(
    features_by_name: Mapping[str, V4ImageFeatures],
    frontend: V4Frontend,
) -> dict[int, Mapping[str, Any]]:
    """Upload each cached image once when many pairs share the same views."""

    return {id(features): _feature_payload(features, frontend.device) for features in features_by_name.values()}


def _evenly_spaced_indices(count: int, limit: int) -> tuple[int, ...]:
    if count < 1:
        return ()
    if limit >= count:
        return tuple(range(count))
    raw = np.linspace(0, count - 1, num=limit)
    return tuple(sorted(set(int(round(value)) for value in raw)))


def estimate_cross_ring_offsets(
    frames_by_ring: Mapping[str, Sequence[PhaseFrame]],
    features_by_name: Mapping[str, V4ImageFeatures],
    frontend: V4Frontend,
    *,
    config: V4PairingConfig = V4PairingConfig(),
    progress: Callable[[int, int, str, float, int], None] | None = None,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], np.ndarray]]:
    """Estimate adjacent-ring phase offsets using bounded learned support."""

    config.validate()
    offsets: dict[tuple[str, str], dict[str, Any]] = {}
    match_cache: dict[tuple[str, str], np.ndarray] = {}
    payload_cache = build_feature_payload_cache(features_by_name, frontend)
    adjacent = list(zip(config.ring_order[:-1], config.ring_order[1:]))
    total_jobs = len(adjacent) * len(config.candidate_offsets) * max(1, config.anchor_count)
    completed_jobs = 0
    for left_id, right_id in adjacent:
        left_frames = tuple(frames_by_ring[left_id])
        right_frames = tuple(frames_by_ring[right_id])
        anchor_indices = _evenly_spaced_indices(len(left_frames), config.anchor_count)
        score_entries: list[tuple[float, float]] = []
        candidate_details: list[dict[str, Any]] = []
        for candidate_offset in config.candidate_offsets:
            normalized_offset = normalize_phase(candidate_offset)
            counts: list[int] = []
            pairs: list[tuple[str, str]] = []
            for anchor_index in anchor_indices:
                anchor = left_frames[anchor_index]
                target_phase = normalize_phase(anchor.phase_01 + normalized_offset)
                right_index = min(
                    range(len(right_frames)),
                    key=lambda index: (circular_distance(target_phase, right_frames[index].phase_01), right_frames[index].filename),
                )
                pair = _normal_pair(anchor.filename, right_frames[right_index].filename)
                if pair not in match_cache:
                    match_cache[pair] = match_feature_pair(
                        features_by_name[pair[0]],
                        features_by_name[pair[1]],
                        frontend,
                        payload_cache=payload_cache,
                    )
                count = int(len(match_cache[pair]))
                counts.append(count)
                pairs.append(pair)
                completed_jobs += 1
                if progress is not None:
                    progress(completed_jobs, total_jobs, f"{left_id}->{right_id}", normalized_offset, count)
            score = float(statistics.median(counts)) if counts else 0.0
            score_entries.append((normalized_offset, score))
            candidate_details.append(
                {
                    "offset_01": normalized_offset,
                    "anchor_match_counts": counts,
                    "median_match_count": score,
                    "total_match_count": int(sum(counts)),
                    "positive_anchor_count": int(sum(value > 0 for value in counts)),
                    "anchor_pairs": [list(pair) for pair in pairs],
                }
            )
        chosen = estimate_circular_phase_offset(score_entries, minimum_margin=config.minimum_offset_margin)
        chosen["left_ring"] = left_id
        chosen["right_ring"] = right_id
        chosen["anchor_count"] = len(anchor_indices)
        chosen["candidate_count"] = len(candidate_details)
        chosen["candidates"] = candidate_details
        offsets[(left_id, right_id)] = chosen
    return offsets, match_cache


def build_v4_pair_schedule(
    frames_by_ring: Mapping[str, Sequence[PhaseFrame]],
    offsets: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    config: V4PairingConfig = V4PairingConfig(),
) -> tuple[tuple[str, str], ...]:
    config.validate()
    cross_offsets = {pair: float(value.get("offset_01", 0.0)) for pair, value in offsets.items()}
    return build_pair_schedule(
        frames_by_ring,
        same_ring_neighbors=config.same_ring_neighbors,
        wider_phase_neighbors=config.wider_phase_neighbors,
        cross_ring_neighbors=config.cross_ring_neighbors,
        cross_ring_offsets=cross_offsets,
        ring_order=config.ring_order,
    )


def pairing_report(
    *,
    records: Sequence[Mapping[str, Any]],
    feature_marker: Mapping[str, Any],
    frames_by_ring: Mapping[str, Sequence[PhaseFrame]],
    offsets: Mapping[tuple[str, str], Mapping[str, Any]],
    pairs: Sequence[tuple[str, str]],
    config: V4PairingConfig = V4PairingConfig(),
) -> dict[str, Any]:
    config.validate()
    # The isolation index is already restricted to selected geometry images and
    # therefore omits the manifest-only selection flag.  Treat a missing flag
    # as selected; an explicit false remains excluded.
    selected_names = {
        str(record.get("relative_path"))
        for record in records
        if bool(record.get("selected_for_geometry", True))
    }
    pair_names = {name for pair in pairs for name in pair}
    return {
        "schema_version": 1,
        "status": "complete",
        "method": "ALIKED-N16Rot + LightGlue mask-restricted phase support",
        "feature_manifest_sha256": sha256_file(RECONSTRUCTION_V4_ROOT / "work" / "features_manifest.json"),
        "feature_model_identity": feature_marker.get("model_identity", {}),
        "config": asdict(config),
        "ring_order": list(config.ring_order),
        "ring_counts": {ring: len(frames_by_ring[ring]) for ring in config.ring_order},
        "selected_geometry_count": len(selected_names),
        "schedule_pair_count": len(pairs),
        "schedule_hash": schedule_hash(pairs),
        "schedule_image_coverage": len(pair_names),
        "all_selected_images_represented": selected_names.issubset(pair_names),
        "cross_ring_offsets": {f"{left}->{right}": dict(value) for (left, right), value in offsets.items()},
        "ambiguous_offset_count": int(sum(bool(value.get("ambiguous")) for value in offsets.values())),
        "estimated_at_epoch": time.time(),
    }


def _camera_params(config: V4SparseConfig) -> str:
    diagonal = math.hypot(36.0, 24.0)
    focal = config.focal_35mm / diagonal * math.hypot(config.image_width, config.image_height)
    return ",".join(f"{value:.12g}" for value in (focal, config.image_width / 2.0, config.image_height / 2.0, 0.0))


def _write_pairs_file(path: Path, pairs: Sequence[tuple[str, str]]) -> None:
    target = assert_output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for first, second in pairs:
        pair = _normal_pair(str(first), str(second))
        if pair in seen:
            raise ValueError("duplicate pair in COLMAP pair file")
        if any(Path(name).name != name or any(char.isspace() for char in name) for name in pair):
            raise ValueError("COLMAP pair filenames must be bare names without whitespace")
        seen.add(pair)
        lines.append(f"{pair[0]} {pair[1]}")
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def database_metrics(database_path: str | Path) -> dict[str, Any]:
    """Read compact, inspectable counts from a verified COLMAP database."""

    path = Path(database_path)
    if not path.is_file():
        raise ValueError(f"COLMAP database is missing: {path}")
    with sqlite3.connect(str(path)) as connection:
        def scalar(query: str) -> int:
            value = connection.execute(query).fetchone()[0]
            return int(value or 0)

        metrics = {
            "image_count": scalar("SELECT COUNT(*) FROM images"),
            "camera_count": scalar("SELECT COUNT(*) FROM cameras"),
            "keypoint_rows": scalar("SELECT COALESCE(SUM(rows), 0) FROM keypoints"),
            "match_pair_rows": scalar("SELECT COUNT(*) FROM matches"),
            "raw_match_rows": scalar("SELECT COALESCE(SUM(rows), 0) FROM matches"),
            "verified_pair_rows": scalar("SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0"),
            "verified_inlier_rows": scalar("SELECT COALESCE(SUM(rows), 0) FROM two_view_geometries"),
        }
    try:
        import pycolmap

        with pycolmap.Database.open(path) as database:
            metrics["pycolmap_verified_image_pairs"] = int(database.num_verified_image_pairs())
            metrics["pycolmap_matched_image_pairs"] = int(database.num_matched_image_pairs())
    except Exception as error:
        metrics["pycolmap_metrics_error"] = f"{type(error).__name__}: {error}"
    return metrics


def build_colmap_database(
    *,
    image_dir: str | Path,
    database_path: str | Path,
    pairs_path: str | Path,
    features_by_name: Mapping[str, V4ImageFeatures],
    matches_by_pair: Mapping[tuple[str, str], np.ndarray],
    sparse_config: V4SparseConfig = V4SparseConfig(),
    min_raw_matches_to_import: int = 8,
    two_view_options: Any | None = None,
) -> dict[str, Any]:
    """Import cached features, write learned matches, and verify with COLMAP."""

    sparse_config.validate()
    image_root = Path(image_dir)
    database = assert_output_path(database_path)
    pair_file = assert_output_path(pairs_path)
    if not image_root.is_dir():
        raise ValueError(f"V4 database image directory is missing: {image_root}")
    if database.exists():
        raise FileExistsError(f"V4 database destination already exists: {database}")
    if min_raw_matches_to_import < 1:
        raise ValueError("min_raw_matches_to_import must be positive")
    names = tuple(sorted(str(name) for name in features_by_name))
    if len(names) < 2:
        raise ValueError("V4 database requires at least two feature images")
    for name in names:
        features_by_name[name].validate()
        if not (image_root / name).is_file():
            raise ValueError(f"V4 database source image is missing: {name}")

    import pycolmap

    database.parent.mkdir(parents=True, exist_ok=True)
    with pycolmap.Database.open(database):
        pass
    reader = pycolmap.ImageReaderOptions(
        camera_model=sparse_config.camera_model,
        camera_params=_camera_params(sparse_config),
    )
    pycolmap.import_images(
        database_path=database,
        image_path=image_root,
        camera_mode=pycolmap.CameraMode.SINGLE,
        image_names=list(names),
        options=reader,
    )
    with sqlite3.connect(str(database)) as connection:
        image_ids = {str(name): int(image_id) for image_id, name in connection.execute("SELECT image_id, name FROM images")}
        camera_count = int(connection.execute("SELECT COUNT(*) FROM cameras").fetchone()[0])
    if set(image_ids) != set(names):
        raise RuntimeError("COLMAP database image names do not match V4 feature cache")
    if camera_count != 1:
        raise RuntimeError(f"V4 shared-intrinsic import produced {camera_count} cameras")

    imported_pairs: list[tuple[str, str]] = []
    raw_match_counts: list[int] = []
    with pycolmap.Database.open(database) as database_handle:
        for name in names:
            features = features_by_name[name]
            database_handle.write_keypoints(image_ids[name], (features.keypoints + 0.5).astype(np.float32, copy=False))
        for original_pair, values in matches_by_pair.items():
            pair = _normal_pair(*original_pair)
            if pair[0] not in image_ids or pair[1] not in image_ids:
                raise ValueError(f"V4 match pair references unknown image: {pair}")
            matches = np.asarray(values)
            if matches.ndim != 2 or matches.shape[1] != 2:
                raise ValueError(f"V4 matches have invalid shape for {pair}: {matches.shape}")
            if len(matches) < min_raw_matches_to_import:
                continue
            matches = matches.astype(np.uint32, copy=False)
            if np.any(matches[:, 0] >= features_by_name[pair[0]].count) or np.any(matches[:, 1] >= features_by_name[pair[1]].count):
                raise ValueError(f"V4 match row exceeds feature count for {pair}")
            database_handle.write_matches(image_ids[pair[0]], image_ids[pair[1]], matches)
            imported_pairs.append(pair)
            raw_match_counts.append(int(len(matches)))
    _write_pairs_file(pair_file, imported_pairs)
    options = two_view_options or pycolmap.TwoViewGeometryOptions()
    pycolmap.verify_matches(database, pair_file, options)
    metrics = database_metrics(database)
    report = {
        "schema_version": 1,
        "status": "complete",
        "database_path": str(database.resolve()),
        "database_sha256": sha256_file(database),
        "pairs_path": str(pair_file.resolve()),
        "pairs_sha256": sha256_file(pair_file),
        "pycolmap_version": str(pycolmap.__version__),
        "camera_contract": {
            "camera_model": sparse_config.camera_model,
            "camera_mode": "SINGLE",
            "camera_params": _camera_params(sparse_config),
            "image_width": sparse_config.image_width,
            "image_height": sparse_config.image_height,
        },
        "feature_image_count": len(names),
        "scheduled_pair_count": len(matches_by_pair),
        "imported_pair_count": len(imported_pairs),
        "raw_match_count": {"min": min(raw_match_counts) if raw_match_counts else 0, "median": statistics.median(raw_match_counts) if raw_match_counts else 0, "max": max(raw_match_counts) if raw_match_counts else 0, "total": int(sum(raw_match_counts))},
        "min_raw_matches_to_import": min_raw_matches_to_import,
        "metrics": metrics,
        "two_view_geometry_options": options.todict() if hasattr(options, "todict") else str(options),
    }
    return report


__all__ = [
    "DEFAULT_RING_ORDER",
    "V4PairingConfig",
    "build_colmap_database",
    "build_feature_payload_cache",
    "build_v4_pair_schedule",
    "database_metrics",
    "estimate_cross_ring_offsets",
    "load_v4_feature_cache",
    "match_feature_pair",
    "pairing_report",
    "selected_phase_frames",
]
