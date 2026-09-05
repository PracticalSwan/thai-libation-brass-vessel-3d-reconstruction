"""Step 11 sparse-component bridge diagnosis and reconstruction helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import sqlite3
import time
from typing import Callable, Sequence

import pycolmap

from analysis_common import SelectedImageRecord, verify_selected_images
from sparse_reconstruction import (
    AttemptMetrics,
    DatabaseMetrics,
    ModelMetrics,
    SparseRunConfig,
    extract_sparse_features,
    map_sparse_database,
    summarize_database,
)


@dataclass(frozen=True)
class BridgeSearchConfig:
    boundaries: tuple[tuple[int, int], ...] = ((73, 74), (145, 146), (203, 204))
    window_size: int = 40
    minimum_sequence_gap: int = 41
    minimum_verified_inliers: int = 15
    minimum_inlier_ratio: float = 0.15
    max_pairs_per_boundary: int = 8
    max_endpoint_reuse: int = 2
    targeted_sequential_overlap: int = 20
    exhaustive_block_size: int = 50
    minimum_registered_images: int = 274
    minimum_sparse_points: int = 1000


@dataclass(frozen=True)
class BridgeCandidate:
    boundary_left: int
    boundary_right: int
    left_index: int
    right_index: int
    left_filename: str
    right_filename: str

    @property
    def sequence_gap(self) -> int:
        return self.right_index - self.left_index


@dataclass(frozen=True)
class BridgePairMetrics:
    candidate: BridgeCandidate
    raw_matches: int
    verified_inliers: int
    inlier_ratio: float
    qualified: bool


@dataclass(frozen=True)
class TargetedGateResult:
    allowed: bool
    reason: str


def _candidate_key(pair: BridgeCandidate) -> tuple[int, int, int, str, str]:
    return (
        pair.boundary_left,
        pair.left_index,
        pair.right_index,
        pair.left_filename,
        pair.right_filename,
    )


def generate_candidate_pairs(
    records: Sequence[SelectedImageRecord],
    config: BridgeSearchConfig = BridgeSearchConfig(),
) -> tuple[BridgeCandidate, ...]:
    if config.window_size < 1:
        raise ValueError("window_size must be positive")
    if config.minimum_sequence_gap < 1:
        raise ValueError("minimum_sequence_gap must be positive")
    by_index = {record.index: record for record in records}
    if len(by_index) != len(records):
        raise ValueError("selected-image indices must be unique")
    if not by_index:
        raise ValueError("selected-image records are required")

    pairs: list[BridgeCandidate] = []
    seen_filenames: set[tuple[str, str]] = set()
    last_index = max(by_index)
    for boundary_left, boundary_right in config.boundaries:
        if boundary_right != boundary_left + 1:
            raise ValueError("bridge boundaries must contain adjacent indices")
        left_start = max(1, boundary_left - config.window_size + 1)
        right_end = min(last_index, boundary_right + config.window_size - 1)
        for left_index in range(left_start, boundary_left + 1):
            left = by_index.get(left_index)
            if left is None:
                raise ValueError(f"selected-image index is missing: {left_index}")
            for right_index in range(boundary_right, right_end + 1):
                right = by_index.get(right_index)
                if right is None:
                    raise ValueError(f"selected-image index is missing: {right_index}")
                if right_index - left_index < config.minimum_sequence_gap:
                    continue
                filename_pair = (left.filename, right.filename)
                if filename_pair in seen_filenames:
                    continue
                seen_filenames.add(filename_pair)
                pairs.append(
                    BridgeCandidate(
                        boundary_left=boundary_left,
                        boundary_right=boundary_right,
                        left_index=left_index,
                        right_index=right_index,
                        left_filename=left.filename,
                        right_filename=right.filename,
                    )
                )
    return tuple(sorted(pairs, key=_candidate_key))


def _as_candidate(pair: BridgeCandidate | BridgePairMetrics) -> BridgeCandidate:
    return pair.candidate if isinstance(pair, BridgePairMetrics) else pair


def write_pair_list(
    path: Path, pairs: Sequence[BridgeCandidate | BridgePairMetrics]
) -> None:
    candidates = [_as_candidate(pair) for pair in pairs]
    filename_pairs: set[tuple[str, str]] = set()
    for candidate in candidates:
        names = (candidate.left_filename, candidate.right_filename)
        if any(any(character.isspace() for character in name) for name in names):
            raise ValueError("imported-pair filenames must not contain whitespace")
        if any(Path(name).name != name or name in {"", ".", ".."} for name in names):
            raise ValueError("imported-pair entries must contain filenames only")
        if names in filename_pairs:
            raise ValueError("duplicate imported image pair")
        filename_pairs.add(names)

    lines = [
        f"{candidate.left_filename} {candidate.right_filename}"
        for candidate in sorted(candidates, key=_candidate_key)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def qualified_bridge(
    pair: BridgePairMetrics, config: BridgeSearchConfig = BridgeSearchConfig()
) -> bool:
    return bool(
        pair.verified_inliers >= config.minimum_verified_inliers
        and pair.inlier_ratio >= config.minimum_inlier_ratio
    )


def select_bridge_pairs(
    metrics: Sequence[BridgePairMetrics],
    config: BridgeSearchConfig = BridgeSearchConfig(),
) -> tuple[BridgePairMetrics, ...]:
    selected: list[BridgePairMetrics] = []
    for boundary_left, boundary_right in config.boundaries:
        ranked = sorted(
            (
                pair
                for pair in metrics
                if (
                    pair.candidate.boundary_left,
                    pair.candidate.boundary_right,
                )
                == (boundary_left, boundary_right)
                and qualified_bridge(pair, config)
            ),
            key=lambda pair: (
                -pair.verified_inliers,
                -pair.inlier_ratio,
                pair.candidate.sequence_gap,
                pair.candidate.left_filename,
                pair.candidate.right_filename,
            ),
        )
        endpoint_use: Counter[str] = Counter()
        for pair in ranked:
            left_name = pair.candidate.left_filename
            right_name = pair.candidate.right_filename
            if (
                endpoint_use[left_name] >= config.max_endpoint_reuse
                or endpoint_use[right_name] >= config.max_endpoint_reuse
            ):
                continue
            selected.append(pair)
            endpoint_use[left_name] += 1
            endpoint_use[right_name] += 1
            boundary_count = sum(
                item.candidate.boundary_left == boundary_left for item in selected
            )
            if boundary_count >= config.max_pairs_per_boundary:
                break
    return tuple(selected)


def summarize_bridge_pairs(
    database_path: Path,
    candidates: Sequence[BridgeCandidate],
    config: BridgeSearchConfig = BridgeSearchConfig(),
) -> tuple[BridgePairMetrics, ...]:
    if not database_path.is_file():
        raise ValueError(f"COLMAP database is missing: {database_path}")
    with sqlite3.connect(str(database_path)) as connection:
        image_ids = {
            str(name): int(image_id)
            for image_id, name in connection.execute(
                "SELECT image_id, name FROM images"
            ).fetchall()
        }
        metrics: list[BridgePairMetrics] = []
        for candidate in candidates:
            try:
                left_id = image_ids[candidate.left_filename]
                right_id = image_ids[candidate.right_filename]
            except KeyError as error:
                raise ValueError(
                    f"candidate image is missing from COLMAP database: {error.args[0]}"
                ) from error
            pair_id = int(pycolmap.image_pair_to_pair_id(left_id, right_id))
            raw_row = connection.execute(
                "SELECT rows FROM matches WHERE pair_id = ?", (pair_id,)
            ).fetchone()
            verified_row = connection.execute(
                "SELECT rows FROM two_view_geometries WHERE pair_id = ?", (pair_id,)
            ).fetchone()
            raw_matches = int(raw_row[0]) if raw_row is not None else 0
            verified_inliers = int(verified_row[0]) if verified_row is not None else 0
            inlier_ratio = verified_inliers / raw_matches if raw_matches > 0 else 0.0
            provisional = BridgePairMetrics(
                candidate=candidate,
                raw_matches=raw_matches,
                verified_inliers=verified_inliers,
                inlier_ratio=inlier_ratio,
                qualified=False,
            )
            metrics.append(
                BridgePairMetrics(
                    candidate=candidate,
                    raw_matches=raw_matches,
                    verified_inliers=verified_inliers,
                    inlier_ratio=inlier_ratio,
                    qualified=qualified_bridge(provisional, config),
                )
            )
    return tuple(metrics)


def run_bridge_diagnostics(
    features_database: Path,
    workspace: Path,
    candidates: Sequence[BridgeCandidate],
    config: BridgeSearchConfig = BridgeSearchConfig(),
) -> tuple[BridgePairMetrics, ...]:
    if not features_database.is_file():
        raise ValueError(f"features database is missing: {features_database}")
    if not candidates:
        raise ValueError("at least one diagnostic image pair is required")
    workspace.mkdir(parents=True, exist_ok=True)
    diagnostic_database = workspace / "diagnostic.db"
    pair_list_path = workspace / "diagnostic_pairs.txt"
    if diagnostic_database.exists():
        if not diagnostic_database.is_file():
            raise ValueError(f"diagnostic database path is not a file: {diagnostic_database}")
        diagnostic_database.unlink()
    shutil.copy2(features_database, diagnostic_database)
    write_pair_list(pair_list_path, candidates)

    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.use_gpu = False
    pairing_options = pycolmap.ImportedPairingOptions()
    pairing_options.match_list_path = pair_list_path
    pycolmap.match_image_pairs(
        database_path=diagnostic_database,
        matching_options=matching_options,
        pairing_options=pairing_options,
        device=pycolmap.Device.cpu,
    )
    return summarize_bridge_pairs(diagnostic_database, candidates, config)


def boundary_bridge_summary(
    metrics: Sequence[BridgePairMetrics],
    selected: Sequence[BridgePairMetrics],
    config: BridgeSearchConfig = BridgeSearchConfig(),
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for boundary_left, boundary_right in config.boundaries:
        boundary_metrics = tuple(
            pair
            for pair in metrics
            if (
                pair.candidate.boundary_left,
                pair.candidate.boundary_right,
            )
            == (boundary_left, boundary_right)
        )
        boundary_selected = tuple(
            pair
            for pair in selected
            if (
                pair.candidate.boundary_left,
                pair.candidate.boundary_right,
            )
            == (boundary_left, boundary_right)
        )
        rows.append(
            {
                "boundary": f"{boundary_left}-{boundary_right}",
                "boundary_left": boundary_left,
                "boundary_right": boundary_right,
                "candidate_count": len(boundary_metrics),
                "nonzero_match_count": sum(
                    pair.raw_matches > 0 for pair in boundary_metrics
                ),
                "qualified_count": sum(
                    qualified_bridge(pair, config) for pair in boundary_metrics
                ),
                "selected_count": len(boundary_selected),
                "maximum_verified_inliers": max(
                    (pair.verified_inliers for pair in boundary_metrics), default=0
                ),
                "maximum_inlier_ratio": max(
                    (pair.inlier_ratio for pair in boundary_metrics), default=0.0
                ),
            }
        )
    return tuple(rows)


def targeted_gate(
    selected_bridges: Sequence[BridgePairMetrics],
    config: BridgeSearchConfig = BridgeSearchConfig(),
) -> TargetedGateResult:
    for boundary_left, boundary_right in config.boundaries:
        found = any(
            (
                pair.candidate.boundary_left,
                pair.candidate.boundary_right,
            )
            == (boundary_left, boundary_right)
            and qualified_bridge(pair, config)
            for pair in selected_bridges
        )
        if not found:
            return TargetedGateResult(
                allowed=False,
                reason=(
                    "no selected qualified bridge for boundary "
                    f"{boundary_left}-{boundary_right}"
                ),
            )
    return TargetedGateResult(
        allowed=True,
        reason="all critical boundaries have a selected qualified bridge",
    )


def _feature_marker_payload(
    metrics: DatabaseMetrics,
    selection_manifest_sha256: str,
    config: SparseRunConfig,
) -> dict[str, object]:
    return {
        "camera_model": config.camera_model,
        "feature_count": metrics.feature_count,
        "image_count": metrics.image_count,
        "max_image_size": config.max_image_size,
        "max_num_features": config.max_num_features,
        "pycolmap_version": str(pycolmap.__version__),
        "selection_manifest_sha256": selection_manifest_sha256,
    }


def _feature_database_layout(
    database_path: Path,
) -> tuple[tuple[int, str, int, int, int, int, int], ...]:
    """Return immutable image/feature metadata used to validate cached DB identity."""
    if not database_path.is_file():
        raise ValueError(f"features database is missing: {database_path}")
    connection = sqlite3.connect(str(database_path))
    try:
        rows = connection.execute(
            """
            SELECT
                images.image_id,
                images.name,
                images.camera_id,
                keypoints.rows,
                keypoints.cols,
                descriptors.rows,
                descriptors.cols
            FROM images
            LEFT JOIN keypoints ON keypoints.image_id = images.image_id
            LEFT JOIN descriptors ON descriptors.image_id = images.image_id
            ORDER BY images.image_id
            """
        ).fetchall()
    finally:
        connection.close()
    return tuple(
        (
            int(image_id),
            str(name),
            int(camera_id),
            int(keypoint_rows) if keypoint_rows is not None else -1,
            int(keypoint_cols) if keypoint_cols is not None else -1,
            int(descriptor_rows) if descriptor_rows is not None else -1,
            int(descriptor_cols) if descriptor_cols is not None else -1,
        )
        for (
            image_id,
            name,
            camera_id,
            keypoint_rows,
            keypoint_cols,
            descriptor_rows,
            descriptor_cols,
        ) in rows
    )


def _validate_feature_database_layout(
    database_path: Path,
    expected_filenames: Sequence[str],
) -> tuple[tuple[int, str, int, int, int, int, int], ...]:
    layout = _feature_database_layout(database_path)
    if len(layout) != len(expected_filenames):
        raise RuntimeError(
            f"feature cache contains feature rows for {len(layout)} images, "
            f"expected {len(expected_filenames)}"
        )
    actual_filenames = {row[1] for row in layout}
    if actual_filenames != set(expected_filenames):
        raise RuntimeError("feature cache image names do not match the selected manifest")
    if any(row[3] < 0 or row[5] < 0 for row in layout):
        raise RuntimeError("feature cache is missing a keypoint or descriptor row")
    if any(row[3] != row[5] for row in layout):
        raise RuntimeError("feature cache keypoint and descriptor counts do not agree")
    return layout


def prepare_feature_cache(
    image_dir: Path,
    selection_manifest: Path,
    work_dir: Path,
    sparse_config: SparseRunConfig = SparseRunConfig(),
    *,
    feature_extractor: Callable[
        [Path, Path, SparseRunConfig], DatabaseMetrics
    ] = extract_sparse_features,
) -> Path:
    sparse_config.validate()
    verified = verify_selected_images(
        image_dir,
        selection_manifest,
        expected_count=sparse_config.expected_images,
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    database_path = work_dir / "features.db"
    marker_path = work_dir / "features_complete.json"

    expected_filenames = tuple(record.filename for record in verified.records)
    if database_path.is_file() and marker_path.is_file():
        try:
            metrics = summarize_database(database_path)
            _validate_feature_database_layout(database_path, expected_filenames)
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            expected_marker = _feature_marker_payload(
                metrics, verified.manifest_sha256, sparse_config
            )
            if (
                marker == expected_marker
                and metrics.image_count == sparse_config.expected_images
                and metrics.feature_count > 0
            ):
                return database_path
        except (
            json.JSONDecodeError,
            OSError,
            sqlite3.DatabaseError,
            RuntimeError,
            ValueError,
        ):
            pass

    marker_path.unlink(missing_ok=True)
    temporary_marker = marker_path.with_suffix(".json.tmp")
    temporary_marker.unlink(missing_ok=True)
    if database_path.exists():
        if not database_path.is_file():
            raise ValueError(f"feature database path is not a file: {database_path}")
        database_path.unlink()

    feature_extractor(image_dir, database_path, sparse_config)
    metrics = summarize_database(database_path)
    if metrics.image_count != sparse_config.expected_images:
        raise RuntimeError(
            f"feature cache contains {metrics.image_count} images, "
            f"expected {sparse_config.expected_images}"
        )
    if metrics.feature_count <= 0:
        raise RuntimeError("feature cache contains no SIFT features")
    _validate_feature_database_layout(database_path, expected_filenames)
    marker = _feature_marker_payload(
        metrics, verified.manifest_sha256, sparse_config
    )
    temporary_marker.write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_marker.replace(marker_path)
    return database_path


def _best_model(models: Sequence[ModelMetrics]) -> ModelMetrics:
    if not models:
        raise ValueError("at least one sparse model is required")
    return max(
        models,
        key=lambda model: (
            model.registered_images,
            model.sparse_points,
            -(
                model.mean_reprojection_error
                if math.isfinite(model.mean_reprojection_error)
                else float("inf")
            ),
        ),
    )


def run_targeted_attempt(
    image_dir: Path,
    features_database: Path,
    output_dir: Path,
    selected_bridges: Sequence[BridgePairMetrics],
    sparse_config: SparseRunConfig = SparseRunConfig(),
    bridge_config: BridgeSearchConfig = BridgeSearchConfig(),
) -> AttemptMetrics:
    sparse_config.validate()
    gate = targeted_gate(selected_bridges, bridge_config)
    if not gate.allowed:
        raise ValueError(gate.reason)
    if not features_database.is_file():
        raise ValueError(f"features database is missing: {features_database}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"targeted sparse output is not empty: {output_dir}")

    work_dir = output_dir.parent / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    targeted_database = work_dir / "targeted.db"
    targeted_pair_list = work_dir / "targeted_bridge_pairs.txt"
    if targeted_database.exists():
        if not targeted_database.is_file():
            raise ValueError(f"targeted database path is not a file: {targeted_database}")
        targeted_database.unlink()
    shutil.copy2(features_database, targeted_database)

    started = time.perf_counter()
    pycolmap.match_sequential(
        database_path=targeted_database,
        pairing_options=pycolmap.SequentialPairingOptions(
            overlap=bridge_config.targeted_sequential_overlap,
            quadratic_overlap=True,
            loop_detection=False,
        ),
        device=pycolmap.Device.cpu,
    )
    write_pair_list(targeted_pair_list, selected_bridges)
    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.use_gpu = False
    pairing_options = pycolmap.ImportedPairingOptions()
    pairing_options.match_list_path = targeted_pair_list
    pycolmap.match_image_pairs(
        database_path=targeted_database,
        matching_options=matching_options,
        pairing_options=pairing_options,
        device=pycolmap.Device.cpu,
    )
    models = map_sparse_database(
        targeted_database, image_dir, output_dir, sparse_config
    )
    runtime_seconds = time.perf_counter() - started
    return AttemptMetrics(
        name="targeted_bridges",
        workspace=output_dir,
        overlap=bridge_config.targeted_sequential_overlap,
        database=summarize_database(targeted_database),
        models=models,
        best_model=_best_model(models),
        runtime_seconds=runtime_seconds,
        pycolmap_version=str(pycolmap.__version__),
    )


def bridge_model_accepted(
    model: ModelMetrics,
    bridge_config: BridgeSearchConfig = BridgeSearchConfig(),
) -> bool:
    return bool(
        model.registered_images >= bridge_config.minimum_registered_images
        and model.sparse_points >= bridge_config.minimum_sparse_points
        and model.camera_count == 1
        and model.camera_model == "SIMPLE_RADIAL"
        and math.isfinite(model.mean_reprojection_error)
    )


def choose_bridge_attempt(
    attempts: Sequence[AttemptMetrics],
    bridge_config: BridgeSearchConfig = BridgeSearchConfig(),
) -> AttemptMetrics:
    if not attempts:
        raise ValueError("at least one Step 11 attempt is required")

    def rank(attempt: AttemptMetrics) -> tuple[bool, int, int, float]:
        model = attempt.best_model
        finite_error = (
            model.mean_reprojection_error
            if math.isfinite(model.mean_reprojection_error)
            else float("inf")
        )
        return (
            bridge_model_accepted(model, bridge_config),
            model.registered_images,
            model.sparse_points,
            -finite_error,
        )

    return max(attempts, key=rank)


def run_exhaustive_attempt(
    image_dir: Path,
    features_database: Path,
    output_dir: Path,
    sparse_config: SparseRunConfig = SparseRunConfig(),
    bridge_config: BridgeSearchConfig = BridgeSearchConfig(),
) -> AttemptMetrics:
    sparse_config.validate()
    if not features_database.is_file():
        raise ValueError(f"features database is missing: {features_database}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"exhaustive sparse output is not empty: {output_dir}")

    work_dir = output_dir.parent / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    exhaustive_database = work_dir / "exhaustive.db"
    if exhaustive_database.exists():
        if not exhaustive_database.is_file():
            raise ValueError(
                f"exhaustive database path is not a file: {exhaustive_database}"
            )
        source_metrics = summarize_database(features_database)
        resumed_metrics = summarize_database(exhaustive_database)
        if (
            resumed_metrics.image_count != source_metrics.image_count
            or resumed_metrics.feature_count != source_metrics.feature_count
            or _feature_database_layout(exhaustive_database)
            != _feature_database_layout(features_database)
        ):
            raise ValueError(
                "partial exhaustive database does not match the verified feature cache"
            )
    else:
        shutil.copy2(features_database, exhaustive_database)

    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.use_gpu = False
    pairing_options = pycolmap.ExhaustivePairingOptions()
    pairing_options.block_size = bridge_config.exhaustive_block_size
    started = time.perf_counter()
    pycolmap.match_exhaustive(
        database_path=exhaustive_database,
        matching_options=matching_options,
        pairing_options=pairing_options,
        device=pycolmap.Device.cpu,
    )
    models = map_sparse_database(
        exhaustive_database, image_dir, output_dir, sparse_config
    )
    runtime_seconds = time.perf_counter() - started
    return AttemptMetrics(
        name="exhaustive",
        workspace=output_dir,
        overlap=0,
        database=summarize_database(exhaustive_database),
        models=models,
        best_model=_best_model(models),
        runtime_seconds=runtime_seconds,
        pycolmap_version=str(pycolmap.__version__),
    )
