"""Step 12 native learned-feature sparse-recovery helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import sys
import time
from typing import Callable, Sequence

import numpy as np
import pycolmap

from analysis_common import verify_selected_images
from sparse_bridging import (
    BridgeCandidate,
    BridgePairMetrics,
    BridgeSearchConfig,
    bridge_model_accepted,
    summarize_bridge_pairs,
    targeted_gate,
    write_pair_list,
)
from sparse_reconstruction import (
    AttemptMetrics,
    DatabaseMetrics,
    ModelMetrics,
    SparseRunConfig,
    build_image_reader_options,
    map_sparse_database,
    summarize_database,
    validate_workspace_boundary,
)


@dataclass(frozen=True)
class LearnedFrontendSpec:
    name: str
    extractor_type: pycolmap.FeatureExtractorType
    matcher_type: pycolmap.FeatureMatcherType
    max_image_size: int = 1600


@dataclass(frozen=True)
class LearnedRecoveryConfig:
    expected_images: int = 288
    minimum_registered_images: int = 274
    minimum_sparse_points: int = 1000
    targeted_overlap: int = 20
    visual_status_pending: str = "pending"


@dataclass(frozen=True)
class FrontendSmokeResult:
    frontend: str
    status: str
    first_features: int
    second_features: int
    raw_matches: int
    exception_type: str = ""
    exception_message: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "frontend": self.frontend,
            "status": self.status,
            "first_features": self.first_features,
            "second_features": self.second_features,
            "raw_matches": self.raw_matches,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
        }


@dataclass(frozen=True)
class LearnedFeatureDatabaseIdentity:
    image_count: int
    feature_count: int
    camera_ids: tuple[int, ...]
    layout: tuple[tuple[int, str, int, int, int, int, int], ...]


ALIKED_FRONTEND = LearnedFrontendSpec(
    name="aliked",
    extractor_type=pycolmap.FeatureExtractorType.ALIKED_N16ROT,
    matcher_type=pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE,
)
LOMA_FRONTEND = LearnedFrontendSpec(
    name="loma",
    extractor_type=pycolmap.FeatureExtractorType.LOMA_B,
    matcher_type=pycolmap.FeatureMatcherType.LOMA_L,
)
FRONTENDS = {
    ALIKED_FRONTEND.name: ALIKED_FRONTEND,
    LOMA_FRONTEND.name: LOMA_FRONTEND,
}


def build_learned_extraction_options(
    frontend: LearnedFrontendSpec,
) -> pycolmap.FeatureExtractionOptions:
    options = pycolmap.FeatureExtractionOptions()
    options.type = frontend.extractor_type
    options.max_image_size = frontend.max_image_size
    options.use_gpu = False
    if not options.check():
        raise ValueError(f"invalid learned extraction options for {frontend.name}")
    return options


def build_learned_matching_options(
    frontend: LearnedFrontendSpec,
) -> pycolmap.FeatureMatchingOptions:
    options = pycolmap.FeatureMatchingOptions()
    options.type = frontend.matcher_type
    options.use_gpu = False
    if not options.check():
        raise ValueError(f"invalid learned matching options for {frontend.name}")
    return options


def _json_option_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_option_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_option_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_capability_snapshot() -> dict[str, object]:
    frontends: dict[str, object] = {}
    for name, frontend in FRONTENDS.items():
        extraction = build_learned_extraction_options(frontend)
        matching = build_learned_matching_options(frontend)
        frontends[name] = {
            "extractor": frontend.extractor_type.name,
            "matcher": frontend.matcher_type.name,
            "max_image_size": frontend.max_image_size,
            "effective_max_image_size": int(extraction.eff_max_image_size()),
            "requires_rgb": bool(extraction.requires_rgb()),
            "requires_opengl": bool(extraction.requires_opengl()),
            "extraction_options_valid": bool(extraction.check()),
            "matching_options_valid": bool(matching.check()),
            "extraction_options": _json_option_value(extraction.todict()),
            "matching_options": _json_option_value(matching.todict()),
        }
    return {
        "python_version": sys.version.split()[0],
        "pycolmap_version": str(pycolmap.__version__),
        "has_cuda": bool(pycolmap.has_cuda),
        "extractor_members": list(pycolmap.FeatureExtractorType.__members__.keys()),
        "matcher_members": list(pycolmap.FeatureMatcherType.__members__.keys()),
        "device_policy": "cpu",
        "use_gpu": False,
        "frontends": frontends,
    }


def _require_rgb_uint8(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise ValueError("smoke image must be a NumPy array")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("smoke image must be an RGB uint8 array")


def _feature_count(features: object) -> int:
    try:
        return len(features)  # type: ignore[arg-type]
    except TypeError:
        array = np.asarray(features)
        return int(array.shape[0]) if array.ndim else 0


def smoke_frontend(
    frontend: LearnedFrontendSpec,
    first_image: np.ndarray,
    second_image: np.ndarray,
    *,
    extractor_factory: Callable[..., object] = pycolmap.FeatureExtractor.create,
    matcher_factory: Callable[..., object] = pycolmap.FeatureMatcher.create,
) -> FrontendSmokeResult:
    first_count = 0
    second_count = 0
    try:
        _require_rgb_uint8(first_image)
        _require_rgb_uint8(second_image)
        extractor = extractor_factory(
            options=build_learned_extraction_options(frontend),
            device=pycolmap.Device.cpu,
        )
        first_keypoints, first_descriptors = extractor.extract_from_uint8_array(  # type: ignore[attr-defined]
            first_image
        )
        second_keypoints, second_descriptors = extractor.extract_from_uint8_array(  # type: ignore[attr-defined]
            second_image
        )
        first_count = _feature_count(first_keypoints)
        second_count = _feature_count(second_keypoints)
        if first_count <= 0 or second_count <= 0:
            raise ValueError("learned smoke extraction returned no features")
        if first_count != _feature_count(first_descriptors) or second_count != _feature_count(
            second_descriptors
        ):
            raise ValueError("learned smoke keypoint and descriptor counts do not agree")
        matcher = matcher_factory(
            options=build_learned_matching_options(frontend),
            device=pycolmap.Device.cpu,
        )
        matches = np.asarray(
            matcher.match(  # type: ignore[attr-defined]
                first_keypoints,
                first_descriptors,
                second_keypoints,
                second_descriptors,
            )
        )
        if matches.ndim != 2 or matches.shape[1] != 2:
            raise ValueError("learned smoke matches must have shape N x 2")
        raw_matches = int(matches.shape[0])
        if raw_matches <= 0:
            raise ValueError("learned smoke matcher returned no matches")
        return FrontendSmokeResult(
            frontend.name, "passed", first_count, second_count, raw_matches
        )
    except Exception as error:
        return FrontendSmokeResult(
            frontend=frontend.name,
            status="blocked",
            first_features=first_count,
            second_features=second_count,
            raw_matches=0,
            exception_type=type(error).__name__,
            exception_message=str(error),
        )


def _learned_database_layout(
    database_path: Path,
) -> tuple[tuple[int, str, int, int, int, int, int], ...]:
    if not database_path.is_file():
        raise ValueError(f"learned feature database is missing: {database_path}")
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


def validate_learned_feature_database(
    database_path: Path,
    expected_filenames: Sequence[str],
    *,
    expected_camera_ids: tuple[int, ...] = (1,),
) -> LearnedFeatureDatabaseIdentity:
    layout = _learned_database_layout(database_path)
    if len(layout) != len(expected_filenames):
        raise RuntimeError(
            f"learned feature cache contains {len(layout)} images, "
            f"expected {len(expected_filenames)}"
        )
    if {row[1] for row in layout} != set(expected_filenames):
        raise RuntimeError("learned feature cache image names do not match the selected manifest")
    camera_ids = tuple(sorted({row[2] for row in layout}))
    if camera_ids != expected_camera_ids:
        raise RuntimeError("learned feature cache camera IDs do not match the shared-camera contract")
    if any(row[3] < 0 or row[5] < 0 for row in layout):
        raise RuntimeError("learned feature cache is missing a keypoint or descriptor row")
    if any(row[3] != row[5] for row in layout):
        raise RuntimeError("learned feature cache keypoint and descriptor counts do not agree")
    if any(row[3] <= 0 for row in layout):
        raise RuntimeError("learned feature cache contains no learned features for an image")
    return LearnedFeatureDatabaseIdentity(
        image_count=len(layout),
        feature_count=sum(row[3] for row in layout),
        camera_ids=camera_ids,
        layout=layout,
    )


def _feature_layout_sha256(identity: LearnedFeatureDatabaseIdentity) -> str:
    encoded = json.dumps(identity.layout, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _feature_marker_payload(
    identity: LearnedFeatureDatabaseIdentity,
    selection_manifest_sha256: str,
    frontend: LearnedFrontendSpec,
    sparse_config: SparseRunConfig,
) -> dict[str, object]:
    return {
        "frontend": frontend.name,
        "extractor_type": frontend.extractor_type.name,
        "matcher_type": frontend.matcher_type.name,
        "max_image_size": frontend.max_image_size,
        "image_count": identity.image_count,
        "feature_count": identity.feature_count,
        "layout_sha256": _feature_layout_sha256(identity),
        "camera_model": sparse_config.camera_model,
        "camera_ids": list(identity.camera_ids),
        "pycolmap_version": str(pycolmap.__version__),
        "device": "cpu",
        "selection_manifest_sha256": selection_manifest_sha256,
    }


def extract_learned_features(
    image_dir: Path,
    database_path: Path,
    frontend: LearnedFrontendSpec,
    sparse_config: SparseRunConfig = SparseRunConfig(),
) -> DatabaseMetrics:
    sparse_config.validate()
    if not image_dir.is_dir():
        raise ValueError(f"selected image directory is missing: {image_dir}")
    if database_path.exists() and (
        not database_path.is_file() or database_path.stat().st_size > 0
    ):
        raise ValueError(f"learned feature database destination is not empty: {database_path}")
    validate_workspace_boundary(image_dir, database_path.parent)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=build_image_reader_options(sparse_config),
        extraction_options=build_learned_extraction_options(frontend),
        device=pycolmap.Device.cpu,
    )
    metrics = summarize_database(database_path)
    if metrics.image_count != sparse_config.expected_images:
        raise RuntimeError(
            f"learned feature database contains {metrics.image_count} images, "
            f"expected {sparse_config.expected_images}"
        )
    return metrics


def prepare_learned_feature_cache(
    image_dir: Path,
    selection_manifest: Path,
    work_dir: Path,
    frontend: LearnedFrontendSpec,
    sparse_config: SparseRunConfig = SparseRunConfig(),
    *,
    feature_extractor: Callable[
        [Path, Path, LearnedFrontendSpec, SparseRunConfig], DatabaseMetrics
    ] = extract_learned_features,
) -> Path:
    sparse_config.validate()
    verified = verify_selected_images(
        image_dir, selection_manifest, expected_count=sparse_config.expected_images
    )
    frontend_work = work_dir / frontend.name
    validate_workspace_boundary(image_dir, frontend_work)
    frontend_work.mkdir(parents=True, exist_ok=True)
    database_path = frontend_work / "features.db"
    marker_path = frontend_work / "features_complete.json"
    temporary_marker = marker_path.with_suffix(".json.tmp")
    expected_filenames = tuple(record.filename for record in verified.records)

    if database_path.is_file() and marker_path.is_file():
        try:
            identity = validate_learned_feature_database(
                database_path, expected_filenames, expected_camera_ids=(1,)
            )
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            expected_marker = _feature_marker_payload(
                identity, verified.manifest_sha256, frontend, sparse_config
            )
            if marker == expected_marker:
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
    temporary_marker.unlink(missing_ok=True)
    if database_path.exists():
        if not database_path.is_file() or database_path.is_symlink():
            raise ValueError(f"learned feature database path is not a regular file: {database_path}")
        database_path.unlink()
    feature_extractor(image_dir, database_path, frontend, sparse_config)
    identity = validate_learned_feature_database(
        database_path, expected_filenames, expected_camera_ids=(1,)
    )
    marker = _feature_marker_payload(
        identity, verified.manifest_sha256, frontend, sparse_config
    )
    temporary_marker.write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_marker.replace(marker_path)
    return database_path


def candidate_identity(
    candidate: BridgeCandidate,
) -> tuple[int, int, int, int, str, str]:
    return (
        candidate.boundary_left,
        candidate.boundary_right,
        candidate.left_index,
        candidate.right_index,
        candidate.left_filename,
        candidate.right_filename,
    )


def verify_step11_candidate_identity(
    candidates: Sequence[BridgeCandidate], step11_candidates_csv: Path
) -> None:
    if not step11_candidates_csv.is_file():
        raise ValueError(f"Step 11 candidate report is missing: {step11_candidates_csv}")
    with step11_candidates_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    try:
        expected = tuple(
            (
                int(row["boundary_left"]),
                int(row["boundary_right"]),
                int(row["left_index"]),
                int(row["right_index"]),
                str(row["left_filename"]),
                str(row["right_filename"]),
            )
            for row in rows
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Step 11 candidate report has an invalid identity schema") from error
    actual = tuple(candidate_identity(candidate) for candidate in candidates)
    if actual != expected:
        raise RuntimeError("Step 12 candidate identity/order does not match Step 11")


def run_learned_diagnostics(
    features_database: Path,
    workspace: Path,
    candidates: Sequence[BridgeCandidate],
    frontend: LearnedFrontendSpec,
    bridge_config: BridgeSearchConfig = BridgeSearchConfig(),
) -> tuple[BridgePairMetrics, ...]:
    if not features_database.is_file():
        raise ValueError(f"learned features database is missing: {features_database}")
    if not candidates:
        raise ValueError("at least one learned diagnostic image pair is required")
    workspace.mkdir(parents=True, exist_ok=True)
    diagnostic_database = workspace / "diagnostic.db"
    pair_list_path = workspace / "diagnostic_pairs.txt"
    if diagnostic_database.exists():
        if not diagnostic_database.is_file() or diagnostic_database.is_symlink():
            raise ValueError(f"diagnostic database path is not a regular file: {diagnostic_database}")
        diagnostic_database.unlink()
    shutil.copy2(features_database, diagnostic_database)
    write_pair_list(pair_list_path, candidates)
    pairing_options = pycolmap.ImportedPairingOptions()
    pairing_options.match_list_path = pair_list_path
    pycolmap.match_image_pairs(
        database_path=diagnostic_database,
        matching_options=build_learned_matching_options(frontend),
        pairing_options=pairing_options,
        device=pycolmap.Device.cpu,
    )
    return summarize_bridge_pairs(diagnostic_database, candidates, bridge_config)


def _best_model(models: Sequence[ModelMetrics]) -> ModelMetrics:
    if not models:
        raise ValueError("at least one learned sparse model is required")
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


def run_learned_targeted_attempt(
    image_dir: Path,
    features_database: Path,
    output_dir: Path,
    selected_bridges: Sequence[BridgePairMetrics],
    frontend: LearnedFrontendSpec,
    sparse_config: SparseRunConfig = SparseRunConfig(),
    bridge_config: BridgeSearchConfig = BridgeSearchConfig(),
) -> AttemptMetrics:
    sparse_config.validate()
    gate = targeted_gate(selected_bridges, bridge_config)
    if not gate.allowed:
        raise ValueError(gate.reason)
    if not features_database.is_file():
        raise ValueError(f"learned features database is missing: {features_database}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"learned targeted sparse output is not empty: {output_dir}")
    validate_workspace_boundary(image_dir, output_dir)

    frontend_work = output_dir.parent / "work" / frontend.name
    frontend_work.mkdir(parents=True, exist_ok=True)
    targeted_database = frontend_work / "targeted.db"
    targeted_pair_list = frontend_work / "targeted_bridge_pairs.txt"
    if targeted_database.exists():
        if not targeted_database.is_file() or targeted_database.is_symlink():
            raise ValueError(f"targeted database path is not a regular file: {targeted_database}")
        targeted_database.unlink()
    shutil.copy2(features_database, targeted_database)

    matching_options = build_learned_matching_options(frontend)
    started = time.perf_counter()
    pycolmap.match_sequential(
        database_path=targeted_database,
        matching_options=matching_options,
        pairing_options=pycolmap.SequentialPairingOptions(
            overlap=bridge_config.targeted_sequential_overlap,
            quadratic_overlap=True,
            loop_detection=False,
        ),
        device=pycolmap.Device.cpu,
    )
    write_pair_list(targeted_pair_list, selected_bridges)
    imported_options = pycolmap.ImportedPairingOptions()
    imported_options.match_list_path = targeted_pair_list
    pycolmap.match_image_pairs(
        database_path=targeted_database,
        matching_options=build_learned_matching_options(frontend),
        pairing_options=imported_options,
        device=pycolmap.Device.cpu,
    )
    models = map_sparse_database(
        targeted_database, image_dir, output_dir, sparse_config
    )
    runtime_seconds = time.perf_counter() - started
    return AttemptMetrics(
        name=f"{frontend.name}_targeted",
        workspace=output_dir,
        overlap=bridge_config.targeted_sequential_overlap,
        database=summarize_database(targeted_database),
        models=tuple(models),
        best_model=_best_model(models),
        runtime_seconds=runtime_seconds,
        pycolmap_version=str(pycolmap.__version__),
    )


def learned_model_metric_accepted(
    model: ModelMetrics,
    bridge_config: BridgeSearchConfig = BridgeSearchConfig(),
) -> bool:
    return bridge_model_accepted(model, bridge_config)
