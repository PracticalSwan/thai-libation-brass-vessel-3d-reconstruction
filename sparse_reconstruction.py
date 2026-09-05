"""Step 10 sparse Structure-from-Motion helpers built on pyCOLMAP."""

from __future__ import annotations

import math
import shutil
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pycolmap


FULL_FRAME_DIAGONAL_MM = math.hypot(36.0, 24.0)
SPARSE_MODEL_FILENAMES = (
    "cameras.bin",
    "images.bin",
    "points3D.bin",
    "rigs.bin",
    "frames.bin",
)


@dataclass(frozen=True)
class SparseRunConfig:
    expected_images: int = 288
    image_width: int = 3072
    image_height: int = 4080
    focal_35mm: float = 26.0
    camera_model: str = "SIMPLE_RADIAL"
    max_image_size: int = 1200
    max_num_features: int = 8192
    baseline_overlap: int = 20
    retry_overlap: int = 40
    minimum_registered_images: int = 274
    minimum_sparse_points: int = 1000
    minimum_matches: int = 15
    min_model_size: int = 10
    random_seed: int = 4213

    def validate(self) -> "SparseRunConfig":
        if self.expected_images < 2:
            raise ValueError("expected_images must be at least two")
        if self.image_width < 1 or self.image_height < 1:
            raise ValueError("image geometry must be positive")
        if not math.isfinite(self.focal_35mm) or self.focal_35mm <= 0:
            raise ValueError("focal_35mm must be finite and positive")
        if self.max_image_size < 1 or self.max_num_features < 1:
            raise ValueError("feature-extraction limits must be positive")
        if self.baseline_overlap < 1 or self.retry_overlap < 1:
            raise ValueError("sequential overlap must be positive")
        if self.minimum_registered_images < 2:
            raise ValueError("minimum_registered_images must be at least two")
        if self.minimum_sparse_points < 1:
            raise ValueError("minimum_sparse_points must be positive")
        if self.camera_model != "SIMPLE_RADIAL":
            raise ValueError("Step 10 camera model is frozen to SIMPLE_RADIAL")
        return self


@dataclass(frozen=True)
class DatabaseMetrics:
    image_count: int
    feature_count: int
    matched_pair_count: int
    verified_pair_count: int


@dataclass(frozen=True)
class ModelMetrics:
    model_path: Path
    registered_images: int
    total_images: int
    sparse_points: int
    observations: int
    mean_track_length: float
    mean_reprojection_error: float
    camera_count: int
    camera_model: str
    camera_params: tuple[float, ...]
    mean_observations_per_registered_image: float = math.nan
    registered_image_names: tuple[str, ...] = field(default_factory=tuple)

    @property
    def registration_fraction(self) -> float:
        return self.registered_images / self.total_images if self.total_images else 0.0

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["model_path"] = self.model_path.as_posix()
        payload["registration_fraction"] = self.registration_fraction
        return payload


@dataclass(frozen=True)
class AttemptMetrics:
    name: str
    workspace: Path
    overlap: int
    database: DatabaseMetrics
    models: tuple[ModelMetrics, ...]
    best_model: ModelMetrics
    runtime_seconds: float = 0.0
    pycolmap_version: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "workspace": self.workspace.as_posix(),
            "overlap": self.overlap,
            "database": asdict(self.database),
            "models": [model.to_dict() for model in self.models],
            "best_model": self.best_model.to_dict(),
            "runtime_seconds": self.runtime_seconds,
            "pycolmap_version": self.pycolmap_version,
        }


def focal_pixels_from_35mm_equivalent(
    width: int, height: int, focal_35mm: float
) -> float:
    if width < 1 or height < 1:
        raise ValueError("image geometry must be positive")
    if not math.isfinite(focal_35mm) or focal_35mm <= 0:
        raise ValueError("focal_35mm must be finite and positive")
    pixel_diagonal = math.hypot(float(width), float(height))
    return focal_35mm / FULL_FRAME_DIAGONAL_MM * pixel_diagonal


def simple_radial_camera_params(
    width: int, height: int, focal_35mm: float
) -> tuple[float, float, float, float]:
    focal = focal_pixels_from_35mm_equivalent(width, height, focal_35mm)
    return focal, width / 2.0, height / 2.0, 0.0


def should_retry(metrics: ModelMetrics, total_images: int) -> bool:
    if total_images < 1:
        raise ValueError("total_images must be positive")
    required_registered = math.ceil(total_images * 0.95)
    required_registered = max(required_registered, 274 if total_images == 288 else 2)
    return bool(
        metrics.registered_images < required_registered
        or metrics.sparse_points < 1000
        or not math.isfinite(metrics.mean_reprojection_error)
        or metrics.camera_count != 1
    )


def attempt_requires_retry(
    attempt: AttemptMetrics, total_images: int, *, meaningful_component_size: int = 10
) -> bool:
    if should_retry(attempt.best_model, total_images):
        return True
    meaningful_models = sum(
        model.registered_images >= meaningful_component_size for model in attempt.models
    )
    return meaningful_models > 1


def choose_best_attempt(attempts: Sequence[AttemptMetrics]) -> AttemptMetrics:
    if not attempts:
        raise ValueError("at least one sparse attempt is required")

    def key(attempt: AttemptMetrics) -> tuple[int, int, float]:
        model = attempt.best_model
        error = model.mean_reprojection_error
        finite_error = error if math.isfinite(error) else float("inf")
        return model.registered_images, model.sparse_points, -finite_error

    return max(attempts, key=key)


def _paths_overlap(first: Path, second: Path) -> bool:
    first_resolved = first.resolve()
    second_resolved = second.resolve()
    return (
        first_resolved == second_resolved
        or first_resolved in second_resolved.parents
        or second_resolved in first_resolved.parents
    )


def validate_workspace_boundary(image_dir: Path, workspace: Path) -> None:
    if _paths_overlap(image_dir, workspace):
        raise ValueError("reconstruction workspace must not overlap selected images")


def summarize_database(database_path: Path) -> DatabaseMetrics:
    if not database_path.is_file():
        raise ValueError(f"COLMAP database is missing: {database_path}")
    connection = sqlite3.connect(str(database_path))
    try:
        image_count = int(connection.execute("SELECT COUNT(*) FROM images").fetchone()[0])
        feature_count = int(
            connection.execute("SELECT COALESCE(SUM(rows), 0) FROM keypoints").fetchone()[0]
        )
        matched_pair_count = int(
            connection.execute("SELECT COUNT(*) FROM matches WHERE rows > 0").fetchone()[0]
        )
        verified_pair_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0"
            ).fetchone()[0]
        )
    finally:
        connection.close()
    return DatabaseMetrics(
        image_count=image_count,
        feature_count=feature_count,
        matched_pair_count=matched_pair_count,
        verified_pair_count=verified_pair_count,
    )


def _camera_summary(reconstruction: pycolmap.Reconstruction) -> tuple[int, str, tuple[float, ...]]:
    camera_count = int(reconstruction.num_cameras())
    if camera_count == 0:
        return 0, "", ()
    camera_ids = sorted(int(camera_id) for camera_id in reconstruction.cameras.keys())
    first = reconstruction.camera(camera_ids[0])
    model_name = str(first.model_name)
    params = tuple(float(value) for value in np.asarray(first.params).reshape(-1))
    return camera_count, model_name, params


def registered_image_names_from_reconstruction(
    reconstruction: pycolmap.Reconstruction,
) -> tuple[str, ...]:
    names = [reconstruction.image(int(image_id)).name for image_id in reconstruction.reg_image_ids()]
    return tuple(sorted(str(name) for name in names))


def summarize_reconstruction(
    model_path: Path, total_images: int
) -> ModelMetrics:
    reconstruction = pycolmap.Reconstruction(model_path)
    camera_count, camera_model, camera_params = _camera_summary(reconstruction)
    registered_images = int(reconstruction.num_reg_images())
    sparse_points = int(reconstruction.num_points3D())
    observations = int(reconstruction.compute_num_observations())
    mean_track_length = float(reconstruction.compute_mean_track_length())
    mean_reprojection_error = float(reconstruction.compute_mean_reprojection_error())
    mean_observations = float(reconstruction.compute_mean_observations_per_reg_image())
    return ModelMetrics(
        model_path=model_path,
        registered_images=registered_images,
        total_images=total_images,
        sparse_points=sparse_points,
        observations=observations,
        mean_track_length=mean_track_length,
        mean_reprojection_error=mean_reprojection_error,
        camera_count=camera_count,
        camera_model=camera_model,
        camera_params=camera_params,
        mean_observations_per_registered_image=mean_observations,
        registered_image_names=registered_image_names_from_reconstruction(reconstruction),
    )


def registered_image_names(model_path: Path) -> tuple[str, ...]:
    return registered_image_names_from_reconstruction(pycolmap.Reconstruction(model_path))


def build_image_reader_options(config: SparseRunConfig) -> pycolmap.ImageReaderOptions:
    config.validate()
    camera_params = simple_radial_camera_params(
        config.image_width, config.image_height, config.focal_35mm
    )
    return pycolmap.ImageReaderOptions(
        camera_model=config.camera_model,
        camera_params=",".join(f"{value:.12g}" for value in camera_params),
    )


def build_feature_extraction_options(
    config: SparseRunConfig,
) -> pycolmap.FeatureExtractionOptions:
    config.validate()
    options = pycolmap.FeatureExtractionOptions()
    options.max_image_size = config.max_image_size
    options.sift.max_num_features = config.max_num_features
    options.use_gpu = False
    return options


def build_incremental_pipeline_options(
    config: SparseRunConfig,
) -> pycolmap.IncrementalPipelineOptions:
    config.validate()
    options = pycolmap.IncrementalPipelineOptions()
    options.min_num_matches = config.minimum_matches
    options.multiple_models = True
    options.min_model_size = config.min_model_size
    options.random_seed = config.random_seed
    options.ba_refine_focal_length = True
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = True
    options.mapper.random_seed = config.random_seed
    options.triangulation.random_seed = config.random_seed
    return options


def extract_sparse_features(
    image_dir: Path,
    database_path: Path,
    config: SparseRunConfig = SparseRunConfig(),
) -> DatabaseMetrics:
    config.validate()
    if not image_dir.is_dir():
        raise ValueError(f"selected image directory is missing: {image_dir}")
    if database_path.exists() and (
        not database_path.is_file() or database_path.stat().st_size > 0
    ):
        raise ValueError(f"feature database destination is not empty: {database_path}")
    database_path.parent.mkdir(parents=True, exist_ok=True)
    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=build_image_reader_options(config),
        extraction_options=build_feature_extraction_options(config),
        device=pycolmap.Device.cpu,
    )
    metrics = summarize_database(database_path)
    if metrics.image_count != config.expected_images:
        raise RuntimeError(
            f"pyCOLMAP database contains {metrics.image_count} images, "
            f"expected {config.expected_images}"
        )
    return metrics


def map_sparse_database(
    database_path: Path,
    image_dir: Path,
    output_dir: Path,
    config: SparseRunConfig = SparseRunConfig(),
) -> tuple[ModelMetrics, ...]:
    config.validate()
    if not database_path.is_file():
        raise ValueError(f"COLMAP database is missing: {database_path}")
    if not image_dir.is_dir():
        raise ValueError(f"selected image directory is missing: {image_dir}")
    validate_workspace_boundary(image_dir, output_dir)
    if output_dir.exists():
        allowed_database = database_path.resolve()
        unexpected = [
            path
            for path in output_dir.iterdir()
            if path.resolve() != allowed_database
        ]
        if unexpected:
            raise ValueError(f"sparse mapping output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    reconstructions = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=output_dir,
        options=build_incremental_pipeline_options(config),
    )
    if not reconstructions:
        raise RuntimeError("pyCOLMAP incremental mapping produced no sparse model")

    model_metrics: list[ModelMetrics] = []
    for model_id in sorted(int(value) for value in reconstructions.keys()):
        model_path = output_dir / str(model_id)
        if not model_path.is_dir():
            reconstructions[model_id].write(model_path)
        model_metrics.append(summarize_reconstruction(model_path, config.expected_images))
    return tuple(model_metrics)


def _attempt_name(overlap: int, config: SparseRunConfig) -> str:
    return "baseline" if overlap == config.baseline_overlap else f"retry_overlap{overlap}"


def run_sparse_attempt(
    image_dir: Path,
    workspace: Path,
    *,
    overlap: int,
    config: SparseRunConfig = SparseRunConfig(),
) -> AttemptMetrics:
    config.validate()
    if overlap < 1:
        raise ValueError("sequential overlap must be positive")
    if not image_dir.is_dir():
        raise ValueError(f"selected image directory is missing: {image_dir}")
    validate_workspace_boundary(image_dir, workspace)
    image_files = sorted(image_dir.glob("*.jpg"))
    if len(image_files) != config.expected_images:
        raise ValueError(
            f"expected {config.expected_images} selected JPEGs, found {len(image_files)}"
        )
    if workspace.exists() and any(workspace.iterdir()):
        raise ValueError(f"sparse attempt workspace is not empty: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    database_path = workspace / "database.db"
    sparse_dir = workspace

    started = time.perf_counter()
    extract_sparse_features(image_dir, database_path, config)
    pycolmap.match_sequential(
        database_path=database_path,
        pairing_options=pycolmap.SequentialPairingOptions(
            overlap=overlap,
            quadratic_overlap=True,
            loop_detection=False,
        ),
        device=pycolmap.Device.cpu,
    )
    database_metrics = summarize_database(database_path)
    model_metrics = map_sparse_database(database_path, image_dir, sparse_dir, config)
    runtime_seconds = time.perf_counter() - started
    best_model = max(
        model_metrics,
        key=lambda model: (
            model.registered_images,
            model.sparse_points,
            -(model.mean_reprojection_error if math.isfinite(model.mean_reprojection_error) else float("inf")),
        ),
    )
    return AttemptMetrics(
        name=_attempt_name(overlap, config),
        workspace=workspace,
        overlap=overlap,
        database=database_metrics,
        models=tuple(model_metrics),
        best_model=best_model,
        runtime_seconds=runtime_seconds,
        pycolmap_version=str(pycolmap.__version__),
    )


def copy_sparse_model(source: Path, destination: Path) -> tuple[Path, ...]:
    if not source.is_dir():
        raise ValueError(f"sparse source model is missing: {source}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for filename in SPARSE_MODEL_FILENAMES:
        source_file = source / filename
        if source_file.is_file():
            target = destination / filename
            shutil.copy2(source_file, target)
            copied.append(target)
    if not copied:
        raise ValueError(f"no COLMAP sparse model files found in {source}")
    return tuple(copied)
