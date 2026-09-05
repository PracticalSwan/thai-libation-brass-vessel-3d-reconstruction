from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from sparse_reconstruction import (
    AttemptMetrics,
    DatabaseMetrics,
    ModelMetrics,
    SparseRunConfig,
    choose_best_attempt,
    focal_pixels_from_35mm_equivalent,
    should_retry,
    simple_radial_camera_params,
    summarize_database,
    validate_workspace_boundary,
)


def _model(
    *,
    registered_images: int,
    sparse_points: int = 2000,
    mean_reprojection_error: float = 0.8,
    camera_count: int = 1,
) -> ModelMetrics:
    return ModelMetrics(
        model_path=Path("model"),
        registered_images=registered_images,
        total_images=288,
        sparse_points=sparse_points,
        observations=6000,
        mean_track_length=3.0,
        mean_reprojection_error=mean_reprojection_error,
        camera_count=camera_count,
        camera_model="SIMPLE_RADIAL",
        camera_params=(3070.0, 1536.0, 2040.0, 0.0),
    )


def _attempt(name: str, model: ModelMetrics, *, overlap: int) -> AttemptMetrics:
    return AttemptMetrics(
        name=name,
        workspace=Path(name),
        overlap=overlap,
        database=DatabaseMetrics(288, 1000, 500, 400),
        models=(model,),
        best_model=model,
    )


def test_focal_pixels_uses_full_frame_diagonal_equivalence():
    focal = focal_pixels_from_35mm_equivalent(3072, 4080, 26.0)
    assert focal == pytest.approx(3070.0, rel=0.01)


def test_simple_radial_initialization_uses_center_and_zero_distortion():
    focal, cx, cy, k = simple_radial_camera_params(3072, 4080, 26.0)
    assert cx == 1536.0
    assert cy == 2040.0
    assert k == 0.0
    assert focal > 3000.0


def test_retry_gate_requires_registration_points_finite_error_and_single_camera():
    accepted = _model(registered_images=274)
    assert not should_retry(accepted, 288)
    assert should_retry(replace(accepted, registered_images=273), 288)
    assert should_retry(replace(accepted, sparse_points=999), 288)
    assert should_retry(replace(accepted, mean_reprojection_error=float("nan")), 288)
    assert should_retry(replace(accepted, camera_count=2), 288)


def test_choose_best_attempt_prioritizes_registration_then_points_then_error():
    fewer_registered = _attempt("a", _model(registered_images=280, sparse_points=5000), overlap=20)
    more_registered = _attempt("b", _model(registered_images=281, sparse_points=1000, mean_reprojection_error=1.5), overlap=40)
    assert choose_best_attempt((fewer_registered, more_registered)).name == "b"

    more_points = _attempt("c", _model(registered_images=281, sparse_points=2000, mean_reprojection_error=1.5), overlap=20)
    assert choose_best_attempt((more_registered, more_points)).name == "c"

    lower_error = _attempt("d", _model(registered_images=281, sparse_points=2000, mean_reprojection_error=0.7), overlap=40)
    assert choose_best_attempt((more_points, lower_error)).name == "d"


def test_workspace_boundary_rejects_overlap_with_input(tmp_path: Path):
    image_dir = tmp_path / "images"
    workspace = tmp_path / "reconstruction"
    image_dir.mkdir()
    workspace.mkdir()
    validate_workspace_boundary(image_dir, workspace)

    with pytest.raises(ValueError):
        validate_workspace_boundary(image_dir, image_dir / "attempt")
    with pytest.raises(ValueError):
        validate_workspace_boundary(image_dir, tmp_path)


def test_summarize_database_counts_images_features_matches_and_verified_pairs(tmp_path: Path):
    database_path = tmp_path / "database.db"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE keypoints(image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB);
            CREATE TABLE matches(pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
            CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
            """
        )
        connection.executemany(
            "INSERT INTO images(image_id, name) VALUES (?, ?)",
            [(1, "a.jpg"), (2, "b.jpg")],
        )
        connection.executemany(
            "INSERT INTO keypoints(image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
            [(1, 10, 4, b"x"), (2, 20, 4, b"y")],
        )
        connection.executemany(
            "INSERT INTO matches(pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",
            [(1, 8, 2, b"m"), (2, 0, 2, b"")],
        )
        connection.executemany(
            "INSERT INTO two_view_geometries(pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",
            [(1, 7, 2, b"g"), (2, 0, 2, b"")],
        )

    metrics = summarize_database(database_path)
    assert metrics == DatabaseMetrics(
        image_count=2,
        feature_count=30,
        matched_pair_count=1,
        verified_pair_count=1,
    )


def test_sparse_config_rejects_invalid_overlap():
    config = SparseRunConfig()
    assert config.baseline_overlap == 20
    assert config.retry_overlap == 40
    with pytest.raises(ValueError):
        replace(config, baseline_overlap=0).validate()
