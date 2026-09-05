from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

import sparse_reconstruction as sparse_module
from sparse_reconstruction import (
    AttemptMetrics,
    DatabaseMetrics,
    ModelMetrics,
    SparseRunConfig,
    build_feature_extraction_options,
    build_image_reader_options,
    build_incremental_pipeline_options,
    choose_best_attempt,
    extract_sparse_features,
    focal_pixels_from_35mm_equivalent,
    map_sparse_database,
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


def test_build_image_reader_options_preserves_step10_camera_initialization():
    options = build_image_reader_options(SparseRunConfig())

    assert options.camera_model == "SIMPLE_RADIAL"
    assert options.camera_params == "3069.05066755,1536,2040,0"


def test_build_feature_extraction_options_preserves_step10_settings():
    options = build_feature_extraction_options(SparseRunConfig())

    assert options.max_image_size == 1200
    assert options.sift.max_num_features == 8192
    assert options.use_gpu is False


def test_build_incremental_pipeline_options_preserves_step10_settings():
    options = build_incremental_pipeline_options(SparseRunConfig())

    assert options.min_num_matches == 15
    assert options.multiple_models is True
    assert options.min_model_size == 10
    assert options.random_seed == 4213
    assert options.ba_refine_focal_length is True
    assert options.ba_refine_principal_point is False
    assert options.ba_refine_extra_params is True
    assert options.mapper.random_seed == 4213
    assert options.triangulation.random_seed == 4213


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


def test_extract_sparse_features_uses_shared_step10_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "a.jpg").write_bytes(b"a")
    (image_dir / "b.jpg").write_bytes(b"b")
    database_path = tmp_path / "features.db"

    def fake_extract_features(**kwargs):
        assert kwargs["image_path"] == image_dir
        assert kwargs["camera_mode"] == sparse_module.pycolmap.CameraMode.SINGLE
        assert kwargs["reader_options"].camera_model == "SIMPLE_RADIAL"
        assert kwargs["extraction_options"].max_image_size == 1200
        assert kwargs["extraction_options"].sift.max_num_features == 8192
        assert kwargs["extraction_options"].use_gpu is False
        assert kwargs["device"] == sparse_module.pycolmap.Device.cpu
        with sqlite3.connect(kwargs["database_path"]) as connection:
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

    monkeypatch.setattr(sparse_module.pycolmap, "extract_features", fake_extract_features)

    metrics = extract_sparse_features(
        image_dir, database_path, replace(SparseRunConfig(), expected_images=2)
    )

    assert metrics == DatabaseMetrics(2, 30, 0, 0)


def test_map_sparse_database_writes_models_in_numeric_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    database_path = tmp_path / "features.db"
    database_path.write_bytes(b"database")
    output_dir = tmp_path / "sparse"

    class FakeReconstruction:
        def write(self, path: Path) -> None:
            path.mkdir(parents=True)

    def fake_incremental_mapping(**kwargs):
        assert kwargs["database_path"] == database_path
        assert kwargs["image_path"] == image_dir
        assert kwargs["output_path"] == output_dir
        assert kwargs["options"].min_num_matches == 15
        return {1: FakeReconstruction(), 0: FakeReconstruction()}

    def fake_summary(path: Path, total_images: int) -> ModelMetrics:
        return _model(registered_images=10 + int(path.name))

    monkeypatch.setattr(
        sparse_module.pycolmap, "incremental_mapping", fake_incremental_mapping
    )
    monkeypatch.setattr(sparse_module, "summarize_reconstruction", fake_summary)

    models = map_sparse_database(
        database_path, image_dir, output_dir, SparseRunConfig()
    )

    assert [model.registered_images for model in models] == [10, 11]


def test_map_sparse_database_rejects_empty_mapping_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    database_path = tmp_path / "features.db"
    database_path.write_bytes(b"database")
    monkeypatch.setattr(
        sparse_module.pycolmap, "incremental_mapping", lambda **kwargs: {}
    )

    with pytest.raises(RuntimeError, match="produced no sparse model"):
        map_sparse_database(
            database_path, image_dir, tmp_path / "sparse", SparseRunConfig()
        )


def test_sparse_config_rejects_invalid_overlap():
    config = SparseRunConfig()
    assert config.baseline_overlap == 20
    assert config.retry_overlap == 40
    with pytest.raises(ValueError):
        replace(config, baseline_overlap=0).validate()
