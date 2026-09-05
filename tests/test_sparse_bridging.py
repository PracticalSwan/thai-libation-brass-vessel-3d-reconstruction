from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sqlite3

import pytest
import pycolmap

from analysis_common import SelectedImageRecord
import sparse_bridging as bridging_module
from sparse_bridging import (
    BridgeCandidate,
    BridgePairMetrics,
    BridgeSearchConfig,
    TargetedGateResult,
    boundary_bridge_summary,
    bridge_model_accepted,
    choose_bridge_attempt,
    generate_candidate_pairs,
    qualified_bridge,
    run_bridge_diagnostics,
    run_exhaustive_attempt,
    run_targeted_attempt,
    select_bridge_pairs,
    summarize_bridge_pairs,
    targeted_gate,
    write_pair_list,
)
from sparse_reconstruction import AttemptMetrics, DatabaseMetrics, ModelMetrics, SparseRunConfig


def _records(count: int = 288) -> tuple[SelectedImageRecord, ...]:
    return tuple(
        SelectedImageRecord(
            index=index,
            filename=f"image{index:03d}.jpg",
            variant="PREPROCESSED",
            width=3072,
            height=4080,
            size_bytes=1,
            sha256="0" * 64,
            decision="ACCEPT",
            reasons="",
        )
        for index in range(1, count + 1)
    )


def _metrics(
    candidate: BridgeCandidate,
    *,
    raw_matches: int = 100,
    verified_inliers: int = 25,
    inlier_ratio: float = 0.25,
) -> BridgePairMetrics:
    return BridgePairMetrics(
        candidate=candidate,
        raw_matches=raw_matches,
        verified_inliers=verified_inliers,
        inlier_ratio=inlier_ratio,
        qualified=verified_inliers >= 15 and inlier_ratio >= 0.15,
    )


def test_candidate_generation_excludes_pairs_already_inside_overlap40():
    pairs = generate_candidate_pairs(_records(), BridgeSearchConfig())

    assert pairs
    assert all(pair.sequence_gap >= 41 for pair in pairs)


def test_candidate_generation_is_bounded_and_deterministic():
    pairs = generate_candidate_pairs(_records(), BridgeSearchConfig())

    assert len(pairs) == 2340
    assert pairs == tuple(
        sorted(
            pairs,
            key=lambda pair: (
                pair.boundary_left,
                pair.left_index,
                pair.right_index,
                pair.left_filename,
                pair.right_filename,
            ),
        )
    )


def test_every_candidate_stays_inside_its_boundary_windows():
    records = _records()
    pairs = generate_candidate_pairs(records, BridgeSearchConfig())
    names = {record.filename for record in records}

    for pair in pairs:
        assert pair.boundary_left - 39 <= pair.left_index <= pair.boundary_left
        assert pair.boundary_right <= pair.right_index <= pair.boundary_right + 39
        assert pair.left_index <= pair.boundary_left < pair.boundary_right <= pair.right_index
        assert pair.left_filename in names
        assert pair.right_filename in names


def test_pair_list_is_deterministic_and_accepts_metrics(tmp_path: Path):
    first = BridgeCandidate(73, 74, 34, 75, "image034.jpg", "image075.jpg")
    second = BridgeCandidate(73, 74, 35, 76, "image035.jpg", "image076.jpg")
    path = tmp_path / "pairs.txt"

    write_pair_list(path, (_metrics(second), first))

    assert path.read_text(encoding="utf-8") == (
        "image034.jpg image075.jpg\n"
        "image035.jpg image076.jpg\n"
    )


def test_pair_list_rejects_duplicates_and_whitespace(tmp_path: Path):
    pair = BridgeCandidate(73, 74, 34, 75, "image034.jpg", "image075.jpg")
    with pytest.raises(ValueError, match="duplicate"):
        write_pair_list(tmp_path / "duplicates.txt", (pair, pair))

    invalid = replace(pair, left_filename="image 034.jpg")
    with pytest.raises(ValueError, match="whitespace"):
        write_pair_list(tmp_path / "whitespace.txt", (invalid,))


def test_qualified_bridge_uses_frozen_inlier_and_ratio_thresholds():
    candidate = BridgeCandidate(73, 74, 34, 75, "image034.jpg", "image075.jpg")
    config = BridgeSearchConfig()

    assert qualified_bridge(_metrics(candidate), config)
    assert not qualified_bridge(_metrics(candidate, verified_inliers=14), config)
    assert not qualified_bridge(_metrics(candidate, inlier_ratio=0.149), config)


def test_selection_limits_endpoint_reuse_and_preserves_rank_order():
    boundary = (73, 74)
    candidates = (
        BridgeCandidate(*boundary, 34, 80, "shared.jpg", "right-a.jpg"),
        BridgeCandidate(*boundary, 34, 81, "shared.jpg", "right-b.jpg"),
        BridgeCandidate(*boundary, 34, 82, "shared.jpg", "right-c.jpg"),
        BridgeCandidate(*boundary, 35, 83, "left-d.jpg", "right-d.jpg"),
    )
    metrics = tuple(
        _metrics(candidate, verified_inliers=50 - index, inlier_ratio=0.5 - index * 0.01)
        for index, candidate in enumerate(candidates)
    )

    selected = select_bridge_pairs(metrics, BridgeSearchConfig())

    assert [pair.candidate.right_filename for pair in selected] == [
        "right-a.jpg",
        "right-b.jpg",
        "right-d.jpg",
    ]
    assert sum(pair.candidate.left_filename == "shared.jpg" for pair in selected) == 2


def _pair_database(path: Path, *, raw_matches: int | None, inliers: int | None) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE matches(pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
            CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
            """
        )
        connection.executemany(
            "INSERT INTO images(image_id, name) VALUES (?, ?)",
            [(1, "image034.jpg"), (2, "image075.jpg")],
        )
        pair_id = pycolmap.image_pair_to_pair_id(1, 2)
        if raw_matches is not None:
            connection.execute(
                "INSERT INTO matches(pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                (pair_id, raw_matches, 2, b"m"),
            )
        if inliers is not None:
            connection.execute(
                "INSERT INTO two_view_geometries(pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                (pair_id, inliers, 2, b"g"),
            )


def test_pair_metrics_use_colmap_pair_id_and_database_rows(tmp_path: Path):
    database_path = tmp_path / "pairs.db"
    _pair_database(database_path, raw_matches=100, inliers=25)
    candidate = BridgeCandidate(73, 74, 34, 75, "image034.jpg", "image075.jpg")

    metrics = summarize_bridge_pairs(
        database_path, (candidate,), BridgeSearchConfig()
    )

    assert metrics == (
        BridgePairMetrics(candidate, 100, 25, 0.25, True),
    )


@pytest.mark.parametrize(
    ("raw_matches", "inliers"),
    ((None, None), (0, 0)),
)
def test_pair_metrics_treat_missing_or_zero_rows_as_unqualified(
    tmp_path: Path, raw_matches: int | None, inliers: int | None
):
    database_path = tmp_path / "pairs.db"
    _pair_database(database_path, raw_matches=raw_matches, inliers=inliers)
    candidate = BridgeCandidate(73, 74, 34, 75, "image034.jpg", "image075.jpg")

    metrics = summarize_bridge_pairs(
        database_path, (candidate,), BridgeSearchConfig()
    )

    assert metrics == (
        BridgePairMetrics(candidate, 0, 0, 0.0, False),
    )


def test_diagnostics_copy_features_match_once_and_return_measured_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    features_database = tmp_path / "features.db"
    features_database.write_bytes(b"feature-cache")
    workspace = tmp_path / "work"
    candidate = BridgeCandidate(73, 74, 34, 75, "image034.jpg", "image075.jpg")
    expected = (_metrics(candidate),)
    calls: list[Path] = []

    def fake_match_image_pairs(**kwargs):
        calls.append(Path(kwargs["database_path"]))
        assert kwargs["matching_options"].use_gpu is False
        assert Path(str(kwargs["pairing_options"].match_list_path)) == (
            workspace / "diagnostic_pairs.txt"
        )
        assert kwargs["device"] == pycolmap.Device.cpu

    def fake_summary(database_path, candidates, config):
        assert database_path == workspace / "diagnostic.db"
        assert candidates == (candidate,)
        return expected

    monkeypatch.setattr(
        bridging_module.pycolmap, "match_image_pairs", fake_match_image_pairs
    )
    monkeypatch.setattr(bridging_module, "summarize_bridge_pairs", fake_summary)

    result = run_bridge_diagnostics(
        features_database, workspace, (candidate,), BridgeSearchConfig()
    )

    assert result == expected
    assert calls == [workspace / "diagnostic.db"]
    assert (workspace / "diagnostic.db").read_bytes() == b"feature-cache"
    assert (workspace / "diagnostic_pairs.txt").read_text(encoding="utf-8") == (
        "image034.jpg image075.jpg\n"
    )


def test_boundary_summary_reports_measured_and_selected_counts():
    config = BridgeSearchConfig()
    first = _metrics(
        BridgeCandidate(73, 74, 34, 75, "image034.jpg", "image075.jpg")
    )
    second = _metrics(
        BridgeCandidate(73, 74, 35, 76, "image035.jpg", "image076.jpg"),
        raw_matches=10,
        verified_inliers=1,
        inlier_ratio=0.1,
    )

    summary = boundary_bridge_summary((first, second), (first,), config)

    assert summary[0] == {
        "boundary": "73-74",
        "boundary_left": 73,
        "boundary_right": 74,
        "candidate_count": 2,
        "nonzero_match_count": 2,
        "qualified_count": 1,
        "selected_count": 1,
        "maximum_verified_inliers": 25,
        "maximum_inlier_ratio": 0.25,
    }
    assert summary[1]["boundary"] == "145-146"
    assert summary[1]["candidate_count"] == 0
    assert summary[2]["boundary"] == "203-204"


def test_targeted_gate_requires_a_selected_pair_for_every_boundary():
    first = _metrics(
        BridgeCandidate(73, 74, 34, 75, "image034.jpg", "image075.jpg")
    )
    second = _metrics(
        BridgeCandidate(145, 146, 106, 147, "image106.jpg", "image147.jpg")
    )

    result = targeted_gate((first, second), BridgeSearchConfig())

    assert result == TargetedGateResult(
        allowed=False,
        reason="no selected qualified bridge for boundary 203-204",
    )


def test_targeted_gate_allows_one_or_more_pairs_for_all_boundaries():
    selected = tuple(
        _metrics(
            BridgeCandidate(
                left,
                right,
                left - 39,
                right + 1,
                f"image{left - 39:03d}.jpg",
                f"image{right + 1:03d}.jpg",
            )
        )
        for left, right in BridgeSearchConfig().boundaries
    )

    assert targeted_gate(selected, BridgeSearchConfig()).allowed is True


def test_targeted_attempt_runs_sequential_then_selected_imported_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    features_database = tmp_path / "features.db"
    features_database.write_bytes(b"features")
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "targeted"
    selected = tuple(
        _metrics(
            BridgeCandidate(
                left,
                right,
                left - 39,
                right + 1,
                f"image{left - 39:03d}.jpg",
                f"image{right + 1:03d}.jpg",
            )
        )
        for left, right in BridgeSearchConfig().boundaries
    )
    model = ModelMetrics(
        model_path=output_dir / "0",
        registered_images=200,
        total_images=288,
        sparse_points=5000,
        observations=15000,
        mean_track_length=3.0,
        mean_reprojection_error=1.0,
        camera_count=1,
        camera_model="SIMPLE_RADIAL",
        camera_params=(3070.0, 1536.0, 2040.0, 0.0),
    )
    calls: list[str] = []

    def fake_sequential(**kwargs):
        calls.append("sequential")
        assert kwargs["pairing_options"].overlap == 20
        assert kwargs["pairing_options"].quadratic_overlap is True
        assert kwargs["pairing_options"].loop_detection is False
        assert kwargs["device"] == pycolmap.Device.cpu

    def fake_imported(**kwargs):
        calls.append("imported")
        assert kwargs["matching_options"].use_gpu is False
        assert Path(str(kwargs["pairing_options"].match_list_path)) == (
            tmp_path / "work" / "targeted_bridge_pairs.txt"
        )

    def fake_map(database_path, actual_image_dir, actual_output_dir, config):
        calls.append("mapping")
        assert database_path == tmp_path / "work" / "targeted.db"
        assert actual_image_dir == image_dir
        assert actual_output_dir == output_dir
        return (model,)

    monkeypatch.setattr(bridging_module.pycolmap, "match_sequential", fake_sequential)
    monkeypatch.setattr(bridging_module.pycolmap, "match_image_pairs", fake_imported)
    monkeypatch.setattr(bridging_module, "map_sparse_database", fake_map)
    monkeypatch.setattr(
        bridging_module,
        "summarize_database",
        lambda path: DatabaseMetrics(288, 1000, 500, 400),
    )

    attempt = run_targeted_attempt(
        image_dir,
        features_database,
        output_dir,
        selected,
        SparseRunConfig(),
        BridgeSearchConfig(),
    )

    assert calls == ["sequential", "imported", "mapping"]
    assert attempt.name == "targeted_bridges"
    assert attempt.best_model == model
    assert attempt.models == (model,)
    assert (tmp_path / "work" / "targeted.db").read_bytes() == b"features"


def _model_metrics(
    path: Path,
    *,
    registered_images: int = 274,
    sparse_points: int = 1000,
    camera_count: int = 1,
    camera_model: str = "SIMPLE_RADIAL",
    mean_reprojection_error: float = 1.0,
) -> ModelMetrics:
    return ModelMetrics(
        model_path=path,
        registered_images=registered_images,
        total_images=288,
        sparse_points=sparse_points,
        observations=3000,
        mean_track_length=3.0,
        mean_reprojection_error=mean_reprojection_error,
        camera_count=camera_count,
        camera_model=camera_model,
        camera_params=(3070.0, 1536.0, 2040.0, 0.0),
    )


def _attempt_metrics(name: str, model: ModelMetrics) -> AttemptMetrics:
    return AttemptMetrics(
        name=name,
        workspace=Path(name),
        overlap=20 if name == "targeted_bridges" else 0,
        database=DatabaseMetrics(288, 1000, 500, 400),
        models=(model,),
        best_model=model,
    )


def test_bridge_acceptance_requires_every_frozen_model_condition():
    accepted = _model_metrics(Path("accepted"))
    config = BridgeSearchConfig()

    assert bridge_model_accepted(accepted, config)
    assert not bridge_model_accepted(replace(accepted, registered_images=273), config)
    assert not bridge_model_accepted(replace(accepted, sparse_points=999), config)
    assert not bridge_model_accepted(replace(accepted, camera_count=2), config)
    assert not bridge_model_accepted(replace(accepted, camera_model="OPENCV"), config)
    assert not bridge_model_accepted(
        replace(accepted, mean_reprojection_error=float("nan")), config
    )


def test_bridge_attempt_ranking_prefers_acceptance_then_metrics():
    accepted = _attempt_metrics(
        "targeted_bridges", _model_metrics(Path("accepted"), registered_images=274)
    )
    nonaccepted = _attempt_metrics(
        "exhaustive",
        _model_metrics(Path("wrong-camera"), registered_images=288, camera_model="OPENCV"),
    )
    assert choose_bridge_attempt((nonaccepted, accepted), BridgeSearchConfig()) == accepted

    more_registered = _attempt_metrics(
        "exhaustive", _model_metrics(Path("more-registered"), registered_images=280)
    )
    assert choose_bridge_attempt((accepted, more_registered), BridgeSearchConfig()) == more_registered

    more_points = _attempt_metrics(
        "targeted_bridges",
        _model_metrics(Path("more-points"), registered_images=280, sparse_points=2000),
    )
    assert choose_bridge_attempt((more_registered, more_points), BridgeSearchConfig()) == more_points

    lower_error = _attempt_metrics(
        "exhaustive",
        _model_metrics(
            Path("lower-error"),
            registered_images=280,
            sparse_points=2000,
            mean_reprojection_error=0.5,
        ),
    )
    assert choose_bridge_attempt((more_points, lower_error), BridgeSearchConfig()) == lower_error


def test_exhaustive_attempt_uses_single_cpu_block50_match_then_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    features_database = tmp_path / "features.db"
    features_database.write_bytes(b"features")
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "exhaustive"
    model = _model_metrics(output_dir / "0", registered_images=250, sparse_points=5000)
    calls: list[str] = []

    def fake_exhaustive(**kwargs):
        calls.append("matching")
        assert kwargs["matching_options"].use_gpu is False
        assert kwargs["pairing_options"].block_size == 50
        assert kwargs["device"] == pycolmap.Device.cpu

    def fake_map(database_path, actual_image_dir, actual_output_dir, config):
        calls.append("mapping")
        assert database_path == tmp_path / "work" / "exhaustive.db"
        assert actual_image_dir == image_dir
        assert actual_output_dir == output_dir
        return (model,)

    monkeypatch.setattr(bridging_module.pycolmap, "match_exhaustive", fake_exhaustive)
    monkeypatch.setattr(bridging_module, "map_sparse_database", fake_map)
    monkeypatch.setattr(
        bridging_module,
        "summarize_database",
        lambda path: DatabaseMetrics(288, 1000, 500, 400),
    )
    monkeypatch.setattr(
        bridging_module,
        "_feature_database_layout",
        lambda path: ((1, "image001.jpg", 1, 10, 4, 10, 128),),
    )

    attempt = run_exhaustive_attempt(
        image_dir,
        features_database,
        output_dir,
        SparseRunConfig(),
        BridgeSearchConfig(),
    )

    assert calls == ["matching", "mapping"]
    assert attempt.name == "exhaustive"
    assert attempt.overlap == 0
    assert attempt.best_model == model
    assert (tmp_path / "work" / "exhaustive.db").read_bytes() == b"features"


def test_exhaustive_attempt_resumes_valid_partial_database_without_recopying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    features_database = tmp_path / "features.db"
    features_database.write_bytes(b"features")
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "exhaustive"
    partial_database = tmp_path / "work" / "exhaustive.db"
    partial_database.parent.mkdir()
    partial_database.write_bytes(b"partial-progress")
    model = _model_metrics(output_dir / "0", registered_images=250, sparse_points=5000)

    def fake_exhaustive(**kwargs):
        assert Path(kwargs["database_path"]).read_bytes() == b"partial-progress"

    monkeypatch.setattr(bridging_module.pycolmap, "match_exhaustive", fake_exhaustive)
    monkeypatch.setattr(
        bridging_module, "map_sparse_database", lambda *args, **kwargs: (model,)
    )
    monkeypatch.setattr(
        bridging_module,
        "summarize_database",
        lambda path: DatabaseMetrics(288, 1000, 500, 400),
    )
    monkeypatch.setattr(
        bridging_module,
        "_feature_database_layout",
        lambda path: ((1, "image001.jpg", 1, 10, 4, 10, 128),),
    )

    attempt = run_exhaustive_attempt(
        image_dir,
        features_database,
        output_dir,
        SparseRunConfig(),
        BridgeSearchConfig(),
    )

    assert attempt.best_model == model
    assert partial_database.read_bytes() == b"partial-progress"


def test_exhaustive_attempt_rejects_partial_database_with_different_feature_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    features_database = tmp_path / "features.db"
    features_database.write_bytes(b"features")
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "exhaustive"
    partial_database = tmp_path / "work" / "exhaustive.db"
    partial_database.parent.mkdir()
    partial_database.write_bytes(b"partial-progress")

    monkeypatch.setattr(
        bridging_module,
        "summarize_database",
        lambda path: DatabaseMetrics(288, 1000, 500, 400),
    )

    def layout(path: Path):
        name = "different.jpg" if path == partial_database else "image001.jpg"
        return ((1, name, 1, 10, 4, 10, 128),)

    monkeypatch.setattr(bridging_module, "_feature_database_layout", layout)

    with pytest.raises(ValueError, match="does not match the verified feature cache"):
        run_exhaustive_attempt(
            image_dir,
            features_database,
            output_dir,
            SparseRunConfig(),
            BridgeSearchConfig(),
        )
