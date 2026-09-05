from __future__ import annotations

import json
import csv
from dataclasses import replace
from pathlib import Path
import sqlite3

import pytest
import pycolmap

from analysis_common import SelectedImageRecord, VerifiedSelectedSet
import run_sparse_bridging as runner_module
import sparse_bridging as bridging_module
from sparse_bridging import BridgeCandidate, BridgePairMetrics, prepare_feature_cache
from sparse_reconstruction import (
    AttemptMetrics,
    DatabaseMetrics,
    ModelMetrics,
    SparseRunConfig,
    summarize_database,
)
from run_sparse_bridging import build_step11_summary, cleanup_transient_work


def _write_feature_database(path: Path, *, image_count: int = 288) -> DatabaseMetrics:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER);
            CREATE TABLE keypoints(image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB);
            CREATE TABLE descriptors(image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB);
            CREATE TABLE matches(pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
            CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
            """
        )
        connection.executemany(
            "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)",
            [(index, f"image{index:03d}.jpg", 1) for index in range(1, image_count + 1)],
        )
        connection.executemany(
            "INSERT INTO keypoints(image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
            [(index, 10, 4, b"x") for index in range(1, image_count + 1)],
        )
        connection.executemany(
            "INSERT INTO descriptors(image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
            [(index, 10, 128, b"y") for index in range(1, image_count + 1)],
        )
        connection.commit()
    finally:
        connection.close()
    return summarize_database(path)


def _verified(image_dir: Path) -> VerifiedSelectedSet:
    records = tuple(
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
        for index in range(1, 289)
    )
    return VerifiedSelectedSet(
        records=records, manifest_sha256="manifest-sha", images_dir=image_dir
    )


def test_feature_cache_marker_is_written_only_after_successful_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    manifest = tmp_path / "selection_manifest.csv"
    manifest.write_text("fixture", encoding="utf-8")
    work_dir = tmp_path / "work"
    monkeypatch.setattr(
        bridging_module, "verify_selected_images", lambda *args, **kwargs: _verified(image_dir)
    )

    def failing_extractor(*args, **kwargs):
        raise RuntimeError("extraction failed")

    with pytest.raises(RuntimeError, match="extraction failed"):
        prepare_feature_cache(
            image_dir,
            manifest,
            work_dir,
            SparseRunConfig(),
            feature_extractor=failing_extractor,
        )

    assert not (work_dir / "features_complete.json").exists()


def test_feature_cache_rebuilds_partial_database_and_records_verified_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    manifest = tmp_path / "selection_manifest.csv"
    manifest.write_text("fixture", encoding="utf-8")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    database_path = work_dir / "features.db"
    database_path.write_bytes(b"partial")
    monkeypatch.setattr(
        bridging_module, "verify_selected_images", lambda *args, **kwargs: _verified(image_dir)
    )

    def extractor(actual_image_dir, actual_database_path, config):
        assert actual_image_dir == image_dir
        assert actual_database_path == database_path
        assert not actual_database_path.exists()
        return _write_feature_database(actual_database_path)

    result = prepare_feature_cache(
        image_dir,
        manifest,
        work_dir,
        SparseRunConfig(),
        feature_extractor=extractor,
    )

    marker = json.loads((work_dir / "features_complete.json").read_text(encoding="utf-8"))
    assert result == database_path
    assert marker == {
        "camera_model": "SIMPLE_RADIAL",
        "feature_count": 2880,
        "image_count": 288,
        "max_image_size": 1200,
        "max_num_features": 8192,
        "pycolmap_version": str(pycolmap.__version__),
        "selection_manifest_sha256": "manifest-sha",
    }


def test_feature_cache_reuses_only_database_with_valid_completion_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    manifest = tmp_path / "selection_manifest.csv"
    manifest.write_text("fixture", encoding="utf-8")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    database_path = work_dir / "features.db"
    _write_feature_database(database_path)
    marker = {
        "camera_model": "SIMPLE_RADIAL",
        "feature_count": 2880,
        "image_count": 288,
        "max_image_size": 1200,
        "max_num_features": 8192,
        "pycolmap_version": str(pycolmap.__version__),
        "selection_manifest_sha256": "manifest-sha",
    }
    (work_dir / "features_complete.json").write_text(
        json.dumps(marker), encoding="utf-8"
    )
    monkeypatch.setattr(
        bridging_module, "verify_selected_images", lambda *args, **kwargs: _verified(image_dir)
    )

    def unexpected_extractor(*args, **kwargs):
        raise AssertionError("valid feature cache must be reused")

    result = prepare_feature_cache(
        image_dir,
        manifest,
        work_dir,
        SparseRunConfig(),
        feature_extractor=unexpected_extractor,
    )

    assert result == database_path


def test_feature_cache_rebuilds_when_per_image_feature_layout_is_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    manifest = tmp_path / "selection_manifest.csv"
    manifest.write_text("fixture", encoding="utf-8")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    database_path = work_dir / "features.db"
    _write_feature_database(database_path)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("DELETE FROM descriptors WHERE image_id = 288")
        connection.commit()
    finally:
        connection.close()
    marker = {
        "camera_model": "SIMPLE_RADIAL",
        "feature_count": 2880,
        "image_count": 288,
        "max_image_size": 1200,
        "max_num_features": 8192,
        "pycolmap_version": str(pycolmap.__version__),
        "selection_manifest_sha256": "manifest-sha",
    }
    (work_dir / "features_complete.json").write_text(
        json.dumps(marker), encoding="utf-8"
    )
    monkeypatch.setattr(
        bridging_module, "verify_selected_images", lambda *args, **kwargs: _verified(image_dir)
    )
    calls = 0

    def extractor(actual_image_dir, actual_database_path, config):
        nonlocal calls
        calls += 1
        assert actual_image_dir == image_dir
        assert not actual_database_path.exists()
        return _write_feature_database(actual_database_path)

    result = prepare_feature_cache(
        image_dir,
        manifest,
        work_dir,
        SparseRunConfig(),
        feature_extractor=extractor,
    )

    assert result == database_path
    assert calls == 1


def _configure_runner_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path]:
    paths = {
        "SELECTED_DIR": tmp_path / "images",
        "SELECTION_MANIFEST": tmp_path / "selection_manifest.csv",
        "BRIDGING_ROOT": tmp_path / "bridging",
        "WORK_DIR": tmp_path / "bridging" / "work",
        "TARGETED_DIR": tmp_path / "bridging" / "targeted",
        "EXHAUSTIVE_DIR": tmp_path / "bridging" / "exhaustive",
        "BEST_DIR": tmp_path / "bridging" / "best",
        "REPORTS_DIR": tmp_path / "bridging" / "reports",
        "PREVIEWS_DIR": tmp_path / "bridging" / "previews",
        "CANDIDATES_CSV": tmp_path / "bridging" / "reports" / "step11_candidates.csv",
        "BOUNDARY_SUMMARY_JSON": tmp_path / "bridging" / "reports" / "step11_boundary_summary.json",
        "TARGETED_REPORT": tmp_path / "bridging" / "reports" / "step11_targeted.json",
        "EXHAUSTIVE_REPORT": tmp_path / "bridging" / "reports" / "step11_exhaustive.json",
        "ATTEMPTS_CSV": tmp_path / "bridging" / "reports" / "step11_attempts.csv",
        "REGISTERED_CSV": tmp_path / "bridging" / "reports" / "step11_registered_images.csv",
        "SUMMARY_JSON": tmp_path / "bridging" / "reports" / "step11_summary.json",
        "CANDIDATE_FIGURE": tmp_path / "bridging" / "previews" / "step11_01_bridge_candidates.png",
        "SPARSE_FIGURE": tmp_path / "bridging" / "previews" / "step11_02_sparse_model.png",
        "REGISTRATION_FIGURE": tmp_path / "bridging" / "previews" / "step11_03_registration.png",
        "PLY_PATH": tmp_path / "bridging" / "best" / "points3D.ply",
        "STEP10_SUMMARY": tmp_path / "step10_summary.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(runner_module, name, path)
    paths["SELECTED_DIR"].mkdir()
    paths["SELECTION_MANIFEST"].write_text("fixture", encoding="utf-8")
    paths["STEP10_SUMMARY"].write_text(
        json.dumps(
            {
                "best_model": {"registered_images": 73, "sparse_points": 6099},
                "selected_attempt": "baseline",
            }
        ),
        encoding="utf-8",
    )
    return paths


def _bridge_metric(left: int, right: int) -> BridgePairMetrics:
    candidate = BridgeCandidate(
        boundary_left=left,
        boundary_right=right,
        left_index=left - 39,
        right_index=right + 1,
        left_filename=f"image{left - 39:03d}.jpg",
        right_filename=f"image{right + 1:03d}.jpg",
    )
    return BridgePairMetrics(
        candidate=candidate,
        raw_matches=100,
        verified_inliers=25,
        inlier_ratio=0.25,
        qualified=True,
    )


def _attempt(path: Path) -> AttemptMetrics:
    model = ModelMetrics(
        model_path=path / "0",
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
    return AttemptMetrics(
        name="targeted_bridges",
        workspace=path,
        overlap=20,
        database=DatabaseMetrics(288, 1000, 500, 400),
        models=(model,),
        best_model=model,
        runtime_seconds=1.0,
        pycolmap_version="4.2.0",
    )


def _write_candidate_rows(path: Path, metrics: tuple[BridgePairMetrics, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for metric in metrics:
        candidate = metric.candidate
        rows.append(
            {
                "boundary": f"{candidate.boundary_left}-{candidate.boundary_right}",
                "boundary_left": candidate.boundary_left,
                "boundary_right": candidate.boundary_right,
                "left_index": candidate.left_index,
                "right_index": candidate.right_index,
                "left_filename": candidate.left_filename,
                "right_filename": candidate.right_filename,
                "sequence_gap": candidate.sequence_gap,
                "raw_matches": metric.raw_matches,
                "verified_inliers": metric.verified_inliers,
                "inlier_ratio": metric.inlier_ratio,
                "qualified": 1,
                "selected": 1,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_diagnose_stage_writes_reports_without_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _configure_runner_paths(tmp_path, monkeypatch)
    verified = _verified(paths["SELECTED_DIR"])
    candidates = tuple(metric.candidate for metric in (
        _bridge_metric(73, 74),
        _bridge_metric(145, 146),
        _bridge_metric(203, 204),
    ))
    metrics = tuple(
        BridgePairMetrics(candidate, 100, 25, 0.25, True)
        for candidate in candidates
    )
    features = paths["WORK_DIR"] / "features.db"
    features.parent.mkdir(parents=True)
    features.write_bytes(b"features")
    monkeypatch.setattr(
        runner_module, "verify_selected_images", lambda *args, **kwargs: verified
    )
    monkeypatch.setattr(
        runner_module, "prepare_feature_cache", lambda *args, **kwargs: features
    )
    monkeypatch.setattr(
        runner_module, "generate_candidate_pairs", lambda *args, **kwargs: candidates
    )
    monkeypatch.setattr(
        runner_module, "run_bridge_diagnostics", lambda *args, **kwargs: metrics
    )
    monkeypatch.setattr(
        runner_module, "select_bridge_pairs", lambda *args, **kwargs: metrics
    )
    monkeypatch.setattr(
        runner_module,
        "run_targeted_attempt",
        lambda *args, **kwargs: pytest.fail("diagnose must not map cameras"),
    )

    result = runner_module.run_stage("diagnose")

    assert result["candidate_count"] == 3
    assert result["selected_bridge_count"] == 3
    with paths["CANDIDATES_CSV"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert all(row["selected"] == "1" for row in rows)
    boundary = json.loads(paths["BOUNDARY_SUMMARY_JSON"].read_text(encoding="utf-8"))
    assert boundary["selection_manifest_sha256"] == "manifest-sha"
    assert len(boundary["boundaries"]) == 3
    assert not paths["TARGETED_REPORT"].exists()


def test_targeted_stage_records_skip_when_a_boundary_has_no_selected_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _configure_runner_paths(tmp_path, monkeypatch)
    _write_candidate_rows(
        paths["CANDIDATES_CSV"],
        (_bridge_metric(73, 74), _bridge_metric(145, 146)),
    )
    paths["BOUNDARY_SUMMARY_JSON"].write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        runner_module,
        "run_targeted_attempt",
        lambda *args, **kwargs: pytest.fail("targeted mapping must be skipped"),
    )

    result = runner_module.run_stage("targeted")

    assert result["status"] == "skipped"
    assert "203-204" in result["reason"]
    assert json.loads(paths["TARGETED_REPORT"].read_text(encoding="utf-8")) == result


def test_targeted_stage_runs_from_completed_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _configure_runner_paths(tmp_path, monkeypatch)
    selected = tuple(
        _bridge_metric(left, right)
        for left, right in ((73, 74), (145, 146), (203, 204))
    )
    _write_candidate_rows(paths["CANDIDATES_CSV"], selected)
    paths["BOUNDARY_SUMMARY_JSON"].write_text("{}", encoding="utf-8")
    verified = _verified(paths["SELECTED_DIR"])
    features = paths["WORK_DIR"] / "features.db"
    features.parent.mkdir(parents=True)
    features.write_bytes(b"features")
    attempt = _attempt(paths["TARGETED_DIR"])
    calls: list[str] = []
    monkeypatch.setattr(
        runner_module,
        "verify_selected_images",
        lambda *args, **kwargs: (calls.append("verify") or verified),
    )
    monkeypatch.setattr(
        runner_module,
        "prepare_feature_cache",
        lambda *args, **kwargs: (calls.append("features") or features),
    )
    monkeypatch.setattr(
        runner_module,
        "run_targeted_attempt",
        lambda *args, **kwargs: (calls.append("targeted") or attempt),
    )

    result = runner_module.run_stage("targeted")

    assert calls == ["verify", "features", "targeted"]
    assert result["status"] == "completed"
    assert result["attempt"]["name"] == "targeted_bridges"
    assert paths["TARGETED_REPORT"].is_file()


def _attempt_payload(*, registered: int, camera_model: str = "SIMPLE_RADIAL") -> dict:
    model = {
        "model_path": "bridging/targeted/0",
        "registered_images": registered,
        "total_images": 288,
        "sparse_points": 5000,
        "observations": 15000,
        "mean_track_length": 3.0,
        "mean_reprojection_error": 1.0,
        "camera_count": 1,
        "camera_model": camera_model,
        "camera_params": [3070.0, 1536.0, 2040.0, 0.0],
        "mean_observations_per_registered_image": 75.0,
        "registered_image_names": [],
    }
    return {
        "name": "targeted_bridges",
        "workspace": "bridging/targeted",
        "overlap": 20,
        "database": {
            "image_count": 288,
            "feature_count": 1000,
            "matched_pair_count": 500,
            "verified_pair_count": 400,
        },
        "models": [model],
        "best_model": model,
        "runtime_seconds": 1.0,
        "pycolmap_version": "4.2.0",
    }


def test_exhaustive_stage_skips_when_targeted_already_passes_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _configure_runner_paths(tmp_path, monkeypatch)
    paths["TARGETED_REPORT"].parent.mkdir(parents=True)
    paths["TARGETED_REPORT"].write_text(
        json.dumps(
            {
                "status": "completed",
                "reason": "",
                "attempt": _attempt_payload(registered=274),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner_module,
        "run_exhaustive_attempt",
        lambda *args, **kwargs: pytest.fail("accepted targeted result must skip fallback"),
    )

    result = runner_module.run_stage("exhaustive")

    assert result == {
        "status": "skipped",
        "reason": "targeted attempt already passes the Step 11 acceptance gate",
        "attempt": None,
    }
    assert json.loads(paths["EXHAUSTIVE_REPORT"].read_text(encoding="utf-8")) == result


def test_exhaustive_stage_runs_once_after_targeted_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    paths = _configure_runner_paths(tmp_path, monkeypatch)
    paths["TARGETED_REPORT"].parent.mkdir(parents=True)
    paths["TARGETED_REPORT"].write_text(
        json.dumps({"status": "skipped", "reason": "no bridges", "attempt": None}),
        encoding="utf-8",
    )
    verified = _verified(paths["SELECTED_DIR"])
    features = paths["WORK_DIR"] / "features.db"
    features.parent.mkdir(parents=True)
    features.write_bytes(b"features")
    exhaustive = _attempt(paths["EXHAUSTIVE_DIR"])
    exhaustive = AttemptMetrics(
        name="exhaustive",
        workspace=exhaustive.workspace,
        overlap=0,
        database=exhaustive.database,
        models=exhaustive.models,
        best_model=exhaustive.best_model,
        runtime_seconds=exhaustive.runtime_seconds,
        pycolmap_version=exhaustive.pycolmap_version,
    )
    calls: list[str] = []
    monkeypatch.setattr(
        runner_module,
        "verify_selected_images",
        lambda *args, **kwargs: (calls.append("verify") or verified),
    )
    monkeypatch.setattr(
        runner_module,
        "prepare_feature_cache",
        lambda *args, **kwargs: (calls.append("features") or features),
    )
    monkeypatch.setattr(
        runner_module,
        "run_exhaustive_attempt",
        lambda *args, **kwargs: (calls.append("exhaustive") or exhaustive),
    )

    result = runner_module.run_stage("exhaustive")

    assert calls == ["verify", "features", "exhaustive"]
    assert result["status"] == "completed"
    assert result["attempt"]["name"] == "exhaustive"
    assert "288 images, CPU SIFT matching, block_size=50" in capsys.readouterr().out


def test_step11_summary_keeps_failed_acceptance_false_and_records_required_evidence(
    tmp_path: Path,
):
    verified = _verified(tmp_path / "images")
    base_attempt = _attempt(tmp_path / "targeted")
    registered_names = tuple(record.filename for record in verified.records[:2])
    selected_model = replace(
        base_attempt.best_model,
        registered_images=2,
        registered_image_names=registered_names,
    )
    selected = replace(base_attempt, best_model=selected_model, models=(selected_model,))
    step10 = {
        "selected_attempt": "baseline",
        "best_model": {
            "registered_images": 73,
            "sparse_points": 6099,
            "mean_reprojection_error": 1.2373052447638215,
            "camera_count": 1,
            "camera_model": "SIMPLE_RADIAL",
        },
    }
    boundary = {
        "candidate_count": 2340,
        "selected_bridge_count": 3,
        "boundaries": [
            {"boundary": "73-74", "candidate_count": 780, "qualified_count": 1, "selected_count": 1},
            {"boundary": "145-146", "candidate_count": 780, "qualified_count": 1, "selected_count": 1},
            {"boundary": "203-204", "candidate_count": 780, "qualified_count": 1, "selected_count": 1},
        ],
    }
    targeted = {"status": "completed", "reason": "", "attempt": {"name": "targeted_bridges"}}
    exhaustive = {"status": "skipped", "reason": "not required", "attempt": None}

    summary = build_step11_summary(
        pycolmap_version="4.2.0",
        selection_manifest_sha256="manifest-sha",
        step10_summary=step10,
        boundary_summary=boundary,
        targeted_report=targeted,
        exhaustive_report=exhaustive,
        selected=selected,
        selected_records=verified.records,
    )

    assert summary["bridge_success"] is False
    assert summary["dense_reconstruction_started"] is False
    assert summary["step10_baseline"]["registered_images"] == 73
    assert summary["critical_boundaries"] == [[73, 74], [145, 146], [203, 204]]
    assert summary["candidate_counts"] == {
        "73-74": 780,
        "145-146": 780,
        "203-204": 780,
    }
    assert summary["selected_attempt"] == "targeted_bridges"
    assert summary["registered_indices"] == [1, 2]
    assert summary["unregistered_indices"][0] == 3
    assert summary["final_camera_params"] == [3070.0, 1536.0, 2040.0, 0.0]
    json.dumps(summary, allow_nan=False)


def test_cleanup_removes_only_enumerated_transient_work_files(tmp_path: Path):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    transient_names = (
        "features.db",
        "features_complete.json",
        "diagnostic.db",
        "diagnostic_pairs.txt",
        "targeted.db",
        "targeted_bridge_pairs.txt",
        "exhaustive.db",
    )
    for name in transient_names:
        (work_dir / name).write_text("temporary", encoding="utf-8")
    preserved = work_dir / "preserve.txt"
    preserved.write_text("keep", encoding="utf-8")

    removed = cleanup_transient_work(work_dir)

    assert {path.name for path in removed} == set(transient_names)
    assert preserved.read_text(encoding="utf-8") == "keep"
    assert work_dir.is_dir()


def test_all_stage_runs_bounded_sequence_once(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []
    monkeypatch.setattr(
        runner_module, "_diagnose_stage", lambda: (calls.append("diagnose") or {})
    )
    monkeypatch.setattr(
        runner_module, "_targeted_stage", lambda: (calls.append("targeted") or {})
    )
    monkeypatch.setattr(
        runner_module, "_exhaustive_stage", lambda: (calls.append("exhaustive") or {})
    )
    monkeypatch.setattr(
        runner_module, "_finalize_stage", lambda: (calls.append("finalize") or {"done": True})
    )

    result = runner_module.run_stage("all")

    assert calls == ["diagnose", "targeted", "exhaustive", "finalize"]
    assert result == {"done": True}
