from __future__ import annotations

import json
from pathlib import Path

from run_sparse_reconstruction import (
    build_step10_summary,
    copy_best_model,
    execute_attempt_sequence,
)
from sparse_reconstruction import AttemptMetrics, DatabaseMetrics, ModelMetrics, SparseRunConfig


def _model(path: Path, registered: int) -> ModelMetrics:
    return ModelMetrics(
        model_path=path,
        registered_images=registered,
        total_images=288,
        sparse_points=3000,
        observations=9000,
        mean_track_length=3.0,
        mean_reprojection_error=0.7,
        camera_count=1,
        camera_model="SIMPLE_RADIAL",
        camera_params=(3070.0, 1536.0, 2040.0, 0.0),
    )


def _attempt(name: str, workspace: Path, overlap: int, registered: int) -> AttemptMetrics:
    model = _model(workspace / "0", registered)
    return AttemptMetrics(
        name=name,
        workspace=workspace,
        overlap=overlap,
        database=DatabaseMetrics(288, 100000, 5000, 4500),
        models=(model,),
        best_model=model,
        runtime_seconds=1.0,
        pycolmap_version="4.2.0",
    )


def test_attempt_sequence_skips_retry_when_baseline_passes(tmp_path: Path):
    calls: list[int] = []

    def runner(image_dir: Path, workspace: Path, *, overlap: int, config: SparseRunConfig):
        calls.append(overlap)
        return _attempt("baseline", workspace, overlap, 280)

    attempts = execute_attempt_sequence(
        image_dir=tmp_path / "images",
        reconstruction_root=tmp_path / "reconstruction",
        runner=runner,
        config=SparseRunConfig(),
    )

    assert calls == [20]
    assert len(attempts) == 1


def test_attempt_sequence_runs_overlap40_only_after_failed_baseline(tmp_path: Path):
    calls: list[int] = []

    def runner(image_dir: Path, workspace: Path, *, overlap: int, config: SparseRunConfig):
        calls.append(overlap)
        registered = 270 if overlap == 20 else 282
        name = "baseline" if overlap == 20 else "retry_overlap40"
        return _attempt(name, workspace, overlap, registered)

    attempts = execute_attempt_sequence(
        image_dir=tmp_path / "images",
        reconstruction_root=tmp_path / "reconstruction",
        runner=runner,
        config=SparseRunConfig(),
    )

    assert calls == [20, 40]
    assert len(attempts) == 2


def test_copy_best_model_copies_only_sparse_files(tmp_path: Path):
    source = tmp_path / "source"
    destination = tmp_path / "best"
    source.mkdir()
    for name in ("cameras.bin", "images.bin", "points3D.bin", "rigs.bin", "frames.bin"):
        (source / name).write_bytes(name.encode("ascii"))
    (source / "database.db").write_bytes(b"database")
    (source / "notes.txt").write_text("not model", encoding="utf-8")

    copied = copy_best_model(source, destination)

    assert {path.name for path in copied} == {
        "cameras.bin",
        "images.bin",
        "points3D.bin",
        "rigs.bin",
        "frames.bin",
    }
    assert not (destination / "database.db").exists()
    assert not (destination / "notes.txt").exists()


def test_step10_summary_explicitly_stops_before_dense(tmp_path: Path):
    attempt = _attempt("baseline", tmp_path / "baseline", 20, 280)
    summary = build_step10_summary(
        attempts=(attempt,),
        selected=attempt,
        pycolmap_version="4.2.0",
        selection_manifest_sha256="abc",
        initial_camera_params=(3070.0, 1536.0, 2040.0, 0.0),
        project_root=tmp_path,
    )

    assert summary["dense_reconstruction_started"] is False
    assert summary["selected_attempt"] == "baseline"
    assert summary["best_model"]["registered_images"] == 280
    json.dumps(summary, allow_nan=False)
