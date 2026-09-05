from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from analysis_common import SelectedImageRecord, VerifiedSelectedSet
from external_learned_recovery import (
    ExternalLearnedConfig,
    FrontendBundle,
    ImageFeatures,
    PairMatchingMetrics,
)
import run_external_learned_recovery as runner
from sparse_bridging import BridgeCandidate, BridgePairMetrics, BridgeSearchConfig
from sparse_reconstruction import AttemptMetrics, DatabaseMetrics, ModelMetrics


def _records(count: int = 288) -> tuple[SelectedImageRecord, ...]:
    return tuple(
        SelectedImageRecord(
            index=index,
            filename=f"image{index:03d}.jpg",
            variant="PREPROCESSED",
            width=3072,
            height=4080,
            size_bytes=100,
            sha256=f"{index:064x}"[-64:],
            decision="ACCEPT",
            reasons="",
        )
        for index in range(1, count + 1)
    )


def _verified(path: Path, count: int = 288) -> VerifiedSelectedSet:
    return VerifiedSelectedSet(_records(count), "manifest-sha", path)


def _features() -> ImageFeatures:
    return ImageFeatures(
        keypoints=np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        descriptors=np.ones((2, 128), dtype=np.float32),
        scores=np.ones(2, dtype=np.float32),
        image_size=np.asarray([3072.0, 4080.0], dtype=np.float32),
    )


def _metric(boundary: tuple[int, int], *, qualified: bool = True) -> BridgePairMetrics:
    left, right = boundary
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
        verified_inliers=25 if qualified else 0,
        inlier_ratio=0.25 if qualified else 0.0,
        qualified=qualified,
    )


def _model(path: Path, *, registered: int = 274, points: int = 2000, error: float = 0.8) -> ModelMetrics:
    names = tuple(record.filename for record in _records()[:registered])
    return ModelMetrics(
        model_path=path,
        registered_images=registered,
        total_images=288,
        sparse_points=points,
        observations=points * 3,
        mean_track_length=3.0,
        mean_reprojection_error=error,
        camera_count=1,
        camera_model="SIMPLE_RADIAL",
        camera_params=(3069.0, 1536.0, 2040.0, 0.0),
        registered_image_names=names,
    )


def _attempt(path: Path, *, registered: int = 274) -> AttemptMetrics:
    model = _model(path / "0", registered=registered)
    return AttemptMetrics(
        name="external_aliked_lightglue",
        workspace=path,
        overlap=20,
        database=DatabaseMetrics(288, 500000, 5600, 4000),
        models=(model,),
        best_model=model,
        runtime_seconds=10.0,
        pycolmap_version="4.2.0",
    )


def _configure_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    root = tmp_path
    output = root / "reconstruction" / "external_learned_recovery"
    paths = {
        "ROOT": root,
        "SELECTED_DIR": root / "selected",
        "SELECTION_MANIFEST": root / "selection_manifest.csv",
        "OUTPUT_ROOT": output,
        "REPORTS_DIR": output / "reports",
        "PREVIEWS_DIR": output / "previews",
        "WORK_DIR": output / "work",
        "FEATURE_CACHE_DIR": output / "work" / "features",
        "MAPPING_OUTPUT_DIR": output / "work" / "mapping_output",
        "BEST_DIR": output / "best",
        "CAPABILITY_JSON": output / "reports" / "step13_capability.json",
        "CANDIDATES_CSV": output / "reports" / "step13_candidates.csv",
        "BOUNDARY_JSON": output / "reports" / "step13_boundary_summary.json",
        "ATTEMPT_JSON": output / "reports" / "step13_attempt.json",
        "SUMMARY_JSON": output / "reports" / "step13_summary.json",
        "REGISTERED_CSV": output / "reports" / "step13_registered_images.csv",
        "BOUNDARY_FIGURE": output / "previews" / "step13_01_boundary_comparison.png",
        "SPARSE_FIGURE": output / "previews" / "step13_02_sparse_model.png",
        "REGISTRATION_FIGURE": output / "previews" / "step13_03_registration.png",
        "MODEL_COMPARISON_FIGURE": output / "previews" / "step13_04_model_comparison.png",
        "PLY_PATH": output / "best" / "points3D.ply",
        "STEP11_CANDIDATES": root / "step11_candidates.csv",
        "STEP10_BEST": root / "step10_best",
        "STEP11_BEST": root / "step11_best",
    }
    for name, value in paths.items():
        monkeypatch.setattr(runner, name, value)
    paths["SELECTED_DIR"].mkdir(parents=True)
    paths["SELECTION_MANIFEST"].write_text("fixture", encoding="utf-8")
    paths["STEP11_CANDIDATES"].write_text("fixture", encoding="utf-8")
    return paths


def _write_boundary_report(path: Path, *, allowed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "targeted_allowed": allowed,
                "targeted_gate_reason": "ok" if allowed else "missing bridge",
                "boundaries": [],
            }
        ),
        encoding="utf-8",
    )


def _write_candidates(path: Path, metrics: tuple[BridgePairMetrics, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for metric in metrics:
        c = metric.candidate
        rows.append(
            {
                "boundary": f"{c.boundary_left}-{c.boundary_right}",
                "boundary_left": c.boundary_left,
                "boundary_right": c.boundary_right,
                "left_index": c.left_index,
                "right_index": c.right_index,
                "left_filename": c.left_filename,
                "right_filename": c.right_filename,
                "sequence_gap": c.sequence_gap,
                "raw_matches": metric.raw_matches,
                "verified_inliers": metric.verified_inliers,
                "inlier_ratio": metric.inlier_ratio,
                "qualified": int(metric.qualified),
                "selected": int(metric.qualified),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_runner_exposes_only_five_step13_stages():
    assert runner.STAGES == ("capability", "diagnose", "map", "finalize", "all")
    assert runner.CONFIG == ExternalLearnedConfig()


def test_capability_verifies_inputs_runs_smoke_and_writes_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _configure_paths(tmp_path, monkeypatch)
    verified = _verified(paths["SELECTED_DIR"])
    calls: list[str] = []
    monkeypatch.setattr(runner, "verify_selected_images", lambda *a, **k: calls.append("verify") or verified)
    monkeypatch.setattr(runner, "runtime_snapshot", lambda *a: {"device": "cuda"})
    monkeypatch.setattr(
        runner,
        "smoke_frontend",
        lambda first, second, config: calls.append(f"smoke:{first.name}:{second.name}")
        or {"status": "passed", "first_features": 100, "second_features": 110, "raw_matches": 50},
    )

    result = runner.run_stage("capability")

    assert calls == ["verify", "smoke:image001.jpg:image002.jpg"]
    assert result["status"] == "passed"
    assert result["input_image_count"] == 288
    assert result["selection_manifest_sha256"] == "manifest-sha"
    assert json.loads(paths["CAPABILITY_JSON"].read_text(encoding="utf-8")) == result
    assert not paths["CAPABILITY_JSON"].with_suffix(".json.tmp").exists()


def test_diagnose_stops_before_full_extraction_when_capability_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _configure_paths(tmp_path, monkeypatch)
    paths["CAPABILITY_JSON"].parent.mkdir(parents=True)
    paths["CAPABILITY_JSON"].write_text(
        json.dumps({"status": "blocked", "smoke": {"status": "blocked"}}), encoding="utf-8"
    )
    monkeypatch.setattr(
        runner,
        "prepare_feature_cache",
        lambda *a, **k: pytest.fail("blocked capability must not extract all images"),
    )

    result = runner.run_stage("diagnose")

    assert result["status"] == "blocked"
    assert not paths["BOUNDARY_JSON"].exists()


def test_diagnose_measures_candidates_and_applies_frozen_gate_without_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _configure_paths(tmp_path, monkeypatch)
    paths["CAPABILITY_JSON"].parent.mkdir(parents=True)
    paths["CAPABILITY_JSON"].write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    verified = _verified(paths["SELECTED_DIR"])
    metrics = tuple(_metric(boundary) for boundary in BridgeSearchConfig().boundaries)
    candidates = tuple(metric.candidate for metric in metrics)
    feature_map = {record.filename: _features() for record in verified.records}
    bundle = FrontendBundle(object(), object(), "cuda")
    calls: list[str] = []
    monkeypatch.setattr(runner, "verify_selected_images", lambda *a, **k: verified)
    monkeypatch.setattr(runner, "prepare_feature_cache", lambda *a, **k: (feature_map, bundle, "manifest-sha"))
    monkeypatch.setattr(runner, "generate_candidate_pairs", lambda *a: candidates)
    monkeypatch.setattr(runner, "verify_step11_candidate_identity", lambda *a: calls.append("identity"))
    monkeypatch.setattr(
        runner,
        "match_pairs",
        lambda *a, **k: (
            {
                (m.candidate.left_filename, m.candidate.right_filename): np.zeros(
                    (120, 2), dtype=np.uint32
                )
                for m in metrics
            },
            PairMatchingMetrics(3, 360, 0.1),
        ),
    )
    monkeypatch.setattr(runner, "build_verified_match_database", lambda *a, **k: calls.append("database"))
    monkeypatch.setattr(runner, "summarize_bridge_pairs", lambda *a: metrics)
    monkeypatch.setattr(runner, "run_external_mapping", lambda *a, **k: pytest.fail("diagnose must not map"))

    result = runner.run_stage("diagnose")

    assert calls == ["identity", "database"]
    assert result["status"] == "completed"
    assert result["candidate_count"] == 3
    assert result["targeted_allowed"] is True
    assert result["lightglue_raw_match_count_pre_verification"] == 360
    assert result["colmap_raw_match_count_post_verification"] == 300
    assert result["raw_matches_removed_during_verification"] == 60
    assert result["raw_match_count"] == 300
    assert paths["CANDIDATES_CSV"].exists()
    assert paths["BOUNDARY_JSON"].exists()


def test_map_stage_records_skip_when_any_boundary_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _configure_paths(tmp_path, monkeypatch)
    _write_boundary_report(paths["BOUNDARY_JSON"], allowed=False)
    monkeypatch.setattr(runner, "run_external_mapping", lambda *a, **k: pytest.fail("gate failure must skip mapping"))

    result = runner.run_stage("map")

    assert result["status"] == "skipped"
    assert result["metric_acceptance_met"] is False
    assert json.loads(paths["ATTEMPT_JSON"].read_text(encoding="utf-8")) == result


def test_map_stage_runs_exactly_one_attempt_after_gate_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _configure_paths(tmp_path, monkeypatch)
    _write_boundary_report(paths["BOUNDARY_JSON"], allowed=True)
    selected = tuple(_metric(boundary) for boundary in BridgeSearchConfig().boundaries)
    _write_candidates(paths["CANDIDATES_CSV"], selected)
    verified = _verified(paths["SELECTED_DIR"])
    feature_map = {record.filename: _features() for record in verified.records}
    bundle = FrontendBundle(object(), object(), "cuda")
    attempt = _attempt(paths["MAPPING_OUTPUT_DIR"])
    calls: list[int] = []
    monkeypatch.setattr(runner, "verify_selected_images", lambda *a, **k: verified)
    monkeypatch.setattr(runner, "prepare_feature_cache", lambda *a, **k: (feature_map, bundle, "manifest-sha"))

    def mapping(*args, **kwargs):
        pair_schedule = args[4]
        calls.append(len(pair_schedule))
        return type("Result", (), {"attempt": attempt, "pair_metrics": PairMatchingMetrics(len(pair_schedule), 5000, 1.0)})()

    monkeypatch.setattr(runner, "run_external_mapping", mapping)

    result = runner.run_stage("map")

    assert len(calls) == 1
    assert calls[0] >= 5550
    assert result["status"] == "completed"
    assert result["metric_acceptance_met"] is True
    assert result["attempt"]["best_model"]["registered_images"] == 274


def test_finalization_uses_step10_local_fallback_when_step13_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _configure_paths(tmp_path, monkeypatch)
    paths["CAPABILITY_JSON"].parent.mkdir(parents=True)
    paths["CAPABILITY_JSON"].write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    paths["ATTEMPT_JSON"].write_text(
        json.dumps({"status": "skipped", "reason": "missing bridge", "metric_acceptance_met": False}),
        encoding="utf-8",
    )
    verified = _verified(paths["SELECTED_DIR"])
    step10 = _model(paths["STEP10_BEST"], registered=73, points=6099, error=1.2373)
    step11 = _model(paths["STEP11_BEST"], registered=73, points=3443, error=1.1989)
    monkeypatch.setattr(runner, "verify_selected_images", lambda *a, **k: verified)
    monkeypatch.setattr(
        runner,
        "summarize_reconstruction",
        lambda path, total: step10 if Path(path) == paths["STEP10_BEST"] else step11,
    )
    monkeypatch.setattr(runner, "render_boundary_comparison", lambda: None)
    monkeypatch.setattr(runner, "render_model_comparison", lambda *a: paths["MODEL_COMPARISON_FIGURE"])
    monkeypatch.setattr(runner, "cleanup_transient_work", lambda *a: ())

    result = runner.run_stage("finalize", visual_status="failed")

    assert result["step13_success"] is False
    assert result["selected_sparse_source"] == "step10_best"
    assert result["local_fallback"]["label"] == "step10"
    assert result["dense_reconstruction_started"] is False


def test_all_stage_never_maps_after_failed_diagnostic_gate(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []
    monkeypatch.setattr(runner, "_capability_stage", lambda: calls.append("capability") or {"status": "passed"})
    monkeypatch.setattr(runner, "_diagnose_stage", lambda: calls.append("diagnose") or {"status": "completed", "targeted_allowed": False})
    monkeypatch.setattr(runner, "_map_stage", lambda: pytest.fail("failed diagnostic gate must not map"))
    monkeypatch.setattr(
        runner,
        "_record_mapping_skip",
        lambda boundary: calls.append("record-skip") or {"status": "skipped"},
    )
    monkeypatch.setattr(runner, "_finalize_stage", lambda visual_status="failed": calls.append("finalize") or {"done": True})

    assert runner.run_stage("all") == {"done": True}
    assert calls == ["capability", "diagnose", "record-skip", "finalize"]


def test_runner_source_contains_no_dense_or_second_frontend_fallback():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "patch_match_stereo" not in source
    assert "stereo_fusion" not in source
    assert "LoMa" not in source
    assert "SuperPoint" not in source
    assert "LoFTR" not in source
