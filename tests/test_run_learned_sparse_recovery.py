from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from analysis_common import SelectedImageRecord, VerifiedSelectedSet
from learned_sparse_recovery import FrontendSmokeResult
import run_learned_sparse_recovery as runner
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
            size_bytes=1,
            sha256="0" * 64,
            decision="ACCEPT",
            reasons="",
        )
        for index in range(1, count + 1)
    )


def _verified(image_dir: Path, count: int = 288) -> VerifiedSelectedSet:
    return VerifiedSelectedSet(_records(count), "manifest-sha", image_dir)


def _metric(boundary: tuple[int, int]) -> BridgePairMetrics:
    left, right = boundary
    candidate = BridgeCandidate(
        left,
        right,
        left - 39,
        right + 1,
        f"image{left - 39:03d}.jpg",
        f"image{right + 1:03d}.jpg",
    )
    return BridgePairMetrics(candidate, 100, 25, 0.25, True)


def _attempt(path: Path, *, registered: int = 274) -> AttemptMetrics:
    names = tuple(record.filename for record in _records()[:registered])
    model = ModelMetrics(
        model_path=path / "0",
        registered_images=registered,
        total_images=288,
        sparse_points=2000,
        observations=6000,
        mean_track_length=3.0,
        mean_reprojection_error=0.8,
        camera_count=1,
        camera_model="SIMPLE_RADIAL",
        camera_params=(3070.0, 1536.0, 2040.0, 0.0),
        registered_image_names=names,
    )
    return AttemptMetrics(
        name=f"{path.name}_targeted",
        workspace=path,
        overlap=20,
        database=DatabaseMetrics(288, 500000, 1500, 900),
        models=(model,),
        best_model=model,
        runtime_seconds=1.5,
        pycolmap_version="4.2.0",
    )


def _configure_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    learned_root = tmp_path / "reconstruction" / "learned_recovery"
    paths = {
        "ROOT": tmp_path,
        "SELECTED_DIR": tmp_path / "images",
        "SELECTION_MANIFEST": tmp_path / "selection_manifest.csv",
        "LEARNED_ROOT": learned_root,
        "ALIKED_DIR": learned_root / "aliked",
        "LOMA_DIR": learned_root / "loma",
        "BEST_DIR": learned_root / "best",
        "REPORTS_DIR": learned_root / "reports",
        "PREVIEWS_DIR": learned_root / "previews",
        "WORK_DIR": learned_root / "work",
        "CAPABILITY_JSON": learned_root / "reports" / "step12_capability.json",
        "ALIKED_CANDIDATES_CSV": learned_root / "reports" / "step12_aliked_candidates.csv",
        "ALIKED_BOUNDARY_JSON": learned_root / "reports" / "step12_aliked_boundary_summary.json",
        "ALIKED_ATTEMPT_JSON": learned_root / "reports" / "step12_aliked_attempt.json",
        "LOMA_CANDIDATES_CSV": learned_root / "reports" / "step12_loma_candidates.csv",
        "LOMA_BOUNDARY_JSON": learned_root / "reports" / "step12_loma_boundary_summary.json",
        "LOMA_ATTEMPT_JSON": learned_root / "reports" / "step12_loma_attempt.json",
        "ATTEMPTS_CSV": learned_root / "reports" / "step12_attempts.csv",
        "REGISTERED_CSV": learned_root / "reports" / "step12_registered_images.csv",
        "SUMMARY_JSON": learned_root / "reports" / "step12_summary.json",
        "BOUNDARY_FIGURE": learned_root / "previews" / "step12_01_boundary_comparison.png",
        "SPARSE_FIGURE": learned_root / "previews" / "step12_02_sparse_model.png",
        "REGISTRATION_FIGURE": learned_root / "previews" / "step12_03_registration.png",
        "FRONTEND_FIGURE": learned_root / "previews" / "step12_04_frontend_comparison.png",
        "PLY_PATH": learned_root / "best" / "points3D.ply",
        "STEP10_SUMMARY": tmp_path / "step10_summary.json",
        "STEP11_EXHAUSTIVE": tmp_path / "step11_exhaustive.json",
        "STEP11_CANDIDATES": tmp_path / "step11_candidates.csv",
    }
    for name, path in paths.items():
        monkeypatch.setattr(runner, name, path)
    paths["SELECTED_DIR"].mkdir()
    paths["SELECTION_MANIFEST"].write_text("fixture", encoding="utf-8")
    paths["STEP10_SUMMARY"].write_text(
        json.dumps(
            {
                "selected_attempt": "baseline",
                "best_model": {
                    "registered_images": 73,
                    "sparse_points": 6099,
                    "mean_reprojection_error": 1.2373,
                    "camera_count": 1,
                    "camera_model": "SIMPLE_RADIAL",
                },
            }
        ),
        encoding="utf-8",
    )
    paths["STEP11_EXHAUSTIVE"].write_text(
        json.dumps(
            {
                "status": "completed",
                "attempt": {
                    "best_model": {
                        "registered_images": 73,
                        "sparse_points": 3443,
                        "mean_reprojection_error": 1.1989,
                        "camera_count": 1,
                        "camera_model": "SIMPLE_RADIAL",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    paths["STEP11_CANDIDATES"].write_text(
        "boundary,boundary_left,boundary_right,left_index,right_index,left_filename,right_filename,verified_inliers\n"
        "73-74,73,74,34,75,a.jpg,b.jpg,0\n",
        encoding="utf-8",
    )
    return paths


def _write_candidate_csv(path: Path, metrics: tuple[BridgePairMetrics, ...]) -> None:
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
                "frontend": "aliked",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_runner_exposes_only_the_seven_bounded_stages():
    assert runner.STAGES == (
        "capability",
        "aliked-diagnose",
        "aliked-map",
        "loma-diagnose",
        "loma-map",
        "finalize",
        "all",
    )
    assert runner.LEARNED_ROOT.name == "learned_recovery"


def test_capability_stage_verifies_input_smokes_only_aliked_and_writes_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _configure_paths(tmp_path, monkeypatch)
    verified = _verified(paths["SELECTED_DIR"])
    calls: list[str] = []
    monkeypatch.setattr(
        runner,
        "verify_selected_images",
        lambda *args, **kwargs: (calls.append("verify") or verified),
    )
    monkeypatch.setattr(
        runner,
        "build_capability_snapshot",
        lambda: {"frontends": {"aliked": {}, "loma": {}}, "pycolmap_version": "4.2.0"},
    )
    monkeypatch.setattr(
        runner,
        "load_rgb_image",
        lambda path: (calls.append(path.name) or np.zeros((4, 4, 3), dtype=np.uint8)),
    )

    def smoke(frontend, first, second):
        calls.append(f"smoke-{frontend.name}")
        return FrontendSmokeResult(frontend.name, "passed", 10, 11, 8)

    monkeypatch.setattr(runner, "smoke_frontend", smoke)

    result = runner.run_stage("capability")

    assert calls == ["verify", "image001.jpg", "image002.jpg", "smoke-aliked"]
    assert result["frontends"]["aliked"]["smoke"]["status"] == "passed"
    assert result["frontends"]["loma"]["smoke"]["status"] == "not_run"
    assert json.loads(paths["CAPABILITY_JSON"].read_text(encoding="utf-8")) == result
    assert not paths["CAPABILITY_JSON"].with_suffix(".json.tmp").exists()
    assert not paths["ALIKED_DIR"].exists()
    assert not paths["LOMA_DIR"].exists()


def test_diagnose_stage_writes_measured_reports_without_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _configure_paths(tmp_path, monkeypatch)
    paths["CAPABILITY_JSON"].parent.mkdir(parents=True)
    paths["CAPABILITY_JSON"].write_text(
        json.dumps(
            {
                "frontends": {
                    "aliked": {"smoke": {"status": "passed"}},
                    "loma": {"smoke": {"status": "not_run"}},
                }
            }
        ),
        encoding="utf-8",
    )
    verified = _verified(paths["SELECTED_DIR"])
    metrics = tuple(_metric(boundary) for boundary in BridgeSearchConfig().boundaries)
    candidates = tuple(metric.candidate for metric in metrics)
    features = paths["WORK_DIR"] / "aliked" / "features.db"
    features.parent.mkdir(parents=True)
    features.write_bytes(b"features")
    calls: list[str] = []
    monkeypatch.setattr(runner, "verify_selected_images", lambda *args, **kwargs: verified)
    monkeypatch.setattr(
        runner,
        "prepare_learned_feature_cache",
        lambda *args, **kwargs: (calls.append("features") or features),
    )
    monkeypatch.setattr(runner, "generate_candidate_pairs", lambda *args: candidates)
    monkeypatch.setattr(
        runner,
        "verify_step11_candidate_identity",
        lambda *args: calls.append("identity"),
    )
    monkeypatch.setattr(
        runner,
        "run_learned_diagnostics",
        lambda *args: (calls.append("diagnostics") or metrics),
    )
    monkeypatch.setattr(
        runner,
        "run_learned_targeted_attempt",
        lambda *args, **kwargs: pytest.fail("diagnosis must not map"),
    )

    result = runner.run_stage("aliked-diagnose")

    assert calls == ["features", "identity", "diagnostics"]
    assert result["status"] == "completed"
    assert result["candidate_count"] == 3
    assert result["targeted_allowed"] is True
    with paths["ALIKED_CANDIDATES_CSV"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert all(row["frontend"] == "aliked" for row in rows)
    assert not paths["ALIKED_ATTEMPT_JSON"].exists()


def test_blocked_diagnose_attempt_preserves_frontend_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _configure_paths(tmp_path, monkeypatch)
    paths["CAPABILITY_JSON"].parent.mkdir(parents=True)
    paths["CAPABILITY_JSON"].write_text(
        json.dumps(
            {
                "frontends": {
                    "aliked": {
                        "smoke": {
                            "status": "blocked",
                            "exception_type": "RuntimeError",
                            "exception_message": "ALIKED feature extraction requires ONNX support.",
                        }
                    },
                    "loma": {"smoke": {"status": "not_run"}},
                }
            }
        ),
        encoding="utf-8",
    )

    result = runner.run_stage("aliked-diagnose")

    assert result["status"] == "blocked"
    assert result["frontend"] == "aliked"
    assert result["extractor_type"] == "ALIKED_N16ROT"
    assert result["matcher_type"] == "ALIKED_LIGHTGLUE"
    assert json.loads(paths["ALIKED_ATTEMPT_JSON"].read_text(encoding="utf-8")) == result


def test_mapping_stage_skips_when_any_boundary_has_no_selected_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _configure_paths(tmp_path, monkeypatch)
    selected = tuple(_metric(boundary) for boundary in ((73, 74), (145, 146)))
    _write_candidate_csv(paths["ALIKED_CANDIDATES_CSV"], selected)
    paths["ALIKED_BOUNDARY_JSON"].write_text(
        json.dumps({"status": "completed", "targeted_allowed": False}), encoding="utf-8"
    )
    monkeypatch.setattr(
        runner,
        "run_learned_targeted_attempt",
        lambda *args, **kwargs: pytest.fail("failed boundary gate must skip mapping"),
    )

    result = runner.run_stage("aliked-map")

    assert result["status"] == "skipped"
    assert "203-204" in result["reason"]
    assert result["attempt"] is None
    assert json.loads(paths["ALIKED_ATTEMPT_JSON"].read_text(encoding="utf-8")) == result


def test_mapping_stage_records_real_attempt_and_metric_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _configure_paths(tmp_path, monkeypatch)
    selected = tuple(_metric(boundary) for boundary in BridgeSearchConfig().boundaries)
    _write_candidate_csv(paths["ALIKED_CANDIDATES_CSV"], selected)
    paths["ALIKED_BOUNDARY_JSON"].write_text(
        json.dumps({"status": "completed", "targeted_allowed": True}), encoding="utf-8"
    )
    verified = _verified(paths["SELECTED_DIR"])
    features = paths["WORK_DIR"] / "aliked" / "features.db"
    features.parent.mkdir(parents=True)
    features.write_bytes(b"features")
    attempt = _attempt(paths["ALIKED_DIR"])
    calls: list[str] = []
    monkeypatch.setattr(runner, "verify_selected_images", lambda *args, **kwargs: verified)
    monkeypatch.setattr(
        runner,
        "prepare_learned_feature_cache",
        lambda *args, **kwargs: (calls.append("features") or features),
    )
    monkeypatch.setattr(
        runner,
        "run_learned_targeted_attempt",
        lambda *args, **kwargs: (calls.append("map") or attempt),
    )

    result = runner.run_stage("aliked-map")

    assert calls == ["features", "map"]
    assert result["status"] == "completed"
    assert result["metric_acceptance_met"] is True
    assert result["attempt"]["best_model"]["registered_images"] == 274


@pytest.mark.parametrize(
    ("boundary_allowed", "attempt_status", "accepted", "required"),
    (
        (False, "skipped", None, True),
        (True, "completed", False, True),
        (True, "completed", True, False),
    ),
)
def test_loma_required_only_after_the_two_approved_aliked_failures(
    boundary_allowed: bool, attempt_status: str, accepted: bool | None, required: bool
):
    boundary = {"status": "completed", "targeted_allowed": boundary_allowed}
    attempt = {
        "status": attempt_status,
        "metric_acceptance_met": accepted,
        "attempt": {} if attempt_status == "completed" else None,
    }
    assert runner.loma_required(boundary, attempt) is required


def test_all_stage_stops_before_every_loma_call_after_aliked_acceptance(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []
    monkeypatch.setattr(runner, "_capability_stage", lambda: calls.append("capability") or {})

    def diagnose(frontend):
        calls.append(f"diagnose-{frontend.name}")
        return {"status": "completed", "targeted_allowed": True}

    def map_frontend(frontend):
        calls.append(f"map-{frontend.name}")
        return {"status": "completed", "metric_acceptance_met": True, "attempt": {}}

    monkeypatch.setattr(runner, "_diagnose_frontend", diagnose)
    monkeypatch.setattr(runner, "_map_frontend", map_frontend)
    monkeypatch.setattr(
        runner,
        "_finalize_stage",
        lambda visual_status="pending": calls.append("finalize") or {"done": True},
    )

    assert runner.run_stage("all") == {"done": True}
    assert calls == ["capability", "diagnose-aliked", "map-aliked", "finalize"]


def test_all_stage_runs_the_single_loma_fallback_once_after_aliked_gate_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []
    monkeypatch.setattr(runner, "_capability_stage", lambda: calls.append("capability") or {})

    def diagnose(frontend):
        calls.append(f"diagnose-{frontend.name}")
        return {
            "status": "completed",
            "targeted_allowed": frontend.name == "loma",
        }

    def map_frontend(frontend):
        calls.append(f"map-{frontend.name}")
        if frontend.name == "aliked":
            return {"status": "skipped", "metric_acceptance_met": None, "attempt": None}
        return {"status": "completed", "metric_acceptance_met": False, "attempt": {}}

    monkeypatch.setattr(runner, "_diagnose_frontend", diagnose)
    monkeypatch.setattr(runner, "_map_frontend", map_frontend)
    monkeypatch.setattr(
        runner,
        "_finalize_stage",
        lambda visual_status="pending": calls.append("finalize") or {"done": True},
    )

    assert runner.run_stage("all") == {"done": True}
    assert calls == [
        "capability",
        "diagnose-aliked",
        "map-aliked",
        "diagnose-loma",
        "map-loma",
        "finalize",
    ]


def test_final_acceptance_rejects_visual_pass_when_metric_gate_failed():
    with pytest.raises(ValueError, match="metric acceptance"):
        runner.derive_final_acceptance(False, "passed")

    assert runner.derive_final_acceptance(True, "pending") == ("pending", False)
    assert runner.derive_final_acceptance(True, "passed") == ("passed", True)
    assert runner.derive_final_acceptance(False, "failed") == ("failed", False)


def test_loma_not_run_reason_preserves_primary_capability_blocker():
    aliked = {
        "status": "blocked",
        "reason": "RuntimeError: ALIKED feature extraction requires ONNX support.",
    }

    reason = runner.loma_not_run_reason(aliked)

    assert "primary learned capability gate was blocked" in reason
    assert "ONNX support" in reason


def test_attempts_csv_keeps_not_run_visible_without_fake_model_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _configure_paths(tmp_path, monkeypatch)
    aliked = {
        "status": "skipped",
        "reason": "boundary gate failed",
        "boundary_gate_allowed": False,
        "attempt": None,
        "metric_acceptance_met": None,
    }
    loma = {
        "status": "not_run",
        "reason": "not required",
        "boundary_gate_allowed": None,
        "attempt": None,
        "metric_acceptance_met": None,
    }

    runner.write_attempts_csv(aliked, loma)

    with paths["ATTEMPTS_CSV"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["frontend"] for row in rows] == ["aliked", "loma"]
    assert [row["status"] for row in rows] == ["skipped", "not_run"]
    assert rows[0]["registered_images"] == ""
    assert rows[1]["sparse_points"] == ""


def test_cleanup_removes_only_allowlisted_regular_files_and_preserves_unknowns(
    tmp_path: Path,
):
    work = tmp_path / "work"
    for frontend in ("aliked", "loma"):
        directory = work / frontend
        directory.mkdir(parents=True)
        (directory / "features.db").write_text("temporary", encoding="utf-8")
        (directory / "diagnostic_pairs.txt").write_text("temporary", encoding="utf-8")
        (directory / "preserve.txt").write_text("keep", encoding="utf-8")

    removed = runner.cleanup_transient_work(work)

    assert {path.name for path in removed} == {"features.db", "diagnostic_pairs.txt"}
    assert (work / "aliked" / "preserve.txt").read_text(encoding="utf-8") == "keep"
    assert (work / "loma" / "preserve.txt").read_text(encoding="utf-8") == "keep"


def test_candidate_figure_rejects_pair_identity_mismatch_before_plotting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _configure_paths(tmp_path, monkeypatch)
    _write_candidate_csv(paths["ALIKED_CANDIDATES_CSV"], (_metric((73, 74)),))

    with pytest.raises(RuntimeError, match="candidate identity/order"):
        runner.render_boundary_comparison()
