"""Local reconstruction boundaries; no downloads or real MVS in unit tests."""
import hashlib
import importlib
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


def domain():
    assert importlib.util.find_spec("local_reconstruction"), "local domain is missing"
    return importlib.import_module("local_reconstruction")


def source_metrics(**changes):
    values = dict(registered_images=73, sparse_points=6099, observations=21351,
                  mean_reprojection_error=1.2373052447638215, camera_count=1,
                  camera_model="SIMPLE_RADIAL",
                  camera_params=(3542.7206959261907, 1536., 2040., -0.01641661056124677))
    values.update(changes)
    return SimpleNamespace(**values)


def test_source_rejects_step13_even_if_metrics_are_valid(tmp_path):
    mod = domain()
    mod.validate_source(tmp_path, tmp_path / "reconstruction/sparse/best", source_metrics())
    with pytest.raises(ValueError, match="Step 10"):
        mod.validate_source(tmp_path, tmp_path / "reconstruction/external_learned_recovery/best", source_metrics())
    with pytest.raises(ValueError, match="metrics"):
        mod.validate_source(tmp_path, tmp_path / "reconstruction/sparse/best", source_metrics(sparse_points=6098))


def test_registered_manifest_uses_model_membership_and_selection_order():
    mod = domain()
    records = [SimpleNamespace(filename=n, index=i, sha256="a" * 64, size_bytes=1)
               for i, n in enumerate(["z.jpg", "a.jpg", "b.jpg"], 1)]
    rows = mod.registered_manifest(records, ["b.jpg", "z.jpg"], expected_count=2)
    assert [r["filename"] for r in rows] == ["z.jpg", "b.jpg"]
    assert [r["selected_index"] for r in rows] == [1, 3]
    assert mod.names_hash(rows) == hashlib.sha256(b"z.jpg\nb.jpg\n").hexdigest()
    with pytest.raises(ValueError):
        mod.registered_manifest(records, ["z.jpg", "missing.jpg"], expected_count=2)
    with pytest.raises(ValueError):
        mod.registered_manifest(records, ["z.jpg", "z.jpg"], expected_count=2)


def test_output_and_cleanup_cannot_escape_local_work(tmp_path):
    mod = domain()
    good = tmp_path / "reconstruction/local_dense/work/dense_workspace"
    assert mod.safe_output(tmp_path, good) == good.resolve()
    for bad in (tmp_path / "IMG20260826122949/x", good / "../../../../sparse/best",
                tmp_path / "reconstruction/local_dense"):
        with pytest.raises(ValueError):
            mod.safe_output(tmp_path, bad)
    with pytest.raises(ValueError):
        mod.cleanup_target(tmp_path, tmp_path / "reconstruction/local_dense/reports")
    assert mod.cleanup_target(tmp_path, good) == good.resolve()


def test_cli_args_preserve_spaces_and_reject_sparse_commands(tmp_path):
    mod = domain()
    exe = tmp_path / "tool dir/colmap.exe"
    args = mod.colmap_command(exe, "stereo_fusion", {"workspace_path": tmp_path / "with spaces", "input_type": "geometric"})
    assert args == [str(exe), "stereo_fusion", "--workspace_path", str(tmp_path / "with spaces"), "--input_type", "geometric"]
    with pytest.raises(ValueError):
        mod.colmap_command(exe, "mapper", {})
    with pytest.raises(ValueError):
        mod.colmap_command(exe.with_suffix(".bat"), "stereo_fusion", {})


def test_subprocess_reports_real_failure_and_timeout(tmp_path):
    mod = domain()
    result = mod.run_command([sys.executable, "-B", "-c", "import sys; print('measured error'); sys.exit(7)"], tmp_path / "fail.log", timeout=10)
    assert result["returncode"] == 7 and result["status"] == "failed"
    assert "measured error" in result["output_tail"]
    result = mod.run_command([sys.executable, "-B", "-c", "import time; time.sleep(5)"], tmp_path / "timeout.log", timeout=.05)
    assert result["status"] == "timeout"


def test_subprocess_records_live_pid_before_waiting(tmp_path):
    mod = domain()
    pids = []
    result = mod.run_command([sys.executable, "-B", "-c", "print('done')"],
                             tmp_path / "pid.log", timeout=10, on_start=pids.append)
    assert len(pids) == 1 and pids[0] > 0
    assert result["returncode"] == 0


@pytest.mark.skipif(os.name != "nt", reason="Windows process-tree contract")
def test_subprocess_timeout_terminates_descendants(tmp_path):
    mod = domain()
    child_pid_path = tmp_path / "child.pid"
    parent_code = (
        "import pathlib,subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(60)']);"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid),encoding='ascii');"
        "time.sleep(60)"
    )
    result = mod.run_command([sys.executable, "-B", "-c", parent_code],
                             tmp_path / "tree-timeout.log", timeout=2)
    assert result["status"] == "timeout"
    child_pid = int(child_pid_path.read_text(encoding="ascii"))
    deadline = time.monotonic() + 5
    while mod.process_tree_pids(child_pid) and time.monotonic() < deadline:
        time.sleep(.05)
    assert mod.process_tree_pids(child_pid) == []


def test_fallback_requires_resource_failure_and_cannot_run_a_third_attempt():
    mod = domain()
    initial = {"max_image_size": 1600, "geom_consistency": True, "filter": True, "source_views": 30}
    retry = mod.dense_fallback([{"config": initial, "status": "failed", "failure_category": "resource"}])
    assert retry["max_image_size"] == 1200 and retry["filter"] is True
    assert retry["geom_consistency"] is True
    for status, category in (("timeout", "timeout"), ("interrupted", "runtime")):
        retry = mod.dense_fallback([{"config": initial, "status": status,
                                     "failure_category": category}])
        assert retry["max_image_size"] == 1200
    for attempts in ([{"config": initial, "status": "failed", "failure_category": "capability"}],
                     [{"config": initial, "status": "completed", "failure_category": None}],
                     [{"config": initial, "status": "failed", "failure_category": "resource"}] * 2):
        with pytest.raises(ValueError):
            mod.dense_fallback(attempts)


def test_dense_gate_rejects_noise_metrics_and_requires_visual_evidence():
    mod = domain()
    metrics = dict(point_count=7000, finite_xyz_fraction=1., bounding_box_min=[0, 0, 0],
                   bounding_box_max=[1, 2, 3], bounding_box_extents=[1, 2, 3], rank=3, has_color=True)
    source = dict(bounding_box_min=[-1, -1, -1], bounding_box_max=[2, 3, 4], bounding_box_extents=[3, 4, 5])
    assert mod.dense_gate(metrics, source, "passed")["passed"]
    assert not mod.dense_gate(metrics, source, "pending")["passed"]
    assert not mod.dense_gate(dict(metrics, point_count=6099), source, "passed")["passed"]
    assert not mod.dense_gate(dict(metrics, finite_xyz_fraction=.99), source, "passed")["passed"]
    assert not mod.dense_gate(dict(metrics, rank=2), source, "passed")["passed"]
    assert not mod.dense_gate(dict(metrics, bounding_box_max=[100, 2, 3]), source, "passed")["passed"]


def test_dense_gate_accepts_a_plausible_robust_core_with_disclosed_rare_outliers():
    mod = domain()
    metrics = dict(point_count=7000, finite_xyz_fraction=1.,
                   bounding_box_min=[-100, -100, -100], bounding_box_max=[100, 100, 100],
                   robust_bounding_box_min=[-.9, -.8, -.7],
                   robust_bounding_box_max=[.9, .8, .7],
                   reference_expanded_fraction=.999, rank=3, has_color=True)
    source = dict(bounding_box_min=[-1, -1, -1], bounding_box_max=[1, 1, 1])
    result = mod.dense_gate(metrics, source, "passed")
    assert result["passed"]
    assert result["checks"]["reference_coverage"]
    assert not mod.dense_gate(dict(metrics, reference_expanded_fraction=.98), source, "passed")["passed"]


def test_integrated_summary_cannot_turn_missing_stages_into_success():
    mod = domain()
    reports = {str(s): {"acceptance": {"passed": True}} for s in range(14, 18)}
    assert mod.integrated_success(reports)
    reports["17"]["acceptance"]["passed"] = False
    assert not mod.integrated_success(reports)
    del reports["17"]
    assert not mod.integrated_success(reports)


def test_json_is_deterministic_and_rejects_nonfinite(tmp_path):
    mod = domain()
    path = tmp_path / "report.json"
    mod.write_json(path, {"z": 1, "a": [2]})
    assert path.read_bytes().startswith(b'{\n  "a"')
    before = path.read_bytes()
    with pytest.raises(ValueError):
        mod.write_json(path, {"bad": float("nan")})
    assert path.read_bytes() == before


def test_mesh_gate_requires_faces_supported_component_and_visual_review():
    mod = domain()
    metrics = dict(vertex_count=100, face_count=180, finite_xyz_fraction=1., rank=3,
                   bounding_box_min=[0, 0, 0], bounding_box_max=[1, 2, 3], bounding_box_extents=[1, 2, 3])
    reference = dict(bounding_box_min=[-1, -1, -1], bounding_box_max=[2, 3, 4], bounding_box_extents=[3, 4, 5])
    assert mod.mesh_gate(metrics, reference, .9, "passed")["passed"]
    assert not mod.mesh_gate(metrics, reference, .2, "passed")["passed"]
    assert mod.mesh_gate(metrics, reference, .3, "passed")["passed"]
    assert not mod.mesh_gate(dict(metrics, face_count=0), reference, .9, "passed")["passed"]
    assert not mod.mesh_gate(metrics, reference, .9, "pending")["passed"]


def test_mesh_choice_prioritizes_validity_over_face_count():
    mod = domain()
    bad = dict(name="huge", acceptance={"passed": False}, metrics={"face_count": 999999})
    good = dict(name="supported", acceptance={"passed": True}, metrics={"face_count": 100})
    assert mod.choose_mesh([bad, good])["name"] == "supported"
    assert mod.choose_mesh([bad]) is None
    with pytest.raises(ValueError):
        mod.choose_mesh([good] * 3)


def test_simplification_decision_uses_measured_mesh_size():
    mod = domain()
    practical = mod.simplification_plan({"face_count": 400_000, "file_size_bytes": 80_000_000})
    assert practical == {"used": False, "target_faces": None,
                         "reason": "Measured mesh is practical for texturing and headless Blender validation"}
    heavy = mod.simplification_plan({"face_count": 1_200_000, "file_size_bytes": 600_000_000})
    assert heavy["used"] is True
    assert 0 < heavy["target_faces"] < 1_200_000
