"""Stage ordering and source workspace checks without executing COLMAP jobs."""
import importlib
import json
from contextlib import nullcontext
from pathlib import Path
import sys

import pytest


def runner():
    assert importlib.util.find_spec("run_local_reconstruction"), "staged runner is missing"
    return importlib.import_module("run_local_reconstruction")


def test_all_sequences_stage_owned_gates_and_finalizes_partial_result():
    mod = runner()
    calls = []
    def stage(name, passed):
        def run():
            calls.append(name)
            return {"acceptance": {"passed": passed}}
        return run
    stages = [stage("prepare", True), stage("stereo", False), stage("mesh", True)]
    result = mod.run_sequence(stages, stage("finalize", False))
    assert calls == ["prepare", "stereo", "finalize"]
    assert result["acceptance"]["passed"] is False


def test_all_does_not_duplicate_stage_acceptance_logic():
    mod = runner()
    calls = []
    def passed():
        calls.append(1)
        return {"acceptance": {"passed": True}, "other_metric": False}
    mod.run_sequence([passed]*4, passed)
    assert len(calls) == 5


def test_depth_map_parser_checks_payload_and_channels(tmp_path):
    mod = runner()
    import numpy as np
    path = tmp_path / "depth.bin"
    path.write_bytes(b"2&3&1&" + np.ones(6, dtype="<f4").tobytes())
    assert mod.inspect_map(path, channels=1) == {"width": 2, "height": 3, "channels": 1, "finite": True, "positive_count": 6}
    with pytest.raises(ValueError):
        mod.inspect_map(path, channels=3)
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(ValueError):
        mod.inspect_map(path, channels=1)


def test_capability_rejects_nocuda_and_wrong_version():
    mod = runner()
    commands = {name: True for name in mod.DENSE_COMMANDS}
    assert mod.capability_gate("COLMAP 4.2.0 (with CUDA)", commands, True)
    assert not mod.capability_gate("COLMAP 4.2.0 (without CUDA)", commands, True)
    assert not mod.capability_gate("COLMAP 4.1.0 (with CUDA)", commands, True)
    commands["patch_match_stereo"] = False
    assert not mod.capability_gate("COLMAP 4.2.0 (with CUDA)", commands, True)


def test_capability_report_records_live_blender_path_and_version(tmp_path, monkeypatch):
    mod = runner()
    colmap = tmp_path / "colmap.exe"
    blender = tmp_path / "blender.exe"
    colmap.write_bytes(b"colmap")
    blender.write_bytes(b"blender")

    def fake_run(args, *unused, **kwargs):
        executable = Path(args[0])
        if executable == blender:
            return {"status": "completed", "output_tail": "Blender 5.2.0 LTS\n", "returncode": 0}
        if executable.name == "nvidia-smi":
            return {"status": "completed", "output_tail": "GPU", "returncode": 0}
        if executable == Path(sys.executable):
            return {"status": "completed", "output_tail": "CUDA", "returncode": 0}
        tail = "COLMAP 4.2.0 (with CUDA)" if args[1:] == ["-h"] else "Options can either be specified"
        return {"status": "completed", "output_tail": tail, "returncode": 0}

    monkeypatch.setattr(mod, "run_command", fake_run)
    report = mod.Pipeline(tmp_path, executable=colmap, blender=blender).capability()
    assert report["blender"]["path"] == str(blender.resolve())
    assert report["blender"]["version"] == "Blender 5.2.0 LTS"
    assert report["blender"]["status"] == "completed"


def test_stereo_refuses_failed_prepare_before_creating_attempt(tmp_path):
    mod = runner()
    pipeline = mod.Pipeline(tmp_path)
    pipeline.save("step14_summary.json", {"acceptance": {"passed": False}})
    with pytest.raises(ValueError, match="Step 14"):
        pipeline.stereo()
    assert not (pipeline.reports / "step15_dense_attempts.json").exists()


def test_visual_review_is_bound_to_actual_artifact_and_preview(tmp_path):
    mod = runner()
    artifact, preview = tmp_path / "model.ply", tmp_path / "preview.png"
    artifact.write_bytes(b"real artifact")
    preview.write_bytes(b"real preview")
    report = {}
    assert mod.visual_review(report, artifact, preview, None, None)["status"] == "pending"
    with pytest.raises(ValueError):
        mod.visual_review(report, artifact, preview, "passed", "")
    review = mod.visual_review(report, artifact, preview, "passed", "Coherent local vessel; outliers disclosed")
    assert review["status"] == "passed"
    artifact.write_bytes(b"modified artifact")
    assert mod.visual_review({"visual_review": review}, artifact, preview, None, None)["status"] == "pending"


def test_pipeline_os_lock_rejects_concurrent_owner(tmp_path):
    mod = runner()
    with mod.pipeline_lock(tmp_path):
        with pytest.raises(OSError):
            with mod.pipeline_lock(tmp_path):
                pytest.fail("Concurrent owner acquired the same reconstruction lock")


def test_mesh_refuses_unaccepted_dense_gate(tmp_path):
    mod = runner()
    pipeline = mod.Pipeline(tmp_path)
    pipeline.save("step15_summary.json", {"acceptance": {"passed": False}})
    with pytest.raises(ValueError, match="Step 15"):
        pipeline.mesh()
    assert not (pipeline.reports / "step16_mesh_attempts.json").exists()


def test_texture_refuses_unaccepted_mesh_gate(tmp_path):
    mod = runner()
    pipeline = mod.Pipeline(tmp_path)
    pipeline.save("step16_summary.json", {"acceptance": {"passed": False}})
    with pytest.raises(ValueError, match="Step 16"):
        pipeline.texture()
    assert not (pipeline.reports / "step17_texture_summary.json").exists()


def test_finalize_is_true_only_when_all_stage_reports_pass(tmp_path):
    mod = runner()
    pipeline = mod.Pipeline(tmp_path)
    for stage in range(14, 17):
        pipeline.save(f"step{stage}_summary.json", {"acceptance": {"passed": True}})
    pipeline.save("step17_texture_summary.json", {"acceptance": {"passed": True}})
    summary = pipeline.finalize()
    assert summary["steps14_17_success"] is True
    pipeline.save("step17_texture_summary.json", {"acceptance": {"passed": False}})
    assert pipeline.finalize()["steps14_17_success"] is False


def test_all_cli_uses_the_tested_run_sequence(monkeypatch, tmp_path):
    mod = runner()
    calls = []

    class FakePipeline:
        def __init__(self, **unused):
            self.work = tmp_path
        def prepare(self): return {"acceptance": {"passed": True}}
        def stereo(self, *unused): return {"acceptance": {"passed": True}}
        def mesh(self, *unused): return {"acceptance": {"passed": True}}
        def texture(self, *unused): return {"acceptance": {"passed": True}}
        def finalize(self): return {"steps14_17_success": True}

    original = mod.run_sequence
    def observed(stages, finalize):
        calls.append("run_sequence")
        return original(stages, finalize)

    monkeypatch.setattr(mod, "Pipeline", FakePipeline)
    monkeypatch.setattr(mod, "pipeline_lock", lambda unused: nullcontext())
    monkeypatch.setattr(mod, "run_sequence", observed)
    monkeypatch.setattr(sys, "argv", ["run_local_reconstruction.py", "--stage", "all"])
    assert mod.main() == 0
    assert calls == ["run_sequence"]


def test_texture_asset_identity_binds_mesh_texture_manifest_and_preview(tmp_path):
    mod = runner()
    output = tmp_path / "attempt_1"
    output.mkdir()
    mesh, texture, preview = output / "mesh.ply", output / "texture.png", tmp_path / "preview.png"
    mesh.write_bytes(b"mesh")
    texture.write_bytes(b"texture")
    preview.write_bytes(b"preview")
    validation = {"output_manifest": [
        {"relative_path": "mesh.ply", "sha256": mod.sha256(mesh)},
        {"relative_path": "texture.png", "sha256": mod.sha256(texture)},
    ]}
    before = mod.texture_asset_identity(output, validation, preview)
    texture.write_bytes(b"changed")
    after = mod.texture_asset_identity(output, validation, preview)
    assert before != after
    assert before["preview_sha256"] == mod.sha256(preview)


def test_texture_attempt_outputs_are_isolated_and_bounded(tmp_path):
    mod = runner()
    pipeline = mod.Pipeline(tmp_path)
    first = pipeline.texture_attempt_dir(1)
    second = pipeline.texture_attempt_dir(2)
    assert first != second
    assert first == (tmp_path / "reconstruction/local_dense/texture/attempt_1").resolve()
    assert second == (tmp_path / "reconstruction/local_dense/texture/attempt_2").resolve()
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "partial.tmp").write_bytes(b"partial")
    (second / "preserve.tmp").write_bytes(b"preserve")
    pipeline.cleanup_texture_attempt(1)
    assert not first.exists()
    assert (second / "preserve.tmp").read_bytes() == b"preserve"
    with pytest.raises(ValueError):
        pipeline.cleanup_texture_attempt(3)


def test_texture_rejects_unrecorded_existing_attempt_output(tmp_path, monkeypatch):
    mod = runner()
    pipeline = mod.Pipeline(tmp_path)
    final_mesh = tmp_path / "reconstruction/local_dense/mesh/final_mesh.ply"
    final_mesh.parent.mkdir(parents=True)
    final_mesh.write_bytes(b"mesh")
    monkeypatch.setattr(pipeline, "require", lambda *unused: {
        "final_mesh_path": final_mesh.relative_to(tmp_path).as_posix()
    })
    monkeypatch.setattr(pipeline, "integrity", lambda *unused: (
        {"protected_unchanged": True}, None, [{"filename": "a.jpg"}]
    ))
    attempt = pipeline.texture_attempt_dir(1)
    attempt.mkdir(parents=True)
    (attempt / "unowned.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="Unrecorded texture attempt"):
        pipeline.texture()
    assert (attempt / "unowned.txt").read_text(encoding="utf-8") == "preserve"


def test_mesh_preview_evidence_discloses_vertex_sampling():
    mod = runner()
    large = mod.mesh_preview_evidence({"vertex_count": 283341, "face_count": 499999})
    small = mod.mesh_preview_evidence({"vertex_count": 1000, "face_count": 2000})
    assert large["mode"] == "sampled_vertices" and large["sampled_vertex_count"] == 100000
    assert small["mode"] == "triangle_surface"


def _mesh_fixture(path):
    from local_reconstruction_io import write_ply
    import numpy as np
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    write_ply(path, xyz, faces=faces)


def _stub_mesh_pipeline(mod, tmp_path, monkeypatch):
    pipeline = mod.Pipeline(tmp_path)
    fused = tmp_path / "reconstruction/local_dense/dense/fused.ply"
    fused.parent.mkdir(parents=True)
    fused.write_bytes(b"dense")
    Path(str(fused) + ".vis").write_bytes(b"visibility")
    workspace = tmp_path / "reconstruction/local_dense/work/dense_workspace"
    workspace.mkdir(parents=True)
    dense = {"fused_path": fused.relative_to(tmp_path).as_posix(),
             "workspace": workspace.relative_to(tmp_path).as_posix(),
             "metrics": {"bounding_box_min": [-1, -1, -1], "bounding_box_max": [2, 2, 2]}}
    monkeypatch.setattr(pipeline, "require", lambda *unused: dense)
    monkeypatch.setattr(pipeline, "integrity", lambda *unused: ({"protected_unchanged": True}, None,
                                                                [{"filename": "a.jpg"}]))
    def render(candidate):
        path = Path(candidate["preview_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"preview")
    monkeypatch.setattr(pipeline, "_render_mesh_candidate", render)
    monkeypatch.setattr(pipeline, "_comparison_preview", lambda candidates: None)
    return pipeline


def test_mesh_records_primary_subprocess_failure_before_one_alternative(tmp_path, monkeypatch):
    mod = runner()
    pipeline = _stub_mesh_pipeline(mod, tmp_path, monkeypatch)
    calls = []
    def command(name, options, *unused, **kwargs):
        calls.append(name)
        if name == "poisson_mesher":
            return {"status": "failed", "returncode": 1, "output_tail": "measured failure"}
        _mesh_fixture(Path(options["output_path"]))
        return {"status": "completed", "returncode": 0, "output_tail": "done"}
    monkeypatch.setattr(pipeline, "command", command)
    report = pipeline.mesh()
    assert calls == ["poisson_mesher", "delaunay_mesher"]
    assert report["status"] == "pending_visual"
    assert report["attempts"][0]["process"]["status"] == "failed"
    assert len(report["attempts"]) == 2


def test_mesh_visual_rejection_persists_and_restart_does_not_add_a_third_candidate(tmp_path, monkeypatch):
    mod = runner()
    pipeline = _stub_mesh_pipeline(mod, tmp_path, monkeypatch)
    calls = []
    def command(name, options, *unused, **kwargs):
        calls.append(name)
        _mesh_fixture(Path(options["output_path"]))
        return {"status": "completed", "returncode": 0, "output_tail": "done"}
    monkeypatch.setattr(pipeline, "command", command)
    first = pipeline.mesh("failed", "Primary is an enclosing shell unrelated to the vessel")
    assert first["status"] == "pending_visual"
    assert first["attempts"][0]["visual_review"]["status"] == "failed"
    assert len(first["attempts"]) == 2
    second = pipeline.mesh()
    assert second["attempts"][0]["visual_review"]["status"] == "failed"
    assert len(second["attempts"]) == 2
    assert calls == ["poisson_mesher", "delaunay_mesher"]


def test_mesh_rejects_unbound_existing_final_output(tmp_path, monkeypatch):
    mod = runner()
    pipeline = _stub_mesh_pipeline(mod, tmp_path, monkeypatch)
    final_path = tmp_path / "reconstruction/local_dense/mesh/final_mesh.ply"
    final_path.parent.mkdir(parents=True)
    final_path.write_bytes(b"unowned mesh")

    def command(unused_name, options, *unused, **unused_kwargs):
        _mesh_fixture(Path(options["output_path"]))
        return {"status": "completed", "returncode": 0, "output_tail": "done"}

    monkeypatch.setattr(pipeline, "command", command)
    with pytest.raises(ValueError, match="Unbound final mesh"):
        pipeline.mesh("passed", "The fixture is a bounded surface for ownership testing")
    assert final_path.read_bytes() == b"unowned mesh"


def test_blender_success_requires_reopenable_preview_and_report(tmp_path, monkeypatch):
    mod = runner()
    (tmp_path / "render_local_reconstruction.py").write_text("pass\n", encoding="utf-8")
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"blender")
    pipeline = mod.Pipeline(tmp_path, blender=blender)
    geometry = tmp_path / "geometry.npz"
    cameras = tmp_path / "cameras.json"
    geometry.write_bytes(b"geometry")
    cameras.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(pipeline, "_preview_inputs", lambda *unused: (geometry, cameras))
    monkeypatch.setattr(mod, "run_command", lambda *args, **kwargs: {
        "status": "completed", "returncode": 0, "output_tail": "Blender quit"
    })
    preview = tmp_path / "reconstruction/local_dense/previews/preview.png"
    report = tmp_path / "reconstruction/local_dense/reports/report.json"
    preview.parent.mkdir(parents=True)
    report.parent.mkdir(parents=True)
    preview.write_bytes(b"stale")
    report.write_text('{"status":"completed"}', encoding="utf-8")
    result = pipeline._run_blender_preview(tmp_path / "mesh.ply", tmp_path / "texture.png", preview, report)
    assert result["status"] == "failed"
    assert "render contract" in result["output_tail"]
    assert not preview.exists() and not report.exists()


def test_texture_resume_may_restore_only_a_missing_source_identity(tmp_path):
    mod = runner()
    rows = [{"filename": "a.jpg"}]
    expected = mod.names_hash(rows)
    assert mod.restorable_source_identity(None, expected) == expected
    assert mod.restorable_source_identity(expected, expected) == expected
    with pytest.raises(ValueError, match="identity"):
        mod.restorable_source_identity("wrong", expected)
