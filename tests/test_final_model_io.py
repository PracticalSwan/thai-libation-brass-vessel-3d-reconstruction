from pathlib import Path

import pytest

from final_model_io import (
    build_file_manifest,
    ensure_under_v2_root,
    required_reports_ready,
    owned_work_path,
    sha256_file,
    write_csv_atomic,
    write_json_atomic,
)


def test_v2_boundary_accepts_owned_output(tmp_path: Path):
    root = tmp_path / "reconstruction" / "reference_assisted_v2"
    path = root / "reports" / "x.json"

    assert ensure_under_v2_root(path, root) == path.resolve()


def test_v2_boundary_rejects_parent_escape(tmp_path: Path):
    root = tmp_path / "reconstruction" / "reference_assisted_v2"

    with pytest.raises(ValueError, match="outside V2 root"):
        ensure_under_v2_root(tmp_path / "IMG20260826122949" / "x.jpg", root)


def test_v2_atomic_writers_create_deterministic_owned_outputs(tmp_path: Path):
    root = tmp_path / "reconstruction" / "reference_assisted_v2"
    json_path = root / "reports" / "record.json"
    csv_path = root / "reports" / "record.csv"

    write_json_atomic(json_path, {"z": 2, "a": 1}, root)
    write_csv_atomic(csv_path, ("name", "value"), ({"name": "x", "value": 3},), root)

    assert json_path.read_text(encoding="utf-8") == '{\n  "a": 1,\n  "z": 2\n}\n'
    assert csv_path.read_text(encoding="utf-8") == "name,value\nx,3\n"
    assert sha256_file(json_path) == "83fa4f3b7a42ed169b77d164e26f385306224aa4a684133a6b40f18a8085fa75"


def test_owned_work_path_is_limited_to_v2_work_tree(tmp_path: Path):
    root = tmp_path / "reconstruction" / "reference_assisted_v2"

    assert owned_work_path(root / "work" / "logs" / "run.log", root) is True
    assert owned_work_path(root / "reports" / "run.json", root) is False
    assert owned_work_path(tmp_path / "outside.tmp", root) is False


def test_required_reports_ready_requires_every_accepted_report(tmp_path: Path):
    root = tmp_path / "reconstruction" / "reference_assisted_v2"
    reports = root / "reports"
    reports.mkdir(parents=True)
    names = (
        "final_cv_fit.json",
        "base_geometry_report.json",
        "ornament_build_report.json",
        "cleanup_report.json",
        "uv_bake_report.json",
        "texture_projection_report.json",
        "lookdev_report.json",
        "final_validation_report.json",
    )
    for name in names:
        write_json_atomic(reports / name, {"accepted": True}, root)

    assert required_reports_ready(root) == tuple(reports / name for name in names)
    write_json_atomic(reports / "lookdev_report.json", {"accepted": False}, root)
    with pytest.raises(ValueError, match="lookdev_report.json is not accepted"):
        required_reports_ready(root)


def test_required_reports_ready_accepts_only_disclosed_texture_fallback(tmp_path: Path):
    root = tmp_path / "reconstruction" / "reference_assisted_v2"
    reports = root / "reports"
    reports.mkdir(parents=True)
    names = (
        "final_cv_fit.json",
        "base_geometry_report.json",
        "ornament_build_report.json",
        "cleanup_report.json",
        "uv_bake_report.json",
        "texture_projection_report.json",
        "lookdev_report.json",
        "final_validation_report.json",
    )
    for name in names:
        write_json_atomic(reports / name, {"accepted": True}, root)

    fallback = {
        "status": "BLOCKED",
        "claim_scope": "photo_informed_component_fusion_fallback_no_visual_qa",
        "projection_method": "component_level_photo_informed_fusion_fallback",
        "blocked_reasons": [
            "exact_projection_geometry_package_missing",
            "per_texel_depth_masked_projection_not_run",
            "component_level_photo_informed_fallback_only",
        ],
        "fallback": {"active": True, "texel_direct_projection": False},
        "direct_projection_percent": 0.0,
        "inferred_fill_percent": 100.0,
    }
    write_json_atomic(reports / "texture_projection_report.json", fallback, root)
    assert required_reports_ready(root) == tuple(reports / name for name in names)

    fallback["direct_projection_percent"] = 1.0
    write_json_atomic(reports / "texture_projection_report.json", fallback, root)
    with pytest.raises(ValueError, match="texture_projection_report.json is not accepted"):
        required_reports_ready(root)


def test_build_file_manifest_records_relative_hash_and_size(tmp_path: Path):
    root = tmp_path / "reconstruction" / "reference_assisted_v2"
    asset = root / "final" / "asset.glb"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"v2")

    record = build_file_manifest(asset, root)

    assert record == {
        "path": "final/asset.glb",
        "size_bytes": 2,
        "sha256": sha256_file(asset),
    }
