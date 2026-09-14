from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

import v4_postfusion as postfusion
from v4_postfusion import (
    _classify_bottom_escape,
    _ring_transition_acceptance,
    audit_final_tile_configs,
    parse_dense_pair_config,
    read_colmap_float_map,
    resolve_colmap_fusion_mask,
)


def test_ring_consistency_threshold_is_diagnostic_not_acceptance_gate():
    result = _ring_transition_acceptance(
        measured_pairs=4,
        selected_pairs=4,
        mean_coverage=0.80,
        mean_consistency=0.10,
    )

    assert result["status"] == "passed"
    assert result["diagnostic_status"] == "below_threshold"
    assert result["diagnostic_threshold"]["passed"] is False


def test_colmap_float_map_reader_validates_dimensions(tmp_path: Path):
    path = tmp_path / "depth.bin"
    path.write_bytes(b"2&3&1&" + (b"\x00\x00\x80?" * 6))
    array = read_colmap_float_map(path)
    assert array.shape == (3, 2, 1)
    assert float(array[0, 0, 0]) == 1.0

    path.write_bytes(b"2&3&1&" + b"\x00")
    with pytest.raises(ValueError, match="payload size mismatch"):
        read_colmap_float_map(path)


def test_depth_pair_consistency_compares_source_camera_z_for_distinct_poses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    depth_dir = tmp_path / "depth"
    mask_dir = tmp_path / "masks"
    depth_dir.mkdir()
    mask_dir.mkdir()

    def write_depth(name: str, value: float) -> None:
        payload = np.full((3, 3), value, dtype="<f4").tobytes()
        (depth_dir / f"{name}.geometric.bin").write_bytes(b"3&3&1&" + payload)

    def write_mask(name: str) -> None:
        Image.new("L", (3, 3), 255).save(mask_dir / f"{name}.png")

    write_depth("reference.jpg", 3.0)
    write_depth("source.jpg", 2.0)
    write_mask("reference.jpg")
    write_mask("source.jpg")

    intrinsics = {"fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 1.0, "width": 3, "height": 3}
    camera_infos = {
        "reference.jpg": {**intrinsics, "matrix": np.eye(3, 4)},
        "source.jpg": {
            **intrinsics,
            "matrix": np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, -1.0],
                ]
            ),
        },
    }
    monkeypatch.setattr(postfusion, "_camera_info", lambda _reconstruction, name: camera_infos[name])

    result = postfusion._depth_pair_consistency(
        "reference.jpg",
        "source.jpg",
        reconstruction=object(),
        depth_dir=depth_dir,
        mask_dir=mask_dir,
    )

    assert result["status"] == "measured"
    assert result["consistent_fraction_at_1pct"] == 1.0
    assert result["relative_depth_median"] == 0.0
    assert result["relative_depth_p95"] == 0.0


def test_colmap_fusion_mask_resolution_requires_appended_png_for_jpeg_names(tmp_path: Path):
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    legacy = mask_dir / "view.jpg"
    legacy.write_bytes(b"legacy")
    assert resolve_colmap_fusion_mask(mask_dir, "view.jpg") == (legacy, "legacy_unextended_image_name")

    appended = mask_dir / "view.jpg.png"
    appended.write_bytes(b"colmap")
    assert resolve_colmap_fusion_mask(mask_dir, "view.jpg") == (appended, "colmap_image_name_plus_png")

    missing, resolution = resolve_colmap_fusion_mask(mask_dir, "missing.jpg")
    assert missing is None
    assert resolution == "missing"


def test_final_tile_audit_reports_actual_cross_ring_support_and_missing_references(tmp_path: Path):
    config = tmp_path / "tile.cfg"
    config.write_text("a.jpg\nb.jpg\nc.jpg\na.jpg\n", encoding="utf-8")
    report = audit_final_tile_configs(
        [config],
        image_names=["a.jpg", "b.jpg", "c.jpg"],
        ring_by_name={"a.jpg": "g7", "b.jpg": "g7", "c.jpg": "g8"},
        max_sources=6,
    )
    assert report["status"] == "failed"
    assert report["observed"]["directed_source_count"] == 2
    assert report["observed"]["cross_ring_directed_source_count"] == 1
    assert report["observed"]["references_without_cross_ring_source"] == ["a.jpg"]
    assert "omit 1 registered references" in " ".join(report["errors"])


def test_historical_567_reference_372_unique_158_duplicate_configuration_is_rejected(tmp_path: Path):
    names = [f"view_{index:03d}.jpg" for index in range(372)]
    ring_by_name = {name: "g1" for name in names}
    first = tmp_path / "tile-000.cfg"
    second = tmp_path / "tile-001.cfg"
    third = tmp_path / "tile-002.cfg"

    def config_text(references: list[str]) -> str:
        return "".join(f"{reference}\nsrc.jpg\n" for reference in references)

    first.write_text(config_text(names), encoding="utf-8")
    second.write_text(config_text(names[:158]), encoding="utf-8")
    third.write_text(config_text(names[:37]), encoding="utf-8")
    report = audit_final_tile_configs(
        [first, second, third],
        image_names=names,
        ring_by_name=ring_by_name,
        max_sources=6,
    )
    assert report["status"] == "failed"
    assert report["observed"]["configured_reference_count_total"] == 567
    assert report["observed"]["reference_count"] == 372
    assert len(report["observed"]["duplicate_reference_writes"]) == 158
    assert report["observed"]["exact_one_reference_write"] is False

    source_only = tmp_path / "tile-003.cfg"
    source_only.write_text("unknown.jpg\nsrc.jpg\n", encoding="utf-8")
    source_only_report = audit_final_tile_configs(
        [source_only],
        image_names=["a.jpg"],
        ring_by_name={"a.jpg": "g1"},
        max_sources=6,
    )
    assert source_only_report["status"] == "failed"
    assert source_only_report["observed"]["source_only_reference_count"] == 1


def test_tile_audit_requires_accepted_sparse_lineage(tmp_path: Path):
    config = tmp_path / "tile.cfg"
    config.write_text("a.jpg\nb.jpg\nb.jpg\na.jpg\n", encoding="utf-8")
    kwargs = {
        "config_paths": [config],
        "image_names": ["a.jpg", "b.jpg"],
        "ring_by_name": {"a.jpg": "g1", "b.jpg": "g2"},
        "max_sources": 6,
    }
    missing = audit_final_tile_configs(**kwargs)
    assert missing["status"] == "failed"
    assert missing["sparse_lineage"]["passed"] is False

    lineage = {
        "status": "accepted_sparse_candidate",
        "sparse_gate_passed": True,
        "accepted_sparse_model_sha256": "a" * 64,
        "accepted_sparse_gate_sha256": "b" * 64,
    }
    accepted = audit_final_tile_configs(**kwargs, sparse_lineage=lineage)
    assert accepted["status"] == "passed"
    assert accepted["sparse_lineage"]["passed"] is True
    assert accepted["observed"]["exact_one_reference_write"] is True
    assert accepted["observed"]["source_only_reference_count"] == 0


def test_image_bottom_escape_does_not_mislabel_upper_points_as_webbing():
    result = _classify_bottom_escape(
        heights=[0.8, 0.7, -1.0, -0.9],
        image_bottom_outside_counts=[3, 1, 0, 2],
        low_quantile=0.5,
    )
    assert result["image_bottom_escape_fraction"] == 0.75
    assert result["geometry_local_webbing_fraction"] == 0.25
    assert result["geometry_local_webbing"].tolist() == [False, False, False, True]


def test_geometry_local_webbing_requires_two_outside_views():
    result = _classify_bottom_escape(
        heights=[-1.0, -0.9, -0.8, 0.5],
        image_bottom_outside_counts=[1, 2, 3, 4],
        low_quantile=0.75,
    )
    assert result["geometry_local_webbing"].tolist() == [False, True, True, False]
    assert result["geometry_local_webbing_fraction"] == 0.5
