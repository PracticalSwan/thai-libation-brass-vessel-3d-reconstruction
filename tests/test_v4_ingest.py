from __future__ import annotations

from pathlib import Path

from PIL import Image

from v4_ingest import (
    EXPECTED_COUNTS,
    audited_role_for_filename,
    build_manifest,
    filename_order,
    role_totals,
)


def test_audited_role_ranges_match_the_fixed_688_contract():
    assert audited_role_for_filename("IMG20260912131808.jpg") == ("appearance_reference", "appearance")
    assert audited_role_for_filename("IMG20260912133652.jpg") == ("empty_board", "empty_g6")
    assert audited_role_for_filename("IMG20260912143718.jpg") == ("geometry", "geo_g8")
    assert audited_role_for_filename("IMG20260912143726.jpg") == ("empty_board", "empty_g8")
    assert audited_role_for_filename("IMG20260912150627_01.jpg") == ("geometry", "geo_g12")
    assert EXPECTED_COUNTS == {"total": 688, "appearance_reference": 158, "empty_board": 107, "geometry": 423}


def test_filename_order_keeps_same_second_suffix_after_base():
    assert filename_order("IMG20260912150627.jpg") < filename_order("IMG20260912150627_01.jpg")


def test_small_fixture_manifest_preserves_source_provenance_without_strict_688_gate(tmp_path: Path):
    source = tmp_path / "CSX4213_Project_V4_Images"
    source.mkdir()
    # Use two audited boundary names so the fixture exercises the same role
    # classifier without weakening the canonical 688-file gate.
    for name, color in (
        ("IMG20260912131808.jpg", (180, 120, 50)),
        ("IMG20260912133528.jpg", (140, 100, 60)),
    ):
        Image.new("RGB", (32, 24), color).save(source / name, format="JPEG")
    result = build_manifest(
        source,
        output_root=None,
        require_expected_count=False,
        require_expected_dimensions=False,
    )
    assert len(result["rows"]) == 2
    assert role_totals(result["rows"]) == {"appearance_reference": 1, "empty_board": 1}
    assert all(row["relative_path"] for row in result["rows"])
    assert all(len(row["sha256"]) == 64 for row in result["rows"])
