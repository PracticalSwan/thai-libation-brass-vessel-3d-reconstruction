from __future__ import annotations

from pathlib import Path

import pytest

import v4_dense
from v4_dense import (
    V4DenseConfig,
    build_dense_pair_adjacency,
    build_prioritized_dense_pair_adjacency,
    build_patch_match_command,
    write_dense_pair_config,
    write_prioritized_dense_pair_config,
    write_dense_smoke_config,
)


def test_patch_match_command_carries_explicit_geometric_switch_values(monkeypatch):
    monkeypatch.setattr(v4_dense, "_colmap_executable", lambda: "colmap.exe")
    command = build_patch_match_command(
        Path("dense"), config=V4DenseConfig(), allow_missing_files=True
    )
    assert command[0:2] == ["colmap.exe", "patch_match_stereo"]
    assert ["--PatchMatchStereo.geom_consistency", "1"] == command[
        command.index("--PatchMatchStereo.geom_consistency") : command.index("--PatchMatchStereo.geom_consistency") + 2
    ]
    assert ["--PatchMatchStereo.filter", "1"] == command[
        command.index("--PatchMatchStereo.filter") : command.index("--PatchMatchStereo.filter") + 2
    ]
    assert ["--PatchMatchStereo.allow_missing_files", "1"] == command[
        command.index("--PatchMatchStereo.allow_missing_files") : command.index("--PatchMatchStereo.allow_missing_files") + 2
    ]


def test_smoke_config_uses_only_cyclic_selected_dependencies(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(v4_dense, "assert_output_path", lambda path: Path(path))
    names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    path = write_dense_smoke_config(tmp_path / "patch-match.cfg", names, source_count=2)
    assert path.read_text(encoding="utf-8").splitlines() == [
        "a.jpg",
        "b.jpg,c.jpg",
        "b.jpg",
        "c.jpg,d.jpg",
        "c.jpg",
        "d.jpg,a.jpg",
        "d.jpg",
        "a.jpg,b.jpg",
    ]
    with pytest.raises(FileExistsError):
        write_dense_smoke_config(path, names)


def test_photometric_command_explicitly_disables_geometric_filtering(monkeypatch):
    monkeypatch.setattr(v4_dense, "_colmap_executable", lambda: "colmap.exe")
    command = build_patch_match_command(
        Path("dense"),
        config=V4DenseConfig(geom_consistency=False, filter=False),
    )
    assert command[command.index("--PatchMatchStereo.geom_consistency") + 1] == "0"
    assert command[command.index("--PatchMatchStereo.filter") + 1] == "0"


def test_pair_config_expands_schedule_bidirectionally_and_supports_chunks(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(v4_dense, "assert_output_path", lambda path: Path(path))
    names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    pairs = [["a.jpg", "c.jpg"], ["b.jpg", "a.jpg"], ["d.jpg", "c.jpg"]]
    assert build_dense_pair_adjacency(names, pairs) == {
        "a.jpg": ("b.jpg", "c.jpg"),
        "b.jpg": ("a.jpg",),
        "c.jpg": ("a.jpg", "d.jpg"),
        "d.jpg": ("c.jpg",),
    }
    assert build_dense_pair_adjacency(names, pairs, max_sources=1) == {
        "a.jpg": ("b.jpg",),
        "b.jpg": ("a.jpg",),
        "c.jpg": ("a.jpg",),
        "d.jpg": ("c.jpg",),
    }
    path = write_dense_pair_config(
        tmp_path / "patch-match.cfg",
        names,
        pairs,
        reference_names=["c.jpg", "a.jpg"],
    )
    assert path.read_text(encoding="utf-8").splitlines() == [
        "a.jpg",
        "b.jpg,c.jpg",
        "c.jpg",
        "a.jpg,d.jpg",
    ]


def test_prioritized_pair_config_reserves_local_and_cross_ring_support(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(v4_dense, "assert_output_path", lambda path: Path(path))
    names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    rings = {"a.jpg": "g1", "b.jpg": "g1", "c.jpg": "g2", "d.jpg": "g2"}
    phases = {name: index / 4 for index, name in enumerate(names)}
    pairs = [
        ["a.jpg", "b.jpg"],
        ["a.jpg", "c.jpg"],
        ["a.jpg", "d.jpg"],
        ["b.jpg", "c.jpg"],
        ["c.jpg", "d.jpg"],
    ]
    adjacency = build_prioritized_dense_pair_adjacency(
        names,
        pairs,
        ring_by_name=rings,
        phase_by_name=phases,
        max_sources=2,
    )
    for reference, sources in adjacency.items():
        assert len(sources) == 2
        assert any(rings[source] == rings[reference] for source in sources)
        assert any(rings[source] != rings[reference] for source in sources)
    path = write_prioritized_dense_pair_config(
        tmp_path / "prioritized.cfg",
        names,
        adjacency,
        reference_names=["a.jpg", "c.jpg"],
    )
    assert path.read_text(encoding="utf-8").splitlines() == [
        "a.jpg",
        "b.jpg,d.jpg",
        "c.jpg",
        "d.jpg,b.jpg",
    ]


def test_prioritized_pair_config_can_reserve_two_cross_ring_sources():
    names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"]
    rings = {"a.jpg": "g1", "b.jpg": "g1", "c.jpg": "g2", "d.jpg": "g2", "e.jpg": "g3"}
    pairs = [
        ["a.jpg", "b.jpg"],
        ["a.jpg", "c.jpg"],
        ["a.jpg", "d.jpg"],
        ["a.jpg", "e.jpg"],
        ["b.jpg", "c.jpg"],
        ["b.jpg", "d.jpg"],
        ["b.jpg", "e.jpg"],
        ["c.jpg", "d.jpg"],
        ["c.jpg", "e.jpg"],
        ["d.jpg", "e.jpg"],
    ]
    adjacency = build_prioritized_dense_pair_adjacency(
        names,
        pairs,
        ring_by_name=rings,
        max_sources=3,
        min_cross_sources=2,
    )
    for reference, sources in adjacency.items():
        assert len(sources) == 3
        assert sum(rings[source] != rings[reference] for source in sources) >= 2
        if any(rings[candidate] == rings[reference] for candidate in names if candidate != reference):
            assert any(rings[source] == rings[reference] for source in sources)
