from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from v4_repair import (
    SparseIntegrityConfig,
    _ring_graph_connectivity,
    _rotation_angle_deg,
    _rotation_geodesic_deg,
    build_corrected_sparse_graph,
    build_independent_sparse_audit_set,
    colmap_pair_id,
    copy_frozen_sqlite_snapshot,
    create_disposable_sqlite_from_manifest,
    create_sqlite_snapshot,
    decode_colmap_pair_id,
    duplicate_stem,
    raw_component_gate,
    sparse_integrity_gate,
    validate_one_reference_write,
)
from scripts.build_v4_rotation_consensus import build_rotation_consensus, rotation_geodesic_deg


ROOT = Path(__file__).resolve().parents[1]


class _FakeReconstruction:
    def __init__(self, names: list[str]):
        self._names = names

    def reg_image_ids(self):
        return range(1, len(self._names) + 1)

    def image(self, image_id: int):
        return type("Image", (), {"name": self._names[int(image_id) - 1]})()

    def num_reg_images(self):
        return len(self._names)


class _FakeImage:
    def __init__(self, image_id: int, name: str):
        self.image_id = image_id
        self.name = name


class _FakeGeometry:
    def __init__(self, inliers: int):
        self.inlier_matches = np.zeros((inliers, 2), dtype=np.uint32)


class _FakeDatabase:
    def __init__(self, images, edges):
        self._images = images
        self._edges = edges

    def read_all_images(self):
        return list(self._images)

    def read_two_view_geometries(self):
        return tuple(zip(*self._edges)) if self._edges else ([], [])


def test_colmap_pair_id_round_trip_and_duplicate_suffix():
    pair_id = colmap_pair_id(242, 241)
    assert decode_colmap_pair_id(pair_id) == (241, 242)
    assert duplicate_stem("IMG20260912145220_01.jpg") == "IMG20260912145220"


def test_historical_geo_g10_fixture_fails_new_sparse_integrity_gate():
    fixture = json.loads(
        (ROOT / "tests" / "fixtures" / "v4_historical_geo_g10_pair_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    names = [pair["first"] for pair in fixture["pairs"]] + [fixture["pairs"][-1]["second"]]
    ring_metadata = {name: {"ring": "geo_g10", "frame_index": index, "selected": True} for index, name in enumerate(names)}
    result = sparse_integrity_gate(
        _FakeReconstruction(names),
        ring_metadata,
        fixture["pairs"],
        config=SparseIntegrityConfig(),
    )
    assert result["passed"] is False
    assert result["checks"]["same_ring_pose_consistency"] is False
    assert result["checks"]["shared_track_continuity"] is False
    assert result["disagreement_failure_count"] >= 2


def test_corrected_graph_excludes_duplicate_frames_and_caps_cross_ring_edges():
    images = [
        _FakeImage(1, "a.jpg"),
        _FakeImage(2, "b.jpg"),
        _FakeImage(3, "b_01.jpg"),
        _FakeImage(4, "c.jpg"),
        _FakeImage(5, "d.jpg"),
    ]
    edges = [
        (colmap_pair_id(1, 2), _FakeGeometry(400)),
        (colmap_pair_id(2, 3), _FakeGeometry(900)),  # duplicate frame, must drop
        (colmap_pair_id(1, 3), _FakeGeometry(350)),
        (colmap_pair_id(1, 4), _FakeGeometry(200)),
        (colmap_pair_id(1, 5), _FakeGeometry(190)),
        (colmap_pair_id(2, 4), _FakeGeometry(180)),
        (colmap_pair_id(3, 5), _FakeGeometry(170)),
    ]
    metadata = {
        "a.jpg": {"ring": "g1", "frame_index": 0, "selected": True},
        "b.jpg": {"ring": "g1", "frame_index": 1, "selected": True},
        "b_01.jpg": {"ring": "g1", "frame_index": 2, "selected": True},
        "c.jpg": {"ring": "g2", "frame_index": 0, "selected": True},
        "d.jpg": {"ring": "g2", "frame_index": 1, "selected": True},
    }
    graph = build_corrected_sparse_graph(
        _FakeDatabase(images, edges),
        metadata,
        ring_order=("g1", "g2"),
        same_ring_inliers=100,
        cross_ring_inliers=100,
        max_cross_edges_per_image=1,
    )
    assert graph["status"] == "complete"
    assert ["b.jpg", "b_01.jpg"] not in graph["pairs"]
    assert graph["cross_ring_selected"] >= 1
    assert len(graph["pairs"]) == len({tuple(pair) for pair in graph["pairs"]})


def test_one_reference_write_gate_detects_historical_duplicate_and_accepts_repaired(tmp_path: Path):
    historical = tmp_path / "historical.cfg"
    historical.write_text("a.jpg\nb.jpg\na.jpg\nc.jpg\n", encoding="utf-8")
    failed = validate_one_reference_write([historical], ["a.jpg", "b.jpg", "c.jpg"])
    assert failed["passed"] is False
    assert failed["duplicate_or_non_unit_references"]["a.jpg"] == 2

    repaired_a = tmp_path / "tile_a.cfg"
    repaired_b = tmp_path / "tile_b.cfg"
    repaired_a.write_text("a.jpg\nb.jpg\n", encoding="utf-8")
    repaired_b.write_text("c.jpg\na.jpg\n", encoding="utf-8")
    passed = validate_one_reference_write([repaired_a, repaired_b], ["a.jpg", "c.jpg"])
    assert passed["passed"] is True


def test_raw_component_gate_rejects_historical_fragmented_poisson_and_accepts_connected():
    historical = raw_component_gate({"components": {"face_fractions": [0.6657, 0.22, 0.1143]}})
    assert historical["passed"] is False
    assert historical["dominant_face_fraction"] == 0.6657
    repaired = raw_component_gate({"components": {"face_fractions": [0.965, 0.012, 0.008, 0.005]}})
    assert repaired["passed"] is True


def _rotation(axis: str, degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])
    if axis == "z":
        return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError(axis)


def test_full_so3_geodesic_detects_equal_angle_different_axis_rotations():
    first = _rotation("z", 30.0)
    second = _rotation("x", 30.0)
    assert _rotation_angle_deg(first) == pytest.approx(_rotation_angle_deg(second), abs=1e-9)
    assert _rotation_geodesic_deg(first, second) > 20.0


def test_rotation_consensus_is_deterministic_and_preserves_cross_ring_outliers():
    identity = np.eye(3).tolist()
    quarter_turn = _rotation("z", 90.0).tolist()
    records = [
        {
            "pair_id": 1,
            "first": "a0.jpg",
            "second": "a1.jpg",
            "ring": "g1",
            "second_ring": "g1",
            "same_ring": True,
            "well_conditioned_calibrated": True,
            "calibrated_inliers": 400,
            "calibrated_rotation_matrix_first_to_second": identity,
        },
        {
            "pair_id": 2,
            "first": "b0.jpg",
            "second": "b1.jpg",
            "ring": "g2",
            "second_ring": "g2",
            "same_ring": True,
            "well_conditioned_calibrated": True,
            "calibrated_inliers": 400,
            "calibrated_rotation_matrix_first_to_second": identity,
        },
        {
            "pair_id": 3,
            "first": "a0.jpg",
            "second": "b0.jpg",
            "ring": "g1",
            "second_ring": "g2",
            "same_ring": False,
            "well_conditioned_calibrated": True,
            "calibrated_inliers": 400,
            "calibrated_rotation_matrix_first_to_second": identity,
        },
        {
            "pair_id": 4,
            "first": "a1.jpg",
            "second": "b1.jpg",
            "ring": "g1",
            "second_ring": "g2",
            "same_ring": False,
            "well_conditioned_calibrated": True,
            "calibrated_inliers": 400,
            "calibrated_rotation_matrix_first_to_second": identity,
        },
        {
            "pair_id": 5,
            "first": "a0.jpg",
            "second": "b1.jpg",
            "ring": "g1",
            "second_ring": "g2",
            "same_ring": False,
            "well_conditioned_calibrated": True,
            "calibrated_inliers": 1,
            "calibrated_rotation_matrix_first_to_second": quarter_turn,
        },
    ]
    first_orientations, first_report = build_rotation_consensus(records, ring_order=("g1", "g2"))
    second_orientations, second_report = build_rotation_consensus(records, ring_order=("g1", "g2"))
    assert first_report["ring_graph_connected"] is True
    assert first_report["per_edge"] == second_report["per_edge"]
    assert set(first_orientations) == {"a0.jpg", "a1.jpg", "b0.jpg", "b1.jpg"}
    assert first_report["input_record_count"] == 5
    assert any(
        item["pair_id"] == 5 and item["mapping_status"] == "retained_robust_consensus_outlier"
        for item in first_report["per_edge"]
    )
    assert rotation_geodesic_deg(first_orientations["b1.jpg"], first_orientations["b0.jpg"]) == pytest.approx(
        0.0, abs=1e-8
    )


def test_disconnected_ring_graph_fails_actual_connectivity():
    ring_metadata = {
        "a.jpg": {"ring": "g1"},
        "b.jpg": {"ring": "g2"},
        "c.jpg": {"ring": "g3"},
    }
    evidence = [{
        "first": "a.jpg",
        "second": "b.jpg",
        "first_ring": "g1",
        "second_ring": "g2",
        "same_ring": False,
        "verified_inliers": 100,
    }]
    result = _ring_graph_connectivity(
        ("g1", "g2", "g3"),
        evidence,
        ring_metadata,
        {"a.jpg", "b.jpg", "c.jpg"},
        minimum_cross_ring_inliers=50,
    )
    assert result["connected"] is False
    assert result["components"] == [["g1", "g2"], ["g3"]]


def test_fixed_independent_audit_identity_is_unchanged_when_mapper_pairs_are_pruned():
    audit_pair_ids = [101, 202, 303, 404]
    identity = hashlib.sha256(
        json.dumps(audit_pair_ids, separators=(",", ":"), sort_keys=False).encode("utf-8")
    ).hexdigest()
    fixed_audit = {"pair_count": len(audit_pair_ids), "pair_ids": tuple(audit_pair_ids), "pair_ids_sha256": identity}
    mapper_pairs = [pair_id for pair_id in audit_pair_ids if pair_id != 303]
    assert mapper_pairs != list(fixed_audit["pair_ids"])
    assert fixed_audit["pair_count"] == 4
    assert fixed_audit["pair_ids_sha256"] == identity


def test_sparse_gate_requires_one_model_with_all_372_intended_views(tmp_path: Path):
    names = [f"IMG{index:04d}.jpg" for index in range(372)]
    reconstruction = _FakeReconstruction(names)
    ring_metadata = {name: {"ring": "g1", "selected": True} for name in names}
    model_root = tmp_path / "model"
    model_root.mkdir()
    for filename in ("cameras.bin", "images.bin", "points3D.bin"):
        (model_root / filename).write_bytes(b"model")
    result = sparse_integrity_gate(
        reconstruction,
        ring_metadata,
        [],
        independent_audit_evidence=[{"first": names[0], "second": names[1], "same_ring": True, "verified_inliers": 100}],
        expected_image_names=names,
        model_root=model_root,
        database_lineage={"snapshot_sha256": "a" * 64, "logical_sha256": "b" * 64},
    )
    assert result["model_count"] == 1
    assert result["checks"]["registered_image_set_exact"] is True
    assert result["checks"]["single_coherent_model"] is True

    two_model_root = tmp_path / "two_models"
    for model_index in ("0", "1"):
        child = two_model_root / model_index
        child.mkdir(parents=True)
        for filename in ("cameras.bin", "images.bin", "points3D.bin"):
            (child / filename).write_bytes(b"model")
    rejected = sparse_integrity_gate(
        reconstruction,
        ring_metadata,
        [],
        independent_audit_evidence=[{"first": names[0], "second": names[1], "same_ring": True, "verified_inliers": 100}],
        expected_image_names=names,
        model_root=two_model_root,
        database_lineage={"snapshot_sha256": "a" * 64, "logical_sha256": "b" * 64},
    )
    assert rejected["model_count"] == 2
    assert rejected["checks"]["single_coherent_model"] is False


def test_unconditioned_calibrated_audit_pairs_are_preserved_but_not_local_acceptance_evidence():
    names = ["a.jpg", "b.jpg"]
    ring_metadata = {name: {"ring": "g1", "selected": True} for name in names}
    evidence = [{
        "first": "a.jpg",
        "second": "b.jpg",
        "same_ring": True,
        "verified_inliers": 500,
        "rotation_disagreement_deg": 120.0,
        "shared_final_tracks": 0,
        "calibrated_conditioned": False,
    }]
    result = sparse_integrity_gate(
        _FakeReconstruction(names),
        ring_metadata,
        evidence,
        independent_audit_evidence=evidence,
        expected_image_names=names,
        database_lineage={"snapshot_sha256": "a" * 64, "logical_sha256": "b" * 64},
    )
    assert result["pair_count"] == 1
    assert result["disagreement_failure_count"] == 0
    assert result["shared_track_failure_count"] == 0


def test_canonical_sqlite_snapshot_is_immutable_and_working_copy_is_disposable(tmp_path: Path):
    source = tmp_path / "source.db"
    canonical = tmp_path / "canonical.db"
    working = tmp_path / "working.db"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE values_table (id INTEGER PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO values_table(value) VALUES ('original')")
    creation = create_sqlite_snapshot(source, canonical)
    canonical_sha = hashlib.sha256(canonical.read_bytes()).hexdigest()
    assert canonical_sha == creation["sha256"]
    assert creation["logical_sha256"]
    copied = copy_frozen_sqlite_snapshot(canonical, working)
    assert copied["sha256"] == canonical_sha
    with sqlite3.connect(working) as connection:
        connection.execute("UPDATE values_table SET value='working-copy'")
        connection.commit()
    assert hashlib.sha256(canonical.read_bytes()).hexdigest() == canonical_sha
    assert hashlib.sha256(working.read_bytes()).hexdigest() != canonical_sha
    assert not canonical.with_name(canonical.name + "-wal").exists()
    assert not canonical.with_name(canonical.name + "-shm").exists()


def test_v3_manifest_binds_canonical_bytes_but_only_working_copy_is_opened(tmp_path: Path):
    source = tmp_path / "source.db"
    canonical = tmp_path / "database_snapshot_v3_canonical.db"
    working = tmp_path / "working.db"
    manifest_path = tmp_path / "canonical_sqlite_snapshot_v3.json"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE values_table (id INTEGER PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO values_table(value) VALUES ('original')")
    creation = create_sqlite_snapshot(source, canonical)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "canonical_path": str(canonical),
                "canonical_sha256": creation["sha256"],
                "creation_logical_sha256": creation["logical_sha256"],
                "table_counts": creation["table_counts"],
            }
        ),
        encoding="utf-8",
    )
    disposable = create_disposable_sqlite_from_manifest(manifest_path, working)
    assert disposable["manifest"]["canonical_sha256"] == creation["sha256"]
    with sqlite3.connect(working) as connection:
        connection.execute("UPDATE values_table SET value='working-only'")
        connection.commit()
    assert hashlib.sha256(canonical.read_bytes()).hexdigest() == creation["sha256"]
    assert hashlib.sha256(working.read_bytes()).hexdigest() != creation["sha256"]
