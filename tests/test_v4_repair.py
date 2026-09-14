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
    prune_colmap_database,
    raw_component_gate,
    sparse_camera_center_translation_evidence,
    sparse_integrity_gate,
    sparse_mask_projection_evidence,
    sparse_track_distribution_evidence,
    validate_exact_mapper_graph,
    validate_one_reference_write,
)
from scripts.build_v4_rotation_consensus import (
    build_ring_aligned_repair,
    build_rotation_consensus,
    build_source_component_pose_repair,
    rotation_geodesic_deg,
)
from scripts.build_v4_mapper_graph_from_consensus import build_mapper_graph
from scripts.build_v4_conflict_free_graph import _select_conflict_free_edges
from scripts.build_v4_match_conflict_free_database import select_conflict_free_match_rows
from scripts.audit_v4_sparse_pair_geometry import radians_to_degrees as audit_radians_to_degrees
from scripts.reestimate_v4_two_view_geometry import (
    _pose_override_payload,
    radians_to_degrees as remap_radians_to_degrees,
)


ROOT = Path(__file__).resolve().parents[1]


def test_conflict_free_graph_rejects_same_image_observation_cycle() -> None:
    """A graph cycle that maps two keypoints in one image must be rejected."""

    def record(pair_id: int, first: str, second: str, frame_distance: int) -> dict[str, object]:
        return {
            "pair_id": pair_id,
            "first": first,
            "second": second,
            "same_ring": True,
            "ring": "ring",
            "second_ring": "ring",
            "local_cycle_status": "consistent",
            "frame_distance": frame_distance,
            "calibrated_inliers": 100,
            "calibrated_tri_angle_deg": 2.0,
            "homography_inlier_fraction": 0.1,
        }

    edges = [
        (colmap_pair_id(1, 2), record(colmap_pair_id(1, 2), "a", "b", 1), np.array([[0, 0]], dtype=np.uint32)),
        (colmap_pair_id(2, 3), record(colmap_pair_id(2, 3), "b", "c", 1), np.array([[0, 0]], dtype=np.uint32)),
        # The third edge connects image 1/keypoint 1 to image 3/keypoint 0,
        # whose component already contains image 1/keypoint 0.
        (colmap_pair_id(1, 3), record(colmap_pair_id(1, 3), "a", "c", 2), np.array([[1, 0]], dtype=np.uint32)),
    ]

    result = _select_conflict_free_edges(edges)

    assert len(result["accepted"]) == 2
    assert len(result["rejected"]) == 1
    assert result["rejected"][0]["reason"] == "observation_component_image_conflict"
    assert result["observation_duplicate_component_count"] == 0


def test_conflict_free_graph_consensus_inlier_first_changes_conflicting_edge_order() -> None:
    """Independent consensus status must outrank raw inlier count when edges conflict."""

    def record(pair_id: int, first: str, second: str, inliers: int, status: str = "") -> dict[str, object]:
        return {
            "pair_id": pair_id,
            "first": first,
            "second": second,
            "same_ring": True,
            "local_cycle_status": "consistent",
            "frame_distance": 1,
            "calibrated_inliers": inliers,
            "calibrated_tri_angle_deg": 2.0,
            "homography_inlier_fraction": 0.1,
            "_consensus_status": status,
        }

    ab = colmap_pair_id(1, 2)
    cd = colmap_pair_id(3, 4)
    ac = colmap_pair_id(1, 3)
    bd = colmap_pair_id(2, 4)
    edges = [
        (ab, record(ab, "a", "b", 2000), np.array([[0, 0]], dtype=np.uint32)),
        (cd, record(cd, "c", "d", 2000), np.array([[0, 0]], dtype=np.uint32)),
        # This high-support consensus outlier conflicts with bd after ab/cd.
        (ac, record(ac, "a", "c", 1000, "retained_robust_consensus_outlier"), np.array([[1, 0]], dtype=np.uint32)),
        # This lower-support independent consensus inlier must win in the
        # versioned policy and cause ac to be rejected instead.
        (bd, record(bd, "b", "d", 10, "retained_robust_consensus_inlier"), np.array([[0, 0]], dtype=np.uint32)),
    ]

    default = _select_conflict_free_edges(edges)
    consensus_first = _select_conflict_free_edges(edges, consensus_inlier_first=True)

    assert {item["pair_id"] for item in default["accepted"]} >= {ab, cd, ac}
    assert any(item["pair_id"] == bd for item in default["rejected"])
    assert {item["pair_id"] for item in consensus_first["accepted"]} >= {ab, cd, bd}
    assert any(item["pair_id"] == ac for item in consensus_first["rejected"])


def test_match_level_conflict_filter_preserves_every_pair_id() -> None:
    """Conflicting rows are pruned without deleting the verified pair edge."""

    def record(pair_id: int, first: str, second: str, inliers: int = 100) -> dict[str, object]:
        return {
            "pair_id": pair_id,
            "first": first,
            "second": second,
            "same_ring": True,
            "local_cycle_status": "consistent",
            "frame_distance": 1,
            "calibrated_inliers": inliers,
            "calibrated_tri_angle_deg": 10.0,
            "homography_inlier_fraction": 0.1,
        }

    image_ids = {"a": 1, "b": 2, "c": 3}
    ab = colmap_pair_id(1, 2)
    bc = colmap_pair_id(2, 3)
    ac = colmap_pair_id(1, 3)
    edges = [
        (ab, record(ab, "a", "b"), np.asarray([[0, 0], [1, 1]], dtype=np.uint32)),
        (bc, record(bc, "b", "c"), np.asarray([[0, 0], [1, 1]], dtype=np.uint32)),
        # The first two rows close an inconsistent cycle and must be pruned;
        # the final row proves the pair itself is retained.
        (ac, record(ac, "a", "c", inliers=1), np.asarray([[1, 0], [0, 1], [2, 2]], dtype=np.uint32)),
    ]

    result = select_conflict_free_match_rows(
        image_ids,
        edges,
        minimum_matches_per_pair=1,
    )

    assert set(result["retained_matches"]) == {ab, bc, ac}
    assert result["pair_count"] == 3
    assert result["pruned_inlier_rows"] > 0
    assert all(len(rows) >= 1 for rows in result["retained_matches"].values())


def test_match_level_conflict_filter_prioritizes_consensus_inlier_cross_ring_rows() -> None:
    """Cross-ring consensus inliers win only after same-ring evidence is ordered."""

    def record(pair_id: int, first: str, second: str, same_ring: bool, inliers: int) -> dict[str, object]:
        return {
            "pair_id": pair_id,
            "first": first,
            "second": second,
            "same_ring": same_ring,
            "local_cycle_status": "consistent",
            "frame_distance": 1,
            "calibrated_inliers": inliers,
            "calibrated_tri_angle_deg": 10.0,
            "homography_inlier_fraction": 0.1,
        }

    image_ids = {"a": 1, "b": 2, "c": 3, "d": 4}
    ab = colmap_pair_id(1, 2)
    cd = colmap_pair_id(3, 4)
    ac = colmap_pair_id(1, 3)
    bd = colmap_pair_id(2, 4)
    edges = [
        (ab, record(ab, "a", "b", True, 100), np.asarray([[0, 0]], dtype=np.uint32)),
        (cd, record(cd, "c", "d", True, 100), np.asarray([[0, 0]], dtype=np.uint32)),
        # The first row conflicts after the consensus inlier is retained; the
        # second row is an independently usable fallback for the same pair.
        (ac, record(ac, "a", "c", False, 1000), np.asarray([[1, 0], [2, 1]], dtype=np.uint32)),
        (bd, record(bd, "b", "d", False, 10), np.asarray([[0, 0]], dtype=np.uint32)),
    ]

    result = select_conflict_free_match_rows(
        image_ids,
        edges,
        consensus_status={ac: "retained_robust_consensus_outlier", bd: "retained_robust_consensus_inlier"},
        minimum_matches_per_pair=1,
    )

    assert set(result["retained_matches"]) == {ab, cd, ac, bd}
    per_pair = {int(item["pair_id"]): item for item in result["per_pair"]}
    assert per_pair[bd]["retained_inlier_rows"] == 1
    assert per_pair[ac]["retained_inlier_rows"] == 1
    np.testing.assert_array_equal(result["retained_matches"][ac], np.asarray([[2, 1]], dtype=np.uint32))


def test_match_level_conflict_filter_caps_only_consensus_outlier_support() -> None:
    """The bounded weight control keeps the pair but limits outlier rows."""

    pair_id = colmap_pair_id(1, 2)
    record = {
        "pair_id": pair_id,
        "first": "a",
        "second": "b",
        "same_ring": False,
        "ring": "g7_r1",
        "second_ring": "geo_g8",
        "local_cycle_status": "consistent",
        "frame_distance": 1,
        "calibrated_inliers": 100,
        "calibrated_tri_angle_deg": 10.0,
        "homography_inlier_fraction": 0.1,
    }
    result = select_conflict_free_match_rows(
        {"a": 1, "b": 2},
        [(pair_id, record, np.asarray([[0, 0], [1, 1], [2, 2]], dtype=np.uint32))],
        consensus_status={pair_id: "retained_robust_consensus_outlier"},
        minimum_matches_per_pair=1,
        consensus_outlier_max_rows_by_transition={"g7_r1:geo_g8": 1},
    )

    assert len(result["retained_matches"][pair_id]) == 1
    assert result["capped_rows"] == 2
    assert result["per_pair"][0]["capped_by_consensus_outlier_limit"] is True


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
        self.camera_id = 1


class _FakeRotation:
    def matrix(self):
        return np.eye(3, dtype=np.float64)


class _FakePose:
    def __init__(self, center):
        self.rotation = _FakeRotation()
        self.translation = -np.asarray(center, dtype=np.float64)


class _FakeProjectedImage(_FakeImage):
    def __init__(self, image_id: int, name: str, center=(0.0, 0.0, 0.0)):
        super().__init__(image_id, name)
        self._center = np.asarray(center, dtype=np.float64)
        self._pose = _FakePose(self._center)

    def cam_from_world(self):
        return self._pose

    def projection_center(self):
        return self._center


class _FakeCamera:
    width = 100
    height = 100

    def img_from_cam(self, points):
        points = np.asarray(points, dtype=np.float64)
        return np.column_stack((points[:, 0] * 50.0 + 50.0, points[:, 1] * 50.0 + 50.0))


class _FakeTrackElement:
    def __init__(self, image_id: int):
        self.image_id = image_id


class _FakePoint:
    def __init__(self, image_ids, xyz=(0.0, 0.0, 1.0)):
        self.track = type("Track", (), {"elements": [_FakeTrackElement(value) for value in image_ids]})()
        self.xyz = np.asarray(xyz, dtype=np.float64)


class _FakeProjectedReconstruction:
    def __init__(self, images, points):
        self._images = {image.image_id: image for image in images}
        self.points3D = {index + 1: point for index, point in enumerate(points)}
        self.cameras = {1: _FakeCamera()}

    def reg_image_ids(self):
        return tuple(sorted(self._images))

    def image(self, image_id):
        return self._images[int(image_id)]

    def find_image_with_name(self, name):
        return next(image for image in self._images.values() if image.name == name)

    def camera(self, camera_id):
        return self.cameras[int(camera_id)]



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


def test_pycolmap_angle_units_are_explicitly_converted_from_radians():
    assert audit_radians_to_degrees(0.05) == pytest.approx(2.864788975654116, abs=1e-12)
    assert remap_radians_to_degrees(np.deg2rad(30.0)) == pytest.approx(30.0, abs=1e-12)


def test_pose_override_binds_audited_translation_and_round_trips_all_geometry():
    import pycolmap

    camera = pycolmap.Camera(
        model="PINHOLE",
        width=100,
        height=100,
        params=[100.0, 100.0, 50.0, 50.0],
    )
    payload = pycolmap.TwoViewGeometry().todict(recursive=False)
    payload.update(
        config=2,
        E=np.zeros((3, 3), dtype=np.float64),
        F=np.zeros((3, 3), dtype=np.float64),
        cam2_from_cam1=pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.array([1.0, 0.0, 0.0])),
        inlier_matches=np.zeros((5, 2), dtype=np.uint32),
        tri_angle=0.1,
    )
    rewritten, angle_deg, round_trip = _pose_override_payload(
        None,
        payload,
        camera,
        camera,
        _rotation("z", 20.0),
        np.array([0.0, 1.0, 0.0]),
    )
    assert angle_deg == pytest.approx(20.0, abs=1e-9)
    assert round_trip["passed"] is True
    assert np.asarray(rewritten.cam2_from_cam1.translation) == pytest.approx([0.0, 1.0, 0.0])


def test_calibrated_ring_repair_uses_largest_reference_gauge_cluster():
    true_rotations = {
        "a.jpg": _rotation("z", 0.0),
        "b.jpg": _rotation("z", 10.0),
        "c.jpg": _rotation("z", 20.0),
        "d.jpg": _rotation("z", 30.0),
    }
    records = []
    names = list(true_rotations)
    for index, (first, second) in enumerate(zip(names, names[1:])):
        records.append({
            "pair_id": index + 1,
            "first": first,
            "second": second,
            "ring": "ring_a",
            "second_ring": "ring_a",
            "same_ring": True,
            "well_conditioned_calibrated": True,
            "calibrated_inliers": 500,
            "calibrated_rotation_matrix_first_to_second": (
                true_rotations[second] @ true_rotations[first].T
            ).tolist(),
        })
    reference = {name: matrix.copy() for name, matrix in true_rotations.items()}
    reference["a.jpg"] = _rotation("x", 90.0)
    repaired, report = build_ring_aligned_repair(
        records,
        reference,
        ring_order=("ring_a",),
        gauge_threshold_deg=15.0,
    )
    assert report["same_ring_reports"][0]["gauge_support_count"] == 3
    assert report["same_ring_reports"][0]["gauge_status"] == "supported"
    for first, second in zip(names, names[1:]):
        expected = true_rotations[second] @ true_rotations[first].T
        assert rotation_geodesic_deg(repaired[second] @ repaired[first].T, expected) < 1e-6


def test_source_component_pose_repair_corrects_island_without_ring_drift():
    rotations = {
        "a0.jpg": _rotation("z", 0.0),
        "a1.jpg": _rotation("z", 10.0),
        # The source pose for a2 is an 80-degree island jump.
        "a2.jpg": _rotation("z", 100.0),
    }
    calibrated = {
        "a0.jpg": rotations["a0.jpg"],
        "a1.jpg": rotations["a1.jpg"],
        "a2.jpg": _rotation("z", 20.0),
    }
    records = [
        {
            "pair_id": 1,
            "first": "a0.jpg",
            "second": "a1.jpg",
            "ring": "ring_a",
            "second_ring": "ring_a",
            "same_ring": True,
            "well_conditioned_calibrated": True,
            "calibrated_inliers": 500,
            "calibrated_rotation_matrix_first_to_second": (
                calibrated["a1.jpg"] @ calibrated["a0.jpg"].T
            ).tolist(),
        },
        {
            "pair_id": 2,
            "first": "a1.jpg",
            "second": "a2.jpg",
            "ring": "ring_a",
            "second_ring": "ring_a",
            "same_ring": True,
            "well_conditioned_calibrated": True,
            "calibrated_inliers": 500,
            "calibrated_rotation_matrix_first_to_second": (
                calibrated["a2.jpg"] @ calibrated["a1.jpg"].T
            ).tolist(),
        },
    ]
    repaired, report = build_source_component_pose_repair(
        records,
        rotations,
        ring_order=("ring_a",),
    )
    assert report["component_count"] == 2
    assert report["assigned_component_count"] == 2
    assert report["ring_reports"][0]["corrected_residual_over_threshold_count"] == 0
    assert rotation_geodesic_deg(repaired["a0.jpg"], rotations["a0.jpg"]) == pytest.approx(0.0, abs=1e-8)
    assert rotation_geodesic_deg(repaired["a1.jpg"], rotations["a1.jpg"]) == pytest.approx(0.0, abs=1e-8)
    assert rotation_geodesic_deg(repaired["a2.jpg"], calibrated["a2.jpg"]) < 1e-5


def test_pair_local_two_view_tracks_are_not_independent_global_continuity():
    pairwise = _FakeProjectedReconstruction(
        [_FakeProjectedImage(1, "a.jpg"), _FakeProjectedImage(2, "b.jpg")],
        [_FakePoint((1, 2)) for _ in range(8)],
    )
    evidence = sparse_track_distribution_evidence(
        pairwise,
        provenance={
            "source": "verified_sqlite_graph",
            "independent_graph": True,
            "independent_graph_sha256": "a" * 64,
            "constructed_pairwise_tracks": True,
        },
    )
    assert evidence["fraction_track_length_eq2"] == 1.0
    assert evidence["checks"]["pair_local_track_partition_rejected"] is False
    assert evidence["passed"] is False


def test_global_track_distribution_requires_genuine_three_view_support():
    global_reconstruction = _FakeProjectedReconstruction(
        [_FakeProjectedImage(1, "a.jpg"), _FakeProjectedImage(2, "b.jpg"), _FakeProjectedImage(3, "c.jpg")],
        [_FakePoint((1, 2, 3, 4)) for _ in range(9)],
    )
    evidence = sparse_track_distribution_evidence(
        global_reconstruction,
        provenance={
            "source": "verified_sqlite_graph",
            "independent_graph": True,
            "independent_graph_sha256": "b" * 64,
            "constructed_pairwise_tracks": False,
        },
    )
    assert evidence["checks"]["multi_view_track_fraction"] is True
    assert evidence["checks"]["track_length_p50"] is True
    assert evidence["passed"] is True


def test_sparse_mask_projection_evidence_is_global_and_per_ring(tmp_path: Path):
    import cv2

    name = "a.jpg"
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:76, 25:76] = 255
    mask_path = tmp_path / "a.png"
    assert cv2.imwrite(str(mask_path), mask)
    points = [
        _FakePoint((1,), xyz=((x - 50) / 50.0, (y - 50) / 50.0, 1.0))
        for x in range(35, 66, 5)
        for y in range(35, 66, 5)
    ]
    reconstruction = _FakeProjectedReconstruction([_FakeProjectedImage(1, name)], points)
    evidence = sparse_mask_projection_evidence(
        reconstruction,
        tmp_path,
        {name: {"ring": "g1", "selected": True}},
    )
    assert evidence["checks"]["all_registered_views_measured"] is True
    assert evidence["checks"]["all_selected_rings_measured"] is True
    assert evidence["checks"]["view_precision_p10"] is True
    assert evidence["passed"] is True


def test_calibrated_translation_checks_candidate_center_direction_and_scale():
    reconstruction = _FakeProjectedReconstruction(
        [_FakeProjectedImage(1, "a.jpg", (0.0, 0.0, 0.0)), _FakeProjectedImage(2, "b.jpg", (1.0, 0.0, 0.0))],
        [],
    )
    evidence = sparse_camera_center_translation_evidence(
        reconstruction,
        [{
            "pair_id": 1,
            "first": "a.jpg",
            "second": "b.jpg",
            "same_ring": True,
            "well_conditioned_calibrated": True,
            "calibrated_translation_first_to_second": [-1.0, 0.0, 0.0],
        }],
    )
    assert evidence["checks"]["direction_error_p90"] is True
    assert evidence["checks"]["positive_projected_scale_fraction"] is True
    assert evidence["checks"]["orthogonal_residual_p90"] is True
    assert evidence["passed"] is True


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


def test_mapper_graph_keeps_full_audit_and_adds_minimum_calibrated_bridge(tmp_path: Path):
    identity = np.eye(3).tolist()
    translation = [1.0, 0.0, 0.0]
    pair_specs = [
        (1, "a0.jpg", "a1.jpg", "g1", "g1", "retained_rotation_synchronization", 300, 8.0, 0.10),
        (2, "b0.jpg", "b1.jpg", "g2", "g2", "retained_rotation_synchronization", 250, 7.0, 0.10),
        (3, "a0.jpg", "b0.jpg", "g1", "g2", "retained_robust_consensus_outlier", 400, 21.0, 0.20),
    ]
    classification_records = []
    mapping_records = []
    consensus_edges = []
    raw_records = []
    for pair_id, first, second, ring, second_ring, status, inliers, tri, homography in pair_specs:
        classification_records.append(
            {
                "pair_id": pair_id,
                "first": first,
                "second": second,
                "ring": ring,
                "second_ring": second_ring,
                "same_ring": ring == second_ring,
                "well_conditioned_calibrated": True,
                "calibrated_inliers": inliers,
                "calibrated_tri_angle_deg": tri,
                "calibrated_cheirality_fraction": 1.0,
                "homography_inlier_fraction": homography,
                "classification_reasons": [],
            }
        )
        mapping_records.append(
            {
                "disposition": "retained_for_mapping",
                "first": first,
                "second": second,
                "independent_audit_record": {
                    "calibrated_reestimate_rotation_matrix_first_to_second": identity,
                    "calibrated_reestimate_translation_first_to_second": translation,
                },
            }
        )
        consensus_edges.append(
            {
                "pair_id": pair_id,
                "first": first,
                "second": second,
                "mapping_status": status,
                "edge_type": "same_ring" if ring == second_ring else "cross_ring",
                "conditioned": True,
            }
        )
        raw_records.append({"pair_id": pair_id})

    raw_audit = {
        "schema_version": 2,
        "angle_units": {"triangulation_angle": "degrees", "rotation_angle": "degrees"},
        "records": raw_records,
    }
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(raw_audit), encoding="utf-8")
    fixed_ids_sha = hashlib.sha256(b"[1,2,3]").hexdigest()
    classification = {
        "schema_version": 2,
        "angle_units": {"triangulation_angle": "degrees", "rotation_angle": "degrees"},
        "classification": classification_records,
        "mapping_records": mapping_records,
        "fixed_pair_count": 3,
        "fixed_pair_ids_sha256": fixed_ids_sha,
    }
    classification_path = tmp_path / "classification.json"
    classification_path.write_text(json.dumps(classification), encoding="utf-8")
    consensus = {
        "classification_sha256": hashlib.sha256(classification_path.read_bytes()).hexdigest(),
        "per_edge": consensus_edges,
    }
    consensus_path = tmp_path / "consensus.json"
    consensus_path.write_text(json.dumps(consensus), encoding="utf-8")
    snapshot = {
        "canonical_sha256": "a" * 64,
        "creation_logical_sha256": "b" * 64,
    }
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    payload = build_mapper_graph(
        classification,
        consensus,
        classification_path=classification_path,
        consensus_path=consensus_path,
        raw_audit_path=raw_path,
        snapshot_manifest_path=snapshot_path,
        raw_audit=raw_audit,
        snapshot_manifest=snapshot,
    )
    assert payload["pair_count"] == 3
    assert payload["bridge_pair_count"] == 1
    assert payload["excluded_consensus_outlier_count"] == 0
    assert payload["fixed_audit_pair_count"] == 3
    assert payload["bridge_pairs"][0]["first"] == "a0.jpg"
    assert payload["bridge_pairs"][0]["second"] == "b0.jpg"


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


def test_mapper_preflight_rejects_extra_verified_pair_ids_before_mapping(tmp_path: Path):
    database = tmp_path / "graph.db"
    keep_pair = colmap_pair_id(1, 2)
    extra_pair = colmap_pair_id(2, 3)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE matches (pair_id INTEGER PRIMARY KEY, rows INTEGER NOT NULL)")
        connection.execute(
            "CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO matches(pair_id, rows) VALUES (?, ?)",
            [(keep_pair, 11), (extra_pair, 13)],
        )
        connection.executemany(
            "INSERT INTO two_view_geometries(pair_id, rows) VALUES (?, ?)",
            [(keep_pair, 7), (extra_pair, 5)],
        )

    preflight = validate_exact_mapper_graph(database, {keep_pair})
    assert preflight["passed"] is False
    assert preflight["nonempty_two_view_geometry_count"] == 2
    assert preflight["extra_two_view_geometry_pair_ids"] == [extra_pair]
    assert preflight["raw_match_pair_ids_exact"] is False


def test_mapper_preflight_rejects_equal_count_but_wrong_pair_id_set(tmp_path: Path):
    """A count-only check must not accept a graph with the wrong pair identity."""

    database = tmp_path / "wrong_identity.db"
    expected_pair = colmap_pair_id(1, 2)
    actual_pair = colmap_pair_id(1, 3)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE matches (pair_id INTEGER PRIMARY KEY, rows INTEGER NOT NULL)")
        connection.execute(
            "CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER NOT NULL)"
        )
        connection.execute("INSERT INTO matches(pair_id, rows) VALUES (?, ?)", (actual_pair, 9))
        connection.execute(
            "INSERT INTO two_view_geometries(pair_id, rows) VALUES (?, ?)", (actual_pair, 7)
        )

    preflight = validate_exact_mapper_graph(database, {expected_pair})

    assert preflight["passed"] is False
    assert preflight["nonempty_two_view_geometry_count"] == 1
    assert preflight["extra_two_view_geometry_pair_ids"] == [actual_pair]
    assert preflight["missing_two_view_geometry_pair_ids"] == [expected_pair]
    assert preflight["actual_two_view_geometry_pair_ids_sha256"] != preflight["expected_pair_ids_sha256"]
    assert preflight["raw_match_pair_ids_exact"] is False


def test_exact_mapper_pruning_removes_verified_and_raw_extra_pairs(tmp_path: Path):
    source = tmp_path / "source.db"
    canonical = tmp_path / "canonical.db"
    manifest_path = tmp_path / "manifest.json"
    pruned = tmp_path / "pruned.db"
    keep_pair = colmap_pair_id(1, 2)
    extra_pair = colmap_pair_id(2, 3)
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        connection.execute("CREATE TABLE matches (pair_id INTEGER PRIMARY KEY, rows INTEGER NOT NULL)")
        connection.execute(
            "CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO images(image_id, name) VALUES (?, ?)",
            [(1, "a.jpg"), (2, "b.jpg"), (3, "c.jpg")],
        )
        connection.executemany(
            "INSERT INTO matches(pair_id, rows) VALUES (?, ?)",
            [(keep_pair, 11), (extra_pair, 13)],
        )
        connection.executemany(
            "INSERT INTO two_view_geometries(pair_id, rows) VALUES (?, ?)",
            [(keep_pair, 7), (extra_pair, 5)],
        )
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
    prune_colmap_database(
        canonical,
        pruned,
        [("a.jpg", "b.jpg")],
        image_ids_by_name={"a.jpg": 1, "b.jpg": 2},
        snapshot_manifest=manifest_path,
    )
    preflight = validate_exact_mapper_graph(pruned, {keep_pair}, canonical_path=canonical)
    assert preflight["passed"] is True
    assert preflight["nonempty_two_view_geometry_count"] == 1
    assert preflight["actual_two_view_geometry_pair_ids_sha256"] == preflight["expected_pair_ids_sha256"]
    assert preflight["actual_raw_match_pair_ids_sha256"] == preflight["expected_pair_ids_sha256"]
    with sqlite3.connect(pruned) as connection:
        assert connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0] == 1
    assert hashlib.sha256(canonical.read_bytes()).hexdigest() == creation["sha256"]
