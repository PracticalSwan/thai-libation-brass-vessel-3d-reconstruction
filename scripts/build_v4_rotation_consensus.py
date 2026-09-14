"""Build a calibrated, robust V4 rotation-consensus sparse candidate.

This is the architecture switch after the versioned GLOMAP candidate failed
the fixed independent audit.  It does not use phase, a turntable prior, or
external/reference geometry.  Same-ring camera relations are synchronized
from the calibrated ALIKED/LightGlue audit matrices.  Cross-ring relations
are retained in the report and assigned deterministic robust-consensus
residuals; they are not silently deleted from the evidence set.

The optional model writer starts from a versioned GLOMAP model only as an
image-derived initialization for camera centers, copies all registered image
keypoints, applies the calibrated rotation field, and re-triangulates the
existing image-derived tracks.  It never opens the canonical SQLite snapshot.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_RING_ORDER = (
    "g7_r1",
    "geo_g8",
    "geo_g9",
    "geo_g10",
    "geo_g11",
    "geo_g12",
)


def rotation_geodesic_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Return the full SO(3) geodesic for first-to-second matrices."""

    first_value = np.asarray(first, dtype=np.float64)
    second_value = np.asarray(second, dtype=np.float64)
    if first_value.shape != (3, 3) or second_value.shape != (3, 3):
        raise ValueError("rotation matrices must be 3x3")
    relative = first_value @ second_value.T
    cosine = np.clip((float(np.trace(relative)) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def calibrated_rotation(record: Mapping[str, Any]) -> np.ndarray | None:
    value = record.get("calibrated_rotation_matrix_first_to_second")
    if value is None:
        value = record.get("calibrated_reestimate_rotation_matrix_first_to_second")
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return None
    return matrix


def _average_rotations(matrices: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    if not matrices:
        return np.eye(3, dtype=np.float64)
    quaternions = Rotation.from_matrix(np.asarray(matrices, dtype=np.float64)).as_quat()
    numeric_weights = np.asarray(weights, dtype=np.float64)
    reference = quaternions[int(np.argmax(numeric_weights))]
    quaternions *= np.where(quaternions @ reference < 0.0, -1.0, 1.0)[:, None]
    accumulator = np.einsum("n,ni,nj->ij", numeric_weights, quaternions, quaternions)
    _, vectors = np.linalg.eigh(accumulator)
    quaternion = vectors[:, -1]
    if float(quaternion @ reference) < 0.0:
        quaternion = -quaternion
    return Rotation.from_quat(quaternion).as_matrix()


def _record_weight(record: Mapping[str, Any]) -> tuple[float, bool]:
    conditioned = bool(record.get("well_conditioned_calibrated"))
    inliers = max(1.0, float(record.get("calibrated_inliers", 1)))
    # Weak/unconditioned rows remain usable for connecting otherwise isolated
    # image nodes, but cannot pull the accepted calibrated synchronization.
    return (inliers if conditioned else max(0.001, inliers * 0.001), conditioned)


def _component_orientation(
    records: Sequence[Mapping[str, Any]],
    ring: str,
    *,
    max_iterations: int = 80,
    robust_scale_deg: float = 3.0,
    weak_scale_deg: float = 15.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    edges: list[tuple[float, str, str, np.ndarray, bool, Mapping[str, Any]]] = []
    nodes: set[str] = set()
    for record in records:
        if not bool(record.get("same_ring")) or str(record.get("ring")) != ring:
            continue
        matrix = calibrated_rotation(record)
        if matrix is None:
            continue
        first, second = str(record["first"]), str(record["second"])
        weight, conditioned = _record_weight(record)
        nodes.update((first, second))
        edges.append((weight, first, second, matrix, conditioned, record))
    if not nodes:
        return {}, {"ring": ring, "node_count": 0, "edge_count": 0, "components": []}

    # Deterministic maximum spanning forest gives an initialization that does
    # not depend on JSON order or a random path through the ring graph.
    parents = {node: node for node in nodes}

    def find(node: str) -> str:
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    tree: list[tuple[float, str, str, np.ndarray, bool, Mapping[str, Any]]] = []
    for edge in sorted(edges, key=lambda item: (-item[0], item[1], item[2])):
        _, first, second, _, _, _ = edge
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parents[root_first] = root_second
            tree.append(edge)

    tree_adjacency: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    for _, first, second, matrix, _, _ in tree:
        tree_adjacency[first].append((second, matrix))
        tree_adjacency[second].append((first, matrix.T))

    components: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        components[find(node)].append(node)
    local: dict[str, np.ndarray] = {}
    anchors: set[str] = set()
    for component in components.values():
        anchor = min(component)
        anchors.add(anchor)
        local[anchor] = np.eye(3, dtype=np.float64)
        queue: deque[str] = deque([anchor])
        while queue:
            current = queue.popleft()
            for neighbor, relation in sorted(tree_adjacency[current], key=lambda item: item[0]):
                if neighbor not in local:
                    local[neighbor] = relation @ local[current]
                    queue.append(neighbor)

    adjacency: dict[str, list[tuple[str, np.ndarray, float, bool, Mapping[str, Any]]]] = defaultdict(list)
    for weight, first, second, matrix, conditioned, record in edges:
        # For updating first, first = R.T @ second; for updating second,
        # second = R @ first.  This is the consistent first->second contract.
        adjacency[first].append((second, matrix.T, weight, conditioned, record))
        adjacency[second].append((first, matrix, weight, conditioned, record))

    iterations = 0
    for iterations in range(max_iterations):
        updated: dict[str, np.ndarray] = {}
        for node in sorted(nodes):
            if node in anchors:
                updated[node] = np.eye(3, dtype=np.float64)
                continue
            candidates: list[np.ndarray] = []
            weights: list[float] = []
            for neighbor, relation, weight, conditioned, _ in adjacency[node]:
                if neighbor not in local:
                    continue
                candidate = relation @ local[neighbor]
                residual = rotation_geodesic_deg(local[node], candidate)
                scale = robust_scale_deg if conditioned else weak_scale_deg
                candidates.append(candidate)
                weights.append(weight / (1.0 + (residual / scale) ** 2))
            updated[node] = _average_rotations(candidates, weights) if candidates else local[node]
        change = max(rotation_geodesic_deg(updated[node], local[node]) for node in nodes)
        local = updated
        if change < 1e-8:
            break

    conditioned_residuals: list[float] = []
    all_residuals: list[float] = []
    for _, first, second, matrix, conditioned, _ in edges:
        residual = rotation_geodesic_deg(local[second], matrix @ local[first])
        all_residuals.append(residual)
        if conditioned:
            conditioned_residuals.append(residual)

    def quantiles(values: Sequence[float]) -> list[float]:
        return [float(value) for value in np.quantile(values, [0, 0.5, 0.9, 0.95, 0.99, 1.0])] if values else []

    report = {
        "ring": ring,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "conditioned_edge_count": len(conditioned_residuals),
        "components": sorted(sorted(component) for component in components.values()),
        "anchor_names": sorted(anchors),
        "iterations": iterations + 1,
        "conditioned_residual_quantiles_deg": quantiles(conditioned_residuals),
        "all_residual_quantiles_deg": quantiles(all_residuals),
        "conditioned_residual_over_10_deg": sum(value > 10.0 for value in conditioned_residuals),
        "conditioned_residual_over_30_deg": sum(value > 30.0 for value in conditioned_residuals),
    }
    return local, report


def _consensus_for_transition(
    values: Sequence[tuple[np.ndarray, float, Mapping[str, Any]]],
    *,
    threshold_deg: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not values:
        raise ValueError("cross-ring transition has no conditioned calibrated evidence")
    best_key: tuple[float, int, float, str, str] | None = None
    best_index = 0
    for index, (candidate, _, record) in enumerate(values):
        residuals = [rotation_geodesic_deg(candidate, other) for other, _, _ in values]
        inlier_residuals = [residual for residual in residuals if residual <= threshold_deg]
        support = sum(
            weight for residual, (_, weight, _) in zip(residuals, values) if residual <= threshold_deg
        )
        key = (
            float(support),
            len(inlier_residuals),
            -float(np.median(inlier_residuals or [float("inf")])),
            str(record.get("first", "")),
            str(record.get("second", "")),
        )
        if best_key is None or key > best_key:
            best_key, best_index = key, index
    medoid = values[best_index][0]
    inliers = [item for item in values if rotation_geodesic_deg(medoid, item[0]) <= threshold_deg]
    consensus = _average_rotations([item[0] for item in inliers], [item[1] for item in inliers])
    residuals = [rotation_geodesic_deg(consensus, item[0]) for item in values]
    return consensus, {
        "edge_count": len(values),
        "inlier_count": len(inliers),
        "inlier_weight": float(sum(item[1] for item in inliers)),
        "support_weight": float(best_key[0]),
        "threshold_deg": threshold_deg,
        "residual_quantiles_deg": [
            float(value) for value in np.quantile(residuals, [0, 0.5, 0.9, 0.95, 0.99, 1.0])
        ],
        "outlier_count": sum(value > threshold_deg for value in residuals),
        "medoid_pair": [str(values[best_index][2]["first"]), str(values[best_index][2]["second"])],
    }


def build_rotation_consensus(
    records: Sequence[Mapping[str, Any]],
    *,
    ring_order: Sequence[str] = DEFAULT_RING_ORDER,
    cross_consensus_threshold_deg: float = 10.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Synchronize calibrated rotations without acquisition-phase priors."""

    order = tuple(str(ring) for ring in ring_order)
    ring_index = {ring: index for index, ring in enumerate(order)}
    observed_rings = {str(record.get("ring")) for record in records} | {
        str(record.get("second_ring")) for record in records
    }
    if observed_rings - set(order):
        raise ValueError(f"ring order omits observed rings: {sorted(observed_rings - set(order))}")
    local: dict[str, np.ndarray] = {}
    ring_reports: list[dict[str, Any]] = []
    for ring in order:
        orientations, report = _component_orientation(records, ring)
        local.update(orientations)
        ring_reports.append(report)

    transition_values: dict[tuple[str, str], list[tuple[np.ndarray, float, Mapping[str, Any]]]] = defaultdict(list)
    per_edge: list[dict[str, Any]] = []
    for record in records:
        matrix = calibrated_rotation(record)
        if matrix is None:
            continue
        first, second = str(record["first"]), str(record["second"])
        first_ring, second_ring = str(record["ring"]), str(record["second_ring"])
        if first_ring == second_ring:
            if first in local and second in local:
                residual = rotation_geodesic_deg(local[second], matrix @ local[first])
                per_edge.append({
                    "pair_id": int(record.get("pair_id", 0)),
                    "first": first,
                    "second": second,
                    "edge_type": "same_ring",
                    "conditioned": bool(record.get("well_conditioned_calibrated")),
                    "synchronization_residual_deg": residual,
                    "mapping_status": "retained_rotation_synchronization",
                })
            continue
        if not bool(record.get("well_conditioned_calibrated")):
            per_edge.append({
                "pair_id": int(record.get("pair_id", 0)),
                "first": first,
                "second": second,
                "edge_type": "cross_ring",
                "conditioned": False,
                "mapping_status": "retained_negative_or_unconditioned_evidence",
            })
            continue
        if first not in local or second not in local:
            continue
        if ring_index[first_ring] > ring_index[second_ring]:
            first, second = second, first
            first_ring, second_ring = second_ring, first_ring
            matrix = matrix.T
        candidate = local[second].T @ matrix @ local[first]
        transition_values[(first_ring, second_ring)].append(
            (candidate, max(1.0, float(record.get("calibrated_inliers", 1))), record)
        )

    gauges: dict[str, np.ndarray] = {order[0]: np.eye(3, dtype=np.float64)}
    transition_reports: list[dict[str, Any]] = []
    consensus_by_transition: dict[tuple[str, str], np.ndarray] = {}
    for first_ring, second_ring in zip(order, order[1:]):
        key = (first_ring, second_ring)
        consensus, report = _consensus_for_transition(
            transition_values.get(key, []), threshold_deg=cross_consensus_threshold_deg
        )
        consensus_by_transition[key] = consensus
        gauges[second_ring] = consensus @ gauges[first_ring]
        transition_reports.append({"first_ring": first_ring, "second_ring": second_ring, **report})

    orientations = {
        image_name: local[image_name] @ gauges[str(record_ring)]
        for image_name, record_ring in _image_ring_map(records).items()
        if image_name in local and str(record_ring) in gauges
    }

    for record in records:
        matrix = calibrated_rotation(record)
        if matrix is None or record.get("same_ring") or not bool(record.get("well_conditioned_calibrated")):
            continue
        first, second = str(record["first"]), str(record["second"])
        predicted = orientations[second] @ orientations[first].T
        residual = rotation_geodesic_deg(predicted, matrix)
        per_edge.append({
            "pair_id": int(record.get("pair_id", 0)),
            "first": first,
            "second": second,
            "edge_type": "cross_ring",
            "conditioned": True,
            "consensus_residual_deg": residual,
            "mapping_status": "retained_robust_consensus_inlier" if residual <= cross_consensus_threshold_deg else "retained_robust_consensus_outlier",
        })

    report = {
        "schema_version": 1,
        "method": "calibrated same-ring rotation synchronization with deterministic robust cross-ring consensus",
        "ring_order": list(order),
        "cross_consensus_threshold_deg": cross_consensus_threshold_deg,
        "input_record_count": len(records),
        "well_conditioned_record_count": sum(bool(record.get("well_conditioned_calibrated")) for record in records),
        "same_ring_reports": ring_reports,
        "cross_ring_transitions": transition_reports,
        "ring_graph_connected": len(gauges) == len(order),
        "per_edge": sorted(per_edge, key=lambda item: (int(item.get("pair_id", 0)), item["first"], item["second"])),
        "retained_evidence_policy": "all classified records remain in per_edge; consensus outliers are not deleted from the independent audit",
    }
    return orientations, report


def _image_ring_map(records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for record in records:
        result[str(record["first"])] = str(record["ring"])
        result[str(record["second"])] = str(record["second_ring"])
    return result


def write_consensus_model(
    source_model: str | Path,
    destination_model: str | Path,
    orientations: Mapping[str, np.ndarray],
    *,
    camera_centers: Mapping[str, np.ndarray] | None = None,
    database: Any | None = None,
    allowed_pair_ids: set[int] | None = None,
    pairwise_tracks: bool = False,
    max_points_per_pair: int = 40,
    min_points_per_pair: int = 20,
    max_reprojection_px: float = 12.0,
    max_median_reprojection_px: float = 6.0,
) -> dict[str, Any]:
    """Copy or rebuild image-derived tracks using synchronized rotations.

    When ``database`` is supplied, verified inlier correspondences are rebuilt
    as observation components from that disposable database.  This avoids
    inheriting the failed GLOMAP track partition while keeping every accepted
    3D observation tied to an original LightGlue/SQLite correspondence.
    """

    import pycolmap

    source_path = Path(source_model).resolve()
    destination_path = Path(destination_model).resolve()
    if destination_path.exists():
        raise FileExistsError(destination_path)
    old = pycolmap.Reconstruction(str(source_path))
    new = pycolmap.Reconstruction()
    for camera in old.cameras.values():
        new.add_camera_with_trivial_rig(pycolmap.Camera(camera.todict()))

    for image_id in sorted(old.reg_image_ids()):
        image = old.image(image_id)
        keypoints = np.asarray([point.xy for point in image.points2D], dtype=np.float64).reshape(-1, 2)
        old_pose = image.cam_from_world()
        rotation = np.asarray(orientations.get(image.name, old_pose.rotation.matrix()), dtype=np.float64)
        if camera_centers is not None and image.name in camera_centers:
            translation = -rotation @ np.asarray(camera_centers[image.name], dtype=np.float64).reshape(3)
        else:
            translation = np.asarray(old_pose.translation, dtype=np.float64)
        copied_image = pycolmap.Image(
            name=image.name,
            keypoints=keypoints,
            camera_id=image.camera_id,
            image_id=image.image_id,
        )
        new.add_image_with_trivial_frame(
            copied_image,
            pycolmap.Rigid3d(pycolmap.Rotation3d(rotation), translation),
        )

    cameras = {int(camera_id): new.camera(int(camera_id)) for camera_id in new.cameras}
    poses = {int(image_id): new.image(int(image_id)).cam_from_world() for image_id in new.reg_image_ids()}
    rejected: Counter[str] = Counter()
    added = 0
    if database is None:
        point_sources: Iterable[tuple[list[Any], np.ndarray]] = (
            (list(point.track.elements), np.asarray(point.color, dtype=np.uint8))
            for point in old.points3D.values()
        )
    else:
        if pairwise_tracks:
            point_sources = _database_pair_track_components(
                database,
                poses,
                old,
                allowed_pair_ids or set(),
                max_reprojection_px=max_reprojection_px,
                max_points_per_pair=max_points_per_pair,
                min_points_per_pair=min_points_per_pair,
            )
        else:
            point_sources = _database_track_components(database, poses, old, allowed_pair_ids or set())

    for source_elements, color in point_sources:
        elements = list(source_elements)
        if len(elements) < 2:
            rejected["short_track"] += 1
            continue
        # Iteratively remove the worst reprojection observation.  The input
        # component is a union of measured inlier matches; pruning a bad
        # observation here does not alter the independent audit or source DB.
        working = elements
        xyz: np.ndarray | None = None
        residuals: list[float] = []
        for _ in range(8):
            camera_matrices: list[np.ndarray] = []
            camera_rays: list[np.ndarray] = []
            observations: list[np.ndarray] = []
            for element in working:
                image_id, point_index = int(element.image_id), int(element.point2D_idx)
                image = old.image(image_id)
                xy = np.asarray(image.point2D(point_index).xy, dtype=np.float64)
                camera = cameras[image.camera_id]
                camera_rays.append(np.asarray(camera.cam_ray_from_img(xy), dtype=np.float64).reshape(3))
                camera_matrices.append(np.asarray(poses[image_id].matrix(), dtype=np.float64))
                observations.append(xy)
            xyz_value = pycolmap.triangulate_multi_view_point(
                camera_matrices, np.asarray(camera_rays, dtype=np.float64)
            )
            if xyz_value is None or not np.isfinite(xyz_value).all():
                xyz = None
                break
            xyz = np.asarray(xyz_value, dtype=np.float64).reshape(3)
            residuals = []
            cheirality = True
            for element, observed in zip(working, observations):
                pose = poses[int(element.image_id)]
                camera_point = np.asarray(pose.rotation.matrix(), dtype=np.float64) @ xyz + np.asarray(pose.translation)
                projected = new.image(int(element.image_id)).project_point(xyz)
                if not np.isfinite(camera_point).all() or float(camera_point[2]) <= 0.0 or projected is None:
                    cheirality = False
                    break
                residuals.append(float(np.linalg.norm(np.asarray(projected) - observed)))
            if not cheirality or not residuals:
                xyz = None
                break
            worst = int(np.argmax(residuals))
            if len(working) > 2 and residuals[worst] > max_reprojection_px:
                working = [element for index, element in enumerate(working) if index != worst]
                continue
            break
        if xyz is None or not residuals:
            rejected["triangulation_or_cheirality"] += 1
            continue
        if max(residuals) > max_reprojection_px or float(np.median(residuals)) > max_median_reprojection_px:
            rejected["reprojection"] += 1
            continue
        try:
            new.add_point3D(xyz, pycolmap.Track(working), color)
        except Exception:
            rejected["point_observation_conflict"] += 1
            continue
        added += 1
    new.update_point_3d_errors()
    destination_path.mkdir(parents=True, exist_ok=False)
    new.write(str(destination_path))
    return {
        "source_model": str(source_path),
        "destination_model": str(destination_path),
        "registered_images": int(new.num_reg_images()),
        "points3D": int(new.num_points3D()),
        "observations": int(new.compute_num_observations()),
        "mean_reprojection_error": float(new.compute_mean_reprojection_error()),
        "mean_track_length": float(new.compute_mean_track_length()),
        "added_points": added,
        "rejected_tracks": dict(rejected),
    }


def _database_track_components(
    database: Any,
    poses: Mapping[int, Any],
    source_reconstruction: Any,
    allowed_pair_ids: set[int],
    *,
    max_component_observations: int = 50,
) -> Iterable[tuple[list[Any], np.ndarray]]:
    """Yield disjoint observation components from verified two-view matches."""

    from v4_repair import decode_colmap_pair_id

    image_by_id = {int(image.image_id): image for image in database.read_all_images()}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    size: dict[tuple[int, int], int] = {}

    def find(node: tuple[int, int]) -> tuple[int, int]:
        parent.setdefault(node, node)
        size.setdefault(node, 1)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(first: tuple[int, int], second: tuple[int, int]) -> None:
        first_root, second_root = find(first), find(second)
        if first_root == second_root:
            return
        if size[first_root] < size[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        size[first_root] += size[second_root]

    pair_ids, geometries = database.read_two_view_geometries()
    for pair_id, geometry in zip(pair_ids, geometries):
        pair_value = int(pair_id)
        if pair_value not in allowed_pair_ids:
            continue
        try:
            first_id, second_id = decode_colmap_pair_id(pair_value)
        except ValueError:
            continue
        if first_id not in image_by_id or second_id not in image_by_id:
            continue
        matches = np.asarray(getattr(geometry, "inlier_matches", np.empty((0, 2), dtype=np.uint32)))
        if matches.ndim != 2 or matches.shape[1] != 2:
            continue
        for first_index, second_index in matches.astype(np.int64, copy=False):
            union((first_id, int(first_index)), (second_id, int(second_index)))

    components: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for node in sorted(parent):
        components[find(node)].append(node)
    for nodes in sorted(components.values(), key=lambda value: value[0]):
        if len(nodes) < 2 or len(nodes) > max_component_observations:
            continue
        image_ids = [node[0] for node in nodes]
        if len(image_ids) != len(set(image_ids)):
            continue
        elements = [__import__("pycolmap").TrackElement(image_id, index) for image_id, index in nodes]
        yield elements, np.asarray([128, 128, 128], dtype=np.uint8)


def _database_pair_track_components(
    database: Any,
    poses: Mapping[int, Any],
    source_reconstruction: Any,
    allowed_pair_ids: set[int],
    *,
    max_reprojection_px: float,
    max_points_per_pair: int,
    min_points_per_pair: int,
) -> Iterable[tuple[list[Any], np.ndarray]]:
    """Yield disjoint, pair-local tracks with measured reprojection support.

    A pair-local track is deliberately limited to the two original matched
    observations.  The allocator chooses a deterministic disjoint subset per
    edge, so one keypoint is never assigned to contradictory 3D points while
    every strong same-ring edge retains enough independent support for the
    unchanged sparse gate.
    """

    import cv2

    from v4_repair import decode_colmap_pair_id

    image_by_id = {int(image.image_id): image for image in database.read_all_images()}
    keypoints: dict[int, np.ndarray] = {}
    cameras: dict[int, Any] = {}
    pair_ids, geometries = database.read_two_view_geometries()
    candidates_by_pair: list[tuple[int, int, int, list[tuple[float, int, int, np.ndarray]]]] = []
    for pair_id, geometry in zip(pair_ids, geometries):
        pair_value = int(pair_id)
        if pair_value not in allowed_pair_ids:
            continue
        try:
            first_id, second_id = decode_colmap_pair_id(pair_value)
        except ValueError:
            continue
        if first_id not in image_by_id or second_id not in image_by_id:
            continue
        first_image, second_image = image_by_id[first_id], image_by_id[second_id]
        first_name, second_name = str(first_image.name), str(second_image.name)
        if first_id not in keypoints:
            keypoints[first_id] = np.asarray(database.read_keypoints(first_id), dtype=np.float64)
        if second_id not in keypoints:
            keypoints[second_id] = np.asarray(database.read_keypoints(second_id), dtype=np.float64)
        if first_image.camera_id not in cameras:
            cameras[first_image.camera_id] = database.read_camera(int(first_image.camera_id))
        if second_image.camera_id not in cameras:
            cameras[second_image.camera_id] = database.read_camera(int(second_image.camera_id))
        matches = np.asarray(getattr(geometry, "inlier_matches", np.empty((0, 2))), dtype=np.int64)
        if matches.ndim != 2 or matches.shape[1] != 2 or len(matches) < min_points_per_pair:
            continue
        pose_first, pose_second = poses.get(first_id), poses.get(second_id)
        if pose_first is None or pose_second is None:
            continue
        normalized_first = np.asarray(
            cameras[first_image.camera_id].cam_from_img(keypoints[first_id][matches[:, 0], :2]),
            dtype=np.float64,
        )
        normalized_second = np.asarray(
            cameras[second_image.camera_id].cam_from_img(keypoints[second_id][matches[:, 1], :2]),
            dtype=np.float64,
        )
        try:
            homogeneous = cv2.triangulatePoints(
                np.asarray(pose_first.matrix(), dtype=np.float64),
                np.asarray(pose_second.matrix(), dtype=np.float64),
                normalized_first.T,
                normalized_second.T,
            )
        except Exception:
            continue
        denominator = homogeneous[3]
        finite = np.isfinite(homogeneous).all(axis=0) & (np.abs(denominator) > 1e-12)
        xyz = np.zeros((3, len(matches)), dtype=np.float64)
        xyz[:, finite] = homogeneous[:3, finite] / denominator[finite]
        camera_first = np.asarray(pose_first.rotation.matrix()) @ xyz + np.asarray(pose_first.translation).reshape(3, 1)
        camera_second = np.asarray(pose_second.rotation.matrix()) @ xyz + np.asarray(pose_second.translation).reshape(3, 1)
        positive = finite & (camera_first[2] > 0.0) & (camera_second[2] > 0.0)
        projected_first = cameras[first_image.camera_id].img_from_cam(camera_first.T)
        projected_second = cameras[second_image.camera_id].img_from_cam(camera_second.T)
        error = np.maximum(
            np.linalg.norm(projected_first - keypoints[first_id][matches[:, 0], :2], axis=1),
            np.linalg.norm(projected_second - keypoints[second_id][matches[:, 1], :2], axis=1),
        )
        good = positive & np.isfinite(error) & (error <= max_reprojection_px)
        candidates = [
            (float(error[index]), int(matches[index, 0]), int(matches[index, 1]), xyz[:, index].copy())
            for index in np.flatnonzero(good)
        ]
        candidates.sort(key=lambda value: (value[0], value[1], value[2]))
        if len(candidates) >= min_points_per_pair:
            candidates_by_pair.append((pair_value, first_id, second_id, candidates))

    used_observations: set[tuple[int, int]] = set()
    # Small support edges are allocated first so high-support pairs cannot
    # starve an acquisition-local transition of its minimum track evidence.
    candidates_by_pair.sort(key=lambda item: (len(item[3]), item[0]))
    for _, first_id, second_id, candidates in candidates_by_pair:
        selected: list[tuple[float, int, int, np.ndarray]] = []
        for candidate in candidates:
            _, first_index, second_index, _ = candidate
            first_key, second_key = (first_id, first_index), (second_id, second_index)
            if first_key in used_observations or second_key in used_observations:
                continue
            selected.append(candidate)
            if len(selected) >= max_points_per_pair:
                break
        if len(selected) < min_points_per_pair:
            continue
        for _, first_index, second_index, xyz in selected:
            used_observations.add((first_id, first_index))
            used_observations.add((second_id, second_index))
            import pycolmap

            yield [pycolmap.TrackElement(first_id, first_index), pycolmap.TrackElement(second_id, second_index)], np.asarray(
                [128, 128, 128], dtype=np.uint8
            )


def recover_translation_direction_edges(
    database: Any,
    records: Sequence[Mapping[str, Any]],
    orientations: Mapping[str, np.ndarray],
    allowed_pair_ids: set[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recover calibrated translation directions from the disposable DB."""

    import cv2

    from v4_repair import decode_colmap_pair_id

    image_by_id = {int(image.image_id): image for image in database.read_all_images()}
    record_by_pair = {int(record["pair_id"]): record for record in records}
    keypoints: dict[int, np.ndarray] = {}
    pair_ids, geometries = database.read_two_view_geometries()
    edges: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    for pair_id, geometry in zip(pair_ids, geometries):
        pair_value = int(pair_id)
        if pair_value not in allowed_pair_ids or pair_value not in record_by_pair:
            continue
        try:
            first_id, second_id = decode_colmap_pair_id(pair_value)
        except ValueError:
            failures["invalid_pair_id"] += 1
            continue
        first_image, second_image = image_by_id.get(first_id), image_by_id.get(second_id)
        if first_image is None or second_image is None:
            failures["missing_image"] += 1
            continue
        first_name, second_name = str(first_image.name), str(second_image.name)
        if first_name not in orientations or second_name not in orientations:
            failures["missing_orientation"] += 1
            continue
        essential = np.asarray(getattr(geometry, "E", np.empty((0, 0))), dtype=np.float64)
        matches = np.asarray(getattr(geometry, "inlier_matches", np.empty((0, 2))), dtype=np.int64)
        if essential.shape != (3, 3) or matches.ndim != 2 or matches.shape[1] != 2 or len(matches) < 8:
            failures["insufficient_geometry"] += 1
            continue
        if first_id not in keypoints:
            keypoints[first_id] = np.asarray(database.read_keypoints(first_id), dtype=np.float64)
        if second_id not in keypoints:
            keypoints[second_id] = np.asarray(database.read_keypoints(second_id), dtype=np.float64)
        try:
            first_pixels = keypoints[first_id][matches[:, 0], :2]
            second_pixels = keypoints[second_id][matches[:, 1], :2]
            first_camera = database.read_camera(int(first_image.camera_id))
            second_camera = database.read_camera(int(second_image.camera_id))
            first_normalized = np.asarray(first_camera.cam_from_img(first_pixels), dtype=np.float64)
            second_normalized = np.asarray(second_camera.cam_from_img(second_pixels), dtype=np.float64)
            _, _, translation, _ = cv2.recoverPose(
                essential,
                first_normalized,
                second_normalized,
                np.eye(3, dtype=np.float64),
            )
        except Exception as error:
            failures[f"recover_pose:{type(error).__name__}"] += 1
            continue
        direction = -np.asarray(orientations[second_name], dtype=np.float64).T @ np.asarray(translation).reshape(3)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(direction).all() or norm <= 1e-9:
            failures["invalid_direction"] += 1
            continue
        direction /= norm
        edges.append({
            "pair_id": pair_value,
            "first": first_name,
            "second": second_name,
            "first_id": first_id,
            "second_id": second_id,
            "direction_world": direction.tolist(),
            "weight": max(1.0, float(record_by_pair[pair_value].get("calibrated_inliers", 1))),
            "same_ring": bool(record_by_pair[pair_value].get("same_ring")),
        })
    return edges, {
        "candidate_edge_count": len(edges),
        "failure_counts": dict(failures),
        "method": "recoverPose translation direction from calibrated essential matrices in disposable DB",
    }


def solve_camera_centers(
    edges: Sequence[Mapping[str, Any]],
    initial_centers: Mapping[str, np.ndarray],
    *,
    max_iterations: int = 25,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Solve image-derived camera centers from robust direction constraints."""

    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import lsqr

    names = sorted(initial_centers)
    if not names:
        raise ValueError("camera-center solve requires at least one image")
    anchor = names[0]
    free_names = [name for name in names if name != anchor]
    free_index = {name: index for index, name in enumerate(free_names)}
    centers = {name: np.asarray(value, dtype=np.float64).reshape(3).copy() for name, value in initial_centers.items()}
    usable = [edge for edge in edges if edge["first"] in centers and edge["second"] in centers]
    if not usable:
        raise ValueError("camera-center solve has no usable edges")
    negative_initial = 0
    for edge in usable:
        first, second = edge["first"], edge["second"]
        direction = np.asarray(edge["direction_world"], dtype=np.float64)
        if float(np.dot(centers[second] - centers[first], direction)) < 0.0:
            negative_initial += 1

    residual_quantiles: list[float] = []
    iterations = 0
    for iterations in range(max_iterations):
        scales: list[float] = []
        residuals: list[float] = []
        for edge in usable:
            direction = np.asarray(edge["direction_world"], dtype=np.float64)
            delta = centers[edge["second"]] - centers[edge["first"]]
            scale = float(np.dot(delta, direction))
            scales.append(max(abs(scale), 1e-4))
            residuals.append(float(np.linalg.norm(delta - scales[-1] * direction)))
        residual_scale = max(float(np.median(residuals)), 1e-4)
        rows = 3 * len(usable)
        columns = 3 * len(free_names)
        matrix = lil_matrix((rows, columns), dtype=np.float64)
        right = np.zeros(rows, dtype=np.float64)
        for edge_index, (edge, scale, residual) in enumerate(zip(usable, scales, residuals)):
            robust = math.sqrt(float(edge["weight"])) / (1.0 + (residual / (3.0 * residual_scale)) ** 2)
            first, second = edge["first"], edge["second"]
            rhs = scale * np.asarray(edge["direction_world"], dtype=np.float64)
            if first == anchor:
                rhs += centers[anchor]
            else:
                first_column = 3 * free_index[first]
                for axis in range(3):
                    matrix[3 * edge_index + axis, first_column + axis] -= robust
            if second == anchor:
                rhs -= centers[anchor]
            else:
                second_column = 3 * free_index[second]
                for axis in range(3):
                    matrix[3 * edge_index + axis, second_column + axis] += robust
            right[3 * edge_index : 3 * edge_index + 3] = robust * rhs
        solution = lsqr(matrix.tocsr(), right, atol=1e-9, btol=1e-9, iter_lim=1000)[0]
        updated = {anchor: centers[anchor].copy()}
        for name, index in free_index.items():
            updated[name] = solution[3 * index : 3 * index + 3]
        change = max(float(np.linalg.norm(updated[name] - centers[name])) for name in names)
        centers = updated
        if change < 1e-7:
            break
    final_residuals = []
    for edge in usable:
        direction = np.asarray(edge["direction_world"], dtype=np.float64)
        delta = centers[edge["second"]] - centers[edge["first"]]
        scale = float(np.dot(delta, direction))
        final_residuals.append(float(np.linalg.norm(delta - max(abs(scale), 1e-4) * direction)))
    if final_residuals:
        residual_quantiles = [
            float(value) for value in np.quantile(final_residuals, [0, 0.5, 0.9, 0.95, 0.99, 1.0])
        ]
    return centers, {
        "image_count": len(names),
        "edge_count": len(usable),
        "anchor": anchor,
        "iterations": iterations + 1,
        "negative_initial_direction_count": negative_initial,
        "direction_residual_quantiles": residual_quantiles,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--source-model", type=Path)
    parser.add_argument("--output-model", type=Path)
    parser.add_argument(
        "--snapshot-manifest",
        type=Path,
        help="v3 canonical SQLite manifest; only a disposable working copy is opened",
    )
    parser.add_argument(
        "--solve-positions",
        action="store_true",
        help="recover translation directions and solve camera centers from the disposable DB",
    )
    parser.add_argument(
        "--pairwise-tracks",
        action="store_true",
        help="allocate disjoint pair-local tracks from measured verified inliers",
    )
    parser.add_argument("--max-points-per-pair", type=int, default=40)
    parser.add_argument("--min-points-per-pair", type=int, default=20)
    parser.add_argument("--ring-order", nargs="*", default=list(DEFAULT_RING_ORDER))
    parser.add_argument("--max-reprojection-px", type=float, default=12.0)
    parser.add_argument("--max-median-reprojection-px", type=float, default=6.0)
    args = parser.parse_args()
    classification_payload = json.loads(args.classification.read_text(encoding="utf-8"))
    records = classification_payload.get("classification", classification_payload)
    if not isinstance(records, list):
        raise ValueError("classification payload must contain a classification list")
    orientations, report = build_rotation_consensus(records, ring_order=args.ring_order)
    report["classification_path"] = str(args.classification.resolve())
    report["classification_sha256"] = _sha256(args.classification)
    report["orientation_count"] = len(orientations)
    if args.source_model is not None or args.output_model is not None:
        if args.source_model is None or args.output_model is None:
            raise ValueError("--source-model and --output-model must be provided together")
        if args.snapshot_manifest is None:
            report["model_write"] = write_consensus_model(
                args.source_model,
                args.output_model,
                orientations,
                max_reprojection_px=args.max_reprojection_px,
                max_median_reprojection_px=args.max_median_reprojection_px,
            )
        else:
            from v4_repair import create_disposable_sqlite_from_manifest

            allowed_pair_ids = {
                int(item["pair_id"])
                for item in report["per_edge"]
                if item.get("mapping_status")
                in {"retained_rotation_synchronization", "retained_robust_consensus_inlier"}
            }
            with tempfile.TemporaryDirectory(prefix="v4_rotation_consensus_db_") as temporary:
                working_path = Path(temporary) / "working.db"
                disposable = create_disposable_sqlite_from_manifest(args.snapshot_manifest, working_path)
                import pycolmap

                database = pycolmap.Database.open(str(working_path))
                try:
                    camera_centers = None
                    if args.solve_positions:
                        source_reconstruction = pycolmap.Reconstruction(str(args.source_model))
                        initial_centers = {
                            str(image.name): -np.asarray(image.cam_from_world().rotation.matrix()).T
                            @ np.asarray(image.cam_from_world().translation, dtype=np.float64)
                            for image in source_reconstruction.images.values()
                            if image.has_pose
                        }
                        translation_edges, translation_report = recover_translation_direction_edges(
                            database, records, orientations, allowed_pair_ids
                        )
                        camera_centers, position_report = solve_camera_centers(
                            translation_edges, initial_centers
                        )
                        report["translation_direction_recovery"] = translation_report
                        report["camera_center_solve"] = position_report
                    report["model_write"] = write_consensus_model(
                        args.source_model,
                        args.output_model,
                        orientations,
                        camera_centers=camera_centers,
                        database=database,
                        allowed_pair_ids=allowed_pair_ids,
                        pairwise_tracks=args.pairwise_tracks,
                        max_points_per_pair=args.max_points_per_pair,
                        min_points_per_pair=args.min_points_per_pair,
                        max_reprojection_px=args.max_reprojection_px,
                        max_median_reprojection_px=args.max_median_reprojection_px,
                    )
                finally:
                    database.close()
            report["working_database_sha256"] = disposable["working"]["sha256"]
            report["snapshot_manifest_path"] = str(args.snapshot_manifest.resolve())
            report["snapshot_manifest_sha256"] = _sha256(args.snapshot_manifest)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "orientation_count": len(orientations),
        "ring_graph_connected": report["ring_graph_connected"],
        "model_write": report.get("model_write"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
