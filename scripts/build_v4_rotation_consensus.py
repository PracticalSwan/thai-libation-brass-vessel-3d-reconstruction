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


def build_ring_aligned_repair(
    records: Sequence[Mapping[str, Any]],
    reference_orientations: Mapping[str, np.ndarray],
    *,
    ring_order: Sequence[str] = DEFAULT_RING_ORDER,
    gauge_threshold_deg: float = 15.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Repair same-ring pose islands while retaining a reference world gauge.

    The calibrated graph determines each ring's relative rotations.  Its
    per-ring gauge is otherwise arbitrary, so this helper aligns the
    synchronized field to the existing image-derived model using the largest
    robust cluster of ``R_reference @ R_local.T`` gauges.  A low-support
    gauge is reported rather than hidden; callers must still run the complete
    sparse gate.  No phase/order prior or synthetic geometry is introduced.
    """

    if not math.isfinite(float(gauge_threshold_deg)) or float(gauge_threshold_deg) <= 0.0:
        raise ValueError("gauge_threshold_deg must be finite and positive")
    reference = {
        str(name): np.asarray(matrix, dtype=np.float64)
        for name, matrix in reference_orientations.items()
    }
    for name, matrix in reference.items():
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError(f"invalid reference orientation for {name}")

    corrected: dict[str, np.ndarray] = {}
    ring_reports: list[dict[str, Any]] = []
    for ring in tuple(str(value) for value in ring_order):
        local, local_report = _component_orientation(records, ring)
        gauge_candidates = [
            (name, reference[name] @ local[name].T)
            for name in sorted(local)
            if name in reference
        ]
        if not gauge_candidates:
            ring_reports.append({
                **local_report,
                "gauge_support_count": 0,
                "gauge_support_fraction": 0.0,
                "gauge_threshold_deg": float(gauge_threshold_deg),
                "gauge_status": "missing_reference_overlap",
            })
            continue

        best_key: tuple[int, float, str] | None = None
        best_inliers: list[tuple[str, np.ndarray]] = []
        for name, candidate in gauge_candidates:
            inliers = [
                item
                for item in gauge_candidates
                if rotation_geodesic_deg(candidate, item[1]) <= gauge_threshold_deg
            ]
            key = (
                len(inliers),
                -float(sum(rotation_geodesic_deg(candidate, item[1]) for item in inliers)),
                name,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_inliers = inliers
        gauge = _average_rotations(
            [matrix for _, matrix in best_inliers],
            [1.0 for _ in best_inliers],
        )
        for name, local_rotation in local.items():
            corrected[name] = gauge @ local_rotation
        change_values = [
            rotation_geodesic_deg(corrected[name], reference[name])
            for name in local
            if name in reference
        ]
        gauge_residuals = [
            rotation_geodesic_deg(gauge, candidate)
            for _, candidate in gauge_candidates
        ]
        support_fraction = len(best_inliers) / len(gauge_candidates)
        ring_reports.append({
            **local_report,
            "gauge_support_count": len(best_inliers),
            "gauge_support_fraction": float(support_fraction),
            "gauge_candidate_count": len(gauge_candidates),
            "gauge_threshold_deg": float(gauge_threshold_deg),
            "gauge_status": "supported" if support_fraction >= 0.5 else "low_support",
            "gauge_residual_quantiles_deg": _rotation_quantiles(gauge_residuals),
            "reference_change_quantiles_deg": _rotation_quantiles(change_values),
        })

    # Keep registered images that have no usable calibrated ring edge at their
    # original image-derived pose; the acceptance gate will expose any harm.
    for name, matrix in reference.items():
        corrected.setdefault(name, matrix.copy())
    report = {
        "schema_version": 1,
        "method": "calibrated same-ring synchronization with largest robust reference-gauge cluster",
        "ring_order": [str(value) for value in ring_order],
        "gauge_threshold_deg": float(gauge_threshold_deg),
        "orientation_count": len(corrected),
        "reference_orientation_count": len(reference),
        "same_ring_reports": ring_reports,
        "low_support_rings": [
            item["ring"]
            for item in ring_reports
            if item.get("gauge_status") != "supported"
        ],
        "retained_reference_pose_policy": "uncovered registered images retain their source-model pose; all candidate poses remain subject to independent sparse gates",
    }
    return corrected, report


def build_source_component_pose_repair(
    records: Sequence[Mapping[str, Any]],
    reference_orientations: Mapping[str, np.ndarray],
    *,
    ring_order: Sequence[str] = DEFAULT_RING_ORDER,
    source_edge_threshold_deg: float = 30.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Repair only source pose islands using calibrated image-derived edges.

    The failed global rotation-consensus candidates integrated small local
    rotation errors around an entire ring, destroying the source model's
    useful camera gauge and mask agreement.  This architecture keeps the
    source pose field as the default, cuts only same-ring edges whose source
    relative rotation disagrees with the calibrated first-to-second rotation
    beyond the sparse gate threshold, and aligns the resulting components with
    the calibrated graph.  One largest component per ring is anchored to the
    source gauge; every other component is reached through deterministic
    calibrated links, preferring same-ring evidence before cross-ring bridges.

    No frame order, turntable orbit, or fixed audit-pair preference is used.
    All conditioned calibrated edges remain in the returned residual report,
    including edges not selected for the component spanning tree.
    """

    if not math.isfinite(float(source_edge_threshold_deg)) or source_edge_threshold_deg <= 0.0:
        raise ValueError("source_edge_threshold_deg must be finite and positive")
    reference = {
        str(name): np.asarray(matrix, dtype=np.float64)
        for name, matrix in reference_orientations.items()
    }
    for name, matrix in reference.items():
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError(f"invalid reference orientation for {name}")

    conditioned_edges: list[dict[str, Any]] = []
    ring_nodes: dict[str, set[str]] = defaultdict(set)
    for record in records:
        matrix = calibrated_rotation(record)
        if matrix is None or not bool(record.get("well_conditioned_calibrated")):
            continue
        first, second = str(record["first"]), str(record["second"])
        if first not in reference or second not in reference:
            continue
        source_relative = reference[second] @ reference[first].T
        source_residual = rotation_geodesic_deg(source_relative, matrix)
        edge = {
            "pair_id": int(record.get("pair_id", 0)),
            "first": first,
            "second": second,
            "ring": str(record.get("ring", "")),
            "second_ring": str(record.get("second_ring", "")),
            "same_ring": bool(record.get("same_ring")),
            "calibrated_inliers": max(1, int(record.get("calibrated_inliers", 1))),
            "calibrated_rotation": matrix,
            "source_relative": source_relative,
            "source_residual_deg": float(source_residual),
        }
        conditioned_edges.append(edge)
        if edge["same_ring"] and edge["ring"] == edge["second_ring"]:
            ring_nodes.setdefault(edge["ring"], set()).update((first, second))

    component_of: dict[str, tuple[str, str]] = {}
    component_nodes: dict[tuple[str, str], list[str]] = {}
    for ring, nodes in sorted(ring_nodes.items()):
        parent = {name: name for name in nodes}

        def find(name: str) -> str:
            while parent[name] != name:
                parent[name] = parent[parent[name]]
                name = parent[name]
            return name

        def union(first: str, second: str) -> None:
            first_root, second_root = find(first), find(second)
            if first_root != second_root:
                parent[second_root] = first_root

        for edge in conditioned_edges:
            if (
                edge["same_ring"]
                and edge["ring"] == ring
                and edge["second_ring"] == ring
                and edge["source_residual_deg"] <= source_edge_threshold_deg
            ):
                union(edge["first"], edge["second"])
        for name in sorted(nodes):
            component = (ring, find(name))
            component_of[name] = component
            component_nodes.setdefault(component, []).append(name)

    # Keep any registered image not covered by a same-ring calibrated edge in
    # a singleton source-gauge component; this is explicit negative evidence,
    # not an invented pose.
    for name in sorted(reference):
        if name not in component_of:
            component = ("uncovered", name)
            component_of[name] = component
            component_nodes.setdefault(component, []).append(name)

    links: dict[tuple[tuple[str, str], tuple[str, str]], list[dict[str, Any]]] = defaultdict(list)
    for edge in conditioned_edges:
        first_component = component_of[edge["first"]]
        second_component = component_of[edge["second"]]
        if first_component == second_component:
            continue
        key = tuple(sorted((first_component, second_component), key=str))
        links[key].append(edge)

    selected_links: list[dict[str, Any]] = []
    for key, candidates in links.items():
        selected = sorted(
            candidates,
            key=lambda edge: (
                -int(edge["same_ring"]),
                -int(edge["calibrated_inliers"]),
                float(edge["source_residual_deg"]),
                int(edge["pair_id"]),
            ),
        )[0]
        selected_links.append(selected)
    selected_link_object_ids = {id(edge) for edge in selected_links}

    # One source-gauge anchor per observed ring preserves the useful mask and
    # translation gauge.  A deterministic best-link expansion reaches pose
    # islands through same-ring evidence before cross-ring links.
    anchors: set[tuple[str, str]] = set()
    for ring in sorted(ring_nodes):
        candidates = [component for component in component_nodes if component[0] == ring]
        if candidates:
            anchors.add(max(candidates, key=lambda component: (len(component_nodes[component]), str(component))))
    corrections: dict[tuple[str, str], np.ndarray] = {
        component: np.eye(3, dtype=np.float64) for component in anchors
    }
    assigned = set(corrections)
    component_graph: dict[tuple[str, str], list[tuple[dict[str, Any], tuple[str, str]]]] = defaultdict(list)
    for edge in selected_links:
        first_component = component_of[edge["first"]]
        second_component = component_of[edge["second"]]
        component_graph[first_component].append((edge, second_component))
        component_graph[second_component].append((edge, first_component))

    while True:
        candidates: list[tuple[int, int, float, int, tuple[str, str], tuple[str, str], dict[str, Any]]] = []
        for source_component in sorted(assigned, key=str):
            for edge, target_component in component_graph[source_component]:
                if target_component in assigned:
                    continue
                candidates.append(
                    (
                        int(edge["same_ring"]),
                        int(edge["calibrated_inliers"]),
                        -float(edge["source_residual_deg"]),
                        -int(edge["pair_id"]),
                        source_component,
                        target_component,
                        edge,
                    )
                )
        if not candidates:
            break
        _, _, _, _, source_component, target_component, edge = max(candidates)
        first_component = component_of[edge["first"]]
        second_component = component_of[edge["second"]]
        matrix = np.asarray(edge["calibrated_rotation"], dtype=np.float64)
        source_relative = np.asarray(edge["source_relative"], dtype=np.float64)
        if source_component == first_component and target_component == second_component:
            corrections[target_component] = matrix @ corrections[source_component] @ source_relative.T
        elif source_component == second_component and target_component == first_component:
            corrections[target_component] = matrix.T @ corrections[source_component] @ source_relative
        else:  # pragma: no cover - component graph construction invariant
            raise RuntimeError("component link orientation is inconsistent")
        assigned.add(target_component)

    corrected = {
        name: corrections.get(component_of[name], np.eye(3, dtype=np.float64)) @ matrix
        for name, matrix in reference.items()
    }
    residual_rows: list[dict[str, Any]] = []
    for edge in conditioned_edges:
        first, second = edge["first"], edge["second"]
        relative = corrected[second] @ corrected[first].T
        corrected_residual = rotation_geodesic_deg(relative, edge["calibrated_rotation"])
        residual_rows.append(
            {
                "pair_id": edge["pair_id"],
                "first": first,
                "second": second,
                "ring": edge["ring"],
                "same_ring": edge["same_ring"],
                "source_residual_deg": edge["source_residual_deg"],
                "corrected_residual_deg": corrected_residual,
                "selected_component_link": id(edge) in selected_link_object_ids,
            }
        )

    def quantiles(values: Sequence[float]) -> list[float]:
        return (
            [float(value) for value in np.quantile(np.asarray(values, dtype=np.float64), [0, 0.5, 0.9, 0.95, 0.99, 1.0])]
            if values
            else []
        )

    ring_reports: list[dict[str, Any]] = []
    for ring in sorted(ring_nodes):
        values = [row for row in residual_rows if row["ring"] == ring and row["same_ring"]]
        ring_reports.append(
            {
                "ring": ring,
                "component_count": sum(1 for component in component_nodes if component[0] == ring),
                "source_residual_quantiles_deg": quantiles([float(row["source_residual_deg"]) for row in values]),
                "corrected_residual_quantiles_deg": quantiles([float(row["corrected_residual_deg"]) for row in values]),
                "corrected_residual_over_threshold_count": sum(
                    float(row["corrected_residual_deg"]) > source_edge_threshold_deg for row in values
                ),
            }
        )

    return corrected, {
        "schema_version": 1,
        "method": "source-gauge component pose repair from calibrated first-to-second rotations",
        "source_edge_threshold_deg": float(source_edge_threshold_deg),
        "reference_orientation_count": len(reference),
        "conditioned_edge_count": len(conditioned_edges),
        "component_count": len(component_nodes),
        "assigned_component_count": len(assigned),
        "unassigned_components": [str(component) for component in sorted(component_nodes, key=str) if component not in assigned],
        "anchors": [
            {"ring": component[0], "component": component[1], "image_count": len(component_nodes[component])}
            for component in sorted(anchors, key=str)
        ],
        "components": [
            {
                "ring": component[0],
                "component": component[1],
                "image_count": len(component_nodes[component]),
                "images": sorted(component_nodes[component]),
                "correction_geodesic_deg": rotation_geodesic_deg(
                    corrections.get(component, np.eye(3, dtype=np.float64)), np.eye(3, dtype=np.float64)
                ),
            }
            for component in sorted(component_nodes, key=str)
        ],
        "selected_component_links": [
            {
                "pair_id": int(edge["pair_id"]),
                "first": edge["first"],
                "second": edge["second"],
                "same_ring": bool(edge["same_ring"]),
                "calibrated_inliers": int(edge["calibrated_inliers"]),
                "source_residual_deg": float(edge["source_residual_deg"]),
            }
            for edge in sorted(selected_links, key=lambda item: int(item["pair_id"]))
        ],
        "ring_reports": ring_reports,
        "residual_evidence": residual_rows,
        "retained_evidence_policy": "all conditioned calibrated edges remain measured; only component gauge corrections are applied",
    }


def _rotation_quantiles(values: Sequence[float]) -> list[float]:
    """Return deterministic rotation quantiles for repair reports."""

    if not values:
        return []
    return [
        float(value)
        for value in np.quantile(np.asarray(values, dtype=np.float64), [0, 0.5, 0.9, 0.95, 0.99, 1.0])
    ]


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
    track_pair_order: str = "pair_id",
    priority_pair_ids: set[int] | None = None,
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
    track_builder_stats: dict[str, Any] = {}
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
            point_sources = _database_track_components(
                database,
                poses,
                old,
                allowed_pair_ids or set(),
                track_builder_stats=track_builder_stats,
                pair_order=track_pair_order,
                priority_pair_ids=priority_pair_ids,
            )

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
        "track_builder": track_builder_stats,
    }


def _database_track_components(
    database: Any,
    poses: Mapping[int, Any],
    source_reconstruction: Any,
    allowed_pair_ids: set[int],
    *,
    max_component_observations: int = 372,
    track_builder_stats: dict[str, Any] | None = None,
    pair_order: str = "pair_id",
    priority_pair_ids: set[int] | None = None,
) -> Iterable[tuple[list[Any], np.ndarray]]:
    """Yield conflict-free multi-view components from verified two-view matches.

    A blind union-find over pair matches can merge a repeated or contradictory
    keypoint into one component that contains several observations from the
    same image.  The old implementation discarded that entire component, which
    could erase most of the graph (including a component spanning all 372
    images).  Track construction now refuses a merge when the two roots already
    contain the same image.  The raw pair graph is untouched; only the derived
    track partition changes, and every yielded observation still comes directly
    from an image-derived verified match.
    """

    from v4_repair import decode_colmap_pair_id

    image_by_id = {int(image.image_id): image for image in database.read_all_images()}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    size: dict[tuple[int, int], int] = {}
    image_sets: dict[tuple[int, int], set[int]] = {}
    conflict_merges = 0
    accepted_merges = 0

    def find(node: tuple[int, int]) -> tuple[int, int]:
        parent.setdefault(node, node)
        size.setdefault(node, 1)
        image_sets.setdefault(node, {int(node[0])})
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(first: tuple[int, int], second: tuple[int, int]) -> None:
        nonlocal conflict_merges, accepted_merges
        first_root, second_root = find(first), find(second)
        if first_root == second_root:
            return
        if image_sets[first_root].intersection(image_sets[second_root]):
            conflict_merges += 1
            return
        if size[first_root] < size[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        size[first_root] += size[second_root]
        image_sets[first_root].update(image_sets[second_root])
        del image_sets[second_root]
        accepted_merges += 1

    if pair_order not in {"pair_id", "strongest_verified_inliers", "fixed_audit_priority"}:
        raise ValueError(
            "pair_order must be 'pair_id', 'strongest_verified_inliers', or 'fixed_audit_priority'"
        )
    pair_ids, geometries = database.read_two_view_geometries()
    pair_records: list[tuple[int, Any, int]] = []
    for pair_id, geometry in zip(pair_ids, geometries):
        matches = np.asarray(
            getattr(geometry, "inlier_matches", np.empty((0, 2), dtype=np.uint32))
        )
        count = int(matches.shape[0]) if matches.ndim == 2 and matches.shape[1] == 2 else 0
        pair_records.append((int(pair_id), geometry, count))
    if pair_order == "strongest_verified_inliers":
        pair_records.sort(key=lambda item: (-item[2], item[0]))
    elif pair_order == "fixed_audit_priority":
        priority = priority_pair_ids or set()
        pair_records.sort(key=lambda item: (0 if item[0] in priority else 1, -item[2], item[0]))
    else:
        pair_records.sort(key=lambda item: item[0])
    observed_match_rows = 0
    for pair_value, geometry, _ in pair_records:
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
        for first_index, second_index in sorted(matches.astype(np.int64, copy=False).tolist()):
            union((first_id, int(first_index)), (second_id, int(second_index)))
            observed_match_rows += 1

    components: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for node in sorted(parent):
        components[find(node)].append(node)
    yielded_components = 0
    rejected_short = 0
    rejected_large = 0
    rejected_conflicting = 0
    for nodes in sorted(components.values(), key=lambda value: value[0]):
        if len(nodes) < 2 or len(nodes) > max_component_observations:
            if len(nodes) < 2:
                rejected_short += 1
            else:
                rejected_large += 1
            continue
        image_ids = [node[0] for node in nodes]
        if len(image_ids) != len(set(image_ids)):
            rejected_conflicting += 1
            continue
        elements = [__import__("pycolmap").TrackElement(image_id, index) for image_id, index in nodes]
        yielded_components += 1
        yield elements, np.asarray([128, 128, 128], dtype=np.uint8)
    if track_builder_stats is not None:
        track_builder_stats.update(
            {
                "method": "conflict_aware_union_find_observation_components",
                "pair_order": pair_order,
                "priority_pair_count": int(len(priority_pair_ids or set())),
                "allowed_pair_count": int(len(allowed_pair_ids)),
                "observed_verified_match_rows": int(observed_match_rows),
                "accepted_component_merges": int(accepted_merges),
                "same_image_conflict_merges_rejected": int(conflict_merges),
                "component_count": int(len(components)),
                "yielded_component_count": int(yielded_components),
                "rejected_short_component_count": int(rejected_short),
                "rejected_large_component_count": int(rejected_large),
                "rejected_conflicting_component_count": int(rejected_conflicting),
                "maximum_component_observations": int(max_component_observations),
            }
        )


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
        calibrated_translation = record_by_pair[pair_value].get(
            "calibrated_reestimate_translation_first_to_second"
        )
        if calibrated_translation is not None:
            direction = -np.asarray(orientations[second_name], dtype=np.float64).T @ np.asarray(
                calibrated_translation,
                dtype=np.float64,
            ).reshape(3)
            norm = float(np.linalg.norm(direction))
            if np.isfinite(direction).all() and norm > 1e-9:
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
                    "translation_source": "calibrated_reestimate",
                })
                continue
            failures["invalid_calibrated_direction"] += 1
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
            "translation_source": "database_essential_fallback",
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
            # Calibrated re-estimation plus cheirality gives a directed
            # first-to-second translation.  Taking abs(scale) would silently
            # turn an inconsistent candidate center into evidence for the
            # opposite direction.  Keep the scale positive and let the
            # robust residual down-weight an initially wrong-sign edge.
            scales.append(max(scale, 1e-4))
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
    final_negative_direction_count = 0
    for edge in usable:
        direction = np.asarray(edge["direction_world"], dtype=np.float64)
        delta = centers[edge["second"]] - centers[edge["first"]]
        scale = float(np.dot(delta, direction))
        if scale <= 0.0:
            final_negative_direction_count += 1
        final_residuals.append(float(np.linalg.norm(delta - max(scale, 1e-4) * direction)))
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
        "negative_final_direction_count": final_negative_direction_count,
        "positive_final_direction_fraction": float(
            (len(usable) - final_negative_direction_count) / max(len(usable), 1)
        ),
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
        "--source-component-pose-repair",
        action="store_true",
        help=(
            "retain the source model gauge and repair only source pose islands "
            "with conditioned calibrated component links"
        ),
    )
    parser.add_argument(
        "--source-pose-track-repair",
        action="store_true",
        help=(
            "retain every source-model camera pose and rebuild only the image-derived "
            "multi-view tracks from the disposable calibrated graph"
        ),
    )
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
    parser.add_argument(
        "--track-pair-order",
        choices=("pair_id", "strongest_verified_inliers", "fixed_audit_priority"),
        default="pair_id",
        help=(
            "deterministic verified-graph track order; strongest_verified_inliers "
            "builds a maximum-support conflict-free partition; fixed_audit_priority "
            "places the independent fixed audit edges first"
        ),
    )
    parser.add_argument(
        "--priority-pairs-manifest",
        type=Path,
        help="independent fixed audit manifest used only to prioritize measured graph edges",
    )
    args = parser.parse_args()
    classification_payload = json.loads(args.classification.read_text(encoding="utf-8"))
    records = classification_payload.get("classification", classification_payload)
    if not isinstance(records, list):
        raise ValueError("classification payload must contain a classification list")
    priority_pair_ids: set[int] = set()
    priority_manifest_sha256: str | None = None
    if args.priority_pairs_manifest is not None:
        priority_payload = json.loads(args.priority_pairs_manifest.read_text(encoding="utf-8"))
        priority_records = priority_payload.get("pairs", priority_payload) if isinstance(priority_payload, dict) else priority_payload
        if not isinstance(priority_records, list):
            raise ValueError("priority-pairs-manifest must contain a pairs list")
        priority_pair_ids = {int(item["pair_id"]) for item in priority_records if isinstance(item, Mapping) and "pair_id" in item}
        priority_manifest_sha256 = _sha256(args.priority_pairs_manifest)
    if args.track_pair_order == "fixed_audit_priority" and not priority_pair_ids:
        raise ValueError("fixed_audit_priority requires a non-empty --priority-pairs-manifest")
    if args.source_pose_track_repair:
        if args.source_model is None:
            raise ValueError("--source-model is required with --source-pose-track-repair")
        import pycolmap

        source_reconstruction = pycolmap.Reconstruction(str(args.source_model))
        orientations = {
            str(image.name): np.asarray(image.cam_from_world().rotation.matrix(), dtype=np.float64)
            for image in source_reconstruction.images.values()
            if image.has_pose
        }
        # Still derive the allowed graph edges from the calibrated consensus
        # policy.  Only the source poses are retained; no mapper-allocated or
        # pair-local tracks are inherited from the source model.
        _, selection_report = build_rotation_consensus(records, ring_order=args.ring_order)
        report = dict(selection_report)
        report["method"] = "source-pose track rebuild from calibrated graph"
        report["source_pose_model"] = str(args.source_model.resolve())
        report["source_pose_model_sha256"] = _sha256(args.source_model / "images.bin")
    elif args.source_component_pose_repair:
        if args.source_model is None:
            raise ValueError("--source-model is required with --source-component-pose-repair")
        import pycolmap

        source_reconstruction = pycolmap.Reconstruction(str(args.source_model))
        reference_orientations = {
            str(image.name): np.asarray(image.cam_from_world().rotation.matrix(), dtype=np.float64)
            for image in source_reconstruction.images.values()
            if image.has_pose
        }
        orientations, pose_report = build_source_component_pose_repair(
            records,
            reference_orientations,
            ring_order=args.ring_order,
        )
        # Keep the existing calibrated rotation-consensus selector as the
        # versioned graph policy; the new architecture changes only the pose
        # field used to write the candidate, not which verified edges are
        # eligible for mapping.
        _, selection_report = build_rotation_consensus(records, ring_order=args.ring_order)
        report = dict(selection_report)
        report["method"] = "source-gauge component pose repair with calibrated graph mapping selection"
        report["pose_repair"] = pose_report
    else:
        orientations, report = build_rotation_consensus(records, ring_order=args.ring_order)
    report["classification_path"] = str(args.classification.resolve())
    report["classification_sha256"] = _sha256(args.classification)
    report["orientation_count"] = len(orientations)
    report["track_provenance"] = {
        "source": "verified_disposable_sqlite_graph",
        "independent_graph": bool(args.snapshot_manifest is not None),
        "independent_graph_sha256": report["classification_sha256"],
        "constructed_pairwise_tracks": bool(args.pairwise_tracks),
        "track_builder": "disjoint_pair_allocator" if args.pairwise_tracks else "union_find_observation_components",
        "track_pair_order": args.track_pair_order,
        "priority_pairs_manifest": str(args.priority_pairs_manifest.resolve()) if args.priority_pairs_manifest else None,
        "priority_pairs_manifest_sha256": priority_manifest_sha256,
        "priority_pair_count": len(priority_pair_ids),
        "pose_repair": (
            "source_pose_track_rebuild"
            if args.source_pose_track_repair
            else "source_component"
            if args.source_component_pose_repair
            else "rotation_consensus"
        ),
    }
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
                        track_pair_order=args.track_pair_order,
                        priority_pair_ids=priority_pair_ids,
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
