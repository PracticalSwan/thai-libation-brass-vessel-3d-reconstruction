"""Select a conflict-free image-derived mapper graph.

The calibrated classification and fixed independent audit are evidence, not a
permission to let incompatible correspondence cycles enter track
establishment.  This step works on a disposable copy of the manifest-bound
SQLite snapshot and greedily retains complete verified pair geometries only
when the observation union-find remains *simple by image*: no connected
observation component may contain two keypoints from the same image.

The default policy is deterministic and evidence ordered: same-ring edges,
locally cycle-consistent edges, short acquisition-local frame gaps, high
calibrated inlier support, usable triangulation angle, and low homography
support are preferred.  A versioned ``consensus_inlier_first`` policy is also
available when an independently built rotation-consensus report is supplied;
it prioritizes robust calibrated consensus inliers before same-ring locality.
That policy exists to test the observed g10/g11 pose-island hypothesis, not to
silently discard contradictory audit evidence.  Rejected pairs remain in the
immutable classification/audit and are reported as mapping-negative evidence;
they are not deleted from source evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v4_repair import (  # noqa: E402
    PAIR_BASE,
    colmap_pair_id,
    create_disposable_sqlite_from_manifest,
    decode_colmap_pair_id,
    load_canonical_sqlite_snapshot_manifest,
)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pair_key(first: str, second: str) -> tuple[str, str]:
    return (str(first), str(second))


def _records_by_pair(classification: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    records = classification.get("classification")
    if not isinstance(records, list):
        raise ValueError("classification payload must contain classification records")
    selected: dict[int, dict[str, Any]] = {}
    for raw in records:
        if not isinstance(raw, Mapping) or not raw.get("well_conditioned_calibrated"):
            continue
        pair_id = int(raw.get("pair_id", 0))
        if pair_id <= 0:
            raise ValueError("well-conditioned classification record lacks pair_id")
        if pair_id in selected:
            raise ValueError(f"duplicate calibrated classification pair_id: {pair_id}")
        selected[pair_id] = dict(raw)
    if not selected:
        raise ValueError("classification contains no well-conditioned calibrated records")
    mapping_records = classification.get("mapping_records")
    if not isinstance(mapping_records, list):
        raise ValueError("classification payload must contain mapping_records")
    mapping_ids: set[int] = set()
    for raw in mapping_records:
        if not isinstance(raw, Mapping):
            raise ValueError("mapping_records must contain objects")
        independent = raw.get("independent_audit_record")
        if not isinstance(independent, Mapping):
            raise ValueError("mapping record lacks independent audit evidence")
        # v3 stores the pair id in the corresponding independent audit record.
        pair_id = int(raw.get("pair_id") or independent.get("pair_id") or 0)
        if pair_id <= 0:
            first, second = str(raw.get("first", "")), str(raw.get("second", ""))
            record = next(
                (
                    item
                    for item in selected.values()
                    if str(item.get("first")) == first and str(item.get("second")) == second
                ),
                None,
            )
            pair_id = int(record.get("pair_id", 0)) if record else 0
        if pair_id <= 0:
            raise ValueError("mapping record lacks a resolvable pair_id")
        mapping_ids.add(pair_id)
    if mapping_ids != set(selected):
        raise ValueError(
            "classification mapping_records do not exactly match well-conditioned records: "
            f"mapping={len(mapping_ids)} conditioned={len(selected)}"
        )
    return selected


def _edge_score(
    record: Mapping[str, Any],
    *,
    consensus_inlier_first: bool = False,
) -> tuple[int | float, ...]:
    """Return the fixed evidence ordering used by the selector.

    ``consensus_inlier_first`` is intentionally opt-in and only meaningful
    when the record was annotated from the independent consensus report.  It
    gives independently retained robust consensus inliers precedence over the
    same-ring/high-inlier heuristic that previously selected contradictory
    cross-ring bridges first.
    """

    same_ring = int(bool(record.get("same_ring")))
    cycle_consistent = int(str(record.get("local_cycle_status", "")) == "consistent")
    frame_distance = int(record.get("frame_distance") or 10**9)
    inliers = int(record.get("calibrated_inliers") or 0)
    triangulation = float(record.get("calibrated_tri_angle_deg") or 0.0)
    homography = float(record.get("homography_inlier_fraction") or 1.0)
    consensus_inlier = int(
        str(record.get("_consensus_status", "")) == "retained_robust_consensus_inlier"
    )
    if consensus_inlier_first:
        return (
            consensus_inlier,
            same_ring,
            cycle_consistent,
            -frame_distance,
            inliers,
            triangulation,
            -homography,
            -int(record.get("pair_id") or 0),
        )
    return (
        same_ring,
        cycle_consistent,
        -frame_distance,
        inliers,
        triangulation,
        -homography,
        -int(record.get("pair_id") or 0),
    )


class _ObservationUnionFind:
    """Rollback-able union-find with a no-duplicate-image invariant."""

    def __init__(self, nodes: Iterable[int]):
        values = list(nodes)
        self.parent = {node: node for node in values}
        self.size = {node: 1 for node in values}
        self.images = {node: {int(node >> 32)} for node in values}

    def find(self, node: int) -> int:
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        return root

    def checkpoint(self) -> int:
        return 0

    def try_union_pairs(self, pairs: np.ndarray) -> bool:
        """Tentatively union all observation pairs, rolling back on conflict."""

        changes: list[tuple[int, int, int, list[int]]] = []
        for first, second in np.asarray(pairs, dtype=np.uint32).reshape(-1, 2):
            first_root = self.find(int(first))
            second_root = self.find(int(second))
            if first_root == second_root:
                continue
            if not self.images[first_root].isdisjoint(self.images[second_root]):
                for root, child, old_size, added in reversed(changes):
                    self.parent[child] = child
                    self.size[root] = old_size
                    self.images[root].difference_update(added)
                return False
            if self.size[first_root] < self.size[second_root]:
                first_root, second_root = second_root, first_root
            added = list(self.images[second_root])
            old_size = self.size[first_root]
            self.parent[second_root] = first_root
            self.size[first_root] += self.size[second_root]
            self.images[first_root].update(added)
            changes.append((first_root, second_root, old_size, added))
        return True

    def duplicate_component_count(self) -> int:
        return sum(
            1
            for root, image_ids in self.images.items()
            if self.parent[root] == root and len(image_ids) < self.size[root]
        )


def _read_verified_edges(
    database_path: Path,
    records: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, int], list[tuple[int, dict[str, Any], np.ndarray]]]:
    uri = f"file:{database_path.as_posix()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        image_ids = {
            str(name): int(image_id)
            for image_id, name in connection.execute("SELECT image_id, name FROM images")
        }
        edges: list[tuple[int, dict[str, Any], np.ndarray]] = []
        for pair_id, record in records.items():
            row = connection.execute(
                "SELECT rows, cols, data FROM two_view_geometries WHERE pair_id = ? AND rows > 0",
                (int(pair_id),),
            ).fetchone()
            if row is None:
                raise ValueError(f"well-conditioned pair is absent from two_view_geometries: {pair_id}")
            rows, cols, data = int(row[0]), int(row[1]), row[2]
            if cols != 2 or data is None:
                raise ValueError(f"invalid verified match payload for pair {pair_id}")
            matches = np.frombuffer(data, dtype=np.uint32).reshape(rows, cols).copy()
            first, second = str(record["first"]), str(record["second"])
            expected = colmap_pair_id(image_ids[first], image_ids[second])
            if expected != int(pair_id):
                raise ValueError(f"classification/database pair ID mismatch for {first} -> {second}")
            edges.append((int(pair_id), dict(record), matches))
        return image_ids, edges
    finally:
        connection.close()


def _select_conflict_free_edges(
    edges: Sequence[tuple[int, Mapping[str, Any], np.ndarray]],
    *,
    consensus_inlier_first: bool = False,
) -> dict[str, Any]:
    nodes: set[int] = set()
    scored: list[tuple[tuple[int, ... | float], int, Mapping[str, Any], np.ndarray]] = []
    for pair_id, record, matches in edges:
        first_id, second_id = decode_colmap_pair_id(pair_id)
        encoded = np.empty_like(matches, dtype=np.uint64)
        encoded[:, 0] = (np.uint64(first_id) << np.uint64(32)) | matches[:, 0].astype(np.uint64)
        encoded[:, 1] = (np.uint64(second_id) << np.uint64(32)) | matches[:, 1].astype(np.uint64)
        nodes.update(int(value) for value in encoded.reshape(-1).tolist())
        # Store encoded observation pairs; the union-find works on one integer
        # namespace, which keeps image identity in the upper 32 bits.
        scored.append(
            (
                _edge_score(record, consensus_inlier_first=consensus_inlier_first),
                pair_id,
                record,
                encoded,
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    union_find = _ObservationUnionFind(nodes)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    accepted_matches = 0
    for score, pair_id, record, encoded in scored:
        # Convert the uint64 observation namespace to a uint32-compatible
        # Python integer array without truncating image IDs.
        pairs = np.asarray(encoded, dtype=np.uint64)
        ok = True
        changes: list[tuple[int, int, int, list[int]]] = []
        for first, second in pairs:
            first_root = union_find.find(int(first))
            second_root = union_find.find(int(second))
            if first_root == second_root:
                continue
            if not union_find.images[first_root].isdisjoint(union_find.images[second_root]):
                ok = False
                break
            if union_find.size[first_root] < union_find.size[second_root]:
                first_root, second_root = second_root, first_root
            added = list(union_find.images[second_root])
            old_size = union_find.size[first_root]
            union_find.parent[second_root] = first_root
            union_find.size[first_root] += union_find.size[second_root]
            union_find.images[first_root].update(added)
            changes.append((first_root, second_root, old_size, added))
        if not ok:
            for root, child, old_size, added in reversed(changes):
                union_find.parent[child] = child
                union_find.size[root] = old_size
                union_find.images[root].difference_update(added)
            rejected.append(
                {
                    "pair_id": int(pair_id),
                    "first": str(record["first"]),
                    "second": str(record["second"]),
                    "reason": "observation_component_image_conflict",
                    "same_ring": bool(record.get("same_ring")),
                    "consensus_status": str(record.get("_consensus_status", "")),
                    "calibrated_inliers": int(record.get("calibrated_inliers") or 0),
                    "frame_distance": int(record.get("frame_distance") or 0),
                }
            )
            continue
        accepted.append(
            {
                "pair_id": int(pair_id),
                "first": str(record["first"]),
                "second": str(record["second"]),
                "same_ring": bool(record.get("same_ring")),
                "consensus_status": str(record.get("_consensus_status", "")),
                "consensus_residual_deg": (
                    float(record["_consensus_residual_deg"])
                    if record.get("_consensus_residual_deg") is not None
                    else None
                ),
                "ring": str(record.get("ring", "")),
                "second_ring": str(record.get("second_ring", "")),
                "frame_distance": int(record.get("frame_distance") or 0),
                "calibrated_inliers": int(record.get("calibrated_inliers") or 0),
                "calibrated_tri_angle_deg": float(record.get("calibrated_tri_angle_deg") or 0.0),
                "homography_inlier_fraction": float(record.get("homography_inlier_fraction") or 1.0),
                "selection_score": list(score),
            }
        )
        accepted_matches += int(len(pairs))
    roots = [root for root in union_find.parent if union_find.parent[root] == root]
    duplicate_components = union_find.duplicate_component_count()
    return {
        "accepted": accepted,
        "rejected": rejected,
        "accepted_match_rows": accepted_matches,
        "observation_node_count": len(nodes),
        "observation_component_count": len(roots),
        "observation_duplicate_component_count": duplicate_components,
        "observation_duplicate_image_extra_count": 0,
    }


def _graph_components(
    image_ids: Mapping[str, int],
    accepted: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    parent = {int(value): int(value) for value in image_ids.values()}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for record in accepted:
        union(image_ids[str(record["first"])], image_ids[str(record["second"])])
    components: dict[int, list[str]] = defaultdict(list)
    for name, image_id in image_ids.items():
        components[find(int(image_id))].append(str(name))
    ring_parent: dict[str, str] = {}
    for record in accepted:
        ring_parent.setdefault(str(record["ring"]), str(record["ring"]))
        ring_parent.setdefault(str(record["second_ring"]), str(record["second_ring"]))

    def rfind(ring: str) -> str:
        while ring_parent[ring] != ring:
            ring_parent[ring] = ring_parent[ring_parent[ring]]
            ring = ring_parent[ring]
        return ring

    for record in accepted:
        first, second = rfind(str(record["ring"])), rfind(str(record["second_ring"]))
        if first != second:
            ring_parent[second] = first
    ring_components: dict[str, list[str]] = defaultdict(list)
    for ring in sorted(ring_parent):
        ring_components[rfind(ring)].append(ring)
    return {
        "image_components": sorted(
            (sorted(values) for values in components.values()),
            key=lambda value: (-len(value), value),
        ),
        "ring_components": sorted(
            (sorted(values) for values in ring_components.values()),
            key=lambda value: (-len(value), value),
        ),
    }


def build_conflict_free_graph(
    snapshot_manifest: str | Path,
    classification_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    excluded_pair_ids: Iterable[int] = (),
    consensus_path: str | Path | None = None,
    exclude_consensus_outlier_transitions: Iterable[str] = (),
    consensus_inlier_first: bool = False,
) -> dict[str, Any]:
    manifest = load_canonical_sqlite_snapshot_manifest(snapshot_manifest)
    classification = json.loads(Path(classification_path).read_text(encoding="utf-8"))
    records = _records_by_pair(classification)
    output = Path(output_path).resolve()
    report_target = Path(report_path).resolve()
    if output.exists() or report_target.exists():
        raise FileExistsError("conflict-free graph output/report already exists; preserve it")
    canonical_sha_before = sha256_file(manifest["canonical_path"])
    excluded_ids = {int(value) for value in excluded_pair_ids}
    transition_exclusions = {
        frozenset(part.strip() for part in str(value).split(":", 1))
        for value in exclude_consensus_outlier_transitions
        if ":" in str(value)
    }
    if consensus_inlier_first and consensus_path is None:
        raise ValueError("consensus_inlier_first requires an independent consensus report")
    consensus_by_pair: dict[int, Mapping[str, Any]] = {}
    if consensus_path is not None:
        consensus_payload = json.loads(Path(consensus_path).read_text(encoding="utf-8"))
        per_edge = consensus_payload.get("per_edge")
        if not isinstance(per_edge, list):
            raise ValueError("consensus report must contain per_edge")
        for item in per_edge:
            if isinstance(item, Mapping) and item.get("pair_id") is not None:
                consensus_by_pair[int(item["pair_id"])] = item
    with tempfile.TemporaryDirectory(prefix="v4_conflict_graph_db_") as temporary:
        working_path = Path(temporary) / "working.db"
        disposable = create_disposable_sqlite_from_manifest(snapshot_manifest, working_path)
        image_ids, edges = _read_verified_edges(working_path, records)
    unknown_exclusions = sorted(excluded_ids - {pair_id for pair_id, _, _ in edges})
    if unknown_exclusions:
        raise ValueError(f"excluded pair IDs are absent from the conditioned graph: {unknown_exclusions}")
    consensus_excluded_ids: set[int] = set()
    if transition_exclusions:
        for pair_id, record, _ in edges:
            consensus = consensus_by_pair.get(int(pair_id), {})
            transition = frozenset(
                (str(record.get("ring", "")), str(record.get("second_ring", "")))
            )
            if (
                not bool(record.get("same_ring"))
                and transition in transition_exclusions
                and consensus.get("mapping_status") == "retained_robust_consensus_outlier"
            ):
                consensus_excluded_ids.add(int(pair_id))
    excluded_ids.update(consensus_excluded_ids)
    edges = [edge for edge in edges if edge[0] not in excluded_ids]
    if consensus_by_pair:
        annotated_edges: list[tuple[int, dict[str, Any], np.ndarray]] = []
        for pair_id, record, matches in edges:
            annotated = dict(record)
            consensus = consensus_by_pair.get(int(pair_id))
            if consensus is not None:
                annotated["_consensus_status"] = str(consensus.get("mapping_status", ""))
                residual = consensus.get("synchronization_residual_deg")
                if residual is None:
                    residual = consensus.get("consensus_residual_deg")
                annotated["_consensus_residual_deg"] = residual
            else:
                annotated["_consensus_status"] = ""
                annotated["_consensus_residual_deg"] = None
            annotated_edges.append((pair_id, annotated, matches))
        edges = annotated_edges
    selected = _select_conflict_free_edges(
        edges,
        consensus_inlier_first=consensus_inlier_first,
    )
    graph = _graph_components(image_ids, selected["accepted"])
    expected_names = sorted(image_ids)
    if len(expected_names) != 372:
        raise ValueError(f"expected 372 images, found {len(expected_names)}")
    if len(graph["image_components"]) != 1:
        raise ValueError("conflict-free graph is not connected over all registered images")
    if len(graph["ring_components"]) != 1:
        raise ValueError("conflict-free graph is not connected over all selected rings")
    if selected["observation_duplicate_component_count"] != 0:
        raise ValueError("selector invariant failed: duplicate-image observation component remains")
    pairs = [
        {
            "pair_id": int(record["pair_id"]),
            "first": str(record["first"]),
            "second": str(record["second"]),
        }
        for record in selected["accepted"]
    ]
    pair_ids_sha = hashlib.sha256(
        json.dumps([int(pair["pair_id"]) for pair in pairs], separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "method": (
            "deterministic observation-conflict-free calibrated graph selection; "
            "complete verified pair geometries are retained only when no connected "
            "observation component contains two keypoints from the same image"
        ),
        "snapshot_manifest": str(Path(snapshot_manifest).resolve()),
        "snapshot_manifest_sha256": sha256_file(snapshot_manifest),
        "canonical_snapshot_sha256": manifest["canonical_sha256"],
        "canonical_creation_logical_sha256": manifest["creation_logical_sha256"],
        "canonical_sha256_before": canonical_sha_before,
        "canonical_snapshot_hash_stable": canonical_sha_before == manifest["canonical_sha256"],
        "classification_path": str(Path(classification_path).resolve()),
        "classification_sha256": sha256_file(classification_path),
        "source_conditioned_pair_count": len(records),
        "source_verified_inlier_rows": int(sum(len(edge[2]) for edge in edges)),
        "excluded_pair_ids": sorted(excluded_ids),
        "excluded_pair_count": len(excluded_ids),
        "consensus_report": str(Path(consensus_path).resolve()) if consensus_path is not None else None,
        "consensus_report_sha256": sha256_file(consensus_path) if consensus_path is not None else None,
        "consensus_inlier_first": bool(consensus_inlier_first),
        "consensus_outlier_transition_exclusions": sorted(
            ":".join(sorted(value)) for value in transition_exclusions
        ),
        "consensus_outlier_excluded_pair_ids": sorted(consensus_excluded_ids),
        "excluded_pair_policy": (
            "explicit evidence-bound mapping exclusion; pairs remain in the fixed independent audit"
        ),
        "selection_policy": {
            "consensus_inlier_first": bool(consensus_inlier_first),
            "consensus_status_required": (
                "retained_robust_consensus_inlier" if consensus_inlier_first else None
            ),
            "same_ring_first": True,
            "local_cycle_consistent_first": True,
            "short_frame_gap_first": True,
            "calibrated_inliers_descending": True,
            "triangulation_angle_descending": True,
            "homography_fraction_ascending": True,
            "pair_id_tiebreak": "ascending",
        },
        "accepted_pair_count": len(pairs),
        "accepted_pair_ids_sha256": pair_ids_sha,
        "accepted_match_rows": selected["accepted_match_rows"],
        "rejected_pair_count": len(selected["rejected"]),
        "rejected_reason_counts": dict(Counter(item["reason"] for item in selected["rejected"])),
        "observation_node_count": selected["observation_node_count"],
        "observation_component_count": selected["observation_component_count"],
        "observation_duplicate_component_count": selected["observation_duplicate_component_count"],
        "observation_duplicate_image_extra_count": selected["observation_duplicate_image_extra_count"],
        "graph_connectivity": graph,
        "pairs": pairs,
        "accepted_edges": selected["accepted"],
        "rejected_edges": selected["rejected"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({**payload, "pairs": pairs}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["output_path"] = str(output)
    payload["output_sha256"] = sha256_file(output)
    report_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--classification", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--exclude-pair-id",
        action="append",
        type=int,
        default=[],
        help="Evidence-identified pair ID to exclude from mapping while retaining it in audit evidence.",
    )
    parser.add_argument("--consensus-report", type=Path)
    parser.add_argument(
        "--exclude-consensus-outlier-transition",
        action="append",
        default=[],
        help="Ring transition written as RING_A:RING_B; exclude all consensus-outlier edges for it.",
    )
    parser.add_argument(
        "--consensus-inlier-first",
        action="store_true",
        help=(
            "Rank independently retained robust consensus inliers before "
            "same-ring locality; requires --consensus-report."
        ),
    )
    args = parser.parse_args()
    result = build_conflict_free_graph(
        args.snapshot_manifest,
        args.classification,
        args.output,
        args.report,
        excluded_pair_ids=args.exclude_pair_id,
        consensus_path=args.consensus_report,
        exclude_consensus_outlier_transitions=args.exclude_consensus_outlier_transition,
        consensus_inlier_first=args.consensus_inlier_first,
    )
    print(json.dumps({
        "accepted_pair_count": result["accepted_pair_count"],
        "rejected_pair_count": result["rejected_pair_count"],
        "image_components": len(result["graph_connectivity"]["image_components"]),
        "ring_components": len(result["graph_connectivity"]["ring_components"]),
        "output": result["output_path"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
