"""Build a match-level conflict-free disposable COLMAP database.

The exact calibrated graph can contain individually verified correspondences
whose union creates a track component with two keypoints from one image.  The
earlier edge-level selector had to discard an entire pair when one such row
conflicted, which removed too many local constraints and created pose-island
diagnostics.  This script keeps every retained classified pair, but retains
only a deterministic, evidence-ordered subset of its verified inlier rows.

Only a disposable copy is opened by SQLite.  The manifest-bound canonical
snapshot is copied by bytes, pruned to the exact retained pair-ID set, and
then the match rows are filtered.  Rejected correspondence rows remain in the
immutable source snapshot and are recorded as disposable-copy provenance.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
    copy_frozen_sqlite_snapshot,
    create_disposable_sqlite_from_manifest,
    load_canonical_sqlite_snapshot_manifest,
    sha256_file,
    sqlite_logical_digest,
    validate_exact_mapper_graph,
)
from scripts.build_v4_conflict_free_graph import (  # noqa: E402
    _read_verified_edges,
    _records_by_pair,
)


class _ObservationUnionFind:
    """Union-find with a one-keypoint-per-image component invariant."""

    def __init__(self) -> None:
        self._lookup: dict[int, int] = {}
        self._parent: list[int] = []
        self._size: list[int] = []
        self._image_mask: list[int] = []

    def _node(self, observation: int, image_id: int) -> int:
        index = self._lookup.get(int(observation))
        if index is None:
            index = len(self._parent)
            self._lookup[int(observation)] = index
            self._parent.append(index)
            self._size.append(1)
            self._image_mask.append(1 << int(image_id))
        return index

    def _find(self, index: int) -> int:
        root = index
        while self._parent[root] != root:
            root = self._parent[root]
        while index != root:
            previous = self._parent[index]
            self._parent[index] = root
            index = previous
        return root

    def add_pair(self, first: tuple[int, int], second: tuple[int, int]) -> bool:
        first_image, first_keypoint = first
        second_image, second_keypoint = second
        first_observation = (int(first_image) << 32) | int(first_keypoint)
        second_observation = (int(second_image) << 32) | int(second_keypoint)
        first_node = self._node(first_observation, first_image)
        second_node = self._node(second_observation, second_image)
        first_root = self._find(first_node)
        second_root = self._find(second_node)
        if first_root == second_root:
            return True
        if self._image_mask[first_root] & self._image_mask[second_root]:
            return False
        if self._size[first_root] < self._size[second_root]:
            first_root, second_root = second_root, first_root
        self._parent[second_root] = first_root
        self._size[first_root] += self._size[second_root]
        self._image_mask[first_root] |= self._image_mask[second_root]
        return True

    @property
    def node_count(self) -> int:
        return len(self._parent)


def _consensus_status_by_pair(consensus_path: str | Path | None) -> dict[int, str]:
    if consensus_path is None:
        return {}
    payload = json.loads(Path(consensus_path).read_text(encoding="utf-8"))
    rows = payload.get("per_edge")
    if not isinstance(rows, list):
        raise ValueError("consensus report must contain per_edge")
    result: dict[int, str] = {}
    for raw in rows:
        if not isinstance(raw, Mapping) or raw.get("pair_id") is None:
            continue
        result[int(raw["pair_id"])] = str(raw.get("mapping_status", ""))
    return result


def _match_score(
    pair_id: int,
    record: Mapping[str, Any],
    consensus_status: Mapping[int, str],
) -> tuple[int | float, ...]:
    """Same-ring first, then independent consensus for cross-ring bridges."""

    same_ring = int(bool(record.get("same_ring")))
    consensus_inlier = int(
        consensus_status.get(int(pair_id), "") == "retained_robust_consensus_inlier"
    )
    cycle_consistent = int(str(record.get("local_cycle_status", "")) == "consistent")
    frame_distance = int(record.get("frame_distance") or 10**9)
    inliers = int(record.get("calibrated_inliers") or 0)
    triangulation = float(record.get("calibrated_tri_angle_deg") or 0.0)
    homography = float(record.get("homography_inlier_fraction") or 1.0)
    return (
        same_ring,
        consensus_inlier,
        cycle_consistent,
        -frame_distance,
        inliers,
        triangulation,
        -homography,
        -int(pair_id),
    )


def _transition_key(record: Mapping[str, Any]) -> str:
    rings = sorted((str(record.get("ring", "")), str(record.get("second_ring", ""))))
    return ":".join(rings)


def select_conflict_free_match_rows(
    image_ids: Mapping[str, int],
    edges: Sequence[tuple[int, Mapping[str, Any], np.ndarray]],
    *,
    consensus_status: Mapping[int, str] | None = None,
    minimum_matches_per_pair: int = 15,
    consensus_outlier_max_rows: int | None = None,
    consensus_outlier_max_rows_by_transition: Mapping[str, int] | None = None,
    consensus_outlier_max_rows_by_pair: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Retain deterministic rows while preserving every pair ID.

    The input rows are already verified two-view inliers from the disposable
    SQLite copy.  A row is retained only when adding it cannot make a connected
    observation component contain two keypoints from one image.  Rows from
    higher-evidence pairs are considered first, but every pair must retain at
    least ``minimum_matches_per_pair`` rows or the operation fails closed.
    """

    if minimum_matches_per_pair < 1:
        raise ValueError("minimum_matches_per_pair must be positive")
    if consensus_outlier_max_rows is not None and consensus_outlier_max_rows < minimum_matches_per_pair:
        raise ValueError(
            "consensus_outlier_max_rows must be at least minimum_matches_per_pair"
        )
    transition_caps = {
        str(key): int(value)
        for key, value in (consensus_outlier_max_rows_by_transition or {}).items()
    }
    if any(value < minimum_matches_per_pair for value in transition_caps.values()):
        raise ValueError(
            "every consensus-outlier transition cap must be at least minimum_matches_per_pair"
        )
    pair_caps = {
        int(key): int(value)
        for key, value in (consensus_outlier_max_rows_by_pair or {}).items()
    }
    if any(value < minimum_matches_per_pair for value in pair_caps.values()):
        raise ValueError(
            "every consensus-outlier pair cap must be at least minimum_matches_per_pair"
        )
    status = dict(consensus_status or {})
    ordered = sorted(
        edges,
        key=lambda item: _match_score(int(item[0]), item[1], status),
        reverse=True,
    )
    union_find = _ObservationUnionFind()
    retained: dict[int, np.ndarray] = {}
    per_pair: list[dict[str, Any]] = []
    conflict_rows = 0
    source_rows = 0
    retained_rows = 0
    capped_rows = 0
    for pair_id, record, raw_matches in ordered:
        first_id = int(image_ids[str(record["first"])])
        second_id = int(image_ids[str(record["second"])])
        matches = np.asarray(raw_matches, dtype=np.uint32).reshape(-1, 2)
        source_rows += int(len(matches))
        kept: list[tuple[int, int]] = []
        is_consensus_outlier = (
            status.get(int(pair_id), "") == "retained_robust_consensus_outlier"
        )
        pair_cap = pair_caps.get(
            int(pair_id), transition_caps.get(_transition_key(record), consensus_outlier_max_rows)
        )
        for first_keypoint, second_keypoint in matches:
            if is_consensus_outlier and pair_cap is not None and len(kept) >= pair_cap:
                capped_rows += 1
                continue
            if union_find.add_pair(
                (first_id, int(first_keypoint)),
                (second_id, int(second_keypoint)),
            ):
                kept.append((int(first_keypoint), int(second_keypoint)))
            else:
                conflict_rows += 1
        if len(kept) < minimum_matches_per_pair:
            raise ValueError(
                "match-level conflict filtering cannot preserve mapper support for "
                f"pair {pair_id}: kept {len(kept)} < {minimum_matches_per_pair}"
            )
        kept_array = np.asarray(kept, dtype="<u4").reshape(-1, 2)
        retained[int(pair_id)] = kept_array
        retained_rows += int(len(kept_array))
        per_pair.append(
            {
                "pair_id": int(pair_id),
                "first": str(record["first"]),
                "second": str(record["second"]),
                "same_ring": bool(record.get("same_ring")),
                "consensus_status": status.get(int(pair_id), ""),
                "source_inlier_rows": int(len(matches)),
                "retained_inlier_rows": int(len(kept_array)),
                "pruned_inlier_rows": int(len(matches) - len(kept_array)),
                "consensus_outlier_row_limit": pair_cap if is_consensus_outlier else None,
                "capped_by_consensus_outlier_limit": bool(is_consensus_outlier and pair_cap is not None),
            }
        )
    if set(retained) != {int(pair_id) for pair_id, _, _ in edges}:
        raise AssertionError("match-level filter changed the exact pair-ID set")
    return {
        "retained_matches": retained,
        "per_pair": sorted(per_pair, key=lambda item: int(item["pair_id"])),
        "pair_count": len(retained),
        "source_inlier_rows": source_rows,
        "retained_inlier_rows": retained_rows,
        "pruned_inlier_rows": conflict_rows,
        "capped_rows": capped_rows,
        "consensus_outlier_max_rows": consensus_outlier_max_rows,
        "consensus_outlier_max_rows_by_transition": dict(sorted(transition_caps.items())),
        "consensus_outlier_max_rows_by_pair": dict(sorted(pair_caps.items())),
        "minimum_retained_rows": min(int(len(value)) for value in retained.values()),
        "maximum_retained_rows": max(int(len(value)) for value in retained.values()),
        "observation_node_count": union_find.node_count,
        "method": (
            "deterministic match-level observation conflict filtering; same-ring "
            "evidence first, then independent consensus-inlier cross-ring bridges"
        ),
    }


def _update_pair_rows(
    connection: sqlite3.Connection,
    table: str,
    retained_matches: Mapping[int, np.ndarray],
) -> None:
    if table not in {"matches", "two_view_geometries"}:
        raise ValueError(f"unsupported COLMAP pair table: {table}")
    for pair_id, matches in retained_matches.items():
        rows = int(len(matches))
        payload = sqlite3.Binary(np.asarray(matches, dtype="<u4").reshape(rows, 2).tobytes())
        connection.execute(
            f"UPDATE {table} SET rows = ?, cols = 2, data = ? WHERE pair_id = ?",
            (rows, payload, int(pair_id)),
        )


def build_match_conflict_free_database(
    snapshot_manifest: str | Path,
    classification_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    consensus_path: str | Path | None = None,
    minimum_matches_per_pair: int = 15,
    consensus_outlier_max_rows: int | None = None,
    consensus_outlier_max_rows_by_transition: Mapping[str, int] | None = None,
    consensus_outlier_max_rows_by_pair: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Create and verify an exact-pair disposable DB with filtered rows."""

    manifest = load_canonical_sqlite_snapshot_manifest(snapshot_manifest)
    canonical = Path(manifest["canonical_path"]).resolve()
    output = Path(output_path).resolve()
    report_target = Path(report_path).resolve()
    if output == canonical:
        raise ValueError("match-filtered output may not be the canonical snapshot")
    if output.exists() or report_target.exists():
        raise FileExistsError("match-filtered output/report already exists; preserve it")
    classification = json.loads(Path(classification_path).read_text(encoding="utf-8"))
    records = _records_by_pair(classification)
    status = _consensus_status_by_pair(consensus_path)
    canonical_before = sha256_file(canonical)

    with tempfile.TemporaryDirectory(prefix="v4_match_conflict_graph_db_") as temporary:
        seed = Path(temporary) / "seed.db"
        create_disposable_sqlite_from_manifest(snapshot_manifest, seed)
        image_ids, edges = _read_verified_edges(seed, records)
        selected = select_conflict_free_match_rows(
            image_ids,
            edges,
            consensus_status=status,
            minimum_matches_per_pair=minimum_matches_per_pair,
            consensus_outlier_max_rows=consensus_outlier_max_rows,
            consensus_outlier_max_rows_by_transition=consensus_outlier_max_rows_by_transition,
            consensus_outlier_max_rows_by_pair=consensus_outlier_max_rows_by_pair,
        )

    copy_frozen_sqlite_snapshot(canonical, output)
    expected_pair_ids = sorted(int(pair_id) for pair_id in selected["retained_matches"])
    connection = sqlite3.connect(str(output))
    try:
        connection.execute("CREATE TEMP TABLE selected_match_pairs(pair_id INTEGER PRIMARY KEY)")
        connection.executemany(
            "INSERT INTO selected_match_pairs(pair_id) VALUES (?)",
            ((pair_id,) for pair_id in expected_pair_ids),
        )
        connection.execute(
            "DELETE FROM matches WHERE pair_id NOT IN (SELECT pair_id FROM selected_match_pairs)"
        )
        connection.execute(
            "DELETE FROM two_view_geometries WHERE pair_id NOT IN (SELECT pair_id FROM selected_match_pairs)"
        )
        _update_pair_rows(connection, "matches", selected["retained_matches"])
        _update_pair_rows(connection, "two_view_geometries", selected["retained_matches"])
        connection.commit()
    finally:
        connection.close()

    preflight = validate_exact_mapper_graph(
        output,
        expected_pair_ids,
        canonical_path=canonical,
        require_raw_matches=True,
    )
    canonical_after = sha256_file(canonical)
    if canonical_after != canonical_before or canonical_after != manifest["canonical_sha256"]:
        raise RuntimeError("canonical SQLite snapshot changed while filtering disposable matches")
    if not preflight["passed"]:
        raise RuntimeError(f"match-filtered exact graph preflight failed: {preflight}")

    report = {
        "schema_version": 1,
        "status": "complete",
        "method": selected["method"],
        "snapshot_manifest": str(Path(snapshot_manifest).resolve()),
        "snapshot_manifest_sha256": manifest["manifest_sha256"],
        "canonical_sqlite_opened": False,
        "canonical_snapshot_sha256": manifest["canonical_sha256"],
        "canonical_creation_logical_sha256": manifest["creation_logical_sha256"],
        "canonical_sha256_before": canonical_before,
        "canonical_sha256_after": canonical_after,
        "canonical_snapshot_hash_stable": True,
        "classification_path": str(Path(classification_path).resolve()),
        "classification_sha256": sha256_file(classification_path),
        "fixed_audit_pair_count": classification.get("fixed_pair_count"),
        "fixed_audit_pair_ids_sha256": classification.get("fixed_pair_ids_sha256"),
        "raw_audit_report": classification.get("raw_audit_report"),
        "raw_audit_report_sha256": classification.get("raw_audit_report_sha256"),
        "consensus_path": str(Path(consensus_path).resolve()) if consensus_path is not None else None,
        "consensus_sha256": sha256_file(consensus_path) if consensus_path is not None else None,
        "minimum_matches_per_pair": int(minimum_matches_per_pair),
        "consensus_outlier_max_rows": consensus_outlier_max_rows,
        "consensus_outlier_max_rows_by_transition": dict(
            sorted((consensus_outlier_max_rows_by_transition or {}).items())
        ),
        "consensus_outlier_max_rows_by_pair": dict(
            sorted((consensus_outlier_max_rows_by_pair or {}).items())
        ),
        "pair_count": selected["pair_count"],
        "expected_pair_ids_sha256": preflight["expected_pair_ids_sha256"],
        "source_inlier_rows": selected["source_inlier_rows"],
        "retained_inlier_rows": selected["retained_inlier_rows"],
        "pruned_inlier_rows": selected["pruned_inlier_rows"],
        "capped_rows": selected["capped_rows"],
        "minimum_retained_rows": selected["minimum_retained_rows"],
        "maximum_retained_rows": selected["maximum_retained_rows"],
        "raw_matches_pruned_to_same_conflict_free_rows": True,
        "observation_node_count": selected["observation_node_count"],
        "pairs": [
            {
                "pair_id": int(item["pair_id"]),
                "first": str(item["first"]),
                "second": str(item["second"]),
            }
            for item in selected["per_pair"]
        ],
        "accepted_pair_ids_sha256": preflight["expected_pair_ids_sha256"],
        "per_pair": selected["per_pair"],
        "output_path": str(output),
        "output_sha256": sha256_file(output),
        "output_logical_digest": sqlite_logical_digest(output),
        "exact_mapper_graph_preflight": preflight,
    }
    report_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--classification", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--consensus-report", type=Path)
    parser.add_argument("--minimum-matches-per-pair", type=int, default=15)
    parser.add_argument(
        "--consensus-outlier-max-rows",
        type=int,
        help="Optional bounded support cap for independently classified cross-ring consensus outliers.",
    )
    parser.add_argument(
        "--consensus-outlier-transition-cap",
        action="append",
        default=[],
        help="Transition cap as RING_A:RING_B=ROWS; may be repeated.",
    )
    parser.add_argument(
        "--consensus-outlier-pair-cap",
        action="append",
        default=[],
        help="Pair-specific cap as PAIR_ID=ROWS; may be repeated.",
    )
    args = parser.parse_args()
    transition_caps: dict[str, int] = {}
    for value in args.consensus_outlier_transition_cap:
        if "=" not in str(value) or ":" not in str(value).split("=", 1)[0]:
            raise ValueError(
                "--consensus-outlier-transition-cap must use RING_A:RING_B=ROWS"
            )
        transition, rows = str(value).split("=", 1)
        left, right = transition.split(":", 1)
        transition_caps[":".join(sorted((left, right)))] = int(rows)
    pair_caps: dict[int, int] = {}
    for value in args.consensus_outlier_pair_cap:
        if "=" not in str(value):
            raise ValueError("--consensus-outlier-pair-cap must use PAIR_ID=ROWS")
        pair_id, rows = str(value).split("=", 1)
        pair_caps[int(pair_id)] = int(rows)
    report = build_match_conflict_free_database(
        args.snapshot_manifest,
        args.classification,
        args.output,
        args.report,
        consensus_path=args.consensus_report,
        minimum_matches_per_pair=args.minimum_matches_per_pair,
        consensus_outlier_max_rows=args.consensus_outlier_max_rows,
        consensus_outlier_max_rows_by_transition=transition_caps,
        consensus_outlier_max_rows_by_pair=pair_caps,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "pair_count": report["pair_count"],
                "retained_inlier_rows": report["retained_inlier_rows"],
                "minimum_retained_rows": report["minimum_retained_rows"],
                "capped_rows": report["capped_rows"],
                "output": report["output_path"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
