"""Build a deterministic mapper graph from calibrated consensus evidence.

This is a graph-selection/reporting step, not a reconstruction solver.  The
complete calibrated classification and fixed audit remain immutable evidence;
only the disposable mapper graph is filtered.  Cross-ring edges marked as
robust-consensus outliers are excluded from the mapper graph, while the
smallest number of strongest calibrated outlier bridges is restored when
needed to keep the image graph connected.  The selected bridge is still
retained as an explicitly labelled audit/negative-evidence relationship.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


RETAINED_STATUSES = frozenset(
    {
        "retained_rotation_synchronization",
        "retained_robust_consensus_inlier",
    }
)
OUTLIER_STATUS = "retained_robust_consensus_outlier"


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair_identity_sha256(pairs: Iterable[Mapping[str, Any]]) -> str:
    pair_ids = [int(pair["pair_id"]) for pair in pairs]
    return hashlib.sha256(
        json.dumps(pair_ids, separators=(",", ":"), sort_keys=False).encode("utf-8")
    ).hexdigest()


def _pair_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return str(record["first"]), str(record["second"])


def _union_find(nodes: Iterable[str]) -> tuple[dict[str, str], dict[str, int]]:
    parent = {str(node): str(node) for node in nodes}
    rank = {str(node): 0 for node in nodes}
    return parent, rank


def _find(parent: dict[str, str], node: str) -> str:
    root = node
    while parent[root] != root:
        root = parent[root]
    while parent[node] != node:
        next_node = parent[node]
        parent[node] = root
        node = next_node
    return root


def _union(parent: dict[str, str], rank: dict[str, int], first: str, second: str) -> bool:
    first_root = _find(parent, first)
    second_root = _find(parent, second)
    if first_root == second_root:
        return False
    if rank[first_root] < rank[second_root]:
        first_root, second_root = second_root, first_root
    parent[second_root] = first_root
    if rank[first_root] == rank[second_root]:
        rank[first_root] += 1
    return True


def _classification_mapping_records(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("mapping_records")
    if not isinstance(records, list):
        raise ValueError("classification payload must contain mapping_records")
    result: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("classification mapping_records must contain objects")
        copied = dict(record)
        if not copied.get("first") or not copied.get("second"):
            raise ValueError("classification mapping record lacks image names")
        independent = copied.get("independent_audit_record")
        if not isinstance(independent, Mapping):
            raise ValueError("mapping record lacks independent calibrated audit payload")
        if independent.get("calibrated_reestimate_rotation_matrix_first_to_second") is None:
            raise ValueError("mapping record lacks calibrated rotation override")
        if independent.get("calibrated_reestimate_translation_first_to_second") is None:
            raise ValueError("mapping record lacks calibrated translation override")
        result.append(copied)
    return result


def _bridge_rank(record: Mapping[str, Any]) -> tuple[float, float, float, int, str, str]:
    """Prefer well-conditioned high-support calibrated bridges deterministically."""

    inliers = float(record.get("calibrated_inliers") or 0.0)
    triangulation = float(record.get("calibrated_tri_angle_deg") or 0.0)
    homography = float(record.get("homography_inlier_fraction") or 1.0)
    return (
        -inliers,
        -triangulation,
        homography,
        int(record.get("pair_id") or 0),
        str(record.get("first", "")),
        str(record.get("second", "")),
    )


def build_mapper_graph(
    classification: Mapping[str, Any],
    consensus: Mapping[str, Any],
    *,
    classification_path: str | Path,
    consensus_path: str | Path,
    raw_audit_path: str | Path,
    snapshot_manifest_path: str | Path,
    raw_audit: Mapping[str, Any],
    snapshot_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an auditable disposable mapper graph.

    The fixed audit remains complete in ``fixed_audit_*`` fields.  The
    ``mapping_records``/``pairs`` lists only control which rows are rewritten
    for the disposable mapper database.
    """

    if int(classification.get("schema_version", 0)) < 2:
        raise ValueError("calibrated classification schema is too old")
    if int(raw_audit.get("schema_version", 0)) < 2:
        raise ValueError("raw calibrated audit schema is too old")
    if classification.get("angle_units") != {
        "triangulation_angle": "degrees",
        "rotation_angle": "degrees",
    }:
        raise ValueError("classification does not declare degree persistence units")
    if raw_audit.get("angle_units") != {
        "triangulation_angle": "degrees",
        "rotation_angle": "degrees",
    }:
        raise ValueError("raw audit does not declare degree persistence units")

    source_records = _classification_mapping_records(classification)
    source_by_key = {_pair_key(record): record for record in source_records}
    if len(source_by_key) != len(source_records):
        raise ValueError("classification mapping records contain duplicate image pairs")

    per_edge = consensus.get("per_edge")
    if not isinstance(per_edge, list):
        raise ValueError("consensus report must contain per_edge")
    classification_sha = sha256_file(classification_path)
    if consensus.get("classification_sha256") != classification_sha:
        raise ValueError("consensus report is not bound to the supplied classification")

    consensus_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in per_edge:
        if not isinstance(edge, Mapping) or not edge.get("first") or not edge.get("second"):
            raise ValueError("consensus per_edge contains an invalid record")
        key = _pair_key(edge)
        if key in consensus_by_key:
            raise ValueError("consensus report contains duplicate image pairs")
        consensus_by_key[key] = dict(edge)

    missing = sorted(set(source_by_key) - set(consensus_by_key))
    if missing:
        raise ValueError(f"consensus report misses {len(missing)} mapping pairs")

    classification_records = classification.get("classification", [])
    if not isinstance(classification_records, list):
        raise ValueError("classification payload must contain classification records")
    classification_by_key = {
        _pair_key(record): dict(record)
        for record in classification_records
        if isinstance(record, Mapping)
    }
    base_records: list[dict[str, Any]] = []
    outlier_records: list[dict[str, Any]] = []
    outlier_candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for key, record in source_by_key.items():
        edge = consensus_by_key[key]
        status = str(edge.get("mapping_status", ""))
        if status in RETAINED_STATUSES:
            base_records.append(record)
        elif status == OUTLIER_STATUS:
            outlier_records.append(record)
            classification_record = classification_by_key.get(key)
            if classification_record is None:
                raise ValueError(f"consensus outlier is missing from classification: {key!r}")
            outlier_candidates.append((record, classification_record))
        else:
            raise ValueError(
                "a retained classification pair has an unsupported consensus status: "
                f"{status!r} for {key!r}"
            )

    all_nodes = sorted(
        {str(record["first"]) for record in source_records}
        | {str(record["second"]) for record in source_records}
    )
    parent, rank = _union_find(all_nodes)
    for record in base_records:
        _union(parent, rank, str(record["first"]), str(record["second"]))

    bridge_records: list[dict[str, Any]] = []
    for mapping_record, classification_record in sorted(
        outlier_candidates,
        key=lambda item: _bridge_rank(item[1]),
    ):
        record = mapping_record
        if _union(parent, rank, str(record["first"]), str(record["second"])):
            bridge_records.append(record)

    roots = {_find(parent, node) for node in all_nodes}
    if len(roots) != 1:
        raise ValueError(
            "consensus-filtered mapper graph cannot be connected with available "
            f"outlier bridges ({len(roots)} components remain)"
        )

    selected_keys = {
        _pair_key(record) for record in base_records + bridge_records
    }
    selected_records = [record for record in source_records if _pair_key(record) in selected_keys]
    # Keep pair ordering tied to the immutable classification order.  The
    # bridge policy changes membership only, never the identity convention.
    pairs = [
        {
            "pair_id": int(record["independent_audit_record"].get("pair_id", record.get("pair_id", 0))),
            "first": str(record["first"]),
            "second": str(record["second"]),
        }
        for record in selected_records
    ]
    if any(pair["pair_id"] == 0 for pair in pairs):
        # Mapping records in v3 carry the pair id inside the independent audit
        # record in some historical reports and at the top level in others.
        by_key = {_pair_key(record): record for record in classification.get("classification", [])}
        for pair in pairs:
            if pair["pair_id"] == 0:
                pair["pair_id"] = int(by_key[(pair["first"], pair["second"])] ["pair_id"])

    audit_records = raw_audit.get("records", [])
    if not isinstance(audit_records, list):
        raise ValueError("raw audit records are missing")
    fixed_ids = [int(record["pair_id"]) for record in audit_records]
    fixed_ids_sha = hashlib.sha256(
        json.dumps(fixed_ids, separators=(",", ":"), sort_keys=False).encode("utf-8")
    ).hexdigest()
    if classification.get("fixed_pair_count") != len(audit_records):
        raise ValueError("classification fixed-pair count does not match raw audit")
    if classification.get("fixed_pair_ids_sha256") != fixed_ids_sha:
        raise ValueError("classification fixed-pair identity does not match raw audit")

    payload = {
        "schema_version": 3,
        "method": (
            "calibrated graph-consensus mapper selection: retain all same-ring "
            "synchronization edges and robust cross-ring inliers; add the minimum "
            "strongest calibrated bridge required for one connected image graph; "
            "all excluded edges remain fixed-audit negative evidence"
        ),
        "classification_path": str(Path(classification_path).resolve()),
        "classification_sha256": classification_sha,
        "consensus_report_path": str(Path(consensus_path).resolve()),
        "consensus_report_sha256": sha256_file(consensus_path),
        "raw_audit_path": str(Path(raw_audit_path).resolve()),
        "raw_audit_sha256": sha256_file(raw_audit_path),
        "raw_audit_report_sha256": sha256_file(raw_audit_path),
        "snapshot_manifest": str(Path(snapshot_manifest_path).resolve()),
        "snapshot_manifest_sha256": sha256_file(snapshot_manifest_path),
        "canonical_snapshot_sha256": snapshot_manifest["canonical_sha256"],
        "canonical_creation_logical_sha256": snapshot_manifest["creation_logical_sha256"],
        "fixed_audit_pair_count": len(audit_records),
        "fixed_pair_count": len(audit_records),
        "fixed_audit_pair_ids_sha256": fixed_ids_sha,
        "fixed_pair_ids_sha256": fixed_ids_sha,
        "source_mapping_pair_count": len(source_records),
        "base_consensus_pair_count": len(base_records),
        "excluded_consensus_outlier_count": len(outlier_records) - len(bridge_records),
        "bridge_pair_count": len(bridge_records),
        "bridge_policy": "strongest_calibrated_inliers_then_triangulation_then_low_homography_then_pair_id",
        "bridge_pairs": [
            {
                "pair_id": int(next(item["pair_id"] for item in pairs if _pair_key(item) == _pair_key(record))),
                "first": str(record["first"]),
                "second": str(record["second"]),
                "calibrated_inliers": classification_by_key[_pair_key(record)].get("calibrated_inliers"),
                "calibrated_tri_angle_deg": classification_by_key[_pair_key(record)].get("calibrated_tri_angle_deg"),
                "calibrated_cheirality_fraction": classification_by_key[_pair_key(record)].get("calibrated_cheirality_fraction"),
                "homography_inlier_fraction": classification_by_key[_pair_key(record)].get("homography_inlier_fraction"),
            }
            for record in bridge_records
        ],
        "excluded_consensus_outlier_pair_ids": [],
        "pair_count": len(pairs),
        "pair_identity_sha256": pair_identity_sha256(pairs),
        "pairs": pairs,
        "mapping_records": selected_records,
        "audit_evidence_policy": (
            "The full fixed audit remains independent and hash-bound; mapper graph "
            "selection does not delete or relabel excluded audit evidence."
        ),
    }
    # Record excluded ids without depending on the selected-pair list.
    selected_key_set = {_pair_key(record) for record in selected_records}
    payload["excluded_consensus_outlier_pair_ids"] = [
        int(next(item["pair_id"] for item in classification.get("classification", []) if _pair_key(item) == _pair_key(record)))
        for record in outlier_records
        if _pair_key(record) not in selected_key_set
    ]
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification", required=True, type=Path)
    parser.add_argument("--consensus", required=True, type=Path)
    parser.add_argument("--raw-audit", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    classification = json.loads(args.classification.read_text(encoding="utf-8"))
    consensus = json.loads(args.consensus.read_text(encoding="utf-8"))
    raw_audit = json.loads(args.raw_audit.read_text(encoding="utf-8"))
    snapshot_manifest = json.loads(args.snapshot_manifest.read_text(encoding="utf-8"))
    payload = build_mapper_graph(
        classification,
        consensus,
        classification_path=args.classification,
        consensus_path=args.consensus,
        raw_audit_path=args.raw_audit,
        snapshot_manifest_path=args.snapshot_manifest,
        raw_audit=raw_audit,
        snapshot_manifest=snapshot_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "pair_count": payload["pair_count"],
                "base_consensus_pair_count": payload["base_consensus_pair_count"],
                "bridge_pair_count": payload["bridge_pair_count"],
                "excluded_consensus_outlier_count": payload["excluded_consensus_outlier_count"],
                "pair_identity_sha256": payload["pair_identity_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
