"""Bind audited calibrated poses to a filtered disposable correspondence DB.

This is intentionally different from ``reestimate_v4_two_view_geometry.py``:
the correspondence filter changes only the mapper support/weight.  It must
not ask a deliberately down-weighted subset to re-estimate a new essential
matrix.  The deterministic calibrated R/t from the independent audit remains
authoritative, and E/F/q/t are rebuilt and round-tripped for every retained
pair before GLOMAP is allowed to read the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v4_repair import (  # noqa: E402
    colmap_pair_id,
    copy_frozen_sqlite_snapshot,
    load_canonical_sqlite_snapshot_manifest,
    sha256_file,
    validate_exact_mapper_graph,
)
from scripts.reestimate_v4_two_view_geometry import _pose_override_payload  # noqa: E402


def _pairs(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = payload.get("pairs")
    if not isinstance(values, list) or not values:
        raise ValueError("graph report must contain a non-empty pairs list")
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("graph pair must be an object")
        first, second = str(value.get("first", "")), str(value.get("second", ""))
        pair_id = int(value.get("pair_id") or 0)
        if not first or not second or pair_id <= 0:
            raise ValueError("graph pair lacks first/second/pair_id")
        if pair_id in seen:
            raise ValueError(f"duplicate graph pair_id: {pair_id}")
        seen.add(pair_id)
        result.append({"pair_id": pair_id, "first": first, "second": second})
    return result


def _mapping_records(payload: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    values = payload.get("mapping_records")
    if not isinstance(values, list) or not values:
        raise ValueError("classification must contain mapping_records")
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("mapping record must be an object")
        first, second = str(value.get("first", "")), str(value.get("second", ""))
        if not first or not second:
            raise ValueError("mapping record lacks ordered image names")
        pair_key = (first, second)
        if pair_key in result:
            raise ValueError(f"duplicate classification image pair: {first} -> {second}")
        independent = value.get("independent_audit_record")
        if not isinstance(independent, Mapping):
            raise ValueError(f"mapping record lacks independent audit record: {first} -> {second}")
        result[pair_key] = value
    return result


def apply_calibrated_pose_overrides(
    input_path: str | Path,
    output_path: str | Path,
    snapshot_manifest: str | Path,
    graph_report: str | Path,
    classification_path: str | Path,
    raw_audit_report: str | Path,
    report_path: str | Path,
    *,
    filtered_match_report: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_canonical_sqlite_snapshot_manifest(snapshot_manifest)
    canonical = Path(manifest["canonical_path"]).resolve()
    source = Path(input_path).resolve()
    destination = Path(output_path).resolve()
    report_target = Path(report_path).resolve()
    if source == canonical or destination == canonical:
        raise ValueError("pose override input/output must be disposable, never canonical")
    if source == destination or not source.is_file():
        raise ValueError("pose override input must be an existing disposable DB and output must differ")
    if destination.exists() or report_target.exists():
        raise FileExistsError("pose override output/report already exists; preserve it")

    graph_payload = json.loads(Path(graph_report).read_text(encoding="utf-8"))
    pairs = _pairs(graph_payload)
    classification_payload = json.loads(Path(classification_path).read_text(encoding="utf-8"))
    mapping = _mapping_records(classification_payload)
    raw_payload = json.loads(Path(raw_audit_report).read_text(encoding="utf-8"))
    snapshot_manifest_sha256 = sha256_file(snapshot_manifest)
    raw_audit_sha256 = sha256_file(raw_audit_report)
    if classification_payload.get("raw_audit_report_sha256") != raw_audit_sha256:
        raise ValueError("classification is not bound to the supplied raw calibrated audit")
    if classification_payload.get("snapshot_manifest_sha256") != snapshot_manifest_sha256:
        raise ValueError("classification is not bound to the supplied snapshot manifest")
    if classification_payload.get("canonical_snapshot_sha256") != manifest["canonical_sha256"]:
        raise ValueError("classification canonical SHA does not match the manifest")
    if raw_payload.get("snapshot_manifest_sha256") != snapshot_manifest_sha256:
        raise ValueError("raw calibrated audit is not bound to the supplied snapshot manifest")

    expected_pair_ids = {int(item["pair_id"]) for item in pairs}
    expected_pair_names = {(str(item["first"]), str(item["second"])) for item in pairs}
    if expected_pair_names != set(mapping):
        raise ValueError("graph pairs and calibrated mapping records do not have the same ordered image-pair set")
    canonical_before = sha256_file(canonical)
    input_preflight = validate_exact_mapper_graph(
        source,
        expected_pair_ids,
        canonical_path=canonical,
        require_raw_matches=True,
    )
    if not input_preflight["passed"]:
        raise RuntimeError(f"filtered input graph is not exact: {input_preflight}")
    copy_frozen_sqlite_snapshot(source, destination)

    import pycolmap

    database = pycolmap.Database.open(str(destination))
    records: list[dict[str, Any]] = []
    try:
        for pair in pairs:
            pair_id = int(pair["pair_id"])
            first = database.read_image_with_name(str(pair["first"]))
            second = database.read_image_with_name(str(pair["second"]))
            geometry = database.read_two_view_geometry(int(first.image_id), int(second.image_id))
            if geometry is None:
                raise RuntimeError(f"missing filtered two-view geometry: {pair_id}")
            payload = geometry.todict(recursive=False)
            filtered_matches = np.asarray(geometry.inlier_matches, dtype=np.uint32).reshape(-1, 2)
            payload["inlier_matches"] = filtered_matches
            independent = mapping[(str(pair["first"]), str(pair["second"]))]["independent_audit_record"]
            matrix = independent.get("calibrated_reestimate_rotation_matrix_first_to_second")
            translation = independent.get("calibrated_reestimate_translation_first_to_second")
            if matrix is None or translation is None:
                raise ValueError(f"calibrated R/t missing for pair {pair_id}")
            rewritten, runtime_angle, round_trip = _pose_override_payload(
                geometry,
                payload,
                database.read_camera(int(first.camera_id)),
                database.read_camera(int(second.camera_id)),
                np.asarray(matrix, dtype=np.float64),
                np.asarray(translation, dtype=np.float64),
            )
            database.update_two_view_geometry(int(first.image_id), int(second.image_id), rewritten)
            records.append(
                {
                    "pair_id": pair_id,
                    "first": str(pair["first"]),
                    "second": str(pair["second"]),
                    "filtered_inlier_rows": int(len(filtered_matches)),
                    "pose_override_applied": True,
                    "runtime_rotation_angle_deg": runtime_angle,
                    "pose_override_round_trip": round_trip,
                }
            )
    finally:
        database.close()

    output_preflight = validate_exact_mapper_graph(
        destination,
        expected_pair_ids,
        canonical_path=canonical,
        require_raw_matches=True,
    )
    canonical_after = sha256_file(canonical)
    if canonical_after != canonical_before or canonical_after != manifest["canonical_sha256"]:
        raise RuntimeError("canonical snapshot changed during direct calibrated pose binding")
    round_trip_count = sum(bool(item["pose_override_round_trip"].get("passed")) for item in records)
    if len(records) != len(expected_pair_ids) or round_trip_count != len(expected_pair_ids):
        raise RuntimeError("direct calibrated pose binding did not pass every pair round trip")
    if not output_preflight["passed"]:
        raise RuntimeError(f"direct calibrated pose output graph is not exact: {output_preflight}")

    report = {
        "schema_version": 1,
        "status": "complete",
        "method": (
            "same-solution calibrated R/t/E/F/q override applied to filtered "
            "verified rows without subset re-estimation"
        ),
        "canonical_sqlite_opened": False,
        "snapshot_manifest": str(Path(snapshot_manifest).resolve()),
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
        "canonical_snapshot_sha256": manifest["canonical_sha256"],
        "canonical_creation_logical_sha256": manifest["creation_logical_sha256"],
        "canonical_sha256_before": canonical_before,
        "canonical_sha256_after": canonical_after,
        "canonical_snapshot_hash_stable": True,
        "input_path": str(source),
        "input_sha256": sha256_file(source),
        "input_filtered_match_report": str(Path(filtered_match_report).resolve()) if filtered_match_report else None,
        "input_filtered_match_report_sha256": sha256_file(filtered_match_report) if filtered_match_report else None,
        "graph_report": str(Path(graph_report).resolve()),
        "graph_report_sha256": sha256_file(graph_report),
        "classification_path": str(Path(classification_path).resolve()),
        "classification_sha256": sha256_file(classification_path),
        "raw_audit_report": str(Path(raw_audit_report).resolve()),
        "raw_audit_report_sha256": raw_audit_sha256,
        "output_path": str(destination),
        "output_sha256": sha256_file(destination),
        "pair_count": len(records),
        "pose_override_applied_count": len(records),
        "pose_override_round_trip_pass_count": round_trip_count,
        "failed_pair_count": 0,
        "input_exact_mapper_graph_preflight": input_preflight,
        "output_exact_mapper_graph_preflight": output_preflight,
        "pairs": records,
    }
    report_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--graph-report", required=True, type=Path)
    parser.add_argument("--classification", required=True, type=Path)
    parser.add_argument("--raw-audit-report", required=True, type=Path)
    parser.add_argument("--filtered-match-report", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = apply_calibrated_pose_overrides(
        args.input,
        args.output,
        args.snapshot_manifest,
        args.graph_report,
        args.classification,
        args.raw_audit_report,
        args.report,
        filtered_match_report=args.filtered_match_report,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "pair_count": report["pair_count"],
                "pose_override_round_trip_pass_count": report["pose_override_round_trip_pass_count"],
                "output": report["output_path"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
