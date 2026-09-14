"""Classify fixed sparse-audit pairs using calibrated raw-match evidence.

The mapper's stored E/qvec can be a defective essential-matrix decomposition.
This report keeps that stored evidence, but marks a calibrated rotation as
usable only when raw-match support, triangulation, cheirality, homography and
local rotation-cycle checks all pass.  Nothing is deleted from the audit set.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v4_repair import load_canonical_sqlite_snapshot_manifest


def _geodesic(first: np.ndarray, second: np.ndarray) -> float:
    product = np.asarray(first, dtype=np.float64) @ np.asarray(second, dtype=np.float64).T
    cosine = np.clip((float(np.trace(product)) - 1.0) / 2.0, -1.0, 1.0)
    return float(math.degrees(math.acos(float(cosine))))


def _is_so3(matrix: np.ndarray) -> bool:
    return bool(
        matrix.shape == (3, 3)
        and np.isfinite(matrix).all()
        and abs(float(np.linalg.det(matrix)) - 1.0) <= 1e-3
        and np.max(np.abs(matrix.T @ matrix - np.eye(3))) <= 1e-3
    )


def _is_translation_direction(value: Any) -> bool:
    vector = np.asarray(value, dtype=np.float64).reshape(-1) if value is not None else np.empty(0)
    norm = float(np.linalg.norm(vector)) if len(vector) == 3 else 0.0
    return bool(len(vector) == 3 and np.isfinite(vector).all() and norm > 1e-12)


def _pair_id_sha(pair_ids: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(pair_ids, separators=(",", ":"), sort_keys=False).encode("utf-8")
    ).hexdigest()


def _cycle_errors(
    records: list[dict[str, Any]],
    accepted: Mapping[int, np.ndarray],
    ring_by_name: Mapping[str, str],
) -> dict[int, list[float]]:
    directed: dict[str, list[tuple[str, np.ndarray, int]]] = defaultdict(list)
    for pair_id, rotation in accepted.items():
        record = next(item for item in records if int(item["pair_id"]) == int(pair_id))
        first, second = str(record["first"]), str(record["second"])
        if ring_by_name.get(first) != ring_by_name.get(second):
            continue
        if abs(int(record.get("frame_distance") or 0)) > 4:
            continue
        directed[first].append((second, rotation, int(pair_id)))
        directed[second].append((first, rotation.T, int(pair_id)))
    errors: dict[int, list[float]] = defaultdict(list)
    for pair_id, direct in accepted.items():
        record = next(item for item in records if int(item["pair_id"]) == int(pair_id))
        first, second = str(record["first"]), str(record["second"])
        ring = ring_by_name.get(first)
        for middle, first_to_middle, first_edge_id in directed.get(first, []):
            if middle in {first, second}:
                continue
            for target, middle_to_target, second_edge_id in directed.get(middle, []):
                if target != second or first_edge_id == pair_id or second_edge_id == pair_id:
                    continue
                if ring_by_name.get(middle) != ring or ring_by_name.get(target) != ring:
                    continue
                path = middle_to_target @ first_to_middle
                errors[pair_id].append(_geodesic(direct, path))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--raw-audit-report", required=True, type=Path)
    parser.add_argument("--audit-manifest", required=True, type=Path)
    parser.add_argument("--ring-metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-inliers", type=int, default=100)
    parser.add_argument("--minimum-triangulation-angle-deg", type=float, default=0.05)
    parser.add_argument("--minimum-cheirality-fraction", type=float, default=0.5)
    parser.add_argument("--maximum-homography-fraction", type=float, default=0.8)
    parser.add_argument("--maximum-cycle-error-deg", type=float, default=20.0)
    args = parser.parse_args()

    snapshot = load_canonical_sqlite_snapshot_manifest(args.snapshot_manifest)
    manifest_sha = hashlib.sha256(args.audit_manifest.read_bytes()).hexdigest()
    snapshot_manifest_sha = hashlib.sha256(args.snapshot_manifest.read_bytes()).hexdigest()
    raw = json.loads(args.raw_audit_report.read_text(encoding="utf-8"))
    audit = json.loads(args.audit_manifest.read_text(encoding="utf-8"))
    ring_payload = json.loads(args.ring_metadata.read_text(encoding="utf-8"))
    ring_by_name = {
        Path(str(item["relative_path"])).name: str(item["logical_ring_id"])
        for item in ring_payload["records"]
    }
    expected_pair_ids = [int(item["pair_id"]) for item in audit["pairs"]]
    observed_pair_ids = [int(item["pair_id"]) for item in raw.get("records", [])]
    expected_angle_units = {
        "triangulation_angle": "degrees",
        "rotation_angle": "degrees",
    }
    if raw.get("angle_units") != expected_angle_units:
        raise ValueError(
            "raw audit does not declare corrected degree units; regenerate it from the immutable snapshot"
        )
    if raw.get("pycolmap_angle_source_units") != {
        "TwoViewGeometry.tri_angle": "radians",
        "Rotation3d.angle": "radians",
    }:
        raise ValueError("raw audit is missing the explicit pyCOLMAP radian source-unit contract")
    if int(raw.get("schema_version", 0)) < 2:
        raise ValueError("raw audit schema predates the explicit calibrated angle-unit contract")
    if raw.get("database_sha256") != snapshot["canonical_sha256"]:
        raise ValueError("raw audit is not bound to the manifest raw SHA")
    if raw.get("snapshot_manifest_sha256") != snapshot_manifest_sha:
        raise ValueError("raw audit is not bound to the v3 snapshot manifest")
    if raw.get("pair_source_sha256") != manifest_sha or observed_pair_ids != expected_pair_ids:
        raise ValueError("raw audit does not contain the fixed independent pair identity")

    pair_metadata = {int(item["pair_id"]): item for item in audit["pairs"]}
    records = []
    for raw_record in raw["records"]:
        record = dict(raw_record)
        metadata = pair_metadata.get(int(record["pair_id"]), {})
        record["ring"] = ring_by_name.get(str(record["first"]))
        record["second_ring"] = ring_by_name.get(str(record["second"]))
        record["same_ring"] = record["ring"] == record["second_ring"]
        record["frame_distance"] = metadata.get("frame_distance")
        records.append(record)
    accepted_matrices: dict[int, np.ndarray] = {}
    classification: list[dict[str, Any]] = []
    for record in records:
        pair_id = int(record["pair_id"])
        matrix = np.asarray(
            record.get("calibrated_reestimate_rotation_matrix_first_to_second"), dtype=np.float64
        )
        config = int(record.get("calibrated_reestimate_config") or -1)
        tri = record.get("calibrated_reestimate_tri_angle_deg")
        tri_rad = record.get("calibrated_reestimate_tri_angle_rad")
        translation = record.get("calibrated_reestimate_translation_first_to_second")
        cheirality = float(record.get("calibrated_reestimate_cheirality_fraction") or 0.0)
        homography = float(record.get("homography_inlier_fraction") or 0.0)
        rank_ratio = float(
            record.get("calibrated_essential_rank3_to_rank1")
            if record.get("calibrated_essential_rank3_to_rank1") is not None
            else (record.get("essential_rank3_to_rank1") or math.inf)
        )
        reasons: list[str] = []
        if config != 2:
            reasons.append("calibrated_config_not_2")
        if not _is_so3(matrix):
            reasons.append("calibrated_rotation_not_so3")
        if not _is_translation_direction(translation):
            reasons.append("calibrated_translation_missing_or_degenerate")
        if tri is not None and tri_rad is not None:
            expected_tri = math.degrees(float(tri_rad))
            if not math.isfinite(float(tri)) or abs(float(tri) - expected_tri) > 1e-8:
                raise ValueError(
                    f"raw audit pair {pair_id} has inconsistent radian/degree triangulation fields"
                )
        if int(record.get("calibrated_reestimate_inliers") or 0) < args.minimum_inliers:
            reasons.append("calibrated_inliers_below_floor")
        if tri is None or not math.isfinite(float(tri)) or float(tri) <= args.minimum_triangulation_angle_deg:
            reasons.append("triangulation_angle_weak")
        if cheirality < args.minimum_cheirality_fraction:
            reasons.append("calibrated_cheirality_weak")
        if homography > args.maximum_homography_fraction:
            reasons.append("homography_dominant")
        if rank_ratio > 1e-3:
            reasons.append("essential_rank_defective")
        if not reasons:
            accepted_matrices[pair_id] = matrix
        classification.append(
            {
                "pair_id": pair_id,
                "first": record["first"],
                "second": record["second"],
                "ring": record["ring"],
                "second_ring": record["second_ring"],
                "same_ring": bool(record["same_ring"]),
                "frame_distance": record["frame_distance"],
                "calibrated_rotation_matrix_first_to_second": matrix.tolist(),
                "calibrated_rotation_deg": float(
                    math.degrees(math.acos(np.clip((float(np.trace(matrix)) - 1.0) / 2.0, -1.0, 1.0)))
                ) if _is_so3(matrix) else None,
                "calibrated_translation_first_to_second": translation,
                "calibrated_inliers": int(record.get("calibrated_reestimate_inliers") or 0),
                "calibrated_tri_angle_deg": tri,
                "calibrated_cheirality_fraction": cheirality,
                "homography_inlier_fraction": homography,
                "essential_rank3_to_rank1": rank_ratio,
                "well_conditioned_calibrated": not reasons,
                "classification_reasons": reasons,
            }
        )

    cycle_errors = _cycle_errors(records, accepted_matrices, ring_by_name)
    for item in classification:
        errors = cycle_errors.get(int(item["pair_id"]), [])
        item["local_cycle_count"] = len(errors)
        item["local_cycle_error_deg"] = float(np.median(errors)) if errors else None
        item["local_cycle_status"] = "consistent" if errors and float(np.median(errors)) <= args.maximum_cycle_error_deg else (
            "inconsistent" if errors else "not_available"
        )
        if item["well_conditioned_calibrated"] and errors and float(np.median(errors)) > args.maximum_cycle_error_deg:
            item["well_conditioned_calibrated"] = False
            item["classification_reasons"].append("local_cycle_inconsistent")

    mapping_records = [
        {
            "disposition": "retained_for_mapping",
            "first": item["first"],
            "second": item["second"],
            "independent_audit_record": {
                "calibrated_reestimate_rotation_matrix_first_to_second": item[
                    "calibrated_rotation_matrix_first_to_second"
                ],
                "calibrated_reestimate_translation_first_to_second": item[
                    "calibrated_translation_first_to_second"
                ],
            },
        }
        for item in classification
        if item["well_conditioned_calibrated"]
    ]
    payload = {
        "schema_version": 2,
        "method": "fixed raw ALIKED/LightGlue audit classification: calibrated rotation, triangulation, cheirality, homography, essential-rank and local cycle consistency",
        "angle_units": expected_angle_units,
        "pycolmap_angle_source_units": raw["pycolmap_angle_source_units"],
        "snapshot_manifest": str(args.snapshot_manifest.resolve()),
        "snapshot_manifest_sha256": snapshot_manifest_sha,
        "canonical_snapshot_sha256": snapshot["canonical_sha256"],
        "canonical_creation_logical_sha256": snapshot["creation_logical_sha256"],
        "audit_manifest": str(args.audit_manifest.resolve()),
        "audit_manifest_sha256": manifest_sha,
        "raw_audit_report": str(args.raw_audit_report.resolve()),
        "raw_audit_report_sha256": hashlib.sha256(args.raw_audit_report.read_bytes()).hexdigest(),
        "fixed_pair_count": len(expected_pair_ids),
        "fixed_pair_ids_sha256": _pair_id_sha(expected_pair_ids),
        "classification": classification,
        "mapping_records": mapping_records,
        "thresholds": {
            "minimum_inliers": args.minimum_inliers,
            "minimum_triangulation_angle_deg": args.minimum_triangulation_angle_deg,
            "minimum_cheirality_fraction": args.minimum_cheirality_fraction,
            "maximum_homography_fraction": args.maximum_homography_fraction,
            "maximum_cycle_error_deg": args.maximum_cycle_error_deg,
        },
        "well_conditioned_count": sum(bool(item["well_conditioned_calibrated"]) for item in classification),
        "contradictory_or_unconditioned_count": sum(not bool(item["well_conditioned_calibrated"]) for item in classification),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "fixed_pair_count": len(expected_pair_ids), "well_conditioned_count": payload["well_conditioned_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
