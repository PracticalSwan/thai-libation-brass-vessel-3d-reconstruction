"""Evaluate one V4 sparse candidate against the immutable independent audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np
import pycolmap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v4_repair import (
    collect_sparse_pair_evidence,
    create_disposable_sqlite_from_manifest,
    load_ring_metadata,
    load_canonical_sqlite_snapshot_manifest,
    sparse_camera_center_translation_evidence,
    sparse_integrity_gate,
    sparse_mask_projection_evidence,
    stable_directory_sha256,
)


def _rotation_geodesic_deg(first: object, second: object) -> float | None:
    first_matrix = np.asarray(first, dtype=float)
    second_matrix = np.asarray(second, dtype=float)
    if first_matrix.shape != (3, 3) or second_matrix.shape != (3, 3):
        return None
    product = first_matrix @ second_matrix.T
    cosine = max(-1.0, min(1.0, (float(np.trace(product)) - 1.0) / 2.0))
    return float(math.degrees(math.acos(cosine)))


def _bind_calibrated_audit_evidence(
    evidence: list[dict],
    raw_audit: dict,
    classification: dict,
) -> list[dict]:
    """Replace stored-E rotation evidence with the fixed raw calibrated audit.

    Candidate tracks/poses still come from the reconstruction, while every
    independent rotation used by the sparse gate comes from the immutable raw
    ALIKED/LightGlue audit and its well-conditioned classification.
    """

    raw_by_pair = {int(item["pair_id"]): item for item in raw_audit.get("records", [])}
    class_by_pair = {int(item["pair_id"]): item for item in classification.get("classification", [])}
    bound: list[dict] = []
    for item in evidence:
        result = dict(item)
        pair_id = int(result["pair_id"])
        raw = raw_by_pair.get(pair_id)
        classified = class_by_pair.get(pair_id)
        if raw is None or classified is None:
            result["calibrated_conditioned"] = False
            result["rotation_disagreement_deg"] = None
            bound.append(result)
            continue
        matrix = raw.get("calibrated_reestimate_rotation_matrix_first_to_second")
        final_matrix = result.get("final_rotation_matrix_first_to_second")
        if matrix is None or final_matrix is None:
            result["calibrated_conditioned"] = False
            result["rotation_disagreement_deg"] = None
            bound.append(result)
            continue
        result["stored_two_view_rotation_matrix_first_to_second"] = result.get(
            "two_view_rotation_matrix_first_to_second"
        )
        result["stored_two_view_rotation_deg"] = result.get("two_view_rotation_deg")
        result["two_view_rotation_matrix_first_to_second"] = matrix
        result["two_view_rotation_deg"] = raw.get("calibrated_reestimate_rotation_angle_deg")
        result["calibrated_reestimate_inliers"] = raw.get("calibrated_reestimate_inliers")
        result["calibrated_reestimate_translation_first_to_second"] = raw.get(
            "calibrated_reestimate_translation_first_to_second"
        )
        result["calibrated_reestimate_tri_angle_deg"] = raw.get("calibrated_reestimate_tri_angle_deg")
        result["calibrated_reestimate_cheirality_fraction"] = raw.get(
            "calibrated_reestimate_cheirality_fraction"
        )
        result["calibrated_conditioned"] = bool(classified.get("well_conditioned_calibrated"))
        result["calibrated_classification_reasons"] = list(
            classified.get("classification_reasons", [])
        )
        result["rotation_disagreement_deg"] = _rotation_geodesic_deg(final_matrix, matrix)
        bound.append(result)
    return bound


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, help="Deprecated compatibility check; never opened.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--ring-metadata", required=True, type=Path)
    parser.add_argument("--audit-manifest", required=True, type=Path)
    parser.add_argument("--exclusions", required=True, type=Path)
    parser.add_argument("--raw-audit-report", required=True, type=Path)
    parser.add_argument("--calibrated-classification", required=True, type=Path)
    parser.add_argument("--output-evidence", required=True, type=Path)
    parser.add_argument("--output-gate", required=True, type=Path)
    parser.add_argument("--snapshot-sha256", required=True)
    parser.add_argument("--logical-sha256", required=True)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--graph-report", type=Path)
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--mask-root", required=True, type=Path)
    parser.add_argument("--track-provenance", required=True, type=Path)
    args = parser.parse_args()

    ring_metadata = load_ring_metadata(args.ring_metadata)
    snapshot_manifest = load_canonical_sqlite_snapshot_manifest(args.snapshot_manifest)
    canonical_path = Path(snapshot_manifest["canonical_path"]).resolve()
    if args.database is not None and args.database.resolve() != canonical_path:
        raise RuntimeError("--database does not match the canonical_path in --snapshot-manifest")
    audit_manifest = json.loads(args.audit_manifest.read_text(encoding="utf-8"))
    pair_ids = [int(item["pair_id"]) for item in audit_manifest["pairs"]]
    raw_audit = json.loads(args.raw_audit_report.read_text(encoding="utf-8"))
    calibrated_classification = json.loads(args.calibrated_classification.read_text(encoding="utf-8"))
    track_provenance = json.loads(args.track_provenance.read_text(encoding="utf-8"))
    if not isinstance(track_provenance, dict):
        raise RuntimeError("track provenance must be a JSON object")
    manifest_sha256 = hashlib.sha256(args.audit_manifest.read_bytes()).hexdigest()
    snapshot_manifest_sha256 = hashlib.sha256(args.snapshot_manifest.read_bytes()).hexdigest()
    calibrated_classification_sha256 = hashlib.sha256(args.calibrated_classification.read_bytes()).hexdigest()
    raw_pair_ids = [int(item["pair_id"]) for item in raw_audit.get("records", []) if isinstance(item, dict) and "pair_id" in item]
    expected_pair_ids_sha256 = hashlib.sha256(
        json.dumps(pair_ids, separators=(",", ":"), sort_keys=False).encode("utf-8")
    ).hexdigest()
    raw_pair_ids_sha256 = hashlib.sha256(
        json.dumps(raw_pair_ids, separators=(",", ":"), sort_keys=False).encode("utf-8")
    ).hexdigest()
    raw_audit_bound = bool(
        raw_audit.get("database_sha256") == str(args.snapshot_sha256).lower()
        and raw_audit.get("canonical_snapshot_hash_stable") is True
        and int(raw_audit.get("pair_count", -1)) == len(pair_ids)
        and raw_audit.get("pair_source_sha256") == manifest_sha256
        and raw_audit.get("snapshot_manifest_sha256") == snapshot_manifest_sha256
        and len(raw_pair_ids) == len(pair_ids)
        and len(set(raw_pair_ids)) == len(raw_pair_ids)
        and raw_pair_ids_sha256 == expected_pair_ids_sha256
        and raw_audit.get("schema_version", 0) >= 2
        and raw_audit.get("angle_units") == {
            "triangulation_angle": "degrees",
            "rotation_angle": "degrees",
        }
        and raw_audit.get("pycolmap_angle_source_units") == {
            "TwoViewGeometry.tri_angle": "radians",
            "Rotation3d.angle": "radians",
        }
    )
    calibrated_audit_bound = bool(
        calibrated_classification.get("raw_audit_report_sha256")
        == hashlib.sha256(args.raw_audit_report.read_bytes()).hexdigest()
        and calibrated_classification.get("snapshot_manifest_sha256") == snapshot_manifest_sha256
        and calibrated_classification.get("fixed_pair_count") == len(pair_ids)
        and calibrated_classification.get("fixed_pair_ids_sha256") == expected_pair_ids_sha256
        and len(calibrated_classification.get("classification", [])) == len(pair_ids)
        and calibrated_classification.get("schema_version", 0) >= 2
        and calibrated_classification.get("angle_units") == {
            "triangulation_angle": "degrees",
            "rotation_angle": "degrees",
        }
    )
    if not calibrated_audit_bound:
        raise RuntimeError("calibrated audit classification is not bound to the fixed raw audit/manifest")
    exclusion_payload = json.loads(args.exclusions.read_text(encoding="utf-8"))
    exclusions = exclusion_payload.get("mapping_exclusions", {})
    canonical_sha_before = hashlib.sha256(canonical_path.read_bytes()).hexdigest()
    requested_snapshot_sha = str(args.snapshot_sha256).lower()
    snapshot_manifest_bound = bool(
        Path(str(snapshot_manifest.get("canonical_path", ""))).resolve() == canonical_path
        and str(snapshot_manifest.get("canonical_sha256", "")).lower() == requested_snapshot_sha
        and str(snapshot_manifest.get("creation_logical_sha256", "")).lower() == str(args.logical_sha256).lower()
    )
    if not snapshot_manifest_bound:
        raise RuntimeError("canonical SQLite snapshot does not match the recorded creation manifest")
    if canonical_sha_before != requested_snapshot_sha:
        raise RuntimeError(
            f"canonical SQLite raw SHA does not match the recorded lineage: "
            f"{canonical_sha_before} != {requested_snapshot_sha}"
        )
    with tempfile.TemporaryDirectory(prefix="v4_sparse_candidate_audit_") as temporary:
        working_path = Path(temporary) / "working.db"
        disposable = create_disposable_sqlite_from_manifest(args.snapshot_manifest, working_path)
        working_copy = disposable["working"]
        database = pycolmap.Database.open(str(working_path))
        reconstruction = pycolmap.Reconstruction(str(args.model))
        try:
            evidence = collect_sparse_pair_evidence(
                database,
                reconstruction,
                ring_metadata,
                pair_ids=pair_ids,
            )
            evidence = _bind_calibrated_audit_evidence(
                evidence,
                raw_audit,
                calibrated_classification,
            )
            classification_by_pair = {
                int(item["pair_id"]): item
                for item in calibrated_classification.get("classification", [])
                if isinstance(item, dict) and "pair_id" in item
            }
            translation_records = []
            for item in raw_audit.get("records", []):
                if not isinstance(item, dict) or "pair_id" not in item:
                    continue
                merged = dict(item)
                merged.update(
                    {
                        "well_conditioned_calibrated": bool(
                            classification_by_pair.get(int(item["pair_id"]), {}).get(
                                "well_conditioned_calibrated", False
                            )
                        )
                    }
                )
                translation_records.append(merged)
            mask_projection = sparse_mask_projection_evidence(
                reconstruction,
                args.mask_root,
                ring_metadata,
            )
            camera_center_translation = sparse_camera_center_translation_evidence(
                reconstruction,
                translation_records,
            )
            gate = sparse_integrity_gate(
                reconstruction,
                ring_metadata,
                evidence,
                independent_audit_evidence=evidence,
                expected_image_names=[
                    name for name, item in ring_metadata.items() if item.get("selected", True)
                ],
                model_root=args.model,
                database_lineage={
                    "snapshot_sha256": requested_snapshot_sha,
                    "logical_sha256": str(args.logical_sha256).lower(),
                    "snapshot_path": str(canonical_path),
                    "working_copy_sha256": working_copy["sha256"],
                },
                expected_model_sha256=args.expected_model_sha256,
                audit_exclusions=exclusions,
                track_provenance=track_provenance,
                mask_projection_evidence=mask_projection,
                camera_center_evidence=camera_center_translation,
            )
        finally:
            database.close()
    canonical_sha_after = hashlib.sha256(canonical_path.read_bytes()).hexdigest()
    canonical_snapshot_hash_stable = canonical_sha_before == canonical_sha_after == requested_snapshot_sha
    gate = dict(gate)
    gate["checks"] = dict(gate.get("checks", {}))
    gate["checks"]["canonical_snapshot_hash_stable"] = canonical_snapshot_hash_stable
    gate["checks"]["independent_raw_audit_bound"] = raw_audit_bound
    gate["checks"]["calibrated_audit_bound"] = calibrated_audit_bound
    gate["checks"]["calibrated_angle_units_bound"] = bool(
        raw_audit.get("angle_units") == {
            "triangulation_angle": "degrees",
            "rotation_angle": "degrees",
        }
        and calibrated_classification.get("angle_units") == {
            "triangulation_angle": "degrees",
            "rotation_angle": "degrees",
        }
    )
    gate["checks"]["canonical_snapshot_lineage_manifest_bound"] = snapshot_manifest_bound
    gate["passed"] = bool(
        gate.get("passed")
        and canonical_snapshot_hash_stable
        and raw_audit_bound
        and calibrated_audit_bound
        and gate["checks"]["calibrated_angle_units_bound"]
        and snapshot_manifest_bound
    )
    graph_lineage = {}
    if args.graph_report:
        graph_payload = json.loads(args.graph_report.read_text(encoding="utf-8"))
        graph_lineage = {
            "path": str(args.graph_report.resolve()),
            "sha256": hashlib.sha256(args.graph_report.read_bytes()).hexdigest(),
            "graph_sha256": graph_payload.get("graph_sha256"),
            "pair_count": graph_payload.get("pair_count"),
        }
    evidence_payload = {
        "schema_version": 2,
        "method": "fixed independent SQLite-snapshot audit set; full first-to-second SO(3) relative-rotation geodesic",
        "database_path": str(canonical_path),
        "database_sha256": canonical_sha_after,
        "canonical_snapshot_sha256_before": canonical_sha_before,
        "canonical_snapshot_sha256_after": canonical_sha_after,
        "canonical_snapshot_hash_stable": canonical_snapshot_hash_stable,
        "canonical_snapshot_lineage_manifest_path": str(args.snapshot_manifest.resolve()),
        "canonical_snapshot_lineage_manifest_sha256": snapshot_manifest_sha256,
        "canonical_snapshot_lineage_manifest_bound": snapshot_manifest_bound,
        "working_database_sha256": working_copy["sha256"],
        "audit_manifest_path": str(args.audit_manifest.resolve()),
        "audit_manifest_sha256": manifest_sha256,
        "raw_audit_report_path": str(args.raw_audit_report.resolve()),
        "raw_audit_report_sha256": hashlib.sha256(args.raw_audit_report.read_bytes()).hexdigest(),
        "raw_audit_bound": raw_audit_bound,
        "calibrated_classification_path": str(args.calibrated_classification.resolve()),
        "calibrated_classification_sha256": calibrated_classification_sha256,
        "calibrated_audit_bound": calibrated_audit_bound,
        "raw_audit_pair_ids_sha256": raw_pair_ids_sha256,
        "track_provenance_path": str(args.track_provenance.resolve()),
        "track_provenance_sha256": hashlib.sha256(args.track_provenance.read_bytes()).hexdigest(),
        "mask_root": str(args.mask_root.resolve()),
        "model_path": str(args.model.resolve()),
        "evidence": evidence,
    }
    gate_payload = dict(gate)
    gate_payload.update(
        {
            "schema_version": 2,
            "method": "strict V4 sparse acceptance from independent SQLite snapshot evidence",
            "database_path": str(canonical_path),
            "model_path": str(args.model.resolve()),
            "model_sha256_expected": args.expected_model_sha256,
            "audit_manifest_sha256": manifest_sha256,
            "raw_audit_report_path": str(args.raw_audit_report.resolve()),
            "raw_audit_report_sha256": hashlib.sha256(args.raw_audit_report.read_bytes()).hexdigest(),
            "calibrated_classification_path": str(args.calibrated_classification.resolve()),
            "calibrated_classification_sha256": calibrated_classification_sha256,
            "calibrated_audit_bound": calibrated_audit_bound,
            "independent_raw_audit_bound": raw_audit_bound,
            "raw_audit_pair_ids_sha256": raw_pair_ids_sha256,
            "expected_pair_ids_sha256": expected_pair_ids_sha256,
            "track_provenance_path": str(args.track_provenance.resolve()),
            "track_provenance_sha256": hashlib.sha256(args.track_provenance.read_bytes()).hexdigest(),
            "mask_root": str(args.mask_root.resolve()),
            "exclusion_report_sha256": hashlib.sha256(args.exclusions.read_bytes()).hexdigest(),
            "graph_lineage": graph_lineage,
            "initializer": {"image_id1": 56, "image_id2": 59},
            "canonical_snapshot_sha256_before": canonical_sha_before,
            "canonical_snapshot_sha256_after": canonical_sha_after,
            "canonical_snapshot_hash_stable": canonical_snapshot_hash_stable,
            "canonical_snapshot_lineage_manifest_path": str(args.snapshot_manifest.resolve()),
            "canonical_snapshot_lineage_manifest_sha256": snapshot_manifest_sha256,
            "canonical_snapshot_lineage_manifest_bound": snapshot_manifest_bound,
            "working_database_sha256": working_copy["sha256"],
            "independent_audit_pair_count_expected": len(pair_ids),
            "independent_audit_pair_count_observed": len(evidence),
            "independent_audit_pair_ids_sha256": hashlib.sha256(
                json.dumps(pair_ids, separators=(",", ":"), sort_keys=False).encode("utf-8")
            ).hexdigest(),
        }
    )
    args.output_evidence.parent.mkdir(parents=True, exist_ok=True)
    args.output_gate.parent.mkdir(parents=True, exist_ok=True)
    args.output_evidence.write_text(json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_gate.write_text(json.dumps(gate_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": gate_payload["passed"], "checks": gate_payload["checks"], "model_sha256": gate_payload.get("model_sha256")}, sort_keys=True))
    return 0 if gate_payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
