"""Freeze the best measured V4 sparse candidate for the realistic-result path.

This does not rewrite any sparse model or turn a failed strict gate into a
pass.  It compares the versioned candidates, selects the strongest candidate
that is still defensible for downstream CV work, and emits a separate,
explicit best-defensible lineage consumed by the fresh dense runner.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v4_repair import stable_directory_sha256


REPAIR_ROOT = PROJECT_ROOT / "reconstruction" / "v4" / "repair" / "sparse_v1"
SELECTION_PATH = REPAIR_ROOT / "best_defensible_sparse_v1_selection.json"
LINEAGE_PATH = REPAIR_ROOT / "best_defensible_sparse_v1.json"
V51_FAILURE_PATH = REPAIR_ROOT / "v51_solved_center_native_joint_ba_failure.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(name: str, gate_name: str, provenance_name: str) -> dict[str, Any]:
    gate_path = REPAIR_ROOT / gate_name
    provenance_path = REPAIR_ROOT / provenance_name
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    model_path = Path(str(provenance["model_directory"]))
    observed_model_sha = stable_directory_sha256(model_path)
    expected_model_sha = str(provenance["model_directory_sha256"]).lower()
    if observed_model_sha != expected_model_sha:
        raise RuntimeError(f"{name}: model bytes changed since provenance was recorded")
    if str(gate.get("model_sha256", "")).lower() != observed_model_sha:
        raise RuntimeError(f"{name}: sparse gate is not bound to the measured model hash")
    mask = gate.get("mask_projection", {}).get("view_summary", {})
    tracks = gate.get("track_distribution", {})
    failures = gate.get("failures", {})
    checks = gate.get("checks", {})
    strict_failures = sorted(key for key, value in checks.items() if value is not True)
    exact_372 = (
        int(gate.get("registered_images", 0)) == 372
        and int(gate.get("model_count", 0)) == 1
        and bool(checks.get("registered_image_set_exact"))
        and bool(checks.get("single_coherent_model"))
    )
    pose_failure_count = len(failures.get("rotation", []))
    shared_failure_count = len(failures.get("shared_tracks", []))
    trajectory_failure_count = len(failures.get("trajectory", []))
    pose_and_trajectory_clean = pose_failure_count == 0 and trajectory_failure_count == 0
    provenance_clean = bool(
        provenance.get("independent_graph")
        and not provenance.get("constructed_pairwise_tracks", True)
        and checks.get("stable_model_hash_bound") is True
        and checks.get("sqlite_snapshot_lineage_bound") is True
        and checks.get("cross_ring_graph_connected") is True
    )
    defensible = bool(exact_372 and pose_and_trajectory_clean and provenance_clean)
    return {
        "name": name,
        "gate_path": str(gate_path.resolve()),
        "gate_sha256": _sha256(gate_path),
        "provenance_path": str(provenance_path.resolve()),
        "provenance_sha256": _sha256(provenance_path),
        "model_directory": str(model_path.resolve()),
        "model_directory_sha256": observed_model_sha,
        "strict_gate_passed": bool(gate.get("passed")),
        "strict_gate_failures": strict_failures,
        "exact_372_model": exact_372,
        "pose_and_trajectory_clean": pose_and_trajectory_clean,
        "provenance_clean": provenance_clean,
        "defensible_for_downstream_cv": defensible,
        "registered_images": int(gate.get("registered_images", 0)),
        "points3D": int(gate.get("sparse_points", 0)),
        "observations": int(gate.get("observations", 0)),
        "mean_track_length": float(tracks.get("mean_track_length", 0.0)),
        "fraction_track_length_ge3": float(tracks.get("fraction_track_length_ge3", 0.0)),
        "mean_reprojection_error": float(gate.get("mean_reprojection_error", float("inf"))),
        "mask_p10": {
            "precision": float(mask.get("precision", {}).get("p10", 0.0)),
            "recall": float(mask.get("recall", {}).get("p10", 0.0)),
            "iou": float(mask.get("iou", {}).get("p10", 0.0)),
        },
        "pose_failure_count": pose_failure_count,
        "shared_track_failure_count": shared_failure_count,
        "trajectory_failure_count": trajectory_failure_count,
        "translation_positive_scale_fraction": float(
            gate.get("camera_center_translation", {}).get("positive_projected_scale_fraction", 0.0)
        ),
        "ring_graph_connected": bool(gate.get("ring_graph_connectivity", {}).get("connected")),
    }


def _selection_score(candidate: dict[str, Any]) -> tuple[float, float, float, float, float]:
    mask = candidate["mask_p10"]
    return (
        float(mask["iou"]),
        float(mask["precision"]),
        float(mask["recall"]),
        float(candidate["fraction_track_length_ge3"]),
        -float(candidate["mean_reprojection_error"]),
    )


def run() -> dict[str, Any]:
    if SELECTION_PATH.exists() or LINEAGE_PATH.exists():
        raise FileExistsError("best-defensible sparse reports already exist; refusing to overwrite")
    candidates = [
        _candidate(
            "v43_direct_calibrated_global",
            "repair_sparse_gate_v43_match_conflict_free_outlier_cap15_direct_calibrated_global.json",
            "glomap_v43_match_conflict_free_outlier_cap15_direct_calibrated_global_track_provenance.json",
        ),
        _candidate(
            "v47_v43_fixed_pose_retriangulation",
            "repair_sparse_gate_v47_v43_fixed_pose_retriangulation.json",
            "track_provenance_v47_v43_fixed_pose_retriangulation.json",
        ),
        _candidate(
            "v49_v26_pose_exact_graph_retriangulation",
            "v49_sparse_gate.json",
            "track_provenance_v49_v26_pose_exact_graph_retriangulation.json",
        ),
        _candidate(
            "v50b_solved_center_native_ba_diagnostic",
            "v50b_sparse_gate.json",
            "track_provenance_v50b_solved_center_native_ba.json",
        ),
    ]
    defensible = [candidate for candidate in candidates if candidate["defensible_for_downstream_cv"]]
    if not defensible:
        raise RuntimeError("no measured 372-view candidate satisfies the best-defensible prerequisites")
    selected = max(defensible, key=_selection_score)
    selection_payload = {
        "schema_version": 1,
        "status": "best_defensible_sparse_selection",
        "method": (
            "compare hash-bound versioned sparse candidates; require exact 372-view one-model lineage, "
            "zero independent pose/trajectory failures, genuine non-pair-local tracks, and connected ring graph; "
            "rank remaining candidates by mask IoU/precision/recall p10, multiview-track fraction, then reprojection"
        ),
        "strict_gate_was_not_weakened": True,
        "dense_source_policy": (
            "Downstream dense is explicitly sourced from this best-defensible record; strict sparse failures "
            "remain visible and are not relabeled as a strict sparse pass."
        ),
        "candidate_comparison": candidates,
        "defensible_candidate_names": [candidate["name"] for candidate in defensible],
        "selected_candidate": selected,
        "final_corrected_ba_attempt": json.loads(V51_FAILURE_PATH.read_text(encoding="utf-8")),
    }
    SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_PATH.write_text(json.dumps(selection_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selection_sha = _sha256(SELECTION_PATH)
    strict_failures = list(selected["strict_gate_failures"])
    lineage_payload = {
        "schema_version": 2,
        "status": "best_defensible_sparse_candidate",
        "best_defensible": True,
        "sparse_gate_passed": False,
        "strict_gate_failures": strict_failures,
        "source_model": selected["model_directory"],
        "source_model_sha256": selected["model_directory_sha256"],
        "sparse_gate": selected["gate_path"],
        "sparse_gate_sha256": selected["gate_sha256"],
        "selection_report": str(SELECTION_PATH.resolve()),
        "selection_report_sha256": selection_sha,
        "track_provenance": selected["provenance_path"],
        "track_provenance_sha256": selected["provenance_sha256"],
        "registered_images": selected["registered_images"],
        "ring_graph_connected": selected["ring_graph_connected"],
        "translation_positive_scale_fraction": selected["translation_positive_scale_fraction"],
        "independent_lineage": {
            "audit_manifest_sha256": "47055ecc2cdc8d9d4f2eda546b6bb71e7e31f85ac7817c7da43465fca512e358",
            "raw_audit_sha256": "39c46fb6e164655b590a776db4d1ec422b4edfb178e3ef492f908b7d5c9ad5a0",
            "classification_sha256": "cdf051b0946201732b9775be8e4cff09609e9e1839a819d3690e7c201c2bf9f7",
            "canonical_snapshot_sha256": "658aad6db7d14185ff6678d9d7fa0ed75dab3cac3d71a1f32a3d49111e73f49a",
            "canonical_logical_sha256": "ec4d9a4c9a548903d71704401e50e666408a933dfdb19cc02cbe8e93fd860f97",
            "pair_ids_sha256": "c01f5037307ff9589efb78baef5ed6dd43c95bb2bbd719af8687e69691c2adf2",
        },
        "dense_lineage_policy": "fresh_dense_from_best_defensible_sparse_only; no historical dense reuse",
    }
    LINEAGE_PATH.write_text(json.dumps(lineage_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "selected": selected["name"],
        "selected_model_sha256": selected["model_directory_sha256"],
        "strict_gate_failures": strict_failures,
        "selection_report": str(SELECTION_PATH.resolve()),
        "selection_report_sha256": selection_sha,
        "lineage_report": str(LINEAGE_PATH.resolve()),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
