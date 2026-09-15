"""Freeze the versioned V4 dense/Poisson package handed to Blender.

The historical full dense and trim-5 Poisson reports were written while the
same sparse bytes were named as V1.  The current authoritative checkpoint is
the V2 sparse report.  This module creates new, immutable handoff records that
rebind those unchanged bytes to V2, records the completed bounded high-ring
attempt, and keeps every strict failure visible.  It never edits a mesh,
reconstruction, Blender scene, or GLB artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v4_config import RECONSTRUCTION_V4_ROOT, sha256_file, write_json
from v4_dense import load_accepted_sparse_lineage, validate_patch_match_runtime_phases
from v4_mesh import raw_mesh_metrics
from v4_repair import raw_component_gate


ANATOMY_KEYS = (
    "bowl_interior",
    "rim",
    "globe_shoulder",
    "continuous_neck",
    "lid_tiers",
    "finial",
    "pedestal_transitions",
    "base",
)

DEFAULT_SPARSE = RECONSTRUCTION_V4_ROOT / "repair" / "sparse_v1" / "best_defensible_sparse_v2.json"
DEFAULT_DENSE = RECONSTRUCTION_V4_ROOT / "reports" / "dense_best_defensible_v1_with_min3.json"
DEFAULT_UPPER = RECONSTRUCTION_V4_ROOT / "reports" / "dense_upper_geo_g12_3072_v2_run.json"
DEFAULT_TRIM10 = RECONSTRUCTION_V4_ROOT / "mesh" / "poisson_best_defensible_v1_depth13_trim10.ply"
DEFAULT_TRIM5 = RECONSTRUCTION_V4_ROOT / "mesh" / "poisson_best_defensible_v1_depth13_trim5.ply"
DEFAULT_ANATOMY = RECONSTRUCTION_V4_ROOT / "previews" / "raw_poisson_best_defensible_v1_depth13_trim5_evidence.json"
DEFAULT_DIAGNOSTIC = RECONSTRUCTION_V4_ROOT / "reports" / "raw_poisson_best_defensible_v1_depth13_trim5_diagnostic.json"


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def _same_hash(left: Any, right: Any) -> bool:
    return str(left).strip().lower() == str(right).strip().lower()


def _compact_upper_recovery(path: Path) -> dict[str, Any]:
    run_path = _require_file(path, "upper recovery run report")
    run = _read_object(run_path)
    patch = run.get("patch_match")
    if not isinstance(patch, Mapping) or patch.get("status") != "completed":
        raise ValueError("upper recovery report does not record a completed production PatchMatch")
    phase = patch.get("runtime_phase_gate")
    if not isinstance(phase, Mapping) or phase.get("passed") is not True:
        phase = validate_patch_match_runtime_phases(Path(str(patch.get("log_path", ""))))
    if phase.get("passed") is not True:
        raise ValueError("upper recovery production PatchMatch phase gate failed")
    counts = run.get("map_counts")
    if not isinstance(counts, Mapping) or any(int(counts.get(key, 0)) != 37 for key in (
        "photometric_depth_count",
        "photometric_normal_count",
        "geometric_depth_count",
        "geometric_normal_count",
    )):
        raise ValueError("upper recovery report does not contain exactly 37 maps of every required type")
    if int(run.get("target_reference_count", 0)) != 37:
        raise ValueError("upper recovery report does not contain exactly 37 target references")
    fused_path = _require_file(Path(str(run.get("fused_path", ""))), "upper recovery fused cloud")
    fused_sha = sha256_file(fused_path)
    if not _same_hash(fused_sha, run.get("fused_sha256")):
        raise ValueError("upper recovery fused cloud hash does not match its run report")
    post_path = _require_file(Path(str(run.get("postfusion_report", ""))), "upper recovery post-fusion report")
    post = _read_object(post_path)
    if not _same_hash(fused_sha, post.get("fused_sha256")):
        raise ValueError("upper recovery fused cloud hash does not match its post-fusion report")
    decision = run.get("decision")
    if not isinstance(decision, Mapping) or decision.get("promote_upper_recovery") is True:
        raise ValueError("upper recovery report is not the bounded unresolved result expected for comparison-only evidence")
    anatomy = post.get("contamination", {}).get("anatomy", {})
    if not isinstance(anatomy, Mapping):
        anatomy = {}
    return {
        "path": str(run_path),
        "sha256": sha256_file(run_path),
        "tag": run.get("tag"),
        "ring": run.get("ring"),
        "status": run.get("status"),
        "target_reference_count": int(run.get("target_reference_count", 0)),
        "map_counts": dict(counts),
        "runtime_phase_gate": dict(phase),
        "patch_match_log": str(patch.get("log_path", "")),
        "patch_match_log_sha256": patch.get("log_sha256"),
        "fused_path": str(fused_path),
        "fused_sha256": fused_sha,
        "postfusion_report": str(post_path),
        "postfusion_report_sha256": sha256_file(post_path),
        "postfusion_status": post.get("status"),
        "postfusion_anatomy_failures": list(anatomy.get("failures", [])),
        "candidate_finial_shape": dict(run.get("candidate_finial_shape", {})),
        "decision": dict(decision),
    }


def rebind_dense_lineage(
    *,
    dense_selection_path: Path,
    sparse_report_path: Path,
    upper_run_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Create a V2-authoritative dense selection without changing old reports."""

    dense_path = _require_file(dense_selection_path, "historical dense selection")
    sparse_path = _require_file(sparse_report_path, "authoritative sparse V2 report")
    payload = _read_object(dense_path)
    selected = payload.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("dense selection has no selected candidate")
    if selected.get("strict_gate_passed") is True:
        raise ValueError("this handoff rebinding is only for the best-defensible dense candidate")
    current_lineage = load_accepted_sparse_lineage(sparse_path)
    old_lineage = selected.get("sparse_lineage")
    if not isinstance(old_lineage, Mapping):
        raise ValueError("historical dense selection has no sparse lineage")
    for key in ("accepted_sparse_model_sha256", "accepted_sparse_gate_sha256"):
        if not _same_hash(old_lineage.get(key), current_lineage.get(key)):
            raise ValueError(f"historical dense selection is not byte-compatible with sparse V2: {key}")

    fused_path = _require_file(Path(str(selected.get("fused_path", ""))), "selected dense fused cloud")
    fused_sha = sha256_file(fused_path)
    if not _same_hash(fused_sha, selected.get("fused_sha256")):
        raise ValueError("selected dense fused cloud hash does not match the historical selection")
    run_path = _require_file(Path(str(selected.get("run_report", ""))), "selected dense run report")
    post_path = _require_file(Path(str(selected.get("postfusion_report", ""))), "selected dense post-fusion report")
    run = _read_object(run_path)
    post = _read_object(post_path)
    for report, label in ((run, "run"), (post, "post-fusion")):
        lineage = report.get("sparse_lineage")
        if not isinstance(lineage, Mapping):
            raise ValueError(f"selected dense {label} report has no sparse lineage")
        for key in ("accepted_sparse_model_sha256", "accepted_sparse_gate_sha256"):
            if not _same_hash(lineage.get(key), current_lineage.get(key)):
                raise ValueError(f"selected dense {label} report is not byte-compatible with sparse V2: {key}")
        if not _same_hash(report.get("fused_sha256"), fused_sha):
            raise ValueError(f"selected dense {label} report is not bound to the fused cloud")

    rebound_selected = dict(selected)
    rebound_selected["sparse_lineage"] = current_lineage
    rebound_selected["fused_sha256"] = fused_sha
    upper = _compact_upper_recovery(upper_run_path)
    report = {
        "schema_version": 1,
        "status": "best_defensible_dense_candidate",
        "best_defensible": True,
        "strict_gate_passed": False,
        "selection_policy": "retain the complete two-phase 372-view dense winner; a separate 3072px geo_g12 recovery is comparison-only unless it resolves a narrower finial and improves the measured baseline",
        "lineage_authority": {
            "accepted_sparse_report": str(sparse_path),
            "accepted_sparse_report_sha256": sha256_file(sparse_path),
            "accepted_sparse_lineage": current_lineage,
        },
        "historical_dense_selection": {
            "path": str(dense_path),
            "sha256": sha256_file(dense_path),
            "historical_sparse_lineage": dict(old_lineage),
        },
        "selected": rebound_selected,
        "upper_ring_recovery": upper,
        "strict_failure_reasons": list(selected.get("strict_failure_reasons", [])),
        "lineage_rebind": {
            "status": "rebound_without_geometry_change",
            "model_bytes_unchanged": True,
            "gate_bytes_unchanged": True,
            "historical_dense_reports_preserved": True,
            "old_sparse_report_path": old_lineage.get("accepted_sparse_report_path"),
            "new_sparse_report_path": str(sparse_path),
        },
        "poisson_policy": "the selected fused cloud may feed only the best-defensible Poisson handoff while strict dense/anatomy failures remain explicit",
    }
    write_json(output_path.resolve(), report)
    return report


def _mesh_summary(path: Path, metrics: Mapping[str, Any], component_gate: Mapping[str, Any]) -> dict[str, Any]:
    components = metrics.get("components") if isinstance(metrics.get("components"), Mapping) else {}
    counts = components.get("face_counts") if isinstance(components.get("face_counts"), Sequence) else []
    face_counts = [int(value) for value in counts if isinstance(value, (int, float)) and int(value) >= 0]
    second_count = face_counts[1] if len(face_counts) > 1 else 0
    return {
        "path": str(path),
        "sha256": metrics.get("source_sha256"),
        "vertex_count": int(metrics.get("vertex_count", 0)),
        "face_count": int(metrics.get("face_count", 0)),
        "component_count": int(components.get("component_count", 0)),
        "dominant_face_fraction": float(components.get("dominant_face_fraction", 0.0)),
        "second_largest_face_fraction": float(component_gate.get("second_largest_face_fraction", 0.0)),
        "largest_non_dominant_face_count": second_count,
        "finite_xyz_fraction": float(metrics.get("finite_xyz_fraction", 0.0)),
        "rank": int(metrics.get("rank", 0)),
        "has_color": bool(metrics.get("has_color", False)),
        "bounding_box_extents": list(metrics.get("bounding_box_extents", [])),
        "component_gate": dict(component_gate),
    }


def detached_component_assessment(
    metrics: Mapping[str, Any],
    component_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Report the measurable component proxy without claiming semantic proof."""

    raw_components = metrics.get("components")
    components = raw_components if isinstance(raw_components, Mapping) else {}
    raw_counts = components.get("face_counts")
    counts = raw_counts if isinstance(raw_counts, Sequence) else []
    second_count = int(counts[1]) if len(counts) > 1 else 0
    second_fraction = float(component_gate.get("second_largest_face_fraction", 0.0))
    thresholds = component_gate.get("thresholds")
    threshold = float(thresholds.get("second_max", 0.02)) if isinstance(thresholds, Mapping) else 0.02
    return {
        "status": "bounded_face_fraction_proxy",
        "passed": False,
        "largest_non_dominant_face_count": second_count,
        "largest_non_dominant_face_fraction": second_fraction,
        "threshold_fraction": threshold,
        "fraction_proxy_passed": second_fraction <= threshold,
        "semantic_detachedness_proven": False,
        "limitation": "Persisted Poisson evidence contains connected-component sizes but no semantic/spatial provenance proving that every detached component is supported anatomy.",
    }


def freeze_poisson(
    *,
    dense_handoff_path: Path,
    mesh_path: Path,
    anatomy_path: Path,
    diagnostic_path: Path,
    comparison_mesh_paths: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Select/hash-bind the strongest connected scan-derived Poisson mesh."""

    dense_path = _require_file(dense_handoff_path, "dense handoff report")
    dense = _read_object(dense_path)
    selected_dense = dense.get("selected")
    if not isinstance(selected_dense, Mapping) or selected_dense.get("strict_gate_passed") is True:
        raise ValueError("Poisson handoff requires the best-defensible, not strict, dense candidate")
    dense_lineage = selected_dense.get("sparse_lineage")
    if not isinstance(dense_lineage, Mapping) or not dense_lineage.get("accepted_sparse_report_path", "").endswith("best_defensible_sparse_v2.json"):
        raise ValueError("dense handoff is not explicitly bound to sparse V2")
    fused_path = _require_file(Path(str(selected_dense.get("fused_path", ""))), "dense fused cloud")
    fused_sha = sha256_file(fused_path)
    if not _same_hash(fused_sha, selected_dense.get("fused_sha256")):
        raise ValueError("dense handoff fused cloud hash changed")

    mesh = _require_file(mesh_path, "selected Poisson mesh")
    metrics = raw_mesh_metrics(mesh)
    component_gate = raw_component_gate(metrics)
    anatomy = _read_object(_require_file(anatomy_path, "Poisson anatomy evidence"))
    if not _same_hash(anatomy.get("mesh_sha256"), metrics.get("source_sha256")):
        raise ValueError("Poisson anatomy evidence is not bound to the selected mesh")
    anatomy_regions = anatomy.get("anatomy_region_evidence")
    if not isinstance(anatomy_regions, Mapping):
        raise ValueError("Poisson anatomy evidence is missing anatomy_region_evidence")
    regions = anatomy_regions.get("regions")
    if not isinstance(regions, Mapping) or set(regions) != set(ANATOMY_KEYS):
        raise ValueError("Poisson anatomy evidence does not contain all eight regions")

    candidates: list[dict[str, Any]] = []
    for candidate_path in [mesh, *comparison_mesh_paths]:
        candidate = _require_file(candidate_path, "Poisson comparison mesh")
        candidate_metrics = metrics if candidate == mesh else raw_mesh_metrics(candidate)
        candidate_gate = component_gate if candidate == mesh else raw_component_gate(candidate_metrics)
        candidates.append(_mesh_summary(candidate, candidate_metrics, candidate_gate))
    selected_summary = candidates[0]
    for candidate in candidates[1:]:
        current_key = (
            bool(selected_summary["component_gate"].get("passed")),
            selected_summary["dominant_face_fraction"],
            -selected_summary["second_largest_face_fraction"],
            selected_summary["finite_xyz_fraction"],
            selected_summary["rank"],
            selected_summary["face_count"],
        )
        candidate_key = (
            bool(candidate["component_gate"].get("passed")),
            candidate["dominant_face_fraction"],
            -candidate["second_largest_face_fraction"],
            candidate["finite_xyz_fraction"],
            candidate["rank"],
            candidate["face_count"],
        )
        if candidate_key > current_key:
            raise ValueError("selected Poisson mesh is not the strongest connected candidate by the recorded policy")

    detached_assessment = detached_component_assessment(metrics, component_gate)
    anatomy_failures = [str(value) for value in anatomy_regions.get("failures", [])]
    if anatomy_regions.get("passed") is not True:
        anatomy_failure_reason = "raw_anatomy_region_gate_failed"
    else:
        anatomy_failure_reason = None
    blocking = ["dense_strict_postfusion_gate_failed", *anatomy_failures]
    if anatomy_failure_reason and anatomy_failure_reason not in blocking:
        blocking.append(anatomy_failure_reason)
    if anatomy_regions.get("no_major_vessel_scale_holes") is not True:
        blocking.append("raw_no_major_vessel_scale_holes_failed")
    blocking.append("unsupported_large_detached_component_not_proven")
    diagnostic = _require_file(diagnostic_path, "historical Poisson diagnostic report")
    diagnostic_payload = _read_object(diagnostic)
    diagnostic_mesh = diagnostic_payload.get("poisson", {}).get("mesh_sha256")
    if not _same_hash(diagnostic_mesh, metrics.get("source_sha256")):
        raise ValueError("historical Poisson diagnostic does not refer to the selected mesh")
    diagnostic_dense = diagnostic_payload.get("dense_selection")
    if not isinstance(diagnostic_dense, Mapping) or not _same_hash(diagnostic_dense.get("selected_fused_sha256"), fused_sha):
        raise ValueError("historical Poisson diagnostic is not bound to the selected dense fused cloud")

    report = {
        "schema_version": 2,
        "status": "best_defensible_poisson_handoff",
        "best_defensible": True,
        "strict_gate_passed": False,
        "handoff_allowed": True,
        "promotion_allowed": False,
        "handoff_policy": "completion-first bounded CV continuation: hand the strongest genuine connected scan-derived Poisson mesh to Blender while keeping strict anatomy and detached-component limitations explicit",
        "dense_selection": {
            "path": str(dense_path),
            "sha256": sha256_file(dense_path),
            "status": dense.get("status"),
            "selected_fused_path": str(fused_path),
            "selected_fused_sha256": fused_sha,
            "strict_gate_passed": selected_dense.get("strict_gate_passed"),
            "strict_failure_reasons": list(selected_dense.get("strict_failure_reasons", [])),
            "sparse_lineage": dict(dense_lineage),
        },
        "poisson": {
            "algorithm": "COLMAP Poisson",
            "selected": selected_summary,
            "attempts": candidates,
            "input_dense_fused_path": str(fused_path),
            "input_dense_fused_sha256": fused_sha,
            "component_gate": dict(component_gate),
            "detached_component_assessment": detached_assessment,
            "historical_diagnostic": {"path": str(diagnostic), "sha256": sha256_file(diagnostic)},
            "generation_provenance": {
                "status": "artifact_and_report_recovered",
                "algorithm": "COLMAP poisson_mesher",
                "exact_historical_invocation_persisted": False,
                "limitation": "The historical trim-5 shell did not retain its exact invoking command; the bytes, COLMAP algorithm identity, metrics, and anatomy evidence are hash-bound.",
            },
        },
        "anatomy_evidence": {
            "path": str(_require_file(anatomy_path, "Poisson anatomy evidence")),
            "sha256": sha256_file(anatomy_path),
            "status": anatomy.get("status"),
            "passed": anatomy_regions.get("passed") is True,
            "failures": anatomy_failures,
            "finial_shape": dict(anatomy_regions.get("finial_shape", {})),
            "no_major_vessel_scale_holes": anatomy_regions.get("no_major_vessel_scale_holes") is True,
            "region_names": list(regions),
            "region_count": len(regions),
        },
        "upper_ring_recovery": dense.get("upper_ring_recovery"),
        "blocking_reasons": blocking,
        "blender_input": {
            "allowed": True,
            "mesh_path": str(mesh),
            "mesh_sha256": metrics.get("source_sha256"),
            "strict_promotion_remains_false": True,
            "required_downstream_constraints": [
                "scan-preserving scripted cleanup only",
                "no manual anatomy completion",
                "carry strict failures into final documentation",
            ],
        },
    }
    write_json(output_path.resolve(), report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    dense = subparsers.add_parser("dense")
    dense.add_argument("--dense-selection", type=Path, default=DEFAULT_DENSE)
    dense.add_argument("--sparse-report", type=Path, default=DEFAULT_SPARSE)
    dense.add_argument("--upper-run", type=Path, default=DEFAULT_UPPER)
    dense.add_argument("--output", type=Path, required=True)
    poisson = subparsers.add_parser("poisson")
    poisson.add_argument("--dense-handoff", type=Path, required=True)
    poisson.add_argument("--mesh", type=Path, default=DEFAULT_TRIM5)
    poisson.add_argument("--anatomy", type=Path, default=DEFAULT_ANATOMY)
    poisson.add_argument("--diagnostic", type=Path, default=DEFAULT_DIAGNOSTIC)
    poisson.add_argument("--comparison-mesh", type=Path, action="append", default=[DEFAULT_TRIM10])
    poisson.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "dense":
        report = rebind_dense_lineage(
            dense_selection_path=args.dense_selection,
            sparse_report_path=args.sparse_report,
            upper_run_path=args.upper_run,
            output_path=args.output,
        )
        print(json.dumps({"status": report["status"], "selected_fused_sha256": report["selected"]["fused_sha256"], "upper_status": report["upper_ring_recovery"]["status"]}, sort_keys=True))
    else:
        report = freeze_poisson(
            dense_handoff_path=args.dense_handoff,
            mesh_path=args.mesh,
            anatomy_path=args.anatomy,
            diagnostic_path=args.diagnostic,
            comparison_mesh_paths=args.comparison_mesh,
            output_path=args.output,
        )
        print(json.dumps({"status": report["status"], "mesh_sha256": report["poisson"]["selected"]["sha256"], "strict_gate_passed": report["strict_gate_passed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
