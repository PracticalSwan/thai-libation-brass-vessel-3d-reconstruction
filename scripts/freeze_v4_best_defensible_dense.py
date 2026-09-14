"""Freeze a complete fresh fused cloud as explicit best-defensible evidence.

This is not a dense acceptance gate.  It requires the production runtime and
lineage invariants (both PatchMatch phases, exactly 372 geometric depth and
normal maps, a hash-bound fused cloud, and measured post-fusion evidence),
then preserves every strict failure reason while selecting the strongest
candidate by measured support, coverage, and contamination.  It never edits
the fused cloud or relabels a strict failure as a pass.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v4_config import RECONSTRUCTION_V4_ROOT, sha256_file, write_json
from v4_dense import dense_typed_file_counts, load_accepted_sparse_lineage, validate_patch_match_runtime_phases


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


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _output_path_from_workspace_setup(run: Mapping[str, Any]) -> Path:
    args = run.get("workspace_setup", {}).get("undistorter", {}).get("args", [])
    if not isinstance(args, list) or "--output_path" not in args:
        raise ValueError("dense run does not record the undistorter output workspace")
    index = args.index("--output_path")
    if index + 1 >= len(args):
        raise ValueError("dense run has an incomplete undistorter output path")
    return Path(str(args[index + 1])).resolve()


def _registered_image_names(model_path: Path) -> set[str]:
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(model_path))
    names = {Path(reconstruction.image(int(image_id)).name).name for image_id in reconstruction.reg_image_ids()}
    if len(names) != 372:
        raise ValueError(f"dense best-defensible source model must contain exactly 372 views, found {len(names)}")
    return names


def _geometric_map_names(workspace: Path, suffix: str) -> set[str]:
    if suffix == "geometric":
        root = workspace / "stereo" / "depth_maps"
    elif suffix == "normal_geometric":
        root = workspace / "stereo" / "normal_maps"
        suffix = "geometric"
    else:
        raise ValueError(f"unsupported dense map suffix: {suffix}")
    marker = f".{suffix}.bin"
    return {path.name[: -len(marker)] for path in root.glob(f"*{marker}")}


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _evidence_summary(postfusion: Mapping[str, Any]) -> dict[str, Any]:
    contamination = postfusion.get("contamination")
    if not isinstance(contamination, Mapping) or contamination.get("status") != "measured":
        raise ValueError("best-defensible dense candidate must have measured contamination evidence")
    anatomy = contamination.get("anatomy")
    if not isinstance(anatomy, Mapping):
        raise ValueError("best-defensible dense candidate must have measured anatomy evidence")
    regions = anatomy.get("regions")
    if not isinstance(regions, Mapping) or set(regions) != set(ANATOMY_KEYS):
        raise ValueError("best-defensible dense candidate must measure all eight anatomy regions")
    region_rows: dict[str, dict[str, float]] = {}
    for key in ANATOMY_KEYS:
        item = regions[key]
        if not isinstance(item, Mapping):
            raise ValueError(f"anatomy evidence is not an object: {key}")
        fields = ("supported_fraction", "projected_coverage_fraction", "connected_support_fraction")
        values = {field: _finite(item.get(field)) for field in fields}
        if any(value is None or not 0.0 <= value <= 1.0 for value in values.values()):
            raise ValueError(f"anatomy evidence is incomplete for {key}")
        region_rows[key] = {field: float(value) for field, value in values.items() if value is not None}
    metrics = contamination.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("best-defensible dense candidate is missing contamination metrics")
    contamination_rows: dict[str, float] = {}
    for key in (
        "board_point_fraction",
        "cloth_or_background_point_fraction",
        "pedestal_board_webbing_point_fraction",
    ):
        item = metrics.get(key)
        value = _finite(item.get("value")) if isinstance(item, Mapping) else None
        if value is None or not 0.0 <= value <= 1.0:
            raise ValueError(f"contamination evidence is incomplete for {key}")
        contamination_rows[key] = float(value)
    region_support = sum(row["supported_fraction"] for row in region_rows.values()) / len(region_rows)
    region_coverage = sum(row["projected_coverage_fraction"] for row in region_rows.values()) / len(region_rows)
    region_connected = sum(row["connected_support_fraction"] for row in region_rows.values()) / len(region_rows)
    contamination_penalty = sum(contamination_rows.values()) / len(contamination_rows)
    return {
        "anatomy_status": anatomy.get("status"),
        "anatomy_passed": anatomy.get("passed") is True,
        "anatomy_failures": [str(value) for value in anatomy.get("failures", [])] if isinstance(anatomy.get("failures", []), list) else [],
        "finial_shape": dict(anatomy.get("finial_shape", {})) if isinstance(anatomy.get("finial_shape"), Mapping) else {},
        "regions": region_rows,
        "contamination": contamination_rows,
        "source_selection_status": postfusion.get("source_selection_audit", {}).get("status"),
        "ring_transition_status": postfusion.get("ring_transition_audit", {}).get("status"),
        "region_support_mean": region_support,
        "region_coverage_mean": region_coverage,
        "region_connected_mean": region_connected,
        "contamination_penalty_mean": contamination_penalty,
    }


def inspect_candidate(run_path: Path) -> dict[str, Any]:
    run = _read_object(run_path)
    phase_records = run.get("phase_records")
    if not isinstance(phase_records, list) or len(phase_records) != 1:
        raise ValueError("best-defensible dense candidate must contain exactly one production tile")
    patch = phase_records[0].get("patch_match") if isinstance(phase_records[0], Mapping) else None
    if not isinstance(patch, Mapping) or patch.get("status") != "completed":
        raise ValueError("production PatchMatch tile did not complete")
    phase = patch.get("runtime_phase_gate")
    if not isinstance(phase, Mapping) or phase.get("passed") is not True:
        phase = validate_patch_match_runtime_phases(Path(str(patch.get("log_path", ""))))
    if int(phase.get("photometric_block_count", 0)) < 1 or int(phase.get("geometric_block_count", 0)) < 1:
        raise ValueError("completed geometric candidate lacks both runtime PatchMatch phases")
    workspace = _output_path_from_workspace_setup(run)
    model_path = Path(str(run.get("accepted_sparse_model", ""))).resolve()
    expected_names = _registered_image_names(model_path)
    counts = dense_typed_file_counts(workspace)
    geometric_depth_names = _geometric_map_names(workspace, "geometric")
    geometric_normal_names = _geometric_map_names(workspace, "normal_geometric")
    if counts.get("geometric_depth_count") != 372 or counts.get("geometric_normal_count") != 372:
        raise ValueError(f"best-defensible candidate does not have exactly 372 geometric maps: {counts}")
    # COLMAP stores geometric depth and normal basenames in separate folders;
    # both sets must cover the same registered image names.
    if geometric_depth_names != expected_names or geometric_normal_names != expected_names:
        raise ValueError("geometric depth/normal map names do not exactly match the 372 registered views")
    postfusion_path = Path(str(run.get("postfusion_report", ""))).resolve()
    postfusion = _read_object(postfusion_path)
    fused_path = Path(str(run.get("fused_path", ""))).resolve()
    if not fused_path.is_file():
        raise FileNotFoundError(f"fresh fused cloud is missing: {fused_path}")
    fused_sha = sha256_file(fused_path)
    if fused_sha != str(run.get("fused_sha256", "")).lower() or fused_sha != str(postfusion.get("fused_sha256", "")).lower():
        raise ValueError("fresh fused cloud hash does not match both dense reports")
    sparse_report_path = Path(str(run.get("accepted_sparse_report", ""))).resolve()
    sparse_lineage = load_accepted_sparse_lineage(sparse_report_path)
    gate = postfusion.get("postfusion_evidence_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("fresh dense candidate is missing its strict post-fusion gate")
    summary = _evidence_summary(postfusion)
    strict_passed = gate.get("passed") is True
    strict_failures = [str(value) for value in gate.get("reasons", [])] if isinstance(gate.get("reasons", []), list) else [str(gate.get("reasons"))]
    return {
        "run_report": str(run_path.resolve()),
        "run_report_sha256": sha256_file(run_path),
        "postfusion_report": str(postfusion_path),
        "postfusion_report_sha256": sha256_file(postfusion_path),
        "fused_path": str(fused_path),
        "fused_sha256": fused_sha,
        "workspace": str(workspace),
        "sparse_lineage": sparse_lineage,
        "runtime_phase_gate": dict(phase),
        "map_counts": counts,
        "registered_view_count": len(expected_names),
        "strict_gate_passed": strict_passed,
        "strict_failure_reasons": strict_failures,
        "evidence_summary": summary,
        "score": [
            1 if strict_passed else 0,
            1 if summary["anatomy_passed"] else 0,
            summary["region_support_mean"],
            summary["region_coverage_mean"],
            summary["region_connected_mean"],
            -summary["contamination_penalty_mean"],
        ],
    }


def freeze(candidates: Sequence[Path], output_path: Path) -> dict[str, Any]:
    if not candidates:
        raise ValueError("at least one fresh dense run report is required")
    inspected = [inspect_candidate(path.resolve()) for path in candidates]
    selected = max(inspected, key=lambda item: tuple(item["score"]))
    selected_strict = bool(selected["strict_gate_passed"])
    report = {
        "schema_version": 1,
        "status": "accepted_dense_candidate" if selected_strict else "best_defensible_dense_candidate",
        "best_defensible": not selected_strict,
        "strict_gate_passed": selected_strict,
        "selection_policy": "complete fresh two-phase 372-view candidates ranked by strict pass, anatomy support/coverage/connectivity, then lower measured contamination; strict failures remain visible",
        "candidate_count": len(inspected),
        "selected": selected,
        "candidates": inspected,
        "poisson_policy": "downstream Poisson may consume this exact fused cloud only as diagnostic when strict_gate_passed=false; raw Poisson promotion remains independently gated",
    }
    write_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=RECONSTRUCTION_V4_ROOT / "reports" / "dense_best_defensible_v1.json")
    args = parser.parse_args()
    report = freeze(args.candidate, args.output.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
