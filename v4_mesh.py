"""Poisson-only raw mesh evidence and visual-gate helpers for V4."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

from local_reconstruction_io import component_summary, ply_metrics, read_ply
from v4_config import RECONSTRUCTION_V4_ROOT, assert_output_path, read_json, sha256_file, write_json
from v4_dense import contamination_assessment, reviewed_evidence_gate


def poisson_command(input_path: Path, output_path: Path, *, depth: int = 13) -> list[str]:
    if depth < 1:
        raise ValueError("Poisson depth must be positive")
    executable = shutil.which("colmap")
    if not executable:
        raise RuntimeError("COLMAP executable is unavailable")
    launcher = Path(executable)
    if launcher.suffix.lower() in {".bat", ".cmd"}:
        direct = launcher.parent / "bin" / "colmap.exe"
        if direct.is_file():
            executable = str(direct)
    return [
        executable,
        "poisson_mesher",
        "--input_path",
        str(input_path),
        "--output_path",
        str(output_path),
        "--PoissonMeshing.depth",
        str(depth),
    ]


def raw_mesh_metrics(path: Path, *, reference_bounds: Mapping[str, Any] | None = None) -> dict[str, Any]:
    data = read_ply(path)
    metrics = ply_metrics(path, dict(reference_bounds) if reference_bounds else None)
    metrics["components"] = component_summary(data)
    # Record the bytes that were actually inspected.  A path/size fingerprint
    # is insufficient evidence because a different mesh can have the same
    # length at the same location.
    metrics["source_sha256"] = sha256_file(Path(path))
    return metrics


def raw_visual_gate(
    *,
    fused_metrics: Mapping[str, Any],
    mesh_metrics: Mapping[str, Any],
    visual_status: str,
    required_regions: Mapping[str, bool],
    board_point_fraction: float,
    cloth_point_fraction: float,
    review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    regions_input_valid = (
        bool(required_regions)
        and all(type(value) is bool for value in required_regions.values())
    )
    regions = {str(key): value for key, value in required_regions.items()} if regions_input_valid else {}
    contamination = contamination_assessment(fused_metrics)
    evidence = reviewed_evidence_gate(review)
    measured = contamination["measured_fractions"]
    board_input_valid = (
        not isinstance(board_point_fraction, bool)
        and isinstance(board_point_fraction, (int, float))
        and measured["board_point_fraction"] is not None
        and float(board_point_fraction) == measured["board_point_fraction"]
    )
    cloth_input_valid = (
        not isinstance(cloth_point_fraction, bool)
        and isinstance(cloth_point_fraction, (int, float))
        and measured["cloth_or_background_point_fraction"] is not None
        and float(cloth_point_fraction) == measured["cloth_or_background_point_fraction"]
    )
    review_regions_match = (
        evidence["passed"]
        and isinstance(review, Mapping)
        and {str(key): value for key, value in review.get("regions", {}).items()} == regions
    )
    checks = {
        "fused_nonzero": int(fused_metrics.get("point_count", 0)) > 0,
        "mesh_vertices": int(mesh_metrics.get("vertex_count", 0)) > 0,
        "mesh_faces": int(mesh_metrics.get("face_count", 0)) > 0,
        "mesh_finite": float(mesh_metrics.get("finite_xyz_fraction", 0.0)) == 1.0,
        "mesh_rank_3": int(mesh_metrics.get("rank", 0)) == 3,
        "required_regions": bool(regions) and all(regions.values()),
        "raw_board_fraction_input_matches_measurement": board_input_valid,
        "raw_cloth_fraction_input_matches_measurement": cloth_input_valid,
        "review_regions_match": review_regions_match,
        "visual_review": visual_status == "passed" and evidence["passed"],
    }
    checks.update(contamination["checks"])
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "required_regions": regions,
        "contamination": contamination,
        "review": evidence,
    }


def preserve_raw_evidence(source_path: Path, canonical_path: Path) -> Path:
    """Copy a verified raw result once; never overwrite an accepted canonical file."""

    source_path = Path(source_path)
    canonical = assert_output_path(canonical_path)
    if not source_path.is_file():
        raise ValueError(f"raw mesh/cloud source is missing: {source_path}")
    if canonical.exists():
        raise ValueError(f"canonical raw evidence already exists: {canonical}")
    canonical.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, canonical)
    return canonical


def write_raw_gate_report(payload: Mapping[str, Any], path: Path | None = None) -> Path:
    return write_json(path or RECONSTRUCTION_V4_ROOT / "reports" / "raw_visual_gate.json", dict(payload))


def finalize_raw_visual_gate(
    report_path: Path,
    *,
    review: Mapping[str, Any],
    review_basis: Sequence[str],
) -> dict[str, Any]:
    """Accept the raw mesh only with explicit, independently approved evidence."""

    evidence = reviewed_evidence_gate(review)
    basis = [str(value).strip() for value in review_basis if str(value).strip()]
    if not evidence["passed"] or not basis:
        raise ValueError("; ".join(evidence["reasons"]) or "raw visual review evidence is incomplete")
    report = read_json(report_path)
    fused_metrics = report.get("fused_metrics", {})
    mesh_metrics = report.get("mesh_metrics", {})
    board_input = fused_metrics.get("board_point_fraction")
    cloth_input = fused_metrics.get("cloth_point_fraction")
    if isinstance(board_input, bool) or not isinstance(board_input, (int, float)):
        raise ValueError("fused board_point_fraction must be explicitly measured")
    if isinstance(cloth_input, bool) or not isinstance(cloth_input, (int, float)):
        raise ValueError("fused cloth_point_fraction must be explicitly measured")
    gate = raw_visual_gate(
        fused_metrics=fused_metrics,
        mesh_metrics=mesh_metrics,
        visual_status="passed",
        required_regions=evidence["regions"],
        board_point_fraction=float(board_input),
        cloth_point_fraction=float(cloth_input),
        review=review,
    )
    if not gate["passed"]:
        reasons = [
            *gate["contamination"]["reasons"],
            *gate["review"]["reasons"],
        ]
        raise ValueError("; ".join(reasons) or "raw quantitative gate failed")
    report["visual_gate"] = {
        "status": "passed",
        "previews": evidence["preview_evidence"],
        "reviewer": evidence["reviewer"],
        "approver": evidence["approver"],
        "regions": evidence["regions"],
        "contamination_findings": dict(review["contamination_findings"]),
        "review_basis": basis,
    }
    report["gate"] = gate
    report["status"] = "complete"
    write_raw_gate_report(report, Path(report_path))
    return report


__all__ = [
    "poisson_command",
    "preserve_raw_evidence",
    "raw_mesh_metrics",
    "raw_visual_gate",
    "finalize_raw_visual_gate",
    "write_raw_gate_report",
]
