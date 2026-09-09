"""Restartable orchestration for the CV-constrained Final V2 vessel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from dataclasses import replace
from typing import Any

from final_model_io import V2_RELATIVE_ROOT, required_reports_ready


PROJECT_ROOT = Path(__file__).resolve().parent
V2_ROOT = PROJECT_ROOT / V2_RELATIVE_ROOT
BLENDER_SCRIPT = PROJECT_ROOT / "build_final_model_blender.py"
DEFAULT_BLENDER = Path(
    "C:/Program Files/Blender Foundation/Blender 5.2/blender.exe"
)

STAGE_ORDER = (
    "analyze",
    "cv-fit",
    "base",
    "geometry-validate",
    "ornament",
    "cleanup",
    "uv-bake",
    "lookdev",
    "final-validate",
    "export",
)

PRE_EXPORT_STAGE = "final-validate"
USER_EXPORT_APPROVAL_REPORT = "user_export_approval.json"

PREREQUISITE_REPORTS = {
    "cv-fit": "final_reference_evidence.json",
    "base": "final_cv_fit.json",
    "geometry-validate": "final_cv_fit.json",
    "ornament": "base_geometry_report.json",
    "cleanup": "ornament_build_report.json",
    "uv-bake": "cleanup_report.json",
    "lookdev": "uv_bake_report.json",
    "final-validate": "lookdev_report.json",
    "export": "final_validation_report.json",
}

CV_CONTINUITY_REPORTS = (
    "registered_view_coverage_report.json",
    "surface_evidence_coverage.json",
)

ADDITIONAL_PREREQUISITE_REPORTS = {
    # Every downstream Blender stage preserves Plan 2's proof that the accepted
    # geometry generalizes beyond the 16 canonical fit views and records which
    # surface regions are direct CV evidence versus bounded inference.
    stage: CV_CONTINUITY_REPORTS
    for stage in ("ornament", "cleanup", "uv-bake", "lookdev", "final-validate", "export")
}


def require_accepted_report(path: Path) -> dict[str, Any]:
    """Load one stage gate and require an explicit accepted=true value."""

    if not path.is_file():
        raise ValueError(f"missing required report: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable required report: {path}") from exc
    if payload.get("accepted") is not True:
        raise ValueError(f"required report is not accepted: {path.name}")
    return payload


def require_user_export_approval(v2_root: Path) -> dict[str, Any]:
    """Require an explicit user approval sidecar before any export operation."""

    path = v2_root / "reports" / USER_EXPORT_APPROVAL_REPORT
    if not path.is_file():
        raise ValueError(
            "export is blocked until the user approves the full Blender model; "
            f"missing {USER_EXPORT_APPROVAL_REPORT}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("approved") is not True:
        raise ValueError("export is blocked because user approval is not true")
    return payload


def discover_blender() -> Path:
    """Resolve the verified Blender 5.2 executable without modifying PATH."""

    configured = os.environ.get("CSX4213_BLENDER")
    candidates = (Path(configured) if configured else None, DEFAULT_BLENDER)
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Blender 5.2 executable not found; set CSX4213_BLENDER to blender.exe"
    )


def prepare_gltf_basecolor_texture(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Bake the master shader's simple color mix into a glTF-compatible texture."""

    import cv2
    import numpy as np

    root = project_root.resolve()
    texture_dir = root / V2_RELATIVE_ROOT / "textures" / "final"
    source = texture_dir / "T_ThaiLibation_BaseColor.png"
    output = texture_dir / "T_ThaiLibation_BaseColor_GLTF.png"
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot read final base-color texture: {source}")
    if image.ndim != 3 or image.shape[2] not in {3, 4}:
        raise ValueError("final base-color texture must be RGB or RGBA")

    bgr = image[..., :3].astype(np.float32) / 255.0
    rgb = bgr[..., ::-1]
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    # Exact master material constants from MAT_FINAL_PolishedThaiBrass:
    # MixRGB(MIX), fac=0.26, Color1=(0.83, 0.49, 0.105), Color2=photo texture.
    constant = np.array([0.83, 0.49, 0.105], dtype=np.float32)
    mixed = 0.74 * constant + 0.26 * linear
    srgb = np.where(
        mixed <= 0.0031308,
        mixed * 12.92,
        1.055 * np.power(mixed, 1.0 / 2.4) - 0.055,
    )
    baked_bgr = np.clip(np.rint(srgb[..., ::-1] * 255.0), 0, 255).astype(np.uint8)
    if image.shape[2] == 4:
        baked = np.dstack([baked_bgr, image[..., 3]])
    else:
        baked = baked_bgr
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), baked):
        raise ValueError(f"failed to write glTF-compatible base-color texture: {output}")
    return {
        "source": source.relative_to(root).as_posix(),
        "output": output.relative_to(root).as_posix(),
        "size_bytes": output.stat().st_size,
    }


def run_blender_stage(stage: str, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Execute one Blender stage in a clean background process with a durable log."""

    if not BLENDER_SCRIPT.is_file():
        raise FileNotFoundError(f"missing Blender build script: {BLENDER_SCRIPT}")
    v2_root = project_root.resolve() / V2_RELATIVE_ROOT
    log_dir = v2_root / "work" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage}.log"
    command = (
        str(discover_blender()),
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "2",
        "--python",
        str(BLENDER_SCRIPT),
        "--",
        stage,
        str(v2_root),
    )
    completed = subprocess.run(
        command,
        cwd=project_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8", newline="\n")
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-40:])
        raise RuntimeError(
            f"Blender stage {stage!r} failed with exit code {completed.returncode}:\n{tail}"
        )
    return {
        "stage": stage,
        "command": list(command),
        "log_path": log_path.relative_to(project_root).as_posix(),
        "returncode": completed.returncode,
    }


def _run_cv_fit(project_root: Path) -> dict[str, Any]:
    from final_cv_model_fit import fit_vessel_model, write_fit_outputs, _normalize_visual_review, acceptance_decision

    require_accepted_report(
        project_root / V2_RELATIVE_ROOT / "reports" / "final_reference_evidence.json"
    )
    review_path = (
        project_root / V2_RELATIVE_ROOT / "reports" / "cv_visual_review.json"
    )
    # Bootstrap diagnostics before asking for their review. The first candidate is
    # deliberately unaccepted; downstream Blender work remains blocked.
    result = fit_vessel_model(project_root, visual_component_review=None)
    report_path, profiles_path = write_fit_outputs(project_root, result)
    candidate = json.loads(report_path.read_text(encoding="utf-8"))
    if review_path.is_file():
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if (review.get("accepted") is True
                and review.get("candidate_sha256") == candidate["candidate_sha256"]):
            components = _normalize_visual_review(review.get("components"))
            result = replace(result, visual_component_review=components,
                             accepted=acceptance_decision(result.metrics_passed, components))
            report_path, profiles_path = write_fit_outputs(project_root, result)
    if not result.accepted:
        raise ValueError("CV fit diagnostics written; numeric pass and matching cv_visual_review.json are required")
    return {
        "stage": "cv-fit",
        "accepted": result.accepted,
        "metrics_passed": result.metrics_passed,
        "aggregate_metrics": dict(result.aggregate_metrics),
        "report": report_path.relative_to(project_root).as_posix(),
        "profiles": profiles_path.relative_to(project_root).as_posix(),
    }


def _run_texture_projection(project_root: Path) -> dict[str, Any]:
    import final_texture_projection as projection

    for name in ("project_final_textures", "build_final_textures", "run_texture_projection"):
        function = getattr(projection, name, None)
        if callable(function):
            return function(project_root, project_root / V2_RELATIVE_ROOT)
    raise AttributeError(
        "final_texture_projection must expose project_final_textures(project_root, v2_root)"
    )


def _run_geometry_validation(project_root: Path) -> dict[str, Any]:
    """Create the Blender checkpoint, then score it with exact CV cameras."""

    from final_geometry_audit import run_geometry_audit

    v2_root = project_root / V2_RELATIVE_ROOT
    blender_result = run_blender_stage("geometry-validate", project_root)
    audit_result = run_geometry_audit(project_root, v2_root)
    return {
        "stage": "geometry-validate",
        "blender": blender_result,
        "audit": audit_result,
    }


def run_stage(stage: str, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Run one named stage after checking its immediate accepted prerequisite."""

    if stage not in STAGE_ORDER:
        raise ValueError(f"unknown Final V2 stage: {stage}")
    root = project_root.resolve()
    v2_root = root / V2_RELATIVE_ROOT
    report_name = PREREQUISITE_REPORTS.get(stage)
    if report_name is not None:
        require_accepted_report(v2_root / "reports" / report_name)
    for additional_report in ADDITIONAL_PREREQUISITE_REPORTS.get(stage, ()):
        require_accepted_report(v2_root / "reports" / additional_report)
    if stage == "export":
        require_user_export_approval(v2_root)
        prepare_gltf_basecolor_texture(root)

    if stage == "analyze":
        from final_reference_evidence import build_final_reference_evidence

        return build_final_reference_evidence(root)
    if stage == "cv-fit":
        return _run_cv_fit(root)
    if stage == "geometry-validate":
        return _run_geometry_validation(root)
    if stage == "lookdev":
        projection_result = _run_texture_projection(root)
        blender_result = run_blender_stage(stage, root)
        return {
            "stage": stage,
            "projection": projection_result,
            "blender": blender_result,
        }
    result = run_blender_stage(stage, root)
    if stage == "export":
        result["promotion_reports"] = [
            path.relative_to(root).as_posix() for path in required_reports_ready(v2_root)
        ]
    return result


def run_all_through(
    stage: str | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Run the complete ordered pipeline, optionally stopping at one stage."""

    if stage is not None and stage not in STAGE_ORDER:
        raise ValueError(f"unknown Final V2 stop stage: {stage}")
    # Until the user explicitly approves the complete Blender model, the default
    # integrated run is intentionally pre-export and stops after final validation.
    target = PRE_EXPORT_STAGE if stage is None else stage
    stop = STAGE_ORDER.index(target) + 1
    results: list[dict[str, Any]] = []
    for current in STAGE_ORDER[:stop]:
        results.append(run_stage(current, project_root))
    return {
        "completed_stages": list(STAGE_ORDER[:stop]),
        "stage_results": results,
        "accepted": False,
        "user_review_required_before_export": target == PRE_EXPORT_STAGE,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all",) + STAGE_ORDER, default="all")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = (
        run_all_through()
        if args.stage == "all"
        else run_stage(args.stage)
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
