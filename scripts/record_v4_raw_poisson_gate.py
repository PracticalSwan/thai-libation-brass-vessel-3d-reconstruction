"""Record the strict, versioned V4 raw-Poisson acceptance evidence.

The script never edits a mesh.  It re-hashes the dense input and raw Poisson
bytes, recomputes connected-component fractions, verifies all eight technical
preview hashes, and requires an explicit visual-inspection statement for the
no-major-hole condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v4_config import write_json
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _preview_evidence(path: Path, *, dense_report_path: Path) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    views = payload.get("views") if isinstance(payload, dict) else None
    if payload.get("schema_version") != 2 or not isinstance(views, dict) or set(views) != set(ANATOMY_KEYS):
        raise ValueError("raw Poisson preview evidence must contain exactly the eight anatomy keys")
    result: dict[str, dict[str, object]] = {}
    paths: set[str] = set()
    hashes: set[str] = set()
    expected_basis_sha = _sha256(dense_report_path)
    for key in ANATOMY_KEYS:
        item = views[key]
        if not isinstance(item, dict):
            raise ValueError(f"raw Poisson preview evidence is not an object: {key}")
        target = Path(str(item.get("path", "")))
        expected = str(item.get("sha256", "")).lower()
        crop = item.get("region_crop")
        if (
            not target.is_file()
            or len(expected) != 64
            or _sha256(target) != expected
            or not isinstance(crop, dict)
            or crop.get("region_name") != key
            or crop.get("projection_scope") != "region_only_face_subset"
            or crop.get("whole_object_projection") is not False
            or crop.get("basis_sha256") != expected_basis_sha
            or int(crop.get("face_count", 0)) <= 0
            or int(crop.get("vertex_count", 0)) <= 0
        ):
            raise ValueError(f"raw Poisson preview hash/path is invalid: {key}")
        paths.add(str(target.resolve()))
        hashes.add(expected)
        result[key] = dict(item)
    if len(paths) != len(ANATOMY_KEYS) or len(hashes) != len(ANATOMY_KEYS):
        raise ValueError("raw Poisson anatomy previews must be eight distinct hash-bound files")
    region_evidence = payload.get("anatomy_region_evidence")
    if (
        not isinstance(region_evidence, dict)
        or region_evidence.get("schema_version") != 2
        or region_evidence.get("status") != "passed"
        or region_evidence.get("passed") is not True
        or region_evidence.get("no_major_vessel_scale_holes") is not True
        or set(region_evidence.get("regions", {})) != set(ANATOMY_KEYS)
        or region_evidence.get("finial_shape", {}).get("resolved_narrow_top_element") is not True
    ):
        raise ValueError("raw Poisson anatomy region measurements are missing or failed")
    if payload.get("basis_source") != str(dense_report_path.resolve()) or payload.get("basis_sha256") != expected_basis_sha:
        raise ValueError("raw Poisson anatomy previews are not bound to the accepted dense vertical basis")
    return result, region_evidence


def record_gate(
    *,
    mesh_path: Path,
    fused_path: Path,
    dense_report_path: Path,
    preview_evidence_path: Path,
    output_path: Path,
    parameters: dict[str, object],
    visual_note: str,
) -> dict[str, object]:
    if not visual_note.strip():
        raise ValueError("an explicit raw-mesh visual inspection note is required")
    mesh = raw_mesh_metrics(mesh_path)
    component_gate = raw_component_gate(mesh)
    if not component_gate["passed"]:
        raise ValueError("raw Poisson component gate failed: " + json.dumps(component_gate, sort_keys=True))
    dense_report = json.loads(dense_report_path.read_text(encoding="utf-8"))
    dense_lineage = dense_report.get("sparse_lineage", {})
    preview_evidence, raw_anatomy = _preview_evidence(
        preview_evidence_path,
        dense_report_path=dense_report_path,
    )
    fused_sha = _sha256(fused_path)
    if str(dense_report.get("fused_sha256", "")).lower() != fused_sha:
        raise ValueError("raw Poisson dense input hash does not match the accepted post-fusion report")
    dense_anatomy = dense_report.get("contamination", {}).get("anatomy", {})
    dense_postfusion_gate = dense_report.get("postfusion_evidence_gate", {})
    if dense_postfusion_gate.get("passed") is not True or dense_anatomy.get("passed") is not True:
        raise ValueError("raw Poisson promotion requires a passing region-specific dense anatomy gate")
    report = {
        "schema_version": 1,
        "status": "accepted",
        "algorithm": "COLMAP Poisson",
        "parameters": dict(parameters),
        "dense_input": {
            "path": str(fused_path.resolve()),
            "sha256": fused_sha,
            "postfusion_report": str(dense_report_path.resolve()),
            "postfusion_report_sha256": _sha256(dense_report_path),
            "postfusion_gate": dense_report.get("postfusion_evidence_gate", {}),
        },
        "sparse_lineage": dense_lineage,
        "raw_mesh": {
            "path": str(mesh_path.resolve()),
            "sha256": mesh["source_sha256"],
            "metrics": mesh,
            "component_gate": component_gate,
        },
        "anatomy_inspection": {
            "status": "passed" if raw_anatomy.get("passed") is True else "failed",
            "preview_evidence": preview_evidence,
            "region_evidence": raw_anatomy,
            "no_major_vessel_scale_holes": bool(raw_anatomy.get("no_major_vessel_scale_holes") is True),
            "inspection_note": visual_note.strip(),
            "inspection_scope": "eight hash-bound region-only technical projections of the unmodified raw Poisson surface",
        },
    }
    write_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path)
    parser.add_argument("fused", type=Path)
    parser.add_argument("dense_report", type=Path)
    parser.add_argument("preview_evidence", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--point-weight", type=float, default=0.0)
    parser.add_argument("--depth", type=int, default=13)
    parser.add_argument("--trim", type=float, default=5.0)
    parser.add_argument("--visual-note", required=True)
    args = parser.parse_args()
    report = record_gate(
        mesh_path=args.mesh.resolve(),
        fused_path=args.fused.resolve(),
        dense_report_path=args.dense_report.resolve(),
        preview_evidence_path=args.preview_evidence.resolve(),
        output_path=args.output.resolve(),
        parameters={
            "point_weight": args.point_weight,
            "depth": args.depth,
            "trim": args.trim,
            "color": True,
        },
        visual_note=args.visual_note,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
