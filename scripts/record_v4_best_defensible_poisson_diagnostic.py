"""Record diagnostic-only Poisson evidence from a best-defensible dense cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v4_config import sha256_file, write_json
from v4_mesh import raw_mesh_metrics
from v4_repair import raw_component_gate


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def record(*, dense_selection_path: Path, mesh_path: Path, anatomy_path: Path, output_path: Path) -> dict[str, Any]:
    dense_selection = _read_object(dense_selection_path.resolve())
    selected = dense_selection.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("dense selection has no selected candidate")
    if selected.get("strict_gate_passed") is True:
        raise ValueError("diagnostic-only Poisson record must not consume a strict dense pass")
    mesh_path = mesh_path.resolve()
    anatomy_path = anatomy_path.resolve()
    if not mesh_path.is_file() or not anatomy_path.is_file():
        raise FileNotFoundError("diagnostic Poisson mesh or anatomy evidence is missing")
    mesh = raw_mesh_metrics(mesh_path)
    component_gate = raw_component_gate(mesh)
    anatomy = _read_object(anatomy_path)
    if anatomy.get("mesh_sha256") != sha256_file(mesh_path):
        raise ValueError("raw Poisson anatomy evidence is not bound to the mesh bytes")
    region_evidence = anatomy.get("anatomy_region_evidence")
    if not isinstance(region_evidence, dict):
        raise ValueError("raw Poisson anatomy evidence is missing region measurements")
    report = {
        "schema_version": 1,
        "status": "diagnostic_only",
        "promotion_allowed": False,
        "dense_selection": {
            "path": str(dense_selection_path.resolve()),
            "sha256": sha256_file(dense_selection_path),
            "status": dense_selection.get("status"),
            "selected_fused_path": selected.get("fused_path"),
            "selected_fused_sha256": selected.get("fused_sha256"),
            "strict_gate_passed": selected.get("strict_gate_passed"),
            "strict_failure_reasons": selected.get("strict_failure_reasons", []),
            "sparse_lineage": selected.get("sparse_lineage"),
        },
        "poisson": {
            "algorithm": "COLMAP Poisson",
            "mesh_path": str(mesh_path),
            "mesh_sha256": mesh["source_sha256"],
            "metrics": mesh,
            "component_gate": component_gate,
        },
        "anatomy_evidence": {
            "path": str(anatomy_path),
            "sha256": sha256_file(anatomy_path),
            "status": anatomy.get("status"),
            "passed": region_evidence.get("passed") is True,
            "failures": list(region_evidence.get("failures", [])),
            "finial_shape": region_evidence.get("finial_shape", {}),
            "no_major_vessel_scale_holes": region_evidence.get("no_major_vessel_scale_holes") is True,
            "region_measurements": region_evidence.get("regions", {}),
        },
        "blocking_reasons": [
            "dense_strict_postfusion_gate_failed",
            "raw_anatomy_region_gate_failed",
            "raw_mesh_promotion_not_authorized",
        ],
    }
    write_json(output_path.resolve(), report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-selection", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--anatomy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = record(
        dense_selection_path=args.dense_selection,
        mesh_path=args.mesh,
        anatomy_path=args.anatomy,
        output_path=args.output,
    )
    print(json.dumps({"status": report["status"], "mesh_sha256": report["poisson"]["mesh_sha256"], "component_gate": report["poisson"]["component_gate"], "anatomy": report["anatomy_evidence"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
