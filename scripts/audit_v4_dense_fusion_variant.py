"""Audit one bounded same-map StereoFusion variant without rerunning PatchMatch."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v4_config import CAPTURE_V4_ROOT, RECONSTRUCTION_V4_ROOT, sha256_file, write_json
from v4_dense import load_accepted_sparse_lineage, postfusion_evidence_gate
from v4_postfusion import (
    build_postfusion_evidence,
    load_ring_by_name,
    summarize_g8_g9_negative_evidence,
)


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _workspace_from_run(run: dict[str, Any]) -> Path:
    args = run.get("workspace_setup", {}).get("undistorter", {}).get("args", [])
    if not isinstance(args, list) or "--output_path" not in args:
        raise ValueError("dense run does not record its undistorted workspace")
    index = args.index("--output_path")
    return Path(str(args[index + 1])).resolve()


def _image_names(model_path: Path) -> list[str]:
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(model_path))
    names = sorted(Path(reconstruction.image(int(image_id)).name).name for image_id in reconstruction.reg_image_ids())
    if len(names) != 372 or len(names) != len(set(names)):
        raise ValueError(f"fusion variant source model must contain exactly 372 unique views, found {len(names)}")
    return names


def audit_variant(
    *,
    base_run_path: Path,
    fused_path: Path,
    tag: str,
    min_num_pixels: int,
    isolation_records: Path,
    postfusion_output: Path,
    run_output: Path,
) -> dict[str, Any]:
    if min_num_pixels < 1:
        raise ValueError("min_num_pixels must be positive")
    base = _read_object(base_run_path)
    fused_path = fused_path.resolve()
    if not fused_path.is_file():
        raise FileNotFoundError(f"fusion variant is missing: {fused_path}")
    workspace = _workspace_from_run(base)
    sparse_model_path = Path(str(base.get("accepted_sparse_model", ""))).resolve()
    sparse_lineage = load_accepted_sparse_lineage(Path(str(base["accepted_sparse_report"])).resolve())
    image_names = _image_names(sparse_model_path)
    ring_by_name = load_ring_by_name(isolation_records.resolve())
    tile_configs = [Path(str(value)).resolve() for value in base.get("tile_configs", [])]
    if len(tile_configs) != 1 or not tile_configs[0].is_file():
        raise ValueError("fusion variant must reuse the single audited production tile config")
    evidence = build_postfusion_evidence(
        fused_path,
        workspace_root=workspace,
        sparse_model_path=workspace / "sparse",
        mask_dir=workspace / "masks",
        image_names=image_names,
        ring_by_name=ring_by_name,
        tile_config_paths=tile_configs,
        sparse_lineage=sparse_lineage,
        negative_evidence=summarize_g8_g9_negative_evidence(
            CAPTURE_V4_ROOT / "manifests" / "source_manifest.csv",
            isolation_records.resolve(),
        ),
        preview_dir=RECONSTRUCTION_V4_ROOT / "previews" / f"dense_{tag}_semantic",
        max_sources=6,
        chunk_size=None,
    )
    gate = postfusion_evidence_gate(
        evidence,
        fused_path=fused_path,
        expected_sparse_model_sha256=sparse_lineage["accepted_sparse_model_sha256"],
    )
    evidence["postfusion_evidence_gate"] = gate
    evidence["fusion_variant"] = {
        "tag": tag,
        "min_num_pixels": min_num_pixels,
        "base_run_report": str(base_run_path.resolve()),
        "base_run_report_sha256": sha256_file(base_run_path),
        "sparse_model_sha256": sparse_lineage["accepted_sparse_model_sha256"],
        "tile_config_sha256": sha256_file(tile_configs[0]),
    }
    write_json(postfusion_output.resolve(), evidence)
    variant = copy.deepcopy(base)
    variant.update(
        {
            "schema_version": 1,
            "tag": tag,
            "status": "dense_evidence_passed" if gate["passed"] else "dense_best_defensible",
            "fused_path": str(fused_path),
            "fused_sha256": sha256_file(fused_path),
            "postfusion_report": str(postfusion_output.resolve()),
            "postfusion_gate": gate,
            "fusion_variant": evidence["fusion_variant"],
            "patch_match_reused": True,
            "patch_match_restart": False,
        }
    )
    write_json(run_output.resolve(), variant)
    return variant


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run-report", type=Path, required=True)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--min-num-pixels", type=int, required=True)
    parser.add_argument("--isolation-records", type=Path, default=RECONSTRUCTION_V4_ROOT / "work" / "isolation_records.json")
    parser.add_argument("--postfusion-output", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_variant(
        base_run_path=args.base_run_report.resolve(),
        fused_path=args.fused.resolve(),
        tag=args.tag,
        min_num_pixels=args.min_num_pixels,
        isolation_records=args.isolation_records.resolve(),
        postfusion_output=args.postfusion_output.resolve(),
        run_output=args.run_output.resolve(),
    )
    print(json.dumps({"status": result["status"], "postfusion_gate": result["postfusion_gate"], "fused_sha256": result["fused_sha256"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
