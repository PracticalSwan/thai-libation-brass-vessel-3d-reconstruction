"""Run the V4 read-only post-fusion contamination and transition audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v4_config import CAPTURE_V4_ROOT, RECONSTRUCTION_V4_ROOT, sha256_file, write_json
from v4_dense import postfusion_evidence_gate
from v4_postfusion import build_postfusion_evidence, load_ring_by_name, summarize_g8_g9_negative_evidence


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-report", type=Path, default=RECONSTRUCTION_V4_ROOT / "reports" / "dense_gate.json")
    parser.add_argument("--isolation-records", type=Path, default=RECONSTRUCTION_V4_ROOT / "work" / "isolation_records.json")
    parser.add_argument("--output", type=Path, default=RECONSTRUCTION_V4_ROOT / "reports" / "dense_contamination_gate.json")
    args = parser.parse_args()
    dense_report = _read_json(args.dense_report)
    fused_path = Path(str(dense_report.get("fused_path", "")))
    if not fused_path.is_file():
        raise FileNotFoundError(f"fused cloud is missing: {fused_path}")
    attempts = dense_report.get("attempts", [])
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("dense report has no attempts")
    attempt = attempts[-1]
    tag = str(attempt.get("tag", ""))
    scratch_root = Path(str(dense_report.get("dense_scratch_root", "")))
    workspace_root = scratch_root / tag
    if not workspace_root.is_dir():
        raise FileNotFoundError(f"dense workspace is missing: {workspace_root}")
    dense_model_path = workspace_root / "sparse"
    if not dense_model_path.is_dir():
        candidate = workspace_root / "sparse" / "0"
        if candidate.is_dir():
            dense_model_path = candidate
    if not dense_model_path.is_dir():
        raise FileNotFoundError(f"undistorted dense sparse model is missing: {dense_model_path}")
    model_index = _read_json(RECONSTRUCTION_V4_ROOT / "reports" / "sparse_report.json")
    models = model_index.get("models", [])
    best = int(model_index.get("best_model_index", 0))
    image_names = [Path(str(name)).name for name in models[best].get("registered_image_names", [])]
    ring_by_name = load_ring_by_name(args.isolation_records)
    configs = sorted((RECONSTRUCTION_V4_ROOT / "work").glob(f"dense_{tag}_patch-match-v4-tile-*.cfg"))
    if not configs:
        raise FileNotFoundError(f"final tile configs are missing for dense tag {tag}")
    evidence = build_postfusion_evidence(
        fused_path,
        workspace_root=workspace_root,
        sparse_model_path=dense_model_path,
        mask_dir=workspace_root / "masks",
        image_names=image_names,
        ring_by_name=ring_by_name,
        tile_config_paths=configs,
        prior_review_estimate={
            "references_without_cross_ring_source_count": 150,
            "cross_ring_directed_source_count": 954,
        },
        preview_dir=RECONSTRUCTION_V4_ROOT / "previews" / "dense_fused_semantic",
        max_sources=6,
        negative_evidence=summarize_g8_g9_negative_evidence(
            CAPTURE_V4_ROOT / "manifests" / "source_manifest.csv",
            args.isolation_records,
        ),
    )
    evidence["postfusion_evidence_gate"] = postfusion_evidence_gate(evidence, fused_path=fused_path)
    evidence["dense_report_sha256_before_update"] = sha256_file(args.dense_report)
    write_json(args.output, evidence)
    dense_report["metrics"] = dict(dense_report.get("metrics", {}))
    contamination = evidence["contamination"]
    dense_report["metrics"]["contamination_metrics"] = contamination["metrics"]
    dense_report["metrics"]["contamination_findings"] = contamination["findings"]
    dense_report["metrics"]["board_point_fraction"] = contamination["metrics"]["board_point_fraction"]["value"]
    dense_report["metrics"]["cloth_point_fraction"] = contamination["metrics"]["cloth_or_background_point_fraction"]["value"]
    dense_report["metrics"]["contamination_measurement_status"] = contamination["status"]
    dense_report["metrics"]["contamination_measurement_method_version"] = contamination["method_version"]
    dense_report["metrics"]["anatomy_evidence"] = contamination["anatomy"]
    dense_report["contamination_gate_report"] = str(args.output.resolve())
    dense_report["source_selection_audit"] = evidence["source_selection_audit"]
    dense_report["ring_transition_audit"] = evidence["ring_transition_audit"]
    dense_report["semantic_previews"] = {key: item["path"] for key, item in evidence["semantic_previews"].items()}
    dense_report["postfusion_evidence_gate"] = evidence["postfusion_evidence_gate"]
    dense_report["quality_risk"] = {
        "max_sources": 6,
        "bounded_graph_deviation": "explicit manifest-order six-source cap instead of original automatic/default source selection",
        "review_claim_is_not_accepted_without_final_tile_audit": True,
    }
    # Keep the dense gate pending until the explicit visual review record is
    # supplied.  The measurement pass must never silently self-approve it.
    dense_report["gate"] = {
        "passed": False,
        "status": "pending_visual_review",
        "reason": "measured post-fusion evidence is recorded; four-view review is still required",
        "contamination_report": str(args.output.resolve()),
    }
    dense_report["status"] = "complete_pending_visual_gate"
    dense_report["visual_gate"] = {
        "status": "pending",
        "evidence": list(dense_report["semantic_previews"].values()),
    }
    write_json(args.dense_report, dense_report)
    print(json.dumps({"output": str(args.output.resolve()), "dense_report": str(args.dense_report.resolve()), "status": evidence["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
