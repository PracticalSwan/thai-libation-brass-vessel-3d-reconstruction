"""Persist an observed dense review decision without auto-approving it."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v4_config import RECONSTRUCTION_V4_ROOT, write_json
from v4_dense import reviewed_evidence_gate


def review_evidence_valid_for_record(evidence: dict[str, object]) -> bool:
    """Return whether the four-view evidence is recordable, even on rejection."""

    checks = evidence.get("checks", {})
    if not isinstance(checks, dict):
        return False
    return bool(
        checks.get("four_semantic_views") is True
        and checks.get("preview_hashes_verified") is True
        and checks.get("preview_views_distinct") is True
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Persist an observed dense review decision without auto-approving it.")
    parser.add_argument(
        "--report",
        type=Path,
        default=RECONSTRUCTION_V4_ROOT / "reports" / "dense_gate.json",
        help="Dense report to annotate; defaults to the canonical V4 report.",
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=RECONSTRUCTION_V4_ROOT / "work" / "dense_visual_review.json",
        help="Hash-anchored visual review JSON to record.",
    )
    parser.add_argument(
        "--reason",
        default="board slab and pedestal-to-board webbing are visibly present; downstream Poisson/Blender work is blocked",
        help="Human-readable rejection reason.",
    )
    args = parser.parse_args()
    report_path = Path(args.report)
    review_path = Path(args.review)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    evidence = reviewed_evidence_gate(review)
    report["visual_gate"] = {
        "status": str(review.get("review_status", "missing")),
        "previews": evidence.get("preview_evidence", {}),
        "reviewer": str(review.get("reviewer", "")),
        "approver": str(review.get("approver", "")),
        "regions": dict(review.get("regions", {})) if isinstance(review.get("regions"), dict) else {},
        "contamination_findings": dict(review.get("contamination_findings", {})) if isinstance(review.get("contamination_findings"), dict) else {},
        "review_basis": list(review.get("review_basis", [])) if isinstance(review.get("review_basis"), list) else [],
        "evidence_gate": evidence,
    }
    report["gate"] = {
        "passed": False,
        "status": "failed_visual_contamination_review",
        "reason": str(args.reason),
        "review_evidence_valid_for_record": review_evidence_valid_for_record(evidence),
        "contamination_findings": report["visual_gate"]["contamination_findings"],
    }
    report["status"] = "complete_rejected_contamination"
    write_json(report_path, report)
    print(json.dumps({"status": report["status"], "gate_passed": False, "reasons": evidence.get("reasons", [])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
