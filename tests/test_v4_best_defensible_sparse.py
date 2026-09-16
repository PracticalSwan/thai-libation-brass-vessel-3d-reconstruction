from __future__ import annotations

import json
from pathlib import Path

from scripts.freeze_v4_best_defensible_sparse import _candidate_process_evidence


ROOT = Path(__file__).resolve().parents[1]


def test_v50b_zero_residual_failed_ba_is_not_downstream_eligible() -> None:
    payload = json.loads(
        (
            ROOT
            / "reconstruction"
            / "v4"
            / "repair"
            / "sparse_v1"
            / "v50b_solved_center_native_diagnostic.json"
        ).read_text(encoding="utf-8")
    )
    evidence = _candidate_process_evidence(payload)

    assert evidence["valid"] is False
    assert evidence["bundle_adjustment_num_residuals"] == 0
    assert "bundle_adjustment_not_successfully_executed" in evidence["reasons"]
    assert "promotion_not_allowed" in evidence["reasons"]
    assert "full_sparse_candidate_not_eligible" in evidence["reasons"]


def test_successful_promotable_ba_process_evidence_is_eligible() -> None:
    evidence = _candidate_process_evidence(
        {
            "promotion_allowed": True,
            "full_sparse_candidate_eligible": True,
            "bundle_adjustment": {
                "summary": {
                    "num_residuals": 1234,
                    "termination_type": "CONVERGENCE",
                }
            },
        }
    )

    assert evidence["valid"] is True
    assert evidence["reasons"] == []
