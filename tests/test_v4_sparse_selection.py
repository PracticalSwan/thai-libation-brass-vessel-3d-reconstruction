from __future__ import annotations

from scripts.freeze_v4_best_defensible_sparse import _candidate_process_evidence


def test_failed_zero_residual_ba_is_not_downstream_defensible() -> None:
    evidence = _candidate_process_evidence(
        {
            "bundle_adjustment": {
                "summary": {
                    "num_residuals": 0,
                    "termination_type": "BundleAdjustmentTerminationType.FAILURE",
                }
            },
            "promotion_allowed": False,
            "full_sparse_candidate_eligible": False,
        }
    )

    assert evidence["valid"] is False
    assert "bundle_adjustment_not_successfully_executed" in evidence["reasons"]
    assert "promotion_not_allowed" in evidence["reasons"]
    assert "full_sparse_candidate_not_eligible" in evidence["reasons"]


def test_successful_candidate_process_evidence_can_be_valid() -> None:
    evidence = _candidate_process_evidence(
        {
            "bundle_adjustment": {
                "summary": {
                    "num_residuals": 1234,
                    "termination_type": "BundleAdjustmentTerminationType.CONVERGENCE",
                }
            },
            "promotion_allowed": True,
            "full_sparse_candidate_eligible": True,
        }
    )

    assert evidence["valid"] is True
    assert evidence["reasons"] == []
