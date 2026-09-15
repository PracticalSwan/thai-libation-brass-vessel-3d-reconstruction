from __future__ import annotations

import json

import pytest

from scripts.freeze_v4_preblender_handoff import detached_component_assessment
from v4_config import sha256_file
from v4_dense import load_accepted_sparse_lineage
from v4_repair import raw_component_gate, stable_directory_sha256


def test_detached_component_assessment_keeps_fraction_proxy_separate_from_semantic_proof() -> None:
    metrics = {
        "components": {"face_counts": [950, 19, 6], "component_count": 3},
        "face_count": 975,
    }
    gate = raw_component_gate(metrics)

    result = detached_component_assessment(metrics, gate)

    assert result["fraction_proxy_passed"] is True
    assert result["passed"] is False
    assert result["semantic_detachedness_proven"] is False
    assert result["largest_non_dominant_face_count"] == 19


def test_detached_component_assessment_reports_large_second_component() -> None:
    metrics = {
        "components": {"face_counts": [600, 300, 75], "component_count": 3},
        "face_count": 975,
    }
    gate = raw_component_gate(metrics)

    result = detached_component_assessment(metrics, gate)

    assert result["fraction_proxy_passed"] is False
    assert result["largest_non_dominant_face_fraction"] > 0.02


def test_sparse_lineage_loader_rejects_selection_report_hash_drift(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "points3D.bin").write_bytes(b"model")
    selection = tmp_path / "selection.json"
    gate = tmp_path / "gate.json"
    track = tmp_path / "track.json"
    selection.write_text("{}", encoding="utf-8")
    gate.write_text("{}", encoding="utf-8")
    track.write_text("{}", encoding="utf-8")
    report = tmp_path / "sparse.json"
    report.write_text(
        json.dumps(
            {
                "status": "best_defensible_sparse_candidate",
                "best_defensible": True,
                "sparse_gate_passed": False,
                "strict_gate_failures": ["mask_projection_per_view"],
                "source_model": str(model),
                "source_model_sha256": stable_directory_sha256(model),
                "selection_report": str(selection),
                "selection_report_sha256": sha256_file(selection),
                "sparse_gate": str(gate),
                "sparse_gate_sha256": sha256_file(gate),
                "track_provenance": str(track),
                "track_provenance_sha256": sha256_file(track),
            }
        ),
        encoding="utf-8",
    )
    selection.write_text("{\"changed\":true}", encoding="utf-8")

    with pytest.raises(ValueError, match="selection-report hash does not match"):
        load_accepted_sparse_lineage(report)
