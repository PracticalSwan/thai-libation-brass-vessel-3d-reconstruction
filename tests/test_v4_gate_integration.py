from __future__ import annotations

from pathlib import Path

from scripts.record_v4_dense_review import review_evidence_valid_for_record


ROOT = Path(__file__).resolve().parents[1]


def _section(source: str, start: str, end: str) -> str:
    begin = source.index(start)
    finish = source.index(end, begin)
    return source[begin:finish]


def test_dense_runtime_no_longer_fabricates_zero_contamination_or_visual_pass():
    source = (ROOT / "run_v4.py").read_text(encoding="utf-8")
    dense = _section(source, "def run_dense(", "def run_dense_visual(")
    visual = _section(source, "def run_dense_visual(", "def run_mesh(")
    assert 'metrics["board_point_fraction"] = 0.0' not in dense
    assert 'metrics["cloth_point_fraction"] = 0.0' not in dense
    assert "dense_visual_review.json" in visual
    assert "review=review" in visual
    assert "review_basis=[str(value) for value in review.get(\"review_basis\", [])]" in visual
    assert 'contamination_report.get("status") != "passed"' in visual
    assert 'contamination_report.get("source_selection_audit", {}).get("status") != "passed"' in visual
    assert 'contamination_report.get("ring_transition_audit", {}).get("status") != "passed"' in visual
    assert 'contamination.get("status") != "measured"' in visual
    assert 'set(semantic_previews) != {"front", "quarter", "side", "top_oblique"}' in visual
    assert "postfusion_evidence_gate" in visual
    assert "postfusion_report=contamination_report" in visual


def test_dense_mask_pipeline_uses_colmap_fusion_filename_contract():
    dense_source = (ROOT / "v4_dense.py").read_text(encoding="utf-8")
    postfusion_source = (ROOT / "v4_postfusion.py").read_text(encoding="utf-8")
    assert 'target = target_dir / f"{basename}.png"' in dense_source
    assert "mask_path / (image_name + \".png\")" in postfusion_source
    assert 'resolver": "colmap_image_name_plus_png"' in postfusion_source
    assert "legacy_unextended_image_name" in postfusion_source


def test_mesh_runtime_requires_postfusion_evidence_and_external_review():
    source = (ROOT / "run_v4.py").read_text(encoding="utf-8")
    mesh = _section(source, "def run_mesh(", "def run_mesh_visual(")
    visual = source[source.index("def run_mesh_visual(") : source.index("def run(", source.index("def run_mesh_visual("))]
    assert "dense_contamination_gate.json" in mesh
    assert 'contamination_report.get("status") != "passed"' in mesh
    assert 'contamination_report.get("source_selection_audit", {}).get("status") != "passed"' in mesh
    assert 'contamination_report.get("ring_transition_audit", {}).get("status") != "passed"' in mesh
    assert "ring_transition_audit" in mesh
    assert "postfusion_evidence_gate" in mesh
    assert "finalize_raw_visual_gate" in visual
    assert "raw_mesh_visual_review.json" in visual
    assert "regions = {" not in visual


def test_dense_review_record_uses_nested_evidence_checks():
    assert review_evidence_valid_for_record(
        {
            "checks": {
                "four_semantic_views": True,
                "preview_hashes_verified": True,
                "preview_views_distinct": True,
            }
        }
    ) is True
    assert review_evidence_valid_for_record(
        {
            "four_semantic_views": True,
            "preview_hashes_verified": True,
            "preview_views_distinct": True,
        }
    ) is False


def test_dense_visual_review_cannot_use_arbitrary_region_keys():
    from v4_dense import dense_gate

    review = {
        "regions": {
            "bowl_and_interior": True,
            "globe_or_shoulder": True,
            "neck_lid_or_finial": True,
            "pedestal_and_base": True,
            "anything_true": True,
        }
    }
    result = dense_gate({}, smoke={"status": "failed"}, visual_status="pending", review=review)
    assert result["passed"] is False
    assert result["review"]["checks"]["required_regions_confirmed"] is False
