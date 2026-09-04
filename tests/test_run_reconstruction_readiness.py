import json
from pathlib import Path

import pytest

from run_reconstruction_readiness import build_step9_summary, load_chosen_mode


def test_build_step9_summary_combines_required_stage_evidence():
    mask = {
        "mask_count": 288,
        "segmentation_manifest_sha256": "abc",
        "checkpoint_sha256": "def",
    }
    benchmark = {
        "chosen_mode": "reconstruction_mask",
        "modes": {"unmasked": {}, "reconstruction_mask": {"qualified": True}},
    }
    connectivity = {
        "chosen_mode": "reconstruction_mask",
        "adjacent_edge_count": 287,
        "strong_adjacent_count": 280,
        "weak_adjacent_count": 7,
        "included_count": 284,
        "excluded_count": 4,
    }
    camera = {
        "record_count": 288,
        "unique_signature_count": 1,
        "camera_group_recommendation": "shared_intrinsics_single_camera_signature",
    }

    summary = build_step9_summary(mask, benchmark, connectivity, camera)

    assert summary["reconstruction_started"] is False
    assert summary["mask_count"] == 288
    assert summary["chosen_feature_mode"] == "reconstruction_mask"
    assert summary["connectivity"]["adjacent_edge_count"] == 287
    assert summary["camera"]["unique_signature_count"] == 1


def test_build_step9_summary_rejects_inconsistent_chosen_mode():
    mask = {"mask_count": 288, "segmentation_manifest_sha256": "abc", "checkpoint_sha256": "def"}
    benchmark = {"chosen_mode": "raw_cnn", "modes": {"unmasked": {}, "raw_cnn": {}}}
    connectivity = {
        "chosen_mode": "unmasked",
        "adjacent_edge_count": 287,
        "strong_adjacent_count": 287,
        "weak_adjacent_count": 0,
        "included_count": 288,
        "excluded_count": 0,
    }
    camera = {
        "record_count": 288,
        "unique_signature_count": 1,
        "camera_group_recommendation": "shared_intrinsics_single_camera_signature",
    }

    with pytest.raises(ValueError):
        build_step9_summary(mask, benchmark, connectivity, camera)


def test_load_chosen_mode_rejects_unknown_value(tmp_path: Path):
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps({"chosen_mode": "mystery"}), encoding="utf-8")

    with pytest.raises(ValueError):
        load_chosen_mode(path)
