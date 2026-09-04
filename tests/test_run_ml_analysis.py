import numpy as np
import pytest

from run_ml_analysis import (
    _status_for_metrics,
    restore_prediction_mask,
    validate_prediction_output_path,
    verify_checkpoint_provenance,
)


def test_restore_prediction_mask_is_source_size_binary():
    small = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    restored = restore_prediction_mask(small, (6, 8))
    assert restored.shape == (8, 6)
    assert restored.dtype == np.uint8
    assert set(np.unique(restored).tolist()) <= {0, 255}


def test_prediction_output_path_cannot_target_source_directories(tmp_path):
    raw = tmp_path / "raw"
    selected = tmp_path / "selected"
    raw.mkdir()
    selected.mkdir()
    with pytest.raises(ValueError):
        validate_prediction_output_path(selected / "prediction.png", raw, selected)
    allowed = tmp_path / "analysis" / "prediction.png"
    validate_prediction_output_path(allowed, raw, selected)


def test_checkpoint_provenance_requires_frozen_manifest_and_test_indices():
    checkpoint = {
        "model_name": "SmallSegCNN",
        "random_initialization": True,
        "pretrained_weights": False,
        "manifest_sha256": "abc",
        "test_indices": [72, 142, 165, 200, 255, 288],
        "config": {"threshold": 0.5},
    }
    verify_checkpoint_provenance(
        checkpoint, "abc", [72, 142, 165, 200, 255, 288]
    )
    with pytest.raises(ValueError):
        verify_checkpoint_provenance(
            checkpoint, "different", [72, 142, 165, 200, 255, 288]
        )


def test_status_reports_background_false_positive_when_precision_is_low():
    assert (
        _status_for_metrics(dice=0.894, iou=0.809, precision=0.832, recall=0.967)
        == "background_false_positive"
    )
