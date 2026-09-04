from pathlib import Path

import numpy as np
import pytest

from reconstruction_masks import (
    MaskRecord,
    cleanup_reconstruction_mask,
    load_mask_manifest,
    restore_binary_mask,
    sha256_file,
    validate_output_directories,
    verify_reconstruction_checkpoint,
    write_mask_manifest,
)


def test_cleanup_keeps_central_vessel_and_removes_detached_border_blob():
    mask = np.zeros((100, 80), dtype=np.uint8)
    mask[20:85, 25:55] = 255
    mask[5:25, 0:10] = 255

    cleaned = cleanup_reconstruction_mask(mask)

    assert cleaned.dtype == np.uint8
    assert cleaned[50, 40] == 255
    assert cleaned[10, 5] == 0
    assert set(np.unique(cleaned).tolist()) <= {0, 255}


def test_cleanup_preserves_hole_inside_anchor_component():
    mask = np.zeros((100, 80), dtype=np.uint8)
    mask[20:85, 20:60] = 255
    mask[40:55, 30:50] = 0

    cleaned = cleanup_reconstruction_mask(mask)

    assert cleaned[45, 40] == 0
    assert cleaned[30, 40] == 255


def test_cleanup_keeps_single_component_even_when_it_touches_border():
    mask = np.zeros((100, 80), dtype=np.uint8)
    mask[15:95, 0:55] = 255

    cleaned = cleanup_reconstruction_mask(mask)

    assert np.array_equal(cleaned, mask)


def test_restore_binary_mask_is_source_size_and_binary():
    small = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    restored = restore_binary_mask(small, (6, 8))

    assert restored.shape == (8, 6)
    assert restored.dtype == np.uint8
    assert set(np.unique(restored).tolist()) <= {0, 255}


def test_checkpoint_provenance_rejects_non_frozen_model_or_manifest():
    checkpoint = {
        "model_name": "SmallSegCNN",
        "random_initialization": True,
        "pretrained_weights": False,
        "manifest_sha256": "frozen",
        "config": {
            "threshold": 0.5,
            "input_height": 384,
            "input_width": 288,
        },
    }

    verify_reconstruction_checkpoint(checkpoint, "frozen")

    bad = dict(checkpoint)
    bad["pretrained_weights"] = True
    with pytest.raises(ValueError):
        verify_reconstruction_checkpoint(bad, "frozen")

    with pytest.raises(ValueError):
        verify_reconstruction_checkpoint(checkpoint, "different")

    bad_threshold = dict(checkpoint)
    bad_threshold["config"] = dict(checkpoint["config"], threshold=0.6)
    with pytest.raises(ValueError):
        verify_reconstruction_checkpoint(bad_threshold, "frozen")


def test_output_directories_cannot_overlap_raw_or_selected(tmp_path: Path):
    raw = tmp_path / "raw"
    selected = tmp_path / "selected"
    allowed_raw = tmp_path / "analysis" / "raw_predictions"
    allowed_clean = tmp_path / "analysis" / "reconstruction_masks"
    raw.mkdir()
    selected.mkdir()

    validate_output_directories(allowed_raw, allowed_clean, raw, selected)

    with pytest.raises(ValueError):
        validate_output_directories(selected / "predictions", allowed_clean, raw, selected)
    with pytest.raises(ValueError):
        validate_output_directories(allowed_raw, raw / "masks", raw, selected)


def test_mask_manifest_roundtrip_validates_hashes_and_binary_geometry(tmp_path: Path):
    root = tmp_path
    raw_dir = root / "analysis" / "ml" / "full_predictions"
    clean_dir = root / "analysis" / "ml" / "reconstruction_masks"
    raw_dir.mkdir(parents=True)
    clean_dir.mkdir(parents=True)
    raw_path = raw_dir / "001_frame_pred.png"
    clean_path = clean_dir / "001_frame_mask.png"
    import cv2

    mask = np.zeros((8, 6), dtype=np.uint8)
    mask[2:7, 1:5] = 255
    assert cv2.imwrite(str(raw_path), mask)
    assert cv2.imwrite(str(clean_path), mask)
    record = MaskRecord(
        selected_index=1,
        filename="frame.jpg",
        source_width=6,
        source_height=8,
        raw_prediction_path=raw_path,
        raw_prediction_sha256=sha256_file(raw_path),
        reconstruction_mask_path=clean_path,
        reconstruction_mask_sha256=sha256_file(clean_path),
        raw_foreground_fraction=float(np.count_nonzero(mask)) / mask.size,
        reconstruction_foreground_fraction=float(np.count_nonzero(mask)) / mask.size,
    )
    manifest = root / "analysis" / "reports" / "reconstruction_mask_manifest.csv"

    write_mask_manifest(manifest, (record,), project_root=root)
    loaded = load_mask_manifest(manifest, project_root=root, expected_count=1)

    assert loaded == (record,)

    clean_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        load_mask_manifest(manifest, project_root=root, expected_count=1)
