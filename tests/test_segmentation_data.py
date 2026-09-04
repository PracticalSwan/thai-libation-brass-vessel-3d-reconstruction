from pathlib import Path

import numpy as np
import torch

from analysis_common import load_selected_manifest
from segmentation_data import (
    SegmentationDataset,
    load_segmentation_manifest,
    validate_binary_mask,
)

ROOT = Path(__file__).resolve().parents[1]


def test_real_segmentation_manifest_is_frozen_24_6_6():
    selected = load_selected_manifest(ROOT / "preprocessing/reports/selection_manifest.csv")
    dataset = load_segmentation_manifest(
        ROOT / "ml_dataset/manifest.csv",
        selected,
        ROOT / "preprocessing/pycolmap_input/images",
        verify_source_hashes=False,
    )
    assert len(dataset.records) == 36
    assert len(dataset.split("train")) == 24
    assert len(dataset.split("val")) == 6
    assert len(dataset.split("test")) == 6
    assert [r.selected_index for r in dataset.split("test")] == [72, 142, 165, 200, 255, 288]


def test_validation_dataset_has_expected_tensor_geometry():
    selected = load_selected_manifest(ROOT / "preprocessing/reports/selection_manifest.csv")
    dataset = load_segmentation_manifest(
        ROOT / "ml_dataset/manifest.csv",
        selected,
        ROOT / "preprocessing/pycolmap_input/images",
        verify_source_hashes=False,
    )
    torch.manual_seed(4213)
    item = SegmentationDataset(
        dataset.split("val"), dataset.images_dir, training=False
    )[0]
    assert tuple(item["image"].shape) == (3, 384, 288)
    assert tuple(item["mask"].shape) == (1, 384, 288)
    assert set(torch.unique(item["mask"]).tolist()).issubset({0.0, 1.0})


def test_binary_mask_rejects_non_binary_values(tmp_path):
    import cv2

    path = tmp_path / "bad.png"
    mask = np.zeros((8, 6), dtype=np.uint8)
    mask[2, 3] = 127
    assert cv2.imwrite(str(path), mask)
    try:
        validate_binary_mask(path, (6, 8))
    except ValueError as error:
        assert "only 0 and 255" in str(error)
    else:
        raise AssertionError("non-binary mask should be rejected")
