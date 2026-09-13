from __future__ import annotations

import numpy as np
import pytest

from v4_isolation import (
    SegmentationCapabilityError,
    build_mvs_image,
    derive_mask_products,
    filter_keypoints,
    keypoint_mask_membership,
    map_keypoints_to_original,
    mask_diagnostics,
    refine_board_leakage,
    segmentation_capability,
)


def test_mask_products_preserve_dimensions_and_thin_support():
    mask = np.zeros((40, 30), dtype=np.uint8)
    mask[8:32, 10:20] = 255
    mask[2:5, 14:16] = 255  # thin finial-like component
    products = derive_mask_products(mask, inward_margin=2, boundary_band=2)
    products.validate((30, 40))
    assert products.full_mask.shape == products.feature_mask.shape == products.mvs_mask.shape
    assert np.all(products.feature_mask <= products.full_mask)
    assert np.all(products.mvs_mask <= products.full_mask)
    assert np.any(products.feature_mask[2:5, 14:16] > 0)


def test_keypoint_filter_excludes_off_mask_boundary_and_preserves_original_coordinates():
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[5:15, 10:20] = 255
    boundary = np.zeros_like(mask)
    boundary[5, 10:20] = 255
    points = np.asarray([[10.2, 5.1], [12.5, 10.5], [2.0, 2.0], [29.0, 19.0], [np.nan, 3.0]], dtype=np.float32)
    keep = keypoint_mask_membership(points, mask, boundary_exclusion=boundary)
    filtered, indices = filter_keypoints(points, mask, boundary_exclusion=boundary)
    assert keep.tolist() == [False, True, False, False, False]
    assert indices.tolist() == [1]
    np.testing.assert_allclose(filtered, points[[1]])


def test_scaled_keypoints_map_back_to_original_full_frame():
    points = np.asarray([[0.0, 0.0], [799.5, 599.5]], dtype=np.float32)
    mapped = map_keypoints_to_original(points, feature_size=(800, 600), source_size=(3072, 4080))
    np.testing.assert_allclose(mapped[1], [3070.08, 4076.6], rtol=1e-5)


def test_mvs_mask_neutralizes_only_outside_support_and_board_refinement_never_adds():
    image = np.full((4, 5, 3), 200, dtype=np.uint8)
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1:3, 1:4] = 255
    result = build_mvs_image(image, mask, outside_value=7)
    assert np.all(result[mask == 0] == 7)
    assert np.all(result[mask > 0] == 200)
    board = np.zeros_like(mask)
    board[1, 1] = 255
    refined = refine_board_leakage(mask, supported_board_pixels=board)
    assert refined[1, 1] == 0
    assert np.all(refined <= mask)
    diagnostics = mask_diagnostics(refined)
    assert diagnostics["foreground_pixels"] == int(np.count_nonzero(refined))


def test_segmentation_capability_reports_missing_official_stack_without_smallseg_fallback():
    capability = segmentation_capability()
    assert capability["smallseg_primary"] is False
    if capability["status"] == "blocked":
        with pytest.raises(SegmentationCapabilityError):
            from v4_isolation import require_segmentation_capability

            require_segmentation_capability()
