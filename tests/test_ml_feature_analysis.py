import cv2
import numpy as np

from geometry_detection import ImageScale, SiftFeatures
from ml_feature_analysis import classify_keypoints_by_mask


def test_keypoints_map_through_explicit_scale_metadata():
    analysis = np.zeros((50, 50, 3), dtype=np.uint8)
    keypoints = (
        cv2.KeyPoint(10.0, 10.0, 1.0),
        cv2.KeyPoint(40.0, 40.0, 1.0),
    )
    features = SiftFeatures(
        analysis_image=analysis,
        keypoints=keypoints,
        descriptors=np.zeros((2, 128), dtype=np.float32),
        scale=ImageScale(
            original_size=(100, 100),
            analysis_size=(50, 50),
            scale_x_to_original=2.0,
            scale_y_to_original=2.0,
        ),
        status="ok",
    )
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[15:30, 15:30] = 255
    result = classify_keypoints_by_mask(features, mask)
    assert result.vessel_indices == (0,)
    assert result.background_indices == (1,)
    assert result.total_keypoints == 2
    assert result.vessel_feature_fraction == 0.5


def test_mask_size_mismatch_is_rejected():
    features = SiftFeatures(
        analysis_image=np.zeros((10, 10, 3), dtype=np.uint8),
        keypoints=(),
        descriptors=None,
        scale=ImageScale((20, 20), (10, 10), 2.0, 2.0),
        status="descriptors_unavailable",
    )
    try:
        classify_keypoints_by_mask(features, np.zeros((19, 20), dtype=np.uint8))
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("mismatched mask size should fail")
