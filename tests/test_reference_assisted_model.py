"""Focused contracts for the reference-assisted pre-clean rebuild."""
from pathlib import Path

import cv2
import numpy as np
from reference_assisted_model import aggregate_profiles, estimate_brass_color, model_profiles, normalized_profile
from run_reference_assisted_model import _normalized_silhouette


def test_normalized_profile_recovers_rectangular_width_ratio():
    mask = np.zeros((100, 80), dtype=np.uint8)
    mask[20:80, 25:55] = 1
    profile = normalized_profile(mask, samples=32)
    assert abs(float(profile["aspect"]) - 0.5) < 0.02
    assert np.allclose(np.asarray(profile["half_width"]), 0.25, atol=0.02)


def test_aggregate_profiles_uses_robust_median():
    y = np.linspace(0, 1, 16)
    profiles = [
        {"y": y, "half_width": np.full(16, 0.20), "aspect": 0.40},
        {"y": y, "half_width": np.full(16, 0.21), "aspect": 0.42},
        {"y": y, "half_width": np.full(16, 0.80), "aspect": 1.60},
    ]
    aggregate = aggregate_profiles(profiles)
    assert abs(float(aggregate["aspect"]) - 0.42) < 0.01


def test_model_profiles_scale_radially_but_preserve_vertical_contract():
    narrow = model_profiles(0.50)
    wide = model_profiles(0.62)
    assert float(wide["radial_scale"]) > float(narrow["radial_scale"])
    assert wide["vessel"][5][1] == narrow["vessel"][5][1]
    assert wide["vessel"][5][0] > narrow["vessel"][5][0]


def test_photo_color_estimate_reads_only_masked_warm_pixels(tmp_path: Path):
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[:] = (20, 20, 20)
    image[5:35, 5:35] = (10, 150, 230)  # BGR warm gold
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[5:35, 5:35] = 255
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)
    result = estimate_brass_color(image_path, mask_path)
    r, g, b = result["median_srgb"]
    assert r > g > b


def test_silhouette_normalization_preserves_aspect():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 30:70] = 1
    normalized = _normalized_silhouette(mask, size=200)
    ys, xs = np.nonzero(normalized)
    height = ys.max() - ys.min() + 1
    width = xs.max() - xs.min() + 1
    assert abs((width / height) - 0.5) < 0.03
