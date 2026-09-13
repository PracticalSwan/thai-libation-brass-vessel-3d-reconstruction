from __future__ import annotations

import numpy as np
import pytest

from v4_matching import (
    PhaseFrame,
    build_pair_schedule,
    circular_distance,
    estimate_circular_phase_offset,
    feature_cache_identity,
    filter_feature_keypoints,
    normalize_phase,
    schedule_hash,
)


def _ring(name: str, count: int, *, start: float = 0.0):
    return [PhaseFrame(f"{name}_{i}.jpg", name, normalize_phase(start + i / count)) for i in range(count)]


def test_circular_schedule_is_deterministic_and_closes_each_ring():
    rings = {"g8": _ring("g8", 4), "g9": _ring("g9", 3)}
    first = build_pair_schedule(rings, same_ring_neighbors=1, wider_phase_neighbors=0, cross_ring_neighbors=0)
    second = build_pair_schedule(rings, same_ring_neighbors=1, wider_phase_neighbors=0, cross_ring_neighbors=0)
    assert first == second
    assert ("g8_0.jpg", "g8_3.jpg") in first or ("g8_3.jpg", "g8_0.jpg") in first
    assert len(first) == len(set(first))
    assert all(left != right and left < right for left, right in first)
    assert schedule_hash(first) == schedule_hash(second)


def test_unequal_ring_counts_use_nearest_phase_not_same_index():
    rings = {"low": _ring("low", 4), "high": _ring("high", 3)}
    pairs = build_pair_schedule(rings, same_ring_neighbors=1, wider_phase_neighbors=0, cross_ring_neighbors=0)
    # low phase .75 is nearest high phase 2/3, whereas same-index pairing
    # would incorrectly select high phase 0.
    assert ("high_2.jpg", "low_3.jpg") in pairs
    assert ("high_0.jpg", "low_3.jpg") not in pairs


def test_cross_ring_offset_and_missing_phase_bridging_are_stable():
    low = _ring("low", 4)
    high = [PhaseFrame("high_a.jpg", "high", 0.25), PhaseFrame("high_b.jpg", "high", 0.5), PhaseFrame("high_c.jpg", "high", 0.75)]
    pairs = build_pair_schedule({"low": low, "high": high}, same_ring_neighbors=1, wider_phase_neighbors=0, cross_ring_neighbors=0, cross_ring_offsets={("low", "high"): 0.25})
    assert ("high_a.jpg", "low_0.jpg") in pairs
    assert len(pairs) == len(set(pairs))


def test_offset_selection_reports_ambiguity_and_identity_includes_mask_sha():
    selected = estimate_circular_phase_offset({0.0: 10.0, 0.25: 7.0, 0.5: 2.0}, minimum_margin=2.0)
    assert selected["offset_01"] == 0.0
    assert selected["ambiguous"] is False
    ambiguous = estimate_circular_phase_offset({0.0: 10.0, 0.25: 9.5}, minimum_margin=1.0)
    assert ambiguous["ambiguous"] is True
    first = feature_cache_identity(source_sha256="a" * 64, feature_mask_sha256="b" * 64, model_identity={"model": "aliked-n16rot", "size": 2048})
    second = feature_cache_identity(source_sha256="a" * 64, feature_mask_sha256="c" * 64, model_identity={"model": "aliked-n16rot", "size": 2048})
    assert first != second


def test_feature_filter_uses_original_coordinate_frame():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 255
    points = np.asarray([[2.1, 2.1], [5.5, 5.5], [9.9, 9.9]], dtype=np.float32)
    filtered, indices = filter_feature_keypoints(points, mask)
    assert indices.tolist() == [0, 1]
    np.testing.assert_allclose(filtered, points[:2])
    assert circular_distance(0.99, 0.01) == pytest.approx(0.02)
