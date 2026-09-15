from __future__ import annotations

import pytest

from scripts.run_v4_upper_dense_recovery import (
    classify_upper_recovery,
    connected_smoke_subset,
    same_ring_adjacency,
)


def test_same_ring_adjacency_filters_cross_ring_sources() -> None:
    adjacency = {
        "a": ("b", "x"),
        "b": ("a", "c"),
        "c": ("b", "y"),
        "x": ("a",),
        "y": ("c",),
    }
    ring_by_name = {"a": "top", "b": "top", "c": "top", "x": "lower", "y": "lower"}

    result = same_ring_adjacency(("a", "b", "c"), adjacency, ring_by_name, ring="top")

    assert result == {"a": ("b",), "b": ("a", "c"), "c": ("b",)}


def test_same_ring_adjacency_rejects_isolated_reference() -> None:
    adjacency = {"a": ("x",), "b": ("a",), "x": ("a",)}
    ring_by_name = {"a": "top", "b": "top", "x": "lower"}

    with pytest.raises(ValueError, match="no same-ring source"):
        same_ring_adjacency(("a", "b"), adjacency, ring_by_name, ring="top")


def test_connected_smoke_subset_is_closed_under_at_least_one_source() -> None:
    adjacency = {
        "a": ("b",),
        "b": ("a", "c"),
        "c": ("b", "d"),
        "d": ("c", "e"),
        "e": ("d",),
    }

    subset = connected_smoke_subset(("a", "b", "c", "d", "e"), adjacency, target_count=3)

    assert 3 <= len(subset) <= 5
    selected = set(subset)
    assert all(any(source in selected for source in adjacency[name]) for name in subset)


def test_classify_upper_recovery_requires_resolved_narrow_finial_and_improvement() -> None:
    baseline = {
        "finial_to_lid_radius_ratio": 1.01,
        "resolved_narrow_top_element": False,
    }
    improved_but_unresolved = {
        "finial_to_lid_radius_ratio": 0.92,
        "resolved_narrow_top_element": False,
    }
    resolved = {
        "finial_to_lid_radius_ratio": 0.80,
        "resolved_narrow_top_element": True,
    }

    unresolved = classify_upper_recovery(baseline, improved_but_unresolved)
    accepted = classify_upper_recovery(baseline, resolved)

    assert unresolved["status"] == "recovery_exhausted_unresolved"
    assert unresolved["promote_upper_recovery"] is False
    assert accepted["status"] == "upper_recovery_improved"
    assert accepted["promote_upper_recovery"] is True
