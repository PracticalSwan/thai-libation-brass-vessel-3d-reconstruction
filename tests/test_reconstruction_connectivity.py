from analysis_common import SelectedImageRecord
from reconstruction_matching import (
    PairGeometryMetrics,
    adjacent_pairs,
    bridge_pairs_for_weak_edges,
    choose_reconstruction_subset,
    is_strong_edge,
)


def _metric(
    pair: tuple[int, int],
    *,
    inliers: int = 20,
    ratio: float = 0.25,
    status: str = "ok",
) -> PairGeometryMetrics:
    return PairGeometryMetrics(
        pair_a=pair[0],
        pair_b=pair[1],
        mode="unmasked",
        keypoints_a=100,
        keypoints_b=100,
        candidate_matches=max(inliers, 1),
        inliers=inliers,
        inlier_ratio=ratio,
        median_sampson_error=0.5,
        p90_sampson_error=1.0,
        grid_coverage=0.5,
        status=status,
    )


def _record(index: int) -> SelectedImageRecord:
    return SelectedImageRecord(
        index=index,
        filename=f"frame_{index:03d}.jpg",
        variant="PREPROCESSED",
        width=3072,
        height=4080,
        size_bytes=1,
        sha256="0" * 64,
        decision="ACCEPT",
        reasons="",
    )


def test_strong_edge_requires_status_inliers_and_ratio():
    assert is_strong_edge(_metric((1, 2), inliers=15, ratio=0.15))
    assert not is_strong_edge(_metric((1, 2), inliers=14, ratio=0.50))
    assert not is_strong_edge(_metric((1, 2), inliers=30, ratio=0.14))
    assert not is_strong_edge(
        _metric((1, 2), inliers=30, ratio=0.50, status="insufficient_geometry")
    )


def test_adjacent_and_weak_bridge_pairs_follow_sequence_order():
    indices = [1, 2, 3, 4]
    assert adjacent_pairs(indices) == ((1, 2), (2, 3), (3, 4))
    rows = (
        _metric((1, 2), inliers=30),
        _metric((2, 3), inliers=5),
        _metric((3, 4), inliers=4),
    )
    assert bridge_pairs_for_weak_edges(indices, rows) == ((1, 3), (2, 4))


def test_subset_excludes_only_weak_middle_frame_with_strong_bridge():
    records = tuple(_record(index) for index in (1, 2, 3))
    adjacent = (
        _metric((1, 2), inliers=5),
        _metric((2, 3), inliers=5),
    )
    bridge = (_metric((1, 3), inliers=25),)

    decisions = choose_reconstruction_subset(records, adjacent, bridge)

    by_index = {decision.selected_index: decision for decision in decisions}
    assert by_index[1].include is True
    assert by_index[3].include is True
    assert by_index[2].include is False
    assert by_index[2].reason == "exclude_weak_bridged"


def test_subset_retains_weak_middle_frame_when_bridge_is_not_strong():
    records = tuple(_record(index) for index in (1, 2, 3))
    adjacent = (
        _metric((1, 2), inliers=5),
        _metric((2, 3), inliers=5),
    )
    bridge = (_metric((1, 3), inliers=8),)

    decisions = choose_reconstruction_subset(records, adjacent, bridge)

    middle = decisions[1]
    assert middle.include is True
    assert middle.reason == "keep_weak_bridge_needed"


def test_subset_keeps_frame_with_any_strong_adjacent_edge():
    records = tuple(_record(index) for index in (1, 2, 3))
    adjacent = (
        _metric((1, 2), inliers=20),
        _metric((2, 3), inliers=5),
    )

    decisions = choose_reconstruction_subset(records, adjacent, ())

    assert decisions[1].include is True
    assert decisions[1].reason == "keep_connected"
