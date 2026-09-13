"""Acquisition-aware phase pairing and mask-filtered learned features.

V4 rings have unequal counts and no physical angle log.  Pairing therefore
uses normalized circular phase and explicit wrap closure, never raw frame index
or a fabricated degree value.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from v4_config import fingerprint
from v4_isolation import filter_keypoints


@dataclass(frozen=True)
class PhaseFrame:
    filename: str
    ring_id: str
    phase_01: float
    selected: bool = True

    def __post_init__(self) -> None:
        if not self.filename or not self.ring_id:
            raise ValueError("phase frames require filename and ring_id")
        if not math.isfinite(float(self.phase_01)):
            raise ValueError("phase_01 must be finite")
        object.__setattr__(self, "phase_01", normalize_phase(float(self.phase_01)))


def normalize_phase(value: float) -> float:
    if not math.isfinite(float(value)):
        raise ValueError("phase must be finite")
    phase = float(value) % 1.0
    # Keep the upper endpoint out of the half-open phase interval.
    return 0.0 if math.isclose(phase, 1.0, abs_tol=1e-12) else phase


def circular_distance(first: float, second: float) -> float:
    delta = abs(normalize_phase(first) - normalize_phase(second))
    return min(delta, 1.0 - delta)


def _frame(value: PhaseFrame | Mapping[str, Any], ring_id: str | None = None) -> PhaseFrame:
    if isinstance(value, PhaseFrame):
        return value
    resolved_ring = ring_id or str(value.get("ring_id") or value.get("logical_ring_id") or "")
    selected = value.get("selected", value.get("selected_for_geometry", True))
    return PhaseFrame(
        filename=str(value.get("filename") or value.get("relative_path") or ""),
        ring_id=resolved_ring,
        phase_01=float(value.get("phase_01", 0.0)),
        selected=bool(selected),
    )


def _normalized_pair(first: str, second: str) -> tuple[str, str]:
    if not first or not second or first == second:
        raise ValueError("image pair requires two distinct filenames")
    return (first, second) if first < second else (second, first)


def _ring_frames(
    frames_by_ring: Mapping[str, Sequence[PhaseFrame | Mapping[str, Any]]],
) -> dict[str, tuple[PhaseFrame, ...]]:
    result: dict[str, tuple[PhaseFrame, ...]] = {}
    for ring_id in sorted(str(key) for key in frames_by_ring):
        frames = tuple(
            sorted(
                (_frame(value, ring_id) for value in frames_by_ring[ring_id]),
                key=lambda item: (item.phase_01, item.filename),
            )
        )
        selected = tuple(frame for frame in frames if frame.selected)
        if len({frame.filename for frame in selected}) != len(selected):
            raise ValueError(f"duplicate selected filename in ring {ring_id}")
        result[ring_id] = selected
    return result


def build_pair_schedule(
    frames_by_ring: Mapping[str, Sequence[PhaseFrame | Mapping[str, Any]]],
    *,
    same_ring_neighbors: int = 1,
    wider_phase_neighbors: int = 2,
    cross_ring_neighbors: int = 1,
    cross_ring_offsets: Mapping[tuple[str, str] | str, float] | None = None,
    ring_order: Sequence[str] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Build a deterministic unordered pair set for unequal circular rings."""

    if same_ring_neighbors < 1 or wider_phase_neighbors < 0 or cross_ring_neighbors < 0:
        raise ValueError("pair neighborhood widths must be nonnegative (same-ring >= 1)")
    rings = _ring_frames(frames_by_ring)
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(first: str, second: str) -> None:
        pair = _normalized_pair(first, second)
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)

    for ring_id, frames in rings.items():
        count = len(frames)
        if count < 2:
            continue
        width = min(count - 1, same_ring_neighbors + wider_phase_neighbors)
        for position, frame in enumerate(frames):
            # Positive circular steps include the explicit phase 1 -> 0 edge.
            for step in range(1, width + 1):
                add(frame.filename, frames[(position + step) % count].filename)

    if ring_order is None:
        ring_ids = sorted(rings)
    else:
        requested = [str(value) for value in ring_order]
        if len(set(requested)) != len(requested):
            raise ValueError("ring_order contains duplicate ring ids")
        missing = [ring_id for ring_id in requested if ring_id not in rings]
        if missing:
            raise ValueError(f"ring_order references unknown rings: {missing}")
        # Keep an explicit acquisition/elevation order authoritative, while
        # retaining any additional supplied rings deterministically at the end.
        ring_ids = requested + [ring_id for ring_id in sorted(rings) if ring_id not in requested]
    for left_index, left_id in enumerate(ring_ids[:-1]):
        right_id = ring_ids[left_index + 1]
        left_frames, right_frames = rings[left_id], rings[right_id]
        if not left_frames or not right_frames:
            continue
        raw_offset = 0.0
        if cross_ring_offsets:
            direct = cross_ring_offsets.get((left_id, right_id), None)  # type: ignore[arg-type]
            if direct is None:
                direct = cross_ring_offsets.get(f"{left_id}->{right_id}", None)  # type: ignore[arg-type]
            if direct is None:
                reverse = cross_ring_offsets.get((right_id, left_id), None)  # type: ignore[arg-type]
                if reverse is None:
                    reverse = cross_ring_offsets.get(f"{right_id}->{left_id}", None)  # type: ignore[arg-type]
                direct = -float(reverse) if reverse is not None else 0.0
            raw_offset = float(direct)
        offset = normalize_phase(float(raw_offset))
        for frame in left_frames:
            target = normalize_phase(frame.phase_01 + offset)
            distances = [circular_distance(target, candidate.phase_01) for candidate in right_frames]
            nearest = min(range(len(right_frames)), key=lambda idx: (distances[idx], right_frames[idx].filename))
            candidate_positions = [
                (nearest + delta) % len(right_frames)
                for delta in range(-cross_ring_neighbors, cross_ring_neighbors + 1)
            ]
            for position in sorted(set(candidate_positions)):
                add(frame.filename, right_frames[position].filename)
    return tuple(pairs)


def estimate_circular_phase_offset(
    scores: Mapping[float, float] | Sequence[tuple[float, float]],
    *,
    minimum_margin: float = 0.0,
) -> dict[str, Any]:
    """Choose the strongest supported offset and expose ambiguity honestly."""

    entries = [(normalize_phase(float(offset)), float(score)) for offset, score in (scores.items() if isinstance(scores, Mapping) else scores)]
    if not entries:
        raise ValueError("at least one phase-offset score is required")
    if not all(math.isfinite(score) for _, score in entries):
        raise ValueError("phase-offset scores must be finite")
    ordered = sorted(entries, key=lambda item: (-item[1], item[0]))
    best_offset, best_score = ordered[0]
    runner_score = ordered[1][1] if len(ordered) > 1 else best_score
    margin = best_score - runner_score
    confidence = 1.0 if best_score == 0 and runner_score == 0 else max(0.0, margin) / max(abs(best_score), 1e-9)
    return {
        "offset_01": best_offset,
        "score": best_score,
        "runner_up_score": runner_score,
        "margin": margin,
        "confidence": float(min(1.0, max(0.0, confidence))),
        "ambiguous": bool(margin < minimum_margin),
        "method": "mask_restricted_learned_inlier_support",
    }


def schedule_hash(pairs: Sequence[tuple[str, str]]) -> str:
    normalized = [_normalized_pair(*pair) for pair in pairs]
    if len(set(normalized)) != len(normalized):
        raise ValueError("pair schedule contains duplicate unordered pairs")
    return fingerprint(normalized)


def feature_cache_identity(
    *,
    source_sha256: str,
    feature_mask_sha256: str,
    model_identity: Mapping[str, Any],
) -> str:
    """Identity required before a vessel-only feature cache may be reused."""

    if len(source_sha256) != 64 or len(feature_mask_sha256) != 64:
        raise ValueError("source and feature-mask identities must be SHA-256 strings")
    return fingerprint(
        {
            "source_sha256": source_sha256,
            "feature_mask_sha256": feature_mask_sha256,
            "model_identity": dict(model_identity),
        }
    )


def filter_feature_keypoints(
    keypoints: np.ndarray,
    feature_mask: np.ndarray,
    *,
    boundary_exclusion: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter ALIKED points and return their original row indices."""

    return filter_keypoints(
        np.asarray(keypoints),
        feature_mask,
        boundary_exclusion=boundary_exclusion,
    )


__all__ = [
    "PhaseFrame",
    "build_pair_schedule",
    "circular_distance",
    "estimate_circular_phase_offset",
    "feature_cache_identity",
    "filter_feature_keypoints",
    "normalize_phase",
    "schedule_hash",
]
