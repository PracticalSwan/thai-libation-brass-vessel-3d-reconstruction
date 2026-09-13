"""Build a negative-only G8/G9 empty-board mask refinement set.

The canonical SAM masks are never overwritten.  This script compares each
selected G8/G9 geometry view with the registered empty-board tail from the
same setup, then vetoes only pixels that (a) are already foreground, (b) agree
with the empty reference, and (c) lie near the lower mask boundary.  The
result is an auditable downstream-dense input, not a replacement segmentation
claim.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v4_config import CAPTURE_V4_ROOT, RECONSTRUCTION_V4_ROOT, assert_output_path, sha256_file, write_json
from v4_isolation import refine_board_leakage


TARGET_RINGS = ("geo_g8", "geo_g9")
SMALL_SIZE = (768, 1020)  # width, height; deterministic 1/4-scale evidence
MAX_MATCH_SCORE = 24.0
BOTTOM_FRACTION = 0.55
BOUNDARY_DISTANCE_PX = 16.0


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _gray(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        image = np.asarray(handle.convert("L"), dtype=np.uint8)
    return cv2.resize(image, SMALL_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)


def _mask(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        image = np.asarray(handle.convert("L"), dtype=np.uint8)
    return image > 127


def _small_mask(mask: np.ndarray) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), SMALL_SIZE, interpolation=cv2.INTER_NEAREST) > 0


def _empty_candidates(rows: list[dict[str, str]], ring: str) -> list[dict[str, str]]:
    cluster = ring.replace("geo_", "empty_")
    result = [row for row in rows if row.get("source_role") == "empty_board" and row.get("source_cluster") == cluster]
    result.sort(key=lambda row: (int(row.get("frame_index") or 0), row.get("relative_path", "")))
    return result


def _geometry_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    result = [
        row
        for row in rows
        if row.get("source_role") == "geometry"
        and row.get("selected_for_geometry", "").lower() == "true"
        and row.get("logical_ring_id") in TARGET_RINGS
    ]
    result.sort(key=lambda row: (TARGET_RINGS.index(row["logical_ring_id"]), int(row.get("frame_index_within_logical_ring") or 0), row.get("relative_path", "")))
    return result


def _refine_one(
    row: dict[str, str],
    *,
    source_root: Path,
    mask_root: Path,
    empties: list[dict[str, str]],
    empty_root: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    name = Path(row["relative_path"]).name
    source_path = source_root / row["relative_path"]
    original_path = mask_root / name
    if not source_path.is_file() or not original_path.is_file():
        raise FileNotFoundError(f"G8/G9 refinement input is missing for {name}")
    original = _mask(original_path)
    small = _small_mask(original)
    ys, _ = np.where(small)
    if len(ys) == 0:
        raise ValueError(f"G8/G9 refinement source mask is empty for {name}")
    geometry = _gray(source_path)
    scores: list[tuple[float, str, dict[str, str]]] = []
    for empty in empties:
        empty_path = empty_root / empty["relative_path"]
        if not empty_path.is_file():
            raise FileNotFoundError(f"registered empty-board frame is missing: {empty_path}")
        candidate = _gray(empty_path)
        score = float(np.median(np.abs(geometry - candidate)[~small]))
        scores.append((score, empty["relative_path"], empty))
    scores.sort(key=lambda item: (item[0], item[1]))
    best_score, best_relative, best_row = scores[0]
    runner_score = scores[1][0] if len(scores) > 1 else None
    if not np.isfinite(best_score) or best_score > MAX_MATCH_SCORE:
        raise ValueError(f"empty-board match confidence is insufficient for {name}: score={best_score:.3f}")
    empty = _gray(empty_root / best_relative)
    diff = cv2.GaussianBlur(np.abs(geometry - empty), (5, 5), 0)
    inside = diff[small]
    p05 = float(np.percentile(inside, 5))
    threshold = float(max(10.0, min(24.0, p05 * 0.75)))
    y_min, y_max = int(ys.min()), int(ys.max())
    yy = np.indices(small.shape)[0]
    distance = cv2.distanceTransform(small.astype(np.uint8), cv2.DIST_L2, 5)
    candidate = small & (diff <= threshold) & (yy >= y_min + BOTTOM_FRACTION * (y_max - y_min)) & (distance <= BOUNDARY_DISTANCE_PX)
    # Drop isolated compression specks while preserving the negative-only
    # property.  Components are measured at evidence scale, then upsampled.
    labels_count, labels = cv2.connectedComponents(candidate.astype(np.uint8), 8)
    supported = np.zeros_like(candidate, dtype=bool)
    for label in range(1, labels_count):
        if int(np.sum(labels == label)) >= 4:
            supported[labels == label] = True
    supported_full = cv2.resize(supported.astype(np.uint8), (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    refined = refine_board_leakage(original, supported_board_pixels=supported_full)
    additions = int(np.sum(refined & ~original))
    removed = int(np.sum(original & ~refined))
    if additions:
        raise AssertionError(f"negative-only refinement added foreground pixels for {name}")
    confidence = float(max(0.0, 1.0 - best_score / 64.0))
    return refined, {
        "relative_path": row["relative_path"],
        "logical_ring_id": row["logical_ring_id"],
        "frame_index_within_logical_ring": int(row.get("frame_index_within_logical_ring") or 0),
        "original_mask_path": str(original_path.resolve()),
        "original_mask_sha256": sha256_file(original_path),
        "refined_mask_sha256": "",
        "empty_reference": {
            "relative_path": best_row["relative_path"],
            "sha256": str(best_row.get("sha256", "")),
            "match_score_median_abs_gray_outside_mask": best_score,
            "runner_up_score": runner_score,
            "confidence": confidence,
            "candidate_count": len(scores),
        },
        "threshold": threshold,
        "bottom_fraction_start": BOTTOM_FRACTION,
        "boundary_distance_px": BOUNDARY_DISTANCE_PX,
        "supported_board_pixels_small": int(np.sum(supported)),
        "removed_pixels": removed,
        "removed_fraction_of_original_mask": float(removed / max(int(np.sum(original)), 1)),
        "added_pixels": additions,
        "negative_only": True,
        "status": "applied",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=CAPTURE_V4_ROOT / "manifests" / "source_manifest.csv")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "CSX4213_Project_V4_Images")
    parser.add_argument("--mask-root", type=Path, default=CAPTURE_V4_ROOT / "derived" / "masks")
    parser.add_argument("--output-root", type=Path, default=CAPTURE_V4_ROOT / "derived" / "masks_g8_g9_negative_refined")
    parser.add_argument("--report", type=Path, default=RECONSTRUCTION_V4_ROOT / "reports" / "g8_g9_negative_refinement.json")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    source_root = args.source_root.resolve()
    mask_root = args.mask_root.resolve()
    output_root = assert_output_path(args.output_root)
    report_path = assert_output_path(args.report)
    if report_path.exists():
        raise FileExistsError(f"G8/G9 refinement report already exists; preserve it before retrying: {report_path}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"G8/G9 refinement output is non-empty; preserve it before retrying: {output_root}")
    rows = _rows(manifest)
    geometry = _geometry_rows(rows)
    if not geometry:
        raise ValueError("no selected G8/G9 geometry rows were found")
    empty_by_ring = {ring: _empty_candidates(rows, ring) for ring in TARGET_RINGS}
    missing_empty = {ring: len(values) for ring, values in empty_by_ring.items() if not values}
    if missing_empty:
        raise ValueError(f"registered empty-board evidence is missing for: {sorted(missing_empty)}")
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for row in geometry:
        ring = row["logical_ring_id"]
        try:
            refined, record = _refine_one(
                row,
                source_root=source_root,
                mask_root=mask_root,
                empties=empty_by_ring[ring],
                empty_root=source_root,
            )
            output_path = output_root / Path(row["relative_path"]).name
            Image.fromarray(np.where(refined, 255, 0).astype(np.uint8), mode="L").save(output_path, format="PNG")
            record["refined_mask_path"] = str(output_path.resolve())
            record["refined_mask_sha256"] = sha256_file(output_path)
            records.append(record)
        except Exception as error:
            failures.append(f"{row.get('relative_path', '')}: {type(error).__name__}: {error}")
    report = {
        "schema_version": 1,
        "status": "failed" if failures else "complete",
        "gate_effect": "downstream_dense_rerun_required",
        "negative_only": True,
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256_file(manifest),
        "source_root": str(source_root),
        "original_mask_root": str(mask_root),
        "output_root": str(output_root),
        "method": {
            "comparison": "same-setup empty-board grayscale median absolute difference outside provisional vessel mask",
            "small_size": list(SMALL_SIZE),
            "max_match_score": MAX_MATCH_SCORE,
            "threshold": "max(10, min(24, 0.75 * inside-mask difference p05))",
            "bottom_fraction_start": BOTTOM_FRACTION,
            "boundary_distance_px": BOUNDARY_DISTANCE_PX,
            "component_min_pixels_small": 4,
            "veto_operation": "refine_board_leakage (foreground intersection minus supported empty-board pixels)",
        },
        "empty_reference_sets": {
            ring: {
                "source_cluster": ring.replace("geo_", "empty_"),
                "count": len(values),
                "frames": [
                    {"relative_path": value["relative_path"], "sha256": value.get("sha256", "")}
                    for value in values
                ],
            }
            for ring, values in empty_by_ring.items()
        },
        "geometry_count": len(geometry),
        "refined_count": len(records),
        "failure_count": len(failures),
        "failures": failures,
        "removed_pixels_total": int(sum(int(item["removed_pixels"]) for item in records)),
        "records": records,
    }
    write_json(report_path, report)
    print(json.dumps({"report": str(report_path.resolve()), "status": report["status"], "geometry_count": len(geometry), "refined_count": len(records), "failure_count": len(failures)}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
