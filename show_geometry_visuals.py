"""Open real Step 6 geometry visuals for a professor-facing demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from analysis_common import (
    load_selected_manifest,
    verify_selected_record,
)
from run_geometry_analysis import (
    analyze_pair,
    analyze_shape_record,
    pair_metrics,
    render_epipolar_figure,
    render_match_figure,
    render_shape_figure,
    shape_metrics,
)


def build_visuals(
    *,
    images_dir: Path,
    selection_manifest: Path,
    mode: str,
    pair: tuple[int, int] = (165, 166),
    supporting_pair: tuple[int, int] = (255, 256),
    shape_indices: tuple[int, ...] = (165, 255),
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Verify needed records and render visuals without opening GUI windows."""
    if mode not in {"matches", "epipolar", "shape", "all"}:
        raise ValueError(f"unknown geometry visual mode: {mode}")
    records = load_selected_manifest(selection_manifest)
    needed_indices: set[int] = set()
    if mode in {"matches", "epipolar", "all"}:
        needed_indices.update(pair)
    if mode in {"matches", "all"}:
        needed_indices.update(supporting_pair)
    if mode in {"shape", "all"}:
        needed_indices.update(shape_indices)
    for index in sorted(needed_indices):
        if index < 1 or index > len(records):
            raise IndexError(
                f"one-based selected-image index out of range: {index}; "
                f"valid range is 1..{len(records)}"
            )
        verify_selected_record(images_dir, records[index - 1])

    visuals: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {"shape": {}}
    primary_analysis = None
    if mode in {"matches", "epipolar", "all"}:
        primary_analysis = analyze_pair(images_dir, records, pair)
        metrics["pair"] = pair_metrics(primary_analysis)
    if mode in {"matches", "all"}:
        visuals[f"SIFT and RANSAC {pair[0]}-{pair[1]}"] = render_match_figure(
            primary_analysis
        )
        if supporting_pair != pair:
            supporting = analyze_pair(images_dir, records, supporting_pair)
            visuals[
                f"SIFT and RANSAC {supporting_pair[0]}-{supporting_pair[1]}"
            ] = render_match_figure(supporting)
            metrics["supporting_pair"] = pair_metrics(supporting)
    if mode in {"epipolar", "all"}:
        visuals[f"Epipolar geometry {pair[0]}-{pair[1]}"] = (
            render_epipolar_figure(primary_analysis)
        )
    if mode in {"shape", "all"}:
        for index in shape_indices:
            analysis = analyze_shape_record(images_dir, records, index)
            visuals[f"Classical shape {index}"] = render_shape_figure(analysis)
            metrics["shape"][str(index)] = shape_metrics(analysis)
    return visuals, metrics


def _display_copy(image: np.ndarray, maximum_size: tuple[int, int] = (1500, 900)) -> np.ndarray:
    maximum_width, maximum_height = maximum_size
    scale = min(
        1.0,
        maximum_width / image.shape[1],
        maximum_height / image.shape[0],
    )
    if scale == 1.0:
        return image.copy()
    return cv2.resize(
        image,
        (round(image.shape[1] * scale), round(image.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )


def show_visuals(
    visuals: dict[str, np.ndarray], *, timeout_ms: int = 0
) -> None:
    """Open labeled windows and close all of them on q, Esc, or timeout."""
    if timeout_ms < 0:
        raise ValueError("timeout_ms must not be negative")
    try:
        for title, image in visuals.items():
            display = _display_copy(image)
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, display.shape[1], display.shape[0])
            cv2.imshow(title, display)
        if timeout_ms:
            cv2.waitKey(timeout_ms)
            return
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key in {ord("q"), 27}:
                return
            if all(cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1 for title in visuals):
                return
    finally:
        cv2.destroyAllWindows()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("matches", "epipolar", "shape", "all"),
        default="all",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("preprocessing/pycolmap_input/images"),
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=Path("preprocessing/reports/selection_manifest.csv"),
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run real analysis/rendering and print metrics without opening windows.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=0,
        help="Close display windows after this many milliseconds; 0 waits for q or Esc.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    visuals, metrics = build_visuals(
        images_dir=args.images_dir,
        selection_manifest=args.selection_manifest,
        mode=args.mode,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    if args.no_display:
        print("No-display smoke mode completed; no GUI windows were opened.", flush=True)
        return 0
    print("Press q or Esc to close all geometry windows.", flush=True)
    show_visuals(visuals, timeout_ms=args.timeout_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
