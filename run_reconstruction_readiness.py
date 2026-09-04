"""Step 9 pre-reconstruction readiness: masks, matching, connectivity, and camera audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis_common import SelectedImageRecord, verify_selected_images
from camera_readiness import CameraRecord, read_camera_record, summarize_camera_records
from geometry_detection import SiftConfig, SiftFeatures, extract_sift
from reconstruction_masks import (
    MaskRecord,
    infer_selected_masks,
    load_mask_manifest,
    sha256_file,
    write_mask_manifest,
)
from reconstruction_matching import (
    BENCHMARK_PAIRS,
    FEATURE_MODES,
    PairGeometryMetrics,
    SubsetDecision,
    choose_reconstruction_subset,
    filter_sift_features,
    is_strong_edge,
    measure_pair,
    summarize_benchmark,
)

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "IMG20260826122949"
SELECTED_DIR = ROOT / "preprocessing" / "pycolmap_input" / "images"
SELECTION_MANIFEST = ROOT / "preprocessing" / "reports" / "selection_manifest.csv"
SEGMENTATION_MANIFEST = ROOT / "ml_dataset" / "manifest.csv"
CHECKPOINT_PATH = ROOT / "analysis" / "ml" / "checkpoints" / "best_small_seg_cnn.pt"
RAW_PREDICTION_DIR = ROOT / "analysis" / "ml" / "full_predictions"
RECONSTRUCTION_MASK_DIR = ROOT / "analysis" / "ml" / "reconstruction_masks"
REPORTS_DIR = ROOT / "analysis" / "reports"
PRESENTATION_DIR = ROOT / "analysis" / "previews" / "presentation"
MASK_MANIFEST = REPORTS_DIR / "reconstruction_mask_manifest.csv"
MASK_SUMMARY = REPORTS_DIR / "step9_masks.json"
BENCHMARK_CSV = REPORTS_DIR / "step9_match_benchmark.csv"
BENCHMARK_JSON = REPORTS_DIR / "step9_match_benchmark.json"
CONNECTIVITY_CSV = REPORTS_DIR / "step9_connectivity.csv"
CONNECTIVITY_JSON = REPORTS_DIR / "step9_connectivity.json"
CAMERA_CSV = REPORTS_DIR / "step9_camera_readiness.csv"
CAMERA_JSON = REPORTS_DIR / "step9_camera_readiness.json"
STEP9_SUMMARY = REPORTS_DIR / "step9_summary.json"
SUBSET_DIR = ROOT / "preprocessing" / "reconstruction_input_v1"
SUBSET_MANIFEST = SUBSET_DIR / "manifest.csv"
SUBSET_README = SUBSET_DIR / "README.md"
MASK_FIGURE = PRESENTATION_DIR / "step9_01_reconstruction_masks.png"
BENCHMARK_FIGURE = PRESENTATION_DIR / "step9_02_match_benchmark.png"
CONNECTIVITY_FIGURE = PRESENTATION_DIR / "step9_03_connectivity.png"
CAMERA_FIGURE = PRESENTATION_DIR / "step9_04_camera_readiness.png"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"required Step 9 report is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Step 9 report must contain a JSON object: {path}")
    return value


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty Step 9 CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_chosen_mode(path: Path = BENCHMARK_JSON) -> str:
    payload = _read_json(path)
    mode = payload.get("chosen_mode")
    if mode not in FEATURE_MODES:
        raise ValueError(f"invalid chosen Step 9 feature mode: {mode}")
    return str(mode)


def build_step9_summary(
    mask: dict[str, object],
    benchmark: dict[str, object],
    connectivity: dict[str, object],
    camera: dict[str, object],
) -> dict[str, object]:
    required_mask = ("mask_count", "segmentation_manifest_sha256", "checkpoint_sha256")
    for key in required_mask:
        if key not in mask:
            raise ValueError(f"Step 9 mask summary missing {key}")
    benchmark_mode = benchmark.get("chosen_mode")
    connectivity_mode = connectivity.get("chosen_mode")
    if benchmark_mode not in FEATURE_MODES:
        raise ValueError("Step 9 benchmark chosen mode is invalid")
    if connectivity_mode != benchmark_mode:
        raise ValueError("Step 9 benchmark/connectivity chosen modes disagree")
    for key in (
        "adjacent_edge_count",
        "strong_adjacent_count",
        "weak_adjacent_count",
        "included_count",
        "excluded_count",
    ):
        if key not in connectivity:
            raise ValueError(f"Step 9 connectivity summary missing {key}")
    for key in ("record_count", "unique_signature_count", "camera_group_recommendation"):
        if key not in camera:
            raise ValueError(f"Step 9 camera summary missing {key}")
    return {
        "mask_count": int(mask["mask_count"]),
        "segmentation_manifest_sha256": mask["segmentation_manifest_sha256"],
        "checkpoint_sha256": mask["checkpoint_sha256"],
        "selection_manifest_sha256": mask.get("selection_manifest_sha256"),
        "chosen_feature_mode": benchmark_mode,
        "benchmark": benchmark.get("modes", {}),
        "connectivity": {
            key: connectivity[key]
            for key in (
                "adjacent_edge_count",
                "strong_adjacent_count",
                "weak_adjacent_count",
                "bridge_edge_count",
                "strong_bridge_count",
                "included_count",
                "excluded_count",
                "excluded_indices",
            )
            if key in connectivity
        },
        "camera": {
            "record_count": camera["record_count"],
            "unique_signature_count": camera["unique_signature_count"],
            "camera_group_recommendation": camera["camera_group_recommendation"],
            "geometry_orientation_consistent": camera.get(
                "geometry_orientation_consistent"
            ),
            "missing_fields": camera.get("missing_fields", {}),
        },
        "reconstruction_started": False,
        "next_boundary": "pyCOLMAP/SfM requires separate explicit authorization",
    }


def _overlay_preview(image: np.ndarray, mask: np.ndarray, width: int = 360) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    height = round(source_height * width / source_width)
    image_small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    shown = image_small.copy()
    inside = mask_small > 0
    tint = np.zeros_like(shown)
    tint[:, :, 1] = 220
    shown[inside] = (0.62 * shown[inside] + 0.38 * tint[inside]).astype(np.uint8)
    contours, _ = cv2.findContours(mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(shown, contours, -1, (0, 0, 240), 2, cv2.LINE_AA)
    return cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)


def _mask_figure(
    selected_records: Sequence[SelectedImageRecord], mask_records: Sequence[MaskRecord]
) -> None:
    selected_by_index = {record.index: record for record in selected_records}
    mask_by_index = {record.selected_index: record for record in mask_records}
    sample_indices = (1, 45, 72, 105, 165, 225, 255, 288)
    fig, axes = plt.subplots(4, 4, figsize=(12, 16))
    for sample_position, index in enumerate(sample_indices):
        selected = selected_by_index[index]
        mask_record = mask_by_index[index]
        image = cv2.imread(str(SELECTED_DIR / selected.filename), cv2.IMREAD_COLOR)
        raw_mask = cv2.imread(str(mask_record.raw_prediction_path), cv2.IMREAD_GRAYSCALE)
        clean_mask = cv2.imread(
            str(mask_record.reconstruction_mask_path), cv2.IMREAD_GRAYSCALE
        )
        if image is None or raw_mask is None or clean_mask is None:
            raise RuntimeError(f"failed to render Step 9 mask sample {index}")
        row = sample_position // 2
        pair = sample_position % 2
        raw_axis = axes[row, pair * 2]
        clean_axis = axes[row, pair * 2 + 1]
        raw_axis.imshow(_overlay_preview(image, raw_mask))
        clean_axis.imshow(_overlay_preview(image, clean_mask))
        raw_axis.set_title(
            f"{index:03d} raw CNN | fg={mask_record.raw_foreground_fraction:.3f}",
            fontsize=9,
        )
        clean_axis.set_title(
            f"{index:03d} reconstruction mask | fg={mask_record.reconstruction_foreground_fraction:.3f}",
            fontsize=9,
        )
        raw_axis.axis("off")
        clean_axis.axis("off")
    fig.suptitle(
        "Step 9A: frozen CNN predictions and deterministic reconstruction masks",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(MASK_FIGURE, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_masks_stage() -> dict[str, object]:
    verified = verify_selected_images(
        SELECTED_DIR, SELECTION_MANIFEST, expected_count=288
    )
    if not CHECKPOINT_PATH.is_file():
        raise ValueError(f"frozen SmallSegCNN checkpoint is missing: {CHECKPOINT_PATH}")
    segmentation_manifest_sha256 = sha256_file(SEGMENTATION_MANIFEST)
    mask_records = infer_selected_masks(
        verified.records,
        verified.images_dir,
        checkpoint_path=CHECKPOINT_PATH,
        segmentation_manifest_sha256=segmentation_manifest_sha256,
        raw_prediction_dir=RAW_PREDICTION_DIR,
        reconstruction_mask_dir=RECONSTRUCTION_MASK_DIR,
        raw_source_dir=RAW_DIR,
    )
    if len(mask_records) != 288:
        raise RuntimeError(f"expected 288 full predictions, found {len(mask_records)}")
    write_mask_manifest(MASK_MANIFEST, mask_records, project_root=ROOT)
    _mask_figure(verified.records, mask_records)
    changed_count = sum(
        record.raw_prediction_sha256 != record.reconstruction_mask_sha256
        for record in mask_records
    )
    summary = {
        "mask_count": len(mask_records),
        "raw_prediction_count": len(mask_records),
        "reconstruction_mask_count": len(mask_records),
        "cleaned_prediction_count": changed_count,
        "mean_raw_foreground_fraction": mean(
            record.raw_foreground_fraction for record in mask_records
        ),
        "mean_reconstruction_foreground_fraction": mean(
            record.reconstruction_foreground_fraction for record in mask_records
        ),
        "selection_manifest_sha256": verified.manifest_sha256,
        "segmentation_manifest_sha256": segmentation_manifest_sha256,
        "checkpoint_sha256": sha256_file(CHECKPOINT_PATH),
        "checkpoint_path": CHECKPOINT_PATH.relative_to(ROOT).as_posix(),
        "raw_prediction_directory": RAW_PREDICTION_DIR.relative_to(ROOT).as_posix(),
        "reconstruction_mask_directory": RECONSTRUCTION_MASK_DIR.relative_to(ROOT).as_posix(),
        "cleanup_rule": {
            "central_roi": "x=25%-75%, y=20%-85%",
            "secondary_minimum_area_fraction": 0.02,
            "secondary_maximum_bbox_gap_diagonal_fraction": 0.03,
            "holes_filled": False,
            "erosion": False,
        },
    }
    _write_json(MASK_SUMMARY, summary)
    return summary


def _mask_for_mode(record: MaskRecord, mode: str) -> np.ndarray | None:
    if mode == "unmasked":
        return None
    path = (
        record.raw_prediction_path
        if mode == "raw_cnn"
        else record.reconstruction_mask_path
    )
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"unreadable Step 9 mask for {record.selected_index}: {path}")
    return mask


def _features_for_mode(
    base: SiftFeatures, mask_record: MaskRecord, mode: str
) -> SiftFeatures:
    mask = _mask_for_mode(mask_record, mode)
    return base if mask is None else filter_sift_features(base, mask)


def _benchmark_figure(summary: dict[str, object]) -> None:
    modes_payload = summary["modes"]
    assert isinstance(modes_payload, dict)
    modes = [mode for mode in FEATURE_MODES if mode in modes_payload]
    total_inliers = [int(modes_payload[mode]["total_inliers"]) for mode in modes]
    ratios = [float(modes_payload[mode]["median_inlier_ratio"]) for mode in modes]
    errors = [
        float(modes_payload[mode]["median_sampson_error"])
        if modes_payload[mode]["median_sampson_error"] is not None
        else math.nan
        for mode in modes
    ]
    labels = [
        f"{mode}\n{'qualified' if modes_payload[mode].get('qualified') else 'not qualified'}"
        for mode in modes
    ]
    x = np.arange(len(modes))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    axes[0].bar(x, total_inliers)
    axes[0].set_xticks(x, labels)
    axes[0].set_title("Total RANSAC inliers")
    axes[0].set_ylabel("20-pair total")
    axes[1].bar(x, ratios)
    axes[1].set_xticks(x, labels)
    axes[1].set_title("Median inlier ratio")
    axes[1].set_ylim(0, max(0.05, max(ratios) * 1.2))
    axes[2].bar(x, errors)
    axes[2].set_xticks(x, labels)
    axes[2].set_title("Median Sampson error")
    axes[2].set_ylabel("analysis pixels²")
    chosen = summary["chosen_mode"]
    fig.suptitle(f"Step 9B: masked vs unmasked geometry benchmark | chosen: {chosen}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(BENCHMARK_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_benchmark_stage() -> dict[str, object]:
    verified = verify_selected_images(
        SELECTED_DIR, SELECTION_MANIFEST, expected_count=288
    )
    mask_records = load_mask_manifest(MASK_MANIFEST, project_root=ROOT, expected_count=288)
    selected_by_index = {record.index: record for record in verified.records}
    masks_by_index = {record.selected_index: record for record in mask_records}
    required_indices = sorted({index for pair in BENCHMARK_PAIRS for index in pair})
    missing = [index for index in required_indices if index not in selected_by_index or index not in masks_by_index]
    if missing:
        raise ValueError(f"benchmark index is missing from selected/mask inputs: {missing[0]}")

    config = SiftConfig()
    feature_cache: dict[int, SiftFeatures] = {}
    for index in required_indices:
        image = cv2.imread(
            str(verified.images_dir / selected_by_index[index].filename), cv2.IMREAD_COLOR
        )
        if image is None:
            raise ValueError(f"unreadable benchmark selected image: {index}")
        feature_cache[index] = extract_sift(image, config)

    rows: list[PairGeometryMetrics] = []
    for pair in BENCHMARK_PAIRS:
        for mode in FEATURE_MODES:
            first = _features_for_mode(feature_cache[pair[0]], masks_by_index[pair[0]], mode)
            second = _features_for_mode(feature_cache[pair[1]], masks_by_index[pair[1]], mode)
            rows.append(measure_pair(first, second, pair=pair, mode=mode, config=config))
    if len(rows) != 60:
        raise RuntimeError(f"expected 60 benchmark rows, found {len(rows)}")
    _write_csv(BENCHMARK_CSV, [row.to_dict() for row in rows])
    summary = summarize_benchmark(rows)
    summary["benchmark_pairs"] = [list(pair) for pair in BENCHMARK_PAIRS]
    summary["row_count"] = len(rows)
    summary["sift_config"] = {
        "maximum_width": config.maximum_width,
        "nfeatures": config.nfeatures,
        "ratio_threshold": config.ratio_threshold,
        "minimum_correspondences": config.minimum_correspondences,
        "ransac_threshold": config.ransac_threshold,
        "confidence": config.confidence,
        "rng_seed": config.rng_seed,
    }
    summary["qualification_rule"] = {
        "minimum_total_inlier_fraction_of_unmasked": 0.95,
        "minimum_median_inlier_ratio_relative_to_unmasked": "not lower",
        "maximum_median_sampson_error_multiplier": 1.10,
    }
    _write_json(BENCHMARK_JSON, summary)
    rendered_summary = _read_json(BENCHMARK_JSON)
    _benchmark_figure(rendered_summary)
    return rendered_summary


def _apply_mode_to_features(
    features: SiftFeatures,
    mask_record: MaskRecord | None,
    mode: str,
) -> SiftFeatures:
    if mode == "unmasked":
        return features
    if mask_record is None:
        raise ValueError("masked connectivity mode requires a mask record")
    return _features_for_mode(features, mask_record, mode)


def _connectivity_figure(
    adjacent_rows: Sequence[PairGeometryMetrics], summary: dict[str, object]
) -> None:
    x = np.arange(1, len(adjacent_rows) + 1)
    inliers = np.asarray([row.inliers for row in adjacent_rows], dtype=float)
    ratios = np.asarray([row.inlier_ratio for row in adjacent_rows], dtype=float)
    strong = np.asarray([is_strong_edge(row) for row in adjacent_rows], dtype=bool)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(x, inliers, linewidth=1.0)
    axes[0].axhline(15, linestyle="--", linewidth=1.0, label="strong threshold: 15 inliers")
    if np.any(~strong):
        axes[0].scatter(x[~strong], inliers[~strong], marker="x", label="weak edge")
    axes[0].set_ylabel("RANSAC inliers")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Adjacent sequence geometry")
    axes[1].plot(x, ratios, linewidth=1.0)
    axes[1].axhline(0.15, linestyle="--", linewidth=1.0, label="strong threshold: 0.15")
    if np.any(~strong):
        axes[1].scatter(x[~strong], ratios[~strong], marker="x", label="weak edge")
    axes[1].set_ylabel("Inlier ratio")
    axes[1].set_xlabel("Adjacent pair start position in selected sequence")
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "Step 9C: full-sequence connectivity | "
        f"mode={summary['chosen_mode']} | strong={summary['strong_adjacent_count']} "
        f"weak={summary['weak_adjacent_count']} | subset={summary['included_count']}/288",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CONNECTIVITY_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_subset_manifest(
    decisions: Sequence[SubsetDecision], chosen_mode: str, mask_records: Sequence[MaskRecord]
) -> None:
    mask_by_index = {record.selected_index: record for record in mask_records}
    rows: list[dict[str, object]] = []
    for decision in decisions:
        mask_path = ""
        if chosen_mode != "unmasked":
            mask_record = mask_by_index[decision.selected_index]
            chosen_path = (
                mask_record.raw_prediction_path
                if chosen_mode == "raw_cnn"
                else mask_record.reconstruction_mask_path
            )
            mask_path = chosen_path.relative_to(ROOT).as_posix()
        rows.append(
            {
                "selected_index": decision.selected_index,
                "filename": decision.filename,
                "include": int(decision.include),
                "reason": decision.reason,
                "selected_image_path": (
                    Path("preprocessing") / "pycolmap_input" / "images" / decision.filename
                ).as_posix(),
                "feature_mode": chosen_mode,
                "mask_path": mask_path,
            }
        )
    _write_csv(SUBSET_MANIFEST, rows)


def _write_subset_readme(summary: dict[str, object]) -> None:
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    text = f"""# Reconstruction Input v1\n\nThis directory records the Step 9C recommended reconstruction subset. It intentionally does **not** duplicate the 288 PREPROCESSED JPEGs. `manifest.csv` references the verified files already stored under `../pycolmap_input/images/`.\n\nMeasured readiness decision:\n\n- feature mode: `{summary['chosen_mode']}`\n- adjacent edges: {summary['adjacent_edge_count']}\n- strong adjacent edges: {summary['strong_adjacent_count']}\n- weak adjacent edges: {summary['weak_adjacent_count']}\n- included images: {summary['included_count']}\n- excluded weak-but-bridged images: {summary['excluded_count']}\n\nAn image is excluded only when neither incident adjacent edge is strong and its immediate predecessor/successor have a strong skip bridge. Weak images needed to preserve sequence coverage remain included.\n\nThis is readiness metadata only. Step 9 did not run pyCOLMAP or any reconstruction.\n"""
    SUBSET_README.write_text(text, encoding="utf-8")


def run_connectivity_stage() -> dict[str, object]:
    chosen_mode = load_chosen_mode(BENCHMARK_JSON)
    verified = verify_selected_images(
        SELECTED_DIR, SELECTION_MANIFEST, expected_count=288
    )
    mask_records = load_mask_manifest(MASK_MANIFEST, project_root=ROOT, expected_count=288)
    mask_by_index = {record.selected_index: record for record in mask_records}
    config = SiftConfig()

    adjacent_rows: list[PairGeometryMetrics] = []
    bridge_rows: list[PairGeometryMetrics] = []
    last_two: list[tuple[SelectedImageRecord, SiftFeatures]] = []
    for record in verified.records:
        image = cv2.imread(str(verified.images_dir / record.filename), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unreadable connectivity selected image: {record.filename}")
        base = extract_sift(image, config)
        current = _apply_mode_to_features(base, mask_by_index.get(record.index), chosen_mode)
        if last_two:
            previous_record, previous_features = last_two[-1]
            adjacent = measure_pair(
                previous_features,
                current,
                pair=(previous_record.index, record.index),
                mode=chosen_mode,
                config=config,
            )
            adjacent_rows.append(adjacent)
            if not is_strong_edge(adjacent) and len(last_two) >= 2:
                bridge_record, bridge_features = last_two[-2]
                bridge_rows.append(
                    measure_pair(
                        bridge_features,
                        current,
                        pair=(bridge_record.index, record.index),
                        mode=chosen_mode,
                        config=config,
                    )
                )
        last_two.append((record, current))
        if len(last_two) > 2:
            last_two.pop(0)

    if len(adjacent_rows) != 287:
        raise RuntimeError(f"expected 287 adjacent connectivity rows, found {len(adjacent_rows)}")
    decisions = choose_reconstruction_subset(
        verified.records, adjacent_rows, bridge_rows
    )
    included = [decision for decision in decisions if decision.include]
    excluded = [decision for decision in decisions if not decision.include]
    connectivity_rows: list[dict[str, object]] = []
    for edge_type, rows in (("adjacent", adjacent_rows), ("bridge", bridge_rows)):
        for row in rows:
            item = row.to_dict()
            item["edge_type"] = edge_type
            item["strong"] = int(is_strong_edge(row))
            connectivity_rows.append(item)
    _write_csv(CONNECTIVITY_CSV, connectivity_rows)
    _write_subset_manifest(decisions, chosen_mode, mask_records)

    summary: dict[str, object] = {
        "chosen_mode": chosen_mode,
        "adjacent_edge_count": len(adjacent_rows),
        "strong_adjacent_count": sum(is_strong_edge(row) for row in adjacent_rows),
        "weak_adjacent_count": sum(not is_strong_edge(row) for row in adjacent_rows),
        "bridge_edge_count": len(bridge_rows),
        "strong_bridge_count": sum(is_strong_edge(row) for row in bridge_rows),
        "included_count": len(included),
        "excluded_count": len(excluded),
        "excluded_indices": [decision.selected_index for decision in excluded],
        "weak_adjacent_pairs": [
            [row.pair_a, row.pair_b]
            for row in adjacent_rows
            if not is_strong_edge(row)
        ],
        "strong_edge_rule": {
            "minimum_inliers": 15,
            "minimum_inlier_ratio": 0.15,
            "required_status": "ok",
        },
        "selection_manifest_sha256": verified.manifest_sha256,
    }
    _write_json(CONNECTIVITY_JSON, summary)
    rendered = _read_json(CONNECTIVITY_JSON)
    _write_subset_readme(rendered)
    _connectivity_figure(adjacent_rows, rendered)
    return rendered


def _camera_figure(summary: dict[str, object]) -> None:
    groups = summary["signature_groups"]
    assert isinstance(groups, list)
    counts = [int(group["frame_count"]) for group in groups]
    labels = [f"signature {index + 1}" for index in range(len(groups))]
    missing = summary["missing_fields"]
    assert isinstance(missing, dict)
    missing_names = [
        "lens_model",
        "focal_length_mm",
        "focal_length_35mm",
        "digital_zoom_ratio",
    ]
    missing_values = [int(missing.get(name, 0)) for name in missing_names]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    axes[0].bar(np.arange(len(counts)), counts)
    axes[0].set_xticks(np.arange(len(counts)), labels, rotation=20)
    axes[0].set_ylabel("Selected frames")
    axes[0].set_title("Camera signature groups")
    missing_bars = axes[1].bar(np.arange(len(missing_values)), missing_values)
    axes[1].set_xticks(
        np.arange(len(missing_values)),
        [name.replace("_", "\n") for name in missing_names],
        fontsize=8,
    )
    axes[1].set_ylim(0, max(1.0, max(missing_values, default=0) * 1.2))
    axes[1].bar_label(missing_bars, labels=[str(value) for value in missing_values], padding=3)
    axes[1].set_ylabel("Missing values")
    axes[1].set_title("Metadata completeness (0 = complete)")
    fig.suptitle(
        "Step 9D: camera readiness | "
        f"{summary['camera_group_recommendation']} | signatures={summary['unique_signature_count']}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CAMERA_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_camera_stage() -> dict[str, object]:
    verified = verify_selected_images(
        SELECTED_DIR, SELECTION_MANIFEST, expected_count=288
    )
    records: list[CameraRecord] = []
    for selected in verified.records:
        raw_path = RAW_DIR / selected.filename
        if not raw_path.is_file():
            raise ValueError(f"selected filename missing from raw source: {selected.filename}")
        camera = read_camera_record(selected.index, raw_path)
        if (camera.width, camera.height) != (selected.width, selected.height):
            raise ValueError(f"raw/selected geometry mismatch for camera audit: {selected.filename}")
        records.append(camera)
    if len(records) != 288:
        raise RuntimeError(f"expected 288 camera records, found {len(records)}")
    _write_csv(CAMERA_CSV, [record.to_dict() for record in records])
    summary = summarize_camera_records(records)
    summary["selection_manifest_sha256"] = verified.manifest_sha256
    summary["metadata_source"] = "immutable raw JPEG EXIF matched by selected filename"
    summary["image_resampling_performed"] = False
    summary["calibration_performed"] = False
    _write_json(CAMERA_JSON, summary)
    rendered = _read_json(CAMERA_JSON)
    _camera_figure(rendered)
    return rendered


def run_summary_stage() -> dict[str, object]:
    mask = _read_json(MASK_SUMMARY)
    benchmark = _read_json(BENCHMARK_JSON)
    connectivity = _read_json(CONNECTIVITY_JSON)
    camera = _read_json(CAMERA_JSON)
    summary = build_step9_summary(mask, benchmark, connectivity, camera)
    _write_json(STEP9_SUMMARY, summary)
    return _read_json(STEP9_SUMMARY)


def run_stage(stage: str) -> dict[str, object]:
    if stage == "masks":
        return run_masks_stage()
    if stage == "benchmark":
        return run_benchmark_stage()
    if stage == "connectivity":
        return run_connectivity_stage()
    if stage == "camera":
        return run_camera_stage()
    if stage == "summary":
        return run_summary_stage()
    if stage == "all":
        run_masks_stage()
        run_benchmark_stage()
        run_connectivity_stage()
        run_camera_stage()
        return run_summary_stage()
    raise ValueError(f"unknown Step 9 stage: {stage}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("masks", "benchmark", "connectivity", "camera", "summary", "all"),
        default="all",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_stage(args.stage)
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
