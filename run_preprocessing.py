"""Orchestrate reports, matching evidence, previews, and pyCOLMAP input export."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from preprocess_images import (
    encode_preprocessed_jpeg,
    preprocess_image,
    preprocess_image_array,
)
from quality_check import (
    QualityRecord,
    analyze_image,
    calibrate_thresholds,
    decide_quality,
)


DEFAULT_REPRESENTATIVE_PAIRS = (
    (15, 16),
    (45, 46),
    (75, 76),
    (105, 106),
    (135, 136),
    (165, 166),
    (195, 196),
    (225, 226),
    (255, 256),
    (280, 281),
)


@dataclass(frozen=True)
class PipelineConfig:
    raw_dir: Path
    baseline_manifest: Path
    reports_dir: Path
    previews_dir: Path
    output_dir: Path
    expected_raw_count: int = 297
    representative_pairs: tuple[tuple[int, int], ...] = (
        DEFAULT_REPRESENTATIVE_PAIRS
    )
    analysis_width: int = 800


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths_overlap(first: Path, second: Path) -> bool:
    first_resolved = first.resolve()
    second_resolved = second.resolve()
    return (
        first_resolved == second_resolved
        or first_resolved in second_resolved.parents
        or second_resolved in first_resolved.parents
    )


def _validate_pipeline_paths(config: PipelineConfig) -> None:
    destinations = {
        "reports": config.reports_dir,
        "previews": config.previews_dir,
        "selected output": config.output_dir,
    }
    for label, destination in destinations.items():
        if _paths_overlap(config.raw_dir, destination):
            raise ValueError(
                f"{label} directory must not overlap the immutable raw directory"
            )
    destination_items = list(destinations.items())
    for index, (first_label, first_path) in enumerate(destination_items):
        for second_label, second_path in destination_items[index + 1 :]:
            if _paths_overlap(first_path, second_path):
                raise ValueError(
                    f"{first_label} and {second_label} directories must not overlap"
                )


def verify_raw_manifest(raw_dir: Path, manifest_path: Path) -> dict[str, Any]:
    """Compare every raw JPEG with the published size/SHA-256 baseline."""
    baseline = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {entry["filename"]: entry for entry in baseline["files"]}
    actual_paths = {
        path.name: path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    }

    missing_files = sorted(set(expected) - set(actual_paths))
    unexpected_files = sorted(set(actual_paths) - set(expected))
    size_mismatches: list[str] = []
    hash_mismatches: list[str] = []
    for filename in sorted(set(expected) & set(actual_paths)):
        path = actual_paths[filename]
        entry = expected[filename]
        if path.stat().st_size != int(entry["size_bytes"]):
            size_mismatches.append(filename)
        if _sha256(path) != entry["sha256"]:
            hash_mismatches.append(filename)

    mismatched_files = set(
        missing_files + unexpected_files + size_mismatches + hash_mismatches
    )
    result = {
        "expected_count": int(baseline["image_count"]),
        "actual_count": len(actual_paths),
        "missing_files": missing_files,
        "unexpected_files": unexpected_files,
        "size_mismatches": size_mismatches,
        "hash_mismatches": hash_mismatches,
        "mismatch_count": len(mismatched_files),
    }
    result["unchanged"] = (
        result["expected_count"] == result["actual_count"]
        and result["mismatch_count"] == 0
    )
    return result


def export_selected_images(
    source_dir: Path,
    output_dir: Path,
    filenames: Iterable[str],
    *,
    variant: str,
) -> list[dict[str, Any]]:
    """Create one clean deterministic directory containing selected images."""
    source_resolved = source_dir.resolve()
    output_resolved = output_dir.resolve()
    if _paths_overlap(source_resolved, output_resolved):
        raise ValueError("output directory must be outside the immutable source tree")
    if variant not in {"RAW", "PREPROCESSED"}:
        raise ValueError("variant must be RAW or PREPROCESSED")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_filenames = {
        path.name
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    existing_entries = list(output_dir.iterdir())
    unexpected_directories = [entry for entry in existing_entries if entry.is_dir()]
    unexpected_files = [
        entry
        for entry in existing_entries
        if entry.is_file() and entry.name not in source_filenames
    ]
    if unexpected_directories:
        raise ValueError(
            f"unexpected directory in output set: {unexpected_directories[0]}"
        )
    if unexpected_files:
        raise ValueError(f"unexpected file in output set: {unexpected_files[0]}")
    for stale in existing_entries:
        stale.unlink()

    manifest: list[dict[str, Any]] = []
    for filename in sorted(set(filenames)):
        if Path(filename).name != filename:
            raise ValueError(f"selection filename must not contain a path: {filename}")
        source_path = source_dir / filename
        output_path = output_dir / filename
        if variant == "RAW":
            shutil.copyfile(source_path, output_path)
        else:
            preprocess_image(source_path, output_path)
        image = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"exported image is unreadable: {output_path}")
        height, width = image.shape[:2]
        manifest.append(
            {
                "filename": filename,
                "variant": variant,
                "width": width,
                "height": height,
                "size_bytes": output_path.stat().st_size,
                "sha256": _sha256(output_path),
            }
        )
    return manifest


def choose_reconstruction_variant(
    comparisons: Sequence[Mapping[str, int]],
) -> str:
    """Prefer preprocessing only when geometric inlier evidence is better."""
    if not comparisons:
        raise ValueError("at least one SIFT comparison is required")
    raw_total = sum(int(row["raw_inliers"]) for row in comparisons)
    preprocessed_total = sum(
        int(row["preprocessed_inliers"]) for row in comparisons
    )
    non_worse_pairs = sum(
        int(row["preprocessed_inliers"]) >= int(row["raw_inliers"])
        for row in comparisons
    )
    required_non_worse = (len(comparisons) + 1) // 2
    if preprocessed_total > raw_total and non_worse_pairs >= required_non_worse:
        return "PREPROCESSED"
    return "RAW"


def validate_run_counts(
    *,
    expected_raw_count: int,
    actual_raw_count: int,
    quality_row_count: int,
    selected_count: int,
    selection_manifest_count: int,
    output_count: int,
) -> None:
    mismatches: list[str] = []
    if actual_raw_count != expected_raw_count:
        mismatches.append(
            f"actual_raw_count={actual_raw_count}, expected_raw_count={expected_raw_count}"
        )
    if quality_row_count != expected_raw_count:
        mismatches.append(
            f"quality_row_count={quality_row_count}, expected_raw_count={expected_raw_count}"
        )
    if selection_manifest_count != selected_count:
        mismatches.append(
            "selection_manifest_count="
            f"{selection_manifest_count}, selected_count={selected_count}"
        )
    if output_count != selected_count:
        mismatches.append(
            f"output_count={output_count}, selected_count={selected_count}"
        )
    if mismatches:
        raise ValueError("run count mismatch: " + "; ".join(mismatches))


def _matching_image(image: np.ndarray, maximum_width: int = 1200) -> np.ndarray:
    if image.shape[1] <= maximum_width:
        return image
    scale = maximum_width / image.shape[1]
    return cv2.resize(
        image,
        (maximum_width, round(image.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _sift_match_metrics(
    first: np.ndarray,
    second: np.ndarray,
    *,
    ratio_threshold: float = 0.75,
) -> dict[str, int | float]:
    first_gray = cv2.cvtColor(_matching_image(first), cv2.COLOR_BGR2GRAY)
    second_gray = cv2.cvtColor(_matching_image(second), cv2.COLOR_BGR2GRAY)
    sift = getattr(cv2, "SIFT_create")(nfeatures=8000)
    first_keypoints, first_descriptors = sift.detectAndCompute(first_gray, None)
    second_keypoints, second_descriptors = sift.detectAndCompute(second_gray, None)
    if first_descriptors is None or second_descriptors is None:
        return {
            "keypoints_1": len(first_keypoints),
            "keypoints_2": len(second_keypoints),
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
        }

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    neighbors = matcher.knnMatch(first_descriptors, second_descriptors, k=2)
    good_matches = [
        pair[0]
        for pair in neighbors
        if len(pair) == 2 and pair[0].distance < ratio_threshold * pair[1].distance
    ]
    inliers = 0
    if len(good_matches) >= 8:
        first_points = np.asarray(
            [first_keypoints[match.queryIdx].pt for match in good_matches],
            dtype=np.float32,
        )
        second_points = np.asarray(
            [second_keypoints[match.trainIdx].pt for match in good_matches],
            dtype=np.float32,
        )
        cv2.setRNGSeed(4213)
        find_fundamental_matrix = getattr(cv2, "findFundamentalMat")
        try:
            _, mask = find_fundamental_matrix(
                first_points,
                second_points,
                cv2.FM_RANSAC,
                1.5,
                0.99,
            )
        except cv2.error:
            mask = None
        if mask is not None:
            inliers = int(np.count_nonzero(mask))

    return {
        "keypoints_1": len(first_keypoints),
        "keypoints_2": len(second_keypoints),
        "good_matches": len(good_matches),
        "inliers": inliers,
        "inlier_ratio": inliers / len(good_matches) if good_matches else 0.0,
    }


def compare_sift_pair(first_path: Path, second_path: Path) -> dict[str, Any]:
    """Compare RAW and conservative-preprocessing correspondence evidence."""
    first = cv2.imread(str(first_path), cv2.IMREAD_COLOR)
    second = cv2.imread(str(second_path), cv2.IMREAD_COLOR)
    if first is None or second is None:
        raise ValueError(f"unreadable SIFT comparison pair: {first_path}, {second_path}")

    first_jpeg = cv2.imdecode(
        np.frombuffer(encode_preprocessed_jpeg(first, quality=95), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    second_jpeg = cv2.imdecode(
        np.frombuffer(encode_preprocessed_jpeg(second, quality=95), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if first_jpeg is None or second_jpeg is None:
        raise ValueError("failed to decode preprocessed JPEG comparison artifact")

    raw = _sift_match_metrics(first, second)
    preprocessed = _sift_match_metrics(
        first_jpeg,
        second_jpeg,
    )
    result: dict[str, Any] = {
        "image_1": first_path.name,
        "image_2": second_path.name,
    }
    result.update({f"raw_{key}": value for key, value in raw.items()})
    result.update(
        {f"preprocessed_{key}": value for key, value in preprocessed.items()}
    )
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_before_after_preview(
    source_path: Path,
    output_path: Path,
    *,
    index: int,
) -> None:
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError(f"unreadable preview source: {source_path}")
    processed = preprocess_image_array(source)
    preview_height = 720
    preview_width = round(source.shape[1] * preview_height / source.shape[0])
    raw_preview = cv2.resize(
        source, (preview_width, preview_height), interpolation=cv2.INTER_AREA
    )
    processed_preview = cv2.resize(
        processed, (preview_width, preview_height), interpolation=cv2.INTER_AREA
    )
    header_height = 54
    canvas = np.full(
        (preview_height + header_height, preview_width * 2, 3),
        245,
        dtype=np.uint8,
    )
    canvas[header_height:, :preview_width] = raw_preview
    canvas[header_height:, preview_width:] = processed_preview
    cv2.putText(
        canvas,
        f"RAW - image {index:03d}",
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "PREPROCESSED - 15% CLAHE luminance blend",
        (preview_width + 18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]
    ):
        raise OSError(f"failed to write preview: {output_path}")


def _save_decision_sheets(
    raw_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    previews_dir: Path,
) -> list[str]:
    outliers = [row for row in rows if row["decision"] != "ACCEPT"]
    outputs: list[str] = []
    if not outliers:
        return outputs
    columns = 5
    rows_per_sheet = 5
    items_per_sheet = columns * rows_per_sheet
    tile_width = 190
    image_height = 252
    label_height = 66
    for start in range(0, len(outliers), items_per_sheet):
        chunk = outliers[start : start + items_per_sheet]
        sheet = np.full(
            (rows_per_sheet * (image_height + label_height), columns * tile_width, 3),
            248,
            dtype=np.uint8,
        )
        for offset, row in enumerate(chunk):
            image = cv2.imread(str(raw_dir / str(row["filename"])), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"unreadable decision preview: {row['filename']}")
            thumb = cv2.resize(
                image,
                (tile_width, image_height),
                interpolation=cv2.INTER_AREA,
            )
            tile_row, tile_column = divmod(offset, columns)
            y = tile_row * (image_height + label_height)
            x = tile_column * tile_width
            sheet[y : y + image_height, x : x + tile_width] = thumb
            label_color = (
                (20, 20, 190) if row["decision"] == "REJECT" else (0, 125, 220)
            )
            cv2.putText(
                sheet,
                f"{int(row['index']):03d} {row['decision']}",
                (x + 5, y + image_height + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                label_color,
                2,
                cv2.LINE_AA,
            )
            reason = str(row["reasons"]).split(";", maxsplit=1)[0][:26]
            cv2.putText(
                sheet,
                reason,
                (x + 5, y + image_height + 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (40, 40, 40),
                1,
                cv2.LINE_AA,
            )
        end = min(start + items_per_sheet, len(outliers))
        filename = f"decisions_{start + 1:03d}_{end:03d}.jpg"
        output_path = previews_dir / filename
        if not cv2.imwrite(
            str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]
        ):
            raise OSError(f"failed to write decision sheet: {output_path}")
        outputs.append(filename)
    return outputs


def _save_sift_chart(
    comparisons: Sequence[Mapping[str, Any]], output_path: Path
) -> None:
    width = 1400
    row_height = 78
    header_height = 90
    height = header_height + len(comparisons) * row_height + 30
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    cv2.putText(
        canvas,
        "Geometrically verified SIFT matches (fundamental-matrix RANSAC inliers)",
        (24, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.76,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "RAW",
        (24, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 80, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "PREPROCESSED",
        (100, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (35, 135, 40),
        2,
        cv2.LINE_AA,
    )
    maximum = max(
        max(int(row["raw_inliers"]), int(row["preprocessed_inliers"]))
        for row in comparisons
    )
    maximum = max(maximum, 1)
    bar_start = 400
    bar_width = width - bar_start - 90
    for index, row in enumerate(comparisons):
        y = header_height + index * row_height
        label = f"{row['image_1']} -> {row['image_2']}"
        cv2.putText(
            canvas,
            label,
            (24, y + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (45, 45, 45),
            1,
            cv2.LINE_AA,
        )
        for offset, key, color in [
            (10, "raw_inliers", (180, 80, 20)),
            (40, "preprocessed_inliers", (35, 135, 40)),
        ]:
            value = int(row[key])
            length = round(bar_width * value / maximum)
            cv2.rectangle(
                canvas,
                (bar_start, y + offset),
                (bar_start + length, y + offset + 20),
                color,
                -1,
            )
            cv2.putText(
                canvas,
                str(value),
                (bar_start + length + 8, y + offset + 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )
    if not cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
        raise OSError(f"failed to write SIFT chart: {output_path}")


def _write_input_readme(
    output_dir: Path,
    *,
    variant: str,
    selected_count: int,
    raw_count: int,
) -> None:
    text = f"""# Final pyCOLMAP Input Set

Use every image in `images/` as the input to the next pyCOLMAP stage.

- Selected variant: **{variant}**
- Selected images: **{selected_count} of {raw_count}**
- Selection rule: include `ACCEPT` and `WARN`; exclude only `REJECT`
- Naming: original deterministic capture filenames

The variant was selected from the representative SIFT matching evidence in
`../reports/sift_matching.json`. `WARN` images remain included because a
warning is a review signal, not an automatic rejection. Do not run pyCOLMAP
until the preprocessing milestone and raw-hash verification are accepted.
"""
    readme_path = output_dir.parent / "README.md"
    readme_path.write_text(text, encoding="utf-8")


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    """Run the complete bounded preprocessing stage without invoking pyCOLMAP."""
    _validate_pipeline_paths(config)
    raw_before = verify_raw_manifest(config.raw_dir, config.baseline_manifest)
    if not raw_before["unchanged"]:
        raise RuntimeError(f"raw baseline verification failed: {raw_before}")
    if raw_before["actual_count"] != config.expected_raw_count:
        raise ValueError(
            "expected raw count mismatch: "
            f"actual_raw_count={raw_before['actual_count']}, "
            f"expected_raw_count={config.expected_raw_count}"
        )

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    config.previews_dir.mkdir(parents=True, exist_ok=True)
    preview_entries = list(config.previews_dir.iterdir())
    allowed_preview_names = {"sift_inliers.png"}
    unexpected_preview_entries = [
        entry
        for entry in preview_entries
        if entry.is_dir()
        or not (
            entry.name in allowed_preview_names
            or entry.name.startswith("before_after_")
            and entry.suffix.lower() == ".jpg"
            or entry.name.startswith("decisions_")
            and entry.suffix.lower() == ".jpg"
        )
    ]
    if unexpected_preview_entries:
        raise ValueError(
            f"unexpected file in preview output: {unexpected_preview_entries[0]}"
        )
    for stale in preview_entries:
        stale.unlink()

    raw_paths = sorted(
        path
        for path in config.raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    )
    records: list[QualityRecord] = []
    for index, path in enumerate(raw_paths, start=1):
        records.append(
            analyze_image(path, index=index, analysis_width=config.analysis_width)
        )
        if index % 25 == 0 or index == len(raw_paths):
            print(f"Quality analysis: {index}/{len(raw_paths)}", flush=True)

    thresholds = calibrate_thresholds(records)
    quality_rows: list[dict[str, Any]] = []
    for record in records:
        decision = decide_quality(record, thresholds)
        quality_rows.append(
            {
                **asdict(record),
                "decision": decision.label,
                "reasons": "; ".join(decision.reasons),
                "selected": decision.label != "REJECT",
            }
        )

    records_by_index = {record.index: record for record in records}
    comparisons: list[dict[str, Any]] = []
    for pair_number, (first_index, second_index) in enumerate(
        config.representative_pairs, start=1
    ):
        try:
            first_record = records_by_index[first_index]
            second_record = records_by_index[second_index]
        except KeyError as error:
            raise ValueError(f"representative pair index is missing: {error}") from error
        comparison = compare_sift_pair(
            config.raw_dir / first_record.filename,
            config.raw_dir / second_record.filename,
        )
        comparison["pair_number"] = pair_number
        comparison["index_1"] = first_index
        comparison["index_2"] = second_index
        comparisons.append(comparison)
        print(
            f"SIFT comparison: {pair_number}/{len(config.representative_pairs)}",
            flush=True,
        )

    variant = choose_reconstruction_variant(comparisons)
    selected_rows = [row for row in quality_rows if bool(row["selected"])]
    selected_filenames = [str(row["filename"]) for row in selected_rows]
    selection_manifest = export_selected_images(
        config.raw_dir,
        config.output_dir,
        selected_filenames,
        variant=variant,
    )

    record_by_filename = {record.filename: record for record in records}
    decisions_by_filename = {
        str(row["filename"]): row for row in quality_rows
    }
    for entry in selection_manifest:
        record = record_by_filename[str(entry["filename"])]
        if (entry["width"], entry["height"]) != (record.width, record.height):
            raise ValueError(f"geometry mismatch in selected output: {entry['filename']}")
        row = decisions_by_filename[str(entry["filename"])]
        entry["decision"] = row["decision"]
        entry["reasons"] = row["reasons"]
        if variant == "RAW" and entry["sha256"] != _sha256(
            config.raw_dir / str(entry["filename"])
        ):
            raise ValueError(f"RAW export changed bytes: {entry['filename']}")

    hashes: dict[str, list[str]] = defaultdict(list)
    for entry in selection_manifest:
        hashes[str(entry["sha256"])].append(str(entry["filename"]))
    duplicate_output_groups = [
        names for names in hashes.values() if len(names) > 1
    ]
    if duplicate_output_groups:
        raise ValueError(f"duplicate selected outputs: {duplicate_output_groups}")

    output_count = len(
        [path for path in config.output_dir.iterdir() if path.is_file()]
    )
    validate_run_counts(
        expected_raw_count=config.expected_raw_count,
        actual_raw_count=len(raw_paths),
        quality_row_count=len(quality_rows),
        selected_count=len(selected_rows),
        selection_manifest_count=len(selection_manifest),
        output_count=output_count,
    )

    raw_after = verify_raw_manifest(config.raw_dir, config.baseline_manifest)
    if not raw_after["unchanged"]:
        raise RuntimeError(f"raw post-run verification failed: {raw_after}")

    _write_csv(config.reports_dir / "quality_decisions.csv", quality_rows)
    _write_json(config.reports_dir / "quality_thresholds.json", asdict(thresholds))
    _write_csv(config.reports_dir / "sift_matching.csv", comparisons)
    raw_total = sum(int(row["raw_inliers"]) for row in comparisons)
    preprocessed_total = sum(
        int(row["preprocessed_inliers"]) for row in comparisons
    )
    non_worse_pairs = sum(
        int(row["preprocessed_inliers"]) >= int(row["raw_inliers"])
        for row in comparisons
    )
    sift_report = {
        "method": {
            "detector_descriptor": "OpenCV SIFT, up to 8000 features",
            "analysis_maximum_width": 1200,
            "matcher": "brute-force L2, k-nearest neighbors with k=2",
            "ratio_test": 0.75,
            "geometric_verification": (
                "fundamental matrix, RANSAC, 1.5 px threshold, 0.99 confidence"
            ),
            "preprocessing": "LAB luminance CLAHE blended at 15%; no geometry change",
            "jpeg_quality": 95,
            "comparison_artifact": (
                "decoded quality-95 JPEG bytes identical to final export encoding"
            ),
        },
        "comparisons": comparisons,
        "aggregate": {
            "pair_count": len(comparisons),
            "raw_inliers_total": raw_total,
            "preprocessed_inliers_total": preprocessed_total,
            "preprocessed_non_worse_pair_count": non_worse_pairs,
        },
        "selection_rule": (
            "Choose PREPROCESSED only if total RANSAC inliers are greater and "
            "it is non-worse on at least half of tested pairs; otherwise choose RAW."
        ),
        "selected_variant": variant,
        "interpretation": (
            f"{variant} was selected: RAW produced {raw_total} total verified "
            f"inliers; PREPROCESSED produced {preprocessed_total}; preprocessing "
            f"was non-worse on {non_worse_pairs} of {len(comparisons)} pairs."
        ),
    }
    _write_json(config.reports_dir / "sift_matching.json", sift_report)
    _write_csv(config.reports_dir / "selection_manifest.csv", selection_manifest)
    _write_json(config.reports_dir / "raw_verification_after.json", raw_after)

    preview_indices = [first for first, _ in config.representative_pairs]
    for index in preview_indices:
        record = records_by_index[index]
        _save_before_after_preview(
            config.raw_dir / record.filename,
            config.previews_dir
            / f"before_after_{index:03d}_{Path(record.filename).stem}.jpg",
            index=index,
        )
    decision_sheet_names = _save_decision_sheets(
        config.raw_dir, quality_rows, config.previews_dir
    )
    _save_sift_chart(comparisons, config.previews_dir / "sift_inliers.png")
    _write_input_readme(
        config.output_dir,
        variant=variant,
        selected_count=len(selected_rows),
        raw_count=len(raw_paths),
    )

    decision_counts = Counter(str(row["decision"]) for row in quality_rows)
    summary = {
        "raw_count": len(raw_paths),
        "quality_row_count": len(quality_rows),
        "accept_count": decision_counts["ACCEPT"],
        "warn_count": decision_counts["WARN"],
        "rejected_count": decision_counts["REJECT"],
        "selected_count": len(selected_rows),
        "selected_output_count": output_count,
        "selected_variant": variant,
        "representative_pair_count": len(comparisons),
        "raw_inliers_total": raw_total,
        "preprocessed_inliers_total": preprocessed_total,
        "raw_unchanged": raw_after["unchanged"],
        "raw_mismatch_count": raw_after["mismatch_count"],
        "duplicate_output_groups": duplicate_output_groups,
        "before_after_preview_count": len(preview_indices),
        "decision_sheet_files": decision_sheet_names,
        "pycolmap_started": False,
    }
    _write_json(config.reports_dir / "preprocessing_summary.json", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run verified QA/preprocessing and prepare pyCOLMAP inputs."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("IMG20260826122949"))
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=Path("preprocessing/reports/raw_manifest_before.json"),
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=Path("preprocessing/reports")
    )
    parser.add_argument(
        "--previews-dir",
        type=Path,
        default=Path("preprocessing/previews/final"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preprocessing/pycolmap_input/images"),
    )
    parser.add_argument("--expected-count", type=int, default=297)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = PipelineConfig(
        raw_dir=args.raw_dir,
        baseline_manifest=args.baseline_manifest,
        reports_dir=args.reports_dir,
        previews_dir=args.previews_dir,
        output_dir=args.output_dir,
        expected_raw_count=args.expected_count,
    )
    summary = run_pipeline(config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
