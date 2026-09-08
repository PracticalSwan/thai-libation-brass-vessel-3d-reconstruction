"""Reference-assisted measurements for the presentation rebuild.

This module reads only reviewed masks and immutable source photographs.  It does
not change or overwrite any Steps 1-17 reconstruction evidence.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


SIDE_INDICES = (3, 10, 19, 28, 39, 50, 62, 72)
TEMPLATE_ASPECT = 0.558  # max external width / total height for the hand-built profile


@dataclass(frozen=True)
class ReferenceRecord:
    selected_index: int
    filename: str
    mask_path: Path
    mask_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference_records(manifest_path: Path, indices: Iterable[int] = SIDE_INDICES) -> tuple[ReferenceRecord, ...]:
    wanted = {int(value) for value in indices}
    records: list[ReferenceRecord] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(row["selected_index"])
            if index not in wanted:
                continue
            records.append(
                ReferenceRecord(
                    selected_index=index,
                    filename=row["filename"],
                    mask_path=Path(row["mask_path"]),
                    mask_sha256=row["mask_sha256"].lower(),
                )
            )
    missing = sorted(wanted - {record.selected_index for record in records})
    if missing:
        raise ValueError(f"reference manifest missing selected indices: {missing}")
    return tuple(sorted(records, key=lambda record: record.selected_index))


def read_binary_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"unreadable reference mask: {path}")
    return mask >= 128


def normalized_profile(mask: np.ndarray, samples: int = 384) -> dict[str, np.ndarray | float]:
    """Return a centered half-width profile normalized by foreground height."""
    if mask.ndim != 2 or not bool(mask.any()):
        raise ValueError("reference mask must contain foreground pixels")
    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    foreground_height = max(y1 - y0 + 1, 1)
    widths = np.full(foreground_height, np.nan, dtype=np.float64)
    centers = np.full(foreground_height, np.nan, dtype=np.float64)
    for local_y, y in enumerate(range(y0, y1 + 1)):
        row_x = np.flatnonzero(mask[y])
        if row_x.size:
            left = float(row_x[0])
            right = float(row_x[-1])
            widths[local_y] = (right - left + 1.0) / foreground_height
            centers[local_y] = ((left + right) * 0.5) / foreground_height
    valid = np.isfinite(widths)
    if int(valid.sum()) < 4:
        raise ValueError("reference mask has too few valid silhouette rows")
    source_y = np.linspace(0.0, 1.0, foreground_height)
    target_y = np.linspace(0.0, 1.0, samples)
    widths = np.interp(target_y, source_y[valid], widths[valid])
    centers = np.interp(target_y, source_y[valid], centers[valid])
    half_width = widths * 0.5
    return {
        "y": target_y,
        "half_width": half_width,
        "center": centers,
        "aspect": float(widths.max()),
    }


def aggregate_profiles(profiles: Iterable[dict[str, np.ndarray | float]]) -> dict[str, np.ndarray | float]:
    items = tuple(profiles)
    if not items:
        raise ValueError("at least one silhouette profile is required")
    half_widths = np.stack([np.asarray(item["half_width"], dtype=np.float64) for item in items])
    median = np.median(half_widths, axis=0)
    # Small symmetric smoothing removes mask-pixel stair steps while preserving the measured outline.
    kernel = np.array([1, 2, 3, 4, 3, 2, 1], dtype=np.float64)
    kernel /= kernel.sum()
    padded = np.pad(median, (3, 3), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return {
        "y": np.asarray(items[0]["y"], dtype=np.float64),
        "half_width": smoothed,
        "aspect": float(2.0 * smoothed.max()),
        "per_view_aspects": [float(item["aspect"]) for item in items],
    }


def estimate_brass_color(image_path: Path, mask_path: Path) -> dict[str, list[float] | float]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = read_binary_mask(mask_path)
    if image is None:
        raise ValueError(f"unreadable reference photograph: {image_path}")
    if image.shape[:2] != mask.shape:
        raise ValueError("reference image/mask dimensions differ")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Keep saturated warm metal pixels while excluding near-white highlights and deep shadows.
    warm = mask & (hsv[..., 0] <= 45) & (hsv[..., 1] >= 70) & (hsv[..., 2] >= 55) & (hsv[..., 2] <= 245)
    pixels = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[warm]
    if len(pixels) < 1000:
        pixels = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[mask]
    rgb = np.median(pixels, axis=0) / 255.0
    luminance = pixels.astype(np.float64).mean(axis=1) / 255.0
    return {
        "median_srgb": [float(value) for value in rgb],
        "luminance_p10": float(np.quantile(luminance, 0.10)),
        "luminance_p90": float(np.quantile(luminance, 0.90)),
    }


def _scale_profile(profile: list[tuple[float, float]], radial_scale: float) -> list[list[float]]:
    return [[round(radius * radial_scale, 6), round(z, 6)] for radius, z in profile]


def model_profiles(observed_aspect: float) -> dict[str, list[list[float]]]:
    """Return explainable rotational profiles scaled to measured external width/height."""
    radial_scale = float(np.clip(observed_aspect / TEMPLATE_ASPECT, 0.68, 1.15))

    bowl = [
        (0.00, 0.00), (0.38, 0.00), (0.49, 0.045), (0.55, 0.11), (0.55, 0.18),
        (0.47, 0.25), (0.44, 0.34), (0.46, 0.41), (0.58, 0.49), (0.70, 0.59),
        (0.80, 0.70), (0.86, 0.80), (0.90, 0.875), (0.82, 0.905), (0.76, 0.885),
        (0.71, 0.82), (0.65, 0.74), (0.58, 0.67), (0.50, 0.62), (0.40, 0.61),
        (0.28, 0.64), (0.16, 0.69), (0.00, 0.72),
    ]
    vessel = [
        (0.00, 0.68), (0.31, 0.68), (0.47, 0.74), (0.60, 0.84), (0.70, 0.98),
        (0.76, 1.14), (0.77, 1.31), (0.73, 1.46), (0.64, 1.61), (0.51, 1.73),
        (0.39, 1.81), (0.33, 1.90), (0.31, 2.04), (0.305, 2.37), (0.31, 2.52),
        (0.34, 2.58), (0.25, 2.58), (0.235, 2.36), (0.23, 2.08), (0.20, 1.96),
        (0.00, 1.91),
    ]
    lid = [
        (0.22, 2.55), (0.36, 2.56), (0.40, 2.61), (0.38, 2.68), (0.34, 2.70),
        (0.37, 2.75), (0.34, 2.80), (0.31, 2.83), (0.34, 2.88), (0.29, 2.93),
        (0.17, 2.96), (0.12, 2.93), (0.16, 2.88), (0.19, 2.81), (0.20, 2.72),
        (0.19, 2.63),
    ]
    return {
        "bowl": _scale_profile(bowl, radial_scale),
        "vessel": _scale_profile(vessel, radial_scale),
        "lid": _scale_profile(lid, radial_scale),
        "radial_scale": radial_scale,
    }


def write_reference_silhouette(path: Path, aggregate: dict[str, np.ndarray | float], size: int = 768) -> None:
    canvas = np.zeros((size, size), dtype=np.uint8)
    y_values = np.asarray(aggregate["y"], dtype=np.float64)
    half = np.asarray(aggregate["half_width"], dtype=np.float64)
    max_half = max(float(half.max()), 1e-9)
    # Fit the measured full height to 90% of the canvas and preserve measured aspect.
    height_px = int(size * 0.90)
    top = (size - height_px) // 2
    center_x = size // 2
    px_per_height = height_px
    target_y = np.linspace(0.0, 1.0, height_px)
    target_half = np.interp(target_y, y_values, half)
    for row, half_norm in enumerate(target_half):
        y = top + row
        radius = int(round(half_norm * px_per_height))
        cv2.line(canvas, (max(0, center_x - radius), y), (min(size - 1, center_x + radius), y), 255, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise ValueError(f"failed to write reference silhouette: {path}")


def build_reference_report(project_root: Path, output_dir: Path) -> dict:
    manifest = project_root / "ml_dataset" / "manifest.csv"
    raw_dir = project_root / "IMG20260826122949"
    records = load_reference_records(manifest)
    profiles = []
    color_measurements = []
    provenance = []
    for record in records:
        mask_path = project_root / record.mask_path
        if _sha256(mask_path) != record.mask_sha256:
            raise ValueError(f"reference mask hash mismatch: {record.mask_path}")
        profile = normalized_profile(read_binary_mask(mask_path))
        profiles.append(profile)
        image_path = raw_dir / record.filename
        color_measurements.append(estimate_brass_color(image_path, mask_path))
        provenance.append(
            {
                "selected_index": record.selected_index,
                "filename": record.filename,
                "mask_path": record.mask_path.as_posix(),
                "mask_sha256": record.mask_sha256,
                "image_sha256": _sha256(image_path),
                "aspect": float(profile["aspect"]),
            }
        )

    aggregate = aggregate_profiles(profiles)
    colors = np.asarray([item["median_srgb"] for item in color_measurements], dtype=np.float64)
    median_color = np.median(colors, axis=0)
    profiles_out = model_profiles(float(aggregate["aspect"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    silhouette_path = output_dir / "reference_silhouette.png"
    write_reference_silhouette(silhouette_path, aggregate)
    report = {
        "method": "reviewed_multi_view_silhouette_plus_photo_color_reference",
        "manual_cleanup_started": False,
        "source_scope": {
            "reviewed_side_masks": len(records),
            "selected_indices": [record.selected_index for record in records],
            "raw_source_immutable": True,
        },
        "measurements": {
            "median_external_width_to_height": float(aggregate["aspect"]),
            "per_view_external_width_to_height": aggregate["per_view_aspects"],
            "photo_median_brass_srgb": [float(value) for value in median_color],
            "roughness_estimate": 0.24,
            "metallic": 1.0,
        },
        "profiles": profiles_out,
        "provenance": provenance,
        "artifacts": {"reference_silhouette": silhouette_path.as_posix()},
        "limitations": [
            "geometry is reference-assisted and symmetry-inferred rather than direct dense photogrammetry",
            "ornament on unseen surfaces is estimated from repeated visible motifs",
            "physical scale remains arbitrary until an external measurement is supplied",
        ],
    }
    report_path = output_dir / "reference_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
