"""Compute deterministic appearance parameters from the eligible V4 images.

This helper deliberately does not localize or project an appearance image onto
the reconstructed surface.  It only computes robust aggregate colour and
roughness statistics from all 158 uncoated appearance-reference images, so the
Blender material remains reproducible and its provenance is explicit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def compute(manifest_path: Path, source_root: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    source_root = source_root.resolve()
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    appearance_rows = [row for row in rows if row.get("source_role") == "appearance_reference"]
    if len(appearance_rows) != 158:
        raise ValueError(f"expected exactly 158 appearance references, found {len(appearance_rows)}")

    selected: list[np.ndarray] = []
    image_records: list[dict[str, Any]] = []
    for row in appearance_rows:
        relative_path = str(row["relative_path"])
        path = source_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB").resize((192, 256)), dtype=np.float32) / 255.0
        # Fixed central crop and deterministic stride.  The colour predicate
        # rejects white/green board background while retaining warm uncoated
        # brass pixels; it is applied identically to every source image.
        flat = rgb[24:232, 24:168].reshape(-1, 3)
        luminance = flat.mean(axis=1)
        mask = (
            (luminance > 0.08)
            & (luminance < 0.88)
            & (flat[:, 0] >= flat[:, 1] - 0.02)
            & (flat[:, 1] >= flat[:, 2] - 0.02)
            & ((flat[:, 0] - flat[:, 2]) > 0.015)
        )
        pixels = flat[mask][::4]
        if len(pixels) == 0:
            raise ValueError(f"appearance colour predicate selected no pixels: {relative_path}")
        selected.append(pixels.astype(np.float64))
        image_records.append(
            {
                "relative_path": relative_path,
                "manifest_sha256": str(row.get("sha256", "")).lower(),
                "selected_pixel_count": int(len(pixels)),
            }
        )

    pixels = np.concatenate(selected, axis=0)
    srgb_median = np.median(pixels, axis=0)
    luminance = pixels.mean(axis=1)
    luminance_std = float(np.std(luminance))
    roughness = float(np.clip(0.25 + 0.5 * luminance_std, 0.20, 0.60))
    return {
        "schema_version": 1,
        "source_role": "appearance_reference",
        "photographic_projection_verified": False,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_root": str(source_root),
        "image_count": len(image_records),
        "image_records": image_records,
        "sampling": {
            "resize": [192, 256],
            "central_crop": [24, 232, 24, 168],
            "pixel_stride_after_filter": 4,
            "predicate": "0.08 < mean(RGB) < 0.88; R >= G-0.02; G >= B-0.02; R-B > 0.015",
        },
        "selected_pixel_count": int(len(pixels)),
        "base_color_srgb_median": [float(value) for value in srgb_median],
        "base_color_linear_median": [float(value) for value in _srgb_to_linear(srgb_median)],
        "luminance_median": float(np.median(luminance)),
        "luminance_std": luminance_std,
        "roughness": roughness,
        "metallic": 0.85,
        "metallic_provenance": "fixed brass-conductor prior; no photographic projection claimed",
        "roughness_provenance": "0.25 + 0.5 * selected-pixel luminance standard deviation, clamped to [0.20, 0.60]",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compute(args.manifest, args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "image_count": report["image_count"], "selected_pixel_count": report["selected_pixel_count"], "base_color_srgb_median": report["base_color_srgb_median"], "roughness": report["roughness"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
