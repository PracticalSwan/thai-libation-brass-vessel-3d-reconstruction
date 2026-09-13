"""Deterministic ingest and source-sequence modelling for the V4 photographs.

The manifest is the first reconstruction artifact.  It accounts for every
incoming JPEG exactly once, keeps source provenance even when a view is not
selected for geometry, and makes the G7 repeated-revolution decision explicit.
No source image is ever rewritten by this module.
"""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageFilter

try:  # OpenCV is optional for the manifest fallback, but preferred for Laplacian QA.
    import cv2
except ImportError:  # pragma: no cover - exercised only in a minimal environment
    cv2 = None

from v4_config import (
    CAPTURE_V4_ROOT,
    V4_SOURCE_ROOT,
    assert_output_path,
    ensure_v4_directories,
    fingerprint,
    sha256_file,
    utc_now,
    write_json,
)


EXPECTED_COUNTS = {
    "total": 688,
    "appearance_reference": 158,
    "empty_board": 107,
    "geometry": 423,
}

EXPECTED_DIMENSIONS = (3072, 4080)


@dataclass(frozen=True)
class RoleRange:
    role: str
    start: str
    end: str
    source_pass_id: str = ""

    @property
    def start_datetime(self) -> datetime:
        return datetime.strptime(self.start, "%Y%m%d%H%M%S")

    @property
    def end_datetime(self) -> datetime:
        return datetime.strptime(self.end, "%Y%m%d%H%M%S")


ROLE_RANGES: tuple[RoleRange, ...] = (
    RoleRange("appearance_reference", "20260912131808", "20260912133317", "appearance"),
    RoleRange("empty_board", "20260912133528", "20260912133652", "empty_g6"),
    RoleRange("geometry", "20260912141912", "20260912142404", "geo_g7"),
    RoleRange("geometry", "20260912143336", "20260912143718", "geo_g8"),
    RoleRange("empty_board", "20260912143726", "20260912143825", "empty_g8"),
    RoleRange("geometry", "20260912144413", "20260912144629", "geo_g9"),
    RoleRange("empty_board", "20260912144636", "20260912144719", "empty_g9"),
    RoleRange("geometry", "20260912145120", "20260912145347", "geo_g10"),
    RoleRange("geometry", "20260912150136", "20260912150348", "geo_g11"),
    RoleRange("geometry", "20260912150514", "20260912150627", "geo_g12"),
)

ELEVATION_ROLES = {
    "geo_g7": "lower_horizontal",
    "geo_g8": "lower_mid",
    "geo_g9": "mid",
    "geo_g10": "mid_high",
    "geo_g11": "high",
    "geo_g12": "steep_high_top_interior",
}

FILENAME_RE = re.compile(r"^IMG(?P<stamp>\d{14})(?:_(?P<suffix>\d+))?\.(?P<ext>jpe?g)$", re.IGNORECASE)

MANIFEST_FIELDS = (
    "relative_path",
    "sha256",
    "byte_size",
    "width",
    "height",
    "format",
    "orientation",
    "make",
    "model",
    "lens_model",
    "focal_length_mm",
    "focal_length_35mm_eq",
    "f_number",
    "iso",
    "exposure_time",
    "exposure_compensation",
    "digital_zoom_ratio",
    "white_balance",
    "exposure_program",
    "datetime_original",
    "source_cluster",
    "source_role",
    "source_pass_id",
    "logical_ring_id",
    "selected_for_geometry",
    "selection_reason",
    "near_duplicate_group",
    "frame_index",
    "frame_index_within_logical_ring",
    "phase_01",
    "angle_deg",
    "rotation_direction",
    "camera_group",
    "decode_ok",
    "sharpness_laplacian",
    "mean_luminance",
    "p01_luminance",
    "p99_luminance",
    "clipping_fraction",
    "contrast_std",
    "object_presence_signal",
    "object_presence_sanity",
)


def filename_order(name: str | Path) -> tuple[datetime, int, str]:
    """Return deterministic timestamp/suffix order for an audited filename."""

    base = Path(name).name
    match = FILENAME_RE.match(base)
    if not match:
        raise ValueError(f"unsupported V4 filename: {base}")
    stamp = datetime.strptime(match.group("stamp"), "%Y%m%d%H%M%S")
    suffix = int(match.group("suffix") or 0)
    return stamp, suffix, base.lower()


def _stamp_for_name(name: str | Path) -> datetime:
    return filename_order(name)[0]


def role_for_filename(name: str | Path) -> dict[str, str] | None:
    """Resolve one audited role/pass from the filename timestamp."""

    stamp = _stamp_for_name(name)
    matches = [
        item
        for item in ROLE_RANGES
        if item.start_datetime <= stamp <= item.end_datetime
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"overlapping audited role ranges for {Path(name).name}")
    item = matches[0]
    return {
        "source_role": item.role,
        "source_pass_id": item.source_pass_id,
        "source_cluster": item.source_pass_id,
    }


def audited_role_for_filename(name: str | Path) -> tuple[str, str]:
    """Strict role helper used by tests and the canonical ingest gate."""

    role = role_for_filename(name)
    if role is None:
        raise ValueError(f"filename is outside the audited V4 ranges: {Path(name).name}")
    return role["source_role"], role["source_pass_id"]


def _safe_float(value: Any, default: str = "") -> str:
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return default
    return f"{number:g}"


def _exif_value(exif: Mapping[Any, Any], name: str, default: Any = "") -> Any:
    # PIL exposes numeric EXIF tags by integer IDs.  The small mapping keeps
    # tests and fallback ingest independent of a particular ExifTool install.
    ids = {
        "orientation": 274,
        "make": 271,
        "model": 272,
        "lens_model": 42036,
        "focal_length_mm": 37386,
        "focal_length_35mm_eq": 41989,
        "f_number": 33437,
        "iso": 34855,
        "exposure_time": 33434,
        "exposure_compensation": 37380,
        "digital_zoom_ratio": 41988,
        "white_balance": 41984,
        "exposure_program": 34850,
        "datetime_original": 36867,
    }
    return exif.get(ids.get(name, name), default)


def _pil_metadata(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            return {
                "ImageWidth": int(image.width),
                "ImageHeight": int(image.height),
                "Orientation": _exif_value(exif, "orientation", 1),
                "Make": _exif_value(exif, "make"),
                "Model": _exif_value(exif, "model"),
                "LensModel": _exif_value(exif, "lens_model"),
                "FocalLength": _safe_float(_exif_value(exif, "focal_length_mm")),
                "FocalLengthIn35mmFormat": _safe_float(_exif_value(exif, "focal_length_35mm_eq")),
                "FNumber": _safe_float(_exif_value(exif, "f_number")),
                "ISO": _safe_float(_exif_value(exif, "iso")),
                "ExposureTime": _safe_float(_exif_value(exif, "exposure_time")),
                "ExposureCompensation": _safe_float(_exif_value(exif, "exposure_compensation")),
                "DigitalZoomRatio": _safe_float(_exif_value(exif, "digital_zoom_ratio")),
                "WhiteBalance": _exif_value(exif, "white_balance"),
                "ExposureProgram": _exif_value(exif, "exposure_program"),
                "DateTimeOriginal": _exif_value(exif, "datetime_original"),
            }
    except Exception as error:
        return {"_error": f"{type(error).__name__}: {error}"}


def _exiftool_metadata(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    executable = shutil.which("exiftool")
    if not executable or not paths:
        return {}
    try:
        completed = subprocess.run(
            [executable, "-j", "-n", "-charset", "filename=UTF8", *map(str, paths)],
            capture_output=True,
            text=True,
            timeout=max(120, len(paths) // 2),
            check=False,
        )
        if completed.returncode != 0:
            return {}
        values = json.loads(completed.stdout)
        if not isinstance(values, list):
            return {}
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in values:
        if not isinstance(item, dict):
            continue
        source = item.get("SourceFile")
        if source:
            result[Path(str(source)).name.lower()] = item
    return result


def _thumbnail(path: Path, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        image.thumbnail(size, Image.Resampling.BILINEAR)
        canvas = Image.new("L", size, color=0)
        left = (size[0] - image.width) // 2
        top = (size[1] - image.height) // 2
        canvas.paste(image, (left, top))
        return np.asarray(canvas, dtype=np.float32)


def _dhash(image: np.ndarray) -> int:
    # A deterministic perceptual signal for same-second/adjacent candidates;
    # it is never used as a standalone frame-rejection decision.
    small = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).resize((33, 32), Image.Resampling.BILINEAR)
    values = np.asarray(small, dtype=np.uint8)
    bits = values[:, 1:] > values[:, :-1]
    result = 0
    for bit in bits.reshape(-1):
        result = (result << 1) | int(bool(bit))
    return result


def _hamming(first: int, second: int) -> int:
    return (first ^ second).bit_count()


def _quality(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    image = _thumbnail(path)
    values = image.astype(np.float32)
    if values.size:
        mean = float(values.mean())
        p01 = float(np.quantile(values, 0.01))
        p99 = float(np.quantile(values, 0.99))
        clipping = float(np.mean((values <= 1.0) | (values >= 254.0)))
        contrast = float(values.std())
    else:  # pragma: no cover - Pillow never creates an empty thumbnail
        mean = p01 = p99 = clipping = contrast = 0.0
    if cv2 is not None:
        sharpness = float(cv2.Laplacian(values, cv2.CV_32F).var())
    else:
        # Pillow's edge filter is a deterministic fallback when OpenCV is absent.
        edge = np.asarray(Image.fromarray(np.clip(values, 0, 255).astype(np.uint8)).filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        sharpness = float(edge.var())
    border = np.concatenate((values[:16].ravel(), values[-16:].ravel(), values[:, :16].ravel(), values[:, -16:].ravel()))
    center = values[48:-48, 48:-48]
    signal = float(abs(float(center.mean()) - float(border.mean())) / (float(border.std()) + 1.0))
    return {
        "sharpness_laplacian": f"{sharpness:.8g}",
        "mean_luminance": f"{mean:.8g}",
        "p01_luminance": f"{p01:.8g}",
        "p99_luminance": f"{p99:.8g}",
        "clipping_fraction": f"{clipping:.8g}",
        "contrast_std": f"{contrast:.8g}",
        "object_presence_signal": f"{signal:.8g}",
    }, image


def _decode(path: Path) -> tuple[bool, int, int, str, str]:
    try:
        with Image.open(path) as image:
            width, height = int(image.width), int(image.height)
            image.verify()
        with Image.open(path) as image:
            image.load()
            image_format = str(image.format or "JPEG")
        return True, width, height, image_format, ""
    except Exception as error:
        return False, 0, 0, "", f"{type(error).__name__}: {error}"


def _normalized_signature(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32)
    values = (values - values.mean()) / max(float(values.std()), 1.0)
    return values[::4, ::4].reshape(-1)


def _signature_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean(np.abs(first - second)))


def detect_revolution_wrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    signatures: Sequence[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Detect a supported repeated cycle using multi-frame recurrence.

    The detector is intentionally conservative and only accepts a candidate
    when both a short neighborhood and a second neighborhood recur, each
    candidate ring is substantial, and the best score is clearly separated
    from the pass baseline.  It does not assume a 71/71 split.
    """

    names = [str(row["relative_path"]) for row in rows]
    n = len(names)
    result: dict[str, Any] = {
        "source_pass_id": str(rows[0].get("source_pass_id", "")) if rows else "",
        "frame_count": n,
        "wrap_detected": False,
        "candidate_split_index": None,
        "candidate_score": None,
        "baseline_score": None,
        "confidence": 0.0,
        "method": "ordered_thumbnail_recurrence_neighborhoods",
    }
    # Only a long pass can provide evidence for a repeated revolution.  This
    # keeps ordinary single-ring passes from being over-segmented by symmetry.
    if n < 100:
        result["reason"] = "pass_too_short_for_repeated_cycle_gate"
        return result
    if signatures is None:
        signatures = [_normalized_signature(_thumbnail(Path(name))) for name in names]
    if len(signatures) != n:
        raise ValueError("wrap signatures must match pass rows")
    min_cycle = max(30, int(round(n * 0.30)))
    low = max(min_cycle, int(round(n * 0.35)))
    high = min(n - min_cycle, int(round(n * 0.75)))
    window = max(5, min(12, n // 16))
    candidates: list[tuple[float, int]] = []
    for split in range(low, high + 1):
        if split + window >= n:
            continue
        first = [_signature_distance(signatures[index], signatures[split + index]) for index in range(window)]
        second_start = max(0, n - window)
        second = [
            _signature_distance(signatures[second_start + index], signatures[split + second_start - n + index])
            for index in range(window)
            if 0 <= split + second_start - n + index < n
        ]
        if not second:
            # Compare a later, non-overlapping neighborhood when the end-window
            # formulation cannot fit for an unequal candidate.
            second = [_signature_distance(signatures[window + index], signatures[split + window + index]) for index in range(window) if split + window + index < n]
        score = float(np.mean(first) + np.mean(second)) / 2.0
        candidates.append((score, split))
    if not candidates:
        result["reason"] = "no_valid_candidate_neighborhood"
        return result
    candidates.sort(key=lambda item: (item[0], item[1]))
    best_score, split = candidates[0]
    baseline = float(np.median([value for value, _ in candidates]))
    second_score = candidates[1][0] if len(candidates) > 1 else baseline
    separation = max(0.0, (second_score - best_score) / max(second_score, 1e-9))
    relative = best_score / max(baseline, 1e-9)
    balanced = min_cycle <= split <= n - min_cycle
    accepted = bool(balanced and relative <= 0.72 and separation >= 0.05)
    result.update(
        {
            "candidate_split_index": int(split),
            "candidate_score": best_score,
            "baseline_score": baseline,
            "confidence": float(min(1.0, max(0.0, (1.0 - relative) * 0.7 + separation * 0.3))),
            "wrap_detected": accepted,
            "candidate_ring_lengths": [int(split), int(n - split)],
            "relative_score": relative,
            "separation": separation,
        }
    )
    if not accepted:
        result["reason"] = "recurrence_not_separated_or_not_balanced"
    else:
        result["reason"] = "multi_neighborhood_recurrence_supported"
    return result


def _empty_row(name: str, frame_index: int, source_root: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    path = source_root / name
    role = role_for_filename(name)
    if role is None:
        raise ValueError(f"unclassified V4 source: {name}")
    decode_ok, width, height, image_format, decode_error = _decode(path)
    quality: dict[str, Any] = {}
    dhash = None
    signature = None
    if decode_ok:
        quality, thumb = _quality(path)
        dhash = _dhash(thumb)
        signature = _normalized_signature(thumb)
    exif = dict(metadata)
    datetime_original = str(exif.get("DateTimeOriginal", "") or "")
    if not datetime_original:
        datetime_original = _stamp_for_name(name).strftime("%Y:%m:%d %H:%M:%S")
    row: dict[str, Any] = {
        "relative_path": name,
        "sha256": sha256_file(path),
        "byte_size": path.stat().st_size,
        "width": width or exif.get("ImageWidth", ""),
        "height": height or exif.get("ImageHeight", ""),
        "format": image_format or "JPEG",
        "orientation": exif.get("Orientation", 1),
        "make": exif.get("Make", ""),
        "model": exif.get("Model", ""),
        "lens_model": exif.get("LensModel", ""),
        "focal_length_mm": exif.get("FocalLength", ""),
        "focal_length_35mm_eq": exif.get("FocalLengthIn35mmFormat", ""),
        "f_number": exif.get("FNumber", ""),
        "iso": exif.get("ISO", ""),
        "exposure_time": exif.get("ExposureTime", ""),
        "exposure_compensation": exif.get("ExposureCompensation", ""),
        "digital_zoom_ratio": exif.get("DigitalZoomRatio", ""),
        "white_balance": exif.get("WhiteBalance", ""),
        "exposure_program": exif.get("ExposureProgram", ""),
        "datetime_original": datetime_original,
        "source_cluster": role["source_cluster"],
        "source_role": role["source_role"],
        "source_pass_id": role["source_pass_id"],
        "logical_ring_id": role["source_pass_id"] if role["source_role"] == "geometry" else "",
        "selected_for_geometry": role["source_role"] == "geometry",
        "selection_reason": "primary_geometry_candidate" if role["source_role"] == "geometry" else "not_geometry_source",
        "near_duplicate_group": "",
        "frame_index": frame_index,
        "frame_index_within_logical_ring": "",
        "phase_01": "",
        "angle_deg": "",
        "rotation_direction": "forward_ordered_phase" if role["source_role"] == "geometry" else "",
        "camera_group": "v4_phone_rear_26mm" if role["source_role"] in {"geometry", "empty_board"} else "appearance_reference",
        "decode_ok": decode_ok,
        "decode_error": decode_error,
        "dhash": dhash,
        "_signature": signature,
        "_timestamp": _stamp_for_name(name),
        **quality,
        "object_presence_sanity": "observed_content_signal" if decode_ok else "decode_failed",
    }
    return row


def _assign_duplicate_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        exact[str(row["sha256"])].append(row)
    group_index = 1
    for members in exact.values():
        if len(members) > 1:
            group = f"sha_{group_index:03d}"
            group_index += 1
            for row in members:
                row["near_duplicate_group"] = group
                if row["source_role"] == "geometry":
                    row["selected_for_geometry"] = False
                    row["selection_reason"] = "byte_identical_duplicate"
    # Same-second pairs are the audited near-duplicate candidates.  Restrict
    # perceptual comparison to that small group so recurrence frames elsewhere
    # are not silently discarded.
    by_stamp: dict[datetime, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stamp[row["_timestamp"]].append(row)
    for members in by_stamp.values():
        if len(members) < 2:
            continue
        if not any(row.get("dhash") is not None for row in members):
            continue
        anchor = members[0]
        for row in members[1:]:
            if anchor.get("dhash") is None or row.get("dhash") is None:
                continue
            if _hamming(int(anchor["dhash"]), int(row["dhash"])) <= 6:
                group = str(anchor.get("near_duplicate_group") or f"near_{group_index:03d}")
                if not anchor.get("near_duplicate_group"):
                    group_index += 1
                anchor["near_duplicate_group"] = group
                row["near_duplicate_group"] = group
                if anchor["source_role"] == "geometry" and row["source_role"] == "geometry":
                    # Higher edge variance is the only tie-breaker; both source
                    # hashes and provenance remain in the manifest.
                    first_sharp = float(anchor.get("sharpness_laplacian") or 0.0)
                    second_sharp = float(row.get("sharpness_laplacian") or 0.0)
                    keep, drop = (anchor, row) if first_sharp >= second_sharp else (row, anchor)
                    drop["selected_for_geometry"] = False
                    drop["selection_reason"] = "near_duplicate_lower_sharpness"
                    keep["selected_for_geometry"] = True
                    keep["selection_reason"] = "near_duplicate_sharper_member"
    return rows


def _resolve_logical_rings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["source_role"] == "geometry":
            passes[str(row["source_pass_id"])].append(row)
    wraps: dict[str, Any] = {}
    for source_pass, members in passes.items():
        members.sort(key=lambda row: filename_order(str(row["relative_path"])))
        signatures = [row["_signature"] for row in members]
        wrap = detect_revolution_wrap(members, signatures=signatures)
        wraps[source_pass] = wrap
        if wrap.get("wrap_detected") and source_pass == "geo_g7":
            split = int(wrap["candidate_split_index"])
            first, second = members[:split], members[split:]
            first_id, second_id = "g7_r1", "g7_r2"
            first_sharp = float(np.median([float(row.get("sharpness_laplacian") or 0.0) for row in first]))
            second_sharp = float(np.median([float(row.get("sharpness_laplacian") or 0.0) for row in second]))
            # A short tail after a strong recurrence is a partial repeated cycle,
            # not a second complete ring.  Keep it as provenance but use the
            # supported first cycle as the geometry primary.  Only near-balanced
            # cycles are ranked by sharpness.
            if len(second) < int(round(len(first) * 0.85)):
                primary_id = first_id
                wrap["cycle_classification"] = "primary_complete_plus_partial_tail"
            else:
                primary_id = first_id if first_sharp >= second_sharp else second_id
                wrap["cycle_classification"] = "two_substantial_repeated_cycles"
            for ring_id, ring_rows in ((first_id, first), (second_id, second)):
                for row in ring_rows:
                    row["logical_ring_id"] = ring_id
                    if ring_id != primary_id and row["selected_for_geometry"]:
                        row["selected_for_geometry"] = False
                        row["selection_reason"] = "partial_repeated_cycle_secondary_ring"
            wrap["logical_ring_ids"] = [first_id, second_id]
            wrap["primary_ring_id"] = primary_id
            wrap["primary_selection_basis"] = "median_sharpness"
        else:
            for row in members:
                row["logical_ring_id"] = source_pass
    ring_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ring_id = str(row.get("logical_ring_id") or "")
        if row["source_role"] == "geometry" and ring_id:
            ring_members[ring_id].append(row)
    for ring_id, members in ring_members.items():
        members.sort(key=lambda row: filename_order(str(row["relative_path"])))
        phase_by_group: dict[str, float] = {}
        for index, row in enumerate(members):
            duplicate_group = str(row.get("near_duplicate_group") or "")
            key = duplicate_group or f"frame_{index}"
            if key not in phase_by_group:
                phase_by_group[key] = index / max(len(members), 1)
            row["frame_index_within_logical_ring"] = index
            row["phase_01"] = f"{phase_by_group[key]:.12f}"
            row["angle_deg"] = ""
            row["elevation_role"] = ELEVATION_ROLES.get(
                str(row["source_pass_id"]), "unknown"
            )
    return wraps


def _public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in MANIFEST_FIELDS}


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    target = assert_output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def role_totals(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    totals = defaultdict(int)
    for row in rows:
        totals[str(row["source_role"])] += 1
    return dict(sorted(totals.items()))


def _validate_roles(rows: Sequence[Mapping[str, Any]], *, strict: bool) -> None:
    totals = role_totals(rows)
    expected = {
        "appearance_reference": EXPECTED_COUNTS["appearance_reference"],
        "empty_board": EXPECTED_COUNTS["empty_board"],
        "geometry": EXPECTED_COUNTS["geometry"],
    }
    if strict and totals != expected:
        raise ValueError(f"V4 role accounting mismatch: expected {expected}, found {totals}")
    if strict and len(rows) != EXPECTED_COUNTS["total"]:
        raise ValueError(f"V4 source count mismatch: expected {EXPECTED_COUNTS['total']}, found {len(rows)}")


def build_manifest(
    source_root: Path = V4_SOURCE_ROOT,
    *,
    output_root: Path | None = CAPTURE_V4_ROOT,
    require_expected_count: bool = True,
    require_expected_dimensions: bool = True,
) -> dict[str, Any]:
    """Build the deterministic V4 manifest and sequence model.

    Set the two ``require_*`` flags to ``False`` only for small unit fixtures;
    the canonical run keeps both gates enabled.
    """

    source_root = Path(source_root).resolve()
    if not source_root.is_dir():
        raise ValueError(f"V4 source directory is missing: {source_root}")
    paths = sorted(
        [path for path in source_root.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}],
        key=lambda path: filename_order(path.name),
    )
    if require_expected_count and len(paths) != EXPECTED_COUNTS["total"]:
        raise ValueError(f"V4 source count mismatch: expected 688, found {len(paths)}")
    for path in paths:
        if role_for_filename(path.name) is None:
            raise ValueError(f"V4 source is outside audited ranges: {path.name}")
    metadata = _exiftool_metadata(paths)
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        row = _empty_row(path.name, index, source_root, metadata.get(path.name.lower(), _pil_metadata(path)))
        if require_expected_dimensions and row["decode_ok"] and (int(row["width"]) != EXPECTED_DIMENSIONS[0] or int(row["height"]) != EXPECTED_DIMENSIONS[1]):
            raise ValueError(f"V4 image dimension mismatch: {path.name} has {row['width']}x{row['height']}")
        rows.append(row)
    _validate_roles(rows, strict=require_expected_count)
    _assign_duplicate_groups(rows)
    wraps = _resolve_logical_rings(rows)
    selected_geometry = [row for row in rows if row["source_role"] == "geometry" and row["selected_for_geometry"]]
    geometry_rows = [row for row in rows if row["source_role"] == "geometry"]
    ring_summary: dict[str, Any] = {}
    # Include excluded ring members as provenance.  In particular, a partial
    # repeated G7 tail must remain visible in sequences.json even though it is
    # not admitted to the reconstruction input.
    for row in geometry_rows:
        ring = str(row["logical_ring_id"])
        ring_summary.setdefault(ring, {"source_pass_id": row["source_pass_id"], "elevation_role": row.get("elevation_role", "unknown"), "frame_count": 0, "selected_count": 0, "rotation_direction": "forward_ordered_phase", "phase_basis": "ordered_visual_phase", "angle_deg": None, "frames": []})
        ring_summary[ring]["frame_count"] += 1
        selected = bool(row["selected_for_geometry"])
        ring_summary[ring]["selected_count"] += int(selected)
        ring_summary[ring]["frames"].append({"filename": row["relative_path"], "phase_01": float(row["phase_01"]), "frame_index": row["frame_index_within_logical_ring"], "selected": selected, "selection_reason": row.get("selection_reason", "")})
    sequences = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "source_root": str(source_root),
        "source_manifest_contract": {"total": len(rows), "role_totals": role_totals(rows), "phase_is_inferred": True, "angle_deg_is_unmeasured": True},
        "logical_rings": ring_summary,
        "wrap_detection": wraps,
        "empty_board_sequences": {"empty_g6": 51, "empty_g8": 31, "empty_g9": 25},
        "g8_boundary": {"last_geometry": "IMG20260912143718.jpg", "first_empty": "IMG20260912143726.jpg"},
        "g9_boundary": {"last_geometry": "IMG20260912144629.jpg", "first_empty": "IMG20260912144636.jpg"},
        "rotation_direction_basis": "ordered timestamp phase; clockwise world direction is not measured",
    }
    exact_groups = [group for group in _group_values(rows, "sha256") if len(group) > 1]
    near_groups = [group for group in _group_values(rows, "near_duplicate_group") if group and len(group) > 1]
    audit = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "source_root": str(source_root),
        "expected_counts": EXPECTED_COUNTS,
        "observed_count": len(rows),
        "role_totals": role_totals(rows),
        "role_accounting_passed": role_totals(rows) == {key: EXPECTED_COUNTS[key] for key in ("appearance_reference", "empty_board", "geometry")},
        "dimensions": {"expected": list(EXPECTED_DIMENSIONS), "observed": sorted({f"{row['width']}x{row['height']}" for row in rows})},
        "decode_failures": [{"filename": row["relative_path"], "error": row.get("decode_error", "")} for row in rows if not row["decode_ok"]],
        "exact_duplicate_groups": [[row["relative_path"] for row in group] for group in exact_groups],
        "near_duplicate_groups": [[row["relative_path"] for row in group] for group in near_groups],
        "selected_geometry_count": len(selected_geometry),
        "wrap_detection": wraps,
        "manifest_fingerprint": fingerprint([(row["relative_path"], row["sha256"]) for row in rows]),
    }
    exclusions = [
        {"relative_path": row["relative_path"], "sha256": row["sha256"], "source_role": row["source_role"], "logical_ring_id": row.get("logical_ring_id", ""), "reason": row["selection_reason"]}
        for row in rows
        if not row["selected_for_geometry"] and row["source_role"] == "geometry"
    ]
    if output_root is not None:
        output_root = Path(output_root).resolve()
        manifest_dir = output_root / "manifests"
        for path in (manifest_dir / "source_manifest.csv", manifest_dir / "sequences.json", manifest_dir / "media_audit.json", manifest_dir / "exclusions.csv"):
            assert_output_path(path)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(manifest_dir / "source_manifest.csv", (_public_row(row) for row in rows), MANIFEST_FIELDS)
        write_json(manifest_dir / "sequences.json", sequences)
        write_json(manifest_dir / "media_audit.json", audit)
        _write_csv(manifest_dir / "exclusions.csv", exclusions, ("relative_path", "sha256", "source_role", "logical_ring_id", "reason"))
    # Internal arrays are removed from the returned JSON-compatible object.
    public_rows = [_public_row(row) for row in rows]
    return {"rows": public_rows, "sequences": sequences, "audit": audit, "exclusions": exclusions}


def _group_values(rows: Sequence[Mapping[str, Any]], field: str) -> list[list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        value = str(row.get(field) or "")
        if value:
            groups[value].append(row)
    return list(groups.values())


def build_authoritative_manifest() -> dict[str, Any]:
    return build_manifest(V4_SOURCE_ROOT, output_root=CAPTURE_V4_ROOT, require_expected_count=True, require_expected_dimensions=True)


__all__ = [
    "EXPECTED_COUNTS",
    "EXPECTED_DIMENSIONS",
    "ELEVATION_ROLES",
    "MANIFEST_FIELDS",
    "ROLE_RANGES",
    "audited_role_for_filename",
    "build_authoritative_manifest",
    "build_manifest",
    "detect_revolution_wrap",
    "filename_order",
    "role_for_filename",
    "role_totals",
]
