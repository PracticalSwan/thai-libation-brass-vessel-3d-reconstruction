"""Step 9D read-only camera/EXIF readiness analysis."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image


_TAG_MAKE = 271
_TAG_MODEL = 272
_TAG_ORIENTATION = 274
_TAG_EXIF_IFD = 34665
_TAG_DATETIME_ORIGINAL = 36867
_TAG_FOCAL_LENGTH = 37386
_TAG_DIGITAL_ZOOM_RATIO = 41988
_TAG_FOCAL_LENGTH_35MM = 41989
_TAG_LENS_MODEL = 42036


@dataclass(frozen=True)
class CameraRecord:
    selected_index: int
    filename: str
    width: int
    height: int
    orientation: int | None
    make: str | None
    model: str | None
    lens_model: str | None
    focal_length_mm: float | None
    focal_length_35mm: float | None
    digital_zoom_ratio: float | None
    datetime_original: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _string_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8", errors="replace")
        except Exception:
            text = repr(value)
    else:
        text = str(value)
    text = text.strip()
    return text or None


def _float_value(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) else None


def _int_value(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _exif_value(top_level: object, nested: dict[int, object], tag: int) -> object:
    if tag in nested:
        return nested[tag]
    try:
        return top_level.get(tag)
    except AttributeError:
        return None


def read_camera_record(selected_index: int, raw_path: Path) -> CameraRecord:
    """Read camera metadata without modifying the raw JPEG."""
    if selected_index < 1:
        raise ValueError("selected_index must be positive")
    if not raw_path.is_file():
        raise ValueError(f"raw camera source is missing: {raw_path}")
    with Image.open(raw_path) as image:
        width, height = image.size
        exif = image.getexif()
        try:
            nested = dict(exif.get_ifd(_TAG_EXIF_IFD)) if exif else {}
        except (KeyError, TypeError, ValueError):
            nested = {}
        return CameraRecord(
            selected_index=selected_index,
            filename=raw_path.name,
            width=int(width),
            height=int(height),
            orientation=_int_value(_exif_value(exif, nested, _TAG_ORIENTATION)),
            make=_string_value(_exif_value(exif, nested, _TAG_MAKE)),
            model=_string_value(_exif_value(exif, nested, _TAG_MODEL)),
            lens_model=_string_value(_exif_value(exif, nested, _TAG_LENS_MODEL)),
            focal_length_mm=_float_value(
                _exif_value(exif, nested, _TAG_FOCAL_LENGTH)
            ),
            focal_length_35mm=_float_value(
                _exif_value(exif, nested, _TAG_FOCAL_LENGTH_35MM)
            ),
            digital_zoom_ratio=_float_value(
                _exif_value(exif, nested, _TAG_DIGITAL_ZOOM_RATIO)
            ),
            datetime_original=_string_value(
                _exif_value(exif, nested, _TAG_DATETIME_ORIGINAL)
            ),
        )


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def camera_signature(record: CameraRecord) -> tuple[object, ...]:
    return (
        record.width,
        record.height,
        record.orientation,
        record.make,
        record.model,
        record.lens_model,
        _rounded(record.focal_length_mm),
        _rounded(record.focal_length_35mm),
        _rounded(record.digital_zoom_ratio),
    )


def summarize_camera_records(
    records: Sequence[CameraRecord],
) -> dict[str, object]:
    if not records:
        raise ValueError("camera readiness requires at least one record")
    signatures = Counter(camera_signature(record) for record in records)
    field_names = (
        "orientation",
        "make",
        "model",
        "lens_model",
        "focal_length_mm",
        "focal_length_35mm",
        "digital_zoom_ratio",
        "datetime_original",
    )
    missing_fields = {
        field: sum(getattr(record, field) is None for record in records)
        for field in field_names
    }
    signature_groups: list[dict[str, object]] = []
    for signature, count in sorted(
        signatures.items(), key=lambda item: (-item[1], repr(item[0]))
    ):
        signature_groups.append(
            {
                "width": signature[0],
                "height": signature[1],
                "orientation": signature[2],
                "make": signature[3],
                "model": signature[4],
                "lens_model": signature[5],
                "focal_length_mm": signature[6],
                "focal_length_35mm": signature[7],
                "digital_zoom_ratio": signature[8],
                "frame_count": count,
            }
        )
    geometry_signatures = {
        (record.width, record.height, record.orientation) for record in records
    }
    recommendation = (
        "shared_intrinsics_single_camera_signature"
        if len(signatures) == 1
        else "separate_intrinsics_by_camera_signature"
    )
    return {
        "record_count": len(records),
        "unique_signature_count": len(signatures),
        "geometry_orientation_group_count": len(geometry_signatures),
        "geometry_orientation_consistent": len(geometry_signatures) == 1,
        "missing_fields": missing_fields,
        "signature_groups": signature_groups,
        "camera_group_recommendation": recommendation,
    }
