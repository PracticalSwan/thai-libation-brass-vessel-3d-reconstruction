"""Verified access to the selected preprocessing image set."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2


_MANIFEST_FIELDS = {
    "filename",
    "variant",
    "width",
    "height",
    "size_bytes",
    "sha256",
    "decision",
    "reasons",
}


@dataclass(frozen=True)
class SelectedImageRecord:
    """One selected image in canonical manifest order."""

    index: int
    filename: str
    variant: str
    width: int
    height: int
    size_bytes: int
    sha256: str
    decision: str
    reasons: str


@dataclass(frozen=True)
class VerifiedSelectedSet:
    """A selected-image directory that matches its manifest."""

    records: tuple[SelectedImageRecord, ...]
    manifest_sha256: str
    images_dir: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_integer(value: str, *, field: str, row_number: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"invalid {field} in selection manifest row {row_number}"
        ) from error
    if parsed <= 0:
        raise ValueError(f"invalid {field} in selection manifest row {row_number}")
    return parsed


def load_selected_manifest(
    manifest_path: Path,
) -> tuple[SelectedImageRecord, ...]:
    """Parse selected rows without changing their canonical CSV order."""
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing_fields = sorted(_MANIFEST_FIELDS - fields)
        if missing_fields:
            raise ValueError(
                "selection manifest missing fields: " + ", ".join(missing_fields)
            )

        records: list[SelectedImageRecord] = []
        seen_filenames: set[str] = set()
        for index, row in enumerate(reader, start=1):
            filename = str(row["filename"])
            if Path(filename).name != filename or filename in {"", ".", ".."}:
                raise ValueError(
                    f"selection manifest filename must be a plain filename: {filename}"
                )
            if filename in seen_filenames:
                raise ValueError(f"selection manifest duplicate filename: {filename}")
            seen_filenames.add(filename)

            sha256 = str(row["sha256"]).lower()
            if len(sha256) != 64 or any(
                character not in "0123456789abcdef" for character in sha256
            ):
                raise ValueError(
                    f"invalid sha256 in selection manifest row {index}"
                )

            records.append(
                SelectedImageRecord(
                    index=index,
                    filename=filename,
                    variant=str(row["variant"]),
                    width=_positive_integer(
                        str(row["width"]), field="width", row_number=index
                    ),
                    height=_positive_integer(
                        str(row["height"]), field="height", row_number=index
                    ),
                    size_bytes=_positive_integer(
                        str(row["size_bytes"]),
                        field="size_bytes",
                        row_number=index,
                    ),
                    sha256=sha256,
                    decision=str(row["decision"]),
                    reasons=str(row["reasons"]),
                )
            )
    return tuple(records)


def path_for_index(
    records: Sequence[SelectedImageRecord],
    images_dir: Path,
    one_based_index: int,
) -> Path:
    """Resolve a selected image by canonical one-based manifest index."""
    if one_based_index < 1 or one_based_index > len(records):
        raise IndexError(
            "one-based selected-image index out of range: "
            f"{one_based_index}; valid range is 1..{len(records)}"
        )
    return images_dir / records[one_based_index - 1].filename


def verify_selected_record(
    images_dir: Path,
    record: SelectedImageRecord,
    *,
    verify_hash: bool = True,
) -> Path:
    """Verify one selected file against its parsed manifest record."""
    path = images_dir / record.filename
    if not path.is_file():
        raise ValueError(f"missing selected image: {record.filename}")
    if path.stat().st_size != record.size_bytes:
        raise ValueError(f"selected image size mismatch: {record.filename}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unreadable selected image: {record.filename}")
    height, width = image.shape[:2]
    if (width, height) != (record.width, record.height):
        raise ValueError(
            "selected image dimension mismatch: "
            f"{record.filename}; expected {record.width}x{record.height}, "
            f"found {width}x{height}"
        )
    if verify_hash and _sha256(path) != record.sha256:
        raise ValueError(f"selected image hash mismatch: {record.filename}")
    return path


def verify_selected_images(
    images_dir: Path,
    manifest_path: Path,
    *,
    expected_count: int | None = None,
) -> VerifiedSelectedSet:
    """Verify a complete selected directory before analysis writes outputs."""
    records = load_selected_manifest(manifest_path)
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(
            f"expected {expected_count} selected records, found {len(records)}"
        )
    if not images_dir.is_dir():
        raise ValueError(f"selected image directory is missing: {images_dir}")

    expected_names = {record.filename for record in records}
    actual_entries = {entry.name: entry for entry in images_dir.iterdir()}
    missing_names = sorted(expected_names - set(actual_entries))
    if missing_names:
        raise ValueError(f"missing selected image: {missing_names[0]}")
    unexpected_names = sorted(set(actual_entries) - expected_names)
    if unexpected_names:
        raise ValueError(f"unexpected selected image: {unexpected_names[0]}")

    for record in records:
        verify_selected_record(images_dir, record)

    return VerifiedSelectedSet(
        records=records,
        manifest_sha256=_sha256(manifest_path),
        images_dir=images_dir.resolve(),
    )
