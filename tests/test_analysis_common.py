import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from analysis_common import (
    load_selected_manifest,
    path_for_index,
    verify_selected_images,
)


FIELDNAMES = (
    "filename",
    "variant",
    "width",
    "height",
    "size_bytes",
    "sha256",
    "decision",
    "reasons",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for name, value in (("second.jpg", 170), ("first.jpg", 90)):
        image = np.full((32, 48, 3), value, dtype=np.uint8)
        image[8:24, 12:36] = 255 - value
        path = images_dir / name
        assert cv2.imwrite(str(path), image)
        rows.append(
            {
                "filename": name,
                "variant": "PREPROCESSED",
                "width": 48,
                "height": 32,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "decision": "ACCEPT",
                "reasons": "",
            }
        )
    manifest = tmp_path / "selection_manifest.csv"
    _write_manifest(manifest, rows)
    return images_dir, manifest, rows


def test_verified_set_preserves_manifest_order_and_one_based_lookup(
    tmp_path: Path,
) -> None:
    """Catches sorting by filename or treating the selected index as zero-based."""
    images_dir, manifest, _ = _build_fixture(tmp_path)

    verified = verify_selected_images(images_dir, manifest, expected_count=2)

    assert [record.index for record in verified.records] == [1, 2]
    assert [record.filename for record in verified.records] == [
        "second.jpg",
        "first.jpg",
    ]
    assert path_for_index(verified.records, images_dir, 1) == images_dir / "second.jpg"
    assert path_for_index(verified.records, images_dir, 2) == images_dir / "first.jpg"
    with pytest.raises(IndexError, match="one-based selected-image index"):
        path_for_index(verified.records, images_dir, 0)
    with pytest.raises(IndexError, match="one-based selected-image index"):
        path_for_index(verified.records, images_dir, 3)


def test_manifest_rejects_duplicate_or_unsafe_filenames(tmp_path: Path) -> None:
    """Catches ambiguous records and paths that could escape the selected set."""
    _, manifest, rows = _build_fixture(tmp_path)
    _write_manifest(manifest, [rows[0], rows[0]])
    with pytest.raises(ValueError, match="duplicate filename"):
        load_selected_manifest(manifest)

    unsafe = dict(rows[0], filename="../outside.jpg")
    _write_manifest(manifest, [unsafe])
    with pytest.raises(ValueError, match="plain filename"):
        load_selected_manifest(manifest)


def test_verification_rejects_missing_and_extra_images(tmp_path: Path) -> None:
    """Catches incomplete or contaminated selected-image directories."""
    images_dir, manifest, _ = _build_fixture(tmp_path)
    (images_dir / "first.jpg").unlink()
    with pytest.raises(ValueError, match="missing selected image: first.jpg"):
        verify_selected_images(images_dir, manifest, expected_count=2)

    images_dir, manifest, _ = _build_fixture(tmp_path / "extra_case")
    extra = np.zeros((32, 48, 3), dtype=np.uint8)
    assert cv2.imwrite(str(images_dir / "extra.jpg"), extra)
    with pytest.raises(ValueError, match="unexpected selected image: extra.jpg"):
        verify_selected_images(images_dir, manifest, expected_count=2)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("width", 49, "dimension mismatch"),
        ("size_bytes", 1, "size mismatch"),
        ("sha256", "0" * 64, "hash mismatch"),
    ],
)
def test_verification_rejects_manifest_mismatches(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    """Catches metadata or byte changes even when filenames remain stable."""
    images_dir, manifest, rows = _build_fixture(tmp_path)
    rows[0][field] = replacement
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match=message):
        verify_selected_images(images_dir, manifest, expected_count=2)


def test_verification_rejects_unreadable_selected_image(tmp_path: Path) -> None:
    """Catches hash-valid files that are no longer decodable project images."""
    images_dir, manifest, rows = _build_fixture(tmp_path)
    path = images_dir / "second.jpg"
    path.write_bytes(b"not a jpeg")
    rows[0]["size_bytes"] = path.stat().st_size
    rows[0]["sha256"] = _sha256(path)
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match="unreadable selected image"):
        verify_selected_images(images_dir, manifest, expected_count=2)


def test_verification_checks_expected_record_count(tmp_path: Path) -> None:
    """Catches accidentally running the real gate against a partial manifest."""
    images_dir, manifest, _ = _build_fixture(tmp_path)

    with pytest.raises(ValueError, match="expected 288 selected records, found 2"):
        verify_selected_images(images_dir, manifest, expected_count=288)
