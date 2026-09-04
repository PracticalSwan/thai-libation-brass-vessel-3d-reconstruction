"""Validated segmentation labels and paired image/mask transforms for Steps 7+8."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from analysis_common import SelectedImageRecord, verify_selected_record

DEFAULT_INPUT_SIZE = (384, 288)  # (height, width)
DEFAULT_SPLIT_COUNTS = {"train": 24, "val": 6, "test": 6}
_ALLOWED_SPLITS = frozenset(DEFAULT_SPLIT_COUNTS)
_REQUIRED_FIELDS = {
    "selected_index",
    "filename",
    "split",
    "view_category",
    "quality_condition",
    "source_width",
    "source_height",
    "source_sha256",
    "mask_path",
    "mask_sha256",
    "annotation_method",
}


@dataclass(frozen=True)
class SegmentationRecord:
    selected_index: int
    filename: str
    split: str
    view_category: str
    quality_condition: str
    source_width: int
    source_height: int
    source_sha256: str
    mask_path: Path
    mask_sha256: str
    annotation_method: str


@dataclass(frozen=True)
class ValidatedSegmentationSet:
    records: tuple[SegmentationRecord, ...]
    manifest_sha256: str
    images_dir: Path
    project_root: Path

    def split(self, name: str) -> tuple[SegmentationRecord, ...]:
        if name not in _ALLOWED_SPLITS:
            raise ValueError(f"invalid split: {name}")
        return tuple(record for record in self.records if record.split == name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_positive_int(value: str, field: str, row_number: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {field} in segmentation row {row_number}") from error
    if parsed <= 0:
        raise ValueError(f"invalid {field} in segmentation row {row_number}")
    return parsed


def _validate_sha256(value: str, field: str, row_number: int) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError(f"invalid {field} in segmentation row {row_number}")
    return normalized


def _resolve_mask_path(project_root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative.name in {"", ".", ".."}:
        raise ValueError(f"mask path must be a safe project-relative path: {value}")
    resolved = (project_root / relative).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError(f"mask path escapes project root: {value}") from error
    return resolved


def validate_binary_mask(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    """Read and validate one source-size binary ground-truth mask."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"missing or unreadable segmentation mask: {path}")
    height, width = mask.shape
    expected_width, expected_height = expected_size
    if (width, height) != (expected_width, expected_height):
        raise ValueError(
            f"mask dimension mismatch for {path.name}: expected "
            f"{expected_width}x{expected_height}, found {width}x{height}"
        )
    values = set(int(value) for value in np.unique(mask))
    if not values.issubset({0, 255}):
        raise ValueError(f"mask must contain only 0 and 255: {path.name}")
    return mask


def load_segmentation_manifest(
    manifest_path: Path,
    selected_records: Sequence[SelectedImageRecord],
    images_dir: Path,
    *,
    expected_counts: Mapping[str, int] | None = DEFAULT_SPLIT_COUNTS,
    verify_source_hashes: bool = True,
) -> ValidatedSegmentationSet:
    """Load the frozen segmentation split and verify label/source provenance."""
    project_root = manifest_path.resolve().parent.parent
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(_REQUIRED_FIELDS - fields)
        if missing:
            raise ValueError("segmentation manifest missing fields: " + ", ".join(missing))
        rows = list(reader)

    selected_by_index = {record.index: record for record in selected_records}
    records: list[SegmentationRecord] = []
    seen_indices: set[int] = set()
    for row_number, row in enumerate(rows, start=1):
        selected_index = _parse_positive_int(row["selected_index"], "selected_index", row_number)
        if selected_index in seen_indices:
            raise ValueError(f"duplicate selected_index in segmentation manifest: {selected_index}")
        seen_indices.add(selected_index)
        if selected_index not in selected_by_index:
            raise ValueError(f"unknown selected_index in segmentation manifest: {selected_index}")

        selected = selected_by_index[selected_index]
        filename = str(row["filename"])
        if filename != selected.filename:
            raise ValueError(
                f"filename mismatch for selected index {selected_index}: "
                f"expected {selected.filename}, found {filename}"
            )
        split = str(row["split"])
        if split not in _ALLOWED_SPLITS:
            raise ValueError(f"invalid segmentation split: {split}")
        width = _parse_positive_int(row["source_width"], "source_width", row_number)
        height = _parse_positive_int(row["source_height"], "source_height", row_number)
        if (width, height) != (selected.width, selected.height):
            raise ValueError(f"source dimensions disagree with selection manifest: {filename}")
        source_sha256 = _validate_sha256(row["source_sha256"], "source_sha256", row_number)
        if source_sha256 != selected.sha256:
            raise ValueError(f"source hash disagrees with selection manifest: {filename}")

        verify_selected_record(images_dir, selected, verify_hash=verify_source_hashes)
        mask_path = _resolve_mask_path(project_root, str(row["mask_path"]))
        validate_binary_mask(mask_path, (width, height))
        mask_sha256 = _validate_sha256(row["mask_sha256"], "mask_sha256", row_number)
        if sha256_file(mask_path) != mask_sha256:
            raise ValueError(f"mask hash mismatch: {mask_path.name}")
        annotation_method = str(row["annotation_method"]).strip()
        if not annotation_method:
            raise ValueError(f"annotation_method is empty for selected index {selected_index}")

        records.append(
            SegmentationRecord(
                selected_index=selected_index,
                filename=filename,
                split=split,
                view_category=str(row["view_category"]),
                quality_condition=str(row["quality_condition"]),
                source_width=width,
                source_height=height,
                source_sha256=source_sha256,
                mask_path=mask_path,
                mask_sha256=mask_sha256,
                annotation_method=annotation_method,
            )
        )

    if expected_counts is not None:
        counts = Counter(record.split for record in records)
        if dict(counts) != dict(expected_counts):
            raise ValueError(
                f"invalid segmentation split counts: found {dict(counts)}, "
                f"expected {dict(expected_counts)}"
            )
    return ValidatedSegmentationSet(
        records=tuple(records),
        manifest_sha256=sha256_file(manifest_path),
        images_dir=images_dir.resolve(),
        project_root=project_root,
    )


def _random_uniform(low: float, high: float) -> float:
    return low + (high - low) * float(torch.rand(1).item())


def _paired_transform(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    training: bool,
    input_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    target_height, target_width = input_size
    image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    if training:
        if float(torch.rand(1).item()) < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        angle = _random_uniform(-5.0, 5.0)
        scale = _random_uniform(0.95, 1.05)
        tx = _random_uniform(-0.03, 0.03) * target_width
        ty = _random_uniform(-0.03, 0.03) * target_height
        matrix = cv2.getRotationMatrix2D((target_width / 2.0, target_height / 2.0), angle, scale)
        matrix[0, 2] += tx
        matrix[1, 2] += ty
        image = cv2.warpAffine(
            image,
            matrix,
            (target_width, target_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        mask = cv2.warpAffine(
            mask,
            matrix,
            (target_width, target_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        contrast = _random_uniform(0.85, 1.15)
        brightness = _random_uniform(-0.15, 0.15) * 255.0
        image = np.clip(image.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(np.ascontiguousarray(image_rgb.transpose(2, 0, 1))).float() / 255.0
    image_tensor = (image_tensor - 0.5) / 0.5
    mask_tensor = torch.from_numpy(np.ascontiguousarray((mask > 0).astype(np.float32))).unsqueeze(0)
    return image_tensor, mask_tensor


class SegmentationDataset(Dataset):
    """PyTorch dataset that never modifies source JPEG or mask files."""

    def __init__(
        self,
        records: Sequence[SegmentationRecord],
        images_dir: Path,
        *,
        training: bool,
        input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
    ) -> None:
        self.records = tuple(records)
        self.images_dir = images_dir
        self.training = training
        self.input_size = input_size
        if not self.records:
            raise ValueError("segmentation dataset requires at least one record")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        image = cv2.imread(str(self.images_dir / record.filename), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unreadable selected image: {record.filename}")
        mask = validate_binary_mask(
            record.mask_path, (record.source_width, record.source_height)
        )
        image_tensor, mask_tensor = _paired_transform(
            image, mask, training=self.training, input_size=self.input_size
        )
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "selected_index": record.selected_index,
            "filename": record.filename,
        }
