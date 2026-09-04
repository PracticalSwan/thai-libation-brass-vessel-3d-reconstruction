"""Step 9A full-sequence frozen CNN inference and reconstruction-mask cleanup."""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from analysis_common import SelectedImageRecord
from cnn_segmentation import SmallSegCNN, logits_to_binary


@dataclass(frozen=True)
class MaskRecord:
    selected_index: int
    filename: str
    source_width: int
    source_height: int
    raw_prediction_path: Path
    raw_prediction_sha256: str
    reconstruction_mask_path: Path
    reconstruction_mask_sha256: str
    raw_foreground_fraction: float
    reconstruction_foreground_fraction: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_MASK_MANIFEST_FIELDS = (
    "selected_index",
    "filename",
    "source_width",
    "source_height",
    "raw_prediction_path",
    "raw_prediction_sha256",
    "reconstruction_mask_path",
    "reconstruction_mask_sha256",
    "raw_foreground_fraction",
    "reconstruction_foreground_fraction",
)


def write_mask_manifest(
    path: Path, records: Sequence[MaskRecord], *, project_root: Path
) -> None:
    if not records:
        raise ValueError("mask manifest requires at least one record")
    root = project_root.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_MASK_MANIFEST_FIELDS))
        writer.writeheader()
        for record in records:
            try:
                raw_relative = record.raw_prediction_path.resolve().relative_to(root)
                clean_relative = record.reconstruction_mask_path.resolve().relative_to(root)
            except ValueError as error:
                raise ValueError("mask manifest paths must be inside project root") from error
            writer.writerow(
                {
                    "selected_index": record.selected_index,
                    "filename": record.filename,
                    "source_width": record.source_width,
                    "source_height": record.source_height,
                    "raw_prediction_path": raw_relative.as_posix(),
                    "raw_prediction_sha256": record.raw_prediction_sha256,
                    "reconstruction_mask_path": clean_relative.as_posix(),
                    "reconstruction_mask_sha256": record.reconstruction_mask_sha256,
                    "raw_foreground_fraction": record.raw_foreground_fraction,
                    "reconstruction_foreground_fraction": record.reconstruction_foreground_fraction,
                }
            )


def _safe_manifest_path(project_root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe mask manifest path: {value}")
    resolved = (project_root.resolve() / relative).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError(f"mask manifest path escapes project root: {value}") from error
    return resolved


def _validate_saved_mask(
    path: Path,
    *,
    expected_width: int,
    expected_height: int,
    expected_sha256: str,
    expected_fraction: float,
) -> None:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"missing or unreadable Step 9 mask: {path}")
    if mask.shape != (expected_height, expected_width):
        raise ValueError(f"Step 9 mask geometry mismatch: {path.name}")
    values = set(int(value) for value in np.unique(mask))
    if not values.issubset({0, 255}):
        raise ValueError(f"Step 9 mask is not binary: {path.name}")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"Step 9 mask hash mismatch: {path.name}")
    actual_fraction = float(np.count_nonzero(mask)) / float(mask.size)
    if abs(actual_fraction - expected_fraction) > 1e-9:
        raise ValueError(f"Step 9 mask foreground fraction mismatch: {path.name}")


def load_mask_manifest(
    path: Path, *, project_root: Path, expected_count: int | None = None
) -> tuple[MaskRecord, ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _MASK_MANIFEST_FIELDS:
            raise ValueError("unexpected reconstruction mask manifest columns")
        rows = list(reader)
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(
            f"expected {expected_count} reconstruction mask rows, found {len(rows)}"
        )

    records: list[MaskRecord] = []
    seen_indices: set[int] = set()
    for row in rows:
        selected_index = int(row["selected_index"])
        if selected_index in seen_indices:
            raise ValueError(f"duplicate reconstruction mask index: {selected_index}")
        seen_indices.add(selected_index)
        width = int(row["source_width"])
        height = int(row["source_height"])
        raw_path = _safe_manifest_path(project_root, row["raw_prediction_path"])
        clean_path = _safe_manifest_path(project_root, row["reconstruction_mask_path"])
        raw_fraction = float(row["raw_foreground_fraction"])
        clean_fraction = float(row["reconstruction_foreground_fraction"])
        _validate_saved_mask(
            raw_path,
            expected_width=width,
            expected_height=height,
            expected_sha256=row["raw_prediction_sha256"],
            expected_fraction=raw_fraction,
        )
        _validate_saved_mask(
            clean_path,
            expected_width=width,
            expected_height=height,
            expected_sha256=row["reconstruction_mask_sha256"],
            expected_fraction=clean_fraction,
        )
        records.append(
            MaskRecord(
                selected_index=selected_index,
                filename=row["filename"],
                source_width=width,
                source_height=height,
                raw_prediction_path=raw_path,
                raw_prediction_sha256=row["raw_prediction_sha256"],
                reconstruction_mask_path=clean_path,
                reconstruction_mask_sha256=row["reconstruction_mask_sha256"],
                raw_foreground_fraction=raw_fraction,
                reconstruction_foreground_fraction=clean_fraction,
            )
        )
    return tuple(records)


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    values = set(int(value) for value in np.unique(mask))
    if not values.issubset({0, 255}):
        raise ValueError("mask must contain only 0 and 255")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _bbox_gap(first: np.ndarray, second: np.ndarray) -> float:
    x1, y1, w1, h1 = (int(value) for value in first[:4])
    x2, y2, w2, h2 = (int(value) for value in second[:4])
    right1 = x1 + w1 - 1
    bottom1 = y1 + h1 - 1
    right2 = x2 + w2 - 1
    bottom2 = y2 + h2 - 1
    gap_x = max(x2 - right1 - 1, x1 - right2 - 1, 0)
    gap_y = max(y2 - bottom1 - 1, y1 - bottom2 - 1, 0)
    return math.hypot(gap_x, gap_y)


def cleanup_reconstruction_mask(mask: np.ndarray) -> np.ndarray:
    """Keep the central vessel component and only nearby meaningful fragments."""
    binary = _binary_mask(mask)
    foreground = (binary > 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground, connectivity=8
    )
    if component_count <= 1:
        return binary
    if component_count == 2:
        return binary

    height, width = binary.shape
    x0 = int(math.floor(width * 0.25))
    x1 = int(math.ceil(width * 0.75))
    y0 = int(math.floor(height * 0.20))
    y1 = int(math.ceil(height * 0.85))
    roi_labels = set(int(value) for value in np.unique(labels[y0:y1, x0:x1]))
    roi_labels.discard(0)
    foreground_labels = list(range(1, component_count))
    anchor_candidates = list(roi_labels) if roi_labels else foreground_labels
    anchor_label = max(anchor_candidates, key=lambda label: int(stats[label, cv2.CC_STAT_AREA]))
    anchor_area = int(stats[anchor_label, cv2.CC_STAT_AREA])
    if anchor_area <= 0:
        return np.zeros_like(binary)

    keep = {anchor_label}
    minimum_secondary_area = max(8, int(math.ceil(anchor_area * 0.02)))
    maximum_gap = math.hypot(width, height) * 0.03
    anchor_stats = stats[anchor_label]
    for label in foreground_labels:
        if label == anchor_label:
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_secondary_area:
            continue
        if _bbox_gap(anchor_stats, stats[label]) <= maximum_gap:
            keep.add(label)

    cleaned = np.isin(labels, tuple(sorted(keep))).astype(np.uint8) * 255
    return cleaned


def restore_binary_mask(mask: np.ndarray, source_size: tuple[int, int]) -> np.ndarray:
    binary = _binary_mask(mask)
    width, height = source_size
    if width < 1 or height < 1:
        raise ValueError("source size must be positive")
    restored = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.where(restored > 0, 255, 0).astype(np.uint8)


def verify_reconstruction_checkpoint(
    checkpoint: dict[str, object], segmentation_manifest_sha256: str
) -> None:
    if checkpoint.get("model_name") != "SmallSegCNN":
        raise ValueError("unexpected reconstruction checkpoint model")
    if checkpoint.get("random_initialization") is not True:
        raise ValueError("checkpoint does not prove random initialization")
    if checkpoint.get("pretrained_weights") is not False:
        raise ValueError("pretrained checkpoint is not allowed")
    if checkpoint.get("manifest_sha256") != segmentation_manifest_sha256:
        raise ValueError("checkpoint segmentation manifest hash mismatch")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint config is missing")
    if float(config.get("threshold", -1.0)) != 0.5:
        raise ValueError("checkpoint threshold must remain frozen at 0.5")
    if int(config.get("input_height", 0)) < 1 or int(config.get("input_width", 0)) < 1:
        raise ValueError("checkpoint input geometry is invalid")


def _paths_overlap(first: Path, second: Path) -> bool:
    first_resolved = first.resolve()
    second_resolved = second.resolve()
    return (
        first_resolved == second_resolved
        or first_resolved in second_resolved.parents
        or second_resolved in first_resolved.parents
    )


def validate_output_directories(
    raw_prediction_dir: Path,
    reconstruction_mask_dir: Path,
    raw_source_dir: Path,
    selected_source_dir: Path,
) -> None:
    outputs = (raw_prediction_dir, reconstruction_mask_dir)
    sources = (raw_source_dir, selected_source_dir)
    for output in outputs:
        for source in sources:
            if _paths_overlap(output, source):
                raise ValueError("Step 9 mask output directory overlaps protected source data")
    if _paths_overlap(raw_prediction_dir, reconstruction_mask_dir):
        raise ValueError("raw prediction and reconstruction-mask directories must be separate")


def prepare_image_tensor(
    image: np.ndarray, input_size: tuple[int, int]
) -> torch.Tensor:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("inference image must be BGR color")
    height, width = input_size
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255.0
    return ((tensor - 0.5) / 0.5).unsqueeze(0)


def infer_selected_masks(
    records: Sequence[SelectedImageRecord],
    images_dir: Path,
    *,
    checkpoint_path: Path,
    segmentation_manifest_sha256: str,
    raw_prediction_dir: Path,
    reconstruction_mask_dir: Path,
    raw_source_dir: Path,
    device: torch.device | None = None,
) -> tuple[MaskRecord, ...]:
    """Run the frozen model once per verified selected image."""
    if not records:
        raise ValueError("selected records must not be empty")
    validate_output_directories(
        raw_prediction_dir,
        reconstruction_mask_dir,
        raw_source_dir,
        images_dir,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    verify_reconstruction_checkpoint(checkpoint, segmentation_manifest_sha256)
    config = checkpoint["config"]
    input_size = (int(config["input_height"]), int(config["input_width"]))
    threshold = float(config["threshold"])
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallSegCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    raw_prediction_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_mask_dir.mkdir(parents=True, exist_ok=True)
    output_records: list[MaskRecord] = []
    with torch.no_grad():
        for record in records:
            image_path = images_dir / record.filename
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"unreadable selected image: {record.filename}")
            source_height, source_width = image.shape[:2]
            if (source_width, source_height) != (record.width, record.height):
                raise ValueError(f"selected image geometry mismatch: {record.filename}")

            tensor = prepare_image_tensor(image, input_size).to(device)
            logits = model(tensor)
            model_mask = (
                logits_to_binary(logits, threshold)[0, 0].detach().cpu().numpy().astype(np.uint8)
                * 255
            )
            cleaned_model_mask = cleanup_reconstruction_mask(model_mask)
            raw_prediction = restore_binary_mask(model_mask, (source_width, source_height))
            reconstruction_mask = restore_binary_mask(
                cleaned_model_mask, (source_width, source_height)
            )

            stem = Path(record.filename).stem
            raw_path = raw_prediction_dir / f"{record.index:03d}_{stem}_pred.png"
            reconstruction_path = (
                reconstruction_mask_dir / f"{record.index:03d}_{stem}_mask.png"
            )
            if not cv2.imwrite(str(raw_path), raw_prediction):
                raise RuntimeError(f"failed to write raw CNN prediction: {raw_path}")
            if not cv2.imwrite(str(reconstruction_path), reconstruction_mask):
                raise RuntimeError(f"failed to write reconstruction mask: {reconstruction_path}")

            output_records.append(
                MaskRecord(
                    selected_index=record.index,
                    filename=record.filename,
                    source_width=source_width,
                    source_height=source_height,
                    raw_prediction_path=raw_path,
                    raw_prediction_sha256=sha256_file(raw_path),
                    reconstruction_mask_path=reconstruction_path,
                    reconstruction_mask_sha256=sha256_file(reconstruction_path),
                    raw_foreground_fraction=float(np.count_nonzero(raw_prediction))
                    / float(raw_prediction.size),
                    reconstruction_foreground_fraction=float(
                        np.count_nonzero(reconstruction_mask)
                    )
                    / float(reconstruction_mask.size),
                )
            )
    return tuple(output_records)
