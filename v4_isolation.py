"""Mask contracts and the official Grounded-SAM2 isolation stage for V4.

The source JPEGs remain immutable.  Runtime code and checkpoints live under
the approved ``reconstruction/v4/work`` boundary, while this module writes
only coordinate-preserving mask products under ``capture_v4/derived``.  No
threshold or private SmallSeg fallback is used when the official models are
unavailable.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import importlib.metadata
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from v4_config import (
    CAPTURE_V4_ROOT,
    RECONSTRUCTION_V4_ROOT,
    V4_SOURCE_ROOT,
    assert_output_path,
    fingerprint,
    sha256_file,
    write_json,
)

try:
    import cv2
except ImportError:  # pragma: no cover - only minimal Python environments
    cv2 = None


class SegmentationCapabilityError(RuntimeError):
    """Raised when the mandated official segmentation stack is unavailable."""


DEFAULT_DINO_PROMPT = "a brass libation vessel."
DEFAULT_DINO_BOX_THRESHOLD = 0.25
DEFAULT_DINO_TEXT_THRESHOLD = 0.20
DEFAULT_REFRESH_INTERVAL = 24


def _package_root(name: str) -> Path | None:
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve().parent


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def official_model_identity(
    *,
    grounding_checkpoint: str | Path | None = None,
    sam2_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Return code/config/checkpoint identities used by the real mask run."""

    dino_root = _package_root("groundingdino")
    sam_root = _package_root("sam2")
    dino_config = dino_root / "config" / "GroundingDINO_SwinT_OGC.py" if dino_root else None
    sam_config = sam_root / "configs" / "sam2.1" / "sam2.1_hiera_s.yaml" if sam_root else None

    def file_identity(path: Path | None) -> dict[str, Any]:
        if path is None:
            return {"path": "", "status": "missing"}
        resolved = path.resolve()
        if not resolved.is_file():
            return {"path": str(resolved), "status": "missing"}
        return {
            "path": str(resolved),
            "status": "available",
            "bytes": int(resolved.stat().st_size),
            "sha256": sha256_file(resolved),
        }

    payload = {
        "groundingdino": {
            "package": _package_version("groundingdino"),
            "root": str(dino_root) if dino_root else "",
            "config": file_identity(dino_config),
            "checkpoint": file_identity(Path(grounding_checkpoint) if grounding_checkpoint else None),
        },
        "sam2": {
            "package": _package_version("SAM-2"),
            "root": str(sam_root) if sam_root else "",
            "config": file_identity(sam_config),
            "checkpoint": file_identity(Path(sam2_checkpoint) if sam2_checkpoint else None),
        },
        "prompt": DEFAULT_DINO_PROMPT,
        "dino_box_threshold": DEFAULT_DINO_BOX_THRESHOLD,
        "dino_text_threshold": DEFAULT_DINO_TEXT_THRESHOLD,
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


@dataclass(frozen=True)
class MaskProducts:
    full_mask: np.ndarray
    feature_mask: np.ndarray
    mvs_mask: np.ndarray
    boundary_exclusion: np.ndarray

    def validate(self, image_size: tuple[int, int]) -> "MaskProducts":
        width, height = image_size
        expected = (height, width)
        for name in ("full_mask", "feature_mask", "mvs_mask", "boundary_exclusion"):
            value = getattr(self, name)
            if value.shape != expected:
                raise ValueError(f"{name} shape {value.shape} does not match source {expected}")
            if value.dtype != np.uint8:
                raise ValueError(f"{name} must use uint8 binary pixels")
            if not set(np.unique(value).tolist()).issubset({0, 255}):
                raise ValueError(f"{name} must contain only 0 and 255")
        if np.any(self.feature_mask > self.full_mask) or np.any(self.mvs_mask > self.full_mask):
            raise ValueError("derived masks cannot add foreground outside the full mask")
        if np.any(self.boundary_exclusion > self.full_mask):
            raise ValueError("boundary exclusion cannot extend outside the full mask")
        return self


def binary_mask(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    if values.dtype.kind not in "buif":
        raise ValueError("mask must be numeric")
    if not np.isfinite(values).all():
        raise ValueError("mask contains non-finite values")
    return np.where(values > 0, 255, 0).astype(np.uint8)


def retain_vessel_component(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Keep the largest connected SAM component and report removed pixels.

    A box-prompted SAM result occasionally returns a tiny disconnected marker
    or board fragment alongside the vessel.  The vessel is the dominant
    connected component in these captures, so removing only smaller
    components is conservative and never expands the accepted support.
    """

    full = binary_mask(mask)
    if cv2 is None or not np.count_nonzero(full):
        return full, 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats((full > 0).astype(np.uint8), 8)
    if count <= 2:
        return full, 0
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cleaned = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    return cleaned, int(np.count_nonzero(full) - np.count_nonzero(cleaned))


def _erode(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask.copy()
    if cv2 is None:
        # A conservative fallback that does not expand or fabricate support.
        return mask.copy()
    kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), dtype=np.uint8)
    return cv2.erode(mask, kernel, iterations=1)


def derive_mask_products(
    full_mask: np.ndarray,
    *,
    inward_margin: int = 2,
    boundary_band: int = 3,
    preserve_thin_geometry: bool = True,
) -> MaskProducts:
    """Derive feature/MVS masks without changing source image coordinates.

    The full and MVS masks retain the accepted vessel support.  The feature
    mask is slightly inward where that is safe; components that would be
    erased by the margin (thin finial/rim evidence) are retained in the
    feature mask and protected by the explicit boundary-exclusion mask.
    """

    if inward_margin < 0 or boundary_band < 0:
        raise ValueError("mask margins must be nonnegative")
    full = binary_mask(full_mask)
    eroded = _erode(full, inward_margin)
    boundary = np.where((full > 0) & (eroded == 0), 255, 0).astype(np.uint8)
    if preserve_thin_geometry:
        # Preserve small connected components (finial/rim fragments) rather
        # than deleting thin real geometry merely for a cleaner edge.
        if cv2 is not None and np.count_nonzero(full):
            count, labels, stats, _ = cv2.connectedComponentsWithStats((full > 0).astype(np.uint8), 8)
            feature = eroded.copy()
            for label in range(1, count):
                component = labels == label
                if np.any(component & (eroded == 0)) and int(stats[label, cv2.CC_STAT_AREA]) < 256:
                    feature[component] = 255
        else:
            feature = full.copy() if np.count_nonzero(eroded) == 0 else eroded
    else:
        feature = eroded
    # Expand the exclusion band only inside the accepted full mask.  Dilation
    # is never applied to the foreground itself, especially at the wooden base.
    if boundary_band > 0 and cv2 is not None and np.count_nonzero(boundary):
        kernel = np.ones((boundary_band * 2 + 1, boundary_band * 2 + 1), dtype=np.uint8)
        boundary = cv2.dilate(boundary, kernel, iterations=1)
        boundary = np.where((boundary > 0) & (full > 0), 255, 0).astype(np.uint8)
    products = MaskProducts(
        full_mask=full,
        feature_mask=np.where(feature > 0, 255, 0).astype(np.uint8),
        mvs_mask=full.copy(),
        boundary_exclusion=boundary,
    )
    return products


def keypoint_mask_membership(
    keypoints: np.ndarray,
    feature_mask: np.ndarray,
    *,
    boundary_exclusion: np.ndarray | None = None,
) -> np.ndarray:
    """Return a boolean keep mask for original full-image keypoint coordinates."""

    points = np.asarray(keypoints, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("keypoints must have shape N x 2")
    accepted = binary_mask(feature_mask)
    boundary = None if boundary_exclusion is None else binary_mask(boundary_exclusion)
    height, width = accepted.shape
    finite = np.isfinite(points).all(axis=1)
    # ALIKED coordinates are in pixel coordinates; floor keeps a point from
    # rounding across the image boundary and preserves exact source geometry.
    xy = np.zeros(points.shape, dtype=np.int64)
    finite_indices = np.flatnonzero(finite)
    if len(finite_indices):
        xy[finite_indices] = np.floor(points[finite_indices]).astype(np.int64, copy=False)
    inside = finite & (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    keep = np.zeros(len(points), dtype=bool)
    valid_indices = np.flatnonzero(inside)
    if len(valid_indices):
        keep[valid_indices] = accepted[xy[valid_indices, 1], xy[valid_indices, 0]] > 0
        if boundary is not None:
            keep[valid_indices] &= boundary[xy[valid_indices, 1], xy[valid_indices, 0]] == 0
    return keep


def filter_keypoints_to_mask(
    keypoints: np.ndarray,
    feature_mask: np.ndarray,
    *,
    boundary_exclusion: np.ndarray | None = None,
) -> np.ndarray:
    """Return the vessel-only subset while retaining original coordinates."""

    points = np.asarray(keypoints)
    return points[keypoint_mask_membership(points, feature_mask, boundary_exclusion=boundary_exclusion)]


def filter_keypoints(
    keypoints: np.ndarray,
    feature_mask: np.ndarray,
    *,
    boundary_exclusion: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(filtered_points, original_indices)`` for database import."""

    keep = keypoint_mask_membership(keypoints, feature_mask, boundary_exclusion=boundary_exclusion)
    points = np.asarray(keypoints)
    return points[keep], np.flatnonzero(keep)


def map_keypoints_to_original(
    keypoints: np.ndarray,
    *,
    feature_size: tuple[int, int],
    source_size: tuple[int, int],
) -> np.ndarray:
    """Map feature-only scaled coordinates back to original image coordinates."""

    fw, fh = feature_size
    sw, sh = source_size
    if min(fw, fh, sw, sh) <= 0:
        raise ValueError("image dimensions must be positive")
    points = np.asarray(keypoints, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("keypoints must have shape N x 2")
    scale = np.asarray([sw / fw, sh / fh], dtype=np.float32)
    return points * scale


def mask_diagnostics(mask: np.ndarray) -> dict[str, Any]:
    """Compute deterministic mask area/bbox/centroid and base-plane warnings."""

    binary = binary_mask(mask)
    ys, xs = np.nonzero(binary)
    height, width = binary.shape
    if len(xs):
        bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
        centroid = [float(xs.mean()), float(ys.mean())]
    else:
        bbox = [0, 0, 0, 0]
        centroid = [None, None]
    area = int(len(xs))
    horizontal_support = 0
    if len(xs):
        base_band = binary[max(0, int(height * 0.72)) :]
        horizontal_support = int(np.max(np.sum(base_band > 0, axis=1))) if base_band.size else 0
    largest_fraction = 0.0
    if cv2 is not None and area:
        count, _, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
        if count > 1:
            largest_fraction = float(np.max(stats[1:, cv2.CC_STAT_AREA]) / area)
    component_count = 1 if area else 0
    second_largest_fraction = 0.0
    if cv2 is not None and area:
        count, _, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
        component_count = max(0, int(count - 1))
        if component_count > 1:
            areas = sorted((int(value) for value in stats[1:, cv2.CC_STAT_AREA]), reverse=True)
            second_largest_fraction = float(areas[1] / max(areas[0], 1))
    return {
        "height": height,
        "width": width,
        "foreground_pixels": area,
        "foreground_fraction": float(area / max(binary.size, 1)),
        "bbox_xywh": bbox,
        "centroid_xy": centroid,
        "largest_component_fraction": largest_fraction,
        "component_count": component_count,
        "second_largest_fraction": second_largest_fraction,
        "max_horizontal_base_support": horizontal_support,
        "base_support_warning": bool(horizontal_support > width * 0.45),
    }


def build_mvs_image(image: np.ndarray, mvs_mask: np.ndarray, *, outside_value: int = 0) -> np.ndarray:
    """Neutralize outside-mask pixels without resizing/cropping the image."""

    values = np.asarray(image)
    mask = binary_mask(mvs_mask)
    if values.ndim not in (2, 3) or values.shape[:2] != mask.shape:
        raise ValueError("image and MVS mask dimensions do not match")
    if not 0 <= outside_value <= 255:
        raise ValueError("outside_value must be within 0..255")
    output = values.copy()
    output[mask == 0] = outside_value
    return output


def write_mask(path: str | Path, mask: np.ndarray, image_size: tuple[int, int]) -> Path:
    target = assert_output_path(path)
    binary = binary_mask(mask)
    width, height = image_size
    if binary.shape != (height, width):
        raise ValueError(f"mask dimensions {binary.shape} do not match {(height, width)}")
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary, mode="L").save(target, format="PNG")
    return target


def segmentation_capability() -> dict[str, Any]:
    """Report import capability without claiming that weights are present."""

    modules = {}
    for name in ("groundingdino", "sam2"):
        modules[name] = bool(importlib.util.find_spec(name))
    return {
        "groundingdino_t": modules["groundingdino"],
        "sam2_1_hiera_small": modules["sam2"],
        "status": "available" if all(modules.values()) else "blocked",
        "missing": [name for name, present in modules.items() if not present],
        "required_models": ["Grounding DINO-T/Swin-T", "SAM 2.1 Hiera-small"],
        "smallseg_primary": False,
    }


def require_segmentation_capability() -> dict[str, Any]:
    capability = segmentation_capability()
    if capability["status"] != "available":
        raise SegmentationCapabilityError(
            "V4 isolation requires official Grounding DINO-T + SAM 2.1 Hiera-small; "
            f"missing imports: {', '.join(capability['missing'])}"
        )
    return capability


def refine_board_leakage(
    provisional_mask: np.ndarray,
    *,
    supported_board_pixels: np.ndarray | None = None,
) -> np.ndarray:
    """Apply only negative board evidence; never add foreground by differencing."""

    mask = binary_mask(provisional_mask)
    if supported_board_pixels is None:
        return mask
    board = binary_mask(supported_board_pixels)
    if board.shape != mask.shape:
        raise ValueError("board evidence and provisional mask dimensions differ")
    return np.where((mask > 0) & (board == 0), 255, 0).astype(np.uint8)


def write_mask_diagnostics(path: str | Path, records: Mapping[str, Mapping[str, Any]]) -> Path:
    return write_json(path, {"schema_version": 1, "records": dict(records)})


def _load_source_rgb(path: Path) -> np.ndarray:
    if cv2 is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unable to decode source image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"))


def _detector_box(
    boxes: Any,
    scores: Any,
    source_size: tuple[int, int],
) -> tuple[list[float], float]:
    """Choose one vessel box from normalized Grounding DINO predictions."""

    width, height = source_size
    box_array = boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes)
    score_array = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
    candidates: list[tuple[float, float, list[float]]] = []
    for box, score in zip(np.asarray(box_array), np.asarray(score_array)):
        values = np.asarray(box, dtype=np.float64).reshape(-1)
        if values.size != 4 or not np.isfinite(values).all():
            continue
        cx, cy, bw, bh = [float(value) for value in values]
        x1 = max(0.0, (cx - bw / 2.0) * width)
        y1 = max(0.0, (cy - bh / 2.0) * height)
        x2 = min(float(width), (cx + bw / 2.0) * width)
        y2 = min(float(height), (cy + bh / 2.0) * height)
        area_fraction = max(0.0, (x2 - x1) * (y2 - y1) / max(width * height, 1))
        confidence = float(np.asarray(score).reshape(-1)[0])
        if x2 <= x1 or y2 <= y1 or not math.isfinite(confidence):
            continue
        # Confidence is primary; area only breaks ties in favour of a box that
        # can contain the complete vessel rather than a tiny texture hit.
        candidates.append((confidence, area_fraction, [x1, y1, x2, y2]))
    if not candidates:
        raise SegmentationCapabilityError("Grounding DINO returned no finite vessel box")
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    confidence, _, box = candidates[0]
    return [float(value) for value in box], float(confidence)


def _box_from_mask(mask: np.ndarray, *, padding_fraction: float = 0.08) -> list[float]:
    diagnostics = mask_diagnostics(mask)
    x, y, width, height = [int(value) for value in diagnostics["bbox_xywh"]]
    if width <= 0 or height <= 0:
        raise SegmentationCapabilityError("SAM 2 produced an empty mask; cannot propagate a box")
    pad_x = max(8, int(round(width * padding_fraction)))
    pad_y = max(8, int(round(height * padding_fraction)))
    mask_height, mask_width = mask.shape
    return [
        float(max(0, x - pad_x)),
        float(max(0, y - pad_y)),
        float(min(mask_width, x + width + pad_x)),
        float(min(mask_height, y + height + pad_y)),
    ]


def _sam_mask(predictor: Any, box: Sequence[float], image_shape: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Run one full-resolution SAM 2 box prompt and choose a valid mask."""

    masks, scores, _ = predictor.predict(
        box=np.asarray(box, dtype=np.float32),
        multimask_output=False,
    )
    mask_array = np.asarray(masks)
    if mask_array.ndim == 2:
        mask_array = mask_array[None, ...]
    score_array = np.asarray(scores).reshape(-1) if scores is not None else np.zeros(len(mask_array))
    candidates: list[tuple[float, float, np.ndarray]] = []
    for index, candidate in enumerate(mask_array):
        binary = binary_mask(candidate)
        diagnostics = mask_diagnostics(binary)
        area = float(diagnostics["foreground_fraction"])
        score = float(score_array[index]) if index < len(score_array) else 0.0
        if binary.shape == image_shape and 0.001 <= area <= 0.75:
            candidates.append((score, -abs(area - 0.05), binary))
    if not candidates:
        raise SegmentationCapabilityError("SAM 2 returned no plausible full-resolution vessel mask")
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    score, _, mask = candidates[0]
    return mask, score


def _save_mvs_image(path: Path, image_rgb: np.ndarray, mask: np.ndarray) -> Path:
    target = assert_output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    values = build_mvs_image(image_rgb, mask, outside_value=0)
    if cv2 is not None:
        ok = cv2.imwrite(
            str(target),
            cv2.cvtColor(values, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )
        if not ok:
            raise IOError(f"failed to write MVS image: {target}")
    else:  # pragma: no cover - cv2 is part of the V4 runtime
        Image.fromarray(values, mode="RGB").save(target, format="JPEG", quality=95)
    return target


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    selected = [
        row
        for row in rows
        if row.get("source_role") == "geometry" and row.get("selected_for_geometry", "").lower() == "true"
    ]
    selected.sort(
        key=lambda row: (
            str(row.get("logical_ring_id", "")),
            int(row.get("frame_index_within_logical_ring") or 0),
            str(row.get("relative_path", "")),
        )
    )
    if not selected:
        raise ValueError(f"manifest has no selected geometry rows: {path}")
    return selected


def _existing_products_valid(record: Mapping[str, Any], *, image_size: tuple[int, int]) -> bool:
    products = record.get("products")
    if not isinstance(products, Mapping):
        return False
    width, height = image_size
    for key in ("full_mask", "feature_mask", "mvs_image"):
        item = products.get(key)
        if not isinstance(item, Mapping):
            return False
        path = Path(str(item.get("path", "")))
        if not path.is_file():
            return False
        expected_hash = str(item.get("sha256", ""))
        if expected_hash and sha256_file(path) != expected_hash:
            return False
        if key.endswith("mask"):
            try:
                with Image.open(path) as handle:
                    if handle.size != (width, height) or handle.mode not in {"L", "1"}:
                        return False
            except Exception:
                return False
        else:
            try:
                with Image.open(path) as handle:
                    if handle.size != (width, height):
                        return False
            except Exception:
                return False
    return True


def run_official_isolation(
    *,
    manifest_path: str | Path = CAPTURE_V4_ROOT / "manifests" / "source_manifest.csv",
    source_root: str | Path = V4_SOURCE_ROOT,
    output_root: str | Path = CAPTURE_V4_ROOT / "derived",
    report_path: str | Path = RECONSTRUCTION_V4_ROOT / "reports" / "isolation_report.json",
    index_path: str | Path = RECONSTRUCTION_V4_ROOT / "work" / "isolation_records.json",
    grounding_checkpoint: str | Path = RECONSTRUCTION_V4_ROOT / "work" / "models" / "groundingdino_swint_ogc.pth",
    sam2_checkpoint: str | Path = RECONSTRUCTION_V4_ROOT / "work" / "models" / "sam2.1_hiera_small.pt",
    refresh_interval: int = DEFAULT_REFRESH_INTERVAL,
    dino_device: str = "cpu",
    sam_device: str = "cuda",
    resume: bool = True,
) -> dict[str, Any]:
    """Generate full-resolution V4 masks with official Grounding DINO + SAM 2.

    Grounding DINO is kept on CPU when its optional custom CUDA extension is
    unavailable on Windows; SAM 2 remains on CUDA.  Detection is refreshed at
    ring starts and periodically, while intervening boxes follow the previous
    accepted mask.  A failed or implausible mask stops the stage rather than
    producing a guessed substitute.
    """

    if refresh_interval < 1:
        raise ValueError("refresh_interval must be positive")
    manifest = Path(manifest_path).resolve()
    source = Path(source_root).resolve()
    derived = Path(output_root).resolve()
    report = assert_output_path(report_path)
    index = assert_output_path(index_path)
    mask_root = assert_output_path(derived / "masks")
    feature_root = assert_output_path(derived / "feature_masks")
    mvs_root = assert_output_path(derived / "mvs_images")
    for directory in (mask_root, feature_root, mvs_root):
        directory.mkdir(parents=True, exist_ok=True)

    capability = require_segmentation_capability()
    model_identity = official_model_identity(
        grounding_checkpoint=grounding_checkpoint,
        sam2_checkpoint=sam2_checkpoint,
    )
    for model_name in ("groundingdino", "sam2"):
        model_data = model_identity[model_name]
        checkpoint = model_data["checkpoint"]
        config = model_data["config"]
        if checkpoint.get("status") != "available" or config.get("status") != "available":
            raise SegmentationCapabilityError(
                f"official {model_name} checkpoint/config is unavailable: {checkpoint.get('path') or config.get('path')}"
            )

    selected = _manifest_rows(manifest)
    existing_records: dict[str, dict[str, Any]] = {}
    if resume and index.is_file():
        try:
            payload = json.loads(index.read_text(encoding="utf-8"))
            for record in payload.get("records", []):
                if isinstance(record, Mapping) and record.get("relative_path"):
                    existing_records[str(record["relative_path"])] = dict(record)
        except (OSError, ValueError, TypeError):
            existing_records = {}

    repaired_count = 0
    # Repair only disconnected components in already-complete products.  This
    # keeps a resumable run from re-running 372 model inferences after a QA
    # rule is tightened, while still refreshing every affected output hash.
    for row in selected:
        old = existing_records.get(row["relative_path"])
        image_size = (int(row["width"]), int(row["height"]))
        if not old or not _existing_products_valid(old, image_size=image_size):
            continue
        full_path = Path(old["products"]["full_mask"]["path"])
        with Image.open(full_path) as handle:
            old_mask = np.asarray(handle.convert("L"), dtype=np.uint8)
        cleaned, removed = retain_vessel_component(old_mask)
        if removed <= 0:
            continue
        source_path = source / row["relative_path"]
        image_rgb = _load_source_rgb(source_path)
        products = derive_mask_products(cleaned, inward_margin=2, boundary_band=3)
        products.validate(image_size)
        mask_path = write_mask(mask_root / Path(row["relative_path"]).name, products.full_mask, image_size)
        feature_path = write_mask(feature_root / Path(row["relative_path"]).name, products.feature_mask, image_size)
        mvs_path = _save_mvs_image(mvs_root / Path(row["relative_path"]).with_suffix(".jpg").name, image_rgb, products.mvs_mask)
        old["mask_diagnostics"] = mask_diagnostics(products.full_mask)
        old["mask_diagnostics"]["removed_disconnected_pixels"] = int(removed)
        old["repair"] = {"type": "retain_largest_connected_component", "removed_pixels": int(removed)}
        old["products"] = {
            "full_mask": {"path": str(mask_path.resolve()), "sha256": sha256_file(mask_path)},
            "feature_mask": {"path": str(feature_path.resolve()), "sha256": sha256_file(feature_path)},
            "mvs_image": {"path": str(mvs_path.resolve()), "sha256": sha256_file(mvs_path)},
        }
        existing_records[row["relative_path"]] = old
        repaired_count += 1

    pending = []
    for row in selected:
        size = (int(row["width"]), int(row["height"]))
        previous = existing_records.get(row["relative_path"])
        if previous and _existing_products_valid(previous, image_size=size):
            continue
        pending.append(row)

    if not pending:
        records = [existing_records[row["relative_path"]] for row in selected]
        summary = {
            "schema_version": 1,
            "status": "complete",
            "capability": capability,
            "model_identity": model_identity,
            "source_manifest": str(manifest),
            "source_manifest_sha256": sha256_file(manifest),
            "selected_geometry_count": len(selected),
            "completed_count": len(records),
            "resumed_without_inference": True,
            "repaired_disconnected_component_count": repaired_count,
            "refresh_interval": refresh_interval,
            "dino_device": dino_device,
            "sam_device": sam_device,
            "board_refinement": {
                "status": "deferred_until_registered_empty_reference",
                "negative_only": True,
                "available_sequences": ["empty_g8", "empty_g9"],
            },
            "records_path": str(index),
        }
        write_json(index, {"schema_version": 1, "records": records})
        write_json(report, summary)
        return summary

    # Imports stay inside the execution function so path/contract tests do not
    # require CUDA or either heavyweight model package.
    import torch
    from groundingdino.util.inference import load_image as dino_load_image
    from groundingdino.util.inference import load_model as dino_load_model
    from groundingdino.util.inference import predict as dino_predict
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    dino_config_path = Path(model_identity["groundingdino"]["config"]["path"])
    dino_model = dino_load_model(
        str(dino_config_path),
        str(Path(grounding_checkpoint).resolve()),
        device=dino_device,
    )
    sam_model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        str(Path(sam2_checkpoint).resolve()),
        device=sam_device,
    )
    predictor = SAM2ImagePredictor(sam_model)
    records = dict(existing_records)
    started = time.time()
    current_ring = None
    previous_mask: np.ndarray | None = None
    previous_box: list[float] | None = None
    try:
        for ordinal, row in enumerate(selected, start=1):
            relative_path = row["relative_path"]
            source_path = source / relative_path
            image_size = (int(row["width"]), int(row["height"]))
            old = records.get(relative_path)
            if old and _existing_products_valid(old, image_size=image_size):
                with Image.open(Path(old["products"]["full_mask"]["path"])) as handle:
                    previous_mask = np.asarray(handle.convert("L"), dtype=np.uint8)
                previous_box = _box_from_mask(previous_mask)
                current_ring = row.get("logical_ring_id")
                continue

            ring = row.get("logical_ring_id") or "unknown"
            if ring != current_ring:
                current_ring = ring
                previous_mask = None
                previous_box = None
            detect = previous_box is None or int(row.get("frame_index_within_logical_ring") or 0) % refresh_interval == 0
            if detect:
                detector_source, detector_image = dino_load_image(str(source_path))
                boxes, scores, phrases = dino_predict(
                    model=dino_model,
                    image=detector_image,
                    caption=DEFAULT_DINO_PROMPT,
                    box_threshold=DEFAULT_DINO_BOX_THRESHOLD,
                    text_threshold=DEFAULT_DINO_TEXT_THRESHOLD,
                    device=dino_device,
                )
                box, detector_score = _detector_box(boxes, scores, image_size)
                image_rgb = np.asarray(detector_source)
                detector_phrase = str(phrases[0]) if phrases else DEFAULT_DINO_PROMPT
            else:
                image_rgb = _load_source_rgb(source_path)
                box = list(previous_box or [])
                detector_score = None
                detector_phrase = "propagated_previous_mask_box"
            if image_rgb.shape[:2] != (image_size[1], image_size[0]):
                raise ValueError(f"decoded image shape mismatch for {relative_path}: {image_rgb.shape}")

            predictor.set_image(image_rgb)
            try:
                full_mask, sam_score = _sam_mask(predictor, box, image_rgb.shape[:2])
            except SegmentationCapabilityError:
                if detect:
                    raise
                detector_source, detector_image = dino_load_image(str(source_path))
                boxes, scores, phrases = dino_predict(
                    model=dino_model,
                    image=detector_image,
                    caption=DEFAULT_DINO_PROMPT,
                    box_threshold=DEFAULT_DINO_BOX_THRESHOLD,
                    text_threshold=DEFAULT_DINO_TEXT_THRESHOLD,
                    device=dino_device,
                )
                box, detector_score = _detector_box(boxes, scores, image_size)
                detector_phrase = str(phrases[0]) if phrases else DEFAULT_DINO_PROMPT
                predictor.set_image(image_rgb)
                full_mask, sam_score = _sam_mask(predictor, box, image_rgb.shape[:2])

            full_mask, removed_component_pixels = retain_vessel_component(full_mask)
            diagnostics = mask_diagnostics(full_mask)
            diagnostics["removed_disconnected_pixels"] = removed_component_pixels
            if diagnostics["foreground_fraction"] < 0.001 or diagnostics["foreground_fraction"] > 0.75:
                raise SegmentationCapabilityError(
                    f"implausible SAM 2 mask for {relative_path}: {diagnostics['foreground_fraction']:.6f}"
                )
            products = derive_mask_products(full_mask, inward_margin=2, boundary_band=3)
            products.validate(image_size)
            mask_path = write_mask(mask_root / Path(relative_path).name, products.full_mask, image_size)
            feature_path = write_mask(feature_root / Path(relative_path).name, products.feature_mask, image_size)
            mvs_path = _save_mvs_image(mvs_root / Path(relative_path).with_suffix(".jpg").name, image_rgb, products.mvs_mask)
            previous_mask = products.full_mask
            previous_box = _box_from_mask(previous_mask)
            records[relative_path] = {
                "relative_path": relative_path,
                "source_sha256": row["sha256"],
                "logical_ring_id": ring,
                "frame_index_within_logical_ring": int(row.get("frame_index_within_logical_ring") or 0),
                "phase_01": row.get("phase_01", ""),
                "detector": {
                    "performed": bool(detect),
                    "device": dino_device,
                    "box_xyxy": [float(value) for value in box],
                    "score": detector_score,
                    "phrase": detector_phrase,
                },
                "sam2": {"model": "sam2.1_hiera_small", "device": sam_device, "score": float(sam_score)},
                "mask_diagnostics": diagnostics,
                "board_refinement": {
                    "status": "deferred_until_registered_empty_reference",
                    "negative_only": True,
                },
                "products": {
                    "full_mask": {"path": str(mask_path.resolve()), "sha256": sha256_file(mask_path)},
                    "feature_mask": {"path": str(feature_path.resolve()), "sha256": sha256_file(feature_path)},
                    "mvs_image": {"path": str(mvs_path.resolve()), "sha256": sha256_file(mvs_path)},
                },
                "model_identity_fingerprint": model_identity["fingerprint"],
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            write_json(index, {"schema_version": 1, "records": [records[row["relative_path"]] for row in selected if row["relative_path"] in records]})
            if ordinal % 10 == 0 or ordinal == len(selected):
                print(f"isolation {ordinal}/{len(selected)} complete", flush=True)
    finally:
        del predictor, sam_model, dino_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ordered_records = [records[row["relative_path"]] for row in selected]
    summary = {
        "schema_version": 1,
        "status": "complete",
        "capability": capability,
        "model_identity": model_identity,
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256_file(manifest),
        "selected_geometry_count": len(selected),
        "completed_count": len(ordered_records),
        "resumed_count": len(selected) - len(pending),
        "inference_count": len(pending),
        "repaired_disconnected_component_count": repaired_count,
        "elapsed_seconds": round(time.time() - started, 3),
        "refresh_interval": refresh_interval,
        "dino_device": dino_device,
        "sam_device": sam_device,
        "board_refinement": {
            "status": "deferred_until_registered_empty_reference",
            "negative_only": True,
            "available_sequences": ["empty_g8", "empty_g9"],
        },
        "records_path": str(index),
    }
    write_json(index, {"schema_version": 1, "records": ordered_records})
    write_json(report, summary)
    return summary


__all__ = [
    "MaskProducts",
    "SegmentationCapabilityError",
    "binary_mask",
    "build_mvs_image",
    "derive_mask_products",
    "filter_keypoints",
    "filter_keypoints_to_mask",
    "keypoint_mask_membership",
    "map_keypoints_to_original",
    "mask_diagnostics",
    "official_model_identity",
    "refine_board_leakage",
    "retain_vessel_component",
    "require_segmentation_capability",
    "segmentation_capability",
    "write_mask",
    "write_mask_diagnostics",
    "run_official_isolation",
]
