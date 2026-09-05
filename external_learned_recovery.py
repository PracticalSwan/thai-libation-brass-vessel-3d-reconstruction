"""Step 13 external ALIKED + LightGlue sparse-recovery helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sqlite3
import sys
import time
from typing import Callable, Mapping, Sequence

import numpy as np
import pycolmap
import torch

from analysis_common import SelectedImageRecord, verify_selected_images
from sparse_bridging import BridgeSearchConfig, bridge_model_accepted
from sparse_reconstruction import (
    AttemptMetrics,
    ModelMetrics,
    SparseRunConfig,
    build_image_reader_options,
    map_sparse_database,
    summarize_database,
    validate_workspace_boundary,
)


PINNED_LIGHTGLUE_COMMIT = "eb42fee2d71449efb0aa5c10549752b5d75384d8"
LIGHTGLUE_REPOSITORY = "https://github.com/cvg/LightGlue.git"


@dataclass(frozen=True)
class ExternalLearnedConfig:
    expected_images: int = 288
    frontend_name: str = "aliked_lightglue"
    aliked_model_name: str = "aliked-n16rot"
    max_image_size: int = 1600
    max_keypoints: int = 4096
    sequential_overlap: int = 20
    pinned_commit: str = PINNED_LIGHTGLUE_COMMIT

    def validate(self) -> "ExternalLearnedConfig":
        if self.expected_images < 2:
            raise ValueError("expected_images must be at least two")
        if self.frontend_name != "aliked_lightglue":
            raise ValueError("Step 13 frontend is frozen to aliked_lightglue")
        if self.aliked_model_name != "aliked-n16rot":
            raise ValueError("Step 13 ALIKED model is frozen to aliked-n16rot")
        if self.max_image_size < 1 or self.max_keypoints < 1:
            raise ValueError("learned inference limits must be positive")
        if self.sequential_overlap < 1:
            raise ValueError("sequential overlap must be positive")
        if self.pinned_commit != PINNED_LIGHTGLUE_COMMIT:
            raise ValueError("Step 13 LightGlue source commit is frozen")
        return self


@dataclass(frozen=True)
class ImageFeatures:
    keypoints: np.ndarray
    descriptors: np.ndarray
    scores: np.ndarray
    image_size: np.ndarray

    @property
    def count(self) -> int:
        return int(self.keypoints.shape[0])


@dataclass(frozen=True)
class FrontendBundle:
    extractor: object
    matcher: object
    device: str


@dataclass(frozen=True)
class PairMatchingMetrics:
    pair_count: int
    raw_match_count: int
    runtime_seconds: float


@dataclass(frozen=True)
class ExternalMappingResult:
    attempt: AttemptMetrics
    pair_metrics: PairMatchingMetrics


def resolve_device(torch_module: object = torch) -> str:
    cuda = getattr(torch_module, "cuda")
    return "cuda" if bool(cuda.is_available()) else "cpu"


def lightglue_provenance() -> dict[str, object]:
    distribution = importlib.metadata.distribution("lightglue")
    direct_url_text = distribution.read_text("direct_url.json")
    direct_url: dict[str, object] | None = None
    if direct_url_text:
        try:
            parsed = json.loads(direct_url_text)
            if isinstance(parsed, dict):
                direct_url = parsed
        except json.JSONDecodeError:
            direct_url = None
    return {
        "distribution_version": distribution.version,
        "direct_url": direct_url,
        "repository": LIGHTGLUE_REPOSITORY,
        "pinned_commit": PINNED_LIGHTGLUE_COMMIT,
    }


def runtime_snapshot(config: ExternalLearnedConfig = ExternalLearnedConfig()) -> dict[str, object]:
    config.validate()
    device = resolve_device()
    gpu_name = ""
    if device == "cuda":
        gpu_name = str(torch.cuda.get_device_name(0))
    try:
        torchvision_version = importlib.metadata.version("torchvision")
    except importlib.metadata.PackageNotFoundError:
        torchvision_version = ""
    try:
        kornia_version = importlib.metadata.version("kornia")
    except importlib.metadata.PackageNotFoundError:
        kornia_version = ""
    return {
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "torchvision_version": torchvision_version,
        "kornia_version": kornia_version,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": str(torch.version.cuda or ""),
        "gpu_name": gpu_name,
        "pycolmap_version": str(pycolmap.__version__),
        "device": device,
        "frontend": config.frontend_name,
        "aliked_model_name": config.aliked_model_name,
        "max_image_size": config.max_image_size,
        "max_keypoints": config.max_keypoints,
        "lightglue": lightglue_provenance(),
    }


def validate_image_features(features: ImageFeatures) -> None:
    keypoints = np.asarray(features.keypoints)
    descriptors = np.asarray(features.descriptors)
    scores = np.asarray(features.scores)
    image_size = np.asarray(features.image_size)
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError("keypoints must have shape N x 2")
    if descriptors.ndim != 2:
        raise ValueError("descriptors must have shape N x D")
    if descriptors.shape[0] != keypoints.shape[0]:
        raise ValueError("descriptor count does not match keypoint count")
    if scores.ndim != 1 or scores.shape[0] != keypoints.shape[0]:
        raise ValueError("score count does not match keypoint count")
    if image_size.shape != (2,) or np.any(image_size <= 0):
        raise ValueError("image_size must contain positive width and height")
    if keypoints.shape[0] <= 0:
        raise ValueError("learned extraction returned no keypoints")
    if not np.isfinite(keypoints).all() or not np.isfinite(descriptors).all() or not np.isfinite(scores).all():
        raise ValueError("learned features must be finite")
    width, height = float(image_size[0]), float(image_size[1])
    tolerance = 1.0
    if (
        np.any(keypoints[:, 0] < -tolerance)
        or np.any(keypoints[:, 0] > width + tolerance)
        or np.any(keypoints[:, 1] < -tolerance)
        or np.any(keypoints[:, 1] > height + tolerance)
    ):
        raise ValueError("learned keypoint coordinates exceed source image bounds")


def validate_features_for_record(
    features: ImageFeatures, record: SelectedImageRecord
) -> None:
    validate_image_features(features)
    source_size = np.asarray([record.width, record.height], dtype=np.float32)
    if not np.array_equal(np.asarray(features.image_size, dtype=np.float32), source_size):
        raise ValueError(
            f"learned feature coordinate frame does not match source image dimensions for "
            f"{record.filename}: {tuple(np.asarray(features.image_size).tolist())} != "
            f"({record.width}, {record.height})"
        )


def colmap_keypoints(features: ImageFeatures) -> np.ndarray:
    validate_image_features(features)
    return (np.asarray(features.keypoints, dtype=np.float32) + 0.5).astype(np.float32, copy=False)


def validate_matches(matches: object, first_count: int, second_count: int) -> np.ndarray:
    array = np.asarray(matches)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("learned matches must have shape N x 2")
    if array.size == 0:
        return np.empty((0, 2), dtype=np.uint32)
    if not np.issubdtype(array.dtype, np.integer):
        if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
            raise ValueError("learned match indices must be integers")
    signed = array.astype(np.int64, copy=False)
    if np.any(signed < 0):
        raise ValueError("learned match indices must be nonnegative")
    if np.any(signed[:, 0] >= first_count) or np.any(signed[:, 1] >= second_count):
        raise ValueError("learned match index is out of bounds")
    return signed.astype(np.uint32, copy=False)


def _normalized_pair(first: str, second: str) -> tuple[str, str]:
    if not first or not second or first == second:
        raise ValueError("image pair requires two distinct filenames")
    return (first, second) if first < second else (second, first)


def generate_sequential_pairs(
    records: Sequence[SelectedImageRecord], *, overlap: int
) -> tuple[tuple[str, str], ...]:
    if overlap < 1:
        raise ValueError("sequential overlap must be positive")
    ordered = tuple(sorted(records, key=lambda record: record.index))
    if len({record.index for record in ordered}) != len(ordered):
        raise ValueError("selected image indices must be unique")
    pairs: list[tuple[str, str]] = []
    for position, record in enumerate(ordered):
        for other in ordered[position + 1 : position + overlap + 1]:
            pairs.append(_normalized_pair(record.filename, other.filename))
    if len(set(pairs)) != len(pairs):
        raise RuntimeError("sequential pair schedule contains duplicates")
    return tuple(pairs)


def merge_pair_schedule(
    sequential_pairs: Sequence[tuple[str, str]],
    bridge_pairs: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    merged: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in (*sequential_pairs, *bridge_pairs):
        normalized = _normalized_pair(*pair)
        if normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
    return tuple(merged)


def _model_rank(model: ModelMetrics) -> tuple[int, int, float]:
    error = model.mean_reprojection_error
    finite_error = error if math.isfinite(error) else float("inf")
    return model.registered_images, model.sparse_points, -finite_error


def choose_local_fallback(
    *candidates: tuple[str, ModelMetrics]
) -> tuple[str, ModelMetrics]:
    if not candidates:
        raise ValueError("at least one local sparse fallback is required")
    return max(candidates, key=lambda item: _model_rank(item[1]))


def external_model_metric_accepted(
    model: ModelMetrics, bridge_config: BridgeSearchConfig = BridgeSearchConfig()
) -> bool:
    return bridge_model_accepted(model, bridge_config)


def load_frontend(
    config: ExternalLearnedConfig = ExternalLearnedConfig(), *, device: str | None = None
) -> FrontendBundle:
    config.validate()
    from lightglue import ALIKED, LightGlue

    resolved = device or resolve_device()
    extractor = ALIKED(
        model_name=config.aliked_model_name,
        max_num_keypoints=config.max_keypoints,
    ).eval().to(resolved)
    matcher = LightGlue(features="aliked").eval().to(resolved)
    return FrontendBundle(extractor=extractor, matcher=matcher, device=resolved)


def extract_image_features(
    image_path: Path,
    bundle: FrontendBundle,
    config: ExternalLearnedConfig = ExternalLearnedConfig(),
) -> ImageFeatures:
    config.validate()
    if not image_path.is_file():
        raise ValueError(f"selected image is missing: {image_path}")
    from lightglue.utils import load_image, rbd

    image = load_image(image_path).to(bundle.device)
    with torch.inference_mode():
        result = rbd(bundle.extractor.extract(image, resize=config.max_image_size))  # type: ignore[attr-defined]
    try:
        features = ImageFeatures(
            keypoints=np.asarray(result["keypoints"].detach().cpu(), dtype=np.float32),
            descriptors=np.asarray(result["descriptors"].detach().cpu(), dtype=np.float32),
            scores=np.asarray(result["keypoint_scores"].detach().cpu(), dtype=np.float32),
            image_size=np.asarray(result["image_size"].detach().cpu(), dtype=np.float32),
        )
    except (KeyError, AttributeError, TypeError) as error:
        raise RuntimeError("ALIKED extraction returned an unsupported feature structure") from error
    validate_image_features(features)
    return features


def _feature_tensor_payload(features: ImageFeatures, device: str) -> dict[str, torch.Tensor]:
    validate_image_features(features)
    return {
        "keypoints": torch.from_numpy(np.asarray(features.keypoints, dtype=np.float32)).to(device)[None],
        "descriptors": torch.from_numpy(np.asarray(features.descriptors, dtype=np.float32)).to(device)[None],
        "keypoint_scores": torch.from_numpy(np.asarray(features.scores, dtype=np.float32)).to(device)[None],
        "image_size": torch.from_numpy(np.asarray(features.image_size, dtype=np.float32)).to(device)[None],
    }


def match_feature_pair(
    first: ImageFeatures,
    second: ImageFeatures,
    bundle: FrontendBundle,
) -> np.ndarray:
    from lightglue.utils import rbd

    first_payload = _feature_tensor_payload(first, bundle.device)
    second_payload = _feature_tensor_payload(second, bundle.device)
    with torch.inference_mode():
        prediction = rbd(
            bundle.matcher({"image0": first_payload, "image1": second_payload})  # type: ignore[operator]
        )
    try:
        matches = prediction["matches"].detach().cpu().numpy()
    except (KeyError, AttributeError, TypeError) as error:
        raise RuntimeError("LightGlue returned an unsupported match structure") from error
    return validate_matches(matches, first.count, second.count)


def smoke_frontend(
    first_image: Path,
    second_image: Path,
    config: ExternalLearnedConfig = ExternalLearnedConfig(),
    *,
    frontend_loader: Callable[..., FrontendBundle] = load_frontend,
) -> dict[str, object]:
    first_count = 0
    second_count = 0
    try:
        bundle = frontend_loader(config)
        first = extract_image_features(first_image, bundle, config)
        second = extract_image_features(second_image, bundle, config)
        first_count, second_count = first.count, second.count
        matches = match_feature_pair(first, second, bundle)
        if len(matches) <= 0:
            raise RuntimeError("LightGlue capability smoke returned no matches")
        return {
            "status": "passed",
            "first_features": first_count,
            "second_features": second_count,
            "raw_matches": int(len(matches)),
            "exception_type": "",
            "exception_message": "",
            "device": bundle.device,
        }
    except Exception as error:
        return {
            "status": "blocked",
            "first_features": first_count,
            "second_features": second_count,
            "raw_matches": 0,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "device": resolve_device(),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_marker_base(
    records: Sequence[SelectedImageRecord],
    selection_manifest_sha256: str,
    config: ExternalLearnedConfig,
    torch_version: str,
    device: str,
) -> dict[str, object]:
    config.validate()
    return {
        "selection_manifest_sha256": selection_manifest_sha256,
        "frontend": config.frontend_name,
        "aliked_model_name": config.aliked_model_name,
        "pinned_lightglue_commit": config.pinned_commit,
        "max_image_size": config.max_image_size,
        "max_keypoints": config.max_keypoints,
        "torch_version": torch_version,
        "device": device,
        "image_count": len(records),
        "images": [
            {
                "index": record.index,
                "filename": record.filename,
                "width": record.width,
                "height": record.height,
            }
            for record in records
        ],
    }


def _layout_sha(items: Sequence[dict[str, object]]) -> str:
    encoded = json.dumps(items, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_feature_cache(
    cache_dir: Path,
    records: Sequence[SelectedImageRecord],
    features_by_name: Mapping[str, ImageFeatures],
    *,
    selection_manifest_sha256: str,
    config: ExternalLearnedConfig,
    torch_version: str,
    device: str,
) -> None:
    config.validate()
    if len(records) != config.expected_images:
        raise ValueError(f"expected {config.expected_images} selected records, found {len(records)}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker_path = cache_dir / "features_complete.json"
    temporary_marker = cache_dir / "features_complete.json.tmp"
    items: list[dict[str, object]] = []
    for record in records:
        try:
            features = features_by_name[record.filename]
        except KeyError as error:
            raise ValueError(f"feature cache is missing {record.filename}") from error
        validate_features_for_record(features, record)
        filename = f"feature_{record.index:04d}.npz"
        output = cache_dir / filename
        np.savez_compressed(
            output,
            keypoints=np.asarray(features.keypoints, dtype=np.float32),
            descriptors=np.asarray(features.descriptors, dtype=np.float32),
            scores=np.asarray(features.scores, dtype=np.float32),
            image_size=np.asarray(features.image_size, dtype=np.float32),
        )
        items.append(
            {
                "index": record.index,
                "filename": record.filename,
                "cache_file": filename,
                "keypoint_count": features.count,
                "cache_sha256": _sha256(output),
            }
        )
    marker = {
        **_cache_marker_base(
            records, selection_manifest_sha256, config, torch_version, device
        ),
        "items": items,
        "layout_sha256": _layout_sha(items),
    }
    temporary_marker.write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_marker.replace(marker_path)


def load_feature_cache(
    cache_dir: Path,
    records: Sequence[SelectedImageRecord],
    *,
    selection_manifest_sha256: str,
    config: ExternalLearnedConfig,
    torch_version: str,
    device: str,
) -> dict[str, ImageFeatures] | None:
    marker_path = cache_dir / "features_complete.json"
    if not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if not isinstance(marker, dict):
            return None
        expected_base = _cache_marker_base(
            records, selection_manifest_sha256, config, torch_version, device
        )
        for key, value in expected_base.items():
            if marker.get(key) != value:
                return None
        items = marker.get("items")
        if not isinstance(items, list) or marker.get("layout_sha256") != _layout_sha(items):
            return None
        if len(items) != len(records):
            return None
        loaded: dict[str, ImageFeatures] = {}
        for record, item in zip(records, items, strict=True):
            if not isinstance(item, dict) or item.get("filename") != record.filename:
                return None
            path = cache_dir / str(item.get("cache_file", ""))
            if not path.is_file() or _sha256(path) != item.get("cache_sha256"):
                return None
            with np.load(path, allow_pickle=False) as values:
                features = ImageFeatures(
                    keypoints=np.asarray(values["keypoints"], dtype=np.float32),
                    descriptors=np.asarray(values["descriptors"], dtype=np.float32),
                    scores=np.asarray(values["scores"], dtype=np.float32),
                    image_size=np.asarray(values["image_size"], dtype=np.float32),
                )
            validate_features_for_record(features, record)
            if features.count != int(item.get("keypoint_count", -1)):
                return None
            loaded[record.filename] = features
        return loaded
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _clear_known_feature_cache(cache_dir: Path) -> None:
    if not cache_dir.exists():
        return
    if not cache_dir.is_dir() or cache_dir.is_symlink():
        raise ValueError(f"Step 13 feature cache path is not a regular directory: {cache_dir}")
    for path in cache_dir.glob("feature_*.npz"):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Step 13 cache entry is not a regular file: {path}")
        path.unlink()
    for name in ("features_complete.json", "features_complete.json.tmp"):
        path = cache_dir / name
        if path.exists():
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Step 13 cache marker is not a regular file: {path}")
            path.unlink()


def prepare_feature_cache(
    image_dir: Path,
    selection_manifest: Path,
    cache_dir: Path,
    config: ExternalLearnedConfig = ExternalLearnedConfig(),
    *,
    bundle: FrontendBundle | None = None,
    extractor: Callable[[Path, FrontendBundle, ExternalLearnedConfig], ImageFeatures] = extract_image_features,
) -> tuple[dict[str, ImageFeatures], FrontendBundle, str]:
    config.validate()
    verified = verify_selected_images(
        image_dir, selection_manifest, expected_count=config.expected_images
    )
    resolved_bundle = bundle or load_frontend(config)
    cached = load_feature_cache(
        cache_dir,
        verified.records,
        selection_manifest_sha256=verified.manifest_sha256,
        config=config,
        torch_version=str(torch.__version__),
        device=resolved_bundle.device,
    )
    if cached is not None:
        return cached, resolved_bundle, verified.manifest_sha256
    _clear_known_feature_cache(cache_dir)
    started = time.perf_counter()
    features = {
        record.filename: extractor(image_dir / record.filename, resolved_bundle, config)
        for record in verified.records
    }
    write_feature_cache(
        cache_dir,
        verified.records,
        features,
        selection_manifest_sha256=verified.manifest_sha256,
        config=config,
        torch_version=str(torch.__version__),
        device=resolved_bundle.device,
    )
    _ = time.perf_counter() - started
    return features, resolved_bundle, verified.manifest_sha256


def match_pairs(
    features_by_name: Mapping[str, ImageFeatures],
    pairs: Sequence[tuple[str, str]],
    bundle: FrontendBundle,
    *,
    matcher: Callable[[ImageFeatures, ImageFeatures, FrontendBundle], np.ndarray] = match_feature_pair,
    progress: Callable[[int, int, tuple[str, str], int], None] | None = None,
) -> tuple[dict[tuple[str, str], np.ndarray], PairMatchingMetrics]:
    started = time.perf_counter()
    matched: dict[tuple[str, str], np.ndarray] = {}
    raw_total = 0
    for index, pair in enumerate(pairs, start=1):
        normalized = _normalized_pair(*pair)
        if normalized in matched:
            raise ValueError("duplicate learned match pair")
        try:
            first = features_by_name[normalized[0]]
            second = features_by_name[normalized[1]]
        except KeyError as error:
            raise ValueError(f"learned pair references missing features: {error.args[0]}") from error
        matches = validate_matches(
            matcher(first, second, bundle), first.count, second.count
        )
        matched[normalized] = matches
        raw_total += int(len(matches))
        if progress is not None:
            progress(index, len(pairs), normalized, len(matches))
    return matched, PairMatchingMetrics(
        pair_count=len(matched),
        raw_match_count=raw_total,
        runtime_seconds=time.perf_counter() - started,
    )


def _write_pair_file(path: Path, pairs: Sequence[tuple[str, str]]) -> None:
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for pair in pairs:
        normalized = _normalized_pair(*pair)
        if normalized in seen:
            raise ValueError("duplicate pair in imported verification list")
        if any(any(char.isspace() for char in name) for name in normalized):
            raise ValueError("COLMAP pair filenames must not contain whitespace")
        if any(Path(name).name != name for name in normalized):
            raise ValueError("COLMAP pair entries must be filenames only")
        seen.add(normalized)
        lines.append(f"{normalized[0]} {normalized[1]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_verified_match_database(
    image_dir: Path,
    database_path: Path,
    features_by_name: Mapping[str, ImageFeatures],
    matches_by_pair: Mapping[tuple[str, str], np.ndarray],
    pairs_path: Path,
    sparse_config: SparseRunConfig = SparseRunConfig(),
) -> None:
    sparse_config.validate()
    if not image_dir.is_dir():
        raise ValueError(f"selected image directory is missing: {image_dir}")
    validate_workspace_boundary(image_dir, database_path.parent)
    if database_path.exists():
        raise ValueError(f"Step 13 database destination already exists: {database_path}")
    names = tuple(features_by_name)
    if len(names) != sparse_config.expected_images:
        raise ValueError(
            f"Step 13 database requires {sparse_config.expected_images} image features, found {len(names)}"
        )
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with pycolmap.Database.open(database_path):
        pass
    pycolmap.import_images(
        database_path=database_path,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        image_names=list(names),
        options=build_image_reader_options(sparse_config),
    )
    with sqlite3.connect(str(database_path)) as connection:
        image_ids = {
            str(name): int(image_id)
            for image_id, name in connection.execute(
                "SELECT image_id, name FROM images ORDER BY image_id"
            ).fetchall()
        }
        camera_count = int(connection.execute("SELECT COUNT(*) FROM cameras").fetchone()[0])
    if set(image_ids) != set(names):
        raise RuntimeError("Step 13 COLMAP database image names do not match external features")
    if camera_count != 1:
        raise RuntimeError("Step 13 COLMAP database does not use exactly one shared camera")

    ordered_pairs = tuple(matches_by_pair)
    with pycolmap.Database.open(database_path) as database:
        for name in names:
            features = features_by_name[name]
            database.write_keypoints(image_ids[name], colmap_keypoints(features))
        for pair in ordered_pairs:
            first_name, second_name = _normalized_pair(*pair)
            if first_name not in image_ids or second_name not in image_ids:
                raise ValueError("Step 13 match pair references unknown image")
            first_features = features_by_name[first_name]
            second_features = features_by_name[second_name]
            matches = validate_matches(
                matches_by_pair[pair], first_features.count, second_features.count
            )
            database.write_matches(
                image_ids[first_name], image_ids[second_name], matches
            )
    _write_pair_file(pairs_path, ordered_pairs)
    pycolmap.verify_matches(
        database_path,
        pairs_path,
        pycolmap.TwoViewGeometryOptions(),
    )


def run_external_mapping(
    image_dir: Path,
    output_dir: Path,
    work_dir: Path,
    features_by_name: Mapping[str, ImageFeatures],
    pairs: Sequence[tuple[str, str]],
    bundle: FrontendBundle,
    sparse_config: SparseRunConfig = SparseRunConfig(),
    *,
    matcher: Callable[[ImageFeatures, ImageFeatures, FrontendBundle], np.ndarray] = match_feature_pair,
    progress: Callable[[int, int, tuple[str, str], int], None] | None = None,
) -> ExternalMappingResult:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Step 13 sparse output is not empty: {output_dir}")
    validate_workspace_boundary(image_dir, output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    database = work_dir / "mapping.db"
    pair_file = work_dir / "mapping_pairs.txt"
    if database.exists():
        raise ValueError(f"Step 13 mapping database already exists: {database}")
    matches, pair_metrics = match_pairs(
        features_by_name, pairs, bundle, matcher=matcher, progress=progress
    )
    build_verified_match_database(
        image_dir,
        database,
        features_by_name,
        matches,
        pair_file,
        sparse_config,
    )
    started = time.perf_counter()
    models = map_sparse_database(database, image_dir, output_dir, sparse_config)
    mapping_seconds = time.perf_counter() - started
    best = max(models, key=_model_rank)
    attempt = AttemptMetrics(
        name="external_aliked_lightglue",
        workspace=output_dir,
        overlap=ExternalLearnedConfig().sequential_overlap,
        database=summarize_database(database),
        models=tuple(models),
        best_model=best,
        runtime_seconds=pair_metrics.runtime_seconds + mapping_seconds,
        pycolmap_version=str(pycolmap.__version__),
    )
    return ExternalMappingResult(attempt=attempt, pair_metrics=pair_metrics)
