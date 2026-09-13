"""V4 ALIKED-N16Rot feature cache with vessel-mask provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
import json
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from v4_config import RECONSTRUCTION_V4_ROOT, assert_output_path, fingerprint, sha256_file, write_json
from v4_isolation import binary_mask
from v4_matching import feature_cache_identity, filter_feature_keypoints


@dataclass(frozen=True)
class V4FeatureConfig:
    model_name: str = "aliked-n16rot"
    max_image_size: int = 2048
    max_keypoints: int = 4096
    detection_threshold: float = 0.20
    device: str = "cuda"

    def validate(self) -> "V4FeatureConfig":
        if self.model_name != "aliked-n16rot":
            raise ValueError("V4 feature extraction is frozen to ALIKED-N16Rot")
        if self.max_image_size < 1 or self.max_keypoints < 1:
            raise ValueError("feature limits must be positive")
        if not 0.0 <= float(self.detection_threshold) <= 1.0:
            raise ValueError("detection_threshold must be in [0, 1]")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("feature device must be cpu or cuda")
        return self


@dataclass(frozen=True)
class V4ImageFeatures:
    keypoints: np.ndarray
    descriptors: np.ndarray
    scores: np.ndarray
    image_size: np.ndarray

    @property
    def count(self) -> int:
        return int(self.keypoints.shape[0])

    def validate(self) -> "V4ImageFeatures":
        keypoints = np.asarray(self.keypoints)
        descriptors = np.asarray(self.descriptors)
        scores = np.asarray(self.scores)
        image_size = np.asarray(self.image_size)
        if keypoints.ndim != 2 or keypoints.shape[1] != 2:
            raise ValueError("keypoints must have shape N x 2")
        if descriptors.ndim != 2 or descriptors.shape[0] != keypoints.shape[0]:
            raise ValueError("descriptors must have shape N x D matching keypoints")
        if scores.ndim != 1 or scores.shape[0] != keypoints.shape[0]:
            raise ValueError("scores must have shape N matching keypoints")
        if image_size.shape != (2,) or np.any(image_size <= 0):
            raise ValueError("image_size must contain positive width and height")
        if self.count < 8:
            raise ValueError(f"ALIKED returned too few vessel-supported keypoints: {self.count}")
        if not all(np.isfinite(values).all() for values in (keypoints, descriptors, scores, image_size)):
            raise ValueError("learned feature arrays must be finite")
        width, height = float(image_size[0]), float(image_size[1])
        if np.any(keypoints[:, 0] < -1.0) or np.any(keypoints[:, 0] > width + 1.0):
            raise ValueError("keypoint x coordinates exceed source bounds")
        if np.any(keypoints[:, 1] < -1.0) or np.any(keypoints[:, 1] > height + 1.0):
            raise ValueError("keypoint y coordinates exceed source bounds")
        return self


@dataclass(frozen=True)
class V4Frontend:
    extractor: Any
    matcher: Any
    device: str


def _distribution_direct_url(name: str) -> dict[str, Any] | None:
    try:
        text = importlib.metadata.distribution(name).read_text("direct_url.json")
    except (importlib.metadata.PackageNotFoundError, FileNotFoundError):
        return None
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def lightglue_identity(config: V4FeatureConfig = V4FeatureConfig()) -> dict[str, Any]:
    config.validate()
    try:
        lightglue_version = importlib.metadata.version("lightglue")
    except importlib.metadata.PackageNotFoundError:
        lightglue_version = "missing"
    try:
        kornia_version = importlib.metadata.version("kornia")
    except importlib.metadata.PackageNotFoundError:
        kornia_version = "missing"
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_available = bool(torch.cuda.is_available())
    except Exception as error:  # pragma: no cover - only minimal environments
        torch_version = f"unavailable:{type(error).__name__}"
        cuda_available = False
    checkpoint_candidates = []
    try:
        import torch

        checkpoint_candidates = [Path(torch.hub.get_dir()) / "checkpoints" / "aliked-n16rot.pth"]
    except Exception:
        pass
    checkpoint = next((path for path in checkpoint_candidates if path.is_file()), None)
    checkpoint_info: dict[str, Any] = {"path": str(checkpoint.resolve()) if checkpoint else "", "status": "missing"}
    if checkpoint is not None:
        checkpoint_info.update({"status": "available", "bytes": int(checkpoint.stat().st_size), "sha256": sha256_file(checkpoint)})
    payload = {
        "frontend": "aliked_lightglue",
        "aliked_model_name": config.model_name,
        "max_image_size": config.max_image_size,
        "max_keypoints": config.max_keypoints,
        "detection_threshold": config.detection_threshold,
        "device": config.device,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "lightglue_version": lightglue_version,
        "lightglue_direct_url": _distribution_direct_url("lightglue"),
        "kornia_version": kornia_version,
        "aliked_checkpoint": checkpoint_info,
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def load_frontend(config: V4FeatureConfig = V4FeatureConfig()) -> V4Frontend:
    config.validate()
    import torch
    from lightglue import ALIKED, LightGlue

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V4 ALIKED/LightGlue requested CUDA but torch reports no CUDA device")
    extractor = ALIKED(
        model_name=config.model_name,
        max_num_keypoints=config.max_keypoints,
        detection_threshold=config.detection_threshold,
    ).eval().to(config.device)
    matcher = LightGlue(
        features="aliked",
        depth_confidence=0.95,
        width_confidence=0.99,
    ).eval().to(config.device)
    return V4Frontend(extractor=extractor, matcher=matcher, device=config.device)


def _extract_raw(image_path: Path, frontend: V4Frontend, config: V4FeatureConfig) -> V4ImageFeatures:
    config.validate()
    if not image_path.is_file():
        raise ValueError(f"feature image is missing: {image_path}")
    import torch
    from lightglue.utils import load_image, rbd

    image = load_image(image_path).to(frontend.device)
    with torch.inference_mode():
        result = rbd(frontend.extractor.extract(image, resize=config.max_image_size))
    try:
        features = V4ImageFeatures(
            keypoints=np.asarray(result["keypoints"].detach().cpu(), dtype=np.float32),
            descriptors=np.asarray(result["descriptors"].detach().cpu(), dtype=np.float32),
            scores=np.asarray(result["keypoint_scores"].detach().cpu(), dtype=np.float32),
            image_size=np.asarray(result["image_size"].detach().cpu(), dtype=np.float32),
        )
    except (KeyError, AttributeError, TypeError) as error:
        raise RuntimeError("ALIKED extraction returned an unsupported feature structure") from error
    return features.validate()


def extract_masked_features(
    image_path: str | Path,
    feature_mask_path: str | Path,
    full_mask_path: str | Path,
    frontend: V4Frontend,
    config: V4FeatureConfig = V4FeatureConfig(),
) -> tuple[V4ImageFeatures, dict[str, Any]]:
    """Extract ALIKED features and discard every point outside the feature mask."""

    raw = _extract_raw(Path(image_path), frontend, config)
    with Image.open(feature_mask_path) as handle:
        feature_mask = np.asarray(handle.convert("L"), dtype=np.uint8)
    with Image.open(full_mask_path) as handle:
        full_mask = np.asarray(handle.convert("L"), dtype=np.uint8)
    if feature_mask.shape != full_mask.shape:
        raise ValueError("feature and full mask dimensions differ")
    boundary = np.where((full_mask > 0) & (feature_mask == 0), 255, 0).astype(np.uint8)
    filtered, indices = filter_feature_keypoints(
        raw.keypoints,
        binary_mask(feature_mask),
        boundary_exclusion=boundary,
    )
    if len(indices) < 8:
        raise ValueError(f"fewer than eight mask-supported ALIKED keypoints for {image_path}: {len(indices)}")
    features = V4ImageFeatures(
        keypoints=np.asarray(filtered, dtype=np.float32),
        descriptors=np.asarray(raw.descriptors[indices], dtype=np.float32),
        scores=np.asarray(raw.scores[indices], dtype=np.float32),
        image_size=np.asarray(raw.image_size, dtype=np.float32),
    ).validate()
    return features, {
        "raw_keypoints": raw.count,
        "mask_supported_keypoints": features.count,
        "discarded_outside_mask": int(raw.count - features.count),
        "feature_mask_sha256": sha256_file(feature_mask_path),
        "full_mask_sha256": sha256_file(full_mask_path),
    }


def _cache_path(cache_dir: Path, relative_path: str) -> Path:
    name = Path(relative_path).name
    return assert_output_path(cache_dir / f"feature_{name}.npz")


def _save_feature(path: Path, features: V4ImageFeatures, *, source_sha256: str, feature_mask_sha256: str, cache_identity: str) -> None:
    features.validate()
    path = assert_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        keypoints=np.asarray(features.keypoints, dtype=np.float32),
        descriptors=np.asarray(features.descriptors, dtype=np.float32),
        scores=np.asarray(features.scores, dtype=np.float32),
        image_size=np.asarray(features.image_size, dtype=np.float32),
        source_sha256=np.asarray(source_sha256),
        feature_mask_sha256=np.asarray(feature_mask_sha256),
        cache_identity=np.asarray(cache_identity),
    )


def _load_feature(path: Path, *, source_sha256: str, feature_mask_sha256: str, cache_identity: str) -> V4ImageFeatures | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as values:
            if str(values["source_sha256"].item()) != source_sha256:
                return None
            if str(values["feature_mask_sha256"].item()) != feature_mask_sha256:
                return None
            if str(values["cache_identity"].item()) != cache_identity:
                return None
            features = V4ImageFeatures(
                keypoints=np.asarray(values["keypoints"], dtype=np.float32),
                descriptors=np.asarray(values["descriptors"], dtype=np.float32),
                scores=np.asarray(values["scores"], dtype=np.float32),
                image_size=np.asarray(values["image_size"], dtype=np.float32),
            )
        return features.validate()
    except (OSError, KeyError, ValueError, TypeError, AttributeError):
        return None


def prepare_v4_feature_cache(
    *,
    isolation_records_path: str | Path = RECONSTRUCTION_V4_ROOT / "work" / "isolation_records.json",
    cache_dir: str | Path = RECONSTRUCTION_V4_ROOT / "work" / "features",
    marker_path: str | Path = RECONSTRUCTION_V4_ROOT / "work" / "features_manifest.json",
    config: V4FeatureConfig = V4FeatureConfig(),
    frontend: V4Frontend | None = None,
) -> tuple[dict[str, V4ImageFeatures], dict[str, Any]]:
    """Build or resume the V4 vessel-only feature cache."""

    config.validate()
    records_payload = json.loads(Path(isolation_records_path).read_text(encoding="utf-8"))
    records = list(records_payload.get("records", []))
    if not records:
        raise ValueError("isolation records are empty")
    cache = assert_output_path(cache_dir)
    marker = assert_output_path(marker_path)
    cache.mkdir(parents=True, exist_ok=True)
    model_identity = lightglue_identity(config)
    resolved_frontend = frontend or load_frontend(config)
    features: dict[str, V4ImageFeatures] = {}
    item_records: list[dict[str, Any]] = []
    started = __import__("time").time()
    for ordinal, record in enumerate(records, start=1):
        relative_path = str(record["relative_path"])
        source_sha = str(record["source_sha256"])
        feature_mask_sha = str(record["products"]["feature_mask"]["sha256"])
        identity = feature_cache_identity(
            source_sha256=source_sha,
            feature_mask_sha256=feature_mask_sha,
            model_identity=model_identity,
        )
        output = _cache_path(cache, relative_path)
        cached = _load_feature(output, source_sha256=source_sha, feature_mask_sha256=feature_mask_sha, cache_identity=identity)
        stats: dict[str, Any]
        if cached is not None:
            current = cached
            stats = {"raw_keypoints": None, "mask_supported_keypoints": current.count, "discarded_outside_mask": None, "resumed": True}
        else:
            current, stats = extract_masked_features(
                record["products"]["mvs_image"]["path"],
                record["products"]["feature_mask"]["path"],
                record["products"]["full_mask"]["path"],
                resolved_frontend,
                config,
            )
            stats["resumed"] = False
            _save_feature(output, current, source_sha256=source_sha, feature_mask_sha256=feature_mask_sha, cache_identity=identity)
        features[relative_path] = current
        item_records.append({
            "relative_path": relative_path,
            "logical_ring_id": record.get("logical_ring_id", ""),
            "frame_index_within_logical_ring": int(record.get("frame_index_within_logical_ring") or 0),
            "source_sha256": source_sha,
            "feature_mask_sha256": feature_mask_sha,
            "cache_identity": identity,
            "path": str(output.resolve()),
            "sha256": sha256_file(output),
            **stats,
        })
        write_json(marker, {"schema_version": 1, "status": "running", "model_identity": model_identity, "config": asdict(config), "completed_count": ordinal, "total_count": len(records), "items": item_records})
        if ordinal % 10 == 0 or ordinal == len(records):
            print(f"features {ordinal}/{len(records)} complete", flush=True)
    counts = [item["mask_supported_keypoints"] for item in item_records]
    summary = {
        "schema_version": 1,
        "status": "complete",
        "model_identity": model_identity,
        "config": asdict(config),
        "selected_geometry_count": len(records),
        "completed_count": len(features),
        "keypoint_count": {"min": min(counts), "median": statistics.median(counts), "max": max(counts)},
        "elapsed_seconds": round(__import__("time").time() - started, 3),
        "items": item_records,
    }
    write_json(marker, summary)
    return features, summary


__all__ = [
    "V4FeatureConfig",
    "V4Frontend",
    "V4ImageFeatures",
    "extract_masked_features",
    "lightglue_identity",
    "load_frontend",
    "prepare_v4_feature_cache",
]
