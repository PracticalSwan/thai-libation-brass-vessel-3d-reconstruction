"""Conservative photometric preprocessing that preserves image geometry."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def preprocess_image_array(
    image: np.ndarray,
    *,
    clip_limit: float = 1.5,
    blend: float = 0.15,
) -> np.ndarray:
    """Mildly enhance luminance contrast without moving image features."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("expected a uint8 BGR image with three channels")
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be between 0 and 1")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced_luminance = clahe.apply(luminance)
    blended_luminance = cv2.addWeighted(
        luminance,
        1.0 - blend,
        enhanced_luminance,
        blend,
        0.0,
    )
    enhanced_lab = cv2.merge((blended_luminance, channel_a, channel_b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def encode_preprocessed_jpeg(image: np.ndarray, *, quality: int = 95) -> bytes:
    """Encode the exact deterministic JPEG artifact used by the final export."""
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100")
    output = preprocess_image_array(image)
    encoded, buffer = cv2.imencode(
        ".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not encoded:
        raise OSError("failed to encode preprocessed JPEG")
    return buffer.tobytes()


def preprocess_image(source_path: Path, output_path: Path) -> None:
    if source_path.resolve() == output_path.resolve():
        raise ValueError("source and output paths must differ")
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unreadable image: {source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        output_path.write_bytes(encode_preprocessed_jpeg(image, quality=95))
        return
    output = preprocess_image_array(image)
    if not cv2.imwrite(str(output_path), output):
        raise OSError(f"failed to write image: {output_path}")
