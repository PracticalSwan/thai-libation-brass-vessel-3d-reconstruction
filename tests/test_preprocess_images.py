from pathlib import Path

import cv2
import numpy as np
import pytest

from preprocess_images import (
    encode_preprocessed_jpeg,
    preprocess_image,
    preprocess_image_array,
)


def _gradient_image(height: int = 120, width: int = 80) -> np.ndarray:
    gradient = np.linspace(45, 205, width, dtype=np.uint8)
    gray = np.tile(gradient, (height, 1))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_preprocessing_preserves_geometry_type_and_is_deterministic() -> None:
    """Catches a photometric operation changing geometry, type, or repeatability."""
    source = _gradient_image()

    first = preprocess_image_array(source)
    second = preprocess_image_array(source)

    assert first.shape == source.shape
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)


def test_preprocessing_changes_photometry_mildly_without_moving_features() -> None:
    """Catches a no-op or an unexpectedly aggressive tonal transform."""
    source = _gradient_image()

    result = preprocess_image_array(source)
    mean_absolute_change = float(
        np.mean(np.abs(result.astype(np.int16) - source.astype(np.int16)))
    )

    assert 0.5 < mean_absolute_change < 15.0


def test_preprocessed_file_is_written_and_readable(tmp_path: Path) -> None:
    """Catches an output path that is missing, invalid, or unreadable by OpenCV."""
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "nested" / "output.png"
    source = _gradient_image(height=64, width=96)
    assert cv2.imwrite(str(source_path), source)

    preprocess_image(source_path, output_path)
    reopened = cv2.imread(str(output_path), cv2.IMREAD_COLOR)

    assert output_path.is_file()
    assert reopened is not None
    assert reopened.shape == source.shape
    assert reopened.dtype == np.uint8


def test_preprocess_rejects_in_place_write_without_changing_source(
    tmp_path: Path,
) -> None:
    """Catches a direct API call overwriting immutable source evidence."""
    source_path = tmp_path / "source.jpg"
    assert cv2.imwrite(str(source_path), _gradient_image())
    original_bytes = source_path.read_bytes()

    with pytest.raises(ValueError, match="source and output paths must differ"):
        preprocess_image(source_path, source_path)

    assert source_path.read_bytes() == original_bytes


def test_jpeg_encoding_is_deterministic_decodable_and_matches_file_export(
    tmp_path: Path,
) -> None:
    """Catches SIFT evaluating bytes different from the final JPEG artifact."""
    source = _gradient_image(height=96, width=128)
    first = encode_preprocessed_jpeg(source, quality=95)
    second = encode_preprocessed_jpeg(source, quality=95)
    decoded = cv2.imdecode(np.frombuffer(first, dtype=np.uint8), cv2.IMREAD_COLOR)
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "output.jpg"
    assert cv2.imwrite(str(source_path), source)

    preprocess_image(source_path, output_path)

    assert first == second == output_path.read_bytes()
    assert decoded is not None
    assert decoded.shape == source.shape
    assert decoded.dtype == np.uint8
