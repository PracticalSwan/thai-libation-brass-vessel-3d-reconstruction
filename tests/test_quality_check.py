from pathlib import Path

import cv2
import numpy as np
import pytest

from quality_check import (
    QualityRecord,
    QualityThresholds,
    analyze_image,
    analyze_image_array,
    calibrate_thresholds,
    decide_quality,
)


def test_sharp_texture_scores_above_blurred_equivalent() -> None:
    """Catches blur scoring that stops responding to lost edge detail."""
    yy, xx = np.indices((256, 256))
    checker = (((xx // 8) + (yy // 8)) % 2 * 255).astype(np.uint8)
    sharp = cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)
    blurred = cv2.GaussianBlur(sharp, (21, 21), 6)

    sharp_record = analyze_image_array(sharp, index=1, filename="sharp.jpg")
    blurred_record = analyze_image_array(blurred, index=2, filename="blurred.jpg")

    assert sharp_record.blur_score > blurred_record.blur_score * 2


def test_clipping_metrics_report_controlled_dark_and_bright_halves() -> None:
    """Catches clipping percentages that use the wrong scale or denominator."""
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    image[:, 40:] = 255

    record = analyze_image_array(image, index=1, filename="clipped.jpg")

    assert record.dark_percent == pytest.approx(50.0)
    assert record.bright_percent == pytest.approx(50.0)


def test_unreadable_input_is_returned_as_an_explicit_record(tmp_path: Path) -> None:
    """Catches unreadable files being silently omitted from the audit."""
    missing = tmp_path / "missing.jpg"

    record = analyze_image(missing, index=7)

    assert record.filename == "missing.jpg"
    assert record.index == 7
    assert record.readable is False
    assert record.error == "unreadable image"


def test_warning_metric_does_not_automatically_reject_image() -> None:
    """Catches warning thresholds being treated as rejection thresholds."""
    thresholds = QualityThresholds(
        blur_warn=100.0,
        blur_reject=10.0,
        brightness_low_warn=50.0,
        brightness_high_warn=210.0,
        contrast_warn=15.0,
        dark_percent_warn=5.0,
        bright_percent_warn=5.0,
        feature_count_warn=100,
        feature_count_reject=20,
    )
    record = QualityRecord(
        index=12,
        filename="soft-but-usable.jpg",
        readable=True,
        width=3072,
        height=4080,
        brightness=130.0,
        contrast=40.0,
        blur_score=60.0,
        dark_percent=0.1,
        bright_percent=0.5,
        feature_count=2000,
    )

    decision = decide_quality(record, thresholds)

    assert decision.label == "WARN"
    assert decision.reasons == ("low sharpness",)


def test_unreadable_and_manual_geometry_exclusions_are_rejected() -> None:
    """Catches true failure cases being retained in the SfM input set."""
    thresholds = QualityThresholds(
        blur_warn=100.0,
        blur_reject=10.0,
        brightness_low_warn=50.0,
        brightness_high_warn=210.0,
        contrast_warn=15.0,
        dark_percent_warn=5.0,
        bright_percent_warn=5.0,
        feature_count_warn=100,
        feature_count_reject=20,
    )
    unreadable = QualityRecord.unreadable(index=3, filename="bad.jpg")
    moved_object = QualityRecord(
        index=289,
        filename="flipped.jpg",
        readable=True,
        width=3072,
        height=4080,
        brightness=140.0,
        contrast=50.0,
        blur_score=200.0,
        dark_percent=0.0,
        bright_percent=1.0,
        feature_count=4000,
    )

    assert decide_quality(unreadable, thresholds).label == "REJECT"
    moved_decision = decide_quality(
        moved_object, thresholds, excluded_indices={289}
    )
    assert moved_decision.label == "REJECT"
    assert moved_decision.reasons == ("separate hand-held/flipped sequence",)


def test_thresholds_are_calibrated_from_eligible_dataset_distribution() -> None:
    """Catches calibration reverting to unrelated hard-coded demo values."""
    records = [
        QualityRecord(
            index=index,
            filename=f"{index}.jpg",
            readable=True,
            width=100,
            height=120,
            brightness=brightness,
            contrast=contrast,
            blur_score=blur,
            dark_percent=dark,
            bright_percent=bright,
            feature_count=features,
        )
        for index, brightness, contrast, blur, dark, bright, features in [
            (1, 100.0, 20.0, 100.0, 0.1, 0.2, 1000),
            (2, 110.0, 30.0, 200.0, 0.2, 0.4, 2000),
            (3, 120.0, 40.0, 300.0, 0.3, 0.6, 3000),
            (4, 130.0, 50.0, 400.0, 0.4, 0.8, 4000),
            (5, 140.0, 60.0, 500.0, 0.5, 1.0, 5000),
            # Must not influence thresholds because it belongs to the moved sequence.
            (289, 255.0, 1.0, 1.0, 99.0, 99.0, 1),
        ]
    ]

    thresholds = calibrate_thresholds(records)

    assert thresholds.blur_warn == pytest.approx(120.0)
    assert thresholds.blur_reject == pytest.approx(75.0)
    assert thresholds.brightness_low_warn == pytest.approx(102.0)
    assert thresholds.brightness_high_warn == pytest.approx(138.0)
    assert thresholds.contrast_warn == pytest.approx(22.0)
    assert thresholds.dark_percent_warn == pytest.approx(0.48)
    assert thresholds.bright_percent_warn == pytest.approx(0.96)
    assert thresholds.feature_count_warn == 1200
    assert thresholds.feature_count_reject == 750
