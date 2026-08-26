import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from run_preprocessing import (
    PipelineConfig,
    choose_reconstruction_variant,
    compare_sift_pair,
    export_selected_images,
    run_pipeline,
    validate_run_counts,
    verify_raw_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _textured_pair() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4213)
    first = np.full((240, 320, 3), 210, dtype=np.uint8)
    for _ in range(90):
        center = (int(rng.integers(10, 310)), int(rng.integers(10, 230)))
        radius = int(rng.integers(2, 8))
        color = tuple(int(value) for value in rng.integers(20, 190, size=3))
        cv2.circle(first, center, radius, color, -1)
    cv2.putText(
        first,
        "CSX4213",
        (55, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (10, 10, 10),
        3,
        cv2.LINE_AA,
    )
    transform = np.asarray([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0]], dtype=np.float32)
    warp_affine = getattr(cv2, "warpAffine")
    second = warp_affine(
        first,
        transform,
        (first.shape[1], first.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )
    return first, second


def test_raw_manifest_verification_detects_content_change(tmp_path: Path) -> None:
    """Catches raw-data mutation even when the filename and byte count survive."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    first = raw_dir / "a.jpg"
    second = raw_dir / "b.jpg"
    first.write_bytes(b"abcd")
    second.write_bytes(b"1234")
    baseline_path = tmp_path / "manifest.json"
    baseline_path.write_text(
        json.dumps(
            {
                "image_count": 2,
                "files": [
                    {
                        "filename": first.name,
                        "sha256": _sha256(first),
                        "size_bytes": first.stat().st_size,
                    },
                    {
                        "filename": second.name,
                        "sha256": _sha256(second),
                        "size_bytes": second.stat().st_size,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    unchanged = verify_raw_manifest(raw_dir, baseline_path)
    second.write_bytes(b"5678")
    changed = verify_raw_manifest(raw_dir, baseline_path)

    assert unchanged["unchanged"] is True
    assert unchanged["mismatch_count"] == 0
    assert changed["unchanged"] is False
    assert changed["hash_mismatches"] == ["b.jpg"]
    assert changed["size_mismatches"] == []
    assert changed["mismatch_count"] == 1


def test_raw_export_is_sorted_exact_readable_and_removes_stale_file(
    tmp_path: Path,
) -> None:
    """Catches nondeterministic naming, lossy RAW export, or stale output leakage."""
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    output_dir.mkdir()
    image = np.full((32, 48, 3), 127, dtype=np.uint8)
    for name in ("b.png", "a.png", "c.png"):
        assert cv2.imwrite(str(source_dir / name), image)
    (output_dir / "c.png").write_bytes((source_dir / "c.png").read_bytes())

    manifest = export_selected_images(
        source_dir,
        output_dir,
        ["b.png", "a.png"],
        variant="RAW",
    )

    assert [row["filename"] for row in manifest] == ["a.png", "b.png"]
    assert [path.name for path in sorted(output_dir.iterdir())] == ["a.png", "b.png"]
    for row in manifest:
        source = source_dir / row["filename"]
        output = output_dir / row["filename"]
        assert output.read_bytes() == source.read_bytes()
        assert cv2.imread(str(output), cv2.IMREAD_COLOR) is not None
        assert row["sha256"] == _sha256(output)


def test_export_rejects_ancestor_destination_and_unknown_files_before_cleanup(
    tmp_path: Path,
) -> None:
    """Catches broad cleanup deleting source ancestors or unrelated user files."""
    workspace = tmp_path / "workspace"
    source_dir = workspace / "raw"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "a.jpg"
    source_path.write_bytes(b"raw evidence")

    with pytest.raises(ValueError, match="outside the immutable source tree"):
        export_selected_images(source_dir, workspace, ["a.jpg"], variant="RAW")
    assert source_path.read_bytes() == b"raw evidence"

    safe_output = tmp_path / "selected"
    safe_output.mkdir()
    note = safe_output / "notes.txt"
    note.write_text("user-owned", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected file in output set"):
        export_selected_images(source_dir, safe_output, ["a.jpg"], variant="RAW")
    assert note.read_text(encoding="utf-8") == "user-owned"
    assert source_path.read_bytes() == b"raw evidence"


def test_pipeline_rejects_raw_preview_overlap_before_any_mutation(
    tmp_path: Path,
) -> None:
    """Catches CLI destination overrides deleting files before raw verification."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source_path = raw_dir / "capture.jpg"
    source_path.write_bytes(b"immutable raw")
    baseline_path = tmp_path / "raw_manifest_before.json"
    baseline_path.write_text(
        json.dumps(
            {
                "image_count": 1,
                "files": [
                    {
                        "filename": source_path.name,
                        "sha256": _sha256(source_path),
                        "size_bytes": source_path.stat().st_size,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = PipelineConfig(
        raw_dir=raw_dir,
        baseline_manifest=baseline_path,
        reports_dir=tmp_path / "reports",
        previews_dir=raw_dir,
        output_dir=tmp_path / "selected" / "images",
        expected_raw_count=1,
        representative_pairs=((1, 1),),
    )

    with pytest.raises(ValueError, match="must not overlap the immutable raw directory"):
        run_pipeline(config)

    assert source_path.read_bytes() == b"immutable raw"


def test_variant_choice_requires_better_inlier_evidence_and_defaults_raw_on_tie() -> None:
    """Catches automatically preferring prettier preprocessing without match evidence."""
    tie = [
        {"raw_inliers": 100, "preprocessed_inliers": 100},
        {"raw_inliers": 120, "preprocessed_inliers": 120},
    ]
    improved = [
        {"raw_inliers": 100, "preprocessed_inliers": 112},
        {"raw_inliers": 120, "preprocessed_inliers": 121},
        {"raw_inliers": 90, "preprocessed_inliers": 88},
    ]

    assert choose_reconstruction_variant(tie) == "RAW"
    assert choose_reconstruction_variant(improved) == "PREPROCESSED"


def test_run_counts_must_agree_across_reports_manifest_and_outputs() -> None:
    """Catches a completed-looking run with missing report rows or images."""
    validate_run_counts(
        expected_raw_count=297,
        actual_raw_count=297,
        quality_row_count=297,
        selected_count=288,
        selection_manifest_count=288,
        output_count=288,
    )

    with pytest.raises(ValueError, match="output_count=287, selected_count=288"):
        validate_run_counts(
            expected_raw_count=297,
            actual_raw_count=297,
            quality_row_count=297,
            selected_count=288,
            selection_manifest_count=288,
            output_count=287,
        )


def test_sift_comparison_produces_geometrically_verified_matches(
    tmp_path: Path,
) -> None:
    """Catches a matching report that lacks usable correspondences or RANSAC evidence."""
    first, second = _textured_pair()
    first_path = tmp_path / "first.jpg"
    second_path = tmp_path / "second.jpg"
    assert cv2.imwrite(str(first_path), first)
    assert cv2.imwrite(str(second_path), second)

    result = compare_sift_pair(first_path, second_path)

    assert result["raw_good_matches"] >= 20
    assert result["raw_inliers"] >= 15
    assert result["preprocessed_good_matches"] >= 20
    assert result["preprocessed_inliers"] >= 15
    assert 0.0 <= result["raw_inlier_ratio"] <= 1.0
    assert 0.0 <= result["preprocessed_inlier_ratio"] <= 1.0


def test_small_pipeline_run_writes_consistent_reports_and_selected_set(
    tmp_path: Path,
) -> None:
    """Catches orchestration paths that pass units but fail as one real workflow."""
    raw_dir = tmp_path / "raw"
    reports_dir = tmp_path / "reports"
    previews_dir = tmp_path / "previews"
    output_dir = tmp_path / "pycolmap_input" / "images"
    raw_dir.mkdir()
    first, second = _textured_pair()
    third = cv2.GaussianBlur(second, (3, 3), 0.5)
    for name, image in [
        ("capture_001.jpg", first),
        ("capture_002.jpg", second),
        ("capture_003.jpg", third),
    ]:
        assert cv2.imwrite(str(raw_dir / name), image)
    baseline_path = reports_dir / "raw_manifest_before.json"
    reports_dir.mkdir()
    files = [
        {
            "filename": path.name,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(raw_dir.iterdir())
    ]
    baseline_path.write_text(
        json.dumps({"image_count": 3, "files": files}), encoding="utf-8"
    )
    config = PipelineConfig(
        raw_dir=raw_dir,
        baseline_manifest=baseline_path,
        reports_dir=reports_dir,
        previews_dir=previews_dir,
        output_dir=output_dir,
        expected_raw_count=3,
        representative_pairs=((1, 2),),
    )

    summary = run_pipeline(config)

    assert summary["raw_count"] == 3
    assert summary["selected_count"] == 3
    assert summary["rejected_count"] == 0
    assert summary["raw_unchanged"] is True
    assert summary["selected_variant"] in {"RAW", "PREPROCESSED"}
    assert "pycolmap_input_directory" not in summary
    assert len(list(output_dir.glob("*.jpg"))) == 3
    for name in [
        "quality_decisions.csv",
        "quality_thresholds.json",
        "raw_verification_after.json",
        "selection_manifest.csv",
        "sift_matching.csv",
        "sift_matching.json",
        "preprocessing_summary.json",
    ]:
        assert (reports_dir / name).is_file()
    assert (output_dir.parent / "README.md").is_file()
    assert list(previews_dir.glob("before_after_*.jpg"))
    sift_report = json.loads(
        (reports_dir / "sift_matching.json").read_text(encoding="utf-8")
    )
    assert sift_report["method"]["jpeg_quality"] == 95
    assert sift_report["method"]["comparison_artifact"] == (
        "decoded quality-95 JPEG bytes identical to final export encoding"
    )
