from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from run_geometry_analysis import GeometryRunConfig, run_geometry_analysis
from show_geometry_visuals import build_visuals


MANIFEST_FIELDS = (
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


def _build_project_fixture(tmp_path: Path) -> tuple[Path, Path]:
    images_dir = tmp_path / "selected" / "images"
    images_dir.mkdir(parents=True)
    rng = np.random.default_rng(4213)
    gray = rng.integers(75, 185, size=(360, 480), dtype=np.uint8)
    first = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x in range(25, 456, 45):
        cv2.circle(first, (x, 55 + (x % 170)), 8, (30, 30, 30), -1)
    cv2.ellipse(first, (240, 220), (92, 125), 12, 0, 360, (0, 180, 255), -1)
    cv2.putText(
        first,
        "CSX4213",
        (135, 225),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (15, 15, 15),
        2,
        cv2.LINE_AA,
    )
    transform = np.array([[1.0, 0.0, 6.0], [0.0, 1.0, 3.0]], dtype=np.float32)
    second = cv2.warpAffine(
        first,
        transform,
        (first.shape[1], first.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )

    rows: list[dict[str, object]] = []
    for name, image in (("capture_b.jpg", first), ("capture_a.jpg", second)):
        path = images_dir / name
        assert cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append(
            {
                "filename": name,
                "variant": "PREPROCESSED",
                "width": 480,
                "height": 360,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "decision": "ACCEPT",
                "reasons": "",
            }
        )
    manifest = tmp_path / "selection_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return images_dir, manifest


def _config(images_dir: Path, manifest: Path, output_root: Path) -> GeometryRunConfig:
    return GeometryRunConfig(
        images_dir=images_dir,
        selection_manifest=manifest,
        output_root=output_root,
        expected_selected_count=2,
        pairs=((1, 2),),
        epipolar_pair=(1, 2),
        shape_indices=(1, 2),
    )


def test_orchestrator_verifies_inputs_before_creating_outputs(tmp_path: Path) -> None:
    """Catches final-looking reports being created from an unverified selected set."""
    images_dir, manifest = _build_project_fixture(tmp_path)
    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8", newline="")))
    rows[0]["sha256"] = "0" * 64
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    output_root = tmp_path / "analysis"

    with pytest.raises(ValueError, match="hash mismatch"):
        run_geometry_analysis(_config(images_dir, manifest, output_root))

    assert not output_root.exists()


def test_orchestrator_rejects_output_inside_selected_images(tmp_path: Path) -> None:
    """Catches reports or figures contaminating the verified selected directory."""
    images_dir, manifest = _build_project_fixture(tmp_path)
    unsafe_output = images_dir / "analysis"

    with pytest.raises(ValueError, match="must not overlap"):
        run_geometry_analysis(_config(images_dir, manifest, unsafe_output))

    assert not unsafe_output.exists()


def test_real_miniature_run_writes_reports_figures_and_provenance(
    tmp_path: Path,
) -> None:
    """Catches units that pass separately but fail as one measured Step 6 flow."""
    images_dir, manifest = _build_project_fixture(tmp_path)
    output_root = tmp_path / "analysis"
    source_hashes = {
        path.name: _sha256(path) for path in sorted(images_dir.glob("*.jpg"))
    }

    summary = run_geometry_analysis(_config(images_dir, manifest, output_root))

    required = {
        "geometry/pair_metrics.csv",
        "geometry/epipolar_metrics.json",
        "geometry/shape_metrics.csv",
        "reports/input_verification.json",
        "reports/geometry_summary.json",
        "previews/presentation/geometry_01_matches_1_2.png",
        "previews/presentation/geometry_02_epipolar_1_2.png",
        "previews/presentation/geometry_03_shape_1.png",
        "previews/presentation/geometry_03_shape_2.png",
        "previews/presentation/geometry_04_summary.png",
    }
    assert summary.complete is True
    assert set(summary.artifacts) == required
    for relative_path in required:
        path = output_root / relative_path
        assert path.is_file()
        if path.suffix == ".png":
            assert cv2.imread(str(path), cv2.IMREAD_COLOR) is not None

    summary_payload = json.loads(
        (output_root / "reports" / "geometry_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary_payload["complete"] is True
    assert summary_payload["source"]["verified_selected_count"] == 2
    assert summary_payload["source"]["selection_manifest_sha256"] == _sha256(
        manifest
    )
    assert summary_payload["configuration"]["sift"]["maximum_width"] == 1200
    assert summary_payload["configuration"]["sift"]["rng_seed"] == 4213
    assert summary_payload["scope_exclusions"] == [
        "Step 7/8 ML and SAM",
        "pyCOLMAP and reconstruction",
        "meshing, texturing, and Blender",
    ]
    assert {
        path.name: _sha256(path) for path in sorted(images_dir.glob("*.jpg"))
    } == source_hashes


def test_visualizer_builds_real_renderings_without_gui_or_source_writes(
    tmp_path: Path,
) -> None:
    """Catches a demo that duplicates fake logic or needs a blocking GUI to run."""
    images_dir, manifest = _build_project_fixture(tmp_path)
    before = {path.name: _sha256(path) for path in images_dir.glob("*.jpg")}

    visuals, metrics = build_visuals(
        images_dir=images_dir,
        selection_manifest=manifest,
        mode="all",
        pair=(1, 2),
        supporting_pair=(1, 2),
        shape_indices=(1, 2),
    )

    assert set(visuals) == {
        "SIFT and RANSAC 1-2",
        "Epipolar geometry 1-2",
        "Classical shape 1",
        "Classical shape 2",
    }
    assert all(image.ndim == 3 and image.size > 0 for image in visuals.values())
    assert metrics["pair"]["candidate_matches"] >= 8
    assert metrics["pair"]["ransac_inliers"] >= 8
    assert metrics["shape"]["1"]["status"] in {"ok", "weak_contour"}
    assert {path.name: _sha256(path) for path in images_dir.glob("*.jpg")} == before
