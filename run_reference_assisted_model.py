"""Orchestrate the reference-assisted pre-clean Blender model."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from reference_assisted_model import build_reference_report


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "reconstruction" / "reference_assisted"
REPORT_DIR = OUTPUT_DIR / "reports"
BLENDER_SCRIPT = PROJECT_ROOT / "build_reference_model_blender.py"
DEFAULT_BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")


def resolve_blender() -> Path:
    if DEFAULT_BLENDER.is_file():
        return DEFAULT_BLENDER
    found = shutil.which("blender")
    if found:
        return Path(found)
    raise RuntimeError("Blender executable not found")


def run_analysis() -> dict:
    return build_reference_report(PROJECT_ROOT, REPORT_DIR)


def run_blender() -> dict:
    reference_report = REPORT_DIR / "reference_report.json"
    if not reference_report.is_file():
        raise RuntimeError("reference report is missing; run analyze first")
    blender = resolve_blender()
    log_dir = OUTPUT_DIR / "work" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "build_blender.log"
    command = [
        str(blender),
        "--background",
        "--python-exit-code",
        "2",
        "--python",
        str(BLENDER_SCRIPT),
        "--",
        str(reference_report),
        str(OUTPUT_DIR),
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
        raise RuntimeError(f"Blender reference build failed with exit code {result.returncode}\n{tail}")
    build_report_path = OUTPUT_DIR / "build_report.json"
    if not build_report_path.is_file():
        raise RuntimeError("Blender exited successfully but build_report.json is missing")
    return json.loads(build_report_path.read_text(encoding="utf-8"))


def _normalized_silhouette(mask: np.ndarray, size: int = 768) -> np.ndarray:
    binary = mask.astype(bool)
    if not binary.any():
        raise ValueError("silhouette contains no foreground")
    ys, xs = np.nonzero(binary)
    crop = binary[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    target_height = int(round(size * 0.90))
    scale = target_height / crop.shape[0]
    target_width = max(1, int(round(crop.shape[1] * scale)))
    resized = cv2.resize(crop.astype(np.uint8) * 255, (target_width, target_height), interpolation=cv2.INTER_NEAREST) >= 128
    canvas = np.zeros((size, size), dtype=bool)
    left = (size - target_width) // 2
    top = (size - target_height) // 2
    if left < 0:
        resized = cv2.resize(resized.astype(np.uint8) * 255, (size, target_height), interpolation=cv2.INTER_NEAREST) >= 128
        left = 0
        target_width = size
    canvas[top : top + target_height, left : left + target_width] = resized[:, :target_width]
    return canvas


def silhouette_iou(reference_path: Path, render_path: Path) -> float:
    reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
    render = cv2.imread(str(render_path), cv2.IMREAD_UNCHANGED)
    if reference is None or render is None:
        raise ValueError("silhouette validation image is unreadable")
    if render.ndim != 3 or render.shape[2] < 4:
        raise ValueError("Blender silhouette render must contain alpha")
    ref_mask = _normalized_silhouette(reference >= 128)
    render_mask = _normalized_silhouette(render[..., 3] >= 16)
    union = np.logical_or(ref_mask, render_mask).sum()
    intersection = np.logical_and(ref_mask, render_mask).sum()
    return float(intersection / union) if union else 0.0


def validate_outputs(minimum_silhouette_iou: float = 0.70) -> dict:
    reference_report_path = REPORT_DIR / "reference_report.json"
    build_report_path = OUTPUT_DIR / "build_report.json"
    if not reference_report_path.is_file() or not build_report_path.is_file():
        raise RuntimeError("reference/build report missing")
    reference = json.loads(reference_report_path.read_text(encoding="utf-8"))
    build = json.loads(build_report_path.read_text(encoding="utf-8"))
    if reference.get("manual_cleanup_started") is not False or build.get("manual_cleanup_started") is not False:
        raise RuntimeError("pre-clean boundary was violated")

    expected = [
        OUTPUT_DIR / "Thai_Libation_Vessel_REFERENCE_ASSISTED_PRE_CLEAN.blend",
        OUTPUT_DIR / "Thai_Libation_Vessel_REFERENCE_ASSISTED_PRE_CLEAN.glb",
        OUTPUT_DIR / "previews" / "reference_front.png",
        OUTPUT_DIR / "previews" / "reference_quarter.png",
        OUTPUT_DIR / "previews" / "reference_side.png",
        OUTPUT_DIR / "previews" / "reference_top_oblique.png",
        OUTPUT_DIR / "previews" / "reference_render_silhouette.png",
    ]
    missing = [path.as_posix() for path in expected if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError("missing reference-assisted outputs: " + ", ".join(missing))

    nonmanifold = build["stats"].get("nonmanifold_edges_by_object", {})
    # The generated assembly uses intentional separate physical components, but every mesh object itself must be closed.
    if nonmanifold:
        raise RuntimeError(f"generated mesh objects contain non-manifold edges: {nonmanifold}")

    iou = silhouette_iou(
        REPORT_DIR / "reference_silhouette.png",
        OUTPUT_DIR / "previews" / "reference_render_silhouette.png",
    )
    validation = {
        "accepted": iou >= minimum_silhouette_iou,
        "manual_cleanup_started": False,
        "silhouette_iou": iou,
        "minimum_silhouette_iou": minimum_silhouette_iou,
        "mesh_object_count": build["stats"]["mesh_object_count"],
        "vertex_count": build["stats"]["vertex_count"],
        "polygon_count": build["stats"]["polygon_count"],
        "nonmanifold_edges_by_object": nonmanifold,
        "artifacts": [path.as_posix() for path in expected],
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "validation_report.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if not validation["accepted"]:
        raise RuntimeError(f"reference-assisted silhouette IoU {iou:.4f} is below {minimum_silhouette_iou:.4f}")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("analyze", "build", "validate", "all"), default="all")
    parser.add_argument("--minimum-silhouette-iou", type=float, default=0.70)
    args = parser.parse_args()

    if args.stage in {"analyze", "all"}:
        print(json.dumps(run_analysis(), indent=2))
    if args.stage in {"build", "all"}:
        print(json.dumps(run_blender(), indent=2))
    if args.stage in {"validate", "all"}:
        print(json.dumps(validate_outputs(args.minimum_silhouette_iou), indent=2))


if __name__ == "__main__":
    main()
