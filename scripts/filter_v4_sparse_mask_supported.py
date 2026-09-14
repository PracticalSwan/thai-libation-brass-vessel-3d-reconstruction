"""Filter an image-derived sparse model by supporting-view mask evidence.

This is a bounded sparse-cleanup diagnostic.  It preserves cameras, image
keypoints, coordinates, colors, and track memberships for retained points;
it only removes a point when every one of its supporting observations is not
inside the immutable vessel mask (optionally dilated by the documented mask
boundary allowance).  It never opens a SQLite database.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import pycolmap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v4_repair import stable_directory_sha256  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mask_path(root: Path, image_name: str) -> Path:
    stem = Path(image_name).stem
    candidates = [root / f"{image_name}.png", root / f"{stem}.png", root / image_name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no mask for registered image {image_name!r} under {root}")


def filter_sparse_model(
    source_model: Path,
    mask_root: Path,
    output_model: Path,
    output_report: Path,
    *,
    minimum_support_fraction: float = 1.0,
    minimum_global_support_fraction: float | None = None,
    minimum_track_length: int = 3,
    dilation_px: int = 5,
) -> dict[str, Any]:
    if output_model.exists():
        raise FileExistsError(output_model)
    if not 0.0 < minimum_support_fraction <= 1.0:
        raise ValueError("minimum_support_fraction must be in (0, 1]")
    if minimum_global_support_fraction is not None and not 0.0 < minimum_global_support_fraction <= 1.0:
        raise ValueError("minimum_global_support_fraction must be in (0, 1]")
    if minimum_track_length < 2:
        raise ValueError("minimum_track_length must be at least 2")
    if dilation_px < 0:
        raise ValueError("dilation_px must be non-negative")

    source = pycolmap.Reconstruction(str(source_model))
    masks: dict[str, np.ndarray] = {}
    global_views: list[tuple[Any, Any, np.ndarray]] = []
    for image_id in source.reg_image_ids():
        image = source.image(int(image_id))
        camera = source.camera(int(image.camera_id))
        mask = cv2.imread(str(_mask_path(mask_root, str(image.name))), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (int(camera.height), int(camera.width)):
            raise ValueError(f"mask shape mismatch for {image.name}")
        foreground = mask > 0
        if dilation_px:
            kernel_size = int(dilation_px) * 2 + 1
            foreground = cv2.dilate(
                foreground.astype(np.uint8),
                np.ones((kernel_size, kernel_size), dtype=np.uint8),
                iterations=1,
            ) > 0
        masks[str(image.name)] = foreground
        global_views.append((image, camera, foreground))

    new = pycolmap.Reconstruction()
    for camera in source.cameras.values():
        new.add_camera_with_trivial_rig(pycolmap.Camera(camera.todict()))
    for image_id in sorted(source.reg_image_ids()):
        image = source.image(int(image_id))
        keypoints = np.asarray([point.xy for point in image.points2D], dtype=np.float64).reshape(-1, 2)
        new.add_image_with_trivial_frame(
            pycolmap.Image(
                name=image.name,
                keypoints=keypoints,
                camera_id=image.camera_id,
                image_id=image.image_id,
            ),
            image.cam_from_world(),
        )

    support_histogram: Counter[str] = Counter()
    global_support_histogram: Counter[str] = Counter()
    rejected = Counter()
    kept_support: list[float] = []
    kept_lengths: list[int] = []
    kept = 0
    for point in source.points3D.values():
        xyz = np.asarray(point.xyz, dtype=np.float64).reshape(3)
        if not np.isfinite(xyz).all():
            rejected["non_finite_point"] += 1
            continue
        elements = list(point.track.elements)
        if len(elements) < minimum_track_length:
            rejected["short_track"] += 1
            continue
        seen_images: set[int] = set()
        supported = 0
        valid = True
        for element in elements:
            image_id = int(element.image_id)
            if image_id in seen_images:
                rejected["duplicate_image_track"] += 1
                valid = False
                break
            seen_images.add(image_id)
            image = source.image(image_id)
            camera = source.camera(int(image.camera_id))
            projected = image.project_point(xyz)
            if projected is None or not np.isfinite(projected).all():
                valid = False
                break
            x, y = np.rint(np.asarray(projected, dtype=np.float64)).astype(np.int64)
            if 0 <= x < int(camera.width) and 0 <= y < int(camera.height):
                supported += int(bool(masks[str(image.name)][y, x]))
        support_fraction = supported / max(len(elements), 1)
        support_histogram[f"{support_fraction:.3f}"] += 1
        if not valid or support_fraction < minimum_support_fraction:
            rejected["support_fraction"] += 1
            continue
        global_support_fraction: float | None = None
        if minimum_global_support_fraction is not None:
            visible = 0
            globally_supported = 0
            for image, camera, foreground in global_views:
                pose = image.cam_from_world()
                camera_point = np.asarray(pose.rotation.matrix(), dtype=np.float64) @ xyz + np.asarray(
                    pose.translation, dtype=np.float64
                ).reshape(3)
                if not np.isfinite(camera_point).all() or camera_point[2] <= 0.0:
                    continue
                projected = camera.img_from_cam(np.asarray([camera_point], dtype=np.float64))
                if projected is None:
                    continue
                projected = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
                if len(projected) != 1 or not np.isfinite(projected[0]).all():
                    continue
                x, y = np.rint(projected[0]).astype(np.int64)
                if not (0 <= x < int(camera.width) and 0 <= y < int(camera.height)):
                    continue
                visible += 1
                globally_supported += int(bool(foreground[y, x]))
            global_support_fraction = globally_supported / max(visible, 1)
            global_support_histogram[f"{global_support_fraction:.3f}"] += 1
            if visible == 0 or global_support_fraction < minimum_global_support_fraction:
                rejected["global_support_fraction"] += 1
                continue
        try:
            new.add_point3D(xyz, pycolmap.Track(elements), np.asarray(point.color, dtype=np.uint8))
        except Exception:
            rejected["point_observation_conflict"] += 1
            continue
        kept += 1
        kept_support.append(float(support_fraction))
        kept_lengths.append(len(elements))

    new.update_point_3d_errors()
    output_model.mkdir(parents=True, exist_ok=False)
    new.write(str(output_model))
    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "supporting-view immutable-mask sparse cleanup",
        "source_model": str(source_model.resolve()),
        "source_model_sha256": stable_directory_sha256(source_model),
        "mask_root": str(mask_root.resolve()),
        "minimum_support_fraction": float(minimum_support_fraction),
        "minimum_global_support_fraction": (
            float(minimum_global_support_fraction) if minimum_global_support_fraction is not None else None
        ),
        "minimum_track_length": int(minimum_track_length),
        "dilation_px": int(dilation_px),
        "model_write": {
            "registered_images": int(new.num_reg_images()),
            "points3D": int(new.num_points3D()),
            "observations": int(new.compute_num_observations()),
            "mean_reprojection_error": float(new.compute_mean_reprojection_error()),
            "mean_track_length": float(new.compute_mean_track_length()),
            "kept_points": int(kept),
            "kept_track_length_ge3_fraction": float(np.mean(np.asarray(kept_lengths) >= 3)) if kept_lengths else 0.0,
            "kept_support_fraction_mean": float(np.mean(kept_support)) if kept_support else 0.0,
            "rejected_points": dict(rejected),
            "support_fraction_histogram": dict(sorted(support_histogram.items())),
            "global_support_fraction_histogram": dict(sorted(global_support_histogram.items())),
        },
        "output_model_sha256": stable_directory_sha256(output_model),
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--mask-root", required=True, type=Path)
    parser.add_argument("--output-model", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--minimum-support-fraction", type=float, default=1.0)
    parser.add_argument("--minimum-global-support-fraction", type=float)
    parser.add_argument("--minimum-track-length", type=int, default=3)
    parser.add_argument("--dilation-px", type=int, default=5)
    args = parser.parse_args()
    report = filter_sparse_model(
        args.source_model,
        args.mask_root,
        args.output_model,
        args.output_report,
        minimum_support_fraction=args.minimum_support_fraction,
        minimum_global_support_fraction=args.minimum_global_support_fraction,
        minimum_track_length=args.minimum_track_length,
        dilation_px=args.dilation_px,
    )
    print(json.dumps(report["model_write"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
