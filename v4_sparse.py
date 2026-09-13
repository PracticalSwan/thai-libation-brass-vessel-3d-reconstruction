"""Narrow V4 sparse-SfM contracts around the approved COLMAP/pyCOLMAP path."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from v4_config import RECONSTRUCTION_V4_ROOT, assert_output_path, fingerprint, write_json


@dataclass(frozen=True)
class V4SparseConfig:
    image_width: int = 3072
    image_height: int = 4080
    focal_35mm: float = 26.0
    camera_model: str = "SIMPLE_RADIAL"
    shared_intrinsic_group: str = "v4_phone_rear_26mm"
    min_num_matches: int = 15
    min_model_size: int = 10
    random_seed: int = 4213

    def validate(self) -> "V4SparseConfig":
        if self.image_width < 1 or self.image_height < 1:
            raise ValueError("sparse image geometry must be positive")
        if not math.isfinite(self.focal_35mm) or self.focal_35mm <= 0:
            raise ValueError("focal_35mm must be finite and positive")
        if self.camera_model != "SIMPLE_RADIAL":
            raise ValueError("V4 starts with SIMPLE_RADIAL; change only from calibration evidence")
        if self.min_num_matches < 1 or self.min_model_size < 2:
            raise ValueError("sparse thresholds must be positive")
        return self


REQUIRED_SPARSE_COMMANDS = (
    "geometric_verifier",
    "mapper",
    "bundle_adjuster",
)


def sparse_camera_contract(config: V4SparseConfig = V4SparseConfig()) -> dict[str, Any]:
    config.validate()
    return {
        "camera_model_initial": config.camera_model,
        "shared_intrinsic_group": config.shared_intrinsic_group,
        "image_width": config.image_width,
        "image_height": config.image_height,
        "focal_35mm_equivalent": config.focal_35mm,
        "split_only_on_evidence": True,
        "camera_count_is_not_a_v4_acceptance_threshold": True,
    }


def sparse_stage_fingerprint(
    *,
    database_sha256: str,
    image_manifest_sha256: str,
    mask_manifest_sha256: str,
    config: V4SparseConfig = V4SparseConfig(),
) -> str:
    config.validate()
    return fingerprint(
        {
            "database_sha256": database_sha256,
            "image_manifest_sha256": image_manifest_sha256,
            "mask_manifest_sha256": mask_manifest_sha256,
            "config": asdict(config),
        }
    )


def ring_coverage_summary(
    records: Sequence[Mapping[str, Any]], registered_names: Sequence[str]
) -> dict[str, dict[str, int]]:
    registered = set(str(name) for name in registered_names)
    result: dict[str, dict[str, int]] = {}
    for record in records:
        if str(record.get("source_role", "geometry")) != "geometry":
            continue
        ring = str(record.get("logical_ring_id") or "unknown")
        item = result.setdefault(ring, {"selected": 0, "registered": 0})
        if bool(record.get("selected_for_geometry", True)):
            item["selected"] += 1
            if str(record.get("relative_path") or record.get("filename")) in registered:
                item["registered"] += 1
    for item in result.values():
        item["registration_fraction"] = (
            item["registered"] / item["selected"] if item["selected"] else 0.0
        )
    return result


def sparse_gate(
    *,
    ring_coverage: Mapping[str, Mapping[str, Any]],
    cross_ring_connections: int,
    board_point_fraction: float | None,
    cloth_point_fraction: float | None,
    trajectory_status: str,
    visual_status: str,
) -> dict[str, Any]:
    """Evaluate V4 sparse evidence without fabricating post-fusion metrics.

    Board/cloth fractions are not measurable from a sparse camera graph.  A
    caller may therefore pass ``None``; the corresponding checks remain false
    and the sparse gate fails closed until real fused-cloud evidence exists.
    """

    coverage_checks = {
        str(ring): int(values.get("registered", 0)) > 0
        for ring, values in ring_coverage.items()
    }
    board_measured = (
        board_point_fraction is not None
        and not isinstance(board_point_fraction, bool)
        and isinstance(board_point_fraction, (int, float))
        and math.isfinite(float(board_point_fraction))
        and 0.0 <= float(board_point_fraction) <= 1.0
    )
    cloth_measured = (
        cloth_point_fraction is not None
        and not isinstance(cloth_point_fraction, bool)
        and isinstance(cloth_point_fraction, (int, float))
        and math.isfinite(float(cloth_point_fraction))
        and 0.0 <= float(cloth_point_fraction) <= 1.0
    )
    checks = {
        "useful_ring_coverage": bool(coverage_checks) and all(coverage_checks.values()),
        "cross_ring_connectivity": int(cross_ring_connections) > 0,
        "coherent_trajectories": trajectory_status == "passed",
        "board_measurement_available": board_measured,
        "board_not_dominant": board_measured and float(board_point_fraction) < 0.20,
        "cloth_measurement_available": cloth_measured,
        "cloth_not_dominant": cloth_measured and float(cloth_point_fraction) < 0.10,
        "visual_review": visual_status == "passed",
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "ring_coverage": {str(key): dict(value) for key, value in ring_coverage.items()},
        "board_point_fraction": float(board_point_fraction) if board_measured else None,
        "cloth_point_fraction": float(cloth_point_fraction) if cloth_measured else None,
        "contamination_measurement_status": "measured_at_sparse_stage"
        if board_measured and cloth_measured
        else "not_measurable_before_fusion",
        "explicitly_no_fixed_registration_count": True,
    }


def write_sparse_gate_report(payload: Mapping[str, Any], path: Path | None = None) -> Path:
    target = path or RECONSTRUCTION_V4_ROOT / "reports" / "sparse_gate.json"
    return write_json(target, dict(payload))


def run_pycolmap_mapping(
    database_path: Path,
    image_dir: Path,
    output_dir: Path,
    *,
    config: V4SparseConfig = V4SparseConfig(),
) -> tuple[Any, ...]:
    """Run the approved pyCOLMAP mapper when a verified V4 database exists.

    This function intentionally performs no fallback matching or alternate
    reconstruction.  Callers must have already passed the vessel-mask and
    geometric-verification gates.
    """

    config.validate()
    if not database_path.is_file() or not image_dir.is_dir():
        raise ValueError("V4 sparse mapping requires an existing database and image directory")
    output_dir = assert_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    import pycolmap

    options = pycolmap.IncrementalPipelineOptions()
    options.min_num_matches = config.min_num_matches
    options.multiple_models = True
    options.min_model_size = config.min_model_size
    options.random_seed = config.random_seed
    options.ba_refine_focal_length = True
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = True
    models = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=output_dir,
        options=options,
    )
    if not models:
        raise RuntimeError("pyCOLMAP returned no V4 sparse model")
    return tuple(models[index] for index in sorted(models))


def summarize_sparse_reconstruction(
    reconstruction: Any,
    *,
    model_path: Path,
    total_images: int,
    ring_by_name: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return measured V4 sparse metrics without imposing old fixed thresholds."""

    if total_images < 1:
        raise ValueError("total_images must be positive")
    registered_names = tuple(
        sorted(str(reconstruction.image(int(image_id)).name) for image_id in reconstruction.reg_image_ids())
    )
    points = np.asarray(
        [np.asarray(point.xyz, dtype=np.float64) for point in reconstruction.points3D.values()],
        dtype=np.float64,
    )
    if points.size:
        points = points.reshape((-1, 3))
        finite_points = points[np.isfinite(points).all(axis=1)]
    else:
        finite_points = np.empty((0, 3), dtype=np.float64)
    centers: dict[str, list[np.ndarray]] = {}
    for name in registered_names:
        image = reconstruction.find_image_with_name(name)
        center = np.asarray(image.projection_center(), dtype=np.float64)
        if not np.isfinite(center).all():
            continue
        ring = str((ring_by_name or {}).get(name, "unknown"))
        centers.setdefault(ring, []).append(center)
    trajectory: dict[str, Any] = {}
    for ring, values in sorted(centers.items()):
        array = np.asarray(values, dtype=np.float64).reshape((-1, 3))
        centroid = np.mean(array, axis=0)
        radii = np.linalg.norm(array - centroid, axis=1)
        trajectory[ring] = {
            "registered_count": int(len(array)),
            "centroid": [float(value) for value in centroid],
            "radius_median": float(np.median(radii)) if len(radii) else 0.0,
            "radius_std": float(np.std(radii)) if len(radii) else 0.0,
            "radius_cv": float(np.std(radii) / max(np.mean(radii), 1e-9)) if len(radii) else 0.0,
        }
    camera_count = int(reconstruction.num_cameras())
    camera_model = ""
    camera_params: list[float] = []
    if camera_count:
        camera_ids = sorted(int(value) for value in reconstruction.cameras.keys())
        camera = reconstruction.camera(camera_ids[0])
        camera_model = str(camera.model_name)
        camera_params = [float(value) for value in np.asarray(camera.params).reshape(-1)]
    reprojection = float(reconstruction.compute_mean_reprojection_error())
    observations = int(reconstruction.compute_num_observations())
    mean_track = float(reconstruction.compute_mean_track_length())
    return {
        "model_path": str(Path(model_path).resolve()),
        "registered_images": len(registered_names),
        "total_images": int(total_images),
        "registration_fraction": len(registered_names) / total_images,
        "registered_image_names": list(registered_names),
        "sparse_points": int(reconstruction.num_points3D()),
        "finite_sparse_points": int(len(finite_points)),
        "observations": observations,
        "mean_track_length": mean_track,
        "mean_reprojection_error": reprojection,
        "mean_observations_per_registered_image": float(reconstruction.compute_mean_observations_per_reg_image()),
        "camera_count": camera_count,
        "camera_model": camera_model,
        "camera_params": camera_params,
        "point_bounds": {
            "min": [float(value) for value in np.min(finite_points, axis=0)] if len(finite_points) else [],
            "max": [float(value) for value in np.max(finite_points, axis=0)] if len(finite_points) else [],
        },
        "trajectory": trajectory,
        "trajectory_ring_count": len(trajectory),
        "trajectory_coherent_candidate": bool(
            len(trajectory) > 0
            and all(value["registered_count"] >= 3 for value in trajectory.values())
        ),
    }


def export_sparse_ply(reconstruction: Any, output_path: Path) -> Path:
    """Export the actual accepted sparse cloud for visual inspection."""

    target = assert_output_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    reconstruction.export_PLY(target)
    if not target.is_file() or target.stat().st_size <= 0:
        raise RuntimeError(f"pyCOLMAP did not produce sparse PLY: {target}")
    return target


def render_sparse_contact_sheet(
    reconstruction: Any,
    output_path: Path,
    *,
    max_points: int = 120_000,
) -> Path:
    """Render PCA-aligned orthographic views of the real sparse cloud."""

    if max_points < 100:
        raise ValueError("max_points is too small for a useful sparse preview")
    points = list(reconstruction.points3D.values())
    if not points:
        raise ValueError("cannot render an empty sparse reconstruction")
    xyz = np.asarray([np.asarray(point.xyz, dtype=np.float64) for point in points], dtype=np.float64).reshape((-1, 3))
    colors = np.asarray([np.asarray(point.color, dtype=np.uint8) for point in points], dtype=np.uint8).reshape((-1, 3))
    finite = np.isfinite(xyz).all(axis=1)
    xyz, colors = xyz[finite], colors[finite]
    if len(xyz) > max_points:
        indices = np.linspace(0, len(xyz) - 1, num=max_points, dtype=np.int64)
        xyz, colors = xyz[indices], colors[indices]
    center = np.median(xyz, axis=0)
    _, _, vt = np.linalg.svd(xyz - center, full_matrices=False)
    axes = vt[:3]
    projected = (xyz - center) @ axes.T
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (1200, 900), (245, 245, 242))
    draw = ImageDraw.Draw(canvas)
    views = ((0, 1, "PCA-XY"), (0, 2, "PCA-XZ"), (1, 2, "PCA-YZ"), (0, 1, "PCA-XY depth"))
    for view_index, (x_axis, y_axis, label) in enumerate(views):
        column, row = view_index % 2, view_index // 2
        left, top = column * 600 + 25, row * 450 + 25
        right, bottom = left + 550, top + 400
        draw.rectangle((left, top, right, bottom), outline=(150, 150, 145), width=1)
        x_values, y_values = projected[:, x_axis], projected[:, y_axis]
        x_low, x_high = np.quantile(x_values, [0.005, 0.995])
        y_low, y_high = np.quantile(y_values, [0.005, 0.995])
        x_span, y_span = max(float(x_high - x_low), 1e-9), max(float(y_high - y_low), 1e-9)
        depth = projected[:, 2] if view_index == 3 else projected[:, (3 - x_axis - y_axis)]
        order = np.argsort(depth)
        for index in order:
            px = int(left + 8 + (float(x_values[index]) - x_low) / x_span * (right - left - 16))
            py = int(bottom - 8 - (float(y_values[index]) - y_low) / y_span * (bottom - top - 28))
            if left + 2 <= px <= right - 2 and top + 22 <= py <= bottom - 2:
                color = tuple(int(value) for value in colors[index])
                draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=color)
        draw.text((left + 8, top + 6), label, fill=(25, 25, 25))
    target = assert_output_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, format="PNG", optimize=True)
    return target


def render_camera_trajectory_contact_sheet(
    reconstruction: Any,
    ring_by_name: Mapping[str, str],
    output_path: Path,
) -> Path:
    """Render registered virtual-camera centers by logical capture ring."""

    registered = []
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(int(image_id))
        center = np.asarray(image.projection_center(), dtype=np.float64)
        if np.isfinite(center).all():
            registered.append((str(ring_by_name.get(str(image.name), "unknown")), center))
    if len(registered) < 3:
        raise ValueError("at least three registered camera centers are required")
    centers = np.asarray([item[1] for item in registered], dtype=np.float64)
    origin = np.mean(centers, axis=0)
    _, _, vt = np.linalg.svd(centers - origin, full_matrices=False)
    projected = (centers - origin) @ vt[:3].T
    labels = sorted({item[0] for item in registered})
    palette = (
        (28, 91, 143),
        (190, 79, 45),
        (42, 132, 75),
        (154, 95, 174),
        (204, 137, 34),
        (79, 79, 79),
    )
    colors = {label: palette[index % len(palette)] for index, label in enumerate(labels)}
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (1200, 560), (245, 245, 242))
    draw = ImageDraw.Draw(canvas)
    for view_index, (x_axis, y_axis, title) in enumerate(((0, 1, "trajectory PCA-XY"), (0, 2, "trajectory PCA-XZ"))):
        left, top = 25 + view_index * 600, 25
        right, bottom = left + 550, top + 480
        draw.rectangle((left, top, right, bottom), outline=(150, 150, 145), width=1)
        x_values, y_values = projected[:, x_axis], projected[:, y_axis]
        x_low, x_high = np.quantile(x_values, [0.01, 0.99])
        y_low, y_high = np.quantile(y_values, [0.01, 0.99])
        x_span, y_span = max(float(x_high - x_low), 1e-9), max(float(y_high - y_low), 1e-9)
        for index, (ring, _) in enumerate(registered):
            px = int(left + 12 + (float(x_values[index]) - x_low) / x_span * (right - left - 24))
            py = int(bottom - 12 - (float(y_values[index]) - y_low) / y_span * (bottom - top - 42))
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=colors[ring])
        draw.text((left + 10, top + 8), title, fill=(25, 25, 25))
        for legend_index, ring in enumerate(labels):
            y = top + 30 + legend_index * 18
            draw.rectangle((left + 10, y, left + 20, y + 10), fill=colors[ring])
            draw.text((left + 25, y - 3), ring, fill=(25, 25, 25))
    target = assert_output_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, format="PNG", optimize=True)
    return target


__all__ = [
    "REQUIRED_SPARSE_COMMANDS",
    "V4SparseConfig",
    "ring_coverage_summary",
    "export_sparse_ply",
    "render_sparse_contact_sheet",
    "render_camera_trajectory_contact_sheet",
    "run_pycolmap_mapping",
    "summarize_sparse_reconstruction",
    "sparse_camera_contract",
    "sparse_gate",
    "sparse_stage_fingerprint",
    "write_sparse_gate_report",
]
