"""Run one pose-fixed, point-only bundle-adjustment diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pycolmap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v4_repair import stable_directory_sha256  # noqa: E402


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "name"):
        return str(value.name)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def refine_points(source_model: Path, output_model: Path, output_report: Path) -> dict:
    if output_model.exists():
        raise FileExistsError(output_model)
    reconstruction = pycolmap.Reconstruction(str(source_model))
    options = pycolmap.BundleAdjustmentOptions()
    options.refine_points3D = True
    config = pycolmap.BundleAdjustmentConfig()
    for image_id in reconstruction.reg_image_ids():
        config.set_constant_rig_from_world_pose(int(reconstruction.image(int(image_id)).frame_id))
    for camera_id in reconstruction.cameras:
        config.set_constant_cam_intrinsics(int(camera_id))
    adjuster = pycolmap.create_default_bundle_adjuster(options, config, reconstruction)
    summary = adjuster.solve()
    reconstruction.update_point_3d_errors()
    output_model.mkdir(parents=True, exist_ok=False)
    reconstruction.write(str(output_model))
    report = {
        "schema_version": 1,
        "method": "pose-fixed point-only pyCOLMAP 4.2 bundle adjustment",
        "source_model": str(source_model.resolve()),
        "source_model_sha256": stable_directory_sha256(source_model),
        "model_write": {
            "registered_images": int(reconstruction.num_reg_images()),
            "points3D": int(reconstruction.num_points3D()),
            "observations": int(reconstruction.compute_num_observations()),
            "mean_reprojection_error": float(reconstruction.compute_mean_reprojection_error()),
            "mean_track_length": float(reconstruction.compute_mean_track_length()),
        },
        "bundle_adjustment_summary": _json_safe(summary.todict()),
        "output_model_sha256": stable_directory_sha256(output_model),
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--output-model", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    args = parser.parse_args()
    report = refine_points(args.source_model, args.output_model, args.output_report)
    print(json.dumps(report["model_write"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
