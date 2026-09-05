"""Run Step 10 sparse Structure-from-Motion with pyCOLMAP only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Sequence

import matplotlib
import numpy as np
import pycolmap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis_common import SelectedImageRecord, verify_selected_images
from sparse_reconstruction import (
    AttemptMetrics,
    DatabaseMetrics,
    ModelMetrics,
    SparseRunConfig,
    attempt_requires_retry,
    choose_best_attempt,
    copy_sparse_model,
    run_sparse_attempt,
    simple_radial_camera_params,
    summarize_reconstruction,
)

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "IMG20260826122949"
SELECTED_DIR = ROOT / "preprocessing" / "pycolmap_input" / "images"
SELECTION_MANIFEST = ROOT / "preprocessing" / "reports" / "selection_manifest.csv"
RECON_ROOT = ROOT / "reconstruction"
SPARSE_ROOT = RECON_ROOT / "sparse"
REPORTS_DIR = RECON_ROOT / "reports"
PREVIEWS_DIR = RECON_ROOT / "previews"
BASELINE_DIR = SPARSE_ROOT / "baseline"
RETRY_DIR = SPARSE_ROOT / "retry_overlap40"
BEST_DIR = SPARSE_ROOT / "best"
BASELINE_REPORT = REPORTS_DIR / "step10_baseline.json"
RETRY_REPORT = REPORTS_DIR / "step10_retry_overlap40.json"
ATTEMPTS_CSV = REPORTS_DIR / "step10_attempts.csv"
SUMMARY_JSON = REPORTS_DIR / "step10_summary.json"
REGISTERED_CSV = REPORTS_DIR / "step10_registered_images.csv"
SPARSE_FIGURE = PREVIEWS_DIR / "step10_01_sparse_model.png"
REGISTRATION_FIGURE = PREVIEWS_DIR / "step10_02_registration.png"
PLY_PATH = BEST_DIR / "points3D.ply"
CONFIG = SparseRunConfig()
STEP9_WEAK_TRANSITIONS = (
    (73, 74),
    (145, 146),
    (203, 204),
    (246, 247),
    (258, 259),
    (266, 267),
    (267, 268),
    (271, 272),
    (272, 273),
    (273, 274),
    (276, 277),
    (279, 280),
    (283, 284),
    (285, 286),
)


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _portable_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    root = project_root.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _portable_model(model: ModelMetrics, project_root: Path) -> dict[str, object]:
    payload = model.to_dict()
    payload["model_path"] = _portable_path(model.model_path, project_root)
    return _json_safe(payload)  # type: ignore[return-value]


def _portable_attempt(attempt: AttemptMetrics, project_root: Path) -> dict[str, object]:
    union_registered = set().union(
        *(set(model.registered_image_names) for model in attempt.models)
    )
    return {
        "name": attempt.name,
        "workspace": _portable_path(attempt.workspace, project_root),
        "overlap": attempt.overlap,
        "database": asdict(attempt.database),
        "models": [_portable_model(model, project_root) for model in attempt.models],
        "best_model": _portable_model(attempt.best_model, project_root),
        "model_union_registered_images": len(union_registered),
        "runtime_seconds": attempt.runtime_seconds,
        "pycolmap_version": attempt.pycolmap_version,
    }


def _model_from_payload(payload: dict[str, object], project_root: Path) -> ModelMetrics:
    raw_path = Path(str(payload["model_path"]))
    model_path = raw_path if raw_path.is_absolute() else project_root / raw_path
    raw_error = payload.get("mean_reprojection_error")
    raw_track = payload.get("mean_track_length")
    raw_mean_obs = payload.get("mean_observations_per_registered_image")
    return ModelMetrics(
        model_path=model_path,
        registered_images=int(payload["registered_images"]),
        total_images=int(payload["total_images"]),
        sparse_points=int(payload["sparse_points"]),
        observations=int(payload["observations"]),
        mean_track_length=float(raw_track) if raw_track is not None else math.nan,
        mean_reprojection_error=float(raw_error) if raw_error is not None else math.nan,
        camera_count=int(payload["camera_count"]),
        camera_model=str(payload["camera_model"]),
        camera_params=tuple(float(value) for value in payload.get("camera_params", [])),
        mean_observations_per_registered_image=(
            float(raw_mean_obs) if raw_mean_obs is not None else math.nan
        ),
        registered_image_names=tuple(str(value) for value in payload.get("registered_image_names", [])),
    )


def _load_attempt(path: Path) -> AttemptMetrics:
    if not path.is_file():
        raise ValueError(f"Step 10 attempt report is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid Step 10 attempt report: {path}")
    workspace_value = Path(str(payload["workspace"]))
    workspace = workspace_value if workspace_value.is_absolute() else ROOT / workspace_value
    models_payload = payload.get("models")
    if not isinstance(models_payload, list) or not models_payload:
        raise ValueError(f"Step 10 attempt has no sparse models: {path}")
    models = tuple(_model_from_payload(item, ROOT) for item in models_payload)
    best_payload = payload.get("best_model")
    if not isinstance(best_payload, dict):
        raise ValueError(f"Step 10 attempt has no best model: {path}")
    best_model = _model_from_payload(best_payload, ROOT)
    database_payload = payload.get("database")
    if not isinstance(database_payload, dict):
        raise ValueError(f"Step 10 attempt database metrics are invalid: {path}")
    return AttemptMetrics(
        name=str(payload["name"]),
        workspace=workspace,
        overlap=int(payload["overlap"]),
        database=DatabaseMetrics(
            image_count=int(database_payload["image_count"]),
            feature_count=int(database_payload["feature_count"]),
            matched_pair_count=int(database_payload["matched_pair_count"]),
            verified_pair_count=int(database_payload["verified_pair_count"]),
        ),
        models=models,
        best_model=best_model,
        runtime_seconds=float(payload.get("runtime_seconds", 0.0)),
        pycolmap_version=str(payload.get("pycolmap_version", "")),
    )


def _verified_attempt(path: Path) -> AttemptMetrics:
    attempt = _load_attempt(path)
    measured = summarize_reconstruction(attempt.best_model.model_path, CONFIG.expected_images)
    if measured.registered_images != attempt.best_model.registered_images:
        raise ValueError(f"stale Step 10 attempt registration count: {path}")
    if measured.sparse_points != attempt.best_model.sparse_points:
        raise ValueError(f"stale Step 10 attempt point count: {path}")
    return attempt


def execute_attempt_sequence(
    image_dir: Path,
    reconstruction_root: Path,
    *,
    runner: Callable[..., AttemptMetrics],
    config: SparseRunConfig,
) -> tuple[AttemptMetrics, ...]:
    baseline = runner(
        image_dir,
        reconstruction_root / "sparse" / "baseline",
        overlap=config.baseline_overlap,
        config=config,
    )
    attempts = [baseline]
    if attempt_requires_retry(baseline, config.expected_images):
        retry = runner(
            image_dir,
            reconstruction_root / "sparse" / "retry_overlap40",
            overlap=config.retry_overlap,
            config=config,
        )
        attempts.append(retry)
    return tuple(attempts)


def copy_best_model(source: Path, destination: Path) -> tuple[Path, ...]:
    return copy_sparse_model(source, destination)


def build_step10_summary(
    *,
    attempts: Sequence[AttemptMetrics],
    selected: AttemptMetrics,
    pycolmap_version: str,
    selection_manifest_sha256: str,
    initial_camera_params: tuple[float, float, float, float],
    project_root: Path,
) -> dict[str, object]:
    best = selected.best_model
    return {
        "pycolmap_version": pycolmap_version,
        "input_image_count": CONFIG.expected_images,
        "selection_manifest_sha256": selection_manifest_sha256,
        "camera_mode": "SINGLE",
        "camera_model": CONFIG.camera_model,
        "initial_camera_params": list(initial_camera_params),
        "feature_extraction": {
            "type": "SIFT",
            "max_image_size": CONFIG.max_image_size,
            "max_num_features": CONFIG.max_num_features,
            "masked": False,
            "device": "cpu",
        },
        "matching": {
            "type": "sequential",
            "quadratic_overlap": True,
            "loop_detection": False,
            "baseline_overlap": CONFIG.baseline_overlap,
            "retry_overlap": CONFIG.retry_overlap,
        },
        "attempts": [_portable_attempt(attempt, project_root) for attempt in attempts],
        "selected_attempt": selected.name,
        "best_model": _portable_model(best, project_root),
        "registration_percentage": 100.0 * best.registration_fraction,
        "retry_used": len(attempts) > 1,
        "acceptance_met": not attempt_requires_retry(selected, CONFIG.expected_images),
        "dense_reconstruction_started": False,
        "next_boundary": "dense reconstruction requires separate authorization",
    }


def _clear_partial_attempt(workspace: Path, report_path: Path) -> None:
    if report_path.exists():
        return
    if workspace.exists():
        shutil.rmtree(workspace)


def _run_attempt_stage(name: str, workspace: Path, report_path: Path, overlap: int) -> AttemptMetrics:
    if report_path.is_file():
        try:
            return _verified_attempt(report_path)
        except (OSError, RuntimeError, ValueError):
            report_path.unlink(missing_ok=True)
            if workspace.exists():
                shutil.rmtree(workspace)
    _clear_partial_attempt(workspace, report_path)
    attempt = run_sparse_attempt(
        SELECTED_DIR,
        workspace,
        overlap=overlap,
        config=CONFIG,
    )
    _write_json(report_path, _portable_attempt(attempt, ROOT))
    return attempt


def _baseline_stage() -> AttemptMetrics:
    verify_selected_images(SELECTED_DIR, SELECTION_MANIFEST, expected_count=CONFIG.expected_images)
    return _run_attempt_stage("baseline", BASELINE_DIR, BASELINE_REPORT, CONFIG.baseline_overlap)


def _retry_stage() -> AttemptMetrics | None:
    baseline = _verified_attempt(BASELINE_REPORT)
    if not attempt_requires_retry(baseline, CONFIG.expected_images):
        return None
    return _run_attempt_stage(
        "retry_overlap40", RETRY_DIR, RETRY_REPORT, CONFIG.retry_overlap
    )


def _write_attempts_csv(attempts: Sequence[AttemptMetrics]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for attempt in attempts:
        model = attempt.best_model
        rows.append(
            {
                "attempt": attempt.name,
                "overlap": attempt.overlap,
                "model_count": len(attempt.models),
                "registered_images": model.registered_images,
                "registration_fraction": model.registration_fraction,
                "sparse_points": model.sparse_points,
                "observations": model.observations,
                "mean_track_length": model.mean_track_length,
                "mean_reprojection_error": model.mean_reprojection_error,
                "camera_count": model.camera_count,
                "camera_model": model.camera_model,
                "runtime_seconds": attempt.runtime_seconds,
                "feature_count": attempt.database.feature_count,
                "matched_pair_count": attempt.database.matched_pair_count,
                "verified_pair_count": attempt.database.verified_pair_count,
            }
        )
    with ATTEMPTS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_registered_csv(
    selected_records: Sequence[SelectedImageRecord], registered_names: set[str]
) -> None:
    rows = [
        {
            "selected_index": record.index,
            "filename": record.filename,
            "registered": int(record.filename in registered_names),
        }
        for record in selected_records
    ]
    with REGISTERED_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _sparse_arrays(model_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reconstruction = pycolmap.Reconstruction(model_path)
    point_ids = sorted(int(value) for value in reconstruction.points3D.keys())
    points = np.asarray(
        [reconstruction.point3D(point_id).xyz for point_id in point_ids], dtype=float
    )
    colors = np.asarray(
        [reconstruction.point3D(point_id).color for point_id in point_ids], dtype=float
    )
    if len(colors):
        colors = colors / 255.0
    camera_centers = np.asarray(
        [
            reconstruction.image(int(image_id)).projection_center()
            for image_id in reconstruction.reg_image_ids()
        ],
        dtype=float,
    )
    return points, colors, camera_centers


def _render_sparse_figure(model_path: Path, model: ModelMetrics) -> None:
    points, colors, camera_centers = _sparse_arrays(model_path)
    if not len(points) or not len(camera_centers):
        raise ValueError("selected sparse model cannot be visualized")
    max_points = 40000
    if len(points) > max_points:
        stride = math.ceil(len(points) / max_points)
        points = points[::stride]
        colors = colors[::stride]
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=1.0,
        c=colors if len(colors) == len(points) else None,
        alpha=0.65,
    )
    axis.scatter(
        camera_centers[:, 0],
        camera_centers[:, 1],
        camera_centers[:, 2],
        s=14,
        marker="^",
        label="registered camera centers",
    )
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.set_title(
        "Step 10 selected sparse component | "
        f"registered={model.registered_images}/{model.total_images} | "
        f"points={model.sparse_points:,} | reproj={model.mean_reprojection_error:.3f}px"
    )
    axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(SPARSE_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _render_registration_figure(
    selected_records: Sequence[SelectedImageRecord], registered_names: set[str]
) -> None:
    indices = np.asarray([record.index for record in selected_records], dtype=int)
    registered = np.asarray(
        [1 if record.filename in registered_names else 0 for record in selected_records],
        dtype=int,
    )
    figure, axis = plt.subplots(figsize=(14, 4.5))
    axis.step(indices, registered, where="mid", linewidth=1.2)
    unregistered_indices = indices[registered == 0]
    if len(unregistered_indices):
        axis.scatter(unregistered_indices, np.zeros_like(unregistered_indices), marker="x", s=30)
    for left, right in STEP9_WEAK_TRANSITIONS:
        axis.axvspan(left, right, alpha=0.08)
    axis.set_ylim(-0.15, 1.2)
    axis.set_yticks([0, 1], ["unregistered", "registered"])
    axis.set_xlabel("Selected sequence index")
    axis.set_title(
        "Step 10 selected-component registration | shaded bands = Step 9 weak transitions"
    )
    axis.grid(axis="x", alpha=0.15)
    figure.tight_layout()
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(REGISTRATION_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _finalize_stage() -> dict[str, object]:
    verified = verify_selected_images(
        SELECTED_DIR, SELECTION_MANIFEST, expected_count=CONFIG.expected_images
    )
    attempts = [_verified_attempt(BASELINE_REPORT)]
    if RETRY_REPORT.is_file():
        attempts.append(_verified_attempt(RETRY_REPORT))
    selected = choose_best_attempt(tuple(attempts))
    copy_best_model(selected.best_model.model_path, BEST_DIR)
    selected_model = summarize_reconstruction(BEST_DIR, CONFIG.expected_images)
    reconstruction = pycolmap.Reconstruction(BEST_DIR)
    reconstruction.export_PLY(PLY_PATH)
    registered_names = set(selected_model.registered_image_names)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    _write_attempts_csv(attempts)
    _write_registered_csv(verified.records, registered_names)
    _render_sparse_figure(BEST_DIR, selected_model)
    _render_registration_figure(verified.records, registered_names)
    initial_camera_params = simple_radial_camera_params(
        CONFIG.image_width, CONFIG.image_height, CONFIG.focal_35mm
    )
    selected_for_summary = AttemptMetrics(
        name=selected.name,
        workspace=selected.workspace,
        overlap=selected.overlap,
        database=selected.database,
        models=selected.models,
        best_model=selected_model,
        runtime_seconds=selected.runtime_seconds,
        pycolmap_version=selected.pycolmap_version,
    )
    summary = build_step10_summary(
        attempts=tuple(attempts),
        selected=selected_for_summary,
        pycolmap_version=str(pycolmap.__version__),
        selection_manifest_sha256=verified.manifest_sha256,
        initial_camera_params=initial_camera_params,
        project_root=ROOT,
    )
    summary["best_model"]["ply_path"] = _portable_path(PLY_PATH, ROOT)  # type: ignore[index]
    summary["unregistered_images"] = [
        record.filename for record in verified.records if record.filename not in registered_names
    ]
    summary["unregistered_indices"] = [
        record.index for record in verified.records if record.filename not in registered_names
    ]
    _write_json(SUMMARY_JSON, summary)
    return summary


def run_stage(stage: str) -> dict[str, object]:
    if stage == "baseline":
        attempt = _baseline_stage()
        return _portable_attempt(attempt, ROOT)
    if stage == "retry":
        attempt = _retry_stage()
        return {"retry_required": False} if attempt is None else _portable_attempt(attempt, ROOT)
    if stage == "finalize":
        return _finalize_stage()
    if stage == "all":
        baseline = _baseline_stage()
        if attempt_requires_retry(baseline, CONFIG.expected_images):
            _retry_stage()
        return _finalize_stage()
    raise ValueError(f"unknown Step 10 stage: {stage}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "baseline", "retry", "finalize"),
        default="all",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_stage(args.stage)
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
