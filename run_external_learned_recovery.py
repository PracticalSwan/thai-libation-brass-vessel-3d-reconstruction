"""Step 13 staged external ALIKED + LightGlue recovery runner."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Sequence

import matplotlib
import numpy as np
import pycolmap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis_common import SelectedImageRecord, path_for_index, verify_selected_images
from external_learned_recovery import (
    ExternalLearnedConfig,
    PairMatchingMetrics,
    build_verified_match_database,
    choose_local_fallback,
    external_model_metric_accepted,
    generate_sequential_pairs,
    match_pairs,
    merge_pair_schedule,
    prepare_feature_cache,
    run_external_mapping,
    runtime_snapshot,
    smoke_frontend,
)
from learned_sparse_recovery import verify_step11_candidate_identity
from sparse_bridging import (
    BridgeCandidate,
    BridgePairMetrics,
    BridgeSearchConfig,
    boundary_bridge_summary,
    generate_candidate_pairs,
    select_bridge_pairs,
    summarize_bridge_pairs,
    targeted_gate,
)
from sparse_reconstruction import (
    AttemptMetrics,
    DatabaseMetrics,
    ModelMetrics,
    SparseRunConfig,
    copy_sparse_model,
    summarize_reconstruction,
)


ROOT = Path(__file__).resolve().parent
SELECTED_DIR = ROOT / "preprocessing" / "pycolmap_input" / "images"
SELECTION_MANIFEST = ROOT / "preprocessing" / "reports" / "selection_manifest.csv"
OUTPUT_ROOT = ROOT / "reconstruction" / "external_learned_recovery"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PREVIEWS_DIR = OUTPUT_ROOT / "previews"
WORK_DIR = OUTPUT_ROOT / "work"
FEATURE_CACHE_DIR = WORK_DIR / "features"
MAPPING_OUTPUT_DIR = WORK_DIR / "mapping_output"
BEST_DIR = OUTPUT_ROOT / "best"
CAPABILITY_JSON = REPORTS_DIR / "step13_capability.json"
CANDIDATES_CSV = REPORTS_DIR / "step13_candidates.csv"
BOUNDARY_JSON = REPORTS_DIR / "step13_boundary_summary.json"
ATTEMPT_JSON = REPORTS_DIR / "step13_attempt.json"
SUMMARY_JSON = REPORTS_DIR / "step13_summary.json"
REGISTERED_CSV = REPORTS_DIR / "step13_registered_images.csv"
BOUNDARY_FIGURE = PREVIEWS_DIR / "step13_01_boundary_comparison.png"
SPARSE_FIGURE = PREVIEWS_DIR / "step13_02_sparse_model.png"
REGISTRATION_FIGURE = PREVIEWS_DIR / "step13_03_registration.png"
MODEL_COMPARISON_FIGURE = PREVIEWS_DIR / "step13_04_model_comparison.png"
PLY_PATH = BEST_DIR / "points3D.ply"
STEP11_CANDIDATES = ROOT / "reconstruction" / "bridging" / "reports" / "step11_candidates.csv"
STEP10_BEST = ROOT / "reconstruction" / "sparse" / "best"
STEP11_BEST = ROOT / "reconstruction" / "bridging" / "best"

CONFIG = ExternalLearnedConfig()
SPARSE_CONFIG = SparseRunConfig()
BRIDGE_CONFIG = BridgeSearchConfig()
STAGES = ("capability", "diagnose", "map", "finalize", "all")
VISUAL_STATUSES = ("pending", "passed", "failed")


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is invalid: {path}")
    return payload


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"CSV report requires at least one row: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _portable_model(model: ModelMetrics) -> dict[str, object]:
    payload = model.to_dict()
    payload["model_path"] = _portable_path(model.model_path)
    return _json_safe(payload)  # type: ignore[return-value]


def _portable_attempt(attempt: AttemptMetrics) -> dict[str, object]:
    return {
        "name": attempt.name,
        "workspace": _portable_path(attempt.workspace),
        "overlap": attempt.overlap,
        "database": asdict(attempt.database),
        "models": [_portable_model(model) for model in attempt.models],
        "best_model": _portable_model(attempt.best_model),
        "runtime_seconds": attempt.runtime_seconds,
        "pycolmap_version": attempt.pycolmap_version,
    }


def _model_from_payload(payload: dict[str, object]) -> ModelMetrics:
    raw_path = Path(str(payload["model_path"]))
    model_path = raw_path if raw_path.is_absolute() else ROOT / raw_path
    return ModelMetrics(
        model_path=model_path,
        registered_images=int(payload["registered_images"]),
        total_images=int(payload["total_images"]),
        sparse_points=int(payload["sparse_points"]),
        observations=int(payload["observations"]),
        mean_track_length=(
            float(payload["mean_track_length"])
            if payload.get("mean_track_length") is not None
            else math.nan
        ),
        mean_reprojection_error=(
            float(payload["mean_reprojection_error"])
            if payload.get("mean_reprojection_error") is not None
            else math.nan
        ),
        camera_count=int(payload["camera_count"]),
        camera_model=str(payload["camera_model"]),
        camera_params=tuple(float(value) for value in payload.get("camera_params", [])),
        mean_observations_per_registered_image=(
            float(payload["mean_observations_per_registered_image"])
            if payload.get("mean_observations_per_registered_image") is not None
            else math.nan
        ),
        registered_image_names=tuple(
            str(value) for value in payload.get("registered_image_names", [])
        ),
    )


def _attempt_from_payload(payload: dict[str, object]) -> AttemptMetrics:
    database = payload.get("database")
    models = payload.get("models")
    best = payload.get("best_model")
    if not isinstance(database, dict) or not isinstance(models, list) or not models:
        raise ValueError("Step 13 completed attempt report is incomplete")
    if not isinstance(best, dict):
        raise ValueError("Step 13 completed attempt report has no best model")
    raw_workspace = Path(str(payload["workspace"]))
    workspace = raw_workspace if raw_workspace.is_absolute() else ROOT / raw_workspace
    return AttemptMetrics(
        name=str(payload["name"]),
        workspace=workspace,
        overlap=int(payload["overlap"]),
        database=DatabaseMetrics(
            image_count=int(database["image_count"]),
            feature_count=int(database["feature_count"]),
            matched_pair_count=int(database["matched_pair_count"]),
            verified_pair_count=int(database["verified_pair_count"]),
        ),
        models=tuple(_model_from_payload(item) for item in models if isinstance(item, dict)),
        best_model=_model_from_payload(best),
        runtime_seconds=float(payload.get("runtime_seconds", 0.0)),
        pycolmap_version=str(payload.get("pycolmap_version", "")),
    )


def _assert_model_matches(expected: ModelMetrics, measured: ModelMetrics) -> None:
    if expected.registered_images != measured.registered_images:
        raise ValueError("Step 13 model registered-image count does not match report")
    if expected.sparse_points != measured.sparse_points:
        raise ValueError("Step 13 model sparse-point count does not match report")
    if expected.camera_count != measured.camera_count or expected.camera_model != measured.camera_model:
        raise ValueError("Step 13 model camera contract does not match report")
    if not math.isclose(
        expected.mean_reprojection_error,
        measured.mean_reprojection_error,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("Step 13 model reprojection error does not match report")


def _progress(index: int, total: int, pair: tuple[str, str], matches: int) -> None:
    if index == 1 or index == total or index % 50 == 0:
        print(
            f"Step 13 matching {index}/{total}: {pair[0]} <-> {pair[1]} ({matches} raw)",
            file=sys.stderr,
            flush=True,
        )


def _capability_stage() -> dict[str, object]:
    verified = verify_selected_images(
        SELECTED_DIR, SELECTION_MANIFEST, expected_count=CONFIG.expected_images
    )
    first = path_for_index(verified.records, SELECTED_DIR, 1)
    second = path_for_index(verified.records, SELECTED_DIR, 2)
    snapshot = runtime_snapshot(CONFIG)
    smoke = smoke_frontend(first, second, CONFIG)
    status = "passed" if smoke.get("status") == "passed" else "blocked"
    payload: dict[str, object] = {
        "status": status,
        "selection_manifest_sha256": verified.manifest_sha256,
        "input_image_count": len(verified.records),
        "smoke_pair": {
            "first_selected_index": 1,
            "first_filename": verified.records[0].filename,
            "second_selected_index": 2,
            "second_filename": verified.records[1].filename,
        },
        "runtime": snapshot,
        "smoke": smoke,
    }
    _write_json(CAPABILITY_JSON, payload)
    return payload


def _write_candidates_csv(
    metrics: Sequence[BridgePairMetrics], selected: Sequence[BridgePairMetrics]
) -> None:
    selected_candidates = {pair.candidate for pair in selected}
    rows: list[dict[str, object]] = []
    for pair in metrics:
        candidate = pair.candidate
        rows.append(
            {
                "boundary": f"{candidate.boundary_left}-{candidate.boundary_right}",
                "boundary_left": candidate.boundary_left,
                "boundary_right": candidate.boundary_right,
                "left_index": candidate.left_index,
                "right_index": candidate.right_index,
                "left_filename": candidate.left_filename,
                "right_filename": candidate.right_filename,
                "sequence_gap": candidate.sequence_gap,
                "raw_matches": pair.raw_matches,
                "verified_inliers": pair.verified_inliers,
                "inlier_ratio": pair.inlier_ratio,
                "qualified": int(pair.qualified),
                "selected": int(candidate in selected_candidates),
                "frontend": CONFIG.frontend_name,
            }
        )
    _write_csv(CANDIDATES_CSV, rows)


def _remove_regular_file(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Step 13 transient path is not a regular file: {path}")
    path.unlink()


def _diagnose_stage() -> dict[str, object]:
    capability = _load_json(CAPABILITY_JSON, label="Step 13 capability report")
    if capability.get("status") != "passed":
        return {
            "status": "blocked",
            "reason": "Step 13 external ALIKED + LightGlue capability gate did not pass",
        }
    verified = verify_selected_images(
        SELECTED_DIR, SELECTION_MANIFEST, expected_count=CONFIG.expected_images
    )
    features, bundle, manifest_sha = prepare_feature_cache(
        SELECTED_DIR, SELECTION_MANIFEST, FEATURE_CACHE_DIR, CONFIG
    )
    if manifest_sha != verified.manifest_sha256:
        raise RuntimeError("Step 13 feature cache manifest provenance changed during diagnostics")
    candidates = generate_candidate_pairs(verified.records, BRIDGE_CONFIG)
    verify_step11_candidate_identity(candidates, STEP11_CANDIDATES)
    pairs = tuple(
        (candidate.left_filename, candidate.right_filename) for candidate in candidates
    )
    matches, match_metrics = match_pairs(
        features, pairs, bundle, progress=_progress
    )
    diagnostic_db = WORK_DIR / "diagnostic.db"
    diagnostic_pairs = WORK_DIR / "diagnostic_pairs.txt"
    _remove_regular_file(diagnostic_db)
    _remove_regular_file(diagnostic_pairs)
    build_verified_match_database(
        SELECTED_DIR,
        diagnostic_db,
        features,
        matches,
        diagnostic_pairs,
        SPARSE_CONFIG,
    )
    metrics = summarize_bridge_pairs(diagnostic_db, candidates, BRIDGE_CONFIG)
    selected = select_bridge_pairs(metrics, BRIDGE_CONFIG)
    _write_candidates_csv(metrics, selected)
    post_verification_raw_matches = sum(pair.raw_matches for pair in metrics)
    if post_verification_raw_matches > match_metrics.raw_match_count:
        raise RuntimeError("COLMAP post-verification raw-match total exceeds imported LightGlue total")
    gate = targeted_gate(selected, BRIDGE_CONFIG)
    payload: dict[str, object] = {
        "status": "completed",
        "frontend": CONFIG.frontend_name,
        "aliked_model_name": CONFIG.aliked_model_name,
        "pinned_lightglue_commit": CONFIG.pinned_commit,
        "selection_manifest_sha256": verified.manifest_sha256,
        "candidate_count": len(metrics),
        "selected_bridge_count": len(selected),
        "lightglue_raw_match_count_pre_verification": match_metrics.raw_match_count,
        "colmap_raw_match_count_post_verification": post_verification_raw_matches,
        "raw_matches_removed_during_verification": (
            match_metrics.raw_match_count - post_verification_raw_matches
        ),
        "raw_match_count": post_verification_raw_matches,
        "matching_runtime_seconds": match_metrics.runtime_seconds,
        "critical_boundaries": [list(boundary) for boundary in BRIDGE_CONFIG.boundaries],
        "boundaries": list(boundary_bridge_summary(metrics, selected, BRIDGE_CONFIG)),
        "targeted_allowed": gate.allowed,
        "targeted_gate_reason": gate.reason,
    }
    _write_json(BOUNDARY_JSON, payload)
    return payload


def _load_selected_bridges() -> tuple[BridgePairMetrics, ...]:
    if not CANDIDATES_CSV.is_file():
        raise ValueError(f"Step 13 candidate report is missing: {CANDIDATES_CSV}")
    with CANDIDATES_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected: list[BridgePairMetrics] = []
    for row in rows:
        if int(row.get("selected", "0")) != 1:
            continue
        candidate = BridgeCandidate(
            boundary_left=int(row["boundary_left"]),
            boundary_right=int(row["boundary_right"]),
            left_index=int(row["left_index"]),
            right_index=int(row["right_index"]),
            left_filename=row["left_filename"],
            right_filename=row["right_filename"],
        )
        selected.append(
            BridgePairMetrics(
                candidate=candidate,
                raw_matches=int(row["raw_matches"]),
                verified_inliers=int(row["verified_inliers"]),
                inlier_ratio=float(row["inlier_ratio"]),
                qualified=bool(int(row["qualified"])),
            )
        )
    return tuple(selected)


def _record_mapping_skip(boundary: dict[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "skipped",
        "reason": str(boundary.get("targeted_gate_reason", "diagnostic boundary gate failed")),
        "frontend": CONFIG.frontend_name,
        "attempt": None,
        "metric_acceptance_met": False,
    }
    _write_json(ATTEMPT_JSON, payload)
    return payload


def _map_stage() -> dict[str, object]:
    boundary = _load_json(BOUNDARY_JSON, label="Step 13 boundary report")
    if boundary.get("status") != "completed":
        raise ValueError("Step 13 diagnostics are not complete")
    if boundary.get("targeted_allowed") is not True:
        return _record_mapping_skip(boundary)
    if ATTEMPT_JSON.is_file():
        existing = _load_json(ATTEMPT_JSON, label="Step 13 attempt report")
        if existing.get("status") == "completed":
            return existing
    verified = verify_selected_images(
        SELECTED_DIR, SELECTION_MANIFEST, expected_count=CONFIG.expected_images
    )
    selected = _load_selected_bridges()
    gate = targeted_gate(selected, BRIDGE_CONFIG)
    if not gate.allowed:
        raise RuntimeError("Step 13 candidate CSV does not satisfy its recorded boundary gate")
    features, bundle, _ = prepare_feature_cache(
        SELECTED_DIR, SELECTION_MANIFEST, FEATURE_CACHE_DIR, CONFIG
    )
    sequential = generate_sequential_pairs(
        verified.records, overlap=CONFIG.sequential_overlap
    )
    if len(sequential) != 5550:
        raise RuntimeError(f"Step 13 sequential pair schedule changed: {len(sequential)}")
    bridge_pairs = tuple(
        (pair.candidate.left_filename, pair.candidate.right_filename) for pair in selected
    )
    schedule = merge_pair_schedule(sequential, bridge_pairs)
    mapping_work = WORK_DIR / "mapping"
    if mapping_work.exists():
        _safe_remove_tree(mapping_work)
    if MAPPING_OUTPUT_DIR.exists():
        _safe_remove_tree(MAPPING_OUTPUT_DIR)
    result = run_external_mapping(
        SELECTED_DIR,
        MAPPING_OUTPUT_DIR,
        mapping_work,
        features,
        schedule,
        bundle,
        SPARSE_CONFIG,
        progress=_progress,
    )
    accepted = external_model_metric_accepted(result.attempt.best_model, BRIDGE_CONFIG)
    payload = {
        "status": "completed",
        "frontend": CONFIG.frontend_name,
        "aliked_model_name": CONFIG.aliked_model_name,
        "pinned_lightglue_commit": CONFIG.pinned_commit,
        "pair_schedule_count": len(schedule),
        "sequential_pair_count": len(sequential),
        "selected_bridge_count": len(selected),
        "pair_matching": asdict(result.pair_metrics),
        "attempt": _portable_attempt(result.attempt),
        "metric_acceptance_met": accepted,
    }
    _write_json(ATTEMPT_JSON, payload)
    return payload


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
    centers = np.asarray(
        [
            reconstruction.image(int(image_id)).projection_center()
            for image_id in reconstruction.reg_image_ids()
        ],
        dtype=float,
    )
    return points, colors, centers


def _render_sparse_model(model: ModelMetrics) -> None:
    points, colors, centers = _sparse_arrays(model.model_path)
    if not len(points) or not len(centers):
        raise ValueError("Step 13 selected sparse model cannot be visualized")
    if len(points) > 40000:
        stride = math.ceil(len(points) / 40000)
        points, colors = points[::stride], colors[::stride]
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        points[:, 0], points[:, 1], points[:, 2], s=1.0,
        c=colors if len(colors) == len(points) else None, alpha=0.65,
    )
    axis.scatter(
        centers[:, 0], centers[:, 1], centers[:, 2], s=14, marker="^",
        label="registered camera centers",
    )
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.set_title(
        f"Step 13 ALIKED + LightGlue | registered={model.registered_images}/288 | "
        f"points={model.sparse_points:,} | reproj={model.mean_reprojection_error:.3f}px"
    )
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(SPARSE_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _render_registration(
    records: Sequence[SelectedImageRecord], registered_names: set[str]
) -> None:
    indices = np.asarray([record.index for record in records], dtype=int)
    registered = np.asarray(
        [1 if record.filename in registered_names else 0 for record in records], dtype=int
    )
    figure, axis = plt.subplots(figsize=(14, 4.5))
    axis.step(indices, registered, where="mid", linewidth=1.2)
    for left, right in BRIDGE_CONFIG.boundaries:
        axis.axvline((left + right) / 2.0, linestyle="--", alpha=0.5)
    axis.set_ylim(-0.15, 1.2)
    axis.set_yticks([0, 1], ["unregistered", "registered"])
    axis.set_xlabel("Selected sequence index")
    axis.set_title("Step 13 strongest single-model registration")
    axis.grid(axis="x", alpha=0.15)
    figure.tight_layout()
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(REGISTRATION_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_boundary_comparison() -> Path | None:
    if not CANDIDATES_CSV.is_file():
        return None
    with CANDIDATES_CSV.open("r", encoding="utf-8", newline="") as handle:
        learned_rows = list(csv.DictReader(handle))
    candidates = tuple(
        BridgeCandidate(
            boundary_left=int(row["boundary_left"]),
            boundary_right=int(row["boundary_right"]),
            left_index=int(row["left_index"]),
            right_index=int(row["right_index"]),
            left_filename=row["left_filename"],
            right_filename=row["right_filename"],
        )
        for row in learned_rows
    )
    verify_step11_candidate_identity(candidates, STEP11_CANDIDATES)
    with STEP11_CANDIDATES.open("r", encoding="utf-8", newline="") as handle:
        sift_rows = list(csv.DictReader(handle))
    figure, axes = plt.subplots(len(BRIDGE_CONFIG.boundaries), 1, figsize=(12, 12))
    for axis, (left, right) in zip(np.atleast_1d(axes), BRIDGE_CONFIG.boundaries, strict=True):
        name = f"{left}-{right}"
        learned = [row for row in learned_rows if row["boundary"] == name]
        sift = [row for row in sift_rows if row["boundary"] == name]
        x = np.arange(len(learned), dtype=int)
        sift_inliers = np.asarray([int(row["verified_inliers"]) for row in sift], dtype=float)
        learned_inliers = np.asarray([int(row["verified_inliers"]) for row in learned], dtype=float)
        axis.scatter(x, sift_inliers, s=8, alpha=0.45, label="Step 11 SIFT")
        axis.scatter(x, learned_inliers, s=8, alpha=0.45, label="Step 13 ALIKED + LightGlue")
        axis.set_ylim(bottom=0)
        axis.set_title(
            f"Boundary {name} | SIFT max={int(sift_inliers.max()) if len(sift_inliers) else 0} | "
            f"learned max={int(learned_inliers.max()) if len(learned_inliers) else 0}"
        )
        axis.set_ylabel("Verified inliers")
        axis.set_xlabel("Frozen candidate ordinal")
        axis.grid(alpha=0.15)
        axis.legend(fontsize=8)
    figure.suptitle("Exact 2,340-pair boundary comparison", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(BOUNDARY_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return BOUNDARY_FIGURE


def render_model_comparison(
    step10: ModelMetrics, step11: ModelMetrics, step13: ModelMetrics | None
) -> Path:
    labels = ["Step 10 SIFT", "Step 11 exhaustive SIFT"]
    models = [step10, step11]
    if step13 is not None:
        labels.append("Step 13 ALIKED + LightGlue")
        models.append(step13)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(models))
    axes[0].bar(x, [model.registered_images for model in models])
    axes[0].axhline(BRIDGE_CONFIG.minimum_registered_images, linestyle="--")
    axes[0].set_ylabel("Registered images in strongest single model")
    axes[1].bar(x, [model.sparse_points for model in models])
    axes[1].axhline(BRIDGE_CONFIG.minimum_sparse_points, linestyle="--")
    axes[1].set_ylabel("Sparse points")
    axes[2].bar(x, [model.mean_reprojection_error for model in models])
    axes[2].set_ylabel("Mean reprojection error (px)")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.15)
    figure.suptitle("Strongest single sparse model comparison", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(MODEL_COMPARISON_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return MODEL_COMPARISON_FIGURE


def _write_registered_csv(
    records: Sequence[SelectedImageRecord], registered_names: set[str]
) -> None:
    rows = [
        {
            "selected_index": record.index,
            "filename": record.filename,
            "registered": int(record.filename in registered_names),
            "selected_frontend": CONFIG.frontend_name,
        }
        for record in records
    ]
    _write_csv(REGISTERED_CSV, rows)


def _safe_remove_tree(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"Step 13 cleanup path is not a regular directory: {path}")
    resolved = path.resolve()
    work_resolved = WORK_DIR.resolve()
    try:
        resolved.relative_to(work_resolved)
    except ValueError as error:
        raise ValueError(f"Step 13 cleanup path escaped work root: {path}") from error
    shutil.rmtree(path)


def cleanup_transient_work(work_dir: Path = WORK_DIR) -> tuple[str, ...]:
    if not work_dir.exists():
        return ()
    if not work_dir.is_dir() or work_dir.is_symlink():
        raise ValueError(f"Step 13 work path is not a regular directory: {work_dir}")
    removed: list[str] = []
    for name in ("features", "mapping", "mapping_output"):
        path = work_dir / name
        if path.exists():
            _safe_remove_tree(path)
            removed.append(name)
    for name in ("diagnostic.db", "diagnostic_pairs.txt"):
        path = work_dir / name
        if path.exists():
            _remove_regular_file(path)
            removed.append(name)
    if work_dir.exists() and not any(work_dir.iterdir()):
        work_dir.rmdir()
    return tuple(removed)


def _finalize_stage(visual_status: str = "pending") -> dict[str, object]:
    if visual_status not in VISUAL_STATUSES:
        raise ValueError(f"invalid Step 13 visual status: {visual_status}")
    verified = verify_selected_images(
        SELECTED_DIR, SELECTION_MANIFEST, expected_count=CONFIG.expected_images
    )
    capability = _load_json(CAPABILITY_JSON, label="Step 13 capability report")
    attempt_report = (
        _load_json(ATTEMPT_JSON, label="Step 13 attempt report")
        if ATTEMPT_JSON.is_file()
        else {
            "status": "not_run",
            "reason": "Step 13 mapping stage did not run",
            "metric_acceptance_met": False,
        }
    )
    step10 = summarize_reconstruction(STEP10_BEST, CONFIG.expected_images)
    step11 = summarize_reconstruction(STEP11_BEST, CONFIG.expected_images)
    fallback_label, fallback_model = choose_local_fallback(
        ("step10", step10), ("step11", step11)
    )

    selected_model: ModelMetrics | None = None
    if attempt_report.get("status") == "completed":
        raw_attempt = attempt_report.get("attempt")
        if not isinstance(raw_attempt, dict):
            raise ValueError("completed Step 13 attempt has no attempt payload")
        attempt = _attempt_from_payload(raw_attempt)
        measured_source = summarize_reconstruction(
            attempt.best_model.model_path, CONFIG.expected_images
        )
        _assert_model_matches(attempt.best_model, measured_source)
        copy_sparse_model(measured_source.model_path, BEST_DIR)
        selected_model = summarize_reconstruction(BEST_DIR, CONFIG.expected_images)
        _assert_model_matches(measured_source, selected_model)
        pycolmap.Reconstruction(BEST_DIR).export_PLY(PLY_PATH)
        registered_names = set(selected_model.registered_image_names)
        _write_registered_csv(verified.records, registered_names)
        _render_sparse_model(selected_model)
        _render_registration(verified.records, registered_names)

    render_boundary_comparison()
    render_model_comparison(step10, step11, selected_model)
    metric_acceptance = bool(
        selected_model is not None
        and external_model_metric_accepted(selected_model, BRIDGE_CONFIG)
    )
    if visual_status == "passed" and not metric_acceptance:
        raise ValueError("Step 13 visual status cannot pass before metric acceptance")
    if not metric_acceptance:
        visual_status = "failed"
    success = bool(metric_acceptance and visual_status == "passed")
    selected_sparse_source = BEST_DIR if success else fallback_model.model_path
    selected_reason = (
        "Step 13 external ALIKED + LightGlue passed metric and visual acceptance."
        if success
        else (
            f"Step 13 did not produce an accepted global model; selected {fallback_label} local "
            "model by registered images, sparse points, then reprojection error."
        )
    )
    local_fallback = {
        "label": fallback_label,
        "model": _portable_model(fallback_model),
        "step10": _portable_model(step10),
        "step11": _portable_model(step11),
    }
    summary: dict[str, object] = {
        "selection_manifest_sha256": verified.manifest_sha256,
        "input_image_count": len(verified.records),
        "frontend": CONFIG.frontend_name,
        "aliked_model_name": CONFIG.aliked_model_name,
        "pinned_lightglue_commit": CONFIG.pinned_commit,
        "critical_boundaries": [list(boundary) for boundary in BRIDGE_CONFIG.boundaries],
        "candidate_count": 2340,
        "sequential_pair_count": 5550,
        "capability": capability,
        "boundary_result": (
            _load_json(BOUNDARY_JSON, label="Step 13 boundary report")
            if BOUNDARY_JSON.is_file()
            else None
        ),
        "attempt_result": attempt_report,
        "best_step13_model": _portable_model(selected_model) if selected_model else None,
        "metric_acceptance_met": metric_acceptance,
        "visual_plausibility_status": visual_status,
        "step13_success": success,
        "local_fallback": local_fallback,
        "selected_sparse_source": _portable_path(selected_sparse_source),
        "selected_sparse_reason": selected_reason,
        "dense_reconstruction_started": False,
        "next_boundary": (
            "Use the accepted Step 13 global sparse model for the next dense-reconstruction phase."
            if success
            else "Sparse recovery is closed; continue downstream from the selected local sparse model."
        ),
    }
    _write_json(SUMMARY_JSON, summary)
    if visual_status != "pending":
        cleanup_transient_work(WORK_DIR)
    return summary


def run_stage(stage: str, *, visual_status: str = "pending") -> dict[str, object]:
    if stage == "capability":
        return _capability_stage()
    if stage == "diagnose":
        return _diagnose_stage()
    if stage == "map":
        return _map_stage()
    if stage == "finalize":
        return _finalize_stage(visual_status)
    if stage == "all":
        capability = _capability_stage()
        if capability.get("status") != "passed":
            return _finalize_stage("failed")
        boundary = _diagnose_stage()
        if boundary.get("status") != "completed":
            return _finalize_stage("failed")
        if boundary.get("targeted_allowed") is not True:
            _record_mapping_skip(boundary)
            return _finalize_stage("failed")
        attempt = _map_stage()
        if attempt.get("metric_acceptance_met") is not True:
            return _finalize_stage("failed")
        return _finalize_stage(visual_status)
    raise ValueError(f"unknown Step 13 stage: {stage}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--visual-status", choices=VISUAL_STATUSES, default="pending")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_stage(args.stage, visual_status=args.visual_status)
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
