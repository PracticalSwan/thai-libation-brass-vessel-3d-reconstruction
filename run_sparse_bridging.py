"""Run the bounded Step 11 sparse-component bridging workflow."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import csv
import json
import math
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import pycolmap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis_common import SelectedImageRecord, verify_selected_images
from sparse_bridging import (
    BridgeCandidate,
    BridgePairMetrics,
    BridgeSearchConfig,
    boundary_bridge_summary,
    bridge_model_accepted,
    choose_bridge_attempt,
    generate_candidate_pairs,
    prepare_feature_cache,
    run_bridge_diagnostics,
    run_exhaustive_attempt,
    run_targeted_attempt,
    select_bridge_pairs,
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
BRIDGING_ROOT = ROOT / "reconstruction" / "bridging"
WORK_DIR = BRIDGING_ROOT / "work"
TARGETED_DIR = BRIDGING_ROOT / "targeted"
EXHAUSTIVE_DIR = BRIDGING_ROOT / "exhaustive"
BEST_DIR = BRIDGING_ROOT / "best"
REPORTS_DIR = BRIDGING_ROOT / "reports"
PREVIEWS_DIR = BRIDGING_ROOT / "previews"
CANDIDATES_CSV = REPORTS_DIR / "step11_candidates.csv"
BOUNDARY_SUMMARY_JSON = REPORTS_DIR / "step11_boundary_summary.json"
TARGETED_REPORT = REPORTS_DIR / "step11_targeted.json"
EXHAUSTIVE_REPORT = REPORTS_DIR / "step11_exhaustive.json"
ATTEMPTS_CSV = REPORTS_DIR / "step11_attempts.csv"
REGISTERED_CSV = REPORTS_DIR / "step11_registered_images.csv"
SUMMARY_JSON = REPORTS_DIR / "step11_summary.json"
CANDIDATE_FIGURE = PREVIEWS_DIR / "step11_01_bridge_candidates.png"
SPARSE_FIGURE = PREVIEWS_DIR / "step11_02_sparse_model.png"
REGISTRATION_FIGURE = PREVIEWS_DIR / "step11_03_registration.png"
PLY_PATH = BEST_DIR / "points3D.ply"
STEP10_SUMMARY = ROOT / "reconstruction" / "reports" / "step10_summary.json"
SPARSE_CONFIG = SparseRunConfig()
BRIDGE_CONFIG = BridgeSearchConfig()


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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
    union_registered = set().union(
        *(set(model.registered_image_names) for model in attempt.models)
    )
    return {
        "name": attempt.name,
        "workspace": _portable_path(attempt.workspace),
        "overlap": attempt.overlap,
        "database": asdict(attempt.database),
        "models": [_portable_model(model) for model in attempt.models],
        "best_model": _portable_model(attempt.best_model),
        "model_union_registered_images": len(union_registered),
        "runtime_seconds": attempt.runtime_seconds,
        "pycolmap_version": attempt.pycolmap_version,
    }


def _model_from_payload(payload: dict[str, object]) -> ModelMetrics:
    raw_path = Path(str(payload["model_path"]))
    model_path = raw_path if raw_path.is_absolute() else ROOT / raw_path
    raw_track = payload.get("mean_track_length")
    raw_error = payload.get("mean_reprojection_error")
    raw_mean_observations = payload.get("mean_observations_per_registered_image")
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
            float(raw_mean_observations)
            if raw_mean_observations is not None
            else math.nan
        ),
        registered_image_names=tuple(
            str(value) for value in payload.get("registered_image_names", [])
        ),
    )


def _attempt_from_payload(payload: dict[str, object]) -> AttemptMetrics:
    database = payload.get("database")
    models_payload = payload.get("models")
    best_payload = payload.get("best_model")
    if not isinstance(database, dict):
        raise ValueError("Step 11 attempt has no database metrics")
    if not isinstance(models_payload, list) or not models_payload:
        raise ValueError("Step 11 attempt has no sparse models")
    if not isinstance(best_payload, dict):
        raise ValueError("Step 11 attempt has no best model")
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
        models=tuple(_model_from_payload(item) for item in models_payload),
        best_model=_model_from_payload(best_payload),
        runtime_seconds=float(payload.get("runtime_seconds", 0.0)),
        pycolmap_version=str(payload.get("pycolmap_version", "")),
    )


def _load_stage_report(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"Step 11 stage report is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") not in {
        "completed",
        "skipped",
    }:
        raise ValueError(f"invalid Step 11 stage report: {path}")
    return payload


def _load_step10_summary() -> dict[str, object]:
    if not STEP10_SUMMARY.is_file():
        raise ValueError(f"Step 10 summary is missing: {STEP10_SUMMARY}")
    payload = json.loads(STEP10_SUMMARY.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("best_model"), dict):
        raise ValueError("Step 10 summary has no best-model evidence")
    best = payload["best_model"]
    if int(best.get("registered_images", -1)) != 73:
        raise ValueError("Step 10 baseline registration evidence is not 73 images")
    if int(best.get("sparse_points", -1)) != 6099:
        raise ValueError("Step 10 baseline sparse-point evidence is not 6099 points")
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
            }
        )
    if not rows:
        raise ValueError("candidate report requires at least one measured pair")
    CANDIDATES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with CANDIDATES_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_selected_bridges() -> tuple[BridgePairMetrics, ...]:
    if not CANDIDATES_CSV.is_file() or not BOUNDARY_SUMMARY_JSON.is_file():
        raise ValueError("completed Step 11 diagnostic reports are required")
    with CANDIDATES_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected: list[BridgePairMetrics] = []
    for row in rows:
        if int(row["selected"]) != 1:
            continue
        candidate = BridgeCandidate(
            boundary_left=int(row["boundary_left"]),
            boundary_right=int(row["boundary_right"]),
            left_index=int(row["left_index"]),
            right_index=int(row["right_index"]),
            left_filename=str(row["left_filename"]),
            right_filename=str(row["right_filename"]),
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


def _diagnose_stage() -> dict[str, object]:
    _load_step10_summary()
    verified = verify_selected_images(
        SELECTED_DIR,
        SELECTION_MANIFEST,
        expected_count=SPARSE_CONFIG.expected_images,
    )
    features_database = prepare_feature_cache(
        SELECTED_DIR, SELECTION_MANIFEST, WORK_DIR, SPARSE_CONFIG
    )
    candidates = generate_candidate_pairs(verified.records, BRIDGE_CONFIG)
    metrics = run_bridge_diagnostics(
        features_database, WORK_DIR, candidates, BRIDGE_CONFIG
    )
    selected = select_bridge_pairs(metrics, BRIDGE_CONFIG)
    _write_candidates_csv(metrics, selected)
    boundaries = boundary_bridge_summary(metrics, selected, BRIDGE_CONFIG)
    gate = targeted_gate(selected, BRIDGE_CONFIG)
    payload: dict[str, object] = {
        "pycolmap_version": str(pycolmap.__version__),
        "selection_manifest_sha256": verified.manifest_sha256,
        "critical_boundaries": [list(boundary) for boundary in BRIDGE_CONFIG.boundaries],
        "window_size": BRIDGE_CONFIG.window_size,
        "minimum_sequence_gap": BRIDGE_CONFIG.minimum_sequence_gap,
        "minimum_verified_inliers": BRIDGE_CONFIG.minimum_verified_inliers,
        "minimum_inlier_ratio": BRIDGE_CONFIG.minimum_inlier_ratio,
        "candidate_count": len(metrics),
        "selected_bridge_count": len(selected),
        "boundaries": list(boundaries),
        "targeted_allowed": gate.allowed,
        "targeted_gate_reason": gate.reason,
    }
    _write_json(BOUNDARY_SUMMARY_JSON, payload)
    return payload


def _targeted_stage() -> dict[str, object]:
    selected = _load_selected_bridges()
    gate = targeted_gate(selected, BRIDGE_CONFIG)
    if not gate.allowed:
        payload: dict[str, object] = {
            "status": "skipped",
            "reason": gate.reason,
            "attempt": None,
        }
        _write_json(TARGETED_REPORT, payload)
        return payload

    verify_selected_images(
        SELECTED_DIR,
        SELECTION_MANIFEST,
        expected_count=SPARSE_CONFIG.expected_images,
    )
    features_database = prepare_feature_cache(
        SELECTED_DIR, SELECTION_MANIFEST, WORK_DIR, SPARSE_CONFIG
    )
    attempt = run_targeted_attempt(
        SELECTED_DIR,
        features_database,
        TARGETED_DIR,
        selected,
        SPARSE_CONFIG,
        BRIDGE_CONFIG,
    )
    payload = {"status": "completed", "reason": "", "attempt": _portable_attempt(attempt)}
    _write_json(TARGETED_REPORT, payload)
    return payload


def _exhaustive_stage() -> dict[str, object]:
    targeted_report = _load_stage_report(TARGETED_REPORT)
    if targeted_report["status"] == "completed":
        attempt_payload = targeted_report.get("attempt")
        if not isinstance(attempt_payload, dict):
            raise ValueError("completed targeted report has no attempt metrics")
        targeted_attempt = _attempt_from_payload(attempt_payload)
        if bridge_model_accepted(targeted_attempt.best_model, BRIDGE_CONFIG):
            payload: dict[str, object] = {
                "status": "skipped",
                "reason": "targeted attempt already passes the Step 11 acceptance gate",
                "attempt": None,
            }
            _write_json(EXHAUSTIVE_REPORT, payload)
            return payload

    verify_selected_images(
        SELECTED_DIR,
        SELECTION_MANIFEST,
        expected_count=SPARSE_CONFIG.expected_images,
    )
    features_database = prepare_feature_cache(
        SELECTED_DIR, SELECTION_MANIFEST, WORK_DIR, SPARSE_CONFIG
    )
    print(
        "Step 11 exhaustive fallback: 288 images, CPU SIFT matching, "
        "block_size=50; this may outlive an interactive tool timeout."
    )
    attempt = run_exhaustive_attempt(
        SELECTED_DIR,
        features_database,
        EXHAUSTIVE_DIR,
        SPARSE_CONFIG,
        BRIDGE_CONFIG,
    )
    payload = {"status": "completed", "reason": "", "attempt": _portable_attempt(attempt)}
    _write_json(EXHAUSTIVE_REPORT, payload)
    return payload


def build_step11_summary(
    *,
    pycolmap_version: str,
    selection_manifest_sha256: str,
    step10_summary: dict[str, object],
    boundary_summary: dict[str, object],
    targeted_report: dict[str, object],
    exhaustive_report: dict[str, object],
    selected: AttemptMetrics,
    selected_records: Sequence[SelectedImageRecord],
    bridge_config: BridgeSearchConfig = BridgeSearchConfig(),
) -> dict[str, object]:
    step10_best = step10_summary.get("best_model")
    if not isinstance(step10_best, dict):
        raise ValueError("Step 10 summary has no best-model evidence")
    boundary_rows = boundary_summary.get("boundaries")
    if not isinstance(boundary_rows, list):
        raise ValueError("Step 11 boundary summary has no boundary rows")
    counts = {
        str(row["boundary"]): int(row["candidate_count"])
        for row in boundary_rows
    }
    qualified_counts = {
        str(row["boundary"]): int(row["qualified_count"])
        for row in boundary_rows
    }
    selected_counts = {
        str(row["boundary"]): int(row["selected_count"])
        for row in boundary_rows
    }
    model = selected.best_model
    registered_names = set(model.registered_image_names)
    registered_records = [
        record for record in selected_records if record.filename in registered_names
    ]
    unregistered_records = [
        record for record in selected_records if record.filename not in registered_names
    ]
    bridge_success = bridge_model_accepted(model, bridge_config)
    next_boundary = (
        "Step 11 is accepted; dense reconstruction is the next separately authorized phase."
        if bridge_success
        else (
            "Step 11 execution is complete but global sparse reconstruction remains "
            "below acceptance and dense reconstruction remains blocked."
        )
    )
    return {
        "pycolmap_version": pycolmap_version,
        "selection_manifest_sha256": selection_manifest_sha256,
        "step10_baseline": {
            "selected_attempt": step10_summary.get("selected_attempt", "baseline"),
            "registered_images": int(step10_best["registered_images"]),
            "sparse_points": int(step10_best["sparse_points"]),
            "mean_reprojection_error": step10_best.get("mean_reprojection_error"),
            "camera_count": int(step10_best.get("camera_count", 0)),
            "camera_model": str(step10_best.get("camera_model", "")),
        },
        "critical_boundaries": [list(boundary) for boundary in bridge_config.boundaries],
        "candidate_count": int(boundary_summary.get("candidate_count", sum(counts.values()))),
        "candidate_counts": counts,
        "qualified_counts": qualified_counts,
        "selected_bridge_counts": selected_counts,
        "targeted_result": targeted_report,
        "exhaustive_result": exhaustive_report,
        "selected_attempt": selected.name,
        "best_model": _portable_model(model),
        "registered_images": [record.filename for record in registered_records],
        "registered_indices": [record.index for record in registered_records],
        "unregistered_images": [record.filename for record in unregistered_records],
        "unregistered_indices": [record.index for record in unregistered_records],
        "final_camera_params": list(model.camera_params),
        "bridge_success": bridge_success,
        "dense_reconstruction_started": False,
        "next_boundary": next_boundary,
    }


def _write_attempts_csv(
    targeted_report: dict[str, object], exhaustive_report: dict[str, object]
) -> None:
    rows: list[dict[str, object]] = []
    for expected_name, report in (
        ("targeted_bridges", targeted_report),
        ("exhaustive", exhaustive_report),
    ):
        status = str(report["status"])
        row: dict[str, object] = {
            "attempt": expected_name,
            "status": status,
            "reason": str(report.get("reason", "")),
            "model_count": "",
            "model_union_registered_images": "",
            "registered_images": "",
            "registration_fraction": "",
            "sparse_points": "",
            "observations": "",
            "mean_track_length": "",
            "mean_reprojection_error": "",
            "camera_count": "",
            "camera_model": "",
            "camera_params": "",
            "runtime_seconds": "",
            "feature_count": "",
            "matched_pair_count": "",
            "verified_pair_count": "",
        }
        if status == "completed":
            raw_attempt = report.get("attempt")
            if not isinstance(raw_attempt, dict):
                raise ValueError(f"completed {expected_name} report has no attempt")
            attempt = _attempt_from_payload(raw_attempt)
            model = attempt.best_model
            row.update(
                {
                    "attempt": attempt.name,
                    "model_count": len(attempt.models),
                    "model_union_registered_images": raw_attempt.get(
                        "model_union_registered_images", ""
                    ),
                    "registered_images": model.registered_images,
                    "registration_fraction": model.registration_fraction,
                    "sparse_points": model.sparse_points,
                    "observations": model.observations,
                    "mean_track_length": model.mean_track_length,
                    "mean_reprojection_error": model.mean_reprojection_error,
                    "camera_count": model.camera_count,
                    "camera_model": model.camera_model,
                    "camera_params": json.dumps(list(model.camera_params)),
                    "runtime_seconds": attempt.runtime_seconds,
                    "feature_count": attempt.database.feature_count,
                    "matched_pair_count": attempt.database.matched_pair_count,
                    "verified_pair_count": attempt.database.verified_pair_count,
                }
            )
        rows.append(row)
    ATTEMPTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with ATTEMPTS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_registered_csv(
    records: Sequence[SelectedImageRecord],
    registered_names: set[str],
    selected_attempt: str,
) -> None:
    rows = [
        {
            "selected_index": record.index,
            "filename": record.filename,
            "registered": int(record.filename in registered_names),
            "selected_attempt": selected_attempt,
        }
        for record in records
    ]
    REGISTERED_CSV.parent.mkdir(parents=True, exist_ok=True)
    with REGISTERED_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_candidate_figure() -> None:
    if not CANDIDATES_CSV.is_file():
        raise ValueError(f"candidate report is missing: {CANDIDATES_CSV}")
    with CANDIDATES_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    figure, axes = plt.subplots(len(BRIDGE_CONFIG.boundaries), 1, figsize=(11, 13))
    for axis, (boundary_left, boundary_right) in zip(
        np.atleast_1d(axes), BRIDGE_CONFIG.boundaries, strict=True
    ):
        boundary_name = f"{boundary_left}-{boundary_right}"
        boundary_rows = [row for row in rows if row["boundary"] == boundary_name]
        left = np.asarray([int(row["left_index"]) for row in boundary_rows], dtype=int)
        right = np.asarray([int(row["right_index"]) for row in boundary_rows], dtype=int)
        inliers = np.asarray(
            [int(row["verified_inliers"]) for row in boundary_rows], dtype=float
        )
        sizes = 10.0 + np.sqrt(np.maximum(inliers, 0.0)) * 5.0
        scatter = axis.scatter(
            left,
            right,
            c=inliers,
            s=sizes,
            cmap="viridis",
            vmin=0.0,
            vmax=max(1.0, float(inliers.max()) if len(inliers) else 1.0),
            alpha=0.65,
            linewidths=0,
        )
        selected_rows = [row for row in boundary_rows if int(row["selected"]) == 1]
        if selected_rows:
            axis.scatter(
                [int(row["left_index"]) for row in selected_rows],
                [int(row["right_index"]) for row in selected_rows],
                marker="*",
                s=150,
                facecolors="none",
                edgecolors="red",
                linewidths=1.3,
                label="selected bridge",
            )
            axis.legend(loc="best", fontsize=8)
        axis.set_title(
            f"Boundary {boundary_name} | candidates={len(boundary_rows)} | "
            f"selected={len(selected_rows)}"
        )
        axis.set_xlabel("Left selected-image index")
        axis.set_ylabel("Right selected-image index")
        axis.grid(alpha=0.15)
        figure.colorbar(scatter, ax=axis, label="Verified geometric inliers")
    figure.suptitle("Step 11 measured non-local bridge candidates", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    CANDIDATE_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(CANDIDATE_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(figure)


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


def _render_sparse_figure(
    model_path: Path, model: ModelMetrics, selected_attempt: str
) -> None:
    points, colors, camera_centers = _sparse_arrays(model_path)
    if not len(points) or not len(camera_centers):
        raise ValueError("selected Step 11 model cannot be visualized")
    if len(points) > 40000:
        stride = math.ceil(len(points) / 40000)
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
        f"Step 11 {selected_attempt} | registered={model.registered_images}/"
        f"{model.total_images} | points={model.sparse_points:,} | "
        f"reproj={model.mean_reprojection_error:.3f}px"
    )
    axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    SPARSE_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(SPARSE_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _render_registration_figure(
    records: Sequence[SelectedImageRecord],
    registered_names: set[str],
    selected_attempt: str,
) -> None:
    indices = np.asarray([record.index for record in records], dtype=int)
    registered = np.asarray(
        [int(record.filename in registered_names) for record in records], dtype=int
    )
    figure, axis = plt.subplots(figsize=(14, 4.5))
    axis.step(indices, registered, where="mid", linewidth=1.2)
    unregistered = indices[registered == 0]
    if len(unregistered):
        axis.scatter(unregistered, np.zeros_like(unregistered), marker="x", s=30)
    for boundary_left, boundary_right in BRIDGE_CONFIG.boundaries:
        axis.axvspan(boundary_left, boundary_right, alpha=0.14, color="tab:red")
        axis.text(
            (boundary_left + boundary_right) / 2,
            1.08,
            f"{boundary_left}-{boundary_right}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_xlim(1, SPARSE_CONFIG.expected_images)
    axis.set_ylim(-0.15, 1.2)
    axis.set_yticks([0, 1], ["unregistered", "registered"])
    axis.set_xlabel("Selected sequence index (1..288)")
    axis.set_title(
        f"Step 11 selected-model registration | {selected_attempt} | "
        f"registered={int(registered.sum())}/{SPARSE_CONFIG.expected_images}"
    )
    axis.grid(axis="x", alpha=0.15)
    figure.tight_layout()
    REGISTRATION_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(REGISTRATION_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _assert_model_matches(reported: ModelMetrics, measured: ModelMetrics) -> None:
    if reported.registered_images != measured.registered_images:
        raise ValueError("Step 11 model registration count does not match its report")
    if reported.sparse_points != measured.sparse_points:
        raise ValueError("Step 11 model sparse-point count does not match its report")
    if reported.camera_count != measured.camera_count:
        raise ValueError("Step 11 model camera count does not match its report")
    if reported.camera_model != measured.camera_model:
        raise ValueError("Step 11 model camera model does not match its report")
    if not math.isclose(
        reported.mean_reprojection_error,
        measured.mean_reprojection_error,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("Step 11 model reprojection error does not match its report")


def _finalize_stage() -> dict[str, object]:
    verified = verify_selected_images(
        SELECTED_DIR,
        SELECTION_MANIFEST,
        expected_count=SPARSE_CONFIG.expected_images,
    )
    step10_summary = _load_step10_summary()
    if not BOUNDARY_SUMMARY_JSON.is_file():
        raise ValueError("Step 11 boundary summary is missing")
    boundary_summary = json.loads(BOUNDARY_SUMMARY_JSON.read_text(encoding="utf-8"))
    if not isinstance(boundary_summary, dict):
        raise ValueError("Step 11 boundary summary is invalid")
    targeted_report = _load_stage_report(TARGETED_REPORT)
    if EXHAUSTIVE_REPORT.is_file():
        exhaustive_report = _load_stage_report(EXHAUSTIVE_REPORT)
    else:
        raw_targeted = targeted_report.get("attempt")
        if targeted_report["status"] != "completed" or not isinstance(raw_targeted, dict):
            raise ValueError("required Step 11 exhaustive result is missing")
        targeted_attempt = _attempt_from_payload(raw_targeted)
        if not bridge_model_accepted(targeted_attempt.best_model, BRIDGE_CONFIG):
            raise ValueError("required Step 11 exhaustive result is missing")
        exhaustive_report = {
            "status": "skipped",
            "reason": "targeted attempt already passes the Step 11 acceptance gate",
            "attempt": None,
        }

    attempts: list[AttemptMetrics] = []
    for report in (targeted_report, exhaustive_report):
        if report["status"] != "completed":
            continue
        raw_attempt = report.get("attempt")
        if not isinstance(raw_attempt, dict):
            raise ValueError("completed Step 11 report has no attempt metrics")
        attempts.append(_attempt_from_payload(raw_attempt))
    selected = choose_bridge_attempt(attempts, BRIDGE_CONFIG)
    measured_source = summarize_reconstruction(
        selected.best_model.model_path, SPARSE_CONFIG.expected_images
    )
    _assert_model_matches(selected.best_model, measured_source)
    copy_sparse_model(measured_source.model_path, BEST_DIR)
    selected_model = summarize_reconstruction(BEST_DIR, SPARSE_CONFIG.expected_images)
    _assert_model_matches(measured_source, selected_model)
    pycolmap.Reconstruction(BEST_DIR).export_PLY(PLY_PATH)

    selected_for_summary = replace(
        selected,
        best_model=selected_model,
        models=tuple(
            selected_model if model.model_path == selected.best_model.model_path else model
            for model in selected.models
        ),
    )
    registered_names = set(selected_model.registered_image_names)
    _write_attempts_csv(targeted_report, exhaustive_report)
    _write_registered_csv(verified.records, registered_names, selected.name)
    _render_candidate_figure()
    _render_sparse_figure(BEST_DIR, selected_model, selected.name)
    _render_registration_figure(verified.records, registered_names, selected.name)
    summary = build_step11_summary(
        pycolmap_version=str(pycolmap.__version__),
        selection_manifest_sha256=verified.manifest_sha256,
        step10_summary=step10_summary,
        boundary_summary=boundary_summary,
        targeted_report=targeted_report,
        exhaustive_report=exhaustive_report,
        selected=selected_for_summary,
        selected_records=verified.records,
        bridge_config=BRIDGE_CONFIG,
    )
    summary["best_model"]["ply_path"] = _portable_path(PLY_PATH)  # type: ignore[index]
    _write_json(SUMMARY_JSON, summary)
    return summary


TRANSIENT_WORK_FILENAMES = (
    "features.db",
    "features_complete.json",
    "diagnostic.db",
    "diagnostic_pairs.txt",
    "targeted.db",
    "targeted_bridge_pairs.txt",
    "exhaustive.db",
)


def cleanup_transient_work(work_dir: Path = WORK_DIR) -> tuple[Path, ...]:
    if not work_dir.exists():
        return ()
    if not work_dir.is_dir() or work_dir.is_symlink():
        raise ValueError(f"Step 11 work path is not a regular directory: {work_dir}")
    resolved_work = work_dir.resolve()
    removed: list[Path] = []
    for filename in TRANSIENT_WORK_FILENAMES:
        path = work_dir / filename
        if path.resolve().parent != resolved_work:
            raise ValueError(f"transient cleanup target escaped work directory: {path}")
        if not path.exists():
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"transient cleanup target is not a regular file: {path}")
        path.unlink()
        removed.append(path)
    if not any(work_dir.iterdir()):
        work_dir.rmdir()
    return tuple(removed)


def run_stage(stage: str) -> dict[str, object]:
    if stage == "diagnose":
        return _diagnose_stage()
    if stage == "targeted":
        return _targeted_stage()
    if stage == "exhaustive":
        return _exhaustive_stage()
    if stage == "finalize":
        return _finalize_stage()
    if stage == "all":
        _diagnose_stage()
        _targeted_stage()
        _exhaustive_stage()
        return _finalize_stage()
    raise ValueError(f"unknown Step 11 stage: {stage}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "diagnose", "targeted", "exhaustive", "finalize"),
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
