"""Run the bounded Step 12 native learned sparse-recovery workflow."""

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
from PIL import Image
import pycolmap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis_common import (
    SelectedImageRecord,
    path_for_index,
    verify_selected_images,
)
from learned_sparse_recovery import (
    ALIKED_FRONTEND,
    LOMA_FRONTEND,
    FrontendSmokeResult,
    LearnedFrontendSpec,
    build_capability_snapshot,
    learned_model_metric_accepted,
    prepare_learned_feature_cache,
    run_learned_diagnostics,
    run_learned_targeted_attempt,
    smoke_frontend,
    verify_step11_candidate_identity,
)
from sparse_bridging import (
    BridgeCandidate,
    BridgePairMetrics,
    BridgeSearchConfig,
    boundary_bridge_summary,
    generate_candidate_pairs,
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
LEARNED_ROOT = ROOT / "reconstruction" / "learned_recovery"
ALIKED_DIR = LEARNED_ROOT / "aliked"
LOMA_DIR = LEARNED_ROOT / "loma"
BEST_DIR = LEARNED_ROOT / "best"
REPORTS_DIR = LEARNED_ROOT / "reports"
PREVIEWS_DIR = LEARNED_ROOT / "previews"
WORK_DIR = LEARNED_ROOT / "work"
CAPABILITY_JSON = REPORTS_DIR / "step12_capability.json"
ALIKED_CANDIDATES_CSV = REPORTS_DIR / "step12_aliked_candidates.csv"
ALIKED_BOUNDARY_JSON = REPORTS_DIR / "step12_aliked_boundary_summary.json"
ALIKED_ATTEMPT_JSON = REPORTS_DIR / "step12_aliked_attempt.json"
LOMA_CANDIDATES_CSV = REPORTS_DIR / "step12_loma_candidates.csv"
LOMA_BOUNDARY_JSON = REPORTS_DIR / "step12_loma_boundary_summary.json"
LOMA_ATTEMPT_JSON = REPORTS_DIR / "step12_loma_attempt.json"
ATTEMPTS_CSV = REPORTS_DIR / "step12_attempts.csv"
REGISTERED_CSV = REPORTS_DIR / "step12_registered_images.csv"
SUMMARY_JSON = REPORTS_DIR / "step12_summary.json"
BOUNDARY_FIGURE = PREVIEWS_DIR / "step12_01_boundary_comparison.png"
SPARSE_FIGURE = PREVIEWS_DIR / "step12_02_sparse_model.png"
REGISTRATION_FIGURE = PREVIEWS_DIR / "step12_03_registration.png"
FRONTEND_FIGURE = PREVIEWS_DIR / "step12_04_frontend_comparison.png"
PLY_PATH = BEST_DIR / "points3D.ply"
STEP10_SUMMARY = ROOT / "reconstruction" / "reports" / "step10_summary.json"
STEP11_EXHAUSTIVE = (
    ROOT / "reconstruction" / "bridging" / "reports" / "step11_exhaustive.json"
)
STEP11_CANDIDATES = (
    ROOT / "reconstruction" / "bridging" / "reports" / "step11_candidates.csv"
)

SPARSE_CONFIG = SparseRunConfig()
BRIDGE_CONFIG = BridgeSearchConfig()
STAGES = (
    "capability",
    "aliked-diagnose",
    "aliked-map",
    "loma-diagnose",
    "loma-map",
    "finalize",
    "all",
)
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
    temporary = path.with_suffix(".json.tmp")
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
    models_payload = payload.get("models")
    best_payload = payload.get("best_model")
    if not isinstance(database, dict):
        raise ValueError("Step 12 attempt has no database metrics")
    if not isinstance(models_payload, list) or not models_payload:
        raise ValueError("Step 12 attempt has no sparse models")
    if not isinstance(best_payload, dict):
        raise ValueError("Step 12 attempt has no best model")
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


def load_rgb_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise ValueError(f"smoke image is missing: {path}")
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _not_run_smoke(frontend: LearnedFrontendSpec) -> FrontendSmokeResult:
    return FrontendSmokeResult(frontend.name, "not_run", 0, 0, 0)


def _capability_stage() -> dict[str, object]:
    verified = verify_selected_images(
        SELECTED_DIR, SELECTION_MANIFEST, expected_count=SPARSE_CONFIG.expected_images
    )
    first_record = verified.records[0]
    second_record = verified.records[1]
    first = load_rgb_image(path_for_index(verified.records, SELECTED_DIR, 1))
    second = load_rgb_image(path_for_index(verified.records, SELECTED_DIR, 2))
    snapshot = build_capability_snapshot()
    frontends = snapshot.get("frontends")
    if not isinstance(frontends, dict):
        raise ValueError("capability snapshot has no frontend options")
    for frontend in (ALIKED_FRONTEND, LOMA_FRONTEND):
        if not isinstance(frontends.get(frontend.name), dict):
            raise ValueError(f"capability snapshot is missing {frontend.name}")
    frontends["aliked"] = {
        **frontends["aliked"],
        "smoke": smoke_frontend(ALIKED_FRONTEND, first, second).to_dict(),
    }
    frontends["loma"] = {
        **frontends["loma"],
        "smoke": _not_run_smoke(LOMA_FRONTEND).to_dict(),
    }
    snapshot.update(
        {
            "selection_manifest_sha256": verified.manifest_sha256,
            "input_image_count": len(verified.records),
            "smoke_pair": {
                "first_selected_index": 1,
                "first_filename": first_record.filename,
                "second_selected_index": 2,
                "second_filename": second_record.filename,
            },
        }
    )
    _write_json(CAPABILITY_JSON, snapshot)
    return snapshot


def _capability_frontend(payload: dict[str, object], name: str) -> dict[str, object]:
    frontends = payload.get("frontends")
    if not isinstance(frontends, dict) or not isinstance(frontends.get(name), dict):
        raise ValueError(f"Step 12 capability report has no {name} evidence")
    return frontends[name]  # type: ignore[return-value]


def _ensure_loma_smoke() -> dict[str, object]:
    capability = _load_json(CAPABILITY_JSON, label="Step 12 capability report")
    loma = _capability_frontend(capability, "loma")
    smoke = loma.get("smoke")
    if not isinstance(smoke, dict):
        raise ValueError("Step 12 capability report has no LoMa smoke state")
    if smoke.get("status") == "not_run":
        verified = verify_selected_images(
            SELECTED_DIR, SELECTION_MANIFEST, expected_count=SPARSE_CONFIG.expected_images
        )
        first = load_rgb_image(path_for_index(verified.records, SELECTED_DIR, 1))
        second = load_rgb_image(path_for_index(verified.records, SELECTED_DIR, 2))
        loma["smoke"] = smoke_frontend(LOMA_FRONTEND, first, second).to_dict()
        _write_json(CAPABILITY_JSON, capability)
    return loma


def _frontend_paths(
    frontend: LearnedFrontendSpec,
) -> tuple[Path, Path, Path, Path]:
    if frontend.name == "aliked":
        return (
            ALIKED_CANDIDATES_CSV,
            ALIKED_BOUNDARY_JSON,
            ALIKED_ATTEMPT_JSON,
            ALIKED_DIR,
        )
    if frontend.name == "loma":
        return (
            LOMA_CANDIDATES_CSV,
            LOMA_BOUNDARY_JSON,
            LOMA_ATTEMPT_JSON,
            LOMA_DIR,
        )
    raise ValueError(f"unsupported Step 12 frontend: {frontend.name}")


def _write_candidates_csv(
    path: Path,
    metrics: Sequence[BridgePairMetrics],
    selected: Sequence[BridgePairMetrics],
    frontend: LearnedFrontendSpec,
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
                "frontend": frontend.name,
            }
        )
    _write_csv(path, rows)


def _load_selected_bridges(path: Path) -> tuple[BridgePairMetrics, ...]:
    if not path.is_file():
        raise ValueError(f"Step 12 candidate report is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
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


def loma_required(
    aliked_boundary: dict[str, object], aliked_attempt: dict[str, object]
) -> bool:
    if aliked_boundary.get("status") != "completed":
        return False
    if aliked_boundary.get("targeted_allowed") is False:
        return True
    return bool(
        aliked_attempt.get("status") == "completed"
        and aliked_attempt.get("metric_acceptance_met") is False
    )


def _diagnose_frontend(frontend: LearnedFrontendSpec) -> dict[str, object]:
    candidates_path, boundary_path, attempt_path, _ = _frontend_paths(frontend)
    capability = _load_json(CAPABILITY_JSON, label="Step 12 capability report")
    if frontend.name == "loma":
        aliked_boundary = _load_json(
            ALIKED_BOUNDARY_JSON, label="ALIKED boundary report"
        )
        aliked_attempt = _load_json(
            ALIKED_ATTEMPT_JSON, label="ALIKED attempt report"
        )
        if not loma_required(aliked_boundary, aliked_attempt):
            payload: dict[str, object] = {
                "status": "not_run",
                "reason": "ALIKED passed metric acceptance; LoMa fallback is not required",
                "frontend": frontend.name,
                "extractor_type": frontend.extractor_type.name,
                "matcher_type": frontend.matcher_type.name,
                "boundary_gate_allowed": None,
                "attempt": None,
                "metric_acceptance_met": None,
            }
            _write_json(attempt_path, payload)
            return payload
        frontend_capability = _ensure_loma_smoke()
    else:
        frontend_capability = _capability_frontend(capability, frontend.name)
    smoke = frontend_capability.get("smoke")
    if not isinstance(smoke, dict) or smoke.get("status") != "passed":
        exception_type = str(smoke.get("exception_type", "")) if isinstance(smoke, dict) else ""
        exception_message = (
            str(smoke.get("exception_message", "")) if isinstance(smoke, dict) else ""
        )
        reason = ": ".join(
            value for value in (exception_type, exception_message) if value
        ) or f"{frontend.name} native smoke did not pass"
        payload = {
            "status": "blocked",
            "reason": reason,
            "frontend": frontend.name,
            "extractor_type": frontend.extractor_type.name,
            "matcher_type": frontend.matcher_type.name,
            "boundary_gate_allowed": None,
            "attempt": None,
            "metric_acceptance_met": None,
        }
        _write_json(attempt_path, payload)
        return payload

    verified = verify_selected_images(
        SELECTED_DIR, SELECTION_MANIFEST, expected_count=SPARSE_CONFIG.expected_images
    )
    features_database = prepare_learned_feature_cache(
        SELECTED_DIR, SELECTION_MANIFEST, WORK_DIR, frontend, SPARSE_CONFIG
    )
    candidates = generate_candidate_pairs(verified.records, BRIDGE_CONFIG)
    verify_step11_candidate_identity(candidates, STEP11_CANDIDATES)
    metrics = run_learned_diagnostics(
        features_database,
        WORK_DIR / frontend.name,
        candidates,
        frontend,
        BRIDGE_CONFIG,
    )
    selected = select_bridge_pairs(metrics, BRIDGE_CONFIG)
    _write_candidates_csv(candidates_path, metrics, selected, frontend)
    gate = targeted_gate(selected, BRIDGE_CONFIG)
    payload = {
        "status": "completed",
        "frontend": frontend.name,
        "extractor_type": frontend.extractor_type.name,
        "matcher_type": frontend.matcher_type.name,
        "pycolmap_version": str(pycolmap.__version__),
        "selection_manifest_sha256": verified.manifest_sha256,
        "critical_boundaries": [list(boundary) for boundary in BRIDGE_CONFIG.boundaries],
        "window_size": BRIDGE_CONFIG.window_size,
        "minimum_sequence_gap": BRIDGE_CONFIG.minimum_sequence_gap,
        "minimum_verified_inliers": BRIDGE_CONFIG.minimum_verified_inliers,
        "minimum_inlier_ratio": BRIDGE_CONFIG.minimum_inlier_ratio,
        "candidate_count": len(metrics),
        "selected_bridge_count": len(selected),
        "boundaries": list(boundary_bridge_summary(metrics, selected, BRIDGE_CONFIG)),
        "targeted_allowed": gate.allowed,
        "targeted_gate_reason": gate.reason,
    }
    _write_json(boundary_path, payload)
    return payload


def _map_frontend(frontend: LearnedFrontendSpec) -> dict[str, object]:
    candidates_path, boundary_path, attempt_path, output_dir = _frontend_paths(frontend)
    if attempt_path.is_file():
        existing = _load_json(attempt_path, label=f"{frontend.name} attempt report")
        if existing.get("status") == "blocked":
            return existing
    boundary = _load_json(boundary_path, label=f"{frontend.name} boundary report")
    if boundary.get("status") != "completed":
        raise ValueError(f"{frontend.name} diagnostics are not completed")
    selected = _load_selected_bridges(candidates_path)
    gate = targeted_gate(selected, BRIDGE_CONFIG)
    if not gate.allowed:
        payload: dict[str, object] = {
            "status": "skipped",
            "reason": gate.reason,
            "frontend": frontend.name,
            "extractor_type": frontend.extractor_type.name,
            "matcher_type": frontend.matcher_type.name,
            "boundary_gate_allowed": False,
            "attempt": None,
            "metric_acceptance_met": None,
        }
        _write_json(attempt_path, payload)
        return payload

    verify_selected_images(
        SELECTED_DIR, SELECTION_MANIFEST, expected_count=SPARSE_CONFIG.expected_images
    )
    features_database = prepare_learned_feature_cache(
        SELECTED_DIR, SELECTION_MANIFEST, WORK_DIR, frontend, SPARSE_CONFIG
    )
    attempt = run_learned_targeted_attempt(
        SELECTED_DIR,
        features_database,
        output_dir,
        selected,
        frontend,
        SPARSE_CONFIG,
        BRIDGE_CONFIG,
    )
    accepted = learned_model_metric_accepted(attempt.best_model, BRIDGE_CONFIG)
    payload = {
        "status": "completed",
        "reason": "",
        "frontend": frontend.name,
        "extractor_type": frontend.extractor_type.name,
        "matcher_type": frontend.matcher_type.name,
        "boundary_gate_allowed": True,
        "attempt": _portable_attempt(attempt),
        "metric_acceptance_met": accepted,
    }
    _write_json(attempt_path, payload)
    return payload


def derive_final_acceptance(
    metric_acceptance_met: bool, visual_status: str
) -> tuple[str, bool]:
    if visual_status not in VISUAL_STATUSES:
        raise ValueError(f"invalid visual plausibility status: {visual_status}")
    if visual_status == "passed" and not metric_acceptance_met:
        raise ValueError("visual status cannot pass before metric acceptance")
    return visual_status, bool(metric_acceptance_met and visual_status == "passed")


def _not_run_attempt(reason: str) -> dict[str, object]:
    return {
        "status": "not_run",
        "reason": reason,
        "boundary_gate_allowed": None,
        "attempt": None,
        "metric_acceptance_met": None,
    }


def _load_attempt_or_not_run(path: Path, reason: str) -> dict[str, object]:
    if not path.is_file():
        return _not_run_attempt(reason)
    payload = _load_json(path, label="Step 12 attempt report")
    if payload.get("status") not in {"completed", "skipped", "blocked", "not_run"}:
        raise ValueError(f"invalid Step 12 attempt status: {path}")
    return payload


def loma_not_run_reason(aliked_report: dict[str, object]) -> str:
    if aliked_report.get("status") == "blocked":
        blocker = str(aliked_report.get("reason", "native learned runtime unavailable"))
        return (
            "LoMa was not run because the primary learned capability gate was blocked: "
            f"{blocker}"
        )
    return "LoMa fallback was not required"


def write_attempts_csv(
    aliked_report: dict[str, object], loma_report: dict[str, object]
) -> None:
    rows: list[dict[str, object]] = []
    for frontend, report in (("aliked", aliked_report), ("loma", loma_report)):
        row: dict[str, object] = {
            "frontend": frontend,
            "status": str(report.get("status", "not_run")),
            "reason": str(report.get("reason", "")),
            "boundary_gate_allowed": (
                ""
                if report.get("boundary_gate_allowed") is None
                else int(bool(report.get("boundary_gate_allowed")))
            ),
            "model_count": "",
            "registered_images": "",
            "registration_fraction": "",
            "sparse_points": "",
            "observations": "",
            "mean_track_length": "",
            "mean_reprojection_error": "",
            "camera_count": "",
            "camera_model": "",
            "runtime_seconds": "",
            "feature_count": "",
            "matched_pair_count": "",
            "verified_pair_count": "",
            "metric_acceptance_met": (
                ""
                if report.get("metric_acceptance_met") is None
                else int(bool(report.get("metric_acceptance_met")))
            ),
        }
        if report.get("status") == "completed":
            raw_attempt = report.get("attempt")
            if not isinstance(raw_attempt, dict):
                raise ValueError(f"completed {frontend} report has no attempt metrics")
            attempt = _attempt_from_payload(raw_attempt)
            model = attempt.best_model
            row.update(
                {
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
        rows.append(row)
    _write_csv(ATTEMPTS_CSV, rows)


def _candidate_key(row: dict[str, str]) -> tuple[int, int, int, int, str, str]:
    return (
        int(row["boundary_left"]),
        int(row["boundary_right"]),
        int(row["left_index"]),
        int(row["right_index"]),
        str(row["left_filename"]),
        str(row["right_filename"]),
    )


def _read_candidate_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def render_boundary_comparison() -> Path | None:
    learned_tables: list[tuple[str, Path]] = []
    if ALIKED_CANDIDATES_CSV.is_file():
        learned_tables.append(("ALIKED-LightGlue", ALIKED_CANDIDATES_CSV))
    if LOMA_CANDIDATES_CSV.is_file():
        learned_tables.append(("LoMa-L", LOMA_CANDIDATES_CSV))
    if not learned_tables:
        return None
    if not STEP11_CANDIDATES.is_file():
        raise ValueError("Step 11 candidate report is missing")
    baseline = _read_candidate_rows(STEP11_CANDIDATES)
    baseline_keys = tuple(_candidate_key(row) for row in baseline)
    tables: list[tuple[str, list[dict[str, str]]]] = [("Step 11 SIFT", baseline)]
    for label, path in learned_tables:
        rows = _read_candidate_rows(path)
        if tuple(_candidate_key(row) for row in rows) != baseline_keys:
            raise RuntimeError("Step 12 candidate identity/order does not match Step 11")
        tables.append((label, rows))

    figure, axes = plt.subplots(len(BRIDGE_CONFIG.boundaries), 1, figsize=(12, 12))
    for axis, (boundary_left, boundary_right) in zip(
        np.atleast_1d(axes), BRIDGE_CONFIG.boundaries, strict=True
    ):
        boundary_name = f"{boundary_left}-{boundary_right}"
        for label, rows in tables:
            values = [
                max(0, int(row["verified_inliers"]))
                for row in rows
                if row["boundary"] == boundary_name
            ]
            axis.plot(
                np.arange(1, len(values) + 1),
                np.asarray(values, dtype=float),
                linewidth=1.0,
                alpha=0.8,
                label=label,
            )
        sift_values = [
            int(row["verified_inliers"])
            for row in baseline
            if row["boundary"] == boundary_name
        ]
        axis.set_title(
            f"Boundary {boundary_name} | same frozen candidate order | "
            f"SIFT max={max(sift_values, default=0)}"
        )
        axis.set_xlabel("Candidate position within the 780-pair boundary window")
        axis.set_ylabel("Verified geometric inliers")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right", fontsize=8)
    figure.suptitle("Step 12 learned frontend vs Step 11 SIFT boundary evidence")
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    BOUNDARY_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(BOUNDARY_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return BOUNDARY_FIGURE


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


def _render_sparse_model(model: ModelMetrics, frontend: str) -> None:
    points, colors, centers = _sparse_arrays(BEST_DIR)
    if not len(points) or not len(centers):
        raise ValueError("selected Step 12 model cannot be visualized")
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
        centers[:, 0], centers[:, 1], centers[:, 2], s=14, marker="^", label="cameras"
    )
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.set_title(
        f"Step 12 {frontend} | registered={model.registered_images}/{model.total_images} | "
        f"points={model.sparse_points:,} | reproj={model.mean_reprojection_error:.3f}px"
    )
    axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    SPARSE_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(SPARSE_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _render_registration(
    records: Sequence[SelectedImageRecord], registered_names: set[str], frontend: str
) -> None:
    indices = np.asarray([record.index for record in records], dtype=int)
    registered = np.asarray(
        [int(record.filename in registered_names) for record in records], dtype=int
    )
    figure, axis = plt.subplots(figsize=(14, 4.5))
    axis.step(indices, registered, where="mid", linewidth=1.2)
    missing = indices[registered == 0]
    if len(missing):
        axis.scatter(missing, np.zeros_like(missing), marker="x", s=28)
    for left, right in BRIDGE_CONFIG.boundaries:
        axis.axvspan(left, right, alpha=0.14, color="tab:red")
        axis.text((left + right) / 2, 1.08, f"{left}-{right}", ha="center", fontsize=8)
    axis.set_xlim(1, SPARSE_CONFIG.expected_images)
    axis.set_ylim(-0.15, 1.2)
    axis.set_yticks([0, 1], ["unregistered", "registered"])
    axis.set_xlabel("Selected sequence index (1..288)")
    axis.set_title(
        f"Step 12 selected single-model registration | {frontend} | "
        f"registered={int(registered.sum())}/{SPARSE_CONFIG.expected_images}"
    )
    axis.grid(axis="x", alpha=0.15)
    figure.tight_layout()
    REGISTRATION_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(REGISTRATION_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _baseline_model_rows() -> list[dict[str, object]]:
    step10 = _load_json(STEP10_SUMMARY, label="Step 10 summary")
    step10_model = step10.get("best_model")
    step11 = _load_json(STEP11_EXHAUSTIVE, label="Step 11 exhaustive report")
    step11_attempt = step11.get("attempt")
    if not isinstance(step10_model, dict):
        raise ValueError("Step 10 summary has no best single model")
    if not isinstance(step11_attempt, dict) or not isinstance(
        step11_attempt.get("best_model"), dict
    ):
        raise ValueError("Step 11 exhaustive report has no best single model")
    return [
        {"label": "Step 10 SIFT", **step10_model},
        {"label": "Step 11 exhaustive SIFT", **step11_attempt["best_model"]},
    ]


def render_frontend_comparison(
    aliked_report: dict[str, object], loma_report: dict[str, object]
) -> Path:
    rows = _baseline_model_rows()
    for label, report in (("Step 12 ALIKED", aliked_report), ("Step 12 LoMa", loma_report)):
        if report.get("status") != "completed":
            continue
        attempt = report.get("attempt")
        if not isinstance(attempt, dict) or not isinstance(attempt.get("best_model"), dict):
            raise ValueError(f"{label} report has no best single model")
        rows.append({"label": label, **attempt["best_model"]})
    labels = [str(row["label"]) for row in rows]
    registered = [int(row["registered_images"]) for row in rows]
    points = [int(row["sparse_points"]) for row in rows]
    errors = [float(row["mean_reprojection_error"]) for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(labels, registered)
    axes[0].axhline(BRIDGE_CONFIG.minimum_registered_images, color="tab:red", linestyle="--")
    axes[0].set_ylabel("Registered images in strongest single model")
    axes[0].set_ylim(bottom=0)
    axes[1].bar(labels, points)
    axes[1].axhline(BRIDGE_CONFIG.minimum_sparse_points, color="tab:red", linestyle="--")
    axes[1].set_ylabel("Sparse points")
    axes[1].set_ylim(bottom=0)
    axes[2].bar(labels, errors)
    axes[2].set_ylabel("Mean reprojection error (px)")
    axes[2].set_ylim(bottom=0)
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=25)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Strongest single sparse model by frontend")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    FRONTEND_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FRONTEND_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return FRONTEND_FIGURE


def _assert_model_matches(reported: ModelMetrics, measured: ModelMetrics) -> None:
    if reported.registered_images != measured.registered_images:
        raise ValueError("Step 12 model registration count does not match its report")
    if reported.sparse_points != measured.sparse_points:
        raise ValueError("Step 12 model sparse-point count does not match its report")
    if reported.camera_count != measured.camera_count:
        raise ValueError("Step 12 model camera count does not match its report")
    if reported.camera_model != measured.camera_model:
        raise ValueError("Step 12 model camera model does not match its report")
    if not math.isclose(
        reported.mean_reprojection_error,
        measured.mean_reprojection_error,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("Step 12 model reprojection error does not match its report")


def _choose_attempt(attempts: Sequence[AttemptMetrics]) -> AttemptMetrics:
    if not attempts:
        raise ValueError("at least one completed Step 12 attempt is required")

    def rank(attempt: AttemptMetrics) -> tuple[bool, int, int, float]:
        model = attempt.best_model
        error = (
            model.mean_reprojection_error
            if math.isfinite(model.mean_reprojection_error)
            else float("inf")
        )
        return (
            learned_model_metric_accepted(model, BRIDGE_CONFIG),
            model.registered_images,
            model.sparse_points,
            -error,
        )

    return max(attempts, key=rank)


def _write_registered_csv(
    records: Sequence[SelectedImageRecord], registered_names: set[str], frontend: str
) -> None:
    rows = [
        {
            "selected_index": record.index,
            "filename": record.filename,
            "registered": int(record.filename in registered_names),
            "selected_frontend": frontend,
        }
        for record in records
    ]
    _write_csv(REGISTERED_CSV, rows)


def _finalize_stage(visual_status: str = "pending") -> dict[str, object]:
    verified = verify_selected_images(
        SELECTED_DIR, SELECTION_MANIFEST, expected_count=SPARSE_CONFIG.expected_images
    )
    capability = _load_json(CAPABILITY_JSON, label="Step 12 capability report")
    aliked_report = _load_attempt_or_not_run(
        ALIKED_ATTEMPT_JSON, "ALIKED mapping stage has not run"
    )
    loma_report = _load_attempt_or_not_run(
        LOMA_ATTEMPT_JSON, loma_not_run_reason(aliked_report)
    )
    write_attempts_csv(aliked_report, loma_report)
    completed: list[tuple[str, AttemptMetrics]] = []
    for frontend, report in (("aliked", aliked_report), ("loma", loma_report)):
        if report.get("status") != "completed":
            continue
        attempt_payload = report.get("attempt")
        if not isinstance(attempt_payload, dict):
            raise ValueError(f"completed {frontend} report has no attempt")
        completed.append((frontend, _attempt_from_payload(attempt_payload)))

    selected_frontend: str | None = None
    selected_attempt: AttemptMetrics | None = None
    selected_model: ModelMetrics | None = None
    registered_names: set[str] = set()
    if completed:
        selected_attempt = _choose_attempt([attempt for _, attempt in completed])
        selected_frontend = next(
            frontend for frontend, attempt in completed if attempt is selected_attempt
        )
        measured_source = summarize_reconstruction(
            selected_attempt.best_model.model_path, SPARSE_CONFIG.expected_images
        )
        _assert_model_matches(selected_attempt.best_model, measured_source)
        copy_sparse_model(measured_source.model_path, BEST_DIR)
        selected_model = summarize_reconstruction(BEST_DIR, SPARSE_CONFIG.expected_images)
        _assert_model_matches(measured_source, selected_model)
        pycolmap.Reconstruction(BEST_DIR).export_PLY(PLY_PATH)
        registered_names = set(selected_model.registered_image_names)
        _write_registered_csv(verified.records, registered_names, selected_frontend)
        _render_sparse_model(selected_model, selected_frontend)
        _render_registration(verified.records, registered_names, selected_frontend)

    render_boundary_comparison()
    render_frontend_comparison(aliked_report, loma_report)
    metric_acceptance = bool(
        selected_model is not None
        and learned_model_metric_accepted(selected_model, BRIDGE_CONFIG)
    )
    visual_status, success = derive_final_acceptance(metric_acceptance, visual_status)
    registered_records = [
        record for record in verified.records if record.filename in registered_names
    ]
    unregistered_records = [
        record for record in verified.records if record.filename not in registered_names
    ]
    best_model_payload: dict[str, object] | None = None
    if selected_model is not None:
        best_model_payload = _portable_model(selected_model)
        best_model_payload["ply_path"] = _portable_path(PLY_PATH)
    summary: dict[str, object] = {
        "pycolmap_version": str(pycolmap.__version__),
        "selection_manifest_sha256": verified.manifest_sha256,
        "input_image_count": len(verified.records),
        "learned_feature_max_image_size": 1600,
        "critical_boundaries": [list(boundary) for boundary in BRIDGE_CONFIG.boundaries],
        "candidate_count": 2340,
        "qualification_thresholds": {
            "minimum_verified_inliers": BRIDGE_CONFIG.minimum_verified_inliers,
            "minimum_inlier_ratio": BRIDGE_CONFIG.minimum_inlier_ratio,
            "max_selected_per_boundary": BRIDGE_CONFIG.max_pairs_per_boundary,
            "max_endpoint_reuse": BRIDGE_CONFIG.max_endpoint_reuse,
        },
        "capability": capability,
        "aliked_result": aliked_report,
        "loma_result": loma_report,
        "selected_frontend": selected_frontend,
        "selected_attempt": selected_attempt.name if selected_attempt else None,
        "best_model": best_model_payload,
        "registered_images": [record.filename for record in registered_records],
        "registered_indices": [record.index for record in registered_records],
        "unregistered_images": [record.filename for record in unregistered_records],
        "unregistered_indices": [record.index for record in unregistered_records],
        "metric_acceptance_met": metric_acceptance,
        "visual_plausibility_status": visual_status,
        "learned_recovery_success": success,
        "dense_reconstruction_started": False,
        "next_boundary": (
            "Step 12 sparse recovery is accepted; dense work still requires separate authorization."
            if success
            else (
                "Choose separately between accepting local-only sparse reconstruction and "
                "authorizing an experimental learned global matching/component alignment phase."
            )
        ),
    }
    _write_json(SUMMARY_JSON, summary)
    cleanup_transient_work(WORK_DIR)
    return summary


TRANSIENT_FILENAMES = (
    "features.db",
    "features_complete.json",
    "diagnostic.db",
    "diagnostic_pairs.txt",
    "targeted.db",
    "targeted_bridge_pairs.txt",
)


def cleanup_transient_work(work_dir: Path = WORK_DIR) -> tuple[Path, ...]:
    if not work_dir.exists():
        return ()
    if not work_dir.is_dir() or work_dir.is_symlink():
        raise ValueError(f"Step 12 work path is not a regular directory: {work_dir}")
    resolved_work = work_dir.resolve()
    removed: list[Path] = []
    for frontend in ("aliked", "loma"):
        directory = work_dir / frontend
        if not directory.exists():
            continue
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"Step 12 frontend work path is not a regular directory: {directory}")
        if directory.resolve().parent != resolved_work:
            raise ValueError(f"Step 12 frontend work path escaped work root: {directory}")
        resolved_directory = directory.resolve()
        for filename in TRANSIENT_FILENAMES:
            path = directory / filename
            if path.resolve().parent != resolved_directory:
                raise ValueError(f"transient cleanup target escaped frontend work directory: {path}")
            if not path.exists():
                continue
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"transient cleanup target is not a regular file: {path}")
            path.unlink()
            removed.append(path)
        if not any(directory.iterdir()):
            directory.rmdir()
    if work_dir.exists() and not any(work_dir.iterdir()):
        work_dir.rmdir()
    return tuple(removed)


def run_stage(stage: str, *, visual_status: str = "pending") -> dict[str, object]:
    if stage == "capability":
        return _capability_stage()
    if stage == "aliked-diagnose":
        return _diagnose_frontend(ALIKED_FRONTEND)
    if stage == "aliked-map":
        return _map_frontend(ALIKED_FRONTEND)
    if stage == "loma-diagnose":
        return _diagnose_frontend(LOMA_FRONTEND)
    if stage == "loma-map":
        return _map_frontend(LOMA_FRONTEND)
    if stage == "finalize":
        return _finalize_stage(visual_status)
    if stage == "all":
        _capability_stage()
        aliked_boundary = _diagnose_frontend(ALIKED_FRONTEND)
        if aliked_boundary.get("status") != "completed":
            return _finalize_stage(visual_status)
        aliked_attempt = _map_frontend(ALIKED_FRONTEND)
        if not loma_required(aliked_boundary, aliked_attempt):
            return _finalize_stage(visual_status)
        loma_boundary = _diagnose_frontend(LOMA_FRONTEND)
        if loma_boundary.get("status") == "completed":
            _map_frontend(LOMA_FRONTEND)
        return _finalize_stage(visual_status)
    raise ValueError(f"unknown Step 12 stage: {stage}")


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
