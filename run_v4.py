"""Restartable V4 entry point for the approved reconstruction path.

The runner intentionally stops at a measured capability gate instead of
creating placeholder masks, sparse models, meshes, or Blender exports.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

from v4_config import (
    CAPTURE_V4_ROOT,
    RECONSTRUCTION_V4_ROOT,
    StageStateStore,
    V4_SOURCE_ROOT,
    ensure_v4_directories,
    record_runtime_identities,
    sha256_file,
    write_json,
)
from v4_ingest import build_authoritative_manifest
from v4_database import (
    V4PairingConfig,
    build_colmap_database,
    build_feature_payload_cache,
    build_v4_pair_schedule,
    estimate_cross_ring_offsets,
    load_v4_feature_cache,
    match_feature_pair,
    pairing_report,
    selected_phase_frames,
)
from v4_features import V4FeatureConfig, lightglue_identity, prepare_v4_feature_cache, load_frontend
from v4_sparse import (
    V4SparseConfig,
    export_sparse_ply,
    render_camera_trajectory_contact_sheet,
    render_sparse_contact_sheet,
    ring_coverage_summary,
    run_pycolmap_mapping,
    sparse_gate,
    summarize_sparse_reconstruction,
)
from v4_dense import (
    V4DenseConfig,
    build_dense_pair_adjacency,
    build_image_undistorter_command,
    build_patch_match_command,
    build_stereo_fusion_command,
    colmap_help,
    dense_depth_file_counts,
    dense_failure_category,
    dense_gate,
    dense_workspace_paths,
    dense_typed_file_counts,
    finalize_dense_visual_gate,
    load_accepted_sparse_lineage,
    postfusion_evidence_gate,
    prune_dense_photometric_maps,
    render_dense_contact_sheet,
    resource_fallback,
    undistort_v4_masks,
    write_dense_image_list,
    write_dense_pair_config,
    write_dense_smoke_config,
    write_dense_report,
)
from v4_mesh import (
    finalize_raw_visual_gate,
    poisson_command,
    raw_mesh_metrics,
    raw_visual_gate,
    write_raw_gate_report,
)
from v4_postfusion import (
    audit_final_tile_configs,
    audit_ring_transitions,
    load_ring_by_name,
    measure_fused_contamination,
    render_semantic_fused_views,
    summarize_g8_g9_negative_evidence,
)
from v4_repair import stable_directory_sha256
from v4_isolation import (
    SegmentationCapabilityError,
    official_model_identity,
    run_official_isolation,
    segmentation_capability,
)


# The user-authorized D: volume is used only for large CUDA/photogrammetry
# intermediates.  Reports, manifests, fused output, meshes, and Blender
# deliverables stay under the project-local V4 boundary.
DENSE_SCRATCH_ROOT = Path(r"D:\Side Projects\CSX4213_V4_Dense_Work")


def _output_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256_file(path) for path in paths if path.is_file()}


def run_paths(store: StageStateStore) -> dict[str, Any]:
    store.begin("paths", config={"source_root": "CSX4213_Project_V4_Images", "writable_roots": ["capture_v4", "reconstruction/v4"]})
    directories = ensure_v4_directories()
    runtime_path = record_runtime_identities()
    outputs = [runtime_path]
    record = store.complete(
        "paths",
        output_paths=outputs,
        output_hashes=_output_hashes(outputs),
        details={"directories": {key: str(value) for key, value in directories.items()}},
    )
    return record


def run_manifest(store: StageStateStore) -> dict[str, Any]:
    source = V4_SOURCE_ROOT
    store.begin("manifest", input_hashes={"source_directory": str(source.resolve())}, config={"expected_counts": {"total": 688, "appearance_reference": 158, "empty_board": 107, "geometry": 423}})
    result = build_authoritative_manifest()
    manifest_dir = CAPTURE_V4_ROOT / "manifests"
    outputs = [
        manifest_dir / "source_manifest.csv",
        manifest_dir / "sequences.json",
        manifest_dir / "media_audit.json",
        manifest_dir / "exclusions.csv",
    ]
    return store.complete(
        "manifest",
        output_paths=outputs,
        output_hashes=_output_hashes(outputs),
        details={
            "observed_count": result["audit"]["observed_count"],
            "role_totals": result["audit"]["role_totals"],
            "selected_geometry_count": result["audit"]["selected_geometry_count"],
            "wrap_detection": result["audit"]["wrap_detection"],
        },
    )


def run_isolation(store: StageStateStore) -> dict[str, Any]:
    capability = segmentation_capability()
    manifest_path = CAPTURE_V4_ROOT / "manifests" / "source_manifest.csv"
    report_path = RECONSTRUCTION_V4_ROOT / "reports" / "isolation_report.json"
    capability_report_path = RECONSTRUCTION_V4_ROOT / "reports" / "isolation_capability.json"
    store.begin(
        "isolation",
        input_hashes={"manifest": sha256_file(manifest_path)},
        config={"refresh_interval": 24, "dino_device": "cpu", "sam_device": "cuda"},
        tools={"segmentation_capability": capability},
    )
    write_json(capability_report_path, capability)
    if capability["status"] != "available":
        message = (
            "V4 isolation is blocked before mask generation: official "
            "Grounding DINO-T + SAM 2.1 Hiera-small imports are unavailable; "
            f"missing={','.join(capability['missing'])}"
        )
        store.fail("isolation", category="capability", message=message, retry_decision="install_or_enable_official_models_then_resume")
        raise SegmentationCapabilityError(message)
    model_identity = official_model_identity(
        grounding_checkpoint=RECONSTRUCTION_V4_ROOT / "work" / "models" / "groundingdino_swint_ogc.pth",
        sam2_checkpoint=RECONSTRUCTION_V4_ROOT / "work" / "models" / "sam2.1_hiera_small.pt",
    )
    store.begin(
        "isolation",
        input_hashes={"manifest": sha256_file(manifest_path)},
        config={"refresh_interval": 24, "dino_device": "cpu", "sam_device": "cuda"},
        tools={"segmentation_capability": capability, "model_identity": model_identity},
    )
    try:
        summary = run_official_isolation(
            manifest_path=manifest_path,
            report_path=report_path,
            index_path=RECONSTRUCTION_V4_ROOT / "work" / "isolation_records.json",
        )
    except Exception as error:
        store.fail(
            "isolation",
            category="runtime",
            message=f"official Grounded-SAM2 isolation failed: {type(error).__name__}: {error}",
            retry_decision="resume_after_diagnosing_official_runtime",
        )
        raise
    outputs = [
        capability_report_path,
        report_path,
        RECONSTRUCTION_V4_ROOT / "work" / "isolation_records.json",
        CAPTURE_V4_ROOT / "derived" / "masks",
        CAPTURE_V4_ROOT / "derived" / "feature_masks",
        CAPTURE_V4_ROOT / "derived" / "mvs_images",
    ]
    return store.complete(
        "isolation",
        output_paths=outputs,
        output_hashes=_output_hashes([capability_report_path, report_path, RECONSTRUCTION_V4_ROOT / "work" / "isolation_records.json"]),
        details=summary,
    )


def _read_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError(f"V4 isolation record index is empty: {path}")
    return [dict(record) for record in records]


def run_matching(store: StageStateStore) -> dict[str, Any]:
    isolation_path = RECONSTRUCTION_V4_ROOT / "work" / "isolation_records.json"
    config = V4FeatureConfig()
    input_hashes = {"isolation_records": sha256_file(isolation_path)}
    model_identity = lightglue_identity(config)
    store.begin(
        "matching",
        input_hashes=input_hashes,
        config=asdict(config),
        tools={"model_identity": model_identity},
    )
    marker_path = RECONSTRUCTION_V4_ROOT / "work" / "features_manifest.json"
    try:
        _, summary = prepare_v4_feature_cache(config=config)
    except Exception as error:
        store.fail(
            "matching",
            category="runtime",
            message=f"V4 ALIKED feature extraction failed: {type(error).__name__}: {error}",
            retry_decision="resume_after_diagnosing_feature_runtime",
        )
        raise
    outputs = [marker_path, RECONSTRUCTION_V4_ROOT / "work" / "features"]
    return store.complete(
        "matching",
        output_paths=outputs,
        output_hashes=_output_hashes([marker_path]),
        details=summary,
    )


def run_pairing(store: StageStateStore) -> dict[str, Any]:
    isolation_path = RECONSTRUCTION_V4_ROOT / "work" / "isolation_records.json"
    feature_marker_path = RECONSTRUCTION_V4_ROOT / "work" / "features_manifest.json"
    records = _read_records(isolation_path)
    pairing_config = V4PairingConfig()
    input_hashes = {
        "isolation_records": sha256_file(isolation_path),
        "feature_manifest": sha256_file(feature_marker_path),
    }
    store.begin("pairing", input_hashes=input_hashes, config=asdict(pairing_config))
    try:
        features, feature_marker = load_v4_feature_cache(feature_marker_path)
        frontend = load_frontend(V4FeatureConfig())
        frames = selected_phase_frames(records, ring_order=pairing_config.ring_order)

        def progress(done: int, total: int, edge: str, offset: float, count: int) -> None:
            if done == 1 or done % 25 == 0 or done == total:
                print(f"offset probes {done}/{total} {edge} offset={offset:.3f} matches={count}", flush=True)

        offsets, _ = estimate_cross_ring_offsets(
            frames,
            features,
            frontend,
            config=pairing_config,
            progress=progress,
        )
        pairs = build_v4_pair_schedule(frames, offsets, config=pairing_config)
        report = pairing_report(
            records=records,
            feature_marker=feature_marker,
            frames_by_ring=frames,
            offsets=offsets,
            pairs=pairs,
            config=pairing_config,
        )
        schedule_path = RECONSTRUCTION_V4_ROOT / "work" / "pair_schedule.json"
        report_path = RECONSTRUCTION_V4_ROOT / "reports" / "pairing_report.json"
        write_json(
            schedule_path,
            {
                "schema_version": 1,
                "status": "complete",
                "schedule_hash": report["schedule_hash"],
                "pairs": [list(pair) for pair in pairs],
                "ring_order": list(pairing_config.ring_order),
                "cross_ring_offsets": report["cross_ring_offsets"],
            },
        )
        write_json(report_path, report)
    except Exception as error:
        store.fail(
            "pairing",
            category="runtime",
            message=f"V4 phase-aware LightGlue pairing failed: {type(error).__name__}: {error}",
            retry_decision="resume_after_diagnosing_pairing_runtime",
        )
        raise
    return store.complete(
        "pairing",
        output_paths=[schedule_path, report_path],
        output_hashes=_output_hashes([schedule_path, report_path]),
        details=report,
    )


def run_database(store: StageStateStore) -> dict[str, Any]:
    isolation_path = RECONSTRUCTION_V4_ROOT / "work" / "isolation_records.json"
    feature_marker_path = RECONSTRUCTION_V4_ROOT / "work" / "features_manifest.json"
    schedule_path = RECONSTRUCTION_V4_ROOT / "work" / "pair_schedule.json"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    pairs = tuple((str(pair[0]), str(pair[1])) for pair in schedule.get("pairs", []))
    if not pairs:
        raise ValueError("V4 pair schedule contains no pairs")
    config = V4FeatureConfig()
    sparse_config = __import__("v4_sparse").V4SparseConfig()
    input_hashes = {
        "isolation_records": sha256_file(isolation_path),
        "feature_manifest": sha256_file(feature_marker_path),
        "pair_schedule": sha256_file(schedule_path),
    }
    store.begin("database", input_hashes=input_hashes, config={"feature": asdict(config), "sparse": asdict(sparse_config)})
    database_path = RECONSTRUCTION_V4_ROOT / "work" / "database.db"
    pairs_path = RECONSTRUCTION_V4_ROOT / "work" / "database_pairs.txt"
    report_path = RECONSTRUCTION_V4_ROOT / "reports" / "database_report.json"
    try:
        features, feature_marker = load_v4_feature_cache(feature_marker_path)
        if str(schedule.get("schedule_hash") or "") != __import__("v4_matching").schedule_hash(pairs):
            raise ValueError("V4 pair schedule hash does not match its pair list")
        frontend = load_frontend(config)
        payload_cache = build_feature_payload_cache(features, frontend)
        matches: dict[tuple[str, str], Any] = {}

        def progress(done: int, total: int, pair: tuple[str, str], count: int) -> None:
            if done == 1 or done % 25 == 0 or done == total:
                print(f"LightGlue pairs {done}/{total} matches={count} {pair[0]} {pair[1]}", flush=True)

        started = __import__("time").time()
        for index, pair in enumerate(pairs, start=1):
            first, second = pair
            matches[pair] = match_feature_pair(
                features[first],
                features[second],
                frontend,
                payload_cache=payload_cache,
            )
            progress(index, len(pairs), pair, len(matches[pair]))
        try:
            import pycolmap

            two_view = pycolmap.TwoViewGeometryOptions()
            two_view.ransac.random_seed = sparse_config.random_seed
            two_view.ransac.num_threads = 1
        except Exception:
            two_view = None
        report = build_colmap_database(
            image_dir=CAPTURE_V4_ROOT / "derived" / "mvs_images",
            database_path=database_path,
            pairs_path=pairs_path,
            features_by_name=features,
            matches_by_pair=matches,
            sparse_config=sparse_config,
            min_raw_matches_to_import=8,
            two_view_options=two_view,
        )
        report.update(
            {
                "feature_manifest_sha256": sha256_file(feature_marker_path),
                "feature_model_identity": feature_marker.get("model_identity", {}),
                "schedule_sha256": sha256_file(schedule_path),
                "lightglue_matching_runtime_seconds": round(__import__("time").time() - started, 3),
            }
        )
        write_json(report_path, report)
        del payload_cache
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception as error:
        store.fail(
            "database",
            category="runtime",
            message=f"V4 LightGlue/COLMAP database construction failed: {type(error).__name__}: {error}",
            retry_decision="resume_after_diagnosing_database_runtime",
        )
        raise
    return store.complete(
        "database",
        output_paths=[database_path, pairs_path, report_path],
        output_hashes=_output_hashes([database_path, pairs_path, report_path]),
        details=report,
    )


def run_sparse(store: StageStateStore) -> dict[str, Any]:
    database_path = RECONSTRUCTION_V4_ROOT / "work" / "database.db"
    pairs_path = RECONSTRUCTION_V4_ROOT / "work" / "database_pairs.txt"
    isolation_path = RECONSTRUCTION_V4_ROOT / "work" / "isolation_records.json"
    config = V4SparseConfig()
    input_hashes = {
        "database": sha256_file(database_path),
        "pairs": sha256_file(pairs_path),
        "isolation_records": sha256_file(isolation_path),
    }
    store.begin("sparse", input_hashes=input_hashes, config=asdict(config))
    output_dir = RECONSTRUCTION_V4_ROOT / "sparse" / "models"
    report_path = RECONSTRUCTION_V4_ROOT / "reports" / "sparse_report.json"
    try:
        records = _read_records(isolation_path)
        ring_by_name = {
            str(record["relative_path"]): str(record.get("logical_ring_id") or "unknown")
            for record in records
        }
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"V4 sparse output is non-empty; preserve it and inspect/reconcile before retrying: {output_dir}"
            )
        started = __import__("time").time()
        models = run_pycolmap_mapping(
            database_path,
            CAPTURE_V4_ROOT / "derived" / "mvs_images",
            output_dir,
            config=config,
        )
        import pycolmap

        model_reports: list[dict[str, Any]] = []
        model_plys: list[Path] = []
        for model_id, model in enumerate(models):
            model_path = output_dir / str(model_id)
            if not model_path.is_dir():
                model.write(model_path)
            metrics = summarize_sparse_reconstruction(
                model,
                model_path=model_path,
                total_images=len(records),
                ring_by_name=ring_by_name,
            )
            ply_path = RECONSTRUCTION_V4_ROOT / "previews" / f"sparse_model_{model_id}.ply"
            export_sparse_ply(model, ply_path)
            preview_path = RECONSTRUCTION_V4_ROOT / "previews" / f"sparse_model_{model_id}_contact.png"
            render_sparse_contact_sheet(model, preview_path)
            metrics["ply_path"] = str(ply_path.resolve())
            metrics["ply_sha256"] = sha256_file(ply_path)
            metrics["contact_sheet_path"] = str(preview_path.resolve())
            metrics["contact_sheet_sha256"] = sha256_file(preview_path)
            model_reports.append(metrics)
            model_plys.append(ply_path)
        if not model_reports:
            raise RuntimeError("V4 sparse mapping returned no model reports")
        best_index = max(
            range(len(model_reports)),
            key=lambda index: (
                int(model_reports[index]["registered_images"]),
                int(model_reports[index]["sparse_points"]),
                -float(model_reports[index]["mean_reprojection_error"])
                if math.isfinite(float(model_reports[index]["mean_reprojection_error"]))
                else float("-inf"),
            ),
        )
        best = model_reports[best_index]
        registered = best["registered_image_names"]
        coverage = ring_coverage_summary(records, registered)
        trajectory_preview_path = RECONSTRUCTION_V4_ROOT / "previews" / "sparse_model_0_trajectory.png"
        render_camera_trajectory_contact_sheet(
            models[best_index],
            ring_by_name,
            trajectory_preview_path,
        )
        schedule = json.loads((RECONSTRUCTION_V4_ROOT / "work" / "pair_schedule.json").read_text(encoding="utf-8"))
        cross_connections = 0
        for first, second in schedule.get("pairs", []):
            if first in registered and second in registered and ring_by_name.get(first) != ring_by_name.get(second):
                cross_connections += 1
        gate = sparse_gate(
            ring_coverage=coverage,
            cross_ring_connections=cross_connections,
            board_point_fraction=None,
            cloth_point_fraction=None,
            trajectory_status="pending_visual_review",
            visual_status="pending",
        )
        report = {
            "schema_version": 1,
            "status": "complete_pending_visual_gate",
            "database_sha256": input_hashes["database"],
            "pairs_sha256": input_hashes["pairs"],
            "pycolmap_version": str(pycolmap.__version__),
            "config": asdict(config),
            "runtime_seconds": round(__import__("time").time() - started, 3),
            "model_count": len(model_reports),
            "best_model_index": best_index,
            "models": model_reports,
            "ring_coverage": coverage,
            "cross_ring_connections": cross_connections,
            "board_point_fraction": None,
            "cloth_point_fraction": None,
            "contamination_measurement_status": "not_measurable_before_fusion",
            "background_fraction_basis": "sparse stage has no fused-cloud denominator; real multi-view contamination measurement is required after fusion",
            "gate": gate,
            "visual_gate": {
                "status": "pending",
                "required_views": ["front", "quarter", "side", "top_oblique"],
                "evidence": [model_reports[best_index]["contact_sheet_path"]],
                "trajectory_evidence": str(trajectory_preview_path.resolve()),
            },
        }
        write_json(report_path, report)
    except Exception as error:
        store.fail(
            "sparse",
            category="runtime",
            message=f"V4 pyCOLMAP sparse mapping failed: {type(error).__name__}: {error}",
            retry_decision="resume_after_diagnosing_sparse_runtime",
        )
        raise
    outputs = [report_path, output_dir, *model_plys]
    return store.complete(
        "sparse",
        output_paths=outputs,
        output_hashes=_output_hashes([report_path, *model_plys]),
        details=report,
    )


def _run_colmap_command(args: list[str], log_path: Path, *, timeout: float = 43_200.0) -> dict[str, Any]:
    """Run a bounded COLMAP child while preserving a capped log and process tree."""

    from local_reconstruction import run_command

    result = run_command(args, log_path, timeout=timeout)
    result["failure_category"] = dense_failure_category(
        result.get("output_tail", ""), status=str(result.get("status", "failed"))
    ) if result.get("status") != "completed" else None
    return result


def _sparse_report() -> tuple[Path, dict[str, Any]]:
    path = RECONSTRUCTION_V4_ROOT / "reports" / "sparse_report.json"
    if not path.is_file():
        raise FileNotFoundError(f"V4 sparse report is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("V4 sparse report must be a JSON object")
    if not bool(payload.get("gate", {}).get("passed", False)):
        raise RuntimeError("V4 dense work is blocked until the sparse visual gate is passed")
    return path, payload


def _dense_image_names(sparse_report: dict[str, Any]) -> list[str]:
    models = sparse_report.get("models", [])
    index = int(sparse_report.get("best_model_index", 0))
    if not isinstance(models, list) or not models or index >= len(models):
        raise ValueError("V4 sparse report has no valid best model")
    names = [Path(str(name)).name for name in models[index].get("registered_image_names", [])]
    if not names or len(names) != len(set(names)):
        raise ValueError("V4 sparse registered image list is empty or duplicated")
    missing = [name for name in names if not (CAPTURE_V4_ROOT / "derived" / "mvs_images" / name).is_file()]
    if missing:
        raise FileNotFoundError(f"V4 dense MVS images are missing ({len(missing)}): {missing[:3]}")
    return names


def _dense_smoke_names(names: list[str], isolation_records: list[dict[str, Any]]) -> list[str]:
    ring_by_name = {Path(str(item.get("relative_path") or item.get("filename"))).name: str(item.get("logical_ring_id") or "unknown") for item in isolation_records}
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(ring_by_name.get(name, "unknown"), []).append(name)
    selected: list[str] = []
    for ring in sorted(groups):
        values = groups[ring]
        selected.extend(values[:2])
    if len(selected) < 6:
        selected = names[: min(12, len(names))]
    return selected[:18]


def _ensure_dense_scratch_root() -> Path:
    root = DENSE_SCRATCH_ROOT
    if root.exists() and root.is_symlink():
        raise RuntimeError(f"D: dense scratch root must not be a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _fresh_dense_tag(base: str, *, workspace_root: Path | None = None) -> str:
    """Choose a non-destructive attempt tag, preserving prior failed workspaces."""

    workspace_root = workspace_root or _ensure_dense_scratch_root()
    candidate = str(base)
    index = 0
    while True:
        root = dense_workspace_paths(candidate, workspace_root=workspace_root)["root"]
        if not root.exists() or not any(root.iterdir()):
            return candidate
        index += 1
        candidate = f"{base}_retry{index}"


def run_dense_preflight(store: StageStateStore) -> dict[str, Any]:
    sparse_path, sparse_report = _sparse_report()
    accepted_sparse_lineage = load_accepted_sparse_lineage()
    config = V4DenseConfig()
    isolation_path = RECONSTRUCTION_V4_ROOT / "work" / "isolation_records.json"
    if not isolation_path.is_file():
        raise FileNotFoundError(f"V4 isolation record index is missing: {isolation_path}")
    input_hashes = {
        "sparse_report": sha256_file(sparse_path),
        "isolation_records": sha256_file(isolation_path),
    }
    store.begin("dense_preflight", input_hashes=input_hashes, config=asdict(config))
    report_path = RECONSTRUCTION_V4_ROOT / "reports" / "dense_preflight.json"
    try:
        helps = {name: colmap_help(name) for name in ("image_undistorter", "patch_match_stereo", "stereo_fusion", "poisson_mesher")}
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "device_0": str(torch.cuda.get_device_name(0)) if torch.cuda.device_count() else "",
        }
        names = _dense_image_names(sparse_report)
        output = {
            "schema_version": 1,
            "status": "complete",
            "sparse_gate": sparse_report.get("gate", {}),
            "image_count": len(names),
            "image_geometry": {"width": 3072, "height": 4080},
            "config": asdict(config),
            "cuda": cuda,
            "colmap_help": {name: text for name, text in helps.items()},
            "approved_fallback": "only diagnosed CUDA OOM permits max_image_size 2000 -> 1600",
        }
        if not cuda["available"]:
            raise RuntimeError("V4 dense CUDA preflight failed: torch.cuda.is_available() is false")
        write_dense_report(output, report_path)
    except Exception as error:
        store.fail(
            "dense_preflight",
            category="capability" if "CUDA" in str(error) or "cuda" in str(error) else "runtime",
            message=f"V4 dense preflight failed: {type(error).__name__}: {error}",
            retry_decision="diagnose_dense_capability_before_run",
        )
        raise
    return store.complete(
        "dense_preflight",
        output_paths=[report_path],
        output_hashes=_output_hashes([report_path]),
        details=output,
    )


def _prepare_dense_workspace(
    *,
    tag: str,
    config: V4DenseConfig,
    image_names: list[str],
    sparse_model_path: Path,
    workspace_root: Path | None = None,
) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    paths = dense_workspace_paths(tag, workspace_root=workspace_root)
    if paths["root"].exists() and any(paths["root"].iterdir()):
        raise FileExistsError(f"V4 dense workspace is non-empty; preserve and inspect before retrying: {paths['root']}")
    paths["root"].mkdir(parents=True, exist_ok=True)
    list_path = write_dense_image_list(image_names, RECONSTRUCTION_V4_ROOT / "work" / f"dense_{tag}_image_list.txt")
    undistort_log = RECONSTRUCTION_V4_ROOT / "work" / f"dense_{tag}_image_undistorter.log"
    undistort_result = _run_colmap_command(
        build_image_undistorter_command(
            image_path=CAPTURE_V4_ROOT / "derived" / "mvs_images",
            input_path=sparse_model_path,
            output_path=paths["root"],
            image_list_path=list_path,
            max_image_size=config.max_image_size,
        ),
        undistort_log,
    )
    if undistort_result.get("status") != "completed":
        return paths, {"status": "failed", "failure_category": undistort_result.get("failure_category"), "undistorter": undistort_result}, {}
    image_outputs = tuple(paths["images"].glob("*")) if paths["images"].is_dir() else ()
    if len(image_outputs) != len(image_names):
        raise RuntimeError(f"COLMAP undistorter produced {len(image_outputs)} images, expected {len(image_names)}")
    mask_summary = undistort_v4_masks(
        sparse_model_path=sparse_model_path,
        source_mask_dir=CAPTURE_V4_ROOT / "derived" / "masks",
        output_mask_dir=paths["masks"],
        image_names=image_names,
        max_image_size=config.max_image_size,
        external_workspace_root=paths["root"],
        manifest_path=RECONSTRUCTION_V4_ROOT / "work" / f"dense_{tag}_masks_manifest.json",
    )
    if len(tuple(paths["masks"].glob("*"))) != len(image_names):
        raise RuntimeError("V4 dense undistorted mask count does not match the image list")
    return paths, {"status": "complete", "undistorter": undistort_result}, mask_summary


def _run_dense_patch_match(
    paths: dict[str, Path],
    config: V4DenseConfig,
    tag: str,
    *,
    allow_missing_files: bool = False,
    config_path: Path | None = None,
) -> dict[str, Any]:
    patch_log = RECONSTRUCTION_V4_ROOT / "work" / f"dense_{tag}_patch_match.log"
    result = _run_colmap_command(
        build_patch_match_command(
            paths["root"],
            config=config,
            config_path=config_path,
            allow_missing_files=allow_missing_files,
        ),
        patch_log,
    )
    counts = dense_depth_file_counts(paths["root"])
    result["depth_counts"] = counts
    result["typed_counts"] = dense_typed_file_counts(paths["root"])
    if (
        allow_missing_files
        and result.get("status") != "completed"
        and counts.get("depth_map_count", 0) > 0
        and counts.get("normal_map_count", 0) > 0
    ):
        # COLMAP returns non-zero when a smoke workspace omits non-selected
        # dependencies, even though it has produced usable maps.  Preserve the
        # original result while treating this bounded, explicitly permitted
        # condition as a passed CUDA smoke.
        result["status_before_missing_file_tolerance"] = result.get("status")
        result["failure_category_before_missing_file_tolerance"] = result.get("failure_category")
        result["status"] = "completed"
        result["failure_category"] = None
        result["completed_with_missing_file_warnings"] = True
    return result


def run_dense(store: StageStateStore) -> dict[str, Any]:
    sparse_path, sparse_report = _sparse_report()
    accepted_sparse_lineage = load_accepted_sparse_lineage()
    isolation_path = RECONSTRUCTION_V4_ROOT / "work" / "isolation_records.json"
    isolation_records = _read_records(isolation_path)
    image_names = _dense_image_names(sparse_report)
    sparse_model_path = Path(str(sparse_report["models"][int(sparse_report.get("best_model_index", 0))]["model_path"]))
    if not sparse_model_path.is_dir():
        raise FileNotFoundError(f"V4 sparse model directory is missing: {sparse_model_path}")
    observed_sparse_hash = stable_directory_sha256(sparse_model_path)
    if observed_sparse_hash != accepted_sparse_lineage["accepted_sparse_model_sha256"]:
        raise RuntimeError(
            "V4 dense work is blocked: the sparse model is not the accepted repaired sparse candidate "
            f"({observed_sparse_hash} != {accepted_sparse_lineage['accepted_sparse_model_sha256']})"
        )
    config = V4DenseConfig()
    preflight_path = RECONSTRUCTION_V4_ROOT / "reports" / "dense_preflight.json"
    input_hashes = {
        "sparse_report": sha256_file(sparse_path),
        "isolation_records": sha256_file(isolation_path),
        "dense_preflight": sha256_file(preflight_path) if preflight_path.is_file() else "",
    }
    store.begin("dense", input_hashes=input_hashes, config=asdict(config))
    report_path = RECONSTRUCTION_V4_ROOT / "reports" / "dense_gate.json"
    attempts: list[dict[str, Any]] = []
    try:
        scratch_root = _ensure_dense_scratch_root()
        smoke_names = _dense_smoke_names(image_names, isolation_records)
        smoke_config = config
        smoke_tag = _fresh_dense_tag("smoke_2000", workspace_root=scratch_root)
        smoke_paths, smoke_setup, smoke_masks = _prepare_dense_workspace(
            tag=smoke_tag,
            config=smoke_config,
            image_names=smoke_names,
            sparse_model_path=sparse_model_path,
            workspace_root=scratch_root,
        )
        smoke_config_path = write_dense_smoke_config(
            RECONSTRUCTION_V4_ROOT / "work" / f"dense_{smoke_tag}_patch-match-v4-smoke.cfg",
            smoke_names,
        )
        smoke_patch = _run_dense_patch_match(
            smoke_paths,
            smoke_config,
            smoke_tag,
            allow_missing_files=True,
            config_path=smoke_config_path,
        )
        smoke_attempt = {"tag": smoke_tag, "config": asdict(smoke_config), "setup": smoke_setup, "mask_summary": smoke_masks, "patch_match_config_path": str(smoke_config_path.resolve()), "patch_match": smoke_patch}
        attempts.append(smoke_attempt)
        if smoke_patch.get("status") != "completed":
            category = str(smoke_patch.get("failure_category") or "runtime")
            if category != "cuda_oom":
                raise RuntimeError(f"V4 CUDA dense smoke failed: {smoke_patch.get('output_tail', '')[-2000:]}")
            smoke_config = resource_fallback(smoke_config, failure_category=category)
            smoke_tag = _fresh_dense_tag("smoke_1600", workspace_root=scratch_root)
            smoke_paths, smoke_setup, smoke_masks = _prepare_dense_workspace(
                tag=smoke_tag,
                config=smoke_config,
                image_names=smoke_names,
                sparse_model_path=sparse_model_path,
                workspace_root=scratch_root,
            )
            smoke_config_path = write_dense_smoke_config(
                RECONSTRUCTION_V4_ROOT / "work" / f"dense_{smoke_tag}_patch-match-v4-smoke.cfg",
                smoke_names,
            )
            smoke_patch = _run_dense_patch_match(
                smoke_paths,
                smoke_config,
                smoke_tag,
                allow_missing_files=True,
                config_path=smoke_config_path,
            )
            attempts.append({"tag": smoke_tag, "config": asdict(smoke_config), "setup": smoke_setup, "mask_summary": smoke_masks, "patch_match_config_path": str(smoke_config_path.resolve()), "patch_match": smoke_patch})
            if smoke_patch.get("status") != "completed":
                raise RuntimeError(f"V4 CUDA dense smoke fallback failed: {smoke_patch.get('output_tail', '')[-2000:]}")
        smoke_summary = {
            "status": "passed" if int(smoke_patch.get("typed_counts", {}).get("geometric_depth_count", 0)) > 0 and int(smoke_patch.get("typed_counts", {}).get("geometric_normal_count", 0)) > 0 else "failed",
            "depth_normal_count": min(int(smoke_patch.get("typed_counts", {}).get("geometric_depth_count", 0)), int(smoke_patch.get("typed_counts", {}).get("geometric_normal_count", 0))),
            "attempt_tag": attempts[-1]["tag"],
        }
        if smoke_summary["status"] != "passed":
            raise RuntimeError("V4 CUDA dense smoke completed without depth and normal maps")
        pair_schedule_path = RECONSTRUCTION_V4_ROOT / "work" / "pair_schedule.json"
        if not pair_schedule_path.is_file():
            raise FileNotFoundError(f"V4 dense pairing schedule is missing: {pair_schedule_path}")
        pair_schedule = json.loads(pair_schedule_path.read_text(encoding="utf-8"))
        if not isinstance(pair_schedule, dict) or not isinstance(pair_schedule.get("pairs"), list):
            raise ValueError("V4 dense pairing schedule must contain a pair list")
        dense_source_limit = 6
        adjacency = build_dense_pair_adjacency(
            image_names,
            pair_schedule["pairs"],
            max_sources=dense_source_limit,
        )

        def execute_full_dense_attempt(
            full_config: V4DenseConfig,
            full_tag: str,
        ) -> tuple[dict[str, Path], dict[str, Any]]:
            """Run disk-bounded photometric+geometric tiles.

            A geometric COLMAP invocation internally computes a photometric
            pass first.  Keeping the tile's reference and source graph local,
            then releasing *all* photometric maps after that tile, avoids
            materialising the complete 372-view photometric set on the
            space-constrained system while preserving the full-resolution
            geometric outputs needed by StereoFusion.
            """

            full_paths, full_setup, full_masks = _prepare_dense_workspace(
                tag=full_tag,
                config=full_config,
                image_names=image_names,
                sparse_model_path=sparse_model_path,
                workspace_root=scratch_root,
            )
            attempt: dict[str, Any] = {
                "tag": full_tag,
                "config": asdict(full_config),
                "setup": full_setup,
                "mask_summary": full_masks,
                "pair_schedule_sha256": sha256_file(pair_schedule_path),
                "phases": [],
            }
            attempts.append(attempt)
            cleanup_manifest = RECONSTRUCTION_V4_ROOT / "work" / f"dense_{full_tag}_photometric_cleanup_manifest.json"
            chunk_size = 24
            geometric_phases: list[dict[str, Any]] = []
            order = {name: index for index, name in enumerate(image_names)}
            for chunk_index, start in enumerate(range(0, len(image_names), chunk_size)):
                references = image_names[start : start + chunk_size]
                tile_set = set(references)
                for name in references:
                    tile_set.update(adjacency[name])
                tile_names = sorted(tile_set, key=order.__getitem__)
                tile_pairs = [
                    pair
                    for pair in pair_schedule["pairs"]
                    if len(pair) == 2 and str(pair[0]).replace("\\", "/") in tile_set
                    and str(pair[1]).replace("\\", "/") in tile_set
                ]
                chunk_config_path = write_dense_pair_config(
                    RECONSTRUCTION_V4_ROOT / "work" / f"dense_{full_tag}_patch-match-v4-tile-{chunk_index:03d}.cfg",
                    tile_names,
                    tile_pairs,
                    reference_names=references,
                    max_sources=dense_source_limit,
                )
                geom_patch = _run_dense_patch_match(
                    full_paths,
                    full_config,
                    f"{full_tag}_tile_{chunk_index:03d}",
                    config_path=chunk_config_path,
                )
                typed_counts = dense_typed_file_counts(full_paths["root"])
                phase: dict[str, Any] = {
                    "type": "geometric_tile",
                    "chunk_index": chunk_index,
                    "reference_start": start,
                    "reference_count": len(references),
                    "references": list(references),
                    "tile_image_count": len(tile_names),
                    "tile_names": tile_names,
                    "config_path": str(chunk_config_path.resolve()),
                    "patch_match": geom_patch,
                    "typed_counts_before_cleanup": typed_counts,
                }
                geometric_phases.append(phase)
                attempt["patch_match"] = geom_patch
                if (
                    geom_patch.get("status") != "completed"
                    or typed_counts.get("geometric_depth_count", 0) < start + len(references)
                    or typed_counts.get("geometric_normal_count", 0) < start + len(references)
                ):
                    raise RuntimeError(
                        f"V4 geometric PatchMatch chunk {chunk_index} failed: "
                        f"{geom_patch.get('output_tail', '')[-3000:]}"
                    )
                phase["photometric_cleanup"] = prune_dense_photometric_maps(
                    full_paths["root"],
                    # The next tile recomputes its own local photometric
                    # dependencies.  No photo map is needed between tiles.
                    keep_names=(),
                    manifest_path=cleanup_manifest,
                )
                phase["typed_counts_after_cleanup"] = dense_typed_file_counts(full_paths["root"])

            final_counts = dense_typed_file_counts(full_paths["root"])
            if (
                final_counts.get("geometric_depth_count") != len(image_names)
                or final_counts.get("geometric_normal_count") != len(image_names)
            ):
                raise RuntimeError(f"V4 geometric output count is incomplete: {final_counts}")
            attempt["strategy"] = "explicit_acquisition_graph_tiled_photometric_geometric"
            attempt["source_limit_per_reference"] = dense_source_limit
            attempt["chunk_size"] = chunk_size
            attempt["geometric_phase_count"] = len(geometric_phases)
            attempt["typed_counts_final"] = final_counts
            attempt["photometric_cleanup_manifest"] = str(cleanup_manifest.resolve())
            return full_paths, attempt

        full_config = smoke_config
        full_tag = _fresh_dense_tag(
            "workspace_tiled6" if full_config.max_image_size == 2000 else "workspace_tiled6_1600",
            workspace_root=scratch_root,
        )
        try:
            full_paths, full_attempt = execute_full_dense_attempt(full_config, full_tag)
        except RuntimeError:
            failed_attempt = attempts[-1] if attempts else {}
            phase_results = failed_attempt.get("phases", []) if isinstance(failed_attempt, dict) else []
            categories = [
                str(phase.get("patch_match", {}).get("failure_category"))
                for phase in phase_results
                if phase.get("patch_match", {}).get("failure_category")
            ]
            category = categories[-1] if categories else "runtime"
            if category != "cuda_oom" or full_config.max_image_size != 2000:
                raise
            full_config = resource_fallback(full_config, failure_category=category)
            full_tag = _fresh_dense_tag("workspace_tiled6_1600", workspace_root=scratch_root)
            full_paths, full_attempt = execute_full_dense_attempt(full_config, full_tag)
        fused_path = RECONSTRUCTION_V4_ROOT / "dense" / ("fused.ply" if full_config.max_image_size == 2000 else "fused_1600.ply")
        if fused_path.exists():
            raise FileExistsError(f"V4 dense fused output already exists; preserve it before retrying: {fused_path}")
        fusion_log = RECONSTRUCTION_V4_ROOT / "work" / f"dense_{full_tag}_stereo_fusion.log"
        fusion_result = _run_colmap_command(
            build_stereo_fusion_command(
                full_paths["root"],
                fused_path,
                mask_path=full_paths["masks"],
                max_image_size=full_config.max_image_size,
            ),
            fusion_log,
        )
        attempts[-1]["stereo_fusion"] = fusion_result
        if fusion_result.get("status") != "completed":
            raise RuntimeError(f"V4 geometric stereo fusion failed: {fusion_result.get('output_tail', '')[-3000:]}")
        from local_reconstruction_io import ply_metrics

        metrics = ply_metrics(fused_path)
        metrics["source_sha256"] = sha256_file(fused_path)
        dense_model_path = full_paths["root"] / "sparse"
        if not dense_model_path.is_dir() and full_paths["sparse"].is_dir():
            dense_model_path = full_paths["sparse"]
        if not dense_model_path.is_dir():
            raise FileNotFoundError(f"undistorted dense sparse model is missing: {dense_model_path}")
        ring_by_name = load_ring_by_name(isolation_path)
        final_tile_configs = sorted(
            RECONSTRUCTION_V4_ROOT / "work" / f"dense_{full_tag}_patch-match-v4-tile-{index:03d}.cfg"
            for index in range(int(full_attempt.get("geometric_phase_count", 0)))
        )
        tile_audit = audit_final_tile_configs(
            final_tile_configs,
            image_names=image_names,
            ring_by_name=ring_by_name,
            max_sources=dense_source_limit,
            sparse_lineage=accepted_sparse_lineage,
            prior_review_estimate={
                "references_without_cross_ring_source_count": 150,
                "cross_ring_directed_source_count": 954,
            },
            chunk_size=24,
        )
        contamination = measure_fused_contamination(
            fused_path,
            workspace_root=full_paths["root"],
            sparse_model_path=dense_model_path,
            mask_dir=full_paths["masks"],
            image_names=image_names,
            ring_by_name=ring_by_name,
        )
        ring_audit = audit_ring_transitions(
            tile_audit=tile_audit,
            sparse_model_path=dense_model_path,
            workspace_root=full_paths["root"],
            mask_dir=full_paths["masks"],
            ring_by_name=ring_by_name,
        )
        negative_evidence = summarize_g8_g9_negative_evidence(
            CAPTURE_V4_ROOT / "manifests" / "source_manifest.csv",
            isolation_path,
        )
        metrics["contamination_metrics"] = contamination["metrics"]
        metrics["contamination_findings"] = contamination["findings"]
        metrics["board_point_fraction"] = contamination["metrics"]["board_point_fraction"]["value"]
        metrics["cloth_point_fraction"] = contamination["metrics"]["cloth_or_background_point_fraction"]["value"]
        metrics["contamination_measurement_status"] = contamination["status"]
        metrics["contamination_measurement_method_version"] = contamination["method_version"]
        metrics["anatomy_evidence"] = contamination["anatomy"]
        preview_path = RECONSTRUCTION_V4_ROOT / "previews" / "dense_fused_contact.png"
        render_dense_contact_sheet(fused_path, preview_path)
        semantic_preview_paths = render_semantic_fused_views(
            fused_path,
            RECONSTRUCTION_V4_ROOT / "previews" / "dense_fused_semantic",
            basis_vectors=contamination["basis_vectors"],
        )
        contamination_report_path = RECONSTRUCTION_V4_ROOT / "reports" / "dense_contamination_gate.json"
        contamination_report = {
            "schema_version": 1,
            "status": "passed"
            if contamination["status"] == "measured"
            and tile_audit["status"] == "passed"
            and ring_audit["status"] == "passed"
            else "failed",
            "fused_path": str(fused_path.resolve()),
            "fused_sha256": sha256_file(fused_path),
            "workspace_root": str(full_paths["root"].resolve()),
            "contamination": contamination,
            "source_selection_audit": tile_audit,
            "sparse_lineage": accepted_sparse_lineage,
            "ring_transition_audit": ring_audit,
            "semantic_previews": {key: str(path.resolve()) for key, path in semantic_preview_paths.items()},
            "g8_g9_negative_evidence": negative_evidence,
        }
        contamination_report["postfusion_evidence_gate"] = postfusion_evidence_gate(
            contamination_report,
            fused_path=fused_path,
            expected_sparse_model_sha256=accepted_sparse_lineage["accepted_sparse_model_sha256"],
        )
        write_json(contamination_report_path, contamination_report)
        report = {
            "schema_version": 1,
            "status": "complete_pending_visual_gate",
            "dense_scratch_root": str(scratch_root),
            "dense_scratch_policy": "user-authorized D: intermediates; project-local reports/fused output",
            "sparse_report_sha256": sha256_file(sparse_path),
            "sparse_model_path": str(sparse_model_path.resolve()),
            "image_count": len(image_names),
            "config": asdict(full_config),
            "attempts": attempts,
            "smoke": smoke_summary,
            "fused_path": str(fused_path.resolve()),
            "fused_sha256": sha256_file(fused_path),
            "metrics": metrics,
            "contamination_gate_report": str(contamination_report_path.resolve()),
            "source_selection_audit": tile_audit,
            "ring_transition_audit": ring_audit,
            "semantic_previews": {key: str(path.resolve()) for key, path in semantic_preview_paths.items()},
            "postfusion_evidence_gate": contamination_report["postfusion_evidence_gate"],
            "visual_gate": {"status": "pending", "evidence": [str(preview_path.resolve()), *[str(path.resolve()) for path in semantic_preview_paths.values()]]},
            "gate": dense_gate(metrics, smoke=smoke_summary, visual_status="pending"),
            "board_point_fraction_basis": contamination["metrics"]["board_point_fraction"]["measurement_method"],
            "quality_risk": {
                "max_sources": dense_source_limit,
                "bounded_graph_deviation": "explicit manifest-order six-source cap instead of original automatic/default source selection",
                "review_claim_is_not_accepted_without_final_tile_audit": True,
            },
        }
        write_dense_report(report, report_path)
    except Exception as error:
        category = "cuda_oom" if any(str(item.get("patch_match", {}).get("failure_category")) == "cuda_oom" for item in attempts) else "runtime"
        store.fail(
            "dense",
            category=category,
            message=f"V4 dense reconstruction failed: {type(error).__name__}: {error}",
            retry_decision="inspect_preserved_workspace_and_logs_before_retry",
        )
        raise
    outputs = [report_path, Path(report["fused_path"]), preview_path]
    return store.complete(
        "dense",
        output_paths=outputs,
        output_hashes=_output_hashes(outputs),
        details=report,
    )


def run_dense_visual(store: StageStateStore) -> dict[str, Any]:
    accepted_sparse_lineage = load_accepted_sparse_lineage()
    report_path = RECONSTRUCTION_V4_ROOT / "reports" / "dense_gate.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"V4 dense report is missing: {report_path}")
    dense_report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(dense_report, dict):
        raise ValueError("V4 dense report must be a JSON object")
    contamination_path = Path(str(dense_report.get("contamination_gate_report", "")))
    if not contamination_path.is_file():
        raise FileNotFoundError(
            "V4 post-fusion contamination evidence is missing; dense acceptance requires the measured report"
        )
    contamination_report = json.loads(contamination_path.read_text(encoding="utf-8"))
    if not isinstance(contamination_report, dict):
        raise ValueError("V4 post-fusion contamination report must be a JSON object")
    postfusion_gate = postfusion_evidence_gate(
        contamination_report,
        fused_path=Path(str(dense_report.get("fused_path", ""))),
        expected_sparse_model_sha256=accepted_sparse_lineage["accepted_sparse_model_sha256"],
    )
    if not postfusion_gate["passed"]:
        raise RuntimeError(
            "V4 dense acceptance is blocked by incomplete post-fusion evidence: "
            + "; ".join(postfusion_gate["reasons"])
        )
    if str(contamination_report.get("fused_sha256", "")).strip().lower() != str(
        dense_report.get("fused_sha256", "")
    ).strip().lower():
        raise RuntimeError("V4 dense acceptance is blocked because post-fusion evidence does not match the fused cloud")
    if contamination_report.get("status") != "passed":
        raise RuntimeError("V4 dense acceptance is blocked until measured fused-cloud contamination evidence passes")
    if contamination_report.get("source_selection_audit", {}).get("status") != "passed":
        raise RuntimeError("V4 dense acceptance is blocked until the final tile source-selection audit passes")
    if contamination_report.get("ring_transition_audit", {}).get("status") != "passed":
        raise RuntimeError("V4 dense acceptance is blocked until fused ring-transition continuity passes")
    contamination = contamination_report.get("contamination", {})
    if contamination.get("status") != "measured":
        raise RuntimeError("V4 dense acceptance requires measured fused-cloud contamination metrics")
    semantic_previews = contamination_report.get("semantic_previews")
    if not isinstance(semantic_previews, dict) or set(semantic_previews) != {"front", "quarter", "side", "top_oblique"}:
        raise RuntimeError("V4 dense acceptance requires four semantic fused-cloud preview records")
    for key, item in semantic_previews.items():
        if not isinstance(item, dict):
            raise RuntimeError(f"V4 semantic preview evidence is missing for {key}")
        preview_path = Path(str(item.get("path", "")))
        expected_hash = str(item.get("sha256", "")).strip().lower()
        if not preview_path.is_file() or len(expected_hash) != 64 or sha256_file(preview_path) != expected_hash:
            raise RuntimeError(f"V4 semantic preview evidence is missing or hash-mismatched for {key}")
    review_path = RECONSTRUCTION_V4_ROOT / "work" / "dense_visual_review.json"
    if not review_path.is_file():
        raise FileNotFoundError(
            f"V4 dense visual review evidence is missing; inspect the real fused cloud and write {review_path}"
        )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if not isinstance(review, dict):
        raise ValueError("V4 dense visual review evidence must be a JSON object")
    report = finalize_dense_visual_gate(
        report_path,
        visual_status="passed",
        review_basis=[str(value) for value in review.get("review_basis", [])],
        review=review,
        postfusion_report=contamination_report,
    )
    store_record = store.load().get("stages", {}).get("dense", {})
    outputs = [Path(value) for value in store_record.get("output_paths", [])]
    return store.complete("dense", output_paths=outputs, output_hashes=_output_hashes(outputs), details=report)


def run_mesh(store: StageStateStore) -> dict[str, Any]:
    accepted_sparse_lineage = load_accepted_sparse_lineage()
    dense_path = RECONSTRUCTION_V4_ROOT / "reports" / "dense_gate.json"
    if not dense_path.is_file():
        raise FileNotFoundError(f"V4 dense report is missing: {dense_path}")
    dense_report = json.loads(dense_path.read_text(encoding="utf-8"))
    if not bool(dense_report.get("gate", {}).get("passed", False)):
        raise RuntimeError("V4 Poisson meshing is blocked until the dense visual gate is passed")
    contamination_path = RECONSTRUCTION_V4_ROOT / "reports" / "dense_contamination_gate.json"
    if not contamination_path.is_file():
        raise FileNotFoundError(f"V4 post-fusion contamination report is missing: {contamination_path}")
    contamination_report = json.loads(contamination_path.read_text(encoding="utf-8"))
    postfusion_gate = postfusion_evidence_gate(
        contamination_report,
        fused_path=Path(str(dense_report.get("fused_path", ""))),
        expected_sparse_model_sha256=accepted_sparse_lineage["accepted_sparse_model_sha256"],
    )
    if not postfusion_gate["passed"]:
        raise RuntimeError(
            "V4 Poisson meshing is blocked by incomplete post-fusion evidence: "
            + "; ".join(postfusion_gate["reasons"])
        )
    if contamination_report.get("status") != "passed":
        raise RuntimeError("V4 Poisson meshing is blocked until measured fused-cloud contamination evidence passes")
    if str(contamination_report.get("fused_sha256", "")).strip().lower() != str(
        dense_report.get("fused_sha256", "")
    ).strip().lower():
        raise RuntimeError("V4 Poisson meshing is blocked because post-fusion evidence does not match the fused cloud")
    if contamination_report.get("source_selection_audit", {}).get("status") != "passed":
        raise RuntimeError("V4 Poisson meshing is blocked until the final tile source-selection audit passes")
    if contamination_report.get("ring_transition_audit", {}).get("status") != "passed":
        raise RuntimeError("V4 Poisson meshing is blocked until fused ring-transition continuity passes")
    contamination = contamination_report.get("contamination", {})
    expected_clean_findings = {
        "board_slab_detected": False,
        "pedestal_board_webbing_detected": False,
        "cloth_or_background_structure_detected": False,
        "vessel_identity_confirmed": True,
    }
    if contamination.get("status") != "measured" or contamination.get("findings") != expected_clean_findings:
        raise RuntimeError("V4 Poisson meshing is blocked by contamination findings or missing measurements")
    fused_path = Path(str(dense_report.get("fused_path", "")))
    if not fused_path.is_file():
        raise FileNotFoundError(f"V4 fused cloud is missing: {fused_path}")
    config = {"algorithm": "Poisson", "depth": 13, "preserve_raw": True}
    input_hashes = {"dense_report": sha256_file(dense_path), "fused": sha256_file(fused_path)}
    store.begin("mesh", input_hashes=input_hashes, config=config)
    output_path = RECONSTRUCTION_V4_ROOT / "mesh" / "poisson_raw.ply"
    report_path = RECONSTRUCTION_V4_ROOT / "reports" / "raw_visual_gate.json"
    if output_path.exists():
        raise FileExistsError(f"V4 raw Poisson mesh already exists; preserve it before retrying: {output_path}")
    try:
        result = _run_colmap_command(
            poisson_command(fused_path, output_path, depth=int(config["depth"])),
            RECONSTRUCTION_V4_ROOT / "work" / "poisson_mesher.log",
        )
        if result.get("status") != "completed":
            raise RuntimeError(f"V4 Poisson mesher failed: {result.get('output_tail', '')[-3000:]}")
        metrics = raw_mesh_metrics(output_path)
        gate = raw_visual_gate(
            fused_metrics=dense_report.get("metrics", {}),
            mesh_metrics=metrics,
            visual_status="pending",
            required_regions={
                "bowl_and_interior": False,
                "globe_or_shoulder": False,
                "neck_lid_or_finial": False,
                "pedestal_and_base": False,
            },
            board_point_fraction=float(dense_report.get("metrics", {}).get("board_point_fraction")),
            cloth_point_fraction=float(dense_report.get("metrics", {}).get("cloth_point_fraction")),
        )
        report = {
            "schema_version": 1,
            "status": "complete_pending_visual_gate",
            "algorithm": "Poisson",
            "depth": int(config["depth"]),
            "fused_path": str(fused_path.resolve()),
            "fused_sha256": sha256_file(fused_path),
            "raw_mesh_path": str(output_path.resolve()),
            "raw_mesh_sha256": sha256_file(output_path),
            "fused_metrics": dense_report.get("metrics", {}),
            "mesh_metrics": metrics,
            "gate": gate,
            "visual_gate": {
                "status": "pending",
                "required_regions": list(gate["required_regions"]),
                "inspection_route": "Blender MCP import of the real raw mesh before any cleanup",
            },
        }
        write_raw_gate_report(report, report_path)
    except Exception as error:
        store.fail(
            "mesh",
            category="runtime",
            message=f"V4 Poisson meshing failed: {type(error).__name__}: {error}",
            retry_decision="preserve_raw_output_and_inspect_before_retry",
        )
        raise
    outputs = [report_path, output_path]
    return store.complete("mesh", output_paths=outputs, output_hashes=_output_hashes(outputs), details=report)


def run_mesh_visual(store: StageStateStore) -> dict[str, Any]:
    report_path = RECONSTRUCTION_V4_ROOT / "reports" / "raw_visual_gate.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"V4 raw mesh report is missing: {report_path}")
    review_path = RECONSTRUCTION_V4_ROOT / "work" / "raw_mesh_visual_review.json"
    if not review_path.is_file():
        raise FileNotFoundError(
            f"V4 raw-mesh visual review evidence is missing; inspect the real Poisson mesh in Blender MCP and write {review_path}"
        )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if not isinstance(review, dict):
        raise ValueError("V4 raw-mesh visual review evidence must be a JSON object")
    report = finalize_raw_visual_gate(
        report_path,
        review=review,
        review_basis=[str(value) for value in review.get("review_basis", [])],
    )
    store_record = store.load().get("stages", {}).get("mesh", {})
    outputs = [Path(value) for value in store_record.get("output_paths", [])]
    return store.complete("mesh", output_paths=outputs, output_hashes=_output_hashes(outputs), details=report)


def run(stage: str) -> dict[str, Any]:
    store = StageStateStore()
    if stage == "paths":
        return run_paths(store)
    if stage == "manifest":
        if not (CAPTURE_V4_ROOT / "manifests").is_dir():
            run_paths(store)
        return run_manifest(store)
    if stage == "isolation":
        if not (CAPTURE_V4_ROOT / "manifests" / "source_manifest.csv").is_file():
            run_manifest(store)
        return run_isolation(store)
    if stage == "matching":
        if not (RECONSTRUCTION_V4_ROOT / "work" / "isolation_records.json").is_file():
            run_isolation(store)
        return run_matching(store)
    if stage == "pairing":
        if not (RECONSTRUCTION_V4_ROOT / "work" / "features_manifest.json").is_file():
            run_matching(store)
        return run_pairing(store)
    if stage == "database":
        if not (RECONSTRUCTION_V4_ROOT / "work" / "pair_schedule.json").is_file():
            run_pairing(store)
        return run_database(store)
    if stage == "sparse":
        if not (RECONSTRUCTION_V4_ROOT / "work" / "database.db").is_file():
            run_database(store)
        return run_sparse(store)
    if stage == "dense_preflight":
        if not (RECONSTRUCTION_V4_ROOT / "reports" / "sparse_report.json").is_file():
            raise FileNotFoundError("V4 sparse report is required before dense preflight")
        return run_dense_preflight(store)
    if stage == "dense":
        if not (RECONSTRUCTION_V4_ROOT / "reports" / "dense_preflight.json").is_file():
            run_dense_preflight(store)
        return run_dense(store)
    if stage == "dense_visual":
        return run_dense_visual(store)
    if stage == "mesh":
        if not (RECONSTRUCTION_V4_ROOT / "reports" / "dense_gate.json").is_file():
            raise FileNotFoundError("V4 dense report is required before Poisson meshing")
        return run_mesh(store)
    if stage == "mesh_visual":
        return run_mesh_visual(store)
    if stage == "all":
        run_paths(store)
        run_manifest(store)
        return run_isolation(store)
    raise ValueError(f"unsupported V4 stage: {stage}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run restartable CSX4213 V4 stages")
    parser.add_argument("--stage", choices=("paths", "manifest", "isolation", "matching", "pairing", "database", "sparse", "dense_preflight", "dense", "dense_visual", "mesh", "mesh_visual", "all"), default="all")
    args = parser.parse_args()
    try:
        result = run(args.stage)
    except Exception as error:
        print(f"V4 stage {args.stage} blocked/failed: {type(error).__name__}: {error}")
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
