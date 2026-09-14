"""Run the affected-only G8/G9 V4 dense rerun and post-fusion evidence.

The script consumes ``dense_priority_rerun_setup.json`` produced by
``prepare_v4_priority_dense_rerun.py``.  It never touches the healthy
``workspace_tiled6`` maps or ``fused.ply``: PatchMatch runs in the D: clone,
first materializes a full photometric source-map set, then only affected tile
references are recomputed with the unchanged 2000px geometric contract, and
fusion writes a new candidate under ``reconstruction/v4/dense``.  Poisson and
Blender are not invoked here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_reconstruction import run_command
from v4_config import RECONSTRUCTION_V4_ROOT, sha256_file, write_json
from v4_dense import (
    V4DenseConfig,
    build_patch_match_command,
    build_stereo_fusion_command,
    dense_failure_category,
    dense_typed_file_counts,
    load_accepted_sparse_lineage,
    postfusion_evidence_gate,
)
from v4_postfusion import (
    audit_final_tile_configs,
    build_postfusion_evidence,
    load_ring_by_name,
)


CHUNK_SIZE = 24
MAX_SOURCES = 6
PATCH_TIMEOUT_SECONDS = 43_200.0
DEFAULT_SETUP = RECONSTRUCTION_V4_ROOT / "reports" / "dense_priority_rerun_setup.json"
DEFAULT_RUN_REPORT = RECONSTRUCTION_V4_ROOT / "reports" / "dense_priority_rerun.json"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _image_names() -> list[str]:
    sparse_report = _read_json(RECONSTRUCTION_V4_ROOT / "reports" / "sparse_report.json")
    models = sparse_report.get("models", [])
    index = int(sparse_report.get("best_model_index", 0))
    if not isinstance(models, list) or not models or not (0 <= index < len(models)):
        raise ValueError("sparse report has no valid best model")
    names = [Path(str(value)).name for value in models[index].get("registered_image_names", [])]
    if not names or len(names) != len(set(names)):
        raise ValueError("registered image names are empty or duplicated")
    return names


def _safe_hash(path: Path) -> str:
    return sha256_file(path) if path.is_file() else ""


def _map_paths(workspace: Path, name: str) -> tuple[Path, Path]:
    depth = workspace / "stereo" / "depth_maps" / f"{name}.geometric.bin"
    normal = workspace / "stereo" / "normal_maps" / f"{name}.geometric.bin"
    return depth, normal


def _assert_map_pair(workspace: Path, name: str) -> dict[str, Any]:
    depth, normal = _map_paths(workspace, name)
    if not depth.is_file() or depth.stat().st_size <= 0:
        raise FileNotFoundError(f"geometric depth map is missing or empty: {depth}")
    if not normal.is_file() or normal.stat().st_size <= 0:
        raise FileNotFoundError(f"geometric normal map is missing or empty: {normal}")
    return {
        "name": name,
        "depth_path": str(depth.resolve()),
        "normal_path": str(normal.resolve()),
        "depth_size_bytes": depth.stat().st_size,
        "normal_size_bytes": normal.stat().st_size,
        "depth_sha256": sha256_file(depth),
        "normal_sha256": sha256_file(normal),
    }


def _run_patch_match(
    workspace: Path,
    config_path: Path,
    tag: str,
    *,
    config: V4DenseConfig | None = None,
) -> dict[str, Any]:
    log_path = RECONSTRUCTION_V4_ROOT / "work" / f"dense_{tag}_patch_match.log"
    config = config or V4DenseConfig(
        max_image_size=2000,
        gpu_index=0,
        geom_consistency=True,
        filter=True,
        input_type="geometric",
        workspace_format="COLMAP",
    )
    started = datetime.now(timezone.utc).isoformat()
    command = build_patch_match_command(workspace, config=config, config_path=config_path)
    result = run_command(command, log_path, timeout=PATCH_TIMEOUT_SECONDS)
    result["failure_category"] = (
        dense_failure_category(result.get("output_tail", ""), status=str(result.get("status", "failed")))
        if result.get("status") != "completed"
        else None
    )
    result["started_at"] = started
    result["config_path"] = str(config_path.resolve())
    result["log_path"] = str(log_path.resolve())
    result["command_contract"] = {
        "max_image_size": config.max_image_size,
        "geom_consistency": config.geom_consistency,
        "filter": config.filter,
        "input_type": config.input_type,
        "workspace_format": config.workspace_format,
        "allow_missing_files": False,
    }
    result["typed_counts"] = dense_typed_file_counts(workspace)
    return result


def _copy_original_maps_for_rollback(workspace: Path, references: list[str]) -> dict[str, Any]:
    backup_root = workspace / "_original_geometric_backup"
    depth_backup = backup_root / "depth_maps"
    normal_backup = backup_root / "normal_maps"
    depth_backup.mkdir(parents=True, exist_ok=True)
    normal_backup.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for name in references:
        depth, normal = _map_paths(workspace, name)
        if not depth.is_file() or not normal.is_file():
            raise FileNotFoundError(f"healthy clone map pair is missing before rerun: {name}")
        depth_target = depth_backup / depth.name
        normal_target = normal_backup / normal.name
        if not depth_target.exists():
            shutil.copy2(depth, depth_target)
        if not normal_target.exists():
            shutil.copy2(normal, normal_target)
        # PatchMatch preserves an existing geometric output rather than
        # replacing it in some COLMAP builds.  Remove the cloned copy only
        # after its rollback hash is secured so the next invocation must write
        # a genuinely fresh affected map.
        if sha256_file(depth_target) != sha256_file(depth):
            raise RuntimeError(f"clone depth map changed before rerun backup: {name}")
        if sha256_file(normal_target) != sha256_file(normal):
            raise RuntimeError(f"clone normal map changed before rerun backup: {name}")
        depth.unlink()
        normal.unlink()
        records.append(
            {
                "name": name,
                "depth_backup": str(depth_target.resolve()),
                "normal_backup": str(normal_target.resolve()),
                "depth_sha256": sha256_file(depth_target),
                "normal_sha256": sha256_file(normal_target),
            }
        )
    return {
        "path": str(backup_root.resolve()),
        "reference_count": len(records),
        "records": records,
    }


def _combined_config_paths(setup: dict[str, Any]) -> list[Path]:
    affected = {int(value) for value in setup.get("affected_tiles", [])}
    priority = {int(key): Path(str(value)) for key, value in setup.get("priority_tile_configs", {}).items()}
    healthy = {int(key): Path(str(value)) for key, value in setup.get("healthy_tile_configs", {}).items()}
    count = len(setup.get("tile_records", []))
    paths: list[Path] = []
    for index in range(count):
        path = priority.get(index) if index in affected else healthy.get(index)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"combined tile config is missing for tile {index}")
        paths.append(path)
    return paths


def _write_run_report(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", type=Path, default=DEFAULT_SETUP)
    parser.add_argument("--report", type=Path, default=DEFAULT_RUN_REPORT)
    parser.add_argument("--workspace", type=Path, default=None, help="optional fresh D: clone override")
    parser.add_argument("--fused-output", type=Path, default=None, help="optional new project-local fused candidate path")
    args = parser.parse_args()
    setup_path = args.setup.resolve()
    report_path = args.report.resolve()
    setup = _read_json(setup_path)
    sparse_lineage = load_accepted_sparse_lineage()
    if setup.get("status") != "prepared":
        raise ValueError("priority rerun setup is not in prepared state")
    rerun_tag = str(setup.get("rerun_tag", "")).strip()
    if not rerun_tag or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in rerun_tag):
        raise ValueError("priority rerun setup has an invalid rerun_tag")
    contract = setup.get("algorithm_contract", {})
    if contract != {
        "max_image_size": 2000,
        "geom_consistency": True,
        "filter": True,
        "input_type": "geometric",
        "workspace_format": "COLMAP",
        "no_algorithm_change": True,
    }:
        raise ValueError(f"priority rerun algorithm contract is not the approved 2000px geometric run: {contract}")
    if report_path.exists():
        raise FileExistsError(f"priority rerun report already exists; preserve it before retrying: {report_path}")
    workspace_value = setup.get("rerun_workspace", {})
    workspace = (
        args.workspace.resolve()
        if args.workspace is not None
        else Path(str(workspace_value.get("path", ""))).resolve()
        if isinstance(workspace_value, dict)
        else Path(str(workspace_value)).resolve()
    )
    if not workspace.is_dir():
        raise FileNotFoundError(f"priority rerun workspace is missing: {workspace}")
    image_names = _image_names()
    if len(image_names) != int(setup.get("image_count", -1)):
        raise ValueError("setup image count no longer matches the sparse registered image list")
    affected_tiles = [int(value) for value in setup.get("affected_tiles", [])]
    if not affected_tiles:
        raise ValueError("setup has no affected dense tiles")
    tile_records = {int(item["tile_index"]): item for item in setup.get("tile_records", []) if isinstance(item, dict)}
    affected_references = [
        name
        for index in affected_tiles
        for name in image_names[index * CHUNK_SIZE : (index + 1) * CHUNK_SIZE]
    ]
    if len(affected_references) != int(setup.get("affected_reference_count", -1)):
        raise ValueError("affected reference count does not match setup tile records")
    combined_configs = _combined_config_paths(setup)
    ring_by_name = load_ring_by_name(RECONSTRUCTION_V4_ROOT / "work" / "isolation_records.json")
    source_audit_before = audit_final_tile_configs(
        combined_configs,
        image_names=image_names,
        ring_by_name=ring_by_name,
        max_sources=MAX_SOURCES,
        sparse_lineage=sparse_lineage,
        prior_review_estimate={
            "references_without_cross_ring_source_count": 150,
            "cross_ring_directed_source_count": 954,
        },
        chunk_size=CHUNK_SIZE,
    )
    if source_audit_before["status"] != "passed":
        raise RuntimeError(f"combined priority/healthy tile configs fail closed before PatchMatch: {source_audit_before['errors']}")
    source_priority = setup.get("source_priority_contract", {})
    min_cross_sources = int(source_priority.get("min_cross_sources", 1)) if isinstance(source_priority, dict) else 1
    weak_cross_support = [
        str(item.get("reference", ""))
        for item in source_audit_before.get("references", [])
        if isinstance(item, dict)
        and int(item.get("cross_ring_source_count", 0)) < min_cross_sources
    ]
    if weak_cross_support:
        raise RuntimeError(
            f"combined priority/healthy tile configs do not satisfy the minimum cross-ring source contract ({min_cross_sources}): {weak_cross_support[:5]}"
        )
    backup = _copy_original_maps_for_rollback(workspace, affected_references)
    photometric_config = Path(str(setup.get("photometric_prep_config", ""))).resolve()
    if not photometric_config.is_file():
        raise FileNotFoundError(f"full photometric preparation config is missing: {photometric_config}")
    photometric_config_lines = [
        line.strip() for line in photometric_config.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(photometric_config_lines) != 2 * len(image_names):
        raise ValueError("full photometric preparation config must contain every registered reference exactly once")
    photometric_config_contract = V4DenseConfig(
        max_image_size=2000,
        gpu_index=0,
        geom_consistency=False,
        filter=False,
        input_type="geometric",
        workspace_format="COLMAP",
    )
    photometric_prep = _run_patch_match(
        workspace,
        photometric_config,
        f"{rerun_tag}_photometric_prep",
        config=photometric_config_contract,
    )
    photometric_counts = photometric_prep.get("typed_counts", {})
    if photometric_prep.get("status") != "completed":
        raise RuntimeError(
            "full photometric source-map preparation failed before affected geometric rerun: "
            f"{photometric_prep.get('output_tail', '')[-3000:]}"
        )
    if (
        int(photometric_counts.get("photometric_depth_count", 0)) != len(image_names)
        or int(photometric_counts.get("photometric_normal_count", 0)) != len(image_names)
    ):
        raise RuntimeError(f"full photometric source-map preparation is incomplete: {photometric_counts}")
    run: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "setup_report": str(setup_path),
        "setup_report_sha256": sha256_file(setup_path),
        "rerun_workspace": str(workspace),
        "setup_workspace_override": bool(args.workspace is not None),
        "healthy_workspace": setup.get("original_workspace", {}).get("path", ""),
        "healthy_fused_sha256": setup.get("original_fused_sha256", ""),
        "algorithm_contract": contract,
        "source_priority_contract": source_priority,
        "affected_tiles": affected_tiles,
        "affected_reference_count": len(affected_references),
        "combined_config_paths": [str(path.resolve()) for path in combined_configs],
        "source_selection_audit_before": source_audit_before,
        "rollback_map_backup": backup,
        "photometric_prep": {
            "config_path": str(photometric_config),
            "config_sha256": sha256_file(photometric_config),
            "command_contract": photometric_prep.get("command_contract", {}),
            "result": photometric_prep,
        },
        "tiles": [],
    }
    _write_run_report(report_path, run)
    for tile_index in affected_tiles:
        tile_record = tile_records.get(tile_index)
        if not tile_record:
            raise ValueError(f"setup tile record is missing for affected tile {tile_index}")
        references = [str(value) for value in tile_record.get("references", [])]
        config_path = Path(str(setup["priority_tile_configs"][str(tile_index)])).resolve()
        result = _run_patch_match(workspace, config_path, f"{rerun_tag}_tile_{tile_index:03d}")
        map_records: list[dict[str, Any]] = []
        map_failures: list[str] = []
        for name in references:
            try:
                map_records.append(_assert_map_pair(workspace, name))
            except (OSError, ValueError) as error:
                map_failures.append(f"{name}: {type(error).__name__}: {error}")
        tile_result: dict[str, Any] = {
            "tile_index": tile_index,
            "references": references,
            "reference_count": len(references),
            "config_path": str(config_path),
            "patch_match": result,
            "map_records": map_records,
            "map_failures": map_failures,
        }
        run["tiles"].append(tile_result)
        run["last_completed_tile"] = tile_index if result.get("status") == "completed" and not map_failures else None
        if result.get("status") != "completed" or map_failures:
            run["status"] = "failed_patch_match"
            run["failure"] = {
                "tile_index": tile_index,
                "failure_category": result.get("failure_category"),
                "output_tail": result.get("output_tail", "")[-4000:],
                "map_failures": map_failures,
            }
            _write_run_report(report_path, run)
            return 1
        tile_result["photometric_cleanup"] = {
            "status": "deferred_until_candidate_fused_output_is_secured",
            "reason": "all source photometric maps remain available for subsequent affected tiles",
        }
        tile_result["typed_counts_after_cleanup"] = dense_typed_file_counts(workspace)
        _write_run_report(report_path, run)
    final_counts = dense_typed_file_counts(workspace)
    if final_counts.get("geometric_depth_count") != len(image_names) or final_counts.get("geometric_normal_count") != len(image_names):
        run["status"] = "failed_incomplete_maps"
        run["final_typed_counts"] = final_counts
        _write_run_report(report_path, run)
        return 1
    run["final_typed_counts"] = final_counts
    run["photometric_cleanup_deferred"] = True
    fused_path = (
        args.fused_output.resolve()
        if args.fused_output is not None
        else RECONSTRUCTION_V4_ROOT / "dense" / "fused_workspace_tiled6_g8g9_priority.ply"
    )
    dense_root = (RECONSTRUCTION_V4_ROOT / "dense").resolve()
    try:
        fused_path.relative_to(dense_root)
    except ValueError as error:
        raise ValueError(f"priority fused candidate must stay under {dense_root}: {fused_path}") from error
    if fused_path.name == "fused.ply":
        raise ValueError("priority rerun may not overwrite the healthy fused.ply")
    if fused_path.exists():
        raise FileExistsError(f"priority fused candidate already exists; preserve it: {fused_path}")
    fusion_log = RECONSTRUCTION_V4_ROOT / "work" / f"dense_{rerun_tag}_stereo_fusion.log"
    fusion_command = build_stereo_fusion_command(
        workspace,
        fused_path,
        mask_path=workspace / "masks",
        max_image_size=2000,
    )
    fusion_result = run_command(fusion_command, fusion_log, timeout=PATCH_TIMEOUT_SECONDS)
    fusion_result["failure_category"] = (
        dense_failure_category(fusion_result.get("output_tail", ""), status=str(fusion_result.get("status", "failed")))
        if fusion_result.get("status") != "completed"
        else None
    )
    run["stereo_fusion"] = fusion_result
    run["fused_path"] = str(fused_path.resolve())
    if fusion_result.get("status") != "completed" or not fused_path.is_file() or fused_path.stat().st_size <= 0:
        run["status"] = "failed_stereo_fusion"
        _write_run_report(report_path, run)
        return 1
    run["fused_sha256"] = sha256_file(fused_path)
    run["fused_size_bytes"] = fused_path.stat().st_size
    try:
        negative_report = _read_json(RECONSTRUCTION_V4_ROOT / "reports" / "g8_g9_negative_refinement.json")
        negative_evidence = {
            "status": negative_report.get("status"),
            "report_path": str((RECONSTRUCTION_V4_ROOT / "reports" / "g8_g9_negative_refinement.json").resolve()),
            "report_sha256": sha256_file(RECONSTRUCTION_V4_ROOT / "reports" / "g8_g9_negative_refinement.json"),
            "geometry_count": negative_report.get("geometry_count"),
            "refined_count": negative_report.get("refined_count"),
            "removed_pixels_total": negative_report.get("removed_pixels_total"),
            "negative_only": negative_report.get("negative_only"),
            "downstream_rerun": True,
        }
        postfusion = build_postfusion_evidence(
            fused_path,
            workspace_root=workspace,
            sparse_model_path=workspace / "sparse",
            mask_dir=workspace / "masks",
            image_names=image_names,
            ring_by_name=ring_by_name,
            tile_config_paths=combined_configs,
            sparse_lineage=sparse_lineage,
            prior_review_estimate={
                "references_without_cross_ring_source_count": 150,
                "cross_ring_directed_source_count": 954,
            },
            negative_evidence=negative_evidence,
            preview_dir=RECONSTRUCTION_V4_ROOT / "previews" / f"dense_{rerun_tag}_semantic",
            max_sources=MAX_SOURCES,
        )
        postfusion["postfusion_evidence_gate"] = postfusion_evidence_gate(
            postfusion,
            fused_path=fused_path,
            expected_sparse_model_sha256=sparse_lineage["accepted_sparse_model_sha256"],
        )
        evidence_path = RECONSTRUCTION_V4_ROOT / "reports" / f"dense_{rerun_tag}_gate.json"
        evidence = {
            "schema_version": 1,
            "status": postfusion.get("status", "failed"),
            "candidate": f"g8_g9_negative_mask_affected_dense_rerun:{rerun_tag}",
            "healthy_run_preserved": True,
            "healthy_fused_sha256": setup.get("original_fused_sha256", ""),
            "setup_report": str(setup_path),
            "setup_report_sha256": sha256_file(setup_path),
            "run_report": str(report_path),
            "run_report_sha256": sha256_file(report_path),
            "postfusion": postfusion,
            "postfusion_evidence_gate": postfusion["postfusion_evidence_gate"],
            "source_selection_audit": postfusion.get("source_selection_audit"),
            "ring_transition_audit": postfusion.get("ring_transition_audit"),
            "contamination": postfusion.get("contamination"),
            "semantic_previews": postfusion.get("semantic_previews"),
            "visual_gate": {
                "status": "pending",
                "required_views": ["front", "quarter", "side", "top_oblique"],
                "evidence": [item.get("path") for item in postfusion.get("semantic_previews", {}).values() if isinstance(item, dict)],
                "review_required_before_dense_acceptance": True,
            },
            "poisson_or_blender_started": False,
        }
        write_json(evidence_path, evidence)
        run["postfusion_evidence_path"] = str(evidence_path.resolve())
        run["status"] = "complete_pending_visual_gate"
        run["completed_at"] = datetime.now(timezone.utc).isoformat()
        _write_run_report(report_path, run)
        print(json.dumps({"status": run["status"], "fused": str(fused_path.resolve()), "evidence": str(evidence_path.resolve())}, indent=2))
        return 0
    except Exception as error:
        run["status"] = "failed_postfusion_audit"
        run["failure"] = {"type": type(error).__name__, "message": str(error)}
        _write_run_report(report_path, run)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
