"""Prepare an isolated affected-only V4 dense rerun.

The healthy ``workspace_tiled6`` run is treated as immutable evidence.  This
script copies it to the user-authorized D: scratch volume, substitutes only
the negative-only G8/G9 masks (after the same COLMAP undistortion), and writes
new tile configs whose bounded source selection explicitly reserves local and
cross-ring support.  It does not invoke PatchMatch or fusion; the companion
runner consumes the auditable setup manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v4_config import RECONSTRUCTION_V4_ROOT, CAPTURE_V4_ROOT, sha256_file, write_json
from v4_dense import (
    V4DenseConfig,
    build_dense_pair_adjacency,
    build_prioritized_dense_pair_adjacency,
    dense_typed_file_counts,
    load_accepted_sparse_lineage,
    undistort_v4_masks,
    write_prioritized_dense_pair_config,
)
from v4_postfusion import audit_final_tile_configs, load_ring_by_name


DENSE_SCRATCH_ROOT = Path(r"D:\Side Projects\CSX4213_V4_Dense_Work")
ORIGINAL_TAG = "workspace_tiled6"
DEFAULT_RERUN_TAG = "workspace_tiled6_g8g9_priority"
CHUNK_SIZE = 24
MAX_SOURCES = 6
DEFAULT_MIN_CROSS_SOURCES = 1


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


def _best_sparse_model_path() -> Path:
    sparse_report = _read_json(RECONSTRUCTION_V4_ROOT / "reports" / "sparse_report.json")
    models = sparse_report.get("models", [])
    index = int(sparse_report.get("best_model_index", 0))
    if not isinstance(models, list) or not models or not (0 <= index < len(models)):
        raise ValueError("sparse report has no valid best model")
    model_path = Path(str(models[index].get("model_path", ""))).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"canonical raw sparse model is missing: {model_path}")
    return model_path


def _isolation_records() -> list[dict[str, Any]]:
    payload = _read_json(RECONSTRUCTION_V4_ROOT / "work" / "isolation_records.json")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("isolation records do not contain a non-empty records list")
    return [dict(value) for value in records if isinstance(value, dict)]


def _workspace_inventory(root: Path) -> dict[str, Any]:
    counts = dense_typed_file_counts(root)
    files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "path": str(root.resolve()),
        "file_count": len(files),
        "bytes": int(sum(path.stat().st_size for path in files)),
        "typed_counts": counts,
    }


def _assert_d_workspace(path: Path) -> Path:
    root = DENSE_SCRATCH_ROOT.resolve()
    candidate = path.expanduser().resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"priority rerun workspace must stay under {root}: {candidate}") from error
    if candidate == root:
        raise ValueError("priority rerun workspace cannot be the D: scratch root itself")
    return candidate


def _copy_workspace(source: Path, target: Path) -> None:
    source = source.resolve()
    target = _assert_d_workspace(target)
    if not source.is_dir():
        raise FileNotFoundError(f"healthy dense workspace is missing: {source}")
    if target.exists():
        raise FileExistsError(f"priority rerun workspace already exists; preserve it: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)


def _prepare_mask_sources(
    *,
    target_root: Path,
    sparse_model_path: Path,
    image_names: list[str],
    refinement_report: dict[str, Any],
    mask_manifest_path: Path,
) -> tuple[Path, set[str], dict[str, Any]]:
    original_root = CAPTURE_V4_ROOT / "derived" / "masks"
    source_root = target_root / "_mask_sources_fullres"
    if source_root.exists():
        raise FileExistsError(f"mask source staging directory already exists: {source_root}")
    shutil.copytree(original_root, source_root)
    records = refinement_report.get("records", [])
    if refinement_report.get("status") != "complete" or not isinstance(records, list):
        raise ValueError("G8/G9 negative refinement report is not complete")
    changed: set[str] = set()
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("negative refinement record is not an object")
        name = Path(str(item.get("relative_path", ""))).name
        refined = Path(str(item.get("refined_mask_path", "")))
        if name not in image_names or not refined.is_file():
            raise FileNotFoundError(f"negative refined mask is missing or unregistered: {name}")
        destination = source_root / name
        shutil.copy2(refined, destination)
        changed.add(name)
    if len(changed) != int(refinement_report.get("refined_count", -1)):
        raise ValueError("negative refinement records contain duplicate or incomplete names")
    if len(changed) != 127:
        raise ValueError(f"expected exactly 127 G8/G9 refined masks, found {len(changed)}")
    if any(int(item.get("added_pixels", 1)) != 0 for item in records if isinstance(item, dict)):
        raise ValueError("negative refinement report contains added foreground pixels")
    if len(tuple(source_root.glob("*"))) != len(image_names):
        raise ValueError("staged full-resolution mask set does not match registered images")
    undistort_summary = undistort_v4_masks(
        # The clone contains the already-undistorted PINHOLE model.  Mask
        # warping must use the canonical raw/distorted sparse camera so the
        # 3072x4080 source bitmap matches its camera dimensions exactly.
        sparse_model_path=sparse_model_path,
        source_mask_dir=source_root,
        output_mask_dir=target_root / "masks",
        image_names=image_names,
        max_image_size=2000,
        external_workspace_root=target_root,
        manifest_path=mask_manifest_path,
    )
    if undistort_summary.get("image_count") != len(image_names):
        raise ValueError("refined undistorted mask count does not match registered images")
    for name in image_names:
        # COLMAP StereoFusion resolves JPEG image masks as
        # ``mask_path / (image_name + '.png')``.  Require that exact output
        # name so a prepared rerun cannot silently fuse without masks.
        output = target_root / "masks" / f"{Path(name).name}.png"
        if not output.is_file() or output.stat().st_size <= 0:
            raise FileNotFoundError(f"refined undistorted mask is missing: {output}")
    return source_root, changed, undistort_summary


def _build_setup(*, target_root: Path, rerun_tag: str, min_cross_sources: int) -> dict[str, Any]:
    sparse_lineage = load_accepted_sparse_lineage()
    image_names = _image_names()
    records = _isolation_records()
    ring_by_name = load_ring_by_name(RECONSTRUCTION_V4_ROOT / "work" / "isolation_records.json")
    phase_by_name: dict[str, float] = {}
    for item in records:
        name = Path(str(item.get("relative_path") or item.get("filename") or "")).name
        if name in image_names:
            try:
                phase_by_name[name] = float(item.get("phase_01"))
            except (TypeError, ValueError):
                continue
    schedule_path = RECONSTRUCTION_V4_ROOT / "work" / "pair_schedule.json"
    schedule = _read_json(schedule_path)
    pairs = schedule.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("pair schedule has no pairs")
    prioritized = build_prioritized_dense_pair_adjacency(
        image_names,
        pairs,
        ring_by_name=ring_by_name,
        phase_by_name=phase_by_name,
        max_sources=MAX_SOURCES,
        min_cross_sources=min_cross_sources,
    )
    original_adjacency = build_dense_pair_adjacency(image_names, pairs, max_sources=MAX_SOURCES)
    changed_report = _read_json(RECONSTRUCTION_V4_ROOT / "reports" / "g8_g9_negative_refinement.json")
    _, changed_masks, mask_summary = _prepare_mask_sources(
        target_root=target_root,
        sparse_model_path=_best_sparse_model_path(),
        image_names=image_names,
        refinement_report=changed_report,
        mask_manifest_path=RECONSTRUCTION_V4_ROOT
        / "work"
        / f"dense_{rerun_tag}_masks_refined_undistorted_manifest.json",
    )
    affected_tiles: list[int] = []
    tile_records: list[dict[str, Any]] = []
    order = {name: index for index, name in enumerate(image_names)}
    for tile_index, start in enumerate(range(0, len(image_names), CHUNK_SIZE)):
        references = image_names[start : start + CHUNK_SIZE]
        dependency = set(references)
        for name in references:
            dependency.update(prioritized[name])
        affected = bool(dependency & changed_masks)
        if affected:
            affected_tiles.append(tile_index)
        tile_records.append(
            {
                "tile_index": tile_index,
                "reference_start": start,
                "references": references,
                "reference_count": len(references),
                "changed_mask_references": sorted(set(references) & changed_masks, key=order.__getitem__),
                "changed_mask_dependencies": sorted(dependency & changed_masks, key=order.__getitem__),
                "affected": affected,
            }
        )
    if not affected_tiles:
        raise ValueError("negative G8/G9 masks do not affect any dense tile")
    config_paths: dict[str, str] = {}
    for tile_index in affected_tiles:
        references = image_names[tile_index * CHUNK_SIZE : (tile_index + 1) * CHUNK_SIZE]
        config_path = RECONSTRUCTION_V4_ROOT / "work" / f"dense_{rerun_tag}_patch-match-v4-tile-{tile_index:03d}.cfg"
        written = write_prioritized_dense_pair_config(
            config_path,
            image_names,
            prioritized,
            # The full photometric pre-pass below materializes every source
            # map.  Keep each geometric tile's reference set limited to its
            # intended 24 images so the rerun cannot silently recompute an
            # unrelated chunk.
            reference_names=references,
        )
        parsed_lines = [line.strip() for line in written.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(parsed_lines) != 2 * len(references):
            raise ValueError(f"priority tile config has an unexpected intended-reference line count: {written}")
        config_paths[str(tile_index)] = str(written.resolve())
    photometric_config_path = RECONSTRUCTION_V4_ROOT / "work" / f"dense_{rerun_tag}_photometric-prep.cfg"
    write_prioritized_dense_pair_config(
        photometric_config_path,
        image_names,
        prioritized,
        reference_names=image_names,
    )
    original_configs = sorted(
        RECONSTRUCTION_V4_ROOT / "work" / f"dense_{ORIGINAL_TAG}_patch-match-v4-tile-{index:03d}.cfg"
        for index in range(len(tile_records))
    )
    if any(not path.is_file() for path in original_configs):
        raise FileNotFoundError("one or more healthy final tile configs are missing")
    original_audit = audit_final_tile_configs(
        original_configs,
        image_names=image_names,
        ring_by_name=ring_by_name,
        max_sources=MAX_SOURCES,
        sparse_lineage=sparse_lineage,
        chunk_size=CHUNK_SIZE,
    )
    graph_support = {
        "reference_count": len(prioritized),
        "references_with_local_support": sum(
            any(ring_by_name[source] == ring_by_name[name] for source in prioritized[name])
            for name in image_names
        ),
        "references_with_cross_ring_support": sum(
            any(ring_by_name[source] != ring_by_name[name] for source in prioritized[name])
            for name in image_names
        ),
        "max_sources": MAX_SOURCES,
        "selection": f"phase-nearest with one local and at least {min_cross_sources} cross-ring source(s) reserved before local-first fill",
        "min_cross_sources": min_cross_sources,
    }
    return {
        "schema_version": 1,
        "status": "prepared",
        "rerun_tag": rerun_tag,
        "original_workspace": _workspace_inventory(DENSE_SCRATCH_ROOT / ORIGINAL_TAG),
        "rerun_workspace": _workspace_inventory(target_root),
        "original_fused_path": str((RECONSTRUCTION_V4_ROOT / "dense" / "fused.ply").resolve()),
        "original_fused_sha256": sha256_file(RECONSTRUCTION_V4_ROOT / "dense" / "fused.ply"),
        "sparse_report": str((RECONSTRUCTION_V4_ROOT / "reports" / "sparse_report.json").resolve()),
        "sparse_lineage": sparse_lineage,
        "pair_schedule": str(schedule_path.resolve()),
        "pair_schedule_sha256": sha256_file(schedule_path),
        "isolation_records": str((RECONSTRUCTION_V4_ROOT / "work" / "isolation_records.json").resolve()),
        "negative_refinement_report": str((RECONSTRUCTION_V4_ROOT / "reports" / "g8_g9_negative_refinement.json").resolve()),
        "negative_refinement_report_sha256": sha256_file(RECONSTRUCTION_V4_ROOT / "reports" / "g8_g9_negative_refinement.json"),
        "staged_fullres_mask_root": str((target_root / "_mask_sources_fullres").resolve()),
        "undistorted_mask_manifest": str(
            (
                RECONSTRUCTION_V4_ROOT
                / "work"
                / f"dense_{rerun_tag}_masks_refined_undistorted_manifest.json"
            ).resolve()
        ),
        "undistorted_mask_summary": mask_summary,
        "image_count": len(image_names),
        "changed_mask_count": len(changed_masks),
        "changed_mask_names": sorted(changed_masks, key=order.__getitem__),
        "chunk_size": CHUNK_SIZE,
        "affected_tiles": affected_tiles,
        "affected_reference_count": sum(tile_records[index]["reference_count"] for index in affected_tiles),
        "tile_records": tile_records,
        "priority_graph": graph_support,
        "original_source_selection_audit": original_audit,
        "priority_tile_configs": config_paths,
        "photometric_prep_config": str(photometric_config_path.resolve()),
        "photometric_prep_reference_count": len(image_names),
        "healthy_tile_configs": {str(index): str(path.resolve()) for index, path in enumerate(original_configs)},
        "combined_config_policy": "full photometric source-map pre-pass, then priority configs replace only affected tile indices; healthy configs remain byte-preserved for unaffected tiles",
        "affected_scope_reason": "G8/G9 negative masks change fused support for their references and every bounded tile whose phase-prioritized local/cross source set depends on those views",
        "algorithm_contract": {
            "max_image_size": 2000,
            "geom_consistency": True,
            "filter": True,
            "input_type": "geometric",
            "workspace_format": "COLMAP",
            "no_algorithm_change": True,
        },
        "source_priority_contract": {
            "min_local_sources": 1,
            "min_cross_sources": min_cross_sources,
            "selection": "phase-nearest cross-ring candidates with local support retained",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=DEFAULT_RERUN_TAG)
    parser.add_argument(
        "--min-cross-sources",
        type=int,
        default=DEFAULT_MIN_CROSS_SOURCES,
        help="minimum cross-ring sources reserved per reference when available (default: 1)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=RECONSTRUCTION_V4_ROOT / "reports" / "dense_priority_rerun_setup.json",
    )
    args = parser.parse_args()
    if not args.tag or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in args.tag):
        raise ValueError("priority rerun tag contains unsupported characters")
    if args.min_cross_sources < 1 or args.min_cross_sources >= MAX_SOURCES:
        raise ValueError("--min-cross-sources must be at least 1 and leave room for local support")
    report_path = args.report.resolve()
    if report_path.exists():
        raise FileExistsError(f"priority rerun setup report already exists; preserve it: {report_path}")
    target_root = _assert_d_workspace(DENSE_SCRATCH_ROOT / args.tag)
    _copy_workspace(DENSE_SCRATCH_ROOT / ORIGINAL_TAG, target_root)
    setup = _build_setup(target_root=target_root, rerun_tag=args.tag, min_cross_sources=args.min_cross_sources)
    write_json(report_path, setup)
    print(
        json.dumps(
            {
                "status": setup["status"],
                "report": str(report_path),
                "rerun_workspace": str(target_root),
                "affected_tiles": setup["affected_tiles"],
                "changed_mask_count": setup["changed_mask_count"],
                "image_count": setup["image_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
