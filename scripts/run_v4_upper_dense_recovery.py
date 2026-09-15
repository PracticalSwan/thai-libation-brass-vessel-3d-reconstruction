"""Run the final bounded 3072px high-ring dense recovery for V4.

This script is deliberately narrow. It uses the frozen best-defensible v47
sparse model, the verified corrected graph, and only the highest acquisition
ring (geo_g12 by default). The target ring is reconstructed in its own fresh
COLMAP workspace so 3072px maps are never mixed with the existing 2000px
production workspace. The result is diagnostic unless it genuinely resolves a
narrow finial; failed recovery remains evidence and does not replace the
selected full dense cloud.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_v4_repaired_dense import (
    DEFAULT_GRAPH_REPORT,
    DEFAULT_SNAPSHOT_MANIFEST,
    _load_verified_graph,
    _model_image_names,
    _read_object,
    _run_colmap,
    _verified_database_support,
)
from run_v4 import _prepare_dense_workspace, _run_dense_patch_match
from v4_config import CAPTURE_V4_ROOT, RECONSTRUCTION_V4_ROOT, sha256_file, write_json
from v4_dense import (
    V4DenseConfig,
    build_stereo_fusion_command,
    dense_typed_file_counts,
    load_accepted_sparse_lineage,
    prune_dense_photometric_maps,
    write_prioritized_dense_pair_config,
)
from v4_postfusion import (
    build_postfusion_evidence,
    load_ring_by_name,
    summarize_g8_g9_negative_evidence,
)
from v4_repair import stable_directory_sha256


DEFAULT_ACCEPTED_REPORT = RECONSTRUCTION_V4_ROOT / "repair" / "sparse_v1" / "best_defensible_sparse_v2.json"
DEFAULT_ISOLATION_RECORDS = RECONSTRUCTION_V4_ROOT / "work" / "isolation_records.json"
DEFAULT_BASELINE_SELECTION = RECONSTRUCTION_V4_ROOT / "reports" / "dense_best_defensible_v1_with_min3.json"
DEFAULT_WORKSPACE_ROOT = Path(r"D:\Side Projects\CSX4213_V4_Dense_Work")
DEFAULT_RING = "geo_g12"
DEFAULT_TAG = "upper_geo_g12_3072_v1"
MAX_SOURCES = 6


def same_ring_adjacency(
    image_names: Sequence[str],
    adjacency: Mapping[str, Sequence[str]],
    ring_by_name: Mapping[str, str],
    *,
    ring: str,
) -> dict[str, tuple[str, ...]]:
    """Restrict an audited adjacency to a self-contained single-ring graph."""

    names = tuple(str(name) for name in image_names)
    allowed = set(names)
    if not names:
        raise ValueError("upper recovery ring is empty")
    result: dict[str, tuple[str, ...]] = {}
    for name in names:
        if str(ring_by_name.get(name, "")) != str(ring):
            raise ValueError(f"upper recovery image is not in {ring}: {name}")
        sources = tuple(
            str(source)
            for source in adjacency.get(name, ())
            if source in allowed and str(ring_by_name.get(source, "")) == str(ring)
        )[:MAX_SOURCES]
        if not sources:
            raise ValueError(f"upper recovery reference has no same-ring source: {name}")
        result[name] = sources
    return result


def connected_smoke_subset(
    image_names: Sequence[str],
    adjacency: Mapping[str, Sequence[str]],
    *,
    target_count: int = 8,
) -> tuple[str, ...]:
    """Choose a deterministic connected smoke set with no source-only nodes."""

    ordered = tuple(str(name) for name in image_names)
    if len(ordered) < 2:
        raise ValueError("upper recovery smoke requires at least two images")
    target_count = max(2, min(int(target_count), len(ordered)))
    order = {name: index for index, name in enumerate(ordered)}
    selected: set[str] = {ordered[0]}
    frontier = [ordered[0]]
    while frontier and len(selected) < target_count:
        current = frontier.pop(0)
        for source in adjacency.get(current, ()):
            if source not in order or source in selected:
                continue
            selected.add(source)
            frontier.append(source)
            if len(selected) >= target_count:
                break
    if len(selected) < target_count:
        for name in ordered:
            if name in selected:
                continue
            if any(source in selected for source in adjacency.get(name, ())):
                selected.add(name)
            if len(selected) >= target_count:
                break
    # Close the set so every selected reference has at least one selected source.
    changed = True
    while changed:
        changed = False
        for name in tuple(sorted(selected, key=order.__getitem__)):
            if any(source in selected for source in adjacency.get(name, ())):
                continue
            source = next((value for value in adjacency.get(name, ()) if value in order), None)
            if source is None:
                raise ValueError(f"upper recovery smoke reference has no graph source: {name}")
            if source not in selected:
                selected.add(source)
                changed = True
    subset = tuple(name for name in ordered if name in selected)
    if any(not any(source in selected for source in adjacency.get(name, ())) for name in subset):
        raise ValueError("upper recovery smoke subset is not source-closed")
    return subset


def classify_upper_recovery(
    baseline_finial: Mapping[str, Any],
    candidate_finial: Mapping[str, Any],
) -> dict[str, Any]:
    """Promote only a genuinely narrower, explicitly resolved top element."""

    def finite_ratio(payload: Mapping[str, Any]) -> float:
        value = payload.get("finial_to_lid_radius_ratio")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return float("inf")
        return float(value)

    baseline_ratio = finite_ratio(baseline_finial)
    candidate_ratio = finite_ratio(candidate_finial)
    candidate_resolved = candidate_finial.get("resolved_narrow_top_element") is True
    improved = candidate_ratio < baseline_ratio
    promote = bool(candidate_resolved and improved)
    return {
        "status": "upper_recovery_improved" if promote else "recovery_exhausted_unresolved",
        "promote_upper_recovery": promote,
        "baseline_finial_to_lid_radius_ratio": baseline_ratio,
        "candidate_finial_to_lid_radius_ratio": candidate_ratio,
        "candidate_resolved_narrow_top_element": candidate_resolved,
        "ratio_improved": improved,
    }


def _restricted_adjacency(
    names: Sequence[str],
    adjacency: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    allowed = set(names)
    restricted: dict[str, tuple[str, ...]] = {}
    for name in names:
        sources = tuple(source for source in adjacency[name] if source in allowed)
        if not sources:
            raise ValueError(f"restricted upper recovery graph isolates {name}")
        restricted[name] = sources
    return restricted


def _geometric_map_names(workspace: Path) -> set[str]:
    root = workspace / "stereo" / "depth_maps"
    marker = ".geometric.bin"
    return {path.name[: -len(marker)] for path in root.glob(f"*{marker}")}


def _baseline_finial(selection_path: Path) -> dict[str, Any]:
    payload = _read_object(selection_path)
    selected = payload.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("baseline dense selection has no selected candidate")
    evidence = selected.get("evidence_summary")
    if not isinstance(evidence, Mapping):
        raise ValueError("baseline dense selection has no evidence summary")
    finial = evidence.get("finial_shape")
    if not isinstance(finial, Mapping):
        raise ValueError("baseline dense selection has no finial shape evidence")
    return dict(finial)


def run(args: argparse.Namespace) -> dict[str, Any]:
    accepted_report = args.accepted_sparse_report.resolve()
    accepted_payload = _read_object(accepted_report)
    lineage = load_accepted_sparse_lineage(accepted_report)
    model_path = Path(str(accepted_payload.get("source_model", ""))).resolve()
    observed_model_sha = stable_directory_sha256(model_path)
    if observed_model_sha != lineage["accepted_sparse_model_sha256"]:
        raise ValueError("upper recovery sparse model bytes do not match the accepted V2 lineage")

    image_names = _model_image_names(model_path)
    ring_by_name = load_ring_by_name(args.isolation_records.resolve())
    if set(image_names) != set(ring_by_name):
        raise ValueError("accepted sparse model and isolation records do not contain the same 372 views")

    support, database_lineage = _verified_database_support(args.snapshot_manifest.resolve(), image_names)
    adjacency, graph_selection = _load_verified_graph(
        args.graph_report.resolve(), image_names, support, ring_by_name
    )
    target_names = tuple(name for name in image_names if str(ring_by_name[name]) == str(args.ring))
    ring_adjacency = same_ring_adjacency(target_names, adjacency, ring_by_name, ring=args.ring)
    baseline_finial = _baseline_finial(args.baseline_selection.resolve())

    result: dict[str, Any] = {
        "schema_version": 1,
        "stage": "bounded_3072_high_ring_dense_recovery",
        "status": "preflight_passed",
        "tag": args.tag,
        "ring": args.ring,
        "max_image_size": args.max_image_size,
        "accepted_sparse_report": str(accepted_report),
        "accepted_sparse_model": str(model_path),
        "accepted_sparse_model_sha256": observed_model_sha,
        "sparse_lineage": lineage,
        "database_lineage": database_lineage,
        "graph_selection": graph_selection,
        "target_reference_count": len(target_names),
        "target_references": list(target_names),
        "baseline_selection": str(args.baseline_selection.resolve()),
        "baseline_selection_sha256": sha256_file(args.baseline_selection.resolve()),
        "baseline_finial_shape": baseline_finial,
    }
    if args.dry_run:
        result["status"] = "dry_run_preflight_passed"
        return result

    config = V4DenseConfig(
        max_image_size=args.max_image_size,
        gpu_index=0,
        geom_consistency=True,
        filter=True,
    )
    scratch_root = args.workspace_root.resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)

    smoke_names = connected_smoke_subset(target_names, ring_adjacency, target_count=args.smoke_count)
    smoke_adjacency = _restricted_adjacency(smoke_names, ring_adjacency)
    smoke_tag = f"{args.tag}_smoke"
    smoke_config_path = RECONSTRUCTION_V4_ROOT / "work" / f"dense_{smoke_tag}_patch-match.cfg"
    write_prioritized_dense_pair_config(
        smoke_config_path,
        smoke_names,
        smoke_adjacency,
        reference_names=smoke_names,
    )
    smoke_paths, smoke_setup, smoke_masks = _prepare_dense_workspace(
        tag=smoke_tag,
        config=config,
        image_names=list(smoke_names),
        sparse_model_path=model_path,
        workspace_root=scratch_root,
    )
    smoke = _run_dense_patch_match(
        smoke_paths,
        config,
        smoke_tag,
        config_path=smoke_config_path,
    )
    result["smoke"] = {
        "image_names": list(smoke_names),
        "config_path": str(smoke_config_path.resolve()),
        "workspace_setup": smoke_setup,
        "mask_summary": smoke_masks,
        "patch_match": smoke,
    }
    if smoke.get("status") != "completed":
        result["status"] = "smoke_failed"
        raise RuntimeError("3072px high-ring CUDA smoke did not complete")
    smoke_counts = dense_typed_file_counts(smoke_paths["root"])
    if smoke_counts.get("geometric_depth_count") != len(smoke_names) or smoke_counts.get("geometric_normal_count") != len(smoke_names):
        raise RuntimeError(f"3072px high-ring smoke produced incomplete geometric maps: {smoke_counts}")

    full_config_path = RECONSTRUCTION_V4_ROOT / "work" / f"dense_{args.tag}_patch-match.cfg"
    write_prioritized_dense_pair_config(
        full_config_path,
        target_names,
        ring_adjacency,
        reference_names=target_names,
    )
    full_paths, setup, mask_summary = _prepare_dense_workspace(
        tag=args.tag,
        config=config,
        image_names=list(target_names),
        sparse_model_path=model_path,
        workspace_root=scratch_root,
    )
    patch = _run_dense_patch_match(
        full_paths,
        config,
        args.tag,
        config_path=full_config_path,
    )
    result["workspace_setup"] = setup
    result["mask_summary"] = mask_summary
    result["patch_match"] = patch
    result["config_path"] = str(full_config_path.resolve())
    if patch.get("status") != "completed":
        result["status"] = "production_patch_match_failed"
        raise RuntimeError("3072px high-ring production PatchMatch did not complete")
    counts = dense_typed_file_counts(full_paths["root"])
    if counts.get("geometric_depth_count") != len(target_names) or counts.get("geometric_normal_count") != len(target_names):
        raise RuntimeError(f"3072px high-ring candidate produced incomplete geometric maps: {counts}")
    if _geometric_map_names(full_paths["root"]) != set(target_names):
        raise RuntimeError("3072px high-ring geometric depth names do not exactly match the target ring")
    result["map_counts"] = counts

    cleanup_manifest = RECONSTRUCTION_V4_ROOT / "work" / f"dense_{args.tag}_photometric_cleanup_manifest.json"
    prune_dense_photometric_maps(full_paths["root"], keep_names=(), manifest_path=cleanup_manifest)

    fused_path = RECONSTRUCTION_V4_ROOT / "dense" / f"fused_{args.tag}.ply"
    if fused_path.exists():
        raise FileExistsError(f"upper recovery fused candidate already exists: {fused_path}")
    fusion_log = RECONSTRUCTION_V4_ROOT / "work" / f"dense_{args.tag}_stereo_fusion.log"
    fusion = _run_colmap(
        build_stereo_fusion_command(
            full_paths["root"],
            fused_path,
            mask_path=full_paths["masks"],
            max_image_size=args.max_image_size,
        ),
        fusion_log,
    )
    result["fusion"] = fusion
    result["fused_path"] = str(fused_path.resolve())
    result["fused_sha256"] = sha256_file(fused_path)

    postfusion = build_postfusion_evidence(
        fused_path,
        workspace_root=full_paths["root"],
        sparse_model_path=full_paths["root"] / "sparse",
        mask_dir=full_paths["masks"],
        image_names=target_names,
        ring_by_name=ring_by_name,
        tile_config_paths=[full_config_path],
        sparse_lineage=lineage,
        negative_evidence=summarize_g8_g9_negative_evidence(
            CAPTURE_V4_ROOT / "manifests" / "source_manifest.csv",
            args.isolation_records.resolve(),
        ),
        preview_dir=RECONSTRUCTION_V4_ROOT / "previews" / f"dense_{args.tag}_semantic",
        max_sources=MAX_SOURCES,
        chunk_size=None,
    )
    postfusion_path = RECONSTRUCTION_V4_ROOT / "reports" / f"dense_{args.tag}_postfusion.json"
    write_json(postfusion_path, postfusion)
    result["postfusion_report"] = str(postfusion_path.resolve())
    result["postfusion_report_sha256"] = sha256_file(postfusion_path)

    candidate_anatomy = postfusion.get("contamination", {}).get("anatomy", {})
    candidate_finial = candidate_anatomy.get("finial_shape", {}) if isinstance(candidate_anatomy, Mapping) else {}
    decision = classify_upper_recovery(baseline_finial, candidate_finial if isinstance(candidate_finial, Mapping) else {})
    result["candidate_finial_shape"] = dict(candidate_finial) if isinstance(candidate_finial, Mapping) else {}
    result["decision"] = decision
    result["status"] = decision["status"]

    report_path = RECONSTRUCTION_V4_ROOT / "reports" / f"dense_{args.tag}_run.json"
    write_json(report_path, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-sparse-report", type=Path, default=DEFAULT_ACCEPTED_REPORT)
    parser.add_argument("--graph-report", type=Path, default=DEFAULT_GRAPH_REPORT)
    parser.add_argument("--snapshot-manifest", type=Path, default=DEFAULT_SNAPSHOT_MANIFEST)
    parser.add_argument("--isolation-records", type=Path, default=DEFAULT_ISOLATION_RECORDS)
    parser.add_argument("--baseline-selection", type=Path, default=DEFAULT_BASELINE_SELECTION)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--ring", default=DEFAULT_RING)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--max-image-size", type=int, default=3072)
    parser.add_argument("--smoke-count", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.max_image_size != 3072:
        raise ValueError("bounded upper recovery is authorized only at max_image_size=3072")
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
