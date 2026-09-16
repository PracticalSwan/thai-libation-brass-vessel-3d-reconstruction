"""Run the one bounded V4 3072px high-ring dense recovery candidate.

This is the final dense escalation authorized by the canonical V4 plan.  It uses
only the frozen best-defensible sparse model, the verified corrected pair graph,
and the existing image-derived masks.  Only geo_g10/g11/g12 are reconstructed;
all PatchMatch sources are restricted to that same high-ring view set so every
source also receives a fresh 3072px photometric/geometric map.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_v4 import _prepare_dense_workspace, _run_dense_patch_match
from scripts.run_v4_repaired_dense import (
    MAX_SOURCES,
    _model_image_names,
    _pair_key,
    _run_colmap,
    _verified_database_support,
)
from v4_config import CAPTURE_V4_ROOT, RECONSTRUCTION_V4_ROOT, sha256_file, write_json
from v4_dense import (
    V4DenseConfig,
    build_stereo_fusion_command,
    dense_typed_file_counts,
    load_accepted_sparse_lineage,
    postfusion_evidence_gate,
    prune_dense_photometric_maps,
    write_prioritized_dense_pair_config,
)
from v4_postfusion import build_postfusion_evidence, load_ring_by_name
from v4_repair import stable_directory_sha256


HIGH_RINGS = ("geo_g10", "geo_g11", "geo_g12")
DEFAULT_SPARSE_REPORT = RECONSTRUCTION_V4_ROOT / "repair" / "sparse_v1" / "best_defensible_sparse_v2.json"
DEFAULT_GRAPH_REPORT = RECONSTRUCTION_V4_ROOT / "repair" / "sparse_v1" / "corrected_sparse_graph_v3.json"
DEFAULT_SNAPSHOT_MANIFEST = RECONSTRUCTION_V4_ROOT / "repair" / "sparse_v1" / "canonical_sqlite_snapshot_v3.json"
DEFAULT_ISOLATION = RECONSTRUCTION_V4_ROOT / "work" / "isolation_records.json"
DEFAULT_WORKSPACE_ROOT = Path(r"D:\Side Projects\CSX4213_V4_Dense_Work")
DEFAULT_TAG = "best_defensible_v1_high_ring_3072"


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _high_ring_adjacency(
    *,
    graph_payload: Mapping[str, Any],
    high_names: Sequence[str],
    support: Mapping[tuple[str, str], int],
    ring_by_name: Mapping[str, str],
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    allowed = set(high_names)
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pairs_raw = graph_payload.get("pairs")
    if not isinstance(pairs_raw, list):
        raise ValueError("corrected sparse graph has no pair list")
    for item in pairs_raw:
        if not isinstance(item, list) or len(item) != 2:
            continue
        first = Path(str(item[0]).replace("\\", "/")).name
        second = Path(str(item[1]).replace("\\", "/")).name
        if first not in allowed or second not in allowed or first == second:
            continue
        inliers = int(support.get(_pair_key(first, second), 0))
        if inliers <= 0:
            continue
        same_ring = str(ring_by_name[first]) == str(ring_by_name[second])
        records[first].append({"source": second, "verified_inliers": inliers, "same_ring": same_ring})
        records[second].append({"source": first, "verified_inliers": inliers, "same_ring": same_ring})

    adjacency: dict[str, tuple[str, ...]] = {}
    counts: dict[str, int] = {}
    for reference in high_names:
        candidates = records.get(reference, [])
        local = sorted((row for row in candidates if row["same_ring"]), key=lambda row: (-int(row["verified_inliers"]), str(row["source"])))
        cross = sorted((row for row in candidates if not row["same_ring"]), key=lambda row: (-int(row["verified_inliers"]), str(row["source"])))
        selected: list[dict[str, Any]] = []
        selected.extend(local[:4])
        selected.extend(cross[:2])
        for row in sorted(candidates, key=lambda value: (-int(value["verified_inliers"]), str(value["source"]))):
            if row not in selected:
                selected.append(row)
            if len(selected) >= MAX_SOURCES:
                break
        selected = selected[:MAX_SOURCES]
        if not selected:
            raise RuntimeError(f"high-ring recovery has no verified source for {reference}")
        adjacency[reference] = tuple(str(row["source"]) for row in selected)
        counts[reference] = len(selected)

    return adjacency, {
        "high_rings": list(HIGH_RINGS),
        "reference_count": len(high_names),
        "verified_high_ring_edge_count": sum(len(rows) for rows in records.values()) // 2,
        "min_sources_per_reference": min(counts.values()),
        "max_sources_per_reference": max(counts.values()),
        "source_selection": "verified corrected-graph inliers only; prefer up to four same-ring and two high-ring cross-ring sources; no phase prior",
    }


def _connected_smoke_names(
    high_names: Sequence[str],
    adjacency: Mapping[str, Sequence[str]],
    ring_by_name: Mapping[str, str],
    *,
    target: int = 15,
) -> list[str]:
    starts = [name for name in high_names if ring_by_name[name] == "geo_g12"]
    if not starts:
        raise RuntimeError("no geo_g12 reference is available for the high-ring CUDA smoke")
    queue: deque[str] = deque([starts[0]])
    seen: set[str] = set()
    ordered: list[str] = []
    while queue and len(ordered) < target:
        name = queue.popleft()
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
        for source in adjacency.get(name, ()):
            if source not in seen:
                queue.append(source)
    if len(ordered) < 4:
        raise RuntimeError("high-ring graph cannot provide a connected CUDA smoke subset")
    return ordered


def _restrict_adjacency(names: Sequence[str], adjacency: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    allowed = set(names)
    result: dict[str, tuple[str, ...]] = {}
    for name in names:
        sources = tuple(source for source in adjacency.get(name, ()) if source in allowed and source != name)
        if not sources:
            raise RuntimeError(f"smoke subset left {name} without an internal verified source")
        result[name] = sources
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    sparse_report = args.accepted_sparse_report.resolve()
    sparse_payload = _read_object(sparse_report)
    sparse_lineage = load_accepted_sparse_lineage(sparse_report)
    model_path = Path(str(sparse_payload.get("source_model", ""))).resolve()
    observed_model_sha = stable_directory_sha256(model_path)
    if observed_model_sha != sparse_lineage["accepted_sparse_model_sha256"]:
        raise ValueError("high-ring recovery sparse model bytes do not match the corrected best-defensible lineage")

    model_names = _model_image_names(model_path)
    ring_by_name = load_ring_by_name(args.isolation_records.resolve())
    high_names = [name for name in model_names if str(ring_by_name.get(name)) in HIGH_RINGS]
    if not high_names:
        raise RuntimeError("no high-ring references were found")
    per_ring = {ring: sum(str(ring_by_name[name]) == ring for name in high_names) for ring in HIGH_RINGS}
    if any(count <= 0 for count in per_ring.values()):
        raise RuntimeError(f"high-ring recovery is missing a required ring: {per_ring}")

    support, database_lineage = _verified_database_support(args.snapshot_manifest.resolve(), model_names)
    graph_payload = _read_object(args.graph_report.resolve())
    adjacency, source_summary = _high_ring_adjacency(
        graph_payload=graph_payload,
        high_names=high_names,
        support=support,
        ring_by_name=ring_by_name,
    )
    source_summary["per_ring_reference_count"] = per_ring
    source_summary["database_lineage"] = database_lineage

    config = V4DenseConfig(max_image_size=3072, gpu_index=0, geom_consistency=True, filter=True)
    scratch_root = args.workspace_root.resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)

    smoke_names = _connected_smoke_names(high_names, adjacency, ring_by_name)
    smoke_adjacency = _restrict_adjacency(smoke_names, adjacency)
    smoke_tag = f"{args.tag}_smoke"
    smoke_cfg = RECONSTRUCTION_V4_ROOT / "work" / f"dense_{smoke_tag}.cfg"
    if smoke_cfg.exists():
        raise FileExistsError(f"refusing to overwrite existing high-ring smoke config: {smoke_cfg}")
    write_prioritized_dense_pair_config(smoke_cfg, smoke_names, smoke_adjacency, reference_names=smoke_names)
    smoke_paths, smoke_setup, smoke_masks = _prepare_dense_workspace(
        tag=smoke_tag,
        config=config,
        image_names=smoke_names,
        sparse_model_path=model_path,
        workspace_root=scratch_root,
    )
    smoke = _run_dense_patch_match(smoke_paths, config, smoke_tag, config_path=smoke_cfg)
    smoke_counts = dense_typed_file_counts(smoke_paths["root"])
    if smoke.get("status") != "completed" or int(smoke_counts.get("geometric_depth_count", 0)) != len(smoke_names):
        raise RuntimeError("3072px high-ring CUDA smoke did not complete with one geometric map per smoke reference")

    full_cfg = RECONSTRUCTION_V4_ROOT / "work" / f"dense_{args.tag}_patch-match-v4-high-ring.cfg"
    if full_cfg.exists():
        raise FileExistsError(f"refusing to overwrite existing high-ring config: {full_cfg}")
    write_prioritized_dense_pair_config(full_cfg, high_names, adjacency, reference_names=high_names)
    full_paths, setup, mask_summary = _prepare_dense_workspace(
        tag=args.tag,
        config=config,
        image_names=high_names,
        sparse_model_path=model_path,
        workspace_root=scratch_root,
    )
    patch = _run_dense_patch_match(full_paths, config, args.tag, config_path=full_cfg)
    typed = dense_typed_file_counts(full_paths["root"])
    if patch.get("status") != "completed" or int(typed.get("geometric_depth_count", 0)) != len(high_names):
        raise RuntimeError("3072px high-ring PatchMatch did not produce the complete expected geometric depth set")
    cleanup_manifest = RECONSTRUCTION_V4_ROOT / "work" / f"dense_{args.tag}_photometric_cleanup_manifest.json"
    prune_dense_photometric_maps(full_paths["root"], keep_names=(), manifest_path=cleanup_manifest)

    fused_path = RECONSTRUCTION_V4_ROOT / "dense" / f"fused_{args.tag}.ply"
    if fused_path.exists():
        raise FileExistsError(f"refusing to overwrite existing high-ring fused cloud: {fused_path}")
    fusion_log = RECONSTRUCTION_V4_ROOT / "work" / f"dense_{args.tag}_stereo_fusion.log"
    fusion = _run_colmap(
        build_stereo_fusion_command(full_paths["root"], fused_path, mask_path=full_paths["masks"], max_image_size=3072),
        fusion_log,
    )

    high_ring_map = {name: str(ring_by_name[name]) for name in high_names}
    evidence = build_postfusion_evidence(
        fused_path,
        workspace_root=full_paths["root"],
        sparse_model_path=full_paths["root"] / "sparse",
        mask_dir=full_paths["masks"],
        image_names=high_names,
        ring_by_name=high_ring_map,
        tile_config_paths=[full_cfg],
        sparse_lineage=sparse_lineage,
        preview_dir=RECONSTRUCTION_V4_ROOT / "previews" / f"dense_{args.tag}_semantic",
        max_sources=MAX_SOURCES,
        chunk_size=None,
    )
    gate = postfusion_evidence_gate(
        evidence,
        fused_path=fused_path,
        expected_sparse_model_sha256=sparse_lineage["accepted_sparse_model_sha256"],
    )
    evidence["postfusion_evidence_gate"] = gate
    evidence_path = RECONSTRUCTION_V4_ROOT / "reports" / f"dense_{args.tag}_postfusion.json"
    write_json(evidence_path, evidence)

    anatomy = evidence.get("contamination", {}).get("anatomy", {})
    result = {
        "schema_version": 1,
        "status": "completed",
        "purpose": "bounded_3072px_high_ring_recovery",
        "high_rings": list(HIGH_RINGS),
        "reference_count": len(high_names),
        "per_ring_reference_count": per_ring,
        "max_image_size": 3072,
        "sparse_report": str(sparse_report),
        "sparse_report_sha256": sha256_file(sparse_report),
        "sparse_model": str(model_path),
        "sparse_model_sha256": observed_model_sha,
        "graph_report": str(args.graph_report.resolve()),
        "graph_report_sha256": sha256_file(args.graph_report.resolve()),
        "source_selection": source_summary,
        "smoke": {
            "reference_count": len(smoke_names),
            "names": smoke_names,
            "config": str(smoke_cfg.resolve()),
            "workspace": str(smoke_paths["root"].resolve()),
            "setup": smoke_setup,
            "mask_summary": smoke_masks,
            "patch_match": smoke,
            "typed_counts": smoke_counts,
        },
        "workspace": str(full_paths["root"].resolve()),
        "workspace_setup": setup,
        "mask_summary": mask_summary,
        "patch_match": patch,
        "typed_counts": typed,
        "config": str(full_cfg.resolve()),
        "config_sha256": sha256_file(full_cfg),
        "fusion": fusion,
        "fused_path": str(fused_path.resolve()),
        "fused_sha256": sha256_file(fused_path),
        "postfusion_report": str(evidence_path.resolve()),
        "postfusion_report_sha256": sha256_file(evidence_path),
        "postfusion_gate_passed": bool(gate.get("passed")),
        "postfusion_failure_reasons": list(gate.get("reasons", [])),
        "anatomy_status": anatomy.get("status"),
        "anatomy_failures": list(anatomy.get("failures", [])),
        "finial_shape": anatomy.get("finial_shape", {}),
        "no_major_vessel_scale_holes": anatomy.get("no_major_vessel_scale_holes"),
    }
    report_path = RECONSTRUCTION_V4_ROOT / "reports" / f"dense_{args.tag}_run.json"
    write_json(report_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-sparse-report", type=Path, default=DEFAULT_SPARSE_REPORT)
    parser.add_argument("--graph-report", type=Path, default=DEFAULT_GRAPH_REPORT)
    parser.add_argument("--snapshot-manifest", type=Path, default=DEFAULT_SNAPSHOT_MANIFEST)
    parser.add_argument("--isolation-records", type=Path, default=DEFAULT_ISOLATION)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(json.dumps({
        "status": result["status"],
        "reference_count": result["reference_count"],
        "fused_sha256": result["fused_sha256"],
        "finial_shape": result["finial_shape"],
        "anatomy_failures": result["anatomy_failures"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
