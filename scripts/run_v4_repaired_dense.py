"""Run a fresh V4 dense candidate from the accepted repaired sparse model.

The runner is deliberately versioned and fail-closed.  It never consumes a
historical dense workspace, opens the frozen SQLite snapshot directly, or
starts PatchMatch before the complete geometric tile set has passed the
one-reference/one-write and accepted-sparse-lineage audit.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v4_config import CAPTURE_V4_ROOT, RECONSTRUCTION_V4_ROOT, sha256_file, write_json
from v4_dense import (
    V4DenseConfig,
    build_patch_match_command,
    build_stereo_fusion_command,
    classify_dense_postfusion_completion,
    dense_typed_file_counts,
    load_accepted_sparse_lineage,
    prune_dense_photometric_maps,
)
from v4_postfusion import (
    audit_final_tile_configs,
    build_postfusion_evidence,
    load_ring_by_name,
    summarize_g8_g9_negative_evidence,
)
from v4_repair import (
    create_disposable_sqlite_from_manifest,
    stable_directory_sha256,
)


DEFAULT_ACCEPTED_REPORT = RECONSTRUCTION_V4_ROOT / "repair" / "sparse_v1" / "accepted_sparse_v4_rotation_consensus_v1.json"
DEFAULT_GRAPH_REPORT = RECONSTRUCTION_V4_ROOT / "repair" / "sparse_v1" / "corrected_sparse_graph_v3.json"
DEFAULT_SNAPSHOT_MANIFEST = RECONSTRUCTION_V4_ROOT / "repair" / "sparse_v1" / "canonical_sqlite_snapshot_v3.json"
DEFAULT_TAG = "repair_rotation_consensus_v1"
DEFAULT_WORKSPACE_ROOT = Path(r"D:\Side Projects\CSX4213_V4_Dense_Work")
MAX_SOURCES = 6


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _pair_key(first: str, second: str) -> tuple[str, str]:
    values = (Path(str(first).replace("\\", "/")).name, Path(str(second).replace("\\", "/")).name)
    if values[0] == values[1]:
        raise ValueError(f"self-pair is not a dense source edge: {values[0]}")
    return tuple(sorted(values))


def _model_image_names(model_path: Path) -> list[str]:
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(model_path))
    names = [Path(reconstruction.image(int(image_id)).name).name for image_id in sorted(reconstruction.reg_image_ids())]
    if len(names) != 372 or len(names) != len(set(names)):
        raise ValueError(f"accepted sparse model must contain exactly 372 unique registered views, found {len(names)}")
    return names


def _verified_database_support(
    snapshot_manifest: Path,
    image_names: Sequence[str],
) -> tuple[dict[tuple[str, str], int], dict[str, Any]]:
    """Read pair support only from a disposable working SQLite copy."""

    import pycolmap

    expected = set(image_names)
    with tempfile.TemporaryDirectory(prefix="v4_repaired_dense_db_") as temporary:
        working_path = Path(temporary) / "working.db"
        disposable = create_disposable_sqlite_from_manifest(snapshot_manifest, working_path)
        database = pycolmap.Database.open(str(working_path))
        try:
            image_by_id = {int(image.image_id): Path(image.name).name for image in database.read_all_images()}
            if set(image_by_id.values()) != expected:
                raise ValueError("disposable v3 database image set does not match the accepted sparse model")
            pair_ids, geometries = database.read_two_view_geometries()
            support: dict[tuple[str, str], int] = {}
            for pair_id, geometry in zip(pair_ids, geometries):
                first_id, second_id = pycolmap.pair_id_to_image_pair(int(pair_id))
                first, second = image_by_id.get(int(first_id)), image_by_id.get(int(second_id))
                if first is None or second is None:
                    continue
                matches = np.asarray(getattr(geometry, "inlier_matches", np.empty((0, 2))), dtype=np.uint32)
                support[_pair_key(first, second)] = int(matches.shape[0]) if matches.ndim == 2 else 0
            result = {
                "manifest_path": str(snapshot_manifest.resolve()),
                "working_copy_sha256": disposable["working"]["sha256"],
                "image_count": len(image_by_id),
                "two_view_geometry_count": len(pair_ids),
                "supported_pair_count": len([value for value in support.values() if value > 0]),
            }
            return support, result
        finally:
            database.close()


def _load_verified_graph(
    graph_path: Path,
    image_names: Sequence[str],
    support: Mapping[tuple[str, str], int],
    ring_by_name: Mapping[str, str],
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    payload = _read_object(graph_path)
    pairs_raw = payload.get("pairs")
    if not isinstance(pairs_raw, list) or not pairs_raw:
        raise ValueError("corrected dense graph has no pair list")
    pairs = [[Path(str(item[0]).replace("\\", "/")).name, Path(str(item[1]).replace("\\", "/")).name] for item in pairs_raw if isinstance(item, list) and len(item) == 2]
    if len(pairs) != len(pairs_raw):
        raise ValueError("corrected dense graph contains malformed pairs")
    graph_sha = hashlib.sha256(json.dumps(pairs, sort_keys=True).encode("utf-8")).hexdigest()
    if graph_sha != str(payload.get("graph_sha256", "")).lower():
        raise ValueError("corrected dense graph pair digest does not match its evidence")
    allowed = set(image_names)
    adjacency: dict[str, set[str]] = {name: set() for name in image_names}
    pair_records: list[dict[str, Any]] = []
    for first, second in pairs:
        if first not in allowed or second not in allowed or first == second:
            raise ValueError(f"corrected dense graph contains an unknown/self image pair: {first}, {second}")
        key = _pair_key(first, second)
        inliers = int(support.get(key, 0))
        if inliers <= 0:
            raise ValueError(f"corrected graph edge has no verified inlier support in disposable database: {key}")
        adjacency[first].add(second)
        adjacency[second].add(first)
        pair_records.append(
            {
                "first": first,
                "second": second,
                "verified_inliers": inliers,
                "same_ring": str(ring_by_name[first]) == str(ring_by_name[second]),
            }
        )
    result: dict[str, tuple[str, ...]] = {}
    source_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in pair_records:
        source_records[record["first"]].append({"source": record["second"], "verified_inliers": record["verified_inliers"], "same_ring": record["same_ring"]})
        source_records[record["second"]].append({"source": record["first"], "verified_inliers": record["verified_inliers"], "same_ring": record["same_ring"]})
    for reference in image_names:
        candidates = source_records.get(reference, [])
        if not candidates:
            raise ValueError(f"corrected dense graph has an isolated registered view: {reference}")
        local = sorted((item for item in candidates if item["same_ring"]), key=lambda item: (-item["verified_inliers"], item["source"]))
        cross = sorted((item for item in candidates if not item["same_ring"]), key=lambda item: (-item["verified_inliers"], item["source"]))
        selected: list[dict[str, Any]] = []
        if local:
            selected.append(local[0])
        if cross:
            selected.append(cross[0])
        for item in sorted(candidates, key=lambda value: (-value["verified_inliers"], value["source"])):
            if item not in selected:
                selected.append(item)
        selected = selected[:MAX_SOURCES]
        result[reference] = tuple(str(item["source"]) for item in selected)
    selection = {
        "schema_version": 1,
        "method": "corrected verified graph plus disposable v3 two-view inlier support; no phase or turntable pose prior",
        "graph_path": str(graph_path.resolve()),
        "graph_file_sha256": sha256_file(graph_path),
        "graph_pair_sha256": graph_sha,
        "graph_pair_count": len(pairs),
        "reference_count": len(result),
        "max_sources": MAX_SOURCES,
        "cross_ring_reference_count": sum(any(str(ring_by_name[source]) != str(ring_by_name[name]) for source in sources) for name, sources in result.items()),
        "references_without_cross_ring_source": sorted(name for name, sources in result.items() if not any(str(ring_by_name[source]) != str(ring_by_name[name]) for source in sources)),
        "verified_edge_count": len(pair_records),
    }
    return result, selection


def _write_tile_configs(
    *,
    tag: str,
    image_names: Sequence[str],
    adjacency: Mapping[str, Sequence[str]],
    lineage: Mapping[str, Any],
    ring_by_name: Mapping[str, str],
) -> tuple[list[Path], dict[str, Any]]:
    from v4_dense import write_prioritized_dense_pair_config

    # A single complete geometric config is intentional.  COLMAP's geometric
    # pass first materializes the photometric maps for every registered view,
    # then consumes those maps for geometric consistency.  Tiling references
    # while allowing sources outside the tile creates source-only dependency
    # writes or missing photometric maps; both are forbidden by the repaired
    # provenance contract.
    path = RECONSTRUCTION_V4_ROOT / "work" / f"dense_{tag}_patch-match-v4-full.cfg"
    if path.exists():
        raise FileExistsError(f"dense full-reference config already exists; refusing to overwrite: {path}")
    write_prioritized_dense_pair_config(path, image_names, adjacency, reference_names=image_names)
    paths = [path]
    audit = audit_final_tile_configs(
        paths,
        image_names=image_names,
        ring_by_name=ring_by_name,
        max_sources=MAX_SOURCES,
        sparse_lineage=lineage,
        chunk_size=None,
    )
    return paths, audit


def _geometric_map_names(workspace: Path, suffix: str) -> set[str]:
    root = Path(workspace) / "stereo" / "depth_maps"
    marker = f".{suffix}.bin"
    return {path.name[: -len(marker)] for path in root.glob(f"*{marker}")}


def _run_colmap(args: list[str], log_path: Path) -> dict[str, Any]:
    from local_reconstruction import run_command

    result = run_command(args, log_path, timeout=43_200.0)
    if result.get("status") != "completed":
        raise RuntimeError(f"COLMAP command failed: {result.get('output_tail', '')[-3000:]}")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    accepted_report = args.accepted_sparse_report.resolve()
    accepted_payload = _read_object(accepted_report)
    lineage = load_accepted_sparse_lineage(accepted_report)
    model_path = Path(str(accepted_payload.get("source_model", ""))).resolve()
    observed_model_sha = stable_directory_sha256(model_path)
    if observed_model_sha != lineage["accepted_sparse_model_sha256"]:
        raise ValueError("accepted sparse model bytes do not match the accepted sparse report")
    image_names = _model_image_names(model_path)
    ring_by_name = load_ring_by_name(args.isolation_records.resolve())
    if set(image_names) != set(ring_by_name):
        raise ValueError("accepted sparse model and isolation records do not contain the same 372 views")
    support, database_lineage = _verified_database_support(args.snapshot_manifest.resolve(), image_names)
    adjacency, selection = _load_verified_graph(args.graph_report.resolve(), image_names, support, ring_by_name)
    selection["database_lineage"] = database_lineage
    selection["accepted_sparse_model_sha256"] = lineage["accepted_sparse_model_sha256"]
    selection["accepted_sparse_gate_sha256"] = lineage["accepted_sparse_gate_sha256"]
    tile_paths, tile_audit = _write_tile_configs(
        tag=args.tag,
        image_names=image_names,
        adjacency=adjacency,
        lineage=lineage,
        ring_by_name=ring_by_name,
    )
    if tile_audit["status"] != "passed":
        raise RuntimeError("fresh repaired dense tile provenance audit failed: " + "; ".join(tile_audit["errors"]))
    audit_path = RECONSTRUCTION_V4_ROOT / "reports" / f"dense_{args.tag}_tile_provenance.json"
    write_json(audit_path, {"selection": selection, "audit": tile_audit, "sparse_lineage": lineage})
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "tile_provenance_passed",
        "tag": args.tag,
        "accepted_sparse_report": str(accepted_report),
        "accepted_sparse_model": str(model_path),
        "accepted_sparse_model_sha256": observed_model_sha,
        "sparse_lineage": lineage,
        "selection_report": selection,
        "tile_audit_report": str(audit_path.resolve()),
        "tile_audit": tile_audit,
        "tile_configs": [str(path.resolve()) for path in tile_paths],
        "patch_match_started": False,
    }
    if args.dry_run:
        result["status"] = "dry_run_dense_provenance_passed"
        return result

    from run_v4 import _prepare_dense_workspace, _run_dense_patch_match

    config = V4DenseConfig(max_image_size=2000, gpu_index=0, geom_consistency=True, filter=True)
    scratch_root = args.workspace_root.resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)
    smoke_names: list[str] = []
    by_ring: dict[str, list[str]] = defaultdict(list)
    for name in image_names:
        by_ring[str(ring_by_name[name])].append(name)
    for ring in sorted(by_ring):
        smoke_names.extend(by_ring[ring][:2])
    smoke_names = smoke_names[:18]
    # A two-per-ring sample is not guaranteed to contain an edge between the
    # selected frames.  Close any isolated smoke reference with its strongest
    # audited graph neighbor; this changes only the diagnostic smoke set, not
    # the 372-reference production tile contract.
    smoke_names = list(dict.fromkeys(smoke_names))
    while True:
        smoke_set = set(smoke_names)
        additions = [
            sorted(adjacency[name], key=lambda value: (value not in smoke_set, value))[0]
            for name in smoke_names
            if not any(source in smoke_set for source in adjacency[name]) and adjacency[name]
        ]
        additions = [name for name in dict.fromkeys(additions) if name not in smoke_set]
        if not additions or len(smoke_names) + len(additions) > 24:
            break
        smoke_names.extend(additions)
    smoke_set = set(smoke_names)
    smoke_pairs = [
        [left, right]
        for left in smoke_names
        for right in adjacency[left]
        if right in smoke_set and left < right
    ]
    if len(smoke_pairs) < 3:
        raise RuntimeError("fresh dense CUDA smoke graph has too few verified local edges")
    smoke_tag = f"{args.tag}_smoke"
    smoke_config_path = RECONSTRUCTION_V4_ROOT / "work" / f"dense_{smoke_tag}.cfg"
    from v4_dense import write_dense_pair_config

    if smoke_config_path.exists():
        raise FileExistsError(f"smoke config already exists; refusing to overwrite: {smoke_config_path}")
    write_dense_pair_config(smoke_config_path, smoke_names, smoke_pairs, reference_names=smoke_names, max_sources=MAX_SOURCES)
    smoke_paths, _, _ = _prepare_dense_workspace(
        tag=smoke_tag,
        config=config,
        image_names=smoke_names,
        sparse_model_path=model_path,
        workspace_root=scratch_root,
    )
    smoke = _run_dense_patch_match(
        smoke_paths,
        config,
        smoke_tag,
        allow_missing_files=True,
        config_path=smoke_config_path,
    )
    if smoke.get("status") != "completed" or int(smoke.get("typed_counts", {}).get("geometric_depth_count", 0)) <= 0:
        raise RuntimeError("fresh repaired CUDA PatchMatch smoke failed or produced no geometric depth")
    result["cuda_smoke"] = smoke
    result["patch_match_started"] = True

    full_paths, setup, mask_summary = _prepare_dense_workspace(
        tag=args.tag,
        config=config,
        image_names=image_names,
        sparse_model_path=model_path,
        workspace_root=scratch_root,
    )
    result["workspace_setup"] = setup
    result["mask_summary"] = mask_summary
    phase_records: list[dict[str, Any]] = []
    cleanup_manifest = RECONSTRUCTION_V4_ROOT / "work" / f"dense_{args.tag}_photometric_cleanup_manifest.json"
    for tile_index, config_path in enumerate(tile_paths):
        patch = _run_dense_patch_match(full_paths, config, f"{args.tag}_tile_{tile_index:03d}", config_path=config_path)
        typed = dense_typed_file_counts(full_paths["root"])
        phase_records.append({"tile_index": tile_index, "config_path": str(config_path.resolve()), "patch_match": patch, "typed_counts": typed})
        if patch.get("status") != "completed":
            raise RuntimeError(f"fresh repaired geometric PatchMatch tile {tile_index} failed")
        prune_dense_photometric_maps(full_paths["root"], keep_names=(), manifest_path=cleanup_manifest)
    depth_names = _geometric_map_names(full_paths["root"], "geometric")
    if depth_names != set(image_names):
        raise RuntimeError(f"fresh repaired geometric depth reference set is not exactly the 372 registered views: {len(depth_names)}")
    fused_path = RECONSTRUCTION_V4_ROOT / "dense" / f"fused_{args.tag}.ply"
    if fused_path.exists():
        raise FileExistsError(f"fused candidate already exists; refusing to overwrite: {fused_path}")
    fusion_log = RECONSTRUCTION_V4_ROOT / "work" / f"dense_{args.tag}_stereo_fusion.log"
    fusion = _run_colmap(
        build_stereo_fusion_command(full_paths["root"], fused_path, mask_path=full_paths["masks"], max_image_size=2000),
        fusion_log,
    )
    postfusion = build_postfusion_evidence(
        fused_path,
        workspace_root=full_paths["root"],
        sparse_model_path=full_paths["root"] / "sparse",
        mask_dir=full_paths["masks"],
        image_names=image_names,
        ring_by_name=ring_by_name,
        tile_config_paths=tile_paths,
        sparse_lineage=lineage,
        negative_evidence=summarize_g8_g9_negative_evidence(
            CAPTURE_V4_ROOT / "manifests" / "source_manifest.csv",
            args.isolation_records.resolve(),
        ),
        preview_dir=RECONSTRUCTION_V4_ROOT / "previews" / f"dense_{args.tag}_semantic",
        max_sources=MAX_SOURCES,
        chunk_size=None,
    )
    from v4_dense import postfusion_evidence_gate

    postfusion_gate = postfusion_evidence_gate(
        postfusion,
        fused_path=fused_path,
        expected_sparse_model_sha256=lineage["accepted_sparse_model_sha256"],
    )
    postfusion_path = RECONSTRUCTION_V4_ROOT / "reports" / f"dense_{args.tag}_postfusion.json"
    postfusion["postfusion_evidence_gate"] = postfusion_gate
    write_json(postfusion_path, postfusion)
    result.update(
        {
            "status": classify_dense_postfusion_completion(postfusion_gate)["status"],
            "phase_records": phase_records,
            "fusion": fusion,
            "fused_path": str(fused_path.resolve()),
            "fused_sha256": sha256_file(fused_path),
            "postfusion_report": str(postfusion_path.resolve()),
            "postfusion_gate": postfusion_gate,
            "dense_completion": classify_dense_postfusion_completion(postfusion_gate),
        }
    )
    report_path = RECONSTRUCTION_V4_ROOT / "reports" / f"dense_{args.tag}_run.json"
    write_json(report_path, result)
    # A strict post-fusion failure is preserved as evidence, but must not
    # discard a complete fresh run.  The explicit best-defensible continuation
    # path consumes this report and keeps every failed metric/reason visible.
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-sparse-report", type=Path, default=DEFAULT_ACCEPTED_REPORT)
    parser.add_argument("--graph-report", type=Path, default=DEFAULT_GRAPH_REPORT)
    parser.add_argument("--snapshot-manifest", type=Path, default=DEFAULT_SNAPSHOT_MANIFEST)
    parser.add_argument("--isolation-records", type=Path, default=RECONSTRUCTION_V4_ROOT / "work" / "isolation_records.json")
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
