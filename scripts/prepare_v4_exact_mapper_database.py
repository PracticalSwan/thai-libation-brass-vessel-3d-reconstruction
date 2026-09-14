"""Create an exact, disposable COLMAP mapper graph from the frozen v3 DB.

The manifest-bound canonical SQLite file is copied as bytes and is never
opened by SQLite or pyCOLMAP.  The disposable copy is physically pruned in
both ``two_view_geometries`` and ``matches`` before any remap or GLOMAP call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v4_repair import (  # noqa: E402
    colmap_pair_id,
    load_canonical_sqlite_snapshot_manifest,
    prune_colmap_database,
    validate_exact_mapper_graph,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pairs(payload: Any) -> list[dict[str, Any]]:
    values = payload.get("pairs", payload.get("mapping_records")) if isinstance(payload, Mapping) else payload
    if not isinstance(values, list) or not values:
        raise ValueError("mapper graph input must contain a non-empty pairs list")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        if isinstance(value, Mapping):
            first, second = str(value.get("first", "")), str(value.get("second", ""))
            declared = int(value["pair_id"]) if value.get("pair_id") is not None else None
        else:
            first, second = str(value[0]), str(value[1])
            declared = None
        if not first or not second:
            raise ValueError("mapper graph pair lacks image names")
        key = (first, second)
        if key in seen:
            raise ValueError(f"mapper graph contains duplicate pair: {key}")
        seen.add(key)
        normalized.append({"first": first, "second": second, "declared_pair_id": declared})
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    manifest = load_canonical_sqlite_snapshot_manifest(args.snapshot_manifest)
    canonical = Path(manifest["canonical_path"]).resolve()
    output = args.output.resolve()
    if output == canonical:
        raise ValueError("exact mapper output may not be the canonical SQLite snapshot")
    if output.exists():
        raise FileExistsError(output)
    graph_payload = json.loads(args.pairs.read_text(encoding="utf-8"))
    graph_pairs = _pairs(graph_payload)
    canonical_before = _sha256(canonical)
    with tempfile.TemporaryDirectory(prefix="v4_exact_mapper_seed_") as temporary:
        seed = Path(temporary) / "seed.db"
        from v4_repair import create_disposable_sqlite_from_manifest

        seed_info = create_disposable_sqlite_from_manifest(args.snapshot_manifest, seed)
        connection = sqlite3.connect(f"file:{seed.as_posix()}?mode=ro&immutable=1", uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            image_ids = {
                str(name): int(image_id)
                for image_id, name in connection.execute("SELECT image_id, name FROM images")
            }
        finally:
            connection.close()
        selected_pairs: list[tuple[str, str]] = []
        expected_pair_ids: set[int] = set()
        for item in graph_pairs:
            first, second = item["first"], item["second"]
            if first not in image_ids or second not in image_ids:
                raise ValueError(f"pair references image absent from disposable DB: {first}, {second}")
            pair_id = colmap_pair_id(image_ids[first], image_ids[second])
            declared = item["declared_pair_id"]
            if declared is not None and declared != pair_id:
                raise ValueError(f"pair_id mismatch for {first} -> {second}")
            expected_pair_ids.add(pair_id)
            selected_pairs.append((first, second))
        prune_colmap_database(
            seed,
            output,
            selected_pairs,
            image_ids_by_name=image_ids,
            snapshot_manifest=args.snapshot_manifest,
        )
        source_seed_sha = _sha256(seed)

    preflight = validate_exact_mapper_graph(
        output,
        expected_pair_ids,
        canonical_path=canonical,
        require_raw_matches=True,
    )
    canonical_after = _sha256(canonical)
    if canonical_after != canonical_before or canonical_after != manifest["canonical_sha256"]:
        raise RuntimeError("canonical SQLite snapshot changed while preparing exact mapper DB")
    if not preflight["passed"]:
        raise RuntimeError(f"exact mapper graph preflight failed: {preflight}")
    report = {
        "schema_version": 1,
        "status": "complete",
        "method": "fresh manifest-bound disposable copy; prune non-retained verified geometries and raw matches",
        "pycolmap_track_loading_contract": preflight["track_loading_path"],
        "canonical_sqlite_opened": False,
        "snapshot_manifest": str(args.snapshot_manifest.resolve()),
        "snapshot_manifest_sha256": manifest["manifest_sha256"],
        "canonical_snapshot_sha256": manifest["canonical_sha256"],
        "canonical_creation_logical_sha256": manifest["creation_logical_sha256"],
        "canonical_sha256_before": canonical_before,
        "canonical_sha256_after": canonical_after,
        "canonical_snapshot_hash_stable": True,
        "graph_input": str(args.pairs.resolve()),
        "graph_input_sha256": _sha256(args.pairs),
        "source_seed_sha256": source_seed_sha,
        "output_path": str(output),
        "output_sha256": _sha256(output),
        "expected_pair_count": len(expected_pair_ids),
        "expected_pair_ids_sha256": preflight["expected_pair_ids_sha256"],
        "raw_matches_pruned": True,
        "exact_mapper_graph_preflight": preflight,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "pair_count": report["expected_pair_count"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
