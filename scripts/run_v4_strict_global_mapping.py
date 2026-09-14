"""Run one versioned, calibrated pyCOLMAP 4.2 global-mapping candidate.

The runner is deliberately narrow: it accepts only a disposable SQLite copy,
records the immutable v3 snapshot lineage, disables a second essential-matrix
decomposition (the calibrated two-view payload has already been re-estimated),
and writes one diagnostic report.  It does not promote a model or start dense
reconstruction; the normal independent sparse gate remains authoritative.
"""

from __future__ import annotations

import argparse
from enum import Enum
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v4_repair import (
    colmap_pair_id,
    load_canonical_sqlite_snapshot_manifest,
    load_ring_metadata,
    stable_directory_sha256,
    validate_exact_mapper_graph,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Convert pyCOLMAP option dictionaries into deterministic JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _registered_names(reconstruction: Any) -> list[str]:
    return sorted(str(reconstruction.image(int(image_id)).name) for image_id in reconstruction.reg_image_ids())


def _record_model(reconstruction: Any, model_root: Path, expected_names: list[str] | None) -> dict[str, Any]:
    names = _registered_names(reconstruction)
    expected = sorted(expected_names or [])
    return {
        "model_root": str(model_root.resolve()),
        "model_sha256": stable_directory_sha256(model_root),
        "registered_images": len(names),
        "registered_image_names_sha256": hashlib.sha256(
            json.dumps(names, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "registered_image_names_match_expected": bool(not expected or names == expected),
        "missing_expected_images": sorted(set(expected) - set(names)),
        "unexpected_registered_images": sorted(set(names) - set(expected)),
        "num_images": int(reconstruction.num_images()),
        "num_reg_images": int(reconstruction.num_reg_images()),
        "points3D": int(reconstruction.num_points3D()),
        "observations": int(reconstruction.compute_num_observations()),
        "mean_observations_per_registered_image": float(
            reconstruction.compute_mean_observations_per_reg_image()
        ),
        "mean_reprojection_error": float(reconstruction.compute_mean_reprojection_error()),
        "mean_track_length": float(reconstruction.compute_mean_track_length()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True, type=Path, help="Disposable remapped DB; canonical is rejected.")
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--ring-metadata", required=True, type=Path)
    parser.add_argument("--graph-report", required=True, type=Path)
    parser.add_argument("--remap-report", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--track-provenance",
        type=Path,
        help="Optional separate provenance report binding the completed model to this graph/run.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Only extract a report from an already completed output model; never rerun mapping.",
    )
    args = parser.parse_args()

    manifest = load_canonical_sqlite_snapshot_manifest(args.snapshot_manifest)
    canonical = Path(manifest["canonical_path"]).resolve()
    database = args.database.resolve()
    if database == canonical:
        raise RuntimeError("strict global mapping may only open a disposable DB copy")
    if not database.is_file():
        raise FileNotFoundError(database)
    if not args.image_root.is_dir():
        raise FileNotFoundError(args.image_root)

    expected_names = sorted(
        name for name, item in load_ring_metadata(args.ring_metadata).items() if item.get("selected", True)
    )
    graph_sha = _sha256(args.graph_report)
    remap_sha = _sha256(args.remap_report)
    graph_payload = json.loads(args.graph_report.read_text(encoding="utf-8"))
    graph_pairs = graph_payload.get("pairs")
    if not isinstance(graph_pairs, list) or not graph_pairs:
        raise ValueError("graph report must contain a non-empty pairs list")
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro&immutable=1", uri=True)
    try:
        image_ids = {
            str(name): int(image_id)
            for image_id, name in connection.execute("SELECT image_id, name FROM images")
        }
    finally:
        connection.close()
    expected_pair_ids: set[int] = set()
    for pair in graph_pairs:
        if not isinstance(pair, Mapping) or not pair.get("first") or not pair.get("second"):
            raise ValueError("graph report contains a pair without image names")
        first, second = str(pair["first"]), str(pair["second"])
        if first not in image_ids or second not in image_ids:
            raise ValueError(f"graph pair references an image absent from the disposable DB: {first}, {second}")
        pair_id = colmap_pair_id(image_ids[first], image_ids[second])
        if pair.get("pair_id") is not None and int(pair["pair_id"]) != pair_id:
            raise ValueError(f"graph pair_id mismatch for {first} -> {second}")
        expected_pair_ids.add(pair_id)
    if len(expected_pair_ids) != len(graph_pairs):
        raise ValueError("graph report contains duplicate image pairs")
    exact_graph_preflight = validate_exact_mapper_graph(
        database,
        expected_pair_ids,
        canonical_path=canonical,
        require_raw_matches=True,
    )
    if not exact_graph_preflight["passed"]:
        raise RuntimeError(
            "strict global mapping refused a non-exact disposable graph; "
            f"prune non-retained matches/two-view geometries first: {exact_graph_preflight}"
        )
    canonical_before = _sha256(canonical)
    if canonical_before != manifest["canonical_sha256"]:
        raise RuntimeError("canonical v3 snapshot does not match its manifest before mapping")

    import pycolmap

    options = pycolmap.GlobalPipelineOptions()
    options.num_threads = -1
    options.random_seed = 4201
    options.min_num_matches = 15
    options.multiple_models = False
    options.min_model_size = len(expected_names)
    # The DB was rebuilt from calibrated ALIKED/LightGlue observations.  Do
    # not decompose E a second time and silently choose a different pose.
    options.decompose_relative_pose = False
    options.mapper.num_threads = -1
    options.mapper.random_seed = 4201
    options.mapper.track_min_num_views_per_track = 3
    options.mapper.global_positioning.min_num_view_per_track = 3
    options.mapper.rotation_averaging.random_seed = 4201
    options.mapper.global_positioning.random_seed = 4201
    options.mapper.bundle_adjustment.min_track_length = 3
    options.mapper.retriangulation.random_seed = 4201
    options.mapper.image_path = args.image_root.resolve()
    options.image_path = args.image_root.resolve()

    report: dict[str, Any] = {
        "schema_version": 1,
        "architecture": "pyCOLMAP 4.2.0 global_mapping (GLOMAP), strict calibrated payload",
        "pycolmap_version": str(pycolmap.__version__),
        "database_path": str(database),
        "database_sha256_before": _sha256(database),
        "snapshot_manifest_path": str(args.snapshot_manifest.resolve()),
        "snapshot_manifest_sha256": _sha256(args.snapshot_manifest),
        "canonical_snapshot_sha256": manifest["canonical_sha256"],
        "canonical_creation_logical_sha256": manifest["creation_logical_sha256"],
        "canonical_sha256_before": canonical_before,
        "ring_metadata_path": str(args.ring_metadata.resolve()),
        "expected_registered_views": len(expected_names),
        "expected_image_names_sha256": hashlib.sha256(
            json.dumps(expected_names, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "graph_report_path": str(args.graph_report.resolve()),
        "graph_report_sha256": graph_sha,
        "expected_mapper_pair_count": len(expected_pair_ids),
        "exact_mapper_graph_preflight": exact_graph_preflight,
        "remap_report_path": str(args.remap_report.resolve()),
        "remap_report_sha256": remap_sha,
        "track_provenance_path": (
            str(args.track_provenance.resolve()) if args.track_provenance is not None else None
        ),
        "options": _json_safe(options.todict()),
        "lineage": {
            "constructed_pairwise_tracks": False,
            "track_builder": "pycolmap_global_mapping_strict_calibrated_graph",
            "source_correspondence": "existing calibrated ALIKED/LightGlue rows only",
            "canonical_sqlite_opened": False,
            "opened_database_is_disposable": True,
        },
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.reuse_existing:
            model_dirs = sorted(
                path
                for path in args.output_root.resolve().iterdir()
                if path.is_dir() and (path / "images.bin").is_file()
            )
            models = {
                int(path.name): pycolmap.Reconstruction(str(path))
                for path in model_dirs
                if path.name.isdigit()
            }
            report["mapping_reused_existing_output"] = True
        else:
            models = pycolmap.global_mapping(
                str(database),
                str(args.image_root.resolve()),
                str(args.output_root.resolve()),
                options,
            )
            report["mapping_reused_existing_output"] = False
        model_records: list[dict[str, Any]] = []
        for model_id, reconstruction in sorted(models.items(), key=lambda item: int(item[0])):
            model_root = args.output_root.resolve() / str(model_id)
            if not model_root.is_dir():
                model_root = args.output_root.resolve()
            record = _record_model(reconstruction, model_root, expected_names)
            record["model_id"] = str(model_id)
            model_records.append(record)
        report["model_count"] = len(model_records)
        report["models"] = model_records
        report["single_model_all_expected_views"] = bool(
            len(model_records) == 1
            and model_records[0]["registered_images"] == len(expected_names)
            and model_records[0]["registered_image_names_match_expected"]
        )
        report["status"] = "completed"
    except Exception as exc:  # preserve a machine-readable negative candidate report
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["model_count"] = 0
        report["models"] = []
    finally:
        report["database_sha256_after"] = _sha256(database)
        report["canonical_sha256_after"] = _sha256(canonical)
        report["canonical_snapshot_hash_stable"] = bool(
            report["canonical_sha256_after"] == canonical_before == manifest["canonical_sha256"]
        )
        report["disposable_database_hash_stable"] = bool(
            report["database_sha256_after"] == report["database_sha256_before"]
        )
        report["canonical_sidecars_present"] = [
            str(path) for path in (canonical.with_name(canonical.name + "-wal"), canonical.with_name(canonical.name + "-shm")) if path.exists()
        ]
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    if args.track_provenance is not None and report.get("status") == "completed":
        models = report.get("models") or []
        if len(models) != 1 or not report.get("single_model_all_expected_views"):
            raise RuntimeError("track provenance requires one completed model containing every expected view")
        model = models[0]
        remap_payload = json.loads(args.remap_report.read_text(encoding="utf-8"))
        graph_sha = _sha256(args.graph_report)
        remap_sha = _sha256(args.remap_report)
        provenance = {
            "schema_version": 1,
            "source": "verified_disposable_sqlite_graph",
            "independent_graph": True,
            "independent_graph_sha256": str(graph_payload.get("classification_sha256", "")),
            "constructed_pairwise_tracks": False,
            "track_builder": "pycolmap_global_mapping_strict_calibrated_graph",
            "model_architecture": "pyCOLMAP 4.2.0 global_mapping from a physically exact calibrated conflict-free graph; genuine multiview track establishment and BA",
            "canonical_snapshot_manifest": str(args.snapshot_manifest.resolve()),
            "canonical_snapshot_manifest_sha256": _sha256(args.snapshot_manifest),
            "canonical_snapshot_sha256": manifest["canonical_sha256"],
            "canonical_creation_logical_sha256": manifest["creation_logical_sha256"],
            "raw_audit_report": remap_payload.get("raw_audit_report"),
            "raw_audit_report_sha256": remap_payload.get("raw_audit_report_sha256"),
            "classification_report": graph_payload.get("classification_path"),
            "classification_report_sha256": graph_payload.get("classification_sha256"),
            "fixed_audit_pair_count": graph_payload.get("fixed_audit_pair_count"),
            "fixed_audit_pair_ids_sha256": graph_payload.get("fixed_audit_pair_ids_sha256"),
            "graph_report": str(args.graph_report.resolve()),
            "graph_report_sha256": graph_sha,
            "mapper_pair_count": len(expected_pair_ids),
            "graph_pair_identity_sha256": graph_payload.get("accepted_pair_ids_sha256", graph_payload.get("pair_identity_sha256")),
            "expected_pair_ids_sha256": exact_graph_preflight["expected_pair_ids_sha256"],
            "remap_report": str(args.remap_report.resolve()),
            "remap_report_sha256": remap_sha,
            "run_report": str(args.report.resolve()),
            "run_report_sha256": _sha256(args.report),
            "model_directory": model["model_root"],
            "model_directory_sha256": model["model_sha256"],
            "model_registered_images": model["registered_images"],
            "model_points3D": model["points3D"],
            "model_observations": model["observations"],
            "model_mean_track_length": model["mean_track_length"],
            "model_mean_reprojection_error": model["mean_reprojection_error"],
            "raw_matches_pruned": True,
            "track_loading_path": exact_graph_preflight["track_loading_path"],
        }
        args.track_provenance.parent.mkdir(parents=True, exist_ok=True)
        args.track_provenance.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(json.dumps({
        "status": report["status"],
        "model_count": report.get("model_count"),
        "single_model_all_expected_views": report.get("single_model_all_expected_views", False),
        "canonical_snapshot_hash_stable": report.get("canonical_snapshot_hash_stable"),
        "report": str(args.report.resolve()),
        "track_provenance": str(args.track_provenance.resolve()) if args.track_provenance else None,
    }, sort_keys=True))
    return 0 if report.get("status") == "completed" and report.get("single_model_all_expected_views") else 2


if __name__ == "__main__":
    raise SystemExit(main())
