"""Audit an image-derived v26 pose field against the exact v43 graph.

This is a bounded diagnostic, not a pose splice or accepted sparse model.  The
v26 model is used only as an image-derived initialization for native pyCOLMAP
``triangulate_points``.  The database is copied to a disposable temporary
working file, and all graph/mask/lineage measurements are taken from that
working copy.  No canonical SQLite path is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
from typing import Any

import numpy as np
import pycolmap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from v4_repair import (  # noqa: E402
    SparseIntegrityConfig,
    load_ring_metadata,
    sparse_camera_center_translation_evidence,
    sparse_mask_projection_evidence,
    stable_directory_sha256,
)
from diagnose_v4_v6_centers_v43_rotations import _load_translation_records  # noqa: E402


WORKSPACE_ROOT = Path(r"D:\Side Projects\CSX4213_V4_Dense_Work\workspace_v4repair_sparse_v1")
DEFAULT_SOURCE_MODEL = WORKSPACE_ROOT / "sparse_candidate_v26_strict_calibrated_global" / "0"
DEFAULT_DATABASE = WORKSPACE_ROOT / "database_v43_match_conflict_free_outlier_cap15_direct_calibrated_exact1530_disposable.db"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "CSX4213_Project_V4_Images"
REPAIR_ROOT = PROJECT_ROOT / "reconstruction" / "v4" / "repair" / "sparse_v1"
DEFAULT_RING_METADATA = REPAIR_ROOT / "selected_ring_metadata_v1.json"
DEFAULT_RAW_AUDIT = REPAIR_ROOT / "independent_audit_geometry_v8_calibrated_units.json"
DEFAULT_CLASSIFICATION = REPAIR_ROOT / "calibrated_pair_classification_v3_calibrated_units.json"
DEFAULT_CANONICAL_MANIFEST = REPAIR_ROOT / "canonical_sqlite_snapshot_v3.json"
DEFAULT_GRAPH_REPORT = REPAIR_ROOT / "match_conflict_free_v39_outlier_cap15_exact1530_preparation_report.json"
DEFAULT_OUTPUT_MODEL = WORKSPACE_ROOT / "sparse_candidate_v49_v26_pose_exact_graph_retriangulation"
DEFAULT_OUTPUT_REPORT = REPAIR_ROOT / "v49_v26_pose_exact_graph_retriangulation_diagnostic.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(manifest_path: Path, source_db: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical = Path(str(manifest["canonical_path"]))
    current = _sha256(canonical)
    sidecars = {
        suffix: canonical.with_name(canonical.name + suffix).exists()
        for suffix in ("-wal", "-shm")
    }
    return {
        "manifest_path": str(manifest_path.resolve()),
        "canonical_path": str(canonical.resolve()),
        "recorded_sha256": str(manifest["canonical_sha256"]),
        "current_sha256": current,
        "raw_sha_equal": current == str(manifest["canonical_sha256"]),
        "creation_logical_sha256": str(manifest.get("creation_logical_sha256", "")),
        "sidecars_present": sidecars,
        "canonical_sqlite_opened": False,
        "source_database_is_disposable": source_db.resolve() != canonical.resolve(),
    }


def _pair_ids(database_path: Path) -> tuple[set[int], set[int], int, int]:
    """Read exact pair IDs from a disposable SQLite file in read-only mode."""
    uri = "file:" + database_path.resolve().as_posix() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        tvg = {int(row[0]) for row in connection.execute("SELECT pair_id FROM two_view_geometries")}
        raw = {int(row[0]) for row in connection.execute("SELECT pair_id FROM matches")}
        # ``rows`` is the encoded correspondence-row count for each pair.  Do
        # not sum one COUNT(*) per pair here: that only reports the number of
        # nonempty groups and can falsely look like a 1,530-row graph.
        tvg_rows = int(
            connection.execute(
                "SELECT COALESCE(SUM(rows), 0) FROM two_view_geometries WHERE rows > 0"
            ).fetchone()[0]
        )
        match_rows = int(
            connection.execute(
                "SELECT COALESCE(SUM(rows), 0) FROM matches WHERE rows > 0"
            ).fetchone()[0]
        )
    finally:
        connection.close()
    return tvg, raw, tvg_rows, match_rows


def _track_summary(reconstruction: Any) -> dict[str, Any]:
    lengths: list[int] = []
    duplicate_image_tracks = 0
    for point in reconstruction.points3D.values():
        image_ids = [int(element.image_id) for element in point.track.elements]
        lengths.append(len(image_ids))
        if len(image_ids) != len(set(image_ids)):
            duplicate_image_tracks += 1
    values = np.asarray(lengths, dtype=np.float64)
    return {
        "point_count_with_tracks": int(values.size),
        "mean_track_length": float(values.mean()) if values.size else 0.0,
        "track_length_p50": float(np.quantile(values, 0.50)) if values.size else 0.0,
        "track_length_p90": float(np.quantile(values, 0.90)) if values.size else 0.0,
        "fraction_track_length_ge3": float(np.mean(values >= 3.0)) if values.size else 0.0,
        "fraction_track_length_eq2": float(np.mean(values == 2.0)) if values.size else 0.0,
        "maximum_track_length": int(values.max()) if values.size else 0,
        "duplicate_image_track_count": int(duplicate_image_tracks),
    }


def _pose_delta(source: Any, result: Any) -> dict[str, Any]:
    rotation_errors: list[float] = []
    center_errors: list[float] = []
    source_by_name = {str(source.image(int(i)).name): source.image(int(i)) for i in source.reg_image_ids()}
    result_by_name = {str(result.image(int(i)).name): result.image(int(i)) for i in result.reg_image_ids()}
    common = sorted(set(source_by_name) & set(result_by_name))
    for name in common:
        first = np.asarray(source_by_name[name].cam_from_world().rotation.matrix(), dtype=np.float64)
        second = np.asarray(result_by_name[name].cam_from_world().rotation.matrix(), dtype=np.float64)
        cosine = float(np.clip((np.trace(second @ first.T) - 1.0) / 2.0, -1.0, 1.0))
        rotation_errors.append(float(np.degrees(np.arccos(cosine))))
        center_errors.append(
            float(
                np.linalg.norm(
                    np.asarray(result_by_name[name].projection_center(), dtype=np.float64)
                    - np.asarray(source_by_name[name].projection_center(), dtype=np.float64)
                )
            )
        )
    return {
        "common_view_count": len(common),
        "rotation_geodesic_max_deg": max(rotation_errors, default=float("inf")),
        "center_delta_max_m": max(center_errors, default=float("inf")),
        "pose_field_unchanged": bool(
            common
            and max(rotation_errors, default=float("inf")) <= 1e-5
            and max(center_errors, default=float("inf")) <= 1e-9
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_model.exists() or args.output_report.exists():
        raise FileExistsError("preserve existing candidate/report; choose a new versioned path")
    graph = json.loads(args.graph_report.read_text(encoding="utf-8"))
    exact_graph = graph.get("exact_mapper_graph_preflight", {})
    expected_pairs = {
        int(item["pair_id"])
        for item in graph.get("pairs", [])
        if isinstance(item, dict) and "pair_id" in item
    }
    if len(expected_pairs) != 1530:
        raise ValueError("graph report does not expose exactly 1,530 expected pair IDs")
    canonical = _canonical_bytes(args.canonical_manifest, args.database)
    source_model = pycolmap.Reconstruction(str(args.source_model))
    source_model_sha = stable_directory_sha256(args.source_model)

    with tempfile.TemporaryDirectory(prefix="v4_v49_exact_graph_") as temporary:
        working_db = Path(temporary) / "exact_graph_working.db"
        shutil.copy2(args.database, working_db)
        tvg_pairs, raw_pairs, tvg_rows, match_rows = _pair_ids(working_db)
        exact_preflight = {
            "expected_pair_count": int(exact_graph.get("expected_pair_count", len(expected_pairs))),
            "actual_two_view_geometry_count": len(tvg_pairs),
            "actual_raw_match_pair_count": len(raw_pairs),
            "two_view_geometry_pair_ids_exact": tvg_pairs == expected_pairs,
            "raw_match_pair_ids_exact": raw_pairs == expected_pairs,
            "two_view_geometry_inlier_rows": tvg_rows,
            "raw_match_rows": match_rows,
            "passed": bool(
                len(expected_pairs) == 1530
                and len(tvg_pairs) == 1530
                and len(raw_pairs) == 1530
                and tvg_pairs == expected_pairs
                and raw_pairs == expected_pairs
            ),
        }
        options = pycolmap.IncrementalPipelineOptions()
        options.extract_colors = False
        options.triangulation.ignore_two_view_tracks = True
        options.triangulation.random_seed = 4201
        options.image_path = args.image_root
        result = pycolmap.triangulate_points(
            source_model,
            str(working_db),
            str(args.image_root),
            str(args.output_model),
            clear_points=True,
            options=options,
            refine_intrinsics=False,
        )
        mask = sparse_mask_projection_evidence(
            result,
            args.mask_root,
            load_ring_metadata(args.ring_metadata),
            config=SparseIntegrityConfig(),
        )
        raw_audit = json.loads(args.raw_audit.read_text(encoding="utf-8"))
        classification = json.loads(args.classification.read_text(encoding="utf-8"))
        translation = sparse_camera_center_translation_evidence(
            result,
            _load_translation_records(raw_audit, classification),
            config=SparseIntegrityConfig(),
        )
        pose_delta = _pose_delta(source_model, result)
        track = _track_summary(result)

    return {
        "schema_version": 1,
        "status": "diagnostic_only",
        "method": "native pyCOLMAP 4.2 triangulate_points from v26 image-derived pose initialization on exact 1,530-pair disposable graph",
        "source_model": str(args.source_model.resolve()),
        "source_model_sha256": source_model_sha,
        "output_model": str(args.output_model.resolve()),
        "output_model_sha256": stable_directory_sha256(args.output_model),
        "registered_images": int(result.num_reg_images()),
        "points3D": int(result.num_points3D()),
        "observations": int(result.compute_num_observations()),
        "mean_reprojection_error": float(result.compute_mean_reprojection_error()),
        "mean_track_length": float(result.compute_mean_track_length()),
        "track_summary": track,
        "pose_delta": pose_delta,
        "exact_graph_preflight": exact_preflight,
        "mask_projection": mask,
        "translation_diagnostic": translation,
        "frozen_canonical_snapshot": canonical,
        "promotion_allowed": False,
        "full_sparse_candidate_eligible": False,
        "next_step_if_positive": "Only a fresh joint pyCOLMAP optimization may consume this initialization; this model is not promotable.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--mask-root", type=Path, default=PROJECT_ROOT / "capture_v4" / "derived" / "masks")
    parser.add_argument("--ring-metadata", type=Path, default=DEFAULT_RING_METADATA)
    parser.add_argument("--raw-audit", type=Path, default=DEFAULT_RAW_AUDIT)
    parser.add_argument("--classification", type=Path, default=DEFAULT_CLASSIFICATION)
    parser.add_argument("--canonical-manifest", type=Path, default=DEFAULT_CANONICAL_MANIFEST)
    parser.add_argument("--graph-report", type=Path, default=DEFAULT_GRAPH_REPORT)
    parser.add_argument("--output-model", type=Path, default=DEFAULT_OUTPUT_MODEL)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_OUTPUT_REPORT)
    args = parser.parse_args()
    payload = run(args)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "registered_images": payload["registered_images"],
                "points3D": payload["points3D"],
                "mask_passed": payload["mask_projection"].get("passed"),
                "output_model": payload["output_model"],
                "report": str(args.output_report.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
