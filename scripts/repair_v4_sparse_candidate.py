"""Build a bounded calibrated pose-repair sparse candidate.

The source model is only an image-derived initialization.  Relative rotations
come from the fixed calibrated ALIKED/LightGlue audit, while observations are
rebuilt from a disposable SQLite copy made from the frozen snapshot manifest.
This script never opens the canonical database and never creates pair-local
tracks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_v4_rotation_consensus import (  # noqa: E402
    build_ring_aligned_repair,
    recover_translation_direction_edges,
    solve_camera_centers,
    write_consensus_model,
)
from v4_repair import (  # noqa: E402
    create_disposable_sqlite_from_manifest,
    stable_directory_sha256,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference_orientations(path: Path) -> dict[str, np.ndarray]:
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(path))
    return {
        str(image.name): np.asarray(image.cam_from_world().rotation.matrix(), dtype=np.float64)
        for image in reconstruction.images.values()
        if image.has_pose
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification", required=True, type=Path)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--output-model", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--gauge-threshold-deg", type=float, default=15.0)
    parser.add_argument("--raw-audit-report", type=Path)
    parser.add_argument("--solve-positions", action="store_true")
    parser.add_argument("--max-reprojection-px", type=float, default=12.0)
    parser.add_argument("--max-median-reprojection-px", type=float, default=6.0)
    args = parser.parse_args()

    classification_payload = json.loads(args.classification.read_text(encoding="utf-8"))
    records = classification_payload.get("classification", classification_payload)
    if not isinstance(records, list) or not records:
        raise ValueError("classification payload must contain a non-empty classification list")
    if args.output_model.exists():
        raise FileExistsError(args.output_model)
    if args.solve_positions and args.raw_audit_report is None:
        raise ValueError("--raw-audit-report is required with --solve-positions")
    if args.raw_audit_report is not None:
        raw_payload = json.loads(args.raw_audit_report.read_text(encoding="utf-8"))
        raw_by_pair = {int(item["pair_id"]): item for item in raw_payload.get("records", [])}
        records = [
            {
                **record,
                **(
                    {
                        "calibrated_reestimate_translation_first_to_second": raw_by_pair[
                            int(record["pair_id"])
                        ].get("calibrated_reestimate_translation_first_to_second")
                    }
                    if int(record["pair_id"]) in raw_by_pair
                    else {}
                ),
            }
            for record in records
        ]

    reference = _reference_orientations(args.source_model)
    orientations, repair_report = build_ring_aligned_repair(
        records,
        reference,
        gauge_threshold_deg=args.gauge_threshold_deg,
    )
    allowed_pair_ids = {
        int(record["pair_id"])
        for record in records
        if bool(record.get("well_conditioned_calibrated"))
    }

    report: dict[str, object] = {
        "schema_version": 1,
        "method": "bounded calibrated same-ring pose repair with disposable union-find tracks",
        "classification_path": str(args.classification.resolve()),
        "classification_sha256": _sha256(args.classification),
        "source_model": str(args.source_model.resolve()),
        "source_model_sha256": stable_directory_sha256(args.source_model),
        "snapshot_manifest_path": str(args.snapshot_manifest.resolve()),
        "snapshot_manifest_sha256": _sha256(args.snapshot_manifest),
        "allowed_pair_count": len(allowed_pair_ids),
        "track_provenance": {
            "source": "verified_disposable_sqlite_graph",
            "independent_graph": True,
            "independent_graph_sha256": _sha256(args.classification),
            "constructed_pairwise_tracks": False,
            "track_builder": "union_find_observation_components",
        },
        # Keep the provenance object directly consumable by the sparse
        # evaluator as well as nested in the human-readable repair report.
        "source": "verified_disposable_sqlite_graph",
        "independent_graph": True,
        "independent_graph_sha256": _sha256(args.classification),
        "constructed_pairwise_tracks": False,
        "track_builder": "union_find_observation_components",
        "repair": repair_report,
    }

    with tempfile.TemporaryDirectory(prefix="v4_sparse_pose_repair_db_") as temporary:
        working_path = Path(temporary) / "working.db"
        disposable = create_disposable_sqlite_from_manifest(args.snapshot_manifest, working_path)
        report["working_database_sha256"] = disposable["working"]["sha256"]
        import pycolmap

        database = pycolmap.Database.open(str(working_path))
        try:
            camera_centers = None
            if args.solve_positions:
                source_reconstruction = pycolmap.Reconstruction(str(args.source_model))
                initial_centers = {
                    str(image.name): -np.asarray(image.cam_from_world().rotation.matrix()).T
                    @ np.asarray(image.cam_from_world().translation, dtype=np.float64)
                    for image in source_reconstruction.images.values()
                    if image.has_pose
                }
                translation_edges, translation_report = recover_translation_direction_edges(
                    database,
                    records,
                    orientations,
                    allowed_pair_ids,
                )
                camera_centers, position_report = solve_camera_centers(
                    translation_edges,
                    initial_centers,
                )
                report["translation_direction_recovery"] = translation_report
                report["camera_center_solve"] = position_report
            report["model_write"] = write_consensus_model(
                args.source_model,
                args.output_model,
                orientations,
                camera_centers=camera_centers,
                database=database,
                allowed_pair_ids=allowed_pair_ids,
                pairwise_tracks=False,
                max_reprojection_px=args.max_reprojection_px,
                max_median_reprojection_px=args.max_median_reprojection_px,
            )
        finally:
            database.close()

    report["output_model_sha256"] = stable_directory_sha256(args.output_model)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_model": str(args.output_model.resolve()),
        "output_model_sha256": report["output_model_sha256"],
        "model_write": report["model_write"],
        "low_support_rings": report["repair"]["low_support_rings"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
