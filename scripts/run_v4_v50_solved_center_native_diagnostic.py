"""Run one bounded solved-center/native-pyCOLMAP sparse diagnostic.

The v6 camera centers are used only as an image-derived initial condition for
the calibrated translation-direction solve.  v43 rotations are used as the
pose initializer because the existing independent SO(3)/trajectory gate has
already accepted that field.  The initializer is never an acceptance model:
all points are cleared and rebuilt by native pyCOLMAP from an exact disposable
1,530-pair graph, followed by one joint bundle-adjustment pass with fixed
intrinsics and one gauge anchor.  The caller must run the unchanged full
independent sparse evaluator on the resulting model.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import shutil
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

from build_v4_rotation_consensus import (  # noqa: E402
    recover_translation_direction_edges,
    solve_camera_centers,
    write_consensus_model,
)
from diagnose_v4_v26_pose_exact_graph_retriangulation import (  # noqa: E402
    _pair_ids,
)
from diagnose_v4_v6_centers_v43_rotations import (  # noqa: E402
    _load_translation_records,
)
from v4_repair import (  # noqa: E402
    SparseIntegrityConfig,
    load_ring_metadata,
    sparse_camera_center_translation_evidence,
    stable_directory_sha256,
)


WORKSPACE_ROOT = Path(r"D:\Side Projects\CSX4213_V4_Dense_Work\workspace_v4repair_sparse_v1")
REPAIR_ROOT = PROJECT_ROOT / "reconstruction" / "v4" / "repair" / "sparse_v1"
DEFAULT_V43_MODEL = WORKSPACE_ROOT / "sparse_candidate_v43_match_conflict_free_outlier_cap15_direct_calibrated_global" / "0"
DEFAULT_V6_MODEL = WORKSPACE_ROOT / "sparse_candidate_v6_glomap_global" / "0"
DEFAULT_DATABASE = WORKSPACE_ROOT / "database_v43_match_conflict_free_outlier_cap15_direct_calibrated_exact1530_disposable.db"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "CSX4213_Project_V4_Images"
DEFAULT_MASK_ROOT = PROJECT_ROOT / "capture_v4" / "derived" / "masks"
DEFAULT_RING_METADATA = REPAIR_ROOT / "selected_ring_metadata_v1.json"
DEFAULT_RAW_AUDIT = REPAIR_ROOT / "independent_audit_geometry_v8_calibrated_units.json"
DEFAULT_CLASSIFICATION = REPAIR_ROOT / "calibrated_pair_classification_v3_calibrated_units.json"
DEFAULT_GRAPH_REPORT = REPAIR_ROOT / "match_conflict_free_v39_outlier_cap15_exact1530_preparation_report.json"
DEFAULT_CANONICAL_MANIFEST = REPAIR_ROOT / "canonical_sqlite_snapshot_v3.json"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "sparse_candidate_v50_solved_center_native_diagnostic"
DEFAULT_REPORT = REPAIR_ROOT / "v50_solved_center_native_diagnostic.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_snapshot(manifest_path: Path, source_database: Path) -> dict[str, Any]:
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
        "source_database_is_disposable": source_database.resolve() != canonical.resolve(),
    }


def _pose_initializer_evidence(
    v43: pycolmap.Reconstruction,
    v6: pycolmap.Reconstruction,
    solved_centers: Mapping[str, np.ndarray],
    expected_names: set[str],
) -> dict[str, Any]:
    v43_by_name = {str(v43.image(int(i)).name): v43.image(int(i)) for i in v43.reg_image_ids()}
    v6_by_name = {str(v6.image(int(i)).name): v6.image(int(i)) for i in v6.reg_image_ids()}
    names = set(v43_by_name)
    rotation_orthogonality = []
    determinants = []
    center_finite = True
    round_trip_residuals = []
    for name in sorted(names):
        image = v43_by_name[name]
        rotation = np.asarray(image.cam_from_world().rotation.matrix(), dtype=np.float64)
        translation = np.asarray(image.cam_from_world().translation, dtype=np.float64)
        center = np.asarray(solved_centers[name], dtype=np.float64)
        rotation_orthogonality.append(float(np.linalg.norm(rotation @ rotation.T - np.eye(3))))
        determinants.append(float(np.linalg.det(rotation)))
        center_finite = center_finite and bool(np.isfinite(center).all())
        round_trip_residuals.append(float(np.linalg.norm(translation + rotation @ np.asarray(image.projection_center(), dtype=np.float64))))
    source_center_delta = [
        float(np.linalg.norm(np.asarray(solved_centers[name]) - np.asarray(v6_by_name[name].projection_center())))
        for name in sorted(names & set(v6_by_name))
    ]
    return {
        "exact_view_set": names == expected_names and names == set(v6_by_name),
        "registered_view_count": len(names),
        "rotation_field_source": "v43 calibrated global model; unchanged as initializer",
        "v43_model_sha256": stable_directory_sha256(DEFAULT_V43_MODEL),
        "v6_model_sha256": stable_directory_sha256(DEFAULT_V6_MODEL),
        "rotation_orthogonality_max": max(rotation_orthogonality, default=float("inf")),
        "rotation_determinant_min": min(determinants, default=float("nan")),
        "solved_center_finite": center_finite,
        "initializer_translation_round_trip_max": max(round_trip_residuals, default=float("inf")),
        "center_delta_from_v6_m": {
            "p50": float(np.quantile(source_center_delta, 0.50)) if source_center_delta else float("inf"),
            "p90": float(np.quantile(source_center_delta, 0.90)) if source_center_delta else float("inf"),
            "max": max(source_center_delta, default=float("inf")),
        },
    }


def _model_summary(reconstruction: pycolmap.Reconstruction) -> dict[str, Any]:
    return {
        "registered_images": int(reconstruction.num_reg_images()),
        "points3D": int(reconstruction.num_points3D()),
        "observations": int(reconstruction.compute_num_observations()),
        "mean_reprojection_error": float(reconstruction.compute_mean_reprojection_error()),
        "mean_track_length": float(reconstruction.compute_mean_track_length()),
    }


def _build_joint_ba_config(
    reconstruction: pycolmap.Reconstruction,
) -> tuple[pycolmap.BundleAdjustmentConfig, int, int]:
    """Build the joint-BA config with every registered image explicitly selected."""

    image_ids = sorted(int(image_id) for image_id in reconstruction.reg_image_ids())
    if not image_ids:
        raise RuntimeError("joint bundle adjustment requires at least one registered image")

    config = pycolmap.BundleAdjustmentConfig()
    # BundleAdjustmentConfig starts with no selected images.  Register every
    # image before applying gauge and intrinsic constraints so residuals are
    # actually present in the joint solve.
    for image_id in image_ids:
        config.add_image(image_id)

    anchor_image_id = image_ids[0]
    config.set_constant_rig_from_world_pose(
        int(reconstruction.image(anchor_image_id).frame_id)
    )
    for camera_id in sorted(int(camera_id) for camera_id in reconstruction.cameras):
        config.set_constant_cam_intrinsics(camera_id)

    residual_count = int(config.num_residuals(reconstruction))
    if residual_count <= 0:
        raise RuntimeError(
            "joint bundle adjustment configuration must contain a nonzero residual count"
        )
    return config, anchor_image_id, residual_count


def _validate_bundle_adjustment_summary(
    summary: Mapping[str, Any],
    *,
    configured_residual_count: int,
) -> dict[str, Any]:
    """Fail closed unless pyCOLMAP reports a real successful BA solve."""

    if configured_residual_count <= 0:
        raise RuntimeError("joint bundle adjustment was configured with zero residuals")
    payload = dict(summary)
    residual_count = int(payload.get("num_residuals", 0))
    if residual_count <= 0:
        raise RuntimeError("bundle adjustment must report a nonzero residual count")
    termination = payload.get("termination_type")
    termination_name = getattr(termination, "name", None)
    if not isinstance(termination_name, str):
        termination_name = str(termination).rsplit(".", 1)[-1].upper()
    accepted_termination_names = {"SUCCESS", "CONVERGENCE", "USER_SUCCESS"}
    if termination_name not in accepted_termination_names:
        raise RuntimeError(
            "bundle adjustment termination is not successful/converged: "
            f"{termination_name}"
        )
    payload["termination_type"] = termination_name
    payload["num_residuals"] = residual_count
    payload["configured_residual_count"] = configured_residual_count
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.initializer_model, args.native_model, args.ba_model, args.report):
        if path.exists():
            raise FileExistsError(f"preserve existing artifact; choose a new versioned path: {path}")
    graph = json.loads(args.graph_report.read_text(encoding="utf-8"))
    expected_pairs = {
        int(item["pair_id"])
        for item in graph.get("pairs", [])
        if isinstance(item, Mapping) and "pair_id" in item
    }
    if len(expected_pairs) != 1530:
        raise ValueError("graph report must expose exactly 1,530 pair IDs")
    canonical = _canonical_snapshot(args.canonical_manifest, args.database)
    if not canonical["raw_sha_equal"] or canonical["sidecars_present"]["-wal"] or canonical["sidecars_present"]["-shm"]:
        raise RuntimeError("frozen canonical SQLite snapshot is not byte-stable")
    if not canonical["source_database_is_disposable"]:
        raise RuntimeError("the source database must not be the canonical SQLite path")

    v43 = pycolmap.Reconstruction(str(args.v43_model))
    v6 = pycolmap.Reconstruction(str(args.v6_model))
    expected_names = {
        str(name)
        for name, item in load_ring_metadata(args.ring_metadata).items()
        if item.get("selected", True)
    }
    v43_by_name = {str(v43.image(int(i)).name): v43.image(int(i)) for i in v43.reg_image_ids()}
    v6_by_name = {str(v6.image(int(i)).name): v6.image(int(i)) for i in v6.reg_image_ids()}
    if set(v43_by_name) != expected_names or set(v6_by_name) != expected_names:
        raise RuntimeError("v43/v6 initializers do not contain the exact 372-view set")
    orientations = {
        name: np.asarray(image.cam_from_world().rotation.matrix(), dtype=np.float64)
        for name, image in v43_by_name.items()
    }
    initial_centers = {
        name: np.asarray(image.projection_center(), dtype=np.float64)
        for name, image in v6_by_name.items()
    }
    raw_audit = json.loads(args.raw_audit.read_text(encoding="utf-8"))
    classification = json.loads(args.classification.read_text(encoding="utf-8"))
    records = _load_translation_records(raw_audit, classification)
    allowed_pair_ids = {
        int(item["pair_id"])
        for item in classification.get("classification", [])
        if isinstance(item, Mapping)
        and "pair_id" in item
        and bool(item.get("well_conditioned_calibrated"))
    }
    if allowed_pair_ids != expected_pairs:
        raise RuntimeError("classification mapping records do not equal the exact graph pair set")

    with tempfile.TemporaryDirectory(prefix="v4_v50_exact_graph_") as temporary:
        working_db = Path(temporary) / "exact_graph_working.db"
        shutil.copy2(args.database, working_db)
        tvg_pairs, raw_pairs, tvg_rows, match_rows = _pair_ids(working_db)
        preflight = {
            "expected_pair_count": len(expected_pairs),
            "actual_two_view_geometry_count": len(tvg_pairs),
            "actual_raw_match_pair_count": len(raw_pairs),
            "two_view_geometry_pair_ids_exact": tvg_pairs == expected_pairs,
            "raw_match_pair_ids_exact": raw_pairs == expected_pairs,
            "two_view_geometry_inlier_rows": tvg_rows,
            "raw_match_rows": match_rows,
            "source_database_sha256": _sha256(args.database),
            "working_database_sha256_before": _sha256(working_db),
        }
        preflight["passed"] = bool(
            preflight["actual_two_view_geometry_count"] == 1530
            and preflight["actual_raw_match_pair_count"] == 1530
            and preflight["two_view_geometry_pair_ids_exact"]
            and preflight["raw_match_pair_ids_exact"]
        )
        if not preflight["passed"]:
            raise RuntimeError("exact graph preflight failed")
        database = pycolmap.Database.open(str(working_db))
        try:
            translation_edges, translation_recovery = recover_translation_direction_edges(
                database,
                records,
                orientations,
                allowed_pair_ids,
            )
            solved_centers, center_solve = solve_camera_centers(translation_edges, initial_centers)
            class _CenterImage:
                def __init__(self, name: str, center: np.ndarray, rotation: np.ndarray) -> None:
                    self.name = name
                    self._center = np.asarray(center, dtype=np.float64)
                    self._pose = pycolmap.Rigid3d(
                        pycolmap.Rotation3d(np.asarray(rotation, dtype=np.float64)),
                        -np.asarray(rotation, dtype=np.float64) @ self._center,
                    )

                def projection_center(self) -> np.ndarray:
                    return self._center

                def cam_from_world(self) -> pycolmap.Rigid3d:
                    return self._pose

            class _CenterReconstruction:
                def __init__(self, centers: Mapping[str, np.ndarray], rotations: Mapping[str, np.ndarray]) -> None:
                    self._images = {
                        index + 1: _CenterImage(name, center, rotations[name])
                        for index, (name, center) in enumerate(sorted(centers.items()))
                    }

                def reg_image_ids(self) -> list[int]:
                    return list(self._images)

                def image(self, image_id: int) -> _CenterImage:
                    return self._images[int(image_id)]

            translation_gate = sparse_camera_center_translation_evidence(
                _CenterReconstruction(solved_centers, orientations),
                records,
                config=SparseIntegrityConfig(),
            )
            if not translation_gate.get("passed"):
                raise RuntimeError("solved-center calibrated translation gate failed")
            initializer_write = write_consensus_model(
                args.v43_model,
                args.initializer_model,
                orientations,
                camera_centers=solved_centers,
            )
        finally:
            database.close()
        working_db_sha_after = _sha256(working_db)

    initializer = pycolmap.Reconstruction(str(args.initializer_model))
    options = pycolmap.IncrementalPipelineOptions()
    options.extract_colors = False
    options.triangulation.ignore_two_view_tracks = True
    options.triangulation.random_seed = 4201
    options.image_path = args.image_root
    # The exact v43 database is already disposable, but never let the native
    # solver mutate that preserved diagnostic input.  Give it one fresh
    # disposable working copy and bind its before/after bytes in the report.
    native_database_tmp = tempfile.TemporaryDirectory(prefix="v4_v50_native_db_")
    native_database = Path(native_database_tmp.name) / "working.db"
    shutil.copy2(args.database, native_database)
    native_database_sha_before = _sha256(native_database)
    try:
        native = pycolmap.triangulate_points(
            initializer,
            str(native_database),
            str(args.image_root),
            str(args.native_model),
            clear_points=True,
            options=options,
            refine_intrinsics=False,
        )
        native_database_sha_after = _sha256(native_database)
    finally:
        native_database_tmp.cleanup()
    native_summary = _model_summary(native)

    ba_options = pycolmap.BundleAdjustmentOptions()
    ba_options.refine_focal_length = False
    ba_options.refine_principal_point = False
    ba_options.refine_extra_params = False
    ba_options.refine_rig_from_world = True
    ba_options.refine_sensor_from_rig = True
    ba_options.refine_points3D = True
    ba_options.ceres.loss_function_type = pycolmap.LossFunctionType.HUBER
    ba_options.ceres.loss_function_scale = 1.0
    ba_options.ceres.solver_options.max_num_iterations = 100
    ba_options.print_summary = False
    config, anchor_image_id, configured_residual_count = _build_joint_ba_config(native)
    adjuster = pycolmap.create_default_bundle_adjuster(ba_options, config, native)
    ba_summary = adjuster.solve()
    raw_ba_summary = ba_summary.todict() if hasattr(ba_summary, "todict") else {}
    if not isinstance(raw_ba_summary, Mapping):
        raise RuntimeError("pyCOLMAP bundle-adjustment summary is not a mapping")
    ba_summary_payload = _validate_bundle_adjustment_summary(
        raw_ba_summary,
        configured_residual_count=configured_residual_count,
    )
    native.update_point_3d_errors()
    args.ba_model.mkdir(parents=True, exist_ok=False)
    native.write(str(args.ba_model))
    ba_model_summary = _model_summary(native)

    canonical_after = _canonical_snapshot(args.canonical_manifest, args.database)
    return {
        "schema_version": 1,
        "status": "diagnostic_only",
        "promotion_allowed": False,
        "full_sparse_candidate_eligible": False,
        "method": "v50 calibrated translation-direction center solve from v6 initialization + v43 rotation initialization; native pyCOLMAP exact-graph retriangulation + one joint BA",
        "v43_model": str(args.v43_model.resolve()),
        "v43_model_sha256": stable_directory_sha256(args.v43_model),
        "v6_model": str(args.v6_model.resolve()),
        "v6_model_sha256": stable_directory_sha256(args.v6_model),
        "exact_graph_preflight": preflight,
        "translation_direction_recovery": translation_recovery,
        "camera_center_solve": center_solve,
        "translation_gate": translation_gate,
        "pose_initializer": _pose_initializer_evidence(v43, v6, solved_centers, expected_names),
        "initializer_model": str(args.initializer_model.resolve()),
        "initializer_model_sha256": stable_directory_sha256(args.initializer_model),
        "initializer_write": initializer_write,
        "native_retriangulation_model": str(args.native_model.resolve()),
        "native_retriangulation_model_sha256": stable_directory_sha256(args.native_model),
        "native_retriangulation_summary": native_summary,
        "bundle_adjustment": {
            "options": {
                "refine_focal_length": False,
                "refine_principal_point": False,
                "refine_extra_params": False,
                "refine_rig_from_world": True,
                "refine_sensor_from_rig": True,
                "refine_points3D": True,
                "loss_function": "HUBER",
                "loss_function_scale": 1.0,
                "max_num_iterations": 100,
                "fixed_gauge_frame_id": int(native.image(anchor_image_id).frame_id),
                "fixed_intrinsic_camera_ids": [int(camera_id) for camera_id in native.cameras],
                "selected_registered_image_ids": [
                    int(image_id) for image_id in native.reg_image_ids()
                ],
                "configured_residual_count": configured_residual_count,
            },
            "summary": ba_summary_payload,
        },
        "ba_model": str(args.ba_model.resolve()),
        "ba_model_sha256": stable_directory_sha256(args.ba_model),
        "ba_model_summary": ba_model_summary,
        "frozen_canonical_snapshot_before": canonical,
        "frozen_canonical_snapshot_after": canonical_after,
        "working_database_sha256_after": working_db_sha_after,
        "working_database_mutated_by_native_solver": working_db_sha_after != preflight["working_database_sha256_before"],
        "native_working_database_sha256_before": native_database_sha_before,
        "native_working_database_sha256_after": native_database_sha_after,
        "native_working_database_mutated_by_solver": native_database_sha_after != native_database_sha_before,
        "next_step": "Run evaluate_v4_sparse_candidate.py unchanged on the BA model; do not promote or start dense from this diagnostic.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v43-model", type=Path, default=DEFAULT_V43_MODEL)
    parser.add_argument("--v6-model", type=Path, default=DEFAULT_V6_MODEL)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--ring-metadata", type=Path, default=DEFAULT_RING_METADATA)
    parser.add_argument("--raw-audit", type=Path, default=DEFAULT_RAW_AUDIT)
    parser.add_argument("--classification", type=Path, default=DEFAULT_CLASSIFICATION)
    parser.add_argument("--graph-report", type=Path, default=DEFAULT_GRAPH_REPORT)
    parser.add_argument("--canonical-manifest", type=Path, default=DEFAULT_CANONICAL_MANIFEST)
    parser.add_argument("--initializer-model", type=Path, default=DEFAULT_OUTPUT_ROOT / "initializer")
    parser.add_argument("--native-model", type=Path, default=DEFAULT_OUTPUT_ROOT / "native_retriangulation")
    parser.add_argument("--ba-model", type=Path, default=DEFAULT_OUTPUT_ROOT / "joint_ba")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    payload = run(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "translation_passed": payload["translation_gate"].get("passed"),
        "registered_images": payload["ba_model_summary"]["registered_images"],
        "points3D": payload["ba_model_summary"]["points3D"],
        "mean_reprojection_error": payload["ba_model_summary"]["mean_reprojection_error"],
        "ba_model": payload["ba_model"],
        "report": str(args.report.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
