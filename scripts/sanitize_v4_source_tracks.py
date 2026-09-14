"""Sanitize an image-derived sparse model's track partition.

This is a bounded architecture diagnostic for the V4 sparse repair.  It keeps
the source model's camera-center gauge and multi-view observations, removes
duplicate-image track members deterministically, verifies that each retained
track is connected by the disposable calibrated match graph, and retriangulates
the retained observations under the calibrated source-component pose repair.
The frozen SQLite snapshot is never opened directly.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import pycolmap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_v4_rotation_consensus import (  # noqa: E402
    build_rotation_consensus,
    build_source_component_pose_repair,
)
from v4_repair import (  # noqa: E402
    create_disposable_sqlite_from_manifest,
    decode_colmap_pair_id,
    stable_directory_sha256,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_orientations(source_model: Path) -> dict[str, np.ndarray]:
    reconstruction = pycolmap.Reconstruction(str(source_model))
    return {
        str(image.name): np.asarray(image.cam_from_world().rotation.matrix(), dtype=np.float64)
        for image in reconstruction.images.values()
        if image.has_pose
    }


def _verified_match_lookup(database: Any, allowed_pair_ids: set[int]) -> dict[tuple[int, int], set[tuple[int, int]]]:
    lookup: dict[tuple[int, int], set[tuple[int, int]]] = {}
    pair_ids, geometries = database.read_two_view_geometries()
    for pair_id, geometry in zip(pair_ids, geometries):
        pair_value = int(pair_id)
        if pair_value not in allowed_pair_ids:
            continue
        first_id, second_id = decode_colmap_pair_id(pair_value)
        matches = np.asarray(getattr(geometry, "inlier_matches", np.empty((0, 2), dtype=np.uint32)))
        if matches.ndim != 2 or matches.shape[1] != 2:
            continue
        lookup[(first_id, second_id)] = {
            (int(first), int(second)) for first, second in matches.astype(np.int64, copy=False).tolist()
        }
    return lookup


def _track_is_graph_connected(
    observations: list[tuple[int, int]],
    verified_matches: dict[tuple[int, int], set[tuple[int, int]]],
) -> tuple[bool, int]:
    parent = {image_id: image_id for image_id, _ in observations}

    def find(image_id: int) -> int:
        while parent[image_id] != image_id:
            parent[image_id] = parent[parent[image_id]]
            image_id = parent[image_id]
        return image_id

    def union(first_id: int, second_id: int) -> None:
        first_root, second_root = find(first_id), find(second_id)
        if first_root != second_root:
            parent[second_root] = first_root

    support = 0
    for index, (first_id, first_index) in enumerate(observations):
        for second_id, second_index in observations[index + 1 :]:
            key = (first_id, second_id) if first_id < second_id else (second_id, first_id)
            matches = verified_matches.get(key)
            if not matches:
                continue
            correspondence = (
                (first_index, second_index)
                if first_id < second_id
                else (second_index, first_index)
            )
            if correspondence in matches:
                union(first_id, second_id)
                support += 1
    return len({find(image_id) for image_id, _ in observations}) == 1, support


def sanitize_source_tracks(
    source_model: Path,
    classification_path: Path,
    snapshot_manifest: Path,
    output_model: Path,
    output_report: Path,
    *,
    preserve_source_points: bool = False,
    preserve_source_pose_translation: bool = False,
) -> dict[str, Any]:
    if output_model.exists():
        raise FileExistsError(output_model)
    classification_payload = json.loads(classification_path.read_text(encoding="utf-8"))
    records = classification_payload.get("classification", classification_payload)
    if not isinstance(records, list) or not records:
        raise ValueError("classification payload must contain a non-empty list")

    source = pycolmap.Reconstruction(str(source_model))
    source_orientations = _source_orientations(source_model)
    corrected_orientations, pose_report = build_source_component_pose_repair(records, source_orientations)
    _, mapping_report = build_rotation_consensus(records)
    allowed_pair_ids = {
        int(item["pair_id"])
        for item in mapping_report["per_edge"]
        if item.get("mapping_status") in {"retained_rotation_synchronization", "retained_robust_consensus_inlier"}
    }

    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "v6 source multi-view track sanitization with calibrated component pose repair",
        "source_model": str(source_model.resolve()),
        "source_model_sha256": stable_directory_sha256(source_model),
        "classification_path": str(classification_path.resolve()),
        "classification_sha256": _sha256(classification_path),
        "snapshot_manifest_path": str(snapshot_manifest.resolve()),
        "snapshot_manifest_sha256": _sha256(snapshot_manifest),
        "allowed_pair_count": len(allowed_pair_ids),
        "track_provenance": {
            "source": "verified_disposable_sqlite_graph",
            "independent_graph": True,
            "independent_graph_sha256": _sha256(classification_path),
            "constructed_pairwise_tracks": False,
            "track_builder": "sanitized_source_multiview_tracks",
            "pose_repair": "source_gauge_component_pose_repair",
            "point_geometry": "source_points_preserved" if preserve_source_points else "retriangulated_under_repaired_poses",
            "pose_translation": "source_pose_translation" if preserve_source_pose_translation else "source_camera_center_preserved",
        },
        "pose_repair": pose_report,
        "mapping_selection": mapping_report,
    }

    with tempfile.TemporaryDirectory(prefix="v4_source_track_sanitize_db_") as temporary:
        working_path = Path(temporary) / "working.db"
        disposable = create_disposable_sqlite_from_manifest(snapshot_manifest, working_path)
        report["working_database_sha256"] = disposable["working"]["sha256"]
        database = pycolmap.Database.open(str(working_path))
        try:
            verified_matches = _verified_match_lookup(database, allowed_pair_ids)
            new = pycolmap.Reconstruction()
            for camera in source.cameras.values():
                new.add_camera_with_trivial_rig(pycolmap.Camera(camera.todict()))
            for image_id in sorted(source.reg_image_ids()):
                image = source.image(image_id)
                keypoints = np.asarray([point.xy for point in image.points2D], dtype=np.float64).reshape(-1, 2)
                rotation = corrected_orientations.get(
                    str(image.name), np.asarray(image.cam_from_world().rotation.matrix(), dtype=np.float64)
                )
                center = np.asarray(image.projection_center(), dtype=np.float64).reshape(3)
                translation = (
                    np.asarray(image.cam_from_world().translation, dtype=np.float64).reshape(3)
                    if preserve_source_pose_translation
                    else -rotation @ center
                )
                new.add_image_with_trivial_frame(
                    pycolmap.Image(
                        name=image.name,
                        keypoints=keypoints,
                        camera_id=image.camera_id,
                        image_id=image.image_id,
                    ),
                    pycolmap.Rigid3d(
                        pycolmap.Rotation3d(rotation),
                        translation,
                    ),
                )

            candidates: list[tuple[int, float, list[Any], np.ndarray, np.ndarray, int]] = []
            rejected = Counter()
            for point in source.points3D.values():
                xyz = np.asarray(point.xyz, dtype=np.float64).reshape(3)
                if not np.isfinite(xyz).all():
                    rejected["non_finite_source_point"] += 1
                    continue
                best_by_image: dict[int, tuple[tuple[float, int], int]] = {}
                for element in point.track.elements:
                    image_id, point_index = int(element.image_id), int(element.point2D_idx)
                    image = source.image(image_id)
                    observed = np.asarray(image.point2D(point_index).xy, dtype=np.float64)
                    projected = image.project_point(xyz)
                    error = (
                        float(np.linalg.norm(np.asarray(projected) - observed))
                        if projected is not None and np.isfinite(projected).all()
                        else float("inf")
                    )
                    candidate = ((error, point_index), point_index)
                    if image_id not in best_by_image or candidate[0] < best_by_image[image_id][0]:
                        best_by_image[image_id] = candidate
                observations = [
                    (image_id, selected[1]) for image_id, selected in sorted(best_by_image.items())
                ]
                if len(observations) < 3:
                    rejected["duplicate_image_sanitization_short"] += 1
                    continue
                connected, graph_support = _track_is_graph_connected(observations, verified_matches)
                if not connected:
                    rejected["verified_graph_disconnected"] += 1
                    continue
                elements = [pycolmap.TrackElement(image_id, point_index) for image_id, point_index in observations]
                candidates.append(
                    (
                        len(elements),
                        float(getattr(point, "error", 0.0)),
                        elements,
                        xyz,
                        np.asarray(point.color, dtype=np.uint8),
                        graph_support,
                    )
                )

            candidates.sort(
                key=lambda item: (
                    -item[0],
                    item[1],
                    [(int(element.image_id), int(element.point2D_idx)) for element in item[2]],
                )
            )
            cameras = {int(camera_id): new.camera(int(camera_id)) for camera_id in new.cameras}
            poses = {int(image_id): new.image(int(image_id)).cam_from_world() for image_id in new.reg_image_ids()}
            used_observations: set[tuple[int, int]] = set()
            lengths: list[int] = []
            graph_supported_tracks = 0
            added = 0
            for _, _, elements, source_xyz, color, graph_support in candidates:
                elements = [
                    element
                    for element in elements
                    if (int(element.image_id), int(element.point2D_idx)) not in used_observations
                ]
                if len(elements) < 3:
                    rejected["observation_reuse_or_short"] += 1
                    continue
                matrices: list[np.ndarray] = []
                rays: list[np.ndarray] = []
                observations: list[np.ndarray] = []
                for element in elements:
                    image = source.image(int(element.image_id))
                    observed = np.asarray(image.point2D(int(element.point2D_idx)).xy, dtype=np.float64)
                    pose = poses[int(element.image_id)]
                    matrices.append(
                        np.hstack(
                            (
                                np.asarray(pose.rotation.matrix(), dtype=np.float64),
                                np.asarray(pose.translation, dtype=np.float64).reshape(3, 1),
                            )
                        )
                    )
                    rays.append(
                        np.asarray(cameras[image.camera_id].cam_ray_from_img(observed), dtype=np.float64).reshape(3)
                    )
                    observations.append(observed)
                if preserve_source_points:
                    xyz = np.asarray(source_xyz, dtype=np.float64).reshape(3)
                else:
                    try:
                        xyz = pycolmap.triangulate_multi_view_point(matrices, np.asarray(rays, dtype=np.float64))
                    except Exception:
                        xyz = None
                if xyz is None or not np.isfinite(xyz).all():
                    rejected["triangulation_or_cheirality"] += 1
                    continue
                residuals: list[float] = []
                valid = True
                for element, observed in zip(elements, observations):
                    pose = poses[int(element.image_id)]
                    camera_point = (
                        np.asarray(pose.rotation.matrix(), dtype=np.float64) @ np.asarray(xyz, dtype=np.float64)
                        + np.asarray(pose.translation, dtype=np.float64)
                    )
                    projected = new.image(int(element.image_id)).project_point(xyz)
                    if camera_point[2] <= 0.0 or projected is None or not np.isfinite(projected).all():
                        valid = False
                        break
                    residuals.append(float(np.linalg.norm(np.asarray(projected) - observed)))
                if not valid or max(residuals) > 12.0 or float(np.median(residuals)) > 6.0:
                    rejected["reprojection"] += 1
                    continue
                try:
                    new.add_point3D(np.asarray(xyz, dtype=np.float64), pycolmap.Track(elements), color)
                except Exception:
                    rejected["point_observation_conflict"] += 1
                    continue
                used_observations.update((int(element.image_id), int(element.point2D_idx)) for element in elements)
                lengths.append(len(elements))
                graph_supported_tracks += int(graph_support > 0)
                added += 1
            new.update_point_3d_errors()
            output_model.mkdir(parents=True, exist_ok=False)
            new.write(str(output_model))
            report["model_write"] = {
                "registered_images": int(new.num_reg_images()),
                "points3D": int(new.num_points3D()),
                "observations": int(new.compute_num_observations()),
                "mean_reprojection_error": float(new.compute_mean_reprojection_error()),
                "mean_track_length": float(new.compute_mean_track_length()),
                "added_points": added,
                "candidate_source_points": len(candidates),
                "graph_supported_track_count": graph_supported_tracks,
                "track_length_ge3_fraction": float(np.mean(np.asarray(lengths) >= 3)) if lengths else 0.0,
                "rejected_tracks": dict(rejected),
            }
        finally:
            database.close()

    report["output_model_sha256"] = stable_directory_sha256(output_model)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--classification", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--output-model", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument(
        "--preserve-source-points",
        action="store_true",
        help="retain each source point coordinate while sanitizing its observation track",
    )
    parser.add_argument(
        "--preserve-source-pose-translation",
        action="store_true",
        help="retain the source pose translation while repairing rotations",
    )
    args = parser.parse_args()
    report = sanitize_source_tracks(
        args.source_model,
        args.classification,
        args.snapshot_manifest,
        args.output_model,
        args.output_report,
        preserve_source_points=args.preserve_source_points,
        preserve_source_pose_translation=args.preserve_source_pose_translation,
    )
    print(json.dumps(report.get("model_write", {}), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
