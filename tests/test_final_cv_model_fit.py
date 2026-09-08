from __future__ import annotations

import json
from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest
import cv2
from scipy.spatial.transform import Rotation

import final_cv_model_fit as fit
from final_cv_model_fit import (
    CameraReference,
    FitResult,
    FitObservation,
    FittedComponentProfile,
    ModelAlignment,
)


ROOT = Path(__file__).resolve().parents[1]


def _camera(rotation_xyzw: tuple[float, float, float, float],
            translation: tuple[float, float, float],
            selected_index: int = 1) -> CameraReference:
    return CameraReference(
        selected_index=selected_index,
        filename="synthetic.jpg",
        image_id=1,
        camera_model="SIMPLE_RADIAL",
        camera_params=(500.0, 64.0, 64.0, 0.0),
        cam_from_world_rotation_xyzw=rotation_xyzw,
        cam_from_world_translation=translation,
        image_size=(128, 128),
    )


def test_sweep_camera_center_gauge_normalization_preserves_projection():
    camera = _camera((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
    original = ModelAlignment(
        scale=2.0,
        rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        translation=(0.2, -0.1, 5.0),
    )
    common_scale = 3.0
    normalized = ModelAlignment(
        scale=common_scale,
        rotation_matrix=original.rotation_matrix,
        translation=original.translation,
    )
    local_points = np.asarray(
        ((0.0, 0.0, 0.0), (0.1, -0.08, 0.4), (-0.12, 0.05, 1.0)),
        dtype=np.float64,
    )

    original_projection = fit.project_world_points(
        camera, fit._model_to_world(original, local_points)
    )
    derived_camera = fit._scale_camera_center_about_anchor(
        camera,
        original.translation,
        common_scale / original.scale,
    )
    normalized_projection = fit.project_world_points(
        derived_camera, fit._model_to_world(normalized, local_points)
    )

    assert original_projection is not None and normalized_projection is not None
    assert normalized_projection == pytest.approx(original_projection, abs=1e-9)
    assert camera.cam_from_world_translation == (0.0, 0.0, 0.0)
    assert derived_camera.cam_from_world_rotation_xyzw == camera.cam_from_world_rotation_xyzw
    assert derived_camera.camera_params == camera.camera_params


def _synthetic_cameras() -> tuple[CameraReference, CameraReference]:
    first = Rotation.from_matrix(
        np.array(((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
    ).as_quat()
    second = Rotation.from_matrix(
        np.array(((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)))
    ).as_quat()
    return (
        _camera(tuple(float(value) for value in first), (0.0, 0.0, 10.0)),
        _camera(tuple(float(value) for value in second), (0.0, 0.0, 20.0), 2),
    )


def _observation(
    camera: CameraReference,
    selected_index: int,
    mask: np.ndarray,
) -> FitObservation:
    top = fit.project_world_point(camera, np.array((0.0, 0.0, 1.0)))
    bottom = fit.project_world_point(camera, np.array((0.0, 0.0, 0.0)))
    assert top is not None and bottom is not None
    return FitObservation(
        selected_index=selected_index,
        filename=f"synthetic_{selected_index:03d}.jpg",
        view_category="normal_side",
        mask=mask,
        axis_top_xy=(float(top[0]), float(top[1])),
        axis_bottom_xy=(float(bottom[0]), float(bottom[1])),
        landmarks=(),
        source_path=Path("synthetic_source.jpg"),
        reviewed_mask_path=Path("synthetic_mask.png"),
    )


def test_step13_cameras_match_authoritative_report_and_manifest():
    cameras = fit.load_step13_cameras(ROOT)

    assert len(cameras) == 266
    assert all(camera.camera_model == "SIMPLE_RADIAL" for camera in cameras)
    assert all(camera.image_size == (3072, 4080) for camera in cameras)
    assert len({camera.camera_params for camera in cameras}) == 1
    assert len({camera.selected_index for camera in cameras}) == 266


def test_step13_loader_rejects_model_that_contradicts_authoritative_report(
    monkeypatch, tmp_path
):
    summary_path = tmp_path / "reports" / "step13_summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps(
            {
                "attempt_result": {
                    "attempt": {
                        "best_model": {
                            "camera_count": 1,
                            "camera_model": "SIMPLE_RADIAL",
                            "camera_params": [250.0, 32.0, 32.0, 0.0],
                            "registered_images": 2,
                            "registered_image_names": ["one.jpg", "two.jpg"],
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    class Image:
        def __init__(self, image_id: int, name: str, camera_id: int):
            self.image_id = image_id
            self.name = name
            self.camera_id = camera_id
            self.cam_from_world = lambda: type(
                "Rigid",
                (),
                {
                    "rotation": type(
                        "Rotation", (), {"quat": np.array((0.0, 0.0, 0.0, 1.0))}
                    )(),
                    "translation": np.array((0.0, 0.0, 5.0)),
                },
            )()
            self.camera = type(
                "Camera",
                (),
                {
                    "model_name": "SIMPLE_RADIAL",
                    "params": np.array((250.0, 32.0, 32.0, 0.0)),
                    "width": 64,
                    "height": 64,
                },
            )()

    class Reconstruction:
        cameras = {1: object()}

        def reg_image_ids(self):
            return (1,)

        def image(self, image_id):
            return Image(image_id, "one.jpg", 1)

    selected_record = type(
        "Selected",
        (),
        {"index": 1, "filename": "one.jpg", "width": 64, "height": 64},
    )()
    monkeypatch.setattr(
        fit, "load_selected_manifest", lambda path: (selected_record,)
    )
    monkeypatch.setattr(
        fit.pycolmap, "Reconstruction", lambda path: Reconstruction()
    )

    with pytest.raises(ValueError, match="authoritative Step 13 report"):
        fit.load_step13_cameras(
            tmp_path,
            summary_path=summary_path,
        )


def test_simple_radial_projection_uses_pycolmap_distortion():
    camera = _camera((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0))

    center = fit.project_world_point(camera, np.array((0.0, 0.0, 5.0)))
    behind = fit.project_world_point(camera, np.array((0.0, 0.0, -5.0)))
    distorted = fit.project_world_point(
        _camera_with_distortion(camera, 0.1),
        np.array((0.2, 0.0, 5.0)),
    )

    assert center is not None
    np.testing.assert_allclose(center, (64.0, 64.0), atol=1e-10)
    assert behind is None
    assert distorted is not None
    assert distorted[0] > 64.0 + 10.0


def _camera_with_distortion(camera: CameraReference, k: float) -> CameraReference:
    return CameraReference(
        selected_index=camera.selected_index,
        filename=camera.filename,
        image_id=camera.image_id,
        camera_model=camera.camera_model,
        camera_params=(*camera.camera_params[:3], k),
        cam_from_world_rotation_xyzw=camera.cam_from_world_rotation_xyzw,
        cam_from_world_translation=camera.cam_from_world_translation,
        image_size=camera.image_size,
    )


def test_axis_alignment_triangulates_two_camera_rays():
    cameras = _synthetic_cameras()
    observations = tuple(
        _observation(camera, index, np.zeros((128, 128), dtype=np.uint8))
        for index, camera in enumerate(cameras, start=1)
    )

    alignment = fit.fit_model_alignment(observations, cameras)

    assert alignment.scale == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(alignment.translation, (0.0, 0.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(
        alignment.rotation_matrix,
        np.identity(3),
        atol=1e-6,
    )


def test_known_radial_profile_is_recovered_from_two_projected_silhouettes():
    cameras = _synthetic_cameras()
    levels = np.linspace(0.0, 1.0, 33)
    radii = 0.08 + 0.14 * np.sin(np.pi * levels)
    known_sections = tuple(
        (float(level), float(radius)) for level, radius in zip(levels, radii, strict=True)
    )
    known_profile = FittedComponentProfile(
        name="outer",
        sections=known_sections,
        measurement_status="synthetic",
    )
    identity_alignment = ModelAlignment(
        scale=1.0,
        rotation_matrix=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        translation=(0.0, 0.0, 0.0),
    )
    observations = tuple(
        _observation(
            camera,
            index,
            fit.render_profile_silhouette(
                known_profile.sections,
                camera,
                identity_alignment,
                camera.image_size,
            ),
        )
        for index, camera in enumerate(cameras, start=1)
    )

    fitted = fit.fit_profiles_from_observations(
        observations,
        cameras,
        identity_alignment,
        control_levels=levels,
    )

    fitted_outer = fitted["outer"]
    assert fitted_outer.measurement_status == "projected_silhouette_fit"
    recovered = np.asarray([radius for _, radius in fitted_outer.sections])
    np.testing.assert_allclose(recovered, radii, atol=0.016)
    assert np.all(np.diff([level for level, _ in fitted_outer.sections]) > 0.0)


def test_profile_fit_uses_width_perpendicular_to_projected_axis():
    base_cameras = _synthetic_cameras()
    rolled_cameras = []
    for camera, roll_degrees in zip(base_cameras, (24.0, -19.0), strict=True):
        base_rotation = Rotation.from_quat(camera.cam_from_world_rotation_xyzw).as_matrix()
        roll = Rotation.from_euler("z", roll_degrees, degrees=True).as_matrix()
        rolled_rotation = Rotation.from_matrix(roll @ base_rotation).as_quat()
        rolled_cameras.append(
            CameraReference(
                selected_index=camera.selected_index,
                filename=camera.filename,
                image_id=camera.image_id,
                camera_model=camera.camera_model,
                camera_params=camera.camera_params,
                cam_from_world_rotation_xyzw=tuple(float(value) for value in rolled_rotation),
                cam_from_world_translation=camera.cam_from_world_translation,
                image_size=camera.image_size,
            )
        )

    levels = np.linspace(0.0, 1.0, 17)
    radii = 0.07 + 0.12 * np.sin(np.pi * levels)
    sections = tuple(
        (float(level), float(radius)) for level, radius in zip(levels, radii, strict=True)
    )
    alignment = ModelAlignment(
        scale=1.0,
        rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        translation=(0.0, 0.0, 0.0),
    )
    observations = tuple(
        _observation(
            camera,
            camera.selected_index,
            fit.render_profile_silhouette(sections, camera, alignment, camera.image_size),
        )
        for camera in rolled_cameras
    )

    fitted = fit.fit_profiles_from_observations(
        observations,
        tuple(rolled_cameras),
        alignment,
        control_levels=levels,
    )

    recovered = np.asarray([radius for _, radius in fitted["outer"].sections])
    np.testing.assert_allclose(recovered, radii, atol=0.02)


def test_mask_lateral_endpoints_follow_projected_axis_normal():
    size = 201
    center = np.asarray((100.0, 100.0))
    angle = np.deg2rad(28.0)
    axis = np.asarray((np.sin(angle), np.cos(angle)))
    lateral = np.asarray((axis[1], -axis[0]))
    yy, xx = np.mgrid[:size, :size]
    points = np.stack((xx, yy), axis=-1).astype(np.float64) - center
    axial = points @ axis
    side = points @ lateral
    mask = np.where((np.abs(axial) <= 70.0) & (np.abs(side) <= 24.0), 255, 0).astype(np.uint8)

    endpoints = fit._mask_lateral_endpoints(mask, tuple(center), axis, lateral)

    assert endpoints is not None
    left, right = endpoints
    assert left[0] < right[0]
    assert abs(left[1] - right[1]) > 10.0
    assert abs(np.linalg.norm(right - left) - 48.0) < 3.0


def test_observed_landmark_span_basis_interpolates_component_height():
    observation = FitObservation(
        selected_index=1,
        filename="synthetic.jpg",
        view_category="normal_side",
        mask=np.zeros((32, 32), dtype=np.uint8),
        axis_top_xy=(16.0, 2.0),
        axis_bottom_xy=(16.0, 30.0),
        landmarks=(
            {"name": "finial_top", "x": 18.0, "y": 4.0, "confidence": "high"},
            {"name": "finial_bottom", "x": 14.0, "y": 20.0, "confidence": "high"},
        ),
        source_path=Path("synthetic.jpg"),
        reviewed_mask_path=Path("synthetic_mask.png"),
    )

    basis = fit._observed_landmark_span_basis(
        observation, "finial_top", "finial_bottom", 0.25
    )

    assert basis is not None
    np.testing.assert_allclose(basis[0], (15.0, 16.0), atol=1e-12)
    assert np.linalg.norm(basis[1]) == pytest.approx(1.0)
    assert np.dot(basis[1], basis[2]) == pytest.approx(0.0, abs=1e-12)


def test_rendered_profile_silhouette_is_a_solid_envelope():
    camera = _synthetic_cameras()[0]
    alignment = ModelAlignment(
        scale=1.0,
        rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        translation=(0.0, 0.0, 0.0),
    )

    silhouette = fit.render_profile_silhouette(
        ((0.0, 0.1), (0.3, 0.2), (0.5, 0.05), (0.7, 0.2), (1.0, 0.05)),
        camera,
        alignment,
        camera.image_size,
    )

    assert silhouette[64, 64] == 255
    foreground = silhouette > 0
    for row in foreground[np.any(foreground, axis=1)]:
        columns = np.flatnonzero(row)
        assert np.all(row[columns[0] : columns[-1] + 1])


def test_axial_camera_sees_end_cap_not_empty_axis_envelope():
    """A top camera still sees a disk when both axis endpoints project together."""
    camera = _camera((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 5.0))
    camera = replace(camera, camera_params=(2000.0, 256.0, 256.0, 0.0), image_size=(512, 512))
    alignment = ModelAlignment(1.0, tuple(map(tuple, np.eye(3))), (0.0, 0.0, 0.0))
    predicted = fit.render_profile_silhouette(
        ((0.0, 0.2), (1.0, 0.2)), camera, alignment, camera.image_size,
        angular_samples=128,
    )
    expected = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(expected, (256, 256), 80, 255, -1)
    assert fit._iou(expected, predicted) >= 0.96
    assert predicted[256, 256] == 255


def test_elevated_cylinder_includes_full_projected_rings():
    """Independent convex cylinder projection catches missing upper/lower arcs."""
    rotation = Rotation.from_euler("x", 55, degrees=True).as_quat()
    camera = _camera(tuple(rotation), (0.0, 0.0, 5.0))
    alignment = ModelAlignment(1.0, tuple(map(tuple, np.eye(3))), (0.0, 0.0, 0.0))
    angles = np.linspace(0, 2 * np.pi, 512, endpoint=False)
    vertices = np.concatenate([
        np.column_stack((0.25*np.cos(angles), 0.25*np.sin(angles), np.full(512, z)))
        for z in (0.0, 1.0)
    ])
    projected = fit.project_world_points(camera, vertices)
    expected = np.zeros((128, 128), dtype=np.uint8)
    cv2.fillConvexPoly(expected, cv2.convexHull(np.rint(projected).astype(np.int32)), 255)
    predicted = fit.render_profile_silhouette(
        ((0.0, 0.25), (1.0, 0.25)), camera, alignment, camera.image_size,
        angular_samples=128,
    )
    assert fit._iou(expected, predicted) >= 0.985


def test_tilted_profile_endpoint_uses_ring_extreme_not_axis_center():
    rotation = Rotation.from_euler("x", 55, degrees=True).as_quat()
    camera = _camera(tuple(rotation), (0.0, 0.0, 5.0))
    alignment = ModelAlignment(
        1.0,
        tuple(map(tuple, np.eye(3))),
        (0.0, 0.0, 0.0),
    )
    radius = 0.25
    endpoint = fit._projected_profile_axial_endpoint(
        camera,
        alignment,
        0.0,
        radius,
        toward_top=False,
        angular_samples=256,
    )
    basis = fit._projected_axis_basis(camera, alignment, 0.0)
    center = fit.project_world_point(camera, np.asarray((0.0, 0.0, 0.0)))
    ring = fit.project_world_points(
        camera,
        fit._circle_world_points(radius, 0.0, angular_samples=256),
    )

    assert endpoint is not None and basis is not None and center is not None
    assert ring is not None
    offsets = (ring - np.asarray(basis[0])) @ basis[1]
    expected = ring[int(np.argmin(offsets))]
    np.testing.assert_allclose(endpoint, expected, atol=1e-9)
    assert np.linalg.norm(np.asarray(endpoint) - np.asarray(center)) > 1.0


def test_tilted_profile_lateral_landmarks_use_ring_tangencies():
    rotation = Rotation.from_euler("x", 55, degrees=True).as_quat()
    camera = _camera(tuple(rotation), (0.0, 0.0, 5.0))
    alignment = ModelAlignment(
        1.0,
        tuple(map(tuple, np.eye(3))),
        (0.0, 0.0, 0.0),
    )
    endpoints = fit._projected_profile_lateral_endpoints(
        camera,
        alignment,
        0.4,
        0.25,
        angular_samples=256,
    )
    center = fit.project_world_point(camera, np.asarray((0.0, 0.0, 0.4)))

    assert endpoints is not None and center is not None
    assert endpoints[0][0] < endpoints[1][0]
    midpoint = (endpoints[0] + endpoints[1]) * 0.5
    assert abs(float(midpoint[1] - center[1])) > 0.25


def test_globe_landmarks_use_globe_profile_not_overlapping_whole_assembly():
    camera = _synthetic_cameras()[0]
    alignment = ModelAlignment(
        scale=1.0,
        rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        translation=(0.0, 0.0, 0.0),
    )
    globe = FittedComponentProfile(
        name="globe",
        sections=((0.35, 0.10), (0.50, 0.10), (0.65, 0.10)),
        measurement_status="synthetic",
    )
    overlapping_bowl = FittedComponentProfile(
        name="bowl_outer",
        sections=((0.35, 0.28), (0.50, 0.28), (0.65, 0.28)),
        measurement_status="synthetic",
    )
    globe_mask = fit.render_profile_silhouette(
        globe.sections, camera, alignment, camera.image_size
    )
    whole_mask = cv2.bitwise_or(
        globe_mask,
        fit.render_profile_silhouette(
            overlapping_bowl.sections, camera, alignment, camera.image_size
        ),
    )
    basis = fit._projected_axis_basis(camera, alignment, 0.50)
    assert basis is not None
    endpoints = fit._mask_lateral_endpoints(globe_mask, *basis)
    assert endpoints is not None
    observation = FitObservation(
        selected_index=1,
        filename="synthetic.jpg",
        view_category="normal_side",
        mask=whole_mask,
        axis_top_xy=(64.0, 39.0),
        axis_bottom_xy=(64.0, 89.0),
        landmarks=(
            {
                "name": "globe_max_left",
                "x": float(endpoints[0][0]),
                "y": float(endpoints[0][1]),
                "confidence": "high",
            },
            {
                "name": "globe_max_right",
                "x": float(endpoints[1][0]),
                "y": float(endpoints[1][1]),
                "confidence": "high",
            },
        ),
        source_path=Path("synthetic.jpg"),
        reviewed_mask_path=Path("synthetic_mask.png"),
    )

    records = fit._landmark_records(
        observation,
        camera,
        alignment,
        whole_mask,
        {"globe_max": 0.50},
        {"globe": globe},
    )

    assert max(record.error_fraction for record in records) <= 1e-9


def test_metric_gate_and_visual_veto_are_fail_closed():
    passing = tuple(
        fit.ViewFitMetrics(
            selected_index=index,
            filename=f"view_{index}.jpg",
            view_category="normal_side",
            silhouette_iou=0.91,
            landmark_median_error_fraction=0.010,
            landmark_max_error_fraction=0.020,
            component_ious={},
            camera_source="step13",
            fit_included=True,
            inclusion_reason="reliable_side_view",
        )
        for index in range(12)
    )
    numeric_pass = fit.fit_metric_gate(passing)
    accepted = fit.acceptance_decision(
        numeric_pass,
        {
            "bowl": "pass",
            "globe": "pass",
            "neck": "pass",
            "lid": "pass",
            "finial": "pass",
        },
    )
    vetoed = fit.acceptance_decision(
        numeric_pass,
        {
            "bowl": "pass",
            "globe": "fail",
            "neck": "pass",
            "lid": "pass",
            "finial": "pass",
        },
    )
    failing = passing[:-1] + (
        fit.ViewFitMetrics(
            selected_index=12,
            filename="view_12.jpg",
            view_category="normal_side",
            silhouette_iou=0.70,
            landmark_median_error_fraction=0.010,
            landmark_max_error_fraction=0.020,
            component_ious={},
            camera_source="step13",
            fit_included=True,
            inclusion_reason="reliable_side_view",
        ),
    )

    assert numeric_pass is True
    assert accepted is True
    assert vetoed is False
    assert fit.fit_metric_gate(failing) is False
    assert fit.acceptance_decision(False, {}) is False


def test_reliable_capture_sweeps_share_one_physical_scale():
    names = tuple(fit.CAPTURE_SWEEPS)
    pose_values = []
    for index, _name in enumerate(names):
        pose_values.extend((float(index), 0.0, 0.0, 0.0, 0.0, float(index + 1)))
    parameters = np.asarray(
        (*pose_values, np.log(3.25), np.log(5.75)), dtype=np.float64
    )

    poses = fit._unpack_shared_scale_sweep_poses(parameters, names)

    reliable_scales = [poses[name].scale for name in fit.RELIABLE_CAPTURE_SWEEPS]
    assert reliable_scales == pytest.approx([3.25, 3.25, 3.25])
    assert poses[fit.TOP_CAPTURE_SWEEP].scale == pytest.approx(5.75)


def test_write_fit_outputs_hashes_the_profiles_written_in_the_same_run(tmp_path: Path):
    profiles = tuple(
        FittedComponentProfile(
            name=name,
            sections=((0.0, 0.1), (1.0, 0.1)),
            measurement_status="synthetic",
        )
        for name in (*fit.REQUIRED_PROFILES, "outer")
    )
    result = FitResult(
        alignment=ModelAlignment(
            scale=1.0,
            rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            translation=(0.0, 0.0, 0.0),
        ),
        profiles=profiles,
        metrics=(),
        metrics_passed=False,
        visual_component_review={name: "pending" for name in fit.VISUAL_REVIEW_COMPONENTS},
        accepted=False,
        aggregate_metrics={},
    )

    report_path, profiles_path = fit.write_fit_outputs(tmp_path, result)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["artifact_hashes"]["final_profiles_sha256"] == fit.sha256_file(profiles_path)
