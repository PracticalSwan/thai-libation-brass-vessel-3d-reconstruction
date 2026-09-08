"""Focused contracts for deterministic multi-view texture projection."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from final_texture_projection import (
    CameraReference,
    CoverageSummary,
    ProjectionConfig,
    ProjectionGeometry,
    ProjectionView,
    OverlapObservation,
    assemble_final_texture_set,
    apply_photometric_correction,
    barycentric_coordinates,
    build_texture_projection_report,
    coverage_gap_plan,
    coverage_summary,
    derive_roughness_and_metallic,
    derive_wear_mask,
    estimate_photometric_harmonization,
    fuse_texel_samples,
    inference_region_mask,
    load_projection_package,
    project_simple_radial,
    rasterize_uv_triangles,
    robust_fuse_colors,
    select_projection_views,
    apply_conservative_gap_fill,
    view_weight,
)


IDENTITY_CAMERA = tuple(float(value) for value in np.eye(4).reshape(-1))


def _camera(selected_index: int = 1, *, k: float = 0.0, translation=(0.0, 0.0, -5.0)):
    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = translation
    return CameraReference(
        camera_id=selected_index + 10,
        selected_index=selected_index,
        width=16,
        height=16,
        focal_length=8.0,
        principal_x=8.0,
        principal_y=8.0,
        radial_distortion=k,
        world_to_camera=tuple(float(value) for value in matrix.reshape(-1)),
    )


def _view(index: int, *, quality="normal", registered=True, camera=None):
    return ProjectionView(
        selected_index=index,
        image_path=Path(f"IMG_{index:03}.jpg"),
        whole_mask_path=Path(f"MASK_{index:03}.png"),
        camera=camera if camera is not None else (_camera(index) if registered else None),
        quality_condition=quality,
        source_sha256=f"{index:064x}",
        view_category="normal_side" if index % 2 == 0 else "elevated_oblique",
        mask_source="reviewed" if index == 72 else "reconstruction",
        step13_registered=registered,
    )


def test_view_weight_rejects_saturation_and_prefers_front_facing_samples():
    assert (
        view_weight(
            cos_view_angle=0.9,
            mask_confidence=1.0,
            luminance=0.995,
            saturation=0.05,
            visible=True,
        )
        == 0.0
    )
    front = view_weight(0.95, 1.0, 0.55, 0.7, True)
    grazing = view_weight(0.25, 1.0, 0.55, 0.7, True)

    assert front > grazing > 0
    assert view_weight(0.9, 1.0, 0.55, 0.7, False) == 0.0


def test_view_weight_rejects_impossible_evidence():
    with pytest.raises(ValueError, match="cos_view_angle"):
        view_weight(1.1, 1.0, 0.5, 0.5, True)
    with pytest.raises(ValueError, match="mask_confidence"):
        view_weight(0.5, 1.1, 0.5, 0.5, True)


def test_barycentric_interpolation_and_deterministic_uv_rasterization():
    triangle_uv = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    values = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    weights = barycentric_coordinates((0.25, 0.25), triangle_uv)

    assert np.allclose(weights, (0.5, 0.25, 0.25))
    assert np.allclose(weights @ values, (0.5, 1.0, 0.0))

    geometry = ProjectionGeometry(
        vertices=np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        loop_uvs=triangle_uv,
        loop_normals=np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ),
        material_ids=np.asarray([0], dtype=np.int32),
    )
    raster = rasterize_uv_triangles(geometry, texture_size=4)

    assert raster.triangle_index[3, 0] == 0
    assert np.allclose(raster.barycentric[3, 0], (0.75, 0.125, 0.125))
    assert raster.triangle_index[0, 3] == -1


def test_simple_radial_projection_applies_distortion_after_perspective_divide():
    camera = _camera(k=-0.5)

    undistorted_x, undistorted_y, undistorted_depth = project_simple_radial(
        camera, (0.0, 0.0, 6.0)
    )
    distorted_x, distorted_y, distorted_depth = project_simple_radial(
        camera, (0.5, 0.0, 6.0)
    )

    assert undistorted_x == pytest.approx(8.0)
    assert undistorted_y == pytest.approx(8.0)
    assert undistorted_depth == pytest.approx(1.0)
    assert 8.0 < distorted_x < 12.0
    assert distorted_y == pytest.approx(8.0)
    assert distorted_depth == pytest.approx(1.0)

    pose_3x4 = replace(
        camera,
        world_to_camera=tuple(float(value) for value in camera.matrix[:3].reshape(-1)),
    )
    assert project_simple_radial(pose_3x4, (0.5, 0.0, 6.0)) == pytest.approx(
        (distorted_x, distorted_y, distorted_depth)
    )


def test_package_loader_validates_blender_projection_contract(tmp_path: Path):
    geometry = ProjectionGeometry(
        vertices=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        loop_uvs=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        loop_normals=np.tile(
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32), (3, 1)
        ),
        material_ids=np.asarray([0], dtype=np.int32),
    )
    np.savez(
        tmp_path / "projection_geometry.npz",
        vertices=geometry.vertices,
        triangles=geometry.triangles,
        loop_uvs=geometry.loop_uvs,
        loop_normals=geometry.loop_normals,
        material_ids=geometry.material_ids,
    )
    blend_hash = "a" * 64
    (tmp_path / "projection_manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "coordinate_system": "right-handed_y_up",
                "source_blend": {
                    "path": "reconstruction/reference_assisted_v2/work/60_UV_BAKE_ACCEPTED.blend",
                    "sha256": blend_hash,
                },
            }
        ),
        encoding="utf-8",
    )

    package = load_projection_package(tmp_path)

    assert package.source_blend_sha256 == blend_hash
    assert package.geometry == geometry

    np.savez(
        tmp_path / "broken.npz",
        vertices=geometry.vertices,
        triangles=np.asarray([[0, 1, 3]], dtype=np.int32),
        loop_uvs=geometry.loop_uvs,
        loop_normals=geometry.loop_normals,
        material_ids=geometry.material_ids,
    )
    with pytest.raises(ValueError, match="triangle index"):
        load_projection_package(tmp_path, geometry_filename="broken.npz")


def test_projection_selection_keeps_registered_coverage_and_provenance():
    def camera_at(index: int, angle: float):
        matrix = np.eye(4, dtype=float)
        matrix[:3, 3] = (5.0 * np.cos(angle), 5.0 * np.sin(angle), 0.0)
        return replace(_camera(index), world_to_camera=tuple(matrix.reshape(-1)))

    views = []
    for index in range(24):
        angle = index * (2.0 * np.pi / 24.0)
        quality = "bright_clipping" if index == 3 else "normal"
        views.append(_view(index, quality=quality, camera=camera_at(index, angle)))
    views.append(_view(267, registered=False, camera=None))

    selection = select_projection_views(views, maximum_views=12, azimuth_bins=6, elevation_bins=1)

    assert len(selection.views) == 12
    assert selection.views[3].selected_index != 3
    assert all(view.step13_registered and view.camera is not None for view in selection.views)
    assert {row["azimuth_bin"] for row in selection.manifest_rows} == set(range(6))
    assert selection.manifest_rows[0]["mask_source"] in {"reviewed", "reconstruction"}
    assert all(row["camera_id"] == row["selected_index"] + 10 for row in selection.manifest_rows)


def test_robust_fusion_suppresses_transient_highlight_and_keeps_repeated_dark_detail():
    colors = np.asarray(
        [
            [120, 90, 50],
            [126, 94, 52],
            [118, 88, 48],
            [255, 252, 248],
        ],
        dtype=np.float32,
    )
    weights = np.asarray([1.0, 0.8, 0.9, 0.2], dtype=np.float32)
    config = ProjectionConfig(minimum_samples_per_texel=2)

    fused = robust_fuse_colors(colors, weights, config)

    assert fused.accepted_count == 3
    assert np.allclose(fused.color_srgb, [121, 90, 50], atol=1.0)
    assert fused.confidence > 0.75

    dark = np.asarray([[20, 15, 10], [22, 16, 11], [21, 15, 10], [250, 248, 240]], dtype=np.float32)
    fused_dark = robust_fuse_colors(dark, np.ones(4, dtype=np.float32), config)
    assert np.allclose(fused_dark.color_srgb, [21, 15, 10], atol=1.0)


def test_fuse_texel_samples_requires_consensus_and_marks_unsupported_texels():
    from final_texture_projection import ProjectionSample

    accepted = (
        ProjectionSample(0, 0, 0, 1, 0, 10, 10, (120, 90, 50), 1.0, True),
        ProjectionSample(0, 0, 1, 2, 0, 11, 10, (124, 92, 51), 0.8, True),
    )
    unsupported = (ProjectionSample(0, 1, 0, 1, 0, 10, 10, (120, 90, 50), 1.0, True),)

    assert fuse_texel_samples(accepted, ProjectionConfig()).accepted_count == 2
    assert fuse_texel_samples(unsupported, ProjectionConfig()) is None


def test_coverage_and_conservative_inference_contract():
    sample_count = np.asarray([[3, 1], [0, 2]], dtype=np.int32)
    confidence = np.asarray([[0.9, 0.4], [0.0, 0.6]], dtype=np.float32)

    summary = coverage_summary(sample_count, confidence, minimum_samples=2)
    inferred = inference_region_mask(sample_count, confidence, minimum_samples=2)

    assert summary.texture_size == 2
    assert summary.supported_texel_count == 2
    assert summary.direct_projection_percent == pytest.approx(50.0)
    assert summary.zero_sample_percent == pytest.approx(25.0)
    assert inferred.tolist() == [[False, True], [True, False]]


def test_roughness_is_evidence_bounded_and_metallic_remains_brass():
    base_color = np.full((2, 2, 3), [180, 120, 50], dtype=np.float32) / 255.0
    highlight_variance = np.asarray([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    recess = np.asarray([[False, False], [True, True]], dtype=bool)

    roughness, metallic = derive_roughness_and_metallic(
        base_color, highlight_variance=highlight_variance, recess_mask=recess
    )

    assert roughness.min() >= 0.14 and roughness.max() <= 0.48
    assert roughness[1, 0] > roughness[0, 0]
    assert np.all(metallic == 1.0)


def test_wear_mask_is_evidence_bounded_and_normalized():
    base_color = np.full((2, 2, 3), [180, 120, 50], dtype=np.float32) / 255.0
    base_color[1, 0] = np.asarray([45, 30, 12], dtype=np.float32) / 255.0
    recess = np.asarray([[False, False], [True, False]], dtype=bool)
    variance = np.asarray([[0.0, 0.2], [0.4, 0.6]], dtype=np.float32)

    wear = derive_wear_mask(base_color, highlight_variance=variance, recess_mask=recess)

    assert wear.shape == (2, 2) and wear.dtype == np.float32
    assert wear.min() >= 0.0 and wear.max() <= 1.0
    assert wear[1, 0] == wear.max()
    assert wear[0, 0] < wear[1, 1]


def test_overlap_harmonization_is_bounded_and_rejects_bad_samples():
    observations = []
    for surface_index in range(6):
        anchor = np.asarray([120, 90, 50], dtype=np.uint8)
        slightly_dark = np.rint(anchor * 0.92).astype(np.uint8)
        slightly_bright = np.rint(anchor * 1.08).astype(np.uint8)
        too_bright = np.rint(anchor * 1.30).astype(np.uint8)
        observations.extend(
            [
                OverlapObservation(
                    surface_id=surface_index,
                    view_index=0,
                    selected_index=10,
                    color_srgb=tuple(int(value) for value in anchor),
                ),
                OverlapObservation(
                    surface_id=surface_index,
                    view_index=1,
                    selected_index=11,
                    color_srgb=tuple(int(value) for value in slightly_bright),
                ),
                OverlapObservation(
                    surface_id=surface_index,
                    view_index=2,
                    selected_index=12,
                    color_srgb=tuple(int(value) for value in too_bright),
                ),
            ]
        )
    observations.append(
        OverlapObservation(
            surface_id=100,
            view_index=1,
            selected_index=11,
            color_srgb=(255, 252, 248),
        )
    )
    observations.append(
        OverlapObservation(
            surface_id=101,
            view_index=2,
            selected_index=12,
            color_srgb=(240, 235, 225),
            specular_confidence=0.9,
        )
    )

    result = estimate_photometric_harmonization(
        observations,
        minimum_overlap_samples=4,
        minimum_gain=0.88,
        maximum_gain=1.12,
    )
    by_view = {correction.view_index: correction for correction in result.corrections}

    assert result.anchor_view_index == 0
    assert by_view[0].accepted and np.allclose(by_view[0].gains, 1.0)
    assert by_view[1].accepted
    assert np.allclose(by_view[1].gains, 1.0 / 1.08, rtol=0.01)
    assert by_view[1].rejected_clipped_count == 1
    assert not by_view[2].accepted
    assert by_view[2].unreliable_reason == "gain_outside_conservative_bounds"
    assert np.allclose(by_view[2].gains, 1.0)
    assert by_view[2].rejected_specular_count == 1
    assert by_view[1].residual_after < by_view[1].residual_before
    assert result.color_calibrated is False
    assert result.method == "anchor_median_rgb_gain"

    corrected = apply_photometric_correction(
        np.asarray([[130, 97, 54], [120, 90, 50]], dtype=np.float32),
        by_view[1],
    )
    assert np.allclose(corrected, [[120, 90, 50], [111, 83, 46]], atol=1.0)
    with pytest.raises(ValueError, match="photometrically unreliable"):
        apply_photometric_correction(
            np.asarray([[156, 117, 65]], dtype=np.float32), by_view[2]
        )


def test_coverage_classes_control_conservative_gap_fill():
    sample_count = np.ones((3, 2), dtype=np.int32)
    confidence = np.ones((3, 2), dtype=np.float32)
    coverage_class = np.asarray(
        [
            ["direct_multi_view", "direct_multi_view"],
            ["reviewed_single_or_detail", "symmetry_repetition"],
            ["hidden_generic_fill", "hidden_generic_fill"],
        ],
        dtype=object,
    )

    with pytest.raises(ValueError, match="square"):
        coverage_gap_plan(sample_count, confidence, coverage_class, minimum_samples=2)

    sample_count = np.asarray([[2, 0], [0, 0]], dtype=np.int32)
    confidence = np.asarray([[0.8, 0.0], [0.0, 0.0]], dtype=np.float32)
    coverage_class = np.asarray(
        [
            ["direct_multi_view", "direct_multi_view"],
            ["reviewed_single_or_detail", "symmetry_repetition"],
        ],
        dtype=object,
    )
    plan = coverage_gap_plan(sample_count, confidence, coverage_class, minimum_samples=2)
    base_color = np.zeros((2, 2, 3), dtype=np.float32)
    base_color[0, 0] = [0.5, 0.35, 0.15]
    symmetry_source = np.full((2, 2, 3), [0.48, 0.34, 0.14], dtype=np.float32)

    filled = apply_conservative_gap_fill(
        plan,
        base_color=base_color,
        symmetry_source_color=symmetry_source,
        generic_brass_color=(0.45, 0.31, 0.12),
    )

    assert plan.projection_defect_mask.tolist() == [[False, True], [False, False]]
    assert plan.reviewed_source_gap_mask.tolist() == [[False, False], [True, False]]
    assert plan.symmetry_fill_mask.tolist() == [[False, False], [False, True]]
    assert plan.generic_fill_mask.tolist() == [[False, False], [False, False]]
    assert np.allclose(filled.base_color[0, 0], [0.5, 0.35, 0.15])
    assert np.allclose(filled.base_color[1, 0], [0.0, 0.0, 0.0])
    assert np.allclose(filled.base_color[1, 1], symmetry_source[1, 1])
    assert filled.unresolved_mask.tolist() == [
        [False, True],
        [True, False],
    ]


def test_final_map_assembly_and_report_contract_are_hash_bounded(tmp_path: Path):
    size = 8
    config = replace(ProjectionConfig(), texture_size=size)
    base_color = np.zeros((size, size, 3), dtype=np.uint8)
    base_color[:] = [120, 90, 50]
    roughness = np.full((size, size), 0.25, dtype=np.float32)
    wear = np.full((size, size), 0.2, dtype=np.float32)
    metallic = np.ones((size, size), dtype=np.float32)
    normal = np.zeros((size, size, 3), dtype=np.float32)
    normal[:] = [0.0, 0.0, 1.0]
    ao = np.ones((size, size), dtype=np.float32)
    inferred = np.zeros((size, size), dtype=bool)
    inferred[:, -1] = True

    result = assemble_final_texture_set(
        output_dir=tmp_path / "final",
        v2_root=tmp_path,
        config=config,
        base_color=base_color,
        roughness=roughness,
        wear=wear,
        metallic=metallic,
        normal=normal,
        ambient_occlusion=ao,
        inferred_region_mask=inferred,
    )
    coverage = coverage_summary(
        np.where(inferred, 0, 2).astype(np.int32),
        np.where(inferred, 0.0, 0.8).astype(np.float32),
        minimum_samples=2,
    )
    selection = select_projection_views(
        [_view(index) for index in range(24)], maximum_views=24, azimuth_bins=6, elevation_bins=1
    )
    report = build_texture_projection_report(
        source_blend_sha256="b" * 64,
        selection=selection,
        coverage=coverage,
        texture_records=result.texture_records,
        config=config,
    )

    expected_names = {
        "T_ThaiLibation_BaseColor.png",
        "T_ThaiLibation_Roughness.png",
        "T_ThaiLibation_Wear.png",
        "T_ThaiLibation_Metallic.png",
        "T_ThaiLibation_Normal.png",
        "T_ThaiLibation_AO.png",
        "T_ThaiLibation_InferredRegionMask.png",
    }
    assert {Path(record["path"]).name for record in result.texture_records} == expected_names
    assert all(len(record["sha256"]) == 64 for record in result.texture_records)
    assert all(record["width"] == size and record["height"] == size for record in result.texture_records)
    assert result.color_spaces["T_ThaiLibation_BaseColor.png"] == "sRGB"
    assert result.color_spaces["T_ThaiLibation_Normal.png"] == "Non-Color"
    assert report["status"] == "BLOCKED"
    assert "texture_size_below_4096" in report["blocked_reasons"]
    assert "photometric_harmonization_missing" in report["blocked_reasons"]
    assert "coverage_class_provenance_missing" in report["blocked_reasons"]
    assert "gap_fill_provenance_missing" in report["blocked_reasons"]
    assert "surface_evidence_provenance_missing" in report["blocked_reasons"]
    assert report["claim_scope"] == "candidate_bound_projection_contract_no_visual_qa"
    assert report["candidate"]["source_blend_sha256"] == "b" * 64
    assert report["direct_projection_percent"] == pytest.approx(87.5)
    assert report["inferred_fill_percent"] == pytest.approx(12.5)
    assert report["source_blend_sha256"] == "b" * 64
    with Image.open(tmp_path / "final" / "T_ThaiLibation_InferredRegionMask.png") as mask:
        assert mask.getextrema() == (0, 255)


def test_ready_report_is_candidate_bound_and_requires_complete_provenance():
    config = ProjectionConfig(texture_size=4096)
    selection = select_projection_views(
        [_view(index) for index in range(24)], maximum_views=24, azimuth_bins=6, elevation_bins=1
    )
    coverage = CoverageSummary(
        texture_size=4096,
        supported_texel_count=15_938_355,
        direct_projection_percent=95.0,
        low_confidence_percent=2.0,
        zero_sample_percent=1.0,
    )
    texture_record_names = [
        "T_ThaiLibation_BaseColor.png",
        "T_ThaiLibation_Roughness.png",
        "T_ThaiLibation_Metallic.png",
        "T_ThaiLibation_Normal.png",
        "T_ThaiLibation_AO.png",
        "T_ThaiLibation_Wear.png",
        "T_ThaiLibation_InferredRegionMask.png",
    ]
    texture_records = [
        {
            "path": f"reconstruction/reference_assisted_v2/textures/final/{name}",
            "sha256": f"{index:064x}",
            "width": 4096,
            "height": 4096,
            "color_space": "sRGB" if name.endswith("BaseColor.png") else "Non-Color",
        }
        for index, name in enumerate(texture_record_names, start=1)
    ]
    photometric = estimate_photometric_harmonization(
        [
            OverlapObservation(str(surface), view_index, view_index, (120, 90, 50))
            for surface in range(2)
            for view_index in range(24)
        ],
        minimum_overlap_samples=2,
    )
    report = build_texture_projection_report(
        source_blend_sha256="d" * 64,
        selection=selection,
        coverage=coverage,
        texture_records=texture_records,
        config=config,
        surface_evidence_sha256="e" * 64,
        photometric_harmonization=photometric,
        coverage_class_percentages={
            "direct_multi_view": 96.0,
            "reviewed_single_or_detail": 1.0,
            "symmetry_repetition": 2.0,
            "hidden_generic_fill": 1.0,
        },
        gap_fill_summary={
            "texture_size": 4096,
            "symmetry_filled_texel_count": 1,
            "generic_filled_texel_count": 1,
            "projection_defect_texel_count": 0,
            "reviewed_source_gap_texel_count": 0,
            "unresolved_texel_count": 0,
        },
    )

    assert report["status"] == "READY_FOR_MATERIALS"
    assert report["candidate"]["source_blend_sha256"] == "d" * 64
    assert report["candidate"]["surface_evidence_sha256"] == "e" * 64
    assert report["photometric_harmonization"]["color_calibrated"] is False
    assert report["coverage_class_percentages"]["hidden_generic_fill"] == 1.0
    assert report["gap_fill_summary"]["unresolved_texel_count"] == 0
    assert report["selected_views"] == list(selection.manifest_rows)
    assert report["projection_method"] == "SIMPLE_RADIAL_depth_masked_UV_projection"
