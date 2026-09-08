from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from final_model_validation import (
    FinalValidationResult,
    aggregate_final_validation,
    build_comparison_panel,
    compare_render_images,
    evaluate_geometry_gate,
    evaluate_visual_identity_gate,
    validate_registered_view_coverage_report,
    validate_export_reimport_report,
    validate_texture_report,
    validate_surface_evidence_coverage_report,
    validate_topology_report,
    validate_uv_bake_report,
)


def test_final_acceptance_requires_every_gate_and_exact_ship_verdict():
    gates = {
        "geometry": True,
        "ornament": True,
        "topology": True,
        "uv_bake": True,
        "texture": True,
        "visual_identity": True,
        "export_reimport": True,
    }

    accepted = aggregate_final_validation(**gates, qa_verdict="SHIP")
    rejected = aggregate_final_validation(**gates, qa_verdict="SHIP WITH NOTES")
    failed_gate = aggregate_final_validation(
        **{**gates, "export_reimport": False}, qa_verdict="SHIP"
    )

    assert accepted == FinalValidationResult(
        geometry_passed=True,
        ornament_passed=True,
        topology_passed=True,
        uv_bake_passed=True,
        texture_passed=True,
        visual_identity_passed=True,
        export_reimport_passed=True,
        qa_verdict="SHIP",
        accepted=True,
    )
    assert rejected.accepted is False
    assert failed_gate.accepted is False


def test_compare_render_images_normalizes_size_and_reports_export_metrics(tmp_path):
    reference = np.zeros((4, 4, 4), dtype=np.uint8)
    reference[:] = (202, 40, 40, 128)
    candidate = np.zeros((8, 8, 4), dtype=np.uint8)
    candidate[:] = (200, 40, 40, 128)
    reference_path = tmp_path / "master.png"
    candidate_path = tmp_path / "reimport.png"
    Image.fromarray(reference, mode="RGBA").save(reference_path)
    Image.fromarray(candidate, mode="RGBA").save(candidate_path)

    result = compare_render_images(reference_path, candidate_path)

    assert result["reference_size"] == (4, 4)
    assert result["candidate_size"] == (8, 8)
    assert result["compared_size"] == (4, 4)
    assert result["transparent"] is True
    assert result["silhouette_iou"] == pytest.approx(1.0)
    assert result["foreground_coverage_difference"] == pytest.approx(0.0)
    assert 0.0 < result["mean_absolute_rgb_difference"] < 0.03
    assert 0.0 < result["percentile_95_absolute_difference"] < 0.03


def test_geometry_gate_enforces_reliable_views_and_component_veto():
    passing_metrics = evaluate_geometry_gate(
        per_view_iou={0: 0.95, 1: 0.91, 2: 0.86, 3: 0.90},
        per_view_landmark_error={0: 0.010, 1: 0.020, 2: 0.018, 3: 0.030},
        component_review={
            "bowl_pedestal": "pass",
            "globe_shoulder": "pass",
            "neck": "pass",
            "lid_finial": "pass",
            "overall_identity": "pass",
        },
    )

    assert passing_metrics["passed"] is True
    assert passing_metrics["metrics"]["median_iou"] == pytest.approx(0.905)
    assert passing_metrics["metrics"]["minimum_iou"] == pytest.approx(0.86)
    assert passing_metrics["metrics"]["p95_landmark_error"] == pytest.approx(0.0285)

    weak_view = evaluate_geometry_gate(
        per_view_iou={0: 0.95, 1: 0.91, 2: 0.83, 3: 0.90},
        per_view_landmark_error={0: 0.010, 1: 0.020, 2: 0.018, 3: 0.030},
        component_review=passing_metrics["component_review"],
    )
    component_veto = evaluate_geometry_gate(
        per_view_iou=passing_metrics["per_view_iou"],
        per_view_landmark_error=passing_metrics["per_view_landmark_error"],
        component_review={
            "bowl_pedestal": "fail",
            "globe_shoulder": "pass",
            "neck": "pass",
            "lid_finial": "pass",
            "overall_identity": "pass",
        },
    )

    assert weak_view["passed"] is False
    assert weak_view["failures"] == ["minimum reliable-view silhouette IoU 0.8300 < 0.8400"]
    assert component_veto["passed"] is False
    assert component_veto["failures"] == ["component visual veto: bowl_pedestal"]


def test_visual_identity_gate_rejects_miss_and_incomplete_review():
    ratings = {
        "camera": "Match",
        "silhouette_proportions": "Close",
        "depth_construction": "Match",
        "hero_ornament": "Match",
        "materials": "Close",
    }
    component_review = {
        "bowl_pedestal": "pass",
        "globe_shoulder": "pass",
        "neck": "pass",
        "lid_finial": "pass",
        "overall_identity": "pass",
    }

    assert evaluate_visual_identity_gate(ratings, component_review)["passed"] is True
    miss = evaluate_visual_identity_gate(
        {**ratings, "hero_ornament": "Miss"}, component_review
    )
    incomplete = evaluate_visual_identity_gate({}, {})

    assert miss["passed"] is False
    assert miss["failures"] == ["visual rating Miss for hero_ornament"]
    assert incomplete["passed"] is False
    assert "missing visual rating: camera" in incomplete["failures"]


def test_topology_report_requires_clean_intentional_export_objects():
    clean = {
        "accepted": True,
        "counts": {
            "non_manifold_edges": 0,
            "loose_vertices": 0,
            "loose_edges": 0,
            "loose_faces": 0,
            "inverted_faces": 0,
            "self_intersections": 0,
        },
        "objects": [{"name": "SM_Vessel_Body_LOW"}],
    }

    assert validate_topology_report(clean)["passed"] is True
    default_name = validate_topology_report(
        {**clean, "objects": [{"name": "Cube.001"}]}
    )
    nonmanifold = validate_topology_report(
        {**clean, "counts": {**clean["counts"], "non_manifold_edges": 1}}
    )

    assert default_name["passed"] is False
    assert default_name["failures"] == ["default Blender object name: Cube.001"]
    assert nonmanifold["passed"] is False
    assert nonmanifold["failures"] == ["non-zero non_manifold_edges: 1"]


def test_uv_bake_report_enforces_uv_main_maps_and_documented_exception():
    report = {
        "accepted": True,
        "meshes": [{"name": "SM_Vessel_Body_LOW", "uv_map": "UV_Main"}],
        "unintended_overlap": False,
        "pack_efficiency": 0.74,
        "texture_resolution": 4096,
        "normal_orientation_verified": True,
        "hero_bake_artifacts": [],
        "bake_maps_accepted": True,
    }

    overlap_failure = validate_uv_bake_report(
        {**report, "unintended_overlap": True}
    )
    exception_pass = validate_uv_bake_report(
        {
            **report,
            "pack_efficiency_exception": "documented UDIM separation",
        }
    )

    assert overlap_failure["passed"] is False
    assert overlap_failure["failures"] == [
        "UV contains unintended unique-detail overlap",
        "UV pack efficiency is below 0.75 without a documented exception",
    ]
    assert exception_pass["passed"] is True


def test_texture_and_export_reports_fail_closed_on_missing_evidence(tmp_path):
    texture_report = {
        "accepted": True,
        "files": [{"path": "textures/baked/base_color.png", "hash": "abc", "size": [4096, 4096]}],
        "missing_images": [],
        "visual_conditions": {
            "bright_polished_brass_match": True,
            "large_regions_not_dark": True,
            "engraving_readable": True,
            "no_baked_moving_highlights": True,
            "no_severe_uv_seams": True,
            "no_unsupported_hero_synthesis": True,
        },
        "visual_match": "Close",
        "inferred_region_percentage": 8.0,
    }
    texture_failure = validate_texture_report(
        {**texture_report, "missing_images": ["textures/baked/normal.png"]}
    )

    export_report = {
        "accepted": True,
        "mesh_count": 6,
        "material_count": 3,
        "image_count": 4,
        "missing_external_files": [],
        "orientation_verified": True,
        "normals_verified": True,
        "visual_inspection_passed": True,
        "render_comparison": {
            "silhouette_iou": 0.996,
            "mean_absolute_rgb_difference": 0.025,
        },
    }
    export_failure = validate_export_reimport_report(
        {**export_report, "render_comparison": {
            "silhouette_iou": 0.990,
            "mean_absolute_rgb_difference": 0.025,
        }}
    )

    assert validate_texture_report(texture_report)["passed"] is True
    assert texture_failure["passed"] is False
    assert texture_failure["failures"] == ["missing texture image: textures/baked/normal.png"]
    assert validate_export_reimport_report(export_report)["passed"] is True
    assert export_failure["passed"] is False
    assert export_failure["failures"] == [
        "export silhouette IoU 0.9900 < 0.9950"
    ]


def test_build_comparison_panel_is_deterministic(tmp_path):
    def image(path: Path, color: tuple[int, int, int]) -> Path:
        array = np.zeros((8, 8, 3), dtype=np.uint8)
        array[:] = color
        path = tmp_path / path
        Image.fromarray(array, mode="RGB").save(path)
        return path

    source = image(Path("source.png"), (255, 0, 0))
    final = image(Path("final.png"), (0, 255, 0))
    overlay = image(Path("overlay.png"), (0, 0, 255))
    difference = image(Path("difference.png"), (255, 255, 0))
    output = tmp_path / "panel.png"

    build_comparison_panel(
        output,
        source=source,
        final_render=final,
        silhouette_overlay=overlay,
        difference=difference,
        source_label="SOURCE 001",
    )
    first = output.read_bytes()
    build_comparison_panel(
        output,
        source=source,
        final_render=final,
        silhouette_overlay=overlay,
        difference=difference,
        source_label="SOURCE 001",
    )

    with Image.open(output) as panel:
        assert panel.size == (652, 228)
        assert panel.mode == "RGB"
    assert output.read_bytes() == first


def test_registered_view_coverage_requires_candidate_provenance_and_rejects_repeated_mismatch():
    sha256 = "a" * 64
    view = {
        "selected_index": 74,
        "filename": "IMG20260826123112345.jpg",
        "sweep": "low",
        "mask_source": "reconstruction/reference_assisted_v2/evidence/geometry_masks/074.png",
        "coordinate_path": "SIMPLE_RADIAL",
        "silhouette_iou": 0.91,
        "foreground_coverage_difference": 0.04,
        "source_quality_condition": "usable",
        "usable": True,
        "classification": "ok",
        "projection_valid": True,
        "camera_consistency": {
            "translation_only_explains_gate_failure": False,
            "rigid_alignment_explains_gate_failure": False,
        },
    }
    report = {
        "accepted": True,
        "blend_path": "reconstruction/reference_assisted_v2/blends/base_geometry.blend",
        "blend_sha256": sha256,
        "final_cv_fit_candidate_sha256": "b" * 64,
        "provenance": {
            "mask_manifest": {"path": "evidence/mask_manifest.json", "sha256": "c" * 64},
            "camera_normalization": {"path": "reports/final_cv_fit.json", "sha256": "d" * 64},
            "camera_alignment": {"path": "reports/final_cv_fit.json", "sha256": "e" * 64},
        },
        "views": [view],
        "inspected_worst_views": [
            {
                "selected_index": 74,
                "filename": view["filename"],
                "classification": "ok",
            }
        ],
    }

    accepted = validate_registered_view_coverage_report(report)
    assert accepted["passed"] is True
    assert accepted["usable_model_mismatch_count"] == 0

    stale = validate_registered_view_coverage_report({**report, "blend_sha256": "not-a-hash"})
    wrong_coordinate_path = validate_registered_view_coverage_report(
        {**report, "views": [{**view, "coordinate_path": "raw_pinhole"}]}
    )
    repeated_mismatch = validate_registered_view_coverage_report(
        {
            **report,
            "views": [
                view,
                {**view, "selected_index": 75, "classification": "model_mismatch"},
                {**view, "selected_index": 76, "classification": "model_mismatch"},
            ],
        }
    )
    unproven_camera_failure = validate_registered_view_coverage_report(
        {
            **report,
            "views": [
                {
                    **view,
                    "silhouette_iou": 0.5,
                    "classification": "camera_failure",
                    "usable": False,
                }
            ],
            "inspected_worst_views": [
                {
                    "selected_index": 74,
                    "filename": view["filename"],
                    "classification": "camera_failure",
                }
            ],
        }
    )

    assert stale["passed"] is False
    assert any("blend_sha256" in failure for failure in stale["failures"])
    assert wrong_coordinate_path["passed"] is False
    assert any("coordinate_path" in failure for failure in wrong_coordinate_path["failures"])
    assert repeated_mismatch["passed"] is False
    assert repeated_mismatch["usable_model_mismatch_count"] == 2
    assert any("repeated usable model_mismatch" in failure for failure in repeated_mismatch["failures"])
    assert unproven_camera_failure["passed"] is False
    assert any("camera_failure is not independently explained" in failure for failure in unproven_camera_failure["failures"])


def test_surface_evidence_coverage_requires_exact_classes_and_sector_provenance():
    sha256 = "a" * 64
    components = {
        component: [
            {
                "sector": "azimuth_000_030/elevation_low",
                "support_class": "direct_multi_view",
                "source_support": {
                    "support_count": 3,
                    "source_views": [3, 19, 50],
                    "camera_component_mask_evidence": [
                        {"selected_index": 3},
                        {"selected_index": 19},
                        {"selected_index": 50},
                    ],
                },
                "provenance": {
                    "source": "reconstruction/reference_assisted_v2/evidence/component_masks",
                    "manifest_sha256": "b" * 64,
                },
            }
        ]
        for component in ("bowl_pedestal", "globe_shoulder", "neck", "lid_finial")
    }
    report = {
        "accepted": True,
        "blend_path": "reconstruction/reference_assisted_v2/blends/base_geometry.blend",
        "blend_sha256": sha256,
        "components": components,
    }

    accepted = validate_surface_evidence_coverage_report(report)
    invalid_class = validate_surface_evidence_coverage_report(
        {
            **report,
            "components": {
                **components,
                "neck": [
                    {**components["neck"][0], "support_class": "photogrammetrically_reconstructed"}
                ],
            },
        }
    )
    empty_component = validate_surface_evidence_coverage_report(
        {**report, "components": {**components, "lid_finial": []}}
    )
    incomplete_support = validate_surface_evidence_coverage_report(
        {
            **report,
            "components": {
                **components,
                "globe_shoulder": [
                    {
                        **components["globe_shoulder"][0],
                        "source_support": {"source_views": [3]},
                        "provenance": {"source": "evidence/components", "manifest_sha256": "short"},
                    }
                ],
            },
        }
    )
    unsupported_direct = validate_surface_evidence_coverage_report(
        {
            **report,
            "components": {
                **components,
                "bowl_pedestal": [
                    {
                        **components["bowl_pedestal"][0],
                        "source_support": {
                            "support_count": 2,
                            "source_views": [3, 19],
                            "camera_component_mask_evidence": [
                                {"selected_index": 3}
                            ],
                        },
                    }
                ],
            },
        }
    )

    assert accepted["passed"] is True
    assert accepted["sector_count"] == 4
    assert invalid_class["passed"] is False
    assert any("support_class" in failure for failure in invalid_class["failures"])
    assert empty_component["passed"] is False
    assert any("lid_finial" in failure for failure in empty_component["failures"])
    assert incomplete_support["passed"] is False
    assert any("support_count" in failure for failure in incomplete_support["failures"])
    assert any("manifest_sha256" in failure for failure in incomplete_support["failures"])
    assert unsupported_direct["passed"] is False
    assert any("camera/component visibility evidence" in failure for failure in unsupported_direct["failures"])
