from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import final_geometry_audit as audit
from final_cv_model_fit import METRIC_THRESHOLDS, CameraReference


def _cv_fit_payload() -> dict:
    return {
        "accepted": True,
        "metrics_passed": True,
        "candidate_sha256": "a" * 64,
        "metric_thresholds": dict(METRIC_THRESHOLDS),
        "aggregate_metrics": {
            "median_silhouette_iou": 0.91,
            "minimum_reliable_silhouette_iou": 0.85,
            "median_landmark_error_fraction": 0.01,
            "p95_landmark_error_fraction": 0.03,
        },
        "visual_component_review": {
            "bowl": "pass",
            "globe": "pass",
            "neck": "pass",
            "lid": "pass",
            "finial": "pass",
        },
        "fit_summary": {
            "pose_calibration": {
                "camera_center_normalization": {
                    "method": "step13_sweep_camera_center_similarity_gauge_normalization",
                    "sweeps": {},
                }
            }
        },
        "sweep_alignments": {},
        "artifact_hashes": {"final_profiles_sha256": "b" * 64},
    }


def test_accepted_plan1_inputs_and_candidate_hashes_are_fail_closed():
    profiles = {
        "coordinate_contract": {
            "axis": "+Z",
            "normalized_set_height": 1.0,
            "scale_status": "relative_no_physical_measurement",
        },
        "profiles": {},
        "fit_metrics": dict(_cv_fit_payload()["aggregate_metrics"]),
    }
    blend_report = {
        "stage": "geometry-validate-blender",
        "blend_path": "reconstruction/reference_assisted_v2/work/30_BASE_GEOMETRY_ACCEPTED.blend",
        "blend_sha256": "c" * 64,
    }
    review = {"accepted": True, "candidate_sha256": "a" * 64}

    audit.validate_accepted_plan1_inputs(_cv_fit_payload(), profiles, "b" * 64, blend_report, "c" * 64, review)

    stale_review = {**review, "candidate_sha256": "d" * 64}
    weakened_thresholds = _cv_fit_payload()
    weakened_thresholds["metric_thresholds"]["median_silhouette_iou"] = 0.89
    stale_blend = {**blend_report, "blend_sha256": "stale"}

    with pytest.raises(ValueError, match="visual review candidate"):
        audit.validate_accepted_plan1_inputs(
            _cv_fit_payload(), profiles, "b" * 64, blend_report, "c" * 64, stale_review
        )
    with pytest.raises(ValueError, match="metric thresholds"):
        audit.validate_accepted_plan1_inputs(
            weakened_thresholds, profiles, "b" * 64, blend_report, "c" * 64, review
        )
    with pytest.raises(ValueError, match="blend SHA"):
        audit.validate_accepted_plan1_inputs(
            _cv_fit_payload(), profiles, "b" * 64, stale_blend, "c" * 64, review
        )


def _camera(index: int, translation: tuple[float, float, float]) -> CameraReference:
    return CameraReference(
        selected_index=index,
        filename=f"IMG{index:010d}.jpg",
        image_id=index,
        camera_model="SIMPLE_RADIAL",
        camera_params=(100.0, 50.0, 50.0, 0.0),
        cam_from_world_rotation_xyzw=(1.0, 0.0, 0.0, 0.0),
        cam_from_world_translation=translation,
        image_size=(100, 100),
    )


def test_sweep_gauge_normalization_is_exact_and_does_not_touch_top_cameras():
    unmodeled = _camera(1, (-4.0, 0.0, 0.0))
    side = _camera(3, (-4.0, 0.0, 0.0))
    top = _camera(206, (-4.0, 0.0, 0.0))
    normalization = {
        "method": "step13_sweep_camera_center_similarity_gauge_normalization",
        "sweeps": {
            "side_003_072": {"anchor_world": [2.0, 0.0, 0.0], "factor": 0.5},
            "low_090_142": {"anchor_world": [2.0, 0.0, 0.0], "factor": 1.0},
            "elevated_148_200": {"anchor_world": [2.0, 0.0, 0.0], "factor": 1.0},
        },
    }

    derived, provenance = audit.derived_audit_cameras(
        (unmodeled, side, top), normalization
    )

    assert derived[1] == unmodeled
    assert derived[3].cam_from_world_translation == (-3.0, 0.0, 0.0)
    assert derived[206].cam_from_world_translation == (-4.0, 0.0, 0.0)
    assert derived[3].cam_from_world_rotation_xyzw == side.cam_from_world_rotation_xyzw
    assert derived[3].camera_params == side.camera_params
    assert provenance["method"] == normalization["method"]
    assert provenance["canonical_translation_refinements_applied"] is False
    assert provenance["source_step13_files_modified"] is False


def test_registered_view_classification_uses_exact_classes_and_repeated_mismatch():
    empty = audit.classify_registered_view(0.0, 0.0, 0.1, mask_valid=False, projection_valid=True)
    projection = audit.classify_registered_view(
        0.5, 0.2, 0.2, mask_valid=True, projection_valid=False
    )
    mismatch = audit.classify_registered_view(
        0.5,
        0.2,
        0.2,
        mask_valid=True,
        projection_valid=True,
    )
    explained_camera = audit.classify_registered_view(
        0.5,
        0.2,
        0.2,
        mask_valid=True,
        projection_valid=True,
        camera_failure_explained=True,
    )
    good = audit.classify_registered_view(
        audit.MINIMUM_IOU_THRESHOLD + 0.01,
        0.2,
        0.2,
        mask_valid=True,
        projection_valid=True,
    )

    assert empty["classification"] == "mask_failure"
    assert empty["usable"] is False
    assert projection["classification"] == "camera_failure"
    assert projection["usable"] is False
    assert mismatch["classification"] == "model_mismatch"
    assert mismatch["usable"] is True
    assert explained_camera["classification"] == "camera_failure"
    assert explained_camera["usable"] is False
    assert good["classification"] == "ok"
    assert good["usable"] is True


def test_silhouette_camera_consistency_separates_translation_from_shape_error():
    observed = np.zeros((100, 100), dtype=np.uint8)
    predicted_shifted = np.zeros_like(observed)
    predicted_centered_wide = np.zeros_like(observed)
    observed[15:85, 45:55] = 255
    predicted_shifted[15:85, 55:65] = 255
    predicted_centered_wide[15:85, 42:58] = 255

    shifted = audit.silhouette_camera_consistency(observed, predicted_shifted)
    centered = audit.silhouette_camera_consistency(observed, predicted_centered_wide)

    assert shifted["passed"] is False
    assert shifted["centroid_offset_object_height"] > 0.10
    assert shifted["centroid_aligned_iou"] == pytest.approx(1.0)
    assert shifted["translation_only_explains_gate_failure"] is True
    assert shifted["rigid_aligned_iou"] == pytest.approx(1.0)
    assert shifted["rigid_alignment_explains_gate_failure"] is True
    assert shifted["diagnostic_only_translation_pixels"] == pytest.approx([-10.0, 0.0])
    assert shifted["diagnostic_only_rotation_degrees"] == pytest.approx(0.0)
    assert shifted["translation_applied_to_metric"] is False
    assert centered["passed"] is True
    assert centered["centroid_offset_object_height"] == pytest.approx(0.0)
    assert centered["centroid_aligned_iou"] < audit.MINIMUM_IOU_THRESHOLD
    assert centered["translation_only_explains_gate_failure"] is False
    assert centered["rigid_aligned_iou"] < audit.MINIMUM_IOU_THRESHOLD
    assert centered["rigid_alignment_explains_gate_failure"] is False


def test_low_iou_shape_error_is_not_suppressed_by_centroid_axis_thresholds():
    observed = np.zeros((100, 100), dtype=np.uint8)
    predicted = np.zeros_like(observed)
    observed[10:90, 46:54] = 255
    predicted[20:80, 41:59] = 255

    consistency = audit.silhouette_camera_consistency(observed, predicted)
    result = audit.classify_registered_view(
        audit._iou(observed, predicted),
        float(np.count_nonzero(observed)) / observed.size,
        float(np.count_nonzero(predicted)) / predicted.size,
        mask_valid=True,
        projection_valid=True,
        camera_failure_explained=bool(
            consistency["translation_only_explains_gate_failure"]
            or consistency["rigid_alignment_explains_gate_failure"]
        ),
    )

    assert consistency["translation_only_explains_gate_failure"] is False
    assert consistency["rigid_alignment_explains_gate_failure"] is False
    assert result["classification"] == "model_mismatch"
    assert result["usable"] is True


def test_surface_sector_visibility_uses_camera_center_in_object_coordinates():
    alignment = audit.ModelAlignment(
        scale=1.0,
        rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        translation=(0.0, 0.0, 0.0),
    )
    camera = _camera(3, (-10.0, 0.0, -0.5))

    front = audit.surface_sector_camera_visibility(
        camera, alignment, azimuth_radians=0.0, z=0.5, radius=0.2
    )
    back = audit.surface_sector_camera_visibility(
        camera, alignment, azimuth_radians=np.pi, z=0.5, radius=0.2
    )

    assert front["visible"] is True
    assert front["facing_cosine"] > 0.9
    assert back["visible"] is False
    assert back["facing_cosine"] < 0.0


def test_surface_support_class_does_not_promote_unseen_sector_to_direct():
    assert audit.surface_support_class([3, 19]) == "direct_multi_view"
    assert audit.surface_support_class([3]) == "reviewed_single_or_detail"
    assert audit.surface_support_class([]) == "symmetry_repetition"
    assert audit.surface_support_class([], allow_symmetry=False) == "hidden_generic_fill"


def test_projected_feature_edge_residuals_are_candidate_sensitive():
    edges = np.zeros((80, 100), dtype=np.uint8)
    edges[:, 50] = 255
    aligned = np.asarray([[50.0, 10.0], [50.0, 40.0], [50.0, 70.0]])
    shifted = aligned + np.asarray([12.0, 0.0])

    aligned_metrics = audit.projected_edge_residuals(aligned, edges, object_height=80.0)
    shifted_metrics = audit.projected_edge_residuals(shifted, edges, object_height=80.0)

    assert aligned_metrics["sample_count"] == 3
    assert aligned_metrics["median_fraction_object_height"] == pytest.approx(0.0)
    assert shifted_metrics["median_fraction_object_height"] > 0.10
    assert shifted_metrics["p90_fraction_object_height"] > aligned_metrics[
        "p90_fraction_object_height"
    ]


def test_projected_axis_consistency_is_independent_of_candidate_width():
    observed = np.zeros((100, 100), dtype=np.uint8)
    observed[10:90, 35:65] = 255
    centered_axis = np.asarray(((50.0, 90.0), (50.0, 10.0)))
    shifted_axis = np.asarray(((65.0, 90.0), (65.0, 10.0)))
    short_axis = np.asarray(((50.0, 80.0), (50.0, 20.0)))

    centered = audit.projected_axis_mask_consistency(observed, centered_axis)
    shifted = audit.projected_axis_mask_consistency(observed, shifted_axis)
    short = audit.projected_axis_mask_consistency(observed, short_axis)

    assert centered["passed"] is True
    assert centered["perpendicular_offset_object_height"] < 0.01
    assert shifted["passed"] is False
    assert shifted["perpendicular_offset_object_height"] > 0.15
    assert short["passed"] is False
    assert short["axial_extent_error_object_height"] > 0.10


def test_unreviewed_mask_requires_two_reviewed_envelope_anomalies_to_fail():
    envelope = {
        "source_views": [3, 10, 19, 28, 39, 50, 62, 72],
        "maximum_bbox_width_per_height": 0.445,
        "maximum_p95_row_width_per_height": 0.430,
        "minimum_solidity": 0.725,
    }
    contaminated = {
        "valid": True,
        "failure_reasons": [],
        "bbox_width_per_height": 0.542,
        "p95_row_width_per_height": 0.449,
        "silhouette_solidity": 0.633,
    }
    one_outlier = {
        **contaminated,
        "bbox_width_per_height": 0.440,
        "silhouette_solidity": 0.730,
    }

    failed = audit.assess_unreviewed_mask_shape(contaminated, envelope)
    retained = audit.assess_unreviewed_mask_shape(one_outlier, envelope)

    assert failed["valid"] is False
    assert failed["failure_reasons"] == [
        "reviewed_shape_envelope_contamination"
    ]
    assert len(failed["reviewed_shape_anomalies"]) >= 2
    assert retained["valid"] is True
    assert retained["reviewed_shape_anomalies"] == [
        "p95_row_width_above_reviewed_envelope"
    ]


def test_connected_lateral_contamination_fails_axial_band_width_envelope():
    envelope = {
        "source_views": [3, 19, 50, 72],
        "maximum_bbox_width_per_height": 0.50,
        "maximum_p95_row_width_per_height": 0.45,
        "minimum_solidity": 0.65,
        "maximum_band_p90_width_per_height": [0.10] * 10,
    }
    quality = {
        "valid": True,
        "failure_reasons": [],
        "bbox_width_per_height": 0.48,
        "p95_row_width_per_height": 0.42,
        "silhouette_solidity": 0.72,
        "band_p90_width_per_height": [
            0.10,
            0.11,
            0.24,
            0.25,
            0.10,
            0.10,
            0.10,
            0.10,
            0.10,
            0.10,
        ],
    }

    result = audit.assess_unreviewed_mask_shape(quality, envelope)

    assert result["valid"] is False
    assert "axial_band_width_contamination" in result["failure_reasons"]
    assert result["reviewed_band_width_anomalies"] == [2, 3]


def test_base_visual_review_is_bound_to_exact_blend_and_render_hashes(tmp_path: Path):
    render = tmp_path / "geometry_validation_front.png"
    render.write_bytes(b"candidate render")
    digest = audit.sha256_file(render)
    review = {
        "accepted": True,
        "blend_sha256": "a" * 64,
        "plan1_candidate_sha256": "b" * 64,
        "components": {
            "bowl_pedestal": "pass",
            "globe_shoulder": "pass",
            "neck": "pass",
            "lid_finial": "pass",
            "overall_identity": "pass",
        },
        "diagnostics": [
            {
                "view": view,
                "path": render.name,
                "sha256": digest,
            }
            for view in audit.REQUIRED_BASE_VISUAL_VIEWS
        ],
    }

    result = audit.validate_base_visual_review(
        review,
        tmp_path,
        "a" * 64,
        "b" * 64,
    )
    stale = audit.validate_base_visual_review(
        review,
        tmp_path,
        "c" * 64,
        "b" * 64,
    )

    assert result["passed"] is True
    assert stale["passed"] is False
    assert "visual review blend SHA is stale" in stale["failures"]


def test_registered_worst_view_review_is_candidate_and_diagnostic_bound(tmp_path: Path):
    diagnostic = tmp_path / "045_overlay.png"
    diagnostic.write_bytes(b"registered diagnostic")
    worst = [
        {
            "selected_index": 45,
            "classification": "camera_failure",
            "diagnostic": diagnostic.name,
        }
    ]
    review = {
        "accepted": True,
        "blend_sha256": "a" * 64,
        "views": [
            {
                "selected_index": 45,
                "classification": "camera_failure",
                "diagnostic": diagnostic.name,
                "diagnostic_sha256": audit.sha256_file(diagnostic),
                "verdict": "classification_confirmed",
            }
        ],
    }

    passed = audit.validate_registered_visual_review(
        review, tmp_path, "a" * 64, worst
    )
    wrong_class = {
        **review,
        "views": [{**review["views"][0], "classification": "model_mismatch"}],
    }

    assert passed["passed"] is True
    assert audit.validate_registered_visual_review(
        wrong_class, tmp_path, "a" * 64, worst
    )["passed"] is False


def test_report_reopening_and_fail_closed_repeated_mismatch(tmp_path: Path):
    view = {
        "selected_index": 4,
        "filename": "IMG4.jpg",
        "sweep": "side_003_072",
        "mask_source": "analysis/ml/reconstruction_masks/004_mask.png",
        "coordinate_path": "SIMPLE_RADIAL",
        "silhouette_iou": 0.5,
        "foreground_coverage_difference": 0.0,
        "source_quality_condition": "usable",
        "usable": True,
        "classification": "model_mismatch",
        "projection_valid": True,
        "camera_consistency": {
            "translation_only_explains_gate_failure": False,
            "rigid_alignment_explains_gate_failure": False,
        },
        "diagnostic": "diagnostics/21_registered_view_coverage/004.png",
    }
    report = {
        "accepted": False,
        "blend_path": "work/30_BASE_GEOMETRY_ACCEPTED.blend",
        "blend_sha256": "a" * 64,
        "final_cv_fit_candidate_sha256": "b" * 64,
        "provenance": {
            "mask_manifest": {"path": "analysis/reports/reconstruction_mask_manifest.csv", "sha256": "c" * 64},
            "camera_normalization": {"path": "reports/final_cv_fit.json", "sha256": "d" * 64},
            "camera_alignment": {"path": "reports/final_cv_fit.json", "sha256": "d" * 64},
        },
        "views": [view],
        "inspected_worst_views": [
            {
                "selected_index": 4,
                "filename": view["filename"],
                "classification": view["classification"],
            }
        ],
    }
    surface = {
        "accepted": False,
        "blend_path": report["blend_path"],
        "blend_sha256": report["blend_sha256"],
        "components": {
            component: [
                {
                    "sector": "azimuth_000_030/elevation_mid",
                    "support_class": "direct_multi_view",
                    "source_support": {
                        "support_count": 2,
                        "source_views": [3, 19],
                        "camera_component_mask_evidence": [
                            {"selected_index": 3},
                            {"selected_index": 19},
                        ],
                    },
                    "provenance": {
                        "source": "evidence/component_masks",
                        "manifest_sha256": "e" * 64,
                    },
                }
            ]
            for component in audit.REQUIRED_SURFACE_COMPONENTS
        },
    }

    registered_path = tmp_path / "registered_view_coverage_report.json"
    surface_path = tmp_path / "surface_evidence_coverage.json"
    registered_path.write_text(json.dumps(report), encoding="utf-8")
    surface_path.write_text(json.dumps(surface), encoding="utf-8")

    registered = audit.reopen_registered_report(registered_path)
    reopened_surface = audit.reopen_surface_report(surface_path)

    assert registered["usable_model_mismatch_count"] == 1
    assert not registered["passed"]
    assert not reopened_surface["passed"]
