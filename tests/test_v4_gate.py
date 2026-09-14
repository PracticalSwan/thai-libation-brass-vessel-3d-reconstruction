from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from v4_dense import (
    dense_gate,
    finalize_dense_visual_gate,
    postfusion_evidence_gate,
    validate_sparse_lineage,
)
from v4_mesh import finalize_raw_visual_gate, raw_visual_gate
from v4_sparse import sparse_gate


def _measured(value: float = 0.0) -> dict[str, str | float]:
    return {
        "value": value,
        "measurement_method": "region-labelled preview grid",
        "evidence_sha256": "a" * 64,
    }


def _metrics() -> dict[str, object]:
    return {
        "point_count": 1000,
        "finite_xyz_fraction": 1.0,
        "rank": 3,
        "board_point_fraction": 0.0,
        "cloth_point_fraction": 0.0,
        "contamination_metrics": {
            "board_point_fraction": _measured(0.0),
            "pedestal_board_webbing_point_fraction": _measured(0.0),
            "cloth_or_background_point_fraction": _measured(0.0),
        },
        "contamination_findings": {
            "board_slab_detected": False,
            "pedestal_board_webbing_detected": False,
            "cloth_or_background_structure_detected": False,
            "vessel_identity_confirmed": True,
        },
    }


def _write_previews(root: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    root.mkdir(parents=True, exist_ok=True)
    for name in ("front", "quarter", "side", "top_oblique"):
        path = root / f"{name}.png"
        path.write_bytes(name.encode("utf-8"))
        result[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return result


def _review(root: Path, **overrides: object) -> dict[str, object]:
    review: dict[str, object] = {
        "review_status": "passed",
        "approval_status": "passed",
        "reviewer": "independent-reviewer",
        "approver": "controller",
        "previews": _write_previews(root),
        "regions": {
            "bowl_and_interior": True,
            "globe_or_shoulder": True,
            "neck_lid_or_finial": True,
            "pedestal_and_base": True,
        },
        "contamination_findings": {
            "board_slab_detected": False,
            "pedestal_board_webbing_detected": False,
            "cloth_or_background_structure_detected": False,
            "vessel_identity_confirmed": True,
        },
    }
    review.update(overrides)
    return review


def _smoke() -> dict[str, object]:
    return {"status": "passed", "depth_normal_count": 2}


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_best_defensible_sparse_lineage_is_explicit_and_does_not_claim_strict_pass() -> None:
    lineage = {
        "status": "best_defensible_sparse_candidate",
        "best_defensible": True,
        "sparse_gate_passed": False,
        "accepted_sparse_model_sha256": "a" * 64,
        "accepted_sparse_gate_sha256": "b" * 64,
        "selection_report_sha256": "c" * 64,
        "strict_gate_failures": ["mask_projection_per_view"],
    }

    result = validate_sparse_lineage(lineage)

    assert result["passed"] is True
    assert result["best_defensible"] is True
    assert result["sparse_gate_passed"] is False


def test_accepted_sparse_lineage_still_requires_a_strict_gate_pass() -> None:
    result = validate_sparse_lineage(
        {
            "status": "accepted_sparse_candidate",
            "best_defensible": True,
            "sparse_gate_passed": False,
            "accepted_sparse_model_sha256": "a" * 64,
            "accepted_sparse_gate_sha256": "b" * 64,
        }
    )

    assert result["passed"] is False
    assert "sparse lineage does not record a passed sparse gate" in result["reasons"]


def _postfusion_report(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    fused = root / "fused.ply"
    fused.write_bytes(b"fused-cloud")
    fused_hash = hashlib.sha256(fused.read_bytes()).hexdigest()
    semantic_root = root / "semantic"
    semantic = {}
    for name in ("front", "quarter", "side", "top_oblique"):
        path = semantic_root / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode("utf-8"))
        semantic[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    anatomy_root = root / "anatomy"
    anatomy = {}
    anatomy_names = (
        "bowl_interior",
        "rim",
        "globe_shoulder",
        "continuous_neck",
        "lid_tiers",
        "finial",
        "pedestal_transitions",
        "base",
    )
    anatomy_regions = {}
    for name in anatomy_names:
        anatomy_regions[name] = {
            "region_name": name,
            "point_count": 500,
            "supported_point_count": 480,
            "supported_fraction": 0.96,
            "projected_coverage_fraction": 0.90,
            "connected_support_fraction": 0.95,
            "height_bin_occupancy_fraction": 1.0,
        }
    for name in anatomy_names:
        path = anatomy_root / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"anatomy-{name}".encode("utf-8"))
        anatomy[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "anatomy_label": name,
            "region_crop": {
                "region_name": name,
                "projection_scope": "region_only_height_crop",
                "whole_object_projection": False,
                "region_measurement_sha256": _canonical_hash(anatomy_regions[name]),
            },
        }
    contamination_evidence = {
        "fused_sha256": fused_hash,
        "workspace_root": str(root),
        "mask_dir": str(root / "masks"),
        "mask_count": 4,
        "camera_count": 4,
        "multi_view_projection": {
            "view_count": 4,
            "mask_count": 4,
            "aggregation": "all_registered_camera_views",
        },
        "mask_resolution": {
            "resolver": "colmap_image_name_plus_png",
            "resolved_count": 4,
            "missing_count": 0,
            "resolution_counts": {"colmap_image_name_plus_png": 4},
            "legacy_fallback_count": 0,
        },
        "sample_count": 4,
        "visible_count": 4,
        "inside_fraction_quantiles": [0.8, 0.9, 1.0],
    }
    contamination_anatomy = {
        "schema_version": 2,
        "status": "passed",
        "passed": True,
        "no_major_vessel_scale_holes": True,
        "regions": anatomy_regions,
        "finial_shape": {
            "resolved_narrow_top_element": True,
            "finial_to_lid_radius_ratio": 0.5,
        },
    }
    contamination_metrics = {
        key: {
            "value": 0.0,
            "count": 0,
            "denominator": 4,
            "measurement_method": "synthetic test projection",
            "evidence_sha256": _canonical_hash(contamination_evidence),
        }
        for key in (
            "board_point_fraction",
            "pedestal_board_webbing_point_fraction",
            "cloth_or_background_point_fraction",
        )
    }
    config = root / "tile-000.cfg"
    config.write_text("a.jpg\nb.jpg\n", encoding="utf-8")
    config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    depth_dir = root / "depth"
    mask_dir = root / "masks"
    sparse_dir = root / "sparse"
    depth_dir.mkdir()
    mask_dir.mkdir()
    sparse_dir.mkdir()
    depth_a = depth_dir / "a.jpg.geometric.bin"
    depth_b = depth_dir / "b.jpg.geometric.bin"
    mask_a = mask_dir / "a.jpg.png"
    mask_b = mask_dir / "b.jpg.png"
    depth_a.write_bytes(b"depth-a")
    depth_b.write_bytes(b"depth-b")
    mask_a.write_bytes(b"mask-a")
    mask_b.write_bytes(b"mask-b")
    measurement = {
        "status": "measured",
        "reference": "a.jpg",
        "source": "b.jpg",
        "sampled": 4,
        "overlap_count": 4,
        "coverage_fraction": 1.0,
        "consistent_fraction_at_1pct": 1.0,
        "relative_depth_median": 0.0,
        "relative_depth_p95": 0.0,
        "reference_mask_path": str(mask_a.resolve()),
        "source_mask_path": str(mask_b.resolve()),
        "reference_mask_resolution": "colmap_image_name_plus_png",
        "source_mask_resolution": "colmap_image_name_plus_png",
        "reference_depth_sha256": hashlib.sha256(depth_a.read_bytes()).hexdigest(),
        "source_depth_sha256": hashlib.sha256(depth_b.read_bytes()).hexdigest(),
        "reference_mask_sha256": hashlib.sha256(mask_a.read_bytes()).hexdigest(),
        "source_mask_sha256": hashlib.sha256(mask_b.read_bytes()).hexdigest(),
    }
    return {
        "status": "passed",
        "fused_path": str(fused),
        "fused_sha256": fused_hash,
        "semantic_previews": semantic,
        "anatomy_previews": anatomy,
        "contamination": {
            "status": "measured",
            "metrics": contamination_metrics,
            "findings": {
                "board_slab_detected": False,
                "pedestal_board_webbing_detected": False,
                "cloth_or_background_structure_detected": False,
                "vessel_identity_confirmed": True,
            },
            "evidence": contamination_evidence,
            "camera_count": 4,
            "ring_counts": {"g1": 2, "g2": 2},
            "anatomy": contamination_anatomy,
        },
        "source_selection_audit": {
            "status": "passed",
            "errors": [],
            "configs": [
                {
                    "path": str(config),
                    "sha256": config_hash,
                    "reference_count": 1,
                    "source_count": 1,
                }
            ],
            "references": [
                {"reference": "a.jpg"},
                {"reference": "b.jpg"},
                {"reference": "c.jpg"},
                {"reference": "d.jpg"},
            ],
            "observed": {
                "reference_count": 4,
                "registered_image_count": 4,
                "configured_reference_count_total": 4,
                "source_only_reference_count": 0,
                "directed_source_count": 4,
                "cross_ring_directed_source_count": 2,
                "references_without_cross_ring_source_count": 0,
                "reference_occurrence_counts": {"a.jpg": 1, "b.jpg": 1, "c.jpg": 1, "d.jpg": 1},
                "duplicate_reference_writes": {},
                "missing_reference_writes": [],
                "exact_one_reference_write": True,
            },
            "deviation_from_original_source_selection": {
                "original_plan": "COLMAP automatic/default source selection",
                "deviation_is_intentional": True,
                "max_sources": 6,
            },
        },
        "sparse_lineage": {
            "status": "accepted_sparse_candidate",
            "sparse_gate_passed": True,
            "accepted_sparse_model_sha256": "a" * 64,
            "accepted_sparse_gate_sha256": "b" * 64,
        },
        "ring_transition_audit": {
            "status": "passed",
            "passed": True,
            "transition_count": 1,
            "transitions": {
                "g1->g2": {
                    "status": "passed",
                    "pair_count": 1,
                    "evaluated_pairs": 1,
                    "selected_pairs": [["a.jpg", "b.jpg"]],
                    "measurements": [measurement],
                }
            },
            "evidence": {
                "workspace_root": str(root),
                "depth_dir": str(depth_dir),
                "mask_dir": str(mask_dir),
                "mask_resolver": "colmap_image_name_plus_png",
                "sparse_model_path": str(sparse_dir),
                "pair_measurement_method": "project_reference_geometric_depth_into_source_and_compare_masked_geometric_depth",
                "evaluated_pair_count": 1,
            },
        },
    }


def _mesh_metrics() -> dict[str, object]:
    return {
        "vertex_count": 10,
        "face_count": 16,
        "finite_xyz_fraction": 1.0,
        "rank": 3,
    }


def _install_json_writer(monkeypatch: pytest.MonkeyPatch, module: object) -> None:
    def write(path: str | Path, payload: object) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    monkeypatch.setattr(module, "write_json", write)


def test_dense_gate_requires_measured_contamination_and_review_evidence(tmp_path: Path):
    metrics = _metrics()
    review = _review(tmp_path)
    result = dense_gate(metrics, smoke=_smoke(), visual_status="passed", review=review)
    assert result["passed"] is True
    assert result["review"]["preview_evidence"]["front"]["sha256"] == review["previews"]["front"]["sha256"]

    unmeasured = dict(metrics)
    unmeasured["contamination_metrics"] = {}
    unmeasured["contamination_findings"] = {}
    result = dense_gate(unmeasured, smoke=_smoke(), visual_status="passed", review=review)
    assert result["passed"] is False
    assert result["contamination"]["reasons"]
    assert result["review"]["passed"] is True


def test_sparse_gate_does_not_fabricate_pre_fusion_contamination_zeroes():
    result = sparse_gate(
        ring_coverage={"geo_g8": {"registered": 1}},
        cross_ring_connections=1,
        board_point_fraction=None,
        cloth_point_fraction=None,
        trajectory_status="pending",
        visual_status="pending",
    )
    assert result["passed"] is False
    assert result["contamination_measurement_status"] == "not_measurable_before_fusion"
    assert result["board_point_fraction"] is None
    assert result["cloth_point_fraction"] is None
    assert result["checks"]["board_measurement_available"] is False
    assert result["checks"]["cloth_measurement_available"] is False


@pytest.mark.parametrize(
    ("key", "value", "check"),
    [
        ("board_slab_detected", True, "board_slab_absent"),
        ("pedestal_board_webbing_detected", "unknown", "pedestal_board_webbing_absent"),
        ("cloth_or_background_structure_detected", None, "cloth_or_background_absent"),
        ("vessel_identity_confirmed", False, "vessel_identity_confirmed"),
        ("vessel_identity_confirmed", "unknown", "vessel_identity_confirmed"),
    ],
)
def test_contamination_decisions_fail_closed(key: str, value: object, check: str):
    metrics = _metrics()
    metrics["contamination_findings"][key] = value  # type: ignore[index]
    result = dense_gate(metrics, smoke=_smoke(), visual_status="passed", review=None)
    assert result["passed"] is False
    assert result["checks"][check] is False


def test_explicit_zero_requires_measurement_provenance_and_matching_inputs(tmp_path: Path):
    metrics = _metrics()
    metrics["contamination_metrics"]["board_point_fraction"] = 0.0  # type: ignore[index]
    result = dense_gate(
        metrics,
        smoke=_smoke(),
        visual_status="passed",
        review=_review(tmp_path),
    )
    assert result["passed"] is False
    assert "board_point_fraction must be an explicit measured-metric object" in result["contamination"]["reasons"]

    metrics = _metrics()
    metrics["board_point_fraction"] = 0.25
    result = dense_gate(
        metrics,
        smoke=_smoke(),
        visual_status="passed",
        review=_review(tmp_path),
    )
    assert result["passed"] is False
    assert result["checks"]["board_fraction_inputs_consistent"] is False

    metrics = _metrics()
    metrics["cloth_point_fraction"] = "not-measured"
    result = dense_gate(
        metrics,
        smoke=_smoke(),
        visual_status="passed",
        review=_review(tmp_path / "views"),
    )
    assert result["passed"] is False
    assert result["checks"]["cloth_fraction_inputs_consistent"] is False


def test_dense_finalization_rejects_bad_preview_hash_self_approval_and_pending_review(tmp_path: Path):
    postfusion = _postfusion_report(tmp_path / "postfusion")
    report_path = tmp_path / "dense_gate.json"
    report_path.write_text(
        json.dumps({"metrics": _metrics(), "smoke": _smoke(), "fused_path": postfusion["fused_path"]}),
        encoding="utf-8",
    )

    duplicate = _review(tmp_path / "views0")
    duplicate["previews"]["side"] = duplicate["previews"]["front"]
    with pytest.raises(ValueError, match="distinct files"):
        finalize_dense_visual_gate(
            report_path,
            visual_status="passed",
            review_basis=["Viewed all four semantic views."],
            review=duplicate,
            postfusion_report=postfusion,
        )

    tampered = _review(tmp_path / "views")
    preview_path = Path(tampered["previews"]["front"]["path"])  # type: ignore[index]
    preview_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="do not match"):
        finalize_dense_visual_gate(
            report_path,
            visual_status="passed",
            review_basis=["Viewed all four semantic views."],
            review=tampered,
            postfusion_report=postfusion,
        )

    self_approved = _review(tmp_path / "views2", reviewer="same-person", approver="Same-Person")
    with pytest.raises(ValueError, match="self-approval"):
        finalize_dense_visual_gate(
            report_path,
            visual_status="passed",
            review_basis=["Viewed all four semantic views."],
            review=self_approved,
            postfusion_report=postfusion,
        )

    pending = _review(tmp_path / "views3", approval_status="pending")
    with pytest.raises(ValueError, match="approval status"):
        finalize_dense_visual_gate(
            report_path,
            visual_status="passed",
            review_basis=["Viewed all four semantic views."],
            review=pending,
            postfusion_report=postfusion,
        )

    complete = _review(tmp_path / "views4")
    with pytest.raises(ValueError, match="review_basis"):
        finalize_dense_visual_gate(
            report_path,
            visual_status="passed",
            review_basis=[""],
            review=complete,
        )


def test_dense_finalization_accepts_only_complete_independent_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import v4_dense

    _install_json_writer(monkeypatch, v4_dense)
    postfusion = _postfusion_report(tmp_path / "postfusion")
    report_path = tmp_path / "dense_gate.json"
    report_path.write_text(
        json.dumps({"metrics": _metrics(), "smoke": _smoke(), "fused_path": postfusion["fused_path"]}),
        encoding="utf-8",
    )
    review = _review(tmp_path / "views")
    report = finalize_dense_visual_gate(
        report_path,
        visual_status="passed",
        review_basis=["Viewed front, quarter, side, and top-oblique fused evidence."],
        review=review,
        postfusion_report=postfusion,
    )
    assert report["status"] == "complete"
    assert report["visual_gate"]["reviewer"] == "independent-reviewer"
    assert report["visual_gate"]["approver"] == "controller"
    assert set(report["visual_gate"]["previews"]) == {"front", "quarter", "side", "top_oblique"}


def test_postfusion_evidence_gate_fails_closed_on_missing_or_status_only_evidence(tmp_path: Path):
    missing = postfusion_evidence_gate(None)
    assert missing["passed"] is False
    assert missing["checks"]["report_object_present"] is False

    report = _postfusion_report(tmp_path / "status-only")
    report["contamination"] = {"status": "measured"}
    report["ring_transition_audit"] = {"status": "passed"}
    checked = postfusion_evidence_gate(report, fused_path=Path(str(report["fused_path"])))
    assert checked["passed"] is False
    assert checked["checks"]["fused_cloud_multiview_contamination"] is False
    assert checked["checks"]["ring_transition_evidence_verified"] is False


def test_postfusion_evidence_gate_rejects_legacy_unextended_fusion_masks(tmp_path: Path):
    report = _postfusion_report(tmp_path / "legacy-mask")
    report["contamination"]["evidence"]["mask_resolution"] = {
        "resolver": "colmap_image_name_plus_png",
        "resolved_count": 4,
        "missing_count": 0,
        "resolution_counts": {"legacy_unextended_image_name": 4},
        "legacy_fallback_count": 4,
    }
    checked = postfusion_evidence_gate(report, fused_path=Path(str(report["fused_path"])))
    assert checked["passed"] is False
    assert checked["checks"]["fusion_mask_resolution_verified"] is False
    assert any("COLMAP's image-name-plus-.png resolver" in reason for reason in checked["reasons"])


def test_postfusion_evidence_gate_requires_exact_accepted_sparse_hash(tmp_path: Path):
    report = _postfusion_report(tmp_path / "sparse-lineage")
    report.pop("sparse_lineage")
    missing = postfusion_evidence_gate(report, fused_path=Path(str(report["fused_path"])))
    assert missing["passed"] is False
    assert missing["checks"]["accepted_sparse_lineage_verified"] is False

    report = _postfusion_report(tmp_path / "sparse-lineage-mismatch")
    mismatched = postfusion_evidence_gate(
        report,
        fused_path=Path(str(report["fused_path"])),
        expected_sparse_model_sha256="c" * 64,
    )
    assert mismatched["passed"] is False
    assert mismatched["checks"]["accepted_sparse_lineage_verified"] is False
    assert any("does not match the accepted repaired sparse model hash" in reason for reason in mismatched["reasons"])


def test_postfusion_evidence_gate_requires_all_eight_anatomy_views(tmp_path: Path):
    report = _postfusion_report(tmp_path / "anatomy")
    checked = postfusion_evidence_gate(report, fused_path=Path(str(report["fused_path"])))
    assert checked["checks"]["anatomy_preview_evidence_verified"] is True
    assert checked["passed"] is True

    report = _postfusion_report(tmp_path / "anatomy-missing")
    report["anatomy_previews"].pop("finial")  # type: ignore[index]
    checked = postfusion_evidence_gate(report, fused_path=Path(str(report["fused_path"])))
    assert checked["passed"] is False
    assert checked["checks"]["anatomy_preview_evidence_verified"] is False
    assert any("eight hashable anatomy" in reason for reason in checked["reasons"])


def test_eight_renamed_whole_object_projections_cannot_satisfy_anatomy_gate(tmp_path: Path):
    report = _postfusion_report(tmp_path / "whole-object-renamed")
    for item in report["anatomy_previews"].values():  # type: ignore[union-attr]
        item["region_crop"]["projection_scope"] = "whole_object"  # type: ignore[index]
        item["region_crop"]["whole_object_projection"] = True  # type: ignore[index]
    checked = postfusion_evidence_gate(report, fused_path=Path(str(report["fused_path"])))
    assert checked["passed"] is False
    assert checked["checks"]["anatomy_preview_evidence_verified"] is False
    assert checked["checks"]["anatomy_region_measurements_verified"] is True


def test_anatomy_region_gate_rejects_unresolved_finial_even_with_hashes(tmp_path: Path):
    report = _postfusion_report(tmp_path / "unresolved-finial")
    report["contamination"]["anatomy"]["finial_shape"]["resolved_narrow_top_element"] = False  # type: ignore[index]
    report["contamination"]["anatomy"]["passed"] = False  # type: ignore[index]
    report["contamination"]["anatomy"]["status"] = "failed"  # type: ignore[index]
    report["contamination"]["anatomy"]["no_major_vessel_scale_holes"] = False  # type: ignore[index]
    checked = postfusion_evidence_gate(report, fused_path=Path(str(report["fused_path"])))
    assert checked["passed"] is False
    assert checked["checks"]["anatomy_region_measurements_verified"] is False
    assert any("anatomy region measurements" in reason for reason in checked["reasons"])


def test_dense_finalization_rejects_a_visual_pass_without_postfusion_report(tmp_path: Path):
    postfusion = _postfusion_report(tmp_path / "postfusion")
    report_path = tmp_path / "dense_gate.json"
    report_path.write_text(
        json.dumps({"metrics": _metrics(), "smoke": _smoke(), "fused_path": postfusion["fused_path"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="post-fusion evidence report"):
        finalize_dense_visual_gate(
            report_path,
            visual_status="passed",
            review_basis=["Viewed all four semantic views."],
            review=_review(tmp_path / "review"),
        )


def test_raw_gate_requires_review_and_matching_measured_quantitative_inputs(tmp_path: Path):
    metrics = _metrics()
    mesh_metrics = _mesh_metrics()
    review = _review(tmp_path)
    result = raw_visual_gate(
        fused_metrics=metrics,
        mesh_metrics=mesh_metrics,
        visual_status="passed",
        required_regions=review["regions"],  # type: ignore[arg-type]
        board_point_fraction=0.0,
        cloth_point_fraction=0.0,
        review=review,
    )
    assert result["passed"] is True

    no_review = raw_visual_gate(
        fused_metrics=metrics,
        mesh_metrics=mesh_metrics,
        visual_status="passed",
        required_regions=review["regions"],  # type: ignore[arg-type]
        board_point_fraction=0.0,
        cloth_point_fraction=0.0,
    )
    assert no_review["passed"] is False
    assert no_review["checks"]["visual_review"] is False

    mismatch = raw_visual_gate(
        fused_metrics=metrics,
        mesh_metrics=mesh_metrics,
        visual_status="passed",
        required_regions=review["regions"],  # type: ignore[arg-type]
        board_point_fraction=0.5,
        cloth_point_fraction=0.0,
        review=review,
    )
    assert mismatch["passed"] is False
    assert mismatch["checks"]["raw_board_fraction_input_matches_measurement"] is False


def test_raw_gate_rejects_unknown_webbing_even_when_quantitative_fraction_is_low(tmp_path: Path):
    metrics = _metrics()
    metrics["contamination_findings"]["pedestal_board_webbing_detected"] = "unknown"  # type: ignore[index]
    review = _review(tmp_path)
    review["contamination_findings"]["pedestal_board_webbing_detected"] = "unknown"  # type: ignore[index]
    result = raw_visual_gate(
        fused_metrics=metrics,
        mesh_metrics=_mesh_metrics(),
        visual_status="passed",
        required_regions=review["regions"],  # type: ignore[arg-type]
        board_point_fraction=0.0,
        cloth_point_fraction=0.0,
        review=review,
    )
    assert result["passed"] is False
    assert result["contamination"]["checks"]["pedestal_board_webbing_absent"] is False


def test_raw_finalization_fails_closed_and_accepts_complete_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import v4_mesh

    _install_json_writer(monkeypatch, v4_mesh)
    report_path = tmp_path / "raw_visual_gate.json"
    report = {
        "fused_metrics": _metrics(),
        "mesh_metrics": _mesh_metrics(),
        "visual_gate": {"status": "pending"},
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="review status must be passed"):
        finalize_raw_visual_gate(
            report_path,
            review={},
            review_basis=["Viewed the raw Poisson mesh."],
        )

    review = _review(tmp_path / "views")
    accepted = finalize_raw_visual_gate(
        report_path,
        review=review,
        review_basis=["Viewed the unmodified raw Poisson mesh from four semantic views."],
    )
    assert accepted["status"] == "complete"
    assert accepted["gate"]["passed"] is True
    assert accepted["visual_gate"]["contamination_findings"]["vessel_identity_confirmed"] is True
