from collections import Counter
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest

import final_reference_evidence as reference_evidence
from final_model_io import sha256_file
from final_reference_evidence import (
    align_detail_pair,
    build_component_mask_evidence,
    build_geometry_mask_evidence,
    build_ornament_inventory,
    extract_ornament_crop,
    refine_geometry_mask,
    FinalReferenceView,
    Landmark,
    build_sift_pair_diagnostic,
    load_reviewed_reference_views,
    load_step13_registration,
    make_labeled_contact_sheet,
    seed_landmarks_from_mask,
    select_canonical_geometry_views,
    split_component_masks,
    validate_landmark_set,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def reference_views():
    return load_reviewed_reference_views(ROOT)


def test_reference_catalog_preserves_real_reviewed_view_categories(reference_views):
    views = reference_views

    assert len(views) == 36
    assert Counter(view.view_category for view in views) == {
        "normal_side": 8,
        "low_angle_pedestal": 8,
        "elevated_oblique": 8,
        "top_down_rim": 8,
        "oblique_detail": 4,
    }
    assert [view.selected_index for view in views] == [
        3,
        10,
        19,
        28,
        39,
        50,
        62,
        72,
        76,
        82,
        90,
        98,
        106,
        114,
        128,
        142,
        148,
        151,
        154,
        165,
        177,
        180,
        188,
        200,
        206,
        209,
        212,
        221,
        230,
        233,
        243,
        255,
        267,
        268,
        278,
        288,
    ]


def test_reference_catalog_keeps_validated_masks_and_registration_truth(reference_views):
    views = reference_views
    by_index = {view.selected_index: view for view in views}

    for view in views:
        assert view.reviewed_mask_path.is_file()
        assert sha256_file(view.reviewed_mask_path) == view.reviewed_mask_sha256
        assert view.cnn_prediction_path.is_file()
        assert view.reconstruction_mask_path.is_file()

    for selected_index in (267, 268, 278, 288):
        assert by_index[selected_index].step13_registered is False
        assert by_index[selected_index].step13_image_id is None

    assert by_index[3].step13_registered is True
    assert by_index[3].step13_image_id is not None


def test_step13_registration_matches_best_model_and_csv():
    registration = load_step13_registration(ROOT)

    assert len(registration) == 288
    assert sum(image_id is not None for image_id in registration.values()) == 266
    assert all(registration[index] is None for index in (267, 268, 278, 288))


def test_labeled_contact_sheet_is_deterministic_and_keeps_cell_size(tmp_path: Path):
    wide = tmp_path / "003_wide.png"
    tall = tmp_path / "090_tall.png"
    Image.new("RGB", (80, 40), (190, 20, 20)).save(wide)
    Image.new("RGB", (40, 80), (20, 20, 190)).save(tall)
    output = tmp_path / "sheet.png"

    result = make_labeled_contact_sheet(
        (wide, tall),
        ("003 | wide.png", "090 | tall.png"),
        output,
        cell_size=(120, 120),
        columns=2,
    )
    first_hash = sha256_file(result)
    make_labeled_contact_sheet(
        (wide, tall),
        ("003 | wide.png", "090 | tall.png"),
        output,
        cell_size=(120, 120),
        columns=2,
    )

    with Image.open(output) as sheet:
        assert sheet.size == (240, 120)
    assert sha256_file(output) == first_hash


def test_labeled_contact_sheet_rejects_mismatched_labels(tmp_path: Path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (20, 20), (1, 2, 3)).save(image_path)

    with pytest.raises(ValueError, match="one label per image"):
        make_labeled_contact_sheet((image_path,), (), tmp_path / "sheet.png")


def test_canonical_geometry_set_uses_four_registered_views_per_category(
    reference_views,
):
    selected = select_canonical_geometry_views(reference_views)

    assert [row["selected_index"] for row in selected] == [
        3,
        19,
        50,
        72,
        90,
        114,
        128,
        142,
        148,
        165,
        188,
        200,
        206,
        221,
        243,
        255,
    ]
    assert Counter(row["view_category"] for row in selected) == {
        "normal_side": 4,
        "low_angle_pedestal": 4,
        "elevated_oblique": 4,
        "top_down_rim": 4,
    }
    assert all(row["step13_registered"] is True for row in selected)


def test_canonical_geometry_set_replaces_failed_preferred_with_same_category(
    reference_views,
):
    views = tuple(
        replace(view, step13_registered=False, step13_image_id=None)
        if view.selected_index == 3
        else view
        for view in reference_views
    )

    selected = select_canonical_geometry_views(views)

    assert selected[0]["selected_index"] == 10
    assert selected[0]["view_category"] == "normal_side"
    assert selected[0]["replacement_for"] == 3
    assert "replacement" in selected[0]["selection_reason"]


def test_sift_diagnostic_reuses_step6_interfaces(monkeypatch, tmp_path: Path):
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    Image.new("RGB", (40, 40), (60, 70, 80)).save(first_path)
    Image.new("RGB", (40, 40), (65, 75, 85)).save(second_path)
    keypoint = cv2.KeyPoint(10.0, 12.0, 1.0)
    features = reference_evidence.SiftFeatures(
        analysis_image=np.zeros((40, 40, 3), dtype=np.uint8),
        keypoints=(keypoint,),
        descriptors=np.ones((1, 128), dtype=np.float32),
        scale=reference_evidence.ImageScale(
            original_size=(40, 40),
            analysis_size=(40, 40),
            scale_x_to_original=1.0,
            scale_y_to_original=1.0,
        ),
        status="ok",
    )
    match = cv2.DMatch(0, 0, 0.1)
    geometry = reference_evidence.GeometryMatchResult(
        fundamental_matrix=None,
        candidate_matches=(match,),
        inlier_mask=np.zeros(1, dtype=bool),
        inlier_matches=(),
        points_a=np.asarray([[10.0, 12.0]], dtype=np.float32),
        points_b=np.asarray([[10.0, 12.0]], dtype=np.float32),
        status="insufficient_geometry",
    )
    calls = {"extract": 0, "match": 0, "estimate": 0}

    def fake_extract(image, config):
        calls["extract"] += 1
        return features

    def fake_match(first, second, ratio_threshold):
        calls["match"] += 1
        return (match,)

    def fake_estimate(first, second, matches, **kwargs):
        calls["estimate"] += 1
        return geometry

    monkeypatch.setattr(reference_evidence, "extract_sift", fake_extract)
    monkeypatch.setattr(reference_evidence, "match_sift", fake_match)
    monkeypatch.setattr(
        reference_evidence, "estimate_fundamental_geometry", fake_estimate
    )
    output = tmp_path / "diagnostic.png"

    report = build_sift_pair_diagnostic(first_path, second_path, output)

    assert calls == {"extract": 2, "match": 1, "estimate": 1}
    assert report["candidate_matches"] == 1
    assert report["ransac_inliers"] == 0
    assert report["fundamental_matrix_status"] == "insufficient_geometry"
    assert report["median_sampson_error"] is None
    assert output.is_file()


def _fake_sift_features(count: int, *, offset: tuple[float, float] = (0.0, 0.0)):
    keypoints = tuple(
        cv2.KeyPoint(
            float(8 + (index % 8) * 4) + offset[0],
            float(8 + (index // 8) * 4) + offset[1],
            1.0,
        )
        for index in range(count)
    )
    return reference_evidence.SiftFeatures(
        analysis_image=np.zeros((48, 48, 3), dtype=np.uint8),
        keypoints=keypoints,
        descriptors=np.ones((count, 128), dtype=np.float32),
        scale=reference_evidence.ImageScale(
            original_size=(48, 48),
            analysis_size=(48, 48),
            scale_x_to_original=1.0,
            scale_y_to_original=1.0,
        ),
        status="ok",
    )


def test_detail_alignment_accepts_strong_sift_without_loading_learned_frontend(
    monkeypatch, tmp_path: Path
):
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    Image.new("RGB", (48, 48), (40, 50, 60)).save(first_path)
    Image.new("RGB", (48, 48), (45, 55, 65)).save(second_path)
    first_features = _fake_sift_features(40)
    second_features = _fake_sift_features(40, offset=(2.0, 1.0))
    matches = tuple(cv2.DMatch(i, i, 0.1) for i in range(40))
    extracted = iter((first_features, second_features))
    monkeypatch.setattr(reference_evidence, "extract_sift", lambda image, config: next(extracted))
    monkeypatch.setattr(
        reference_evidence,
        "match_sift",
        lambda first, second, ratio_threshold: matches,
    )
    monkeypatch.setattr(
        reference_evidence,
        "load_frontend",
        lambda *args, **kwargs: pytest.fail("learned frontend must not load"),
    )

    decision = align_detail_pair(first_path, second_path, tmp_path / "sift.png")

    assert decision.method == "sift"
    assert decision.match_count == 40
    assert decision.inlier_count >= 15
    assert decision.homography is not None


def test_detail_alignment_falls_back_to_frozen_aliked_lightglue(
    monkeypatch, tmp_path: Path
):
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    Image.new("RGB", (48, 48), (40, 50, 60)).save(first_path)
    Image.new("RGB", (48, 48), (45, 55, 65)).save(second_path)
    weak = _fake_sift_features(5)
    monkeypatch.setattr(reference_evidence, "extract_sift", lambda image, config: weak)
    monkeypatch.setattr(
        reference_evidence,
        "match_sift",
        lambda first, second, ratio_threshold: tuple(
            cv2.DMatch(i, i, 0.1) for i in range(5)
        ),
    )
    bundle = object()
    monkeypatch.setattr(reference_evidence, "load_frontend", lambda config: bundle)
    points = np.asarray(
        [[8 + (index % 8) * 4, 8 + (index // 8) * 4] for index in range(40)],
        dtype=np.float32,
    )
    learned = iter(
        (
            reference_evidence.ImageFeatures(
                keypoints=points,
                descriptors=np.ones((40, 128), dtype=np.float32),
                scores=np.ones(40, dtype=np.float32),
                image_size=np.asarray([48, 48], dtype=np.float32),
            ),
            reference_evidence.ImageFeatures(
                keypoints=points + np.asarray([2.0, 1.0], dtype=np.float32),
                descriptors=np.ones((40, 128), dtype=np.float32),
                scores=np.ones(40, dtype=np.float32),
                image_size=np.asarray([48, 48], dtype=np.float32),
            ),
        )
    )
    monkeypatch.setattr(
        reference_evidence,
        "extract_image_features",
        lambda path, actual_bundle, config: next(learned),
    )
    monkeypatch.setattr(
        reference_evidence,
        "match_feature_pair",
        lambda first, second, actual_bundle: np.column_stack(
            (np.arange(40), np.arange(40))
        ),
    )

    decision = align_detail_pair(first_path, second_path, tmp_path / "learned.png")

    assert decision.method == "aliked_lightglue"
    assert decision.match_count == 40
    assert decision.inlier_count >= 15
    assert decision.homography is not None
    assert "SIFT rejected" in decision.reason


def _synthetic_vessel_mask() -> np.ndarray:
    mask = np.zeros((400, 200), dtype=np.uint8)
    for y in range(20, 381):
        fraction = (y - 20) / 360.0
        if fraction < 0.07:
            radius = 10
        elif fraction < 0.16:
            radius = int(10 + (fraction - 0.07) / 0.09 * 16)
        elif fraction < 0.46:
            radius = 14
        elif fraction < 0.67:
            radius = int(14 + np.sin((fraction - 0.46) / 0.21 * np.pi) * 42)
        elif fraction < 0.80:
            radius = 64
        elif fraction < 0.88:
            radius = 28
        else:
            radius = int(28 + (fraction - 0.88) / 0.12 * 26)
        mask[y, 100 - radius : 101 + radius] = 255
    return mask


def test_landmark_seeding_records_source_and_normalized_coordinates():
    landmarks = seed_landmarks_from_mask(
        _synthetic_vessel_mask(), view_category="normal_side"
    )

    validate_landmark_set(landmarks, (200, 400), view_category="normal_side")
    by_name = {landmark.name: landmark for landmark in landmarks}
    assert len(by_name) == 24
    assert by_name["axis_top"].y < by_name["axis_bottom"].y
    assert by_name["axis_bottom"].y >= by_name["foot_left"].y
    assert by_name["axis_bottom"].y >= by_name["foot_right"].y
    assert by_name["globe_max_left"].x < by_name["globe_max_right"].x
    assert by_name["foot_left"].normalized_x == pytest.approx(
        by_name["foot_left"].x / 199.0
    )
    assert all(landmark.annotation_method == "cv_seeded" for landmark in landmarks)


def test_landmark_seeding_marks_a_cropped_axis_endpoint_low_confidence():
    mask = _synthetic_vessel_mask()
    mask[380:, 90:111] = 255

    landmarks = seed_landmarks_from_mask(mask, view_category="normal_side")
    by_name = {landmark.name: landmark for landmark in landmarks}

    assert by_name["axis_bottom"].y == 399.0
    assert by_name["axis_bottom"].confidence == "low"


def test_landmark_validation_rejects_out_of_bounds_point():
    landmarks = list(
        seed_landmarks_from_mask(
            _synthetic_vessel_mask(), view_category="normal_side"
        )
    )
    landmarks[0] = replace(landmarks[0], x=-1.0, normalized_x=-0.01)

    with pytest.raises(ValueError, match="outside image bounds"):
        validate_landmark_set(landmarks, (200, 400), view_category="normal_side")


def test_component_masks_partition_the_reviewed_whole_mask():
    whole = _synthetic_vessel_mask()
    landmarks = seed_landmarks_from_mask(whole, view_category="normal_side")

    components = split_component_masks(whole, landmarks)

    assert set(components) >= {
        "main_vessel",
        "receiving_bowl_pedestal",
        "lid_finial",
        "neck",
        "globe",
    }
    assert all(mask.dtype == np.uint8 and mask.shape == whole.shape for mask in components.values())
    union = np.zeros_like(whole, dtype=bool)
    for mask in components.values():
        assert np.count_nonzero(mask & cv2.bitwise_not(whole)) == 0
        if mask is components["main_vessel"] or mask is components["receiving_bowl_pedestal"]:
            continue
    union |= components["main_vessel"] > 0
    union |= components["receiving_bowl_pedestal"] > 0
    assert np.array_equal(union, whole > 0)
    assert not np.any(
        (components["main_vessel"] > 0)
        & (components["receiving_bowl_pedestal"] > 0)
    )


def test_component_masks_preserve_legitimate_disconnected_upper_geometry():
    whole = _synthetic_vessel_mask()
    landmarks = seed_landmarks_from_mask(whole, view_category="normal_side")
    by_name = {landmark.name: landmark for landmark in landmarks}
    gap_y = int(round((by_name["lid_lower_left"].y + by_name["neck_top_left"].y) * 0.5))
    whole[max(0, gap_y - 2) : gap_y + 3] = 0
    assert np.count_nonzero(whole[:gap_y]) > 0
    assert np.count_nonzero(whole[gap_y + 3 :]) > 0

    components = split_component_masks(whole, landmarks)

    union = (components["main_vessel"] > 0) | (components["receiving_bowl_pedestal"] > 0)
    assert np.array_equal(union, whole > 0)
    assert np.count_nonzero(components["lid_finial"]) > 0


def test_ornament_crop_rejects_bbox_outside_source(tmp_path: Path):
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 48), (100, 90, 70)).save(source)

    with pytest.raises(ValueError, match="inside source"):
        extract_ornament_crop(source, (0, 0, 65, 48), tmp_path / "crop.png")


def test_ornament_inventory_is_source_traceable_and_omits_chain():
    inventory = build_ornament_inventory()

    family_ids = {family.family_id for family in inventory["families"]}
    assert {
        "ORB_GLOBE_CROSSHATCH",
        "ORB_GLOBE_HERO_MOTIF",
        "ORB_NECK_LOTUS",
        "ORB_BOWL_FLAME_BAND",
    } <= family_ids
    assert inventory["chain"] == {
        "supported": False,
        "decision": "omit_from_v2",
        "evidence_requirement": "two_independent_source_views",
    }
    assert all(family.primary_view_indices for family in inventory["families"])


def test_geometry_mask_refinement_is_deterministic_and_bounded():
    height, width = 240, 160
    image = np.full((height, width, 3), 55, dtype=np.uint8)
    cv2.ellipse(image, (80, 118), (37, 88), 0, 0, 360, (45, 185, 235), -1)
    reviewed = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(reviewed, (80, 118), (44, 96), 0, 0, 360, 255, -1)
    original_reviewed = reviewed.copy()

    first = refine_geometry_mask(image, reviewed, maximum_dimension=240)
    second = refine_geometry_mask(image, reviewed, maximum_dimension=240)

    assert np.array_equal(first, second)
    assert np.array_equal(reviewed, original_reviewed)
    assert first.dtype == np.uint8 and first.shape == reviewed.shape
    assert 0 < np.count_nonzero(first) < np.count_nonzero(reviewed)
    allowed = cv2.dilate(reviewed, np.ones((17, 17), dtype=np.uint8)) > 0
    assert not np.any((first > 0) & ~allowed)


def test_geometry_mask_refinement_removes_low_saturation_connected_background():
    height, width = 260, 180
    image = np.full((height, width, 3), 70, dtype=np.uint8)
    cv2.ellipse(image, (90, 112), (40, 88), 0, 0, 360, (30, 170, 235), -1)
    cv2.rectangle(image, (76, 198), (104, 259), (105, 105, 105), -1)

    reviewed = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(reviewed, (90, 112), (45, 94), 0, 0, 360, 255, -1)
    cv2.rectangle(reviewed, (74, 194), (106, 259), 255, -1)

    refined = refine_geometry_mask(image, reviewed, maximum_dimension=260)

    object_region = refined[:205]
    connected_background = refined[215:]
    assert np.count_nonzero(object_region) > 8_000
    assert np.count_nonzero(connected_background) < 250


def test_geometry_mask_refinement_trims_narrow_reflection_tail_below_foot():
    height, width = 280, 190
    image = np.full((height, width, 3), 65, dtype=np.uint8)
    cv2.ellipse(image, (95, 105), (42, 82), 0, 0, 360, (25, 165, 230), -1)
    cv2.rectangle(image, (55, 180), (135, 230), (30, 155, 220), -1)
    cv2.rectangle(image, (78, 231), (112, 270), (75, 125, 155), -1)

    reviewed = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(reviewed, (95, 105), (47, 88), 0, 0, 360, 255, -1)
    cv2.rectangle(reviewed, (51, 176), (139, 232), 255, -1)
    cv2.rectangle(reviewed, (74, 231), (116, 272), 255, -1)

    refined = refine_geometry_mask(image, reviewed, maximum_dimension=280)

    assert np.count_nonzero(refined[180:231]) > 2_500
    assert np.count_nonzero(refined[245:]) < 150


def test_component_mask_evidence_creates_fresh_output_directory(tmp_path: Path):
    source = tmp_path / "preprocessing" / "pycolmap_input" / "images" / "x.png"
    reviewed = tmp_path / "reviewed.png"
    source.parent.mkdir(parents=True)
    whole = _synthetic_vessel_mask()
    Image.fromarray(np.dstack((whole, whole, whole))).save(source)
    Image.fromarray(whole).save(reviewed)
    landmarks = seed_landmarks_from_mask(whole, view_category="normal_side")
    view = FinalReferenceView(
        selected_index=1,
        filename="x.png",
        split="train",
        view_category="normal_side",
        quality_condition="normal",
        source_sha256=sha256_file(source),
        reviewed_mask_path=reviewed,
        reviewed_mask_sha256=sha256_file(reviewed),
        cnn_prediction_path=reviewed,
        reconstruction_mask_path=reviewed,
        step13_registered=True,
        step13_image_id=1,
    )
    report = {
        "views": [
            {
                "selected_index": 1,
                "landmarks": [reference_evidence.asdict(item) for item in landmarks],
            }
        ]
    }

    result = build_component_mask_evidence(tmp_path, (view,), report)

    assert result["view_count"] == 1
    assert (tmp_path / "reconstruction" / "reference_assisted_v2" / "evidence" / "component_masks" / "001_main_vessel.png").is_file()


def test_source_review_mask_patch_is_bounded_and_does_not_change_input():
    mask = np.full((100, 80), 255, np.uint8)
    before = mask.copy()
    review = {"display_size": [80, 100], "mask_patches": [
        {"rows": [80, 99], "polygon": [[20, 80], [20, 90], [60, 90], [60, 80]]}
    ]}
    result = reference_evidence.apply_reviewed_mask_patches(mask, review)
    np.testing.assert_array_equal(mask, before)
    np.testing.assert_array_equal(result[:80], before[:80])
    assert result[85, 40] == 255
    assert not np.any(result[92:])


def test_source_review_does_not_treat_hidden_points_as_measured(reference_views):
    view = next(view for view in reference_views if view.selected_index == 90)
    review = reference_evidence.load_source_review(ROOT, view)
    landmarks = reference_evidence.reviewed_landmarks(review, (3072, 4080), view_category=view.view_category)
    by_name = {record.name: record for record in landmarks}
    assert by_name["finial_top"].confidence == "low"
    assert by_name["globe_bottom_transition"].confidence == "low"
    assert by_name["neck_base_left"].confidence == "high"
    assert by_name["axis_bottom"].confidence == "low"
    assert by_name["lid_max_left"].y == pytest.approx(62 * 4080 / 900)
    assert all(
        record.annotation_method == "cv_seeded_visually_reviewed"
        for record in landmarks
    )
    with pytest.raises(ValueError, match="hash/name mismatch"):
        reference_evidence.load_source_review(ROOT, replace(view, source_sha256="bad"))


def test_reference_visual_review_requires_exact_board_hashes(tmp_path: Path):
    v2_root = tmp_path / "reconstruction" / "reference_assisted_v2"
    report_dir = v2_root / "reports"
    board_dir = v2_root / "evidence" / "annotations"
    report_dir.mkdir(parents=True)
    board_dir.mkdir(parents=True)
    landmark_board = board_dir / "landmark_overlays_board.png"
    landmark_board.write_bytes(b"landmark-board")
    geometry_report = {"overlay_board_sha256": "geometry-hash"}
    review = {
        "accepted": True,
        "geometry_mask_board_sha256": "geometry-hash",
        "landmark_board_sha256": sha256_file(landmark_board),
    }
    review_path = report_dir / "reference_visual_review.json"
    review_path.write_text(__import__("json").dumps(review), encoding="utf-8")

    accepted = reference_evidence.load_reference_visual_review(
        tmp_path, geometry_report, landmark_board
    )
    assert accepted is not None
    assert accepted["accepted"] is True

    landmark_board.write_bytes(b"changed-board")
    assert (
        reference_evidence.load_reference_visual_review(
            tmp_path, geometry_report, landmark_board
        )
        is None
    )
