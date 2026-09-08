import inspect
from pathlib import Path

import pytest

import build_final_model_blender as builder
import final_blender_finish as finisher


def test_blender_52_uses_current_eevee_engine_identifier() -> None:
    assert builder.RENDER_ENGINE == "BLENDER_EEVEE"


def test_profile_measurements_record_normalized_neck_globe_ratio() -> None:
    profiles = builder.validate_profile_payload(_profiles_payload())

    measurements = builder.profile_measurements(profiles)

    assert measurements["normalized_set_height"] == 1.0
    assert measurements["globe_max_diameter"] == pytest.approx(0.4)
    assert measurements["neck_length"] == pytest.approx(1.0)
    assert measurements["neck_length_to_globe_diameter"] == pytest.approx(2.5)
    assert measurements["profile_import_max_abs_delta"] == 0.0


def _profiles_payload() -> dict:
    names = builder.REQUIRED_PROFILES
    return {
        "coordinate_contract": {
            "axis": "+Z",
            "normalized_set_height": 1.0,
            "scale_status": "relative_no_physical_measurement",
        },
        "profiles": {
            name: [[0.0, 0.1], [0.5, 0.2], [1.0, 0.1]]
            for name in names
        },
        "profile_provenance": {
            name: {"measurement_status": "measured", "source_views": [1, 2]}
            for name in names
        },
    }


def test_profile_payload_requires_normalized_positive_monotonic_profiles():
    payload = _profiles_payload()

    result = builder.validate_profile_payload(payload)

    assert tuple(result) == builder.REQUIRED_PROFILES
    assert result["globe"] == ((0.0, 0.1), (0.5, 0.2), (1.0, 0.1))

    descending = _profiles_payload()
    descending["profiles"]["globe"] = [[0.5, 0.2], [0.4, 0.1]]
    with pytest.raises(ValueError, match="strictly increasing"):
        builder.validate_profile_payload(descending)

    negative = _profiles_payload()
    negative["profiles"]["lid"][1][1] = -0.01
    with pytest.raises(ValueError, match="negative radius"):
        builder.validate_profile_payload(negative)


def test_closed_shell_profile_joins_outer_and_reversed_inner_without_axis_cap():
    outer = ((0.2, 0.12), (0.4, 0.20))
    inner = ((0.2, 0.09), (0.4, 0.17))

    shell = builder.closed_shell_profile(outer, inner)

    assert shell == (
        (0.2, 0.12),
        (0.4, 0.20),
        (0.4, 0.17),
        (0.2, 0.09),
    )


def test_source_reviewed_finalization_profiles_make_a_compact_round_globe():
    globe = tuple((0.379 + index * 0.029, 0.18 + index * 0.002) for index in range(9))
    shoulder = tuple((0.482 + index * 0.019, 0.17 - index * 0.015) for index in range(9))
    bowl_outer = tuple((0.205 + index * 0.030, 0.10 + index * 0.0135) for index in range(9))
    profiles = {"globe": globe, "shoulder": shoulder, "bowl_outer": bowl_outer}

    corrected = builder.source_reviewed_finalization_profiles(profiles)

    assert [z for z, _ in corrected["globe"]] == [z for z, _ in globe]
    assert [z for z, _ in corrected["shoulder"]] == [z for z, _ in shoulder]
    target_max = max(radius for _, radius in bowl_outer) / builder.FINAL_GLOBE_BOWL_RADIUS_RATIO
    assert max(radius for _, radius in corrected["globe"]) == pytest.approx(target_max)
    assert corrected["globe"][0][1] < globe[0][1] * 0.5
    assert corrected["shoulder"][-1][1] == shoulder[-1][1]
    assert corrected["bowl_outer"] == bowl_outer


def test_receiving_bowl_clearance_applies_only_to_the_upper_lip_and_cavity():
    bowl_inner = ((0.20, 0.09), (0.35, 0.16), (0.385, 0.172), (0.415, 0.180), (0.445, 0.183))
    bowl_outer = ((0.20, 0.102), (0.35, 0.190), (0.385, 0.196), (0.415, 0.204), (0.445, 0.208))
    globe = ((0.379, 0.187), (0.408, 0.193), (0.438, 0.201), (0.467, 0.188))

    corrected_outer = finisher.source_supported_bowl_outer_profile(bowl_outer, globe)
    corrected_inner = finisher.source_supported_bowl_inner_profile(bowl_inner, corrected_outer, globe)

    assert corrected_outer[:3] == bowl_outer[:3]
    assert corrected_outer[-1][1] >= bowl_outer[-1][1]
    top_globe_radius = builder._interpolate_radius(globe, corrected_outer[-1][0])
    assert corrected_outer[-1][1] >= (
        top_globe_radius * finisher.BOWL_RIM_TO_GLOBE_RADIUS_RATIO - 1e-12
    )
    assert corrected_inner[0] == bowl_inner[0]
    for z, radius in corrected_inner:
        outer_radius = builder._interpolate_radius(corrected_outer, z)
        assert radius <= outer_radius - finisher.BOWL_MIN_WALL_THICKNESS + 1e-12
        if z >= globe[0][0]:
            globe_radius = builder._interpolate_radius(globe, z)
            assert radius > globe_radius
    assert corrected_inner[-1][1] > bowl_inner[-1][1]


def test_lathe_mesh_data_is_closed_and_deterministic():
    profile = ((0.0, 0.0), (0.0, 0.2), (0.4, 0.2), (0.4, 0.0))

    vertices, faces = builder.lathe_mesh_data(profile, segments=8, closed=True)
    vertices_again, faces_again = builder.lathe_mesh_data(profile, segments=8, closed=True)

    assert (vertices, faces) == (vertices_again, faces_again)
    assert len(vertices) == 18
    assert len(faces) == 24
    assert all(len(face) in {3, 4} for face in faces)


def test_stage_checkpoints_remain_inside_v2_and_stop_before_export(tmp_path: Path):
    checkpoints = builder.stage_checkpoint_paths(tmp_path)

    assert checkpoints["base"] == tmp_path / "work" / "base_geometry.blend"
    assert checkpoints["final-validate"] == (
        tmp_path / "final" / "Thai_Libation_Vessel_FINAL.blend"
    )
    assert "export" not in checkpoints
    assert builder.BLENDER_STAGES[-1] == "final-validate"


def test_surface_parts_are_source_named_and_exclude_v1_shortcuts():
    assert set(builder.BASE_SURFACE_OBJECTS) == {
        "SM_Pedestal",
        "SM_Bowl",
        "SM_VesselBody",
        "SM_VesselNeck",
        "SM_Lid",
        "SM_Finial",
    }
    forbidden = ("Chain", "Diamond", "Floral", "Lotus", "Cavity_Shadow")
    assert not any(token in name for name in builder.BASE_SURFACE_OBJECTS for token in forbidden)


def test_ornament_manifest_requires_source_crops_and_explicit_chain_omission():
    families = []
    for family_id in builder.REQUIRED_ORNAMENT_FAMILIES:
        families.append(
            {
                "family_id": family_id,
                "host_component": "vessel_globe",
                "representation": "highpoly_bake",
                "primary_view_indices": [3, 19],
                "repeat_mode": "radial_repetition",
                "repeat_count": 6,
                "confidence": "high",
                "support_count": 2,
                "anchor_crop": {
                    "crop_path": f"evidence/{family_id}.png",
                    "crop_sha256": "a" * 64,
                    "source_sha256": "b" * 64,
                },
            }
        )
    payload = {
        "accepted": True,
        "families": families,
        "chain": {"supported": False, "decision": "omit_from_v2"},
    }

    validated = builder.validate_ornament_manifest(payload)

    assert set(validated) == set(builder.REQUIRED_ORNAMENT_FAMILIES)
    with pytest.raises(ValueError, match="chain"):
        builder.validate_ornament_manifest(
            {**payload, "chain": {"supported": True, "decision": "model"}}
        )
    families[0] = {**families[0], "support_count": 0}
    with pytest.raises(ValueError, match="source support"):
        builder.validate_ornament_manifest({**payload, "families": families})


def test_surface_relief_points_follow_profile_without_changing_macro_radius():
    profile = ((0.2, 0.10), (0.4, 0.20))
    path = ((-0.5, 0.0), (0.0, 0.5), (0.5, 1.0))

    points = builder.surface_relief_points(
        profile,
        path,
        center_angle=0.0,
        angular_width=0.4,
        z_min=0.2,
        z_max=0.4,
        relief_offset=0.002,
    )

    assert len(points) == 3
    assert points[0][2] == pytest.approx(0.2)
    assert points[1][2] == pytest.approx(0.3)
    assert points[2][2] == pytest.approx(0.4)
    for point, expected_radius in zip(points, (0.102, 0.152, 0.202)):
        assert (point[0] ** 2 + point[1] ** 2) ** 0.5 == pytest.approx(
            expected_radius
        )


def test_ornament_blueprint_keeps_source_provenance_and_inference_visible():
    families = {}
    for family_id in builder.REQUIRED_ORNAMENT_FAMILIES:
        families[family_id] = {
            "family_id": family_id,
            "host_component": "vessel_globe",
            "representation": "highpoly_bake",
            "primary_view_indices": [148, 165, 267],
            "repeat_mode": "radial_repetition",
            "repeat_count": 6,
            "hidden_repetition_inferred": True,
            "anchor_crop": {
                "crop_path": f"evidence/{family_id}.png",
                "crop_sha256": "a" * 64,
                "source_sha256": "b" * 64,
            },
        }

    blueprint = builder.build_ornament_blueprint(families)

    assert set(blueprint) == set(builder.REQUIRED_ORNAMENT_FAMILIES)
    hero = blueprint["ORB_GLOBE_HERO_MOTIF"]
    assert hero["host_profile"] == "globe"
    assert hero["repeat_count"] == 6
    assert hero["placement_class"] == "symmetry_or_repetition_inferred"
    assert hero["source_views"] == [148, 165, 267]
    assert hero["crop_sha256"] == "a" * 64
    assert blueprint["ORB_GLOBE_CROSSHATCH"]["geometry_kind"] == "field_source"
    assert blueprint["ORB_SHOULDER_RINGS"]["geometry_kind"] == "explicit_rings"


def test_ornament_blueprint_refuses_unbounded_radial_count():
    family = {
        "family_id": "ORB_GLOBE_HERO_MOTIF",
        "host_component": "vessel_globe",
        "representation": "highpoly_bake",
        "primary_view_indices": [267, 268],
        "repeat_mode": "radial_repetition",
        "repeat_count": 2,
        "hidden_repetition_inferred": True,
        "anchor_crop": {
            "crop_path": "evidence/hero.png",
            "crop_sha256": "a" * 64,
            "source_sha256": "b" * 64,
        },
    }
    families = {
        family_id: {
            **family,
            "family_id": family_id,
            "repeat_mode": "surface_field",
            "repeat_count": None,
        }
        for family_id in builder.REQUIRED_ORNAMENT_FAMILIES
    }
    families["ORB_GLOBE_HERO_MOTIF"] = family

    with pytest.raises(ValueError, match="repeat count"):
        builder.build_ornament_blueprint(families)


def test_review_wireframe_uses_renderable_geometry_not_viewport_show_wire():
    source = inspect.getsource(builder._render_review_views)

    assert "_render_true_wireframe" in source
    assert "show_wire" not in source
    helper = inspect.getsource(builder._render_true_wireframe)
    assert '"WIREFRAME"' in helper
    assert "modifiers.remove" in helper
