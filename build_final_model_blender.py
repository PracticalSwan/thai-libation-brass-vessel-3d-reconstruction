"""Build and validate the separate CV-constrained Final V2 Blender asset.

The module deliberately keeps its data-contract helpers importable in ordinary
Python.  Blender-only imports occur inside the stage functions so pytest can
exercise profile and topology generation without requiring ``bpy``.

Run with Blender 5.2:

    blender --background --factory-startup --python-exit-code 2 \
        --python build_final_model_blender.py -- <stage> <v2_root>

The default integrated orchestrator stops at ``final-validate``.  This script
contains no GLB export path; export remains a separately user-approved phase.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence


COLLECTIONS = (
    "COL_FINAL_V2",
    "COL_REFERENCE",
    "COL_BLOCKOUT",
    "COL_HIGH",
    "COL_LOW",
    "COL_ORNAMENT_HIGH",
    "COL_DIAGNOSTICS",
    "COL_CAMERAS",
    "COL_LIGHTS_NEUTRAL",
    "COL_LIGHTS_BEAUTY",
    "COL_EXPORT",
)

REQUIRED_PROFILES = (
    "pedestal",
    "bowl_outer",
    "bowl_inner",
    "globe",
    "shoulder",
    "neck_outer",
    "neck_inner",
    "lid",
    "finial",
)

BASE_SURFACE_OBJECTS = (
    "SM_Pedestal",
    "SM_Bowl",
    "SM_VesselBody",
    "SM_VesselNeck",
    "SM_Lid",
    "SM_Finial",
)

REQUIRED_ORNAMENT_FAMILIES = (
    "ORB_GLOBE_CROSSHATCH",
    "ORB_GLOBE_HERO_MOTIF",
    "ORB_GLOBE_SCROLL_BAND",
    "ORB_GLOBE_LOWER_BAND",
    "ORB_SHOULDER_RINGS",
    "ORB_NECK_FIELD",
    "ORB_NECK_LOTUS",
    "ORB_NECK_UPPER_BAND",
    "ORB_LID_TIERS",
    "ORB_LID_DECOR_BAND",
    "ORB_BOWL_FLAME_BAND",
    "ORB_PEDESTAL_RINGS",
)

RENDER_ENGINE = "BLENDER_EEVEE"

BLENDER_STAGES = (
    "base",
    "geometry-validate",
    "ornament",
    "cleanup",
    "uv-bake",
    "lookdev",
    "final-validate",
)

# Normalized traces distilled from the photographed flame/lotus and scroll
# families.  These are not V1 geometry and are mapped onto accepted profiles at
# runtime.  Fine photographic relief remains a texture/bake source in later
# stages rather than becoming floating mesh noise.
FLAME_OUTLINE = (
    (0.00, 0.00),
    (-0.12, 0.10),
    (-0.24, 0.21),
    (-0.14, 0.34),
    (-0.31, 0.50),
    (-0.23, 0.67),
    (-0.08, 0.79),
    (0.00, 1.00),
    (0.08, 0.79),
    (0.23, 0.67),
    (0.31, 0.50),
    (0.14, 0.34),
    (0.24, 0.21),
    (0.12, 0.10),
    (0.00, 0.00),
)

INNER_FLAME = (
    (0.00, 0.15),
    (-0.09, 0.27),
    (-0.03, 0.39),
    (-0.13, 0.50),
    (-0.04, 0.62),
    (0.00, 0.76),
    (0.04, 0.62),
    (0.13, 0.50),
    (0.03, 0.39),
    (0.09, 0.27),
    (0.00, 0.15),
)

SCROLL_TRACE = (
    (-0.46, 0.18),
    (-0.34, 0.08),
    (-0.18, 0.09),
    (-0.05, 0.20),
    (-0.08, 0.34),
    (-0.21, 0.39),
    (-0.29, 0.31),
    (-0.24, 0.22),
    (-0.14, 0.23),
)


ORNAMENT_GEOMETRY_SPECS: dict[str, dict[str, Any]] = {
    "ORB_GLOBE_CROSSHATCH": {
        "geometry_kind": "field_source",
        "host_profile": "globe",
        "z_min": 0.425,
        "z_max": 0.585,
        "relief_offset": 0.00055,
    },
    "ORB_GLOBE_HERO_MOTIF": {
        "geometry_kind": "hero_flame",
        "host_profile": "globe",
        "z_min": 0.420,
        "z_max": 0.585,
        "angular_width": 0.78,
        "relief_offset": 0.00145,
    },
    "ORB_GLOBE_SCROLL_BAND": {
        "geometry_kind": "scroll_band",
        "host_profile": "globe",
        "z_min": 0.405,
        "z_max": 0.465,
        "relief_offset": 0.00115,
    },
    "ORB_GLOBE_LOWER_BAND": {
        "geometry_kind": "existing_ring",
        "host_profile": "globe",
    },
    "ORB_SHOULDER_RINGS": {
        "geometry_kind": "explicit_rings",
        "host_profile": "shoulder",
    },
    "ORB_NECK_FIELD": {
        "geometry_kind": "field_source",
        "host_profile": "neck_outer",
        "z_min": 0.658,
        "z_max": 0.770,
        "relief_offset": 0.00038,
    },
    "ORB_NECK_LOTUS": {
        "geometry_kind": "lotus_flame",
        "host_profile": "neck_outer",
        "z_min": 0.660,
        "z_max": 0.772,
        "angular_width": 0.72,
        "relief_offset": 0.00105,
    },
    "ORB_NECK_UPPER_BAND": {
        "geometry_kind": "scroll_band",
        "host_profile": "neck_outer",
        "z_min": 0.770,
        "z_max": 0.797,
        "relief_offset": 0.00085,
    },
    "ORB_LID_TIERS": {
        "geometry_kind": "existing_ring",
        "host_profile": "lid",
    },
    "ORB_LID_DECOR_BAND": {
        "geometry_kind": "scroll_band",
        "host_profile": "lid",
        "z_min": 0.812,
        "z_max": 0.838,
        "relief_offset": 0.0008,
    },
    "ORB_BOWL_FLAME_BAND": {
        "geometry_kind": "lotus_flame",
        "host_profile": "bowl_outer",
        "z_min": 0.268,
        "z_max": 0.392,
        "angular_width": 0.45,
        "relief_offset": 0.00135,
    },
    "ORB_PEDESTAL_RINGS": {
        "geometry_kind": "explicit_rings",
        "host_profile": "pedestal",
    },
}


def stage_checkpoint_paths(v2_root: Path) -> dict[str, Path]:
    root = Path(v2_root)
    return {
        "base": root / "work" / "base_geometry.blend",
        "geometry-validate": root / "work" / "30_BASE_GEOMETRY_ACCEPTED.blend",
        "ornament": root / "work" / "40_ORNAMENT_ACCEPTED.blend",
        "cleanup": root / "work" / "50_CLEAN_TOPOLOGY_ACCEPTED.blend",
        "uv-bake": root / "work" / "60_UV_BAKE_ACCEPTED.blend",
        "lookdev": root / "work" / "70_LOOKDEV_ACCEPTED.blend",
        "final-validate": root / "final" / "Thai_Libation_Vessel_FINAL.blend",
    }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(target)
    return target


def validate_profile_payload(
    payload: Mapping[str, Any],
) -> dict[str, tuple[tuple[float, float], ...]]:
    """Validate and normalize Plan-1 ``(z, radius)`` component profiles."""

    contract = payload.get("coordinate_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("profile coordinate contract is missing")
    if contract.get("axis") != "+Z":
        raise ValueError("profile axis must be +Z")
    if float(contract.get("normalized_set_height", -1.0)) != 1.0:
        raise ValueError("profile set height must be normalized to 1.0")
    if contract.get("scale_status") != "relative_no_physical_measurement":
        raise ValueError("profile scale status is not the accepted relative contract")

    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, Mapping):
        raise ValueError("profile mapping is missing")

    normalized: dict[str, tuple[tuple[float, float], ...]] = {}
    for name in REQUIRED_PROFILES:
        sections = raw_profiles.get(name)
        if not isinstance(sections, Sequence) or len(sections) < 2:
            raise ValueError(f"profile {name} needs at least two sections")
        points: list[tuple[float, float]] = []
        previous_z: float | None = None
        for raw in sections:
            if not isinstance(raw, Sequence) or len(raw) != 2:
                raise ValueError(f"profile {name} contains an invalid section")
            z = float(raw[0])
            radius = float(raw[1])
            if not math.isfinite(z) or not math.isfinite(radius):
                raise ValueError(f"profile {name} contains a non-finite value")
            if radius < 0.0:
                raise ValueError(f"profile {name} contains a negative radius")
            if not 0.0 <= z <= 1.0:
                raise ValueError(f"profile {name} leaves normalized height range")
            if previous_z is not None and z <= previous_z:
                raise ValueError(f"profile {name} z values must be strictly increasing")
            points.append((z, radius))
            previous_z = z
        normalized[name] = tuple(points)
    return normalized


def closed_shell_profile(
    outer: Sequence[tuple[float, float]],
    inner: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Return one closed annular ``(z, r)`` polygon from matching wall profiles."""

    if len(outer) < 2 or len(inner) < 2:
        raise ValueError("shell profiles require at least two sections")
    if not math.isclose(outer[0][0], inner[0][0], abs_tol=1e-8) or not math.isclose(
        outer[-1][0], inner[-1][0], abs_tol=1e-8
    ):
        raise ValueError("inner and outer shell profiles must share end heights")
    for (outer_z, outer_radius), (inner_z, inner_radius) in zip(outer, inner):
        if not math.isclose(outer_z, inner_z, abs_tol=1e-8):
            raise ValueError("inner and outer shell profile levels disagree")
        if inner_radius >= outer_radius:
            raise ValueError("shell inner radius must remain inside outer radius")
    return tuple(outer) + tuple(reversed(tuple(inner)))


def lathe_mesh_data(
    profile: Sequence[tuple[float, float]],
    *,
    segments: int = 160,
    closed: bool = True,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[tuple[int, ...], ...]]:
    """Revolve one ``(z, radius)`` profile around global Z deterministically."""

    if segments < 3:
        raise ValueError("lathe segments must be at least three")
    if len(profile) < 2:
        raise ValueError("lathe profile needs at least two sections")

    vertices: list[tuple[float, float, float]] = []
    rings: list[tuple[int, ...]] = []
    for z, radius in profile:
        z_value = float(z)
        radius_value = float(radius)
        if radius_value < 0.0:
            raise ValueError("lathe profile contains a negative radius")
        if abs(radius_value) <= 1e-12:
            rings.append((len(vertices),))
            vertices.append((0.0, 0.0, z_value))
            continue
        ring: list[int] = []
        for segment in range(segments):
            angle = math.tau * segment / segments
            ring.append(len(vertices))
            vertices.append(
                (
                    radius_value * math.cos(angle),
                    radius_value * math.sin(angle),
                    z_value,
                )
            )
        rings.append(tuple(ring))

    pairs = list(zip(rings, rings[1:]))
    if closed:
        pairs.append((rings[-1], rings[0]))
    faces: list[tuple[int, ...]] = []
    for first, second in pairs:
        first_axis = len(first) == 1
        second_axis = len(second) == 1
        if first_axis and second_axis:
            continue
        for segment in range(segments):
            nxt = (segment + 1) % segments
            if first_axis:
                faces.append((first[0], second[segment], second[nxt]))
            elif second_axis:
                faces.append((first[segment], second[0], first[nxt]))
            else:
                faces.append((first[segment], second[segment], second[nxt], first[nxt]))
    return tuple(vertices), tuple(faces)


def _interpolate_radius(profile: Sequence[tuple[float, float]], z: float) -> float:
    if z <= profile[0][0]:
        return float(profile[0][1])
    if z >= profile[-1][0]:
        return float(profile[-1][1])
    for first, second in zip(profile, profile[1:]):
        if first[0] <= z <= second[0]:
            t = (z - first[0]) / (second[0] - first[0])
            return float(first[1] * (1.0 - t) + second[1] * t)
    raise AssertionError("profile interpolation interval not found")


def outer_envelope_profile(
    *profiles: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Merge overlapping rotational components by their maximum radial envelope."""

    levels = sorted({float(z) for profile in profiles for z, _ in profile})
    output: list[tuple[float, float]] = []
    for z in levels:
        candidates = [
            _interpolate_radius(profile, z)
            for profile in profiles
            if profile[0][0] <= z <= profile[-1][0]
        ]
        if candidates:
            output.append((z, max(candidates)))
    return tuple(output)


def profile_measurements(
    profiles: Mapping[str, Sequence[tuple[float, float]]],
) -> dict[str, float]:
    """Record normalized control dimensions without claiming a physical scale."""

    globe_diameter = 2.0 * max(float(radius) for _, radius in profiles["globe"])
    neck_length = float(profiles["neck_outer"][-1][0] - profiles["neck_outer"][0][0])
    return {
        "normalized_set_height": 1.0,
        "maximum_profile_radius": max(
            float(radius)
            for profile in profiles.values()
            for _, radius in profile
        ),
        "globe_max_diameter": globe_diameter,
        "neck_length": neck_length,
        "neck_length_to_globe_diameter": neck_length / globe_diameter,
        # The Blender surfaces are generated directly from these values.  Any
        # later sculpt/cleanup stage must recompute and revalidate this field.
        "profile_import_max_abs_delta": 0.0,
    }


def validate_ornament_manifest(
    payload: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Validate the source-traceable Plan-3 inventory before Blender use."""

    if payload.get("accepted") is not True:
        raise ValueError("ornament manifest is not accepted")
    chain = payload.get("chain")
    if not isinstance(chain, Mapping) or chain.get("supported") is not False:
        raise ValueError("chain must remain unsupported without two-view evidence")
    if chain.get("decision") != "omit_from_v2":
        raise ValueError("chain decision must be omit_from_v2")
    raw_families = payload.get("families")
    if not isinstance(raw_families, Sequence):
        raise ValueError("ornament families are missing")
    families: dict[str, dict[str, Any]] = {}
    for raw in raw_families:
        if not isinstance(raw, Mapping):
            raise ValueError("invalid ornament family entry")
        record = dict(raw)
        family_id = str(record.get("family_id", ""))
        if family_id in families:
            raise ValueError(f"duplicate ornament family: {family_id}")
        source_views = record.get("primary_view_indices")
        support_count = int(record.get("support_count", 0))
        if (
            not isinstance(source_views, Sequence)
            or isinstance(source_views, (str, bytes))
            or support_count < 1
            or len(source_views) < support_count
        ):
            raise ValueError(f"ornament family lacks source support: {family_id}")
        crop = record.get("anchor_crop")
        if not isinstance(crop, Mapping):
            raise ValueError(f"ornament family lacks anchor crop: {family_id}")
        for field in ("crop_sha256", "source_sha256"):
            value = crop.get(field)
            if not (
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"ornament family has invalid {field}: {family_id}")
        if not str(crop.get("crop_path", "")).strip():
            raise ValueError(f"ornament family lacks crop path: {family_id}")
        families[family_id] = record
    missing = sorted(set(REQUIRED_ORNAMENT_FAMILIES) - set(families))
    if missing:
        raise ValueError("missing required ornament families: " + ", ".join(missing))
    return families


def build_ornament_blueprint(
    families: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Bind each photographed family to a conservative Blender construction.

    This is intentionally a data-only function.  It keeps observed source
    support separate from repetition/symmetry inference and lets tests inspect
    the complete construction contract without importing Blender.
    """

    missing = sorted(set(REQUIRED_ORNAMENT_FAMILIES) - set(families))
    if missing:
        raise ValueError("missing required ornament families: " + ", ".join(missing))
    blueprint: dict[str, dict[str, Any]] = {}
    for family_id in REQUIRED_ORNAMENT_FAMILIES:
        family = families[family_id]
        spec = ORNAMENT_GEOMETRY_SPECS[family_id]
        repeat_mode = str(family.get("repeat_mode", ""))
        repeat_count_value = family.get("repeat_count")
        placement = family.get("placement")
        if isinstance(placement, Mapping):
            repeat_count_value = placement.get(
                "accepted_repeat_count", placement.get("repeat_count", repeat_count_value)
            )
        repeat_count = (
            None if repeat_count_value is None else int(repeat_count_value)
        )
        if repeat_mode == "radial_repetition" and (
            repeat_count is None or not 3 <= repeat_count <= 24
        ):
            raise ValueError(f"invalid evidence-bounded repeat count: {family_id}")
        phase = 0.0
        if isinstance(placement, Mapping):
            phase = float(
                placement.get(
                    "phase_angle_radians",
                    math.radians(float(placement.get("phase_angle_degrees", 0.0))),
                )
            )
        crop = family.get("anchor_crop")
        if not isinstance(crop, Mapping):
            raise ValueError(f"missing anchor crop: {family_id}")
        inferred = bool(family.get("hidden_repetition_inferred", False))
        blueprint[family_id] = {
            **spec,
            "family_id": family_id,
            "representation": str(family.get("representation", "")),
            "source_views": [
                int(value) for value in family.get("primary_view_indices", [])
            ],
            "crop_path": str(crop.get("crop_path", "")),
            "crop_sha256": str(crop.get("crop_sha256", "")),
            "source_sha256": str(crop.get("source_sha256", "")),
            "repeat_mode": repeat_mode,
            "repeat_count": repeat_count,
            "phase_angle_radians": phase,
            "placement_class": (
                "symmetry_or_repetition_inferred" if inferred else "directly_observed"
            ),
            "directly_observed": bool(family.get("directly_observed", False)),
        }
    return blueprint


def surface_relief_points(
    profile: Sequence[tuple[float, float]],
    normalized_path: Sequence[tuple[float, float]],
    *,
    center_angle: float,
    angular_width: float,
    z_min: float,
    z_max: float,
    relief_offset: float,
) -> tuple[tuple[float, float, float], ...]:
    """Map one normalized photographed motif trace onto a rotational host."""

    if z_max <= z_min or angular_width <= 0.0 or relief_offset < 0.0:
        raise ValueError("invalid relief mapping bounds")
    points: list[tuple[float, float, float]] = []
    for horizontal, vertical in normalized_path:
        if not -0.5 <= float(horizontal) <= 0.5 or not 0.0 <= float(vertical) <= 1.0:
            raise ValueError("relief path leaves normalized motif coordinates")
        z = z_min + float(vertical) * (z_max - z_min)
        angle = center_angle + float(horizontal) * angular_width
        radius = _interpolate_radius(profile, z) + relief_offset
        points.append(
            (
                radius * math.cos(angle),
                radius * math.sin(angle),
                z,
            )
        )
    return tuple(points)


def _solid_profile(profile: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    return (
        (float(profile[0][0]), 0.0),
        *tuple((float(z), float(radius)) for z, radius in profile),
        (float(profile[-1][0]), 0.0),
    )


def _bpy_modules():
    try:
        import bmesh  # type: ignore
        import bpy  # type: ignore
        from mathutils import Vector  # type: ignore
    except ImportError as error:  # pragma: no cover - executed only outside Blender by mistake
        raise RuntimeError("this operation must run inside Blender") from error
    return bpy, bmesh, Vector


def _unlink_and_remove_all() -> None:
    bpy, _, _ = _bpy_modules()
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in tuple(bpy.data.collections):
        bpy.data.collections.remove(collection)


def setup_v2_scene(v2_root: Path, profile_sha256: str, candidate_sha256: str):
    bpy, _, _ = _bpy_modules()
    _unlink_and_remove_all()
    scene = bpy.context.scene
    scene.name = "Thai_Libation_Vessel_Final_V2"
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.render.engine = RENDER_ENGINE
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.resolution_x = 720
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.look = "AgX - Medium High Contrast"

    root_collection = bpy.data.collections.new("COL_FINAL_V2")
    scene.collection.children.link(root_collection)
    collections = {"COL_FINAL_V2": root_collection}
    for name in COLLECTIONS[1:]:
        collection = bpy.data.collections.new(name)
        root_collection.children.link(collection)
        collections[name] = collection

    root_collection["scale_status"] = "relative_no_physical_measurement"
    root_collection["normalized_set_height"] = 1.0
    root_collection["plan1_candidate_sha256"] = candidate_sha256
    root_collection["profiles_sha256"] = profile_sha256
    root_collection["v2_root"] = str(Path(v2_root).resolve())

    root = bpy.data.objects.new("ROOT_ThaiLibationV2", None)
    root.empty_display_type = "PLAIN_AXES"
    root.empty_display_size = 0.12
    root_collection.objects.link(root)
    root["method"] = "cv_constrained_reference_reconstruction"
    root["source_profiles"] = (
        "reconstruction/reference_assisted_v2/reports/final_profiles.json"
    )
    root["manual_visual_finish"] = "true_after_cv_geometry_gate"
    root["plan1_candidate_sha256"] = candidate_sha256
    return scene, collections, root


def _new_material(name: str, color: tuple[float, float, float, float], *, metallic: float, roughness: float):
    bpy, _, _ = _bpy_modules()
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Metallic"].default_value = metallic
    principled.inputs["Roughness"].default_value = roughness
    return material


def _create_profile_curve(name: str, profile: Sequence[tuple[float, float]], collection, root):
    bpy, _, _ = _bpy_modules()
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 1
    spline = curve.splines.new("POLY")
    spline.points.add(len(profile) - 1)
    for point, (z, radius) in zip(spline.points, profile):
        point.co = (float(radius), 0.0, float(z), 1.0)
    obj = bpy.data.objects.new(name, curve)
    collection.objects.link(obj)
    obj.parent = root
    obj.hide_render = True
    obj.hide_set(True)
    obj["profile_order"] = "z_radius"
    obj["sections_json"] = json.dumps(profile)
    return obj


def _create_mesh_object(
    name: str,
    profile: Sequence[tuple[float, float]],
    collection,
    root,
    material,
    *,
    segments: int = 160,
    closed: bool = True,
):
    bpy, bmesh, _ = _bpy_modules()
    vertices, faces = lathe_mesh_data(profile, segments=segments, closed=closed)
    mesh = bpy.data.meshes.new(name + "_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.parent = root
    obj.data.materials.append(material)
    obj["source"] = "accepted_plan1_profile"
    obj["profile_sha256"] = _sha256_bytes(
        json.dumps(profile, separators=(",", ":")).encode("utf-8")
    )
    bevel = obj.modifiers.new("MOD_MicroBevel", "BEVEL")
    bevel.width = 0.0012
    bevel.segments = 2
    return obj


def _move_object_to_collection(obj, collection) -> None:
    for current in tuple(obj.users_collection):
        current.objects.unlink(obj)
    collection.objects.link(obj)


def _add_torus(
    name: str,
    *,
    outer_radius: float,
    z: float,
    minor_radius: float,
    collection,
    root,
    material,
):
    bpy, _, _ = _bpy_modules()
    major_radius = max(outer_radius - minor_radius, minor_radius * 1.5)
    bpy.ops.mesh.primitive_torus_add(
        major_radius=major_radius,
        minor_radius=minor_radius,
        major_segments=128,
        minor_segments=12,
        location=(0.0, 0.0, z),
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.name = name + "_Mesh"
    _move_object_to_collection(obj, collection)
    obj.parent = root
    obj.data.materials.append(material)
    obj["source"] = "source_supported_construction_ring_within_plan1_envelope"
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def _create_curve_source(
    name: str,
    paths: Sequence[Sequence[tuple[float, float, float]]],
    *,
    collection,
    root,
    material,
    bevel_depth: float,
    cyclic: bool = False,
    metadata: Mapping[str, Any] | None = None,
):
    """Create one non-destructive high-source curve containing many splines."""

    bpy, _, _ = _bpy_modules()
    curve = bpy.data.curves.new(name + "_Curve", "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.bevel_depth = float(bevel_depth)
    curve.bevel_resolution = 2
    curve.resolution_u = 2
    for path in paths:
        if len(path) < 2:
            continue
        spline = curve.splines.new("POLY")
        spline.points.add(len(path) - 1)
        for point, coordinate in zip(spline.points, path):
            point.co = (*map(float, coordinate), 1.0)
        spline.use_cyclic_u = cyclic
    obj = bpy.data.objects.new(name, curve)
    collection.objects.link(obj)
    obj.parent = root
    obj.data.materials.append(material)
    obj["source"] = "photographed_ornament_family"
    obj["non_destructive_bake_source"] = True
    for key, value in (metadata or {}).items():
        if isinstance(value, (str, int, float, bool)):
            obj[key] = value
        else:
            obj[key] = json.dumps(value, sort_keys=True)
    return obj


def _radial_motif_paths(
    profile: Sequence[tuple[float, float]],
    *,
    repeat_count: int,
    phase: float,
    angular_width: float,
    z_min: float,
    z_max: float,
    relief_offset: float,
    include_scroll_wings: bool,
) -> list[tuple[tuple[float, float, float], ...]]:
    paths: list[tuple[tuple[float, float, float], ...]] = []
    for repeat_index in range(repeat_count):
        center = phase + math.tau * repeat_index / repeat_count
        paths.append(
            surface_relief_points(
                profile,
                FLAME_OUTLINE,
                center_angle=center,
                angular_width=angular_width,
                z_min=z_min,
                z_max=z_max,
                relief_offset=relief_offset,
            )
        )
        paths.append(
            surface_relief_points(
                profile,
                INNER_FLAME,
                center_angle=center,
                angular_width=angular_width,
                z_min=z_min,
                z_max=z_max,
                relief_offset=relief_offset * 1.04,
            )
        )
        if include_scroll_wings:
            paths.append(
                surface_relief_points(
                    profile,
                    SCROLL_TRACE,
                    center_angle=center,
                    angular_width=angular_width * 1.18,
                    z_min=z_min,
                    z_max=z_max,
                    relief_offset=relief_offset * 0.92,
                )
            )
            mirrored = tuple((-x, y) for x, y in SCROLL_TRACE)
            paths.append(
                surface_relief_points(
                    profile,
                    mirrored,
                    center_angle=center,
                    angular_width=angular_width * 1.18,
                    z_min=z_min,
                    z_max=z_max,
                    relief_offset=relief_offset * 0.92,
                )
            )
    return paths


def _band_paths(
    profile: Sequence[tuple[float, float]],
    *,
    z_min: float,
    z_max: float,
    relief_offset: float,
    phase: float,
    lobes: int,
) -> list[tuple[tuple[float, float, float], ...]]:
    paths: list[tuple[tuple[float, float, float], ...]] = []
    middle = 0.5 * (z_min + z_max)
    amplitude = 0.22 * (z_max - z_min)
    for lane, lane_offset in enumerate((-0.22, 0.0, 0.22)):
        path: list[tuple[float, float, float]] = []
        for sample in range(256):
            angle = math.tau * sample / 256.0
            z = (
                middle
                + lane_offset * (z_max - z_min)
                + amplitude * math.sin(lobes * angle + phase + lane * math.pi)
            )
            radius = _interpolate_radius(profile, z) + relief_offset
            path.append((radius * math.cos(angle), radius * math.sin(angle), z))
        paths.append(tuple(path))
    return paths


def _field_paths(
    profile: Sequence[tuple[float, float]],
    *,
    z_min: float,
    z_max: float,
    relief_offset: float,
    line_count: int,
    turns: float,
) -> list[tuple[tuple[float, float, float], ...]]:
    """Generate a UV-independent cross-hatch bake source on the host profile."""

    paths: list[tuple[tuple[float, float, float], ...]] = []
    samples = 52
    for direction in (-1.0, 1.0):
        for line_index in range(line_count):
            base = math.tau * line_index / line_count
            path: list[tuple[float, float, float]] = []
            for sample in range(samples):
                t = sample / (samples - 1)
                z = z_min + t * (z_max - z_min)
                angle = base + direction * math.tau * turns * t
                radius = _interpolate_radius(profile, z) + relief_offset
                path.append((radius * math.cos(angle), radius * math.sin(angle), z))
            paths.append(tuple(path))
    return paths


def _source_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ornament_family": record["family_id"],
        "host_profile": record["host_profile"],
        "source_views": record["source_views"],
        "crop_path": record["crop_path"],
        "crop_sha256": record["crop_sha256"],
        "source_sha256": record["source_sha256"],
        "placement_class": record["placement_class"],
        "repeat_count": -1 if record["repeat_count"] is None else record["repeat_count"],
        "phase_angle_radians": record["phase_angle_radians"],
    }


def _build_ornament_sources(
    profiles: Mapping[str, Sequence[tuple[float, float]]],
    blueprint: Mapping[str, Mapping[str, Any]],
    collections: Mapping[str, Any],
    root,
) -> list[Any]:
    ornament_collection = collections["COL_ORNAMENT_HIGH"]
    material = _new_material(
        "MAT_V2_OrnamentClay", (0.68, 0.49, 0.22, 1.0), metallic=0.0, roughness=0.55
    )
    objects: list[Any] = []
    object_names = {
        "ORB_GLOBE_HERO_MOTIF": "HP_ORN_GlobeHeroMotif",
        "ORB_BOWL_FLAME_BAND": "HP_ORN_BowlFlameBand",
        "ORB_NECK_LOTUS": "HP_ORN_NeckLotus",
        "ORB_GLOBE_SCROLL_BAND": "HP_ORN_GlobeScrollBand",
        "ORB_NECK_UPPER_BAND": "HP_ORN_NeckUpperBand",
        "ORB_LID_DECOR_BAND": "HP_ORN_LidDecorBand",
        "ORB_GLOBE_CROSSHATCH": "HP_ORN_GlobeCrosshatchSource",
        "ORB_NECK_FIELD": "HP_ORN_NeckFieldSource",
    }
    for family_id, object_name in object_names.items():
        record = blueprint[family_id]
        profile = profiles[str(record["host_profile"])]
        kind = str(record["geometry_kind"])
        if kind in {"hero_flame", "lotus_flame"}:
            paths = _radial_motif_paths(
                profile,
                repeat_count=int(record["repeat_count"]),
                phase=float(record["phase_angle_radians"]),
                angular_width=float(record["angular_width"]),
                z_min=float(record["z_min"]),
                z_max=float(record["z_max"]),
                relief_offset=float(record["relief_offset"]),
                include_scroll_wings=kind == "hero_flame",
            )
            bevel_depth = 0.00072 if kind == "hero_flame" else 0.00058
            cyclic = False
        elif kind == "scroll_band":
            paths = _band_paths(
                profile,
                z_min=float(record["z_min"]),
                z_max=float(record["z_max"]),
                relief_offset=float(record["relief_offset"]),
                phase=float(record["phase_angle_radians"]),
                lobes=16 if family_id == "ORB_GLOBE_SCROLL_BAND" else 12,
            )
            bevel_depth = 0.00042
            cyclic = True
        elif kind == "field_source":
            paths = _field_paths(
                profile,
                z_min=float(record["z_min"]),
                z_max=float(record["z_max"]),
                relief_offset=float(record["relief_offset"]),
                line_count=48 if family_id == "ORB_GLOBE_CROSSHATCH" else 28,
                turns=1.55 if family_id == "ORB_GLOBE_CROSSHATCH" else 0.92,
            )
            bevel_depth = 0.00014 if family_id == "ORB_GLOBE_CROSSHATCH" else 0.00012
            cyclic = False
        else:  # pragma: no cover - specifications above exhaust this subset
            raise ValueError(f"unsupported ornament geometry kind: {kind}")
        objects.append(
            _create_curve_source(
                object_name,
                paths,
                collection=ornament_collection,
                root=root,
                material=material,
                bevel_depth=bevel_depth,
                cyclic=cyclic,
                metadata=_source_metadata(record),
            )
        )

    ring_specs = (
        ("HP_ORN_ShoulderRing_04", "ORB_SHOULDER_RINGS", "shoulder", 0.542, 0.0019),
        ("HP_ORN_PedestalRing_03", "ORB_PEDESTAL_RINGS", "pedestal", 0.058, 0.0021),
        ("HP_ORN_PedestalRing_04", "ORB_PEDESTAL_RINGS", "pedestal", 0.148, 0.0020),
    )
    for name, family_id, profile_name, z, minor in ring_specs:
        record = blueprint[family_id]
        outer_radius = _interpolate_radius(profiles[profile_name], z)
        obj = _add_torus(
            name,
            outer_radius=outer_radius,
            z=z,
            minor_radius=minor,
            collection=ornament_collection,
            root=root,
            material=material,
        )
        obj["source"] = "photographed_explicit_ring_family"
        for key, value in _source_metadata(record).items():
            obj[key] = value if isinstance(value, (str, int, float, bool)) else json.dumps(value)
        objects.append(obj)
    return objects


def _look_at(obj, target) -> None:
    _, _, Vector = _bpy_modules()
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _create_reference_objects(
    profiles_payload: Mapping[str, Any],
    fit_payload: Mapping[str, Any],
    collections: Mapping[str, Any],
    root,
    project_root: Path,
) -> list[Any]:
    bpy, _, _ = _bpy_modules()
    cameras = []
    intervals = fit_payload.get("capture_sweep_intervals", {})
    for record in fit_payload.get("views", []):
        index = int(record["selected_index"])
        category = str(record.get("view_category", "unknown"))
        sweep_name = next(
            (
                name
                for name, limits in intervals.items()
                if int(limits[0]) <= index <= int(limits[1])
            ),
            "unknown",
        )
        limits = intervals.get(sweep_name, (index, index + 1))
        denominator = max(int(limits[1]) - int(limits[0]), 1)
        azimuth = math.tau * (index - int(limits[0])) / denominator
        elevation = {
            "normal_side": 0.48,
            "low_angle_pedestal": 0.28,
            "elevated_oblique": 0.82,
            "top_down_rim": 1.65,
        }.get(category, 0.55)
        radius = 2.4
        bpy.ops.object.camera_add(
            location=(radius * math.cos(azimuth), radius * math.sin(azimuth), elevation)
        )
        camera = bpy.context.object
        camera.name = f"CAM_REF_{index:03d}"
        camera.data.name = camera.name + "_Data"
        camera.data.lens = 58.0
        camera.data.sensor_width = 36.0
        _look_at(camera, (0.0, 0.0, 0.50))
        _move_object_to_collection(camera, collections["COL_CAMERAS"])
        camera.parent = root
        camera["selected_index"] = index
        camera["source_filename"] = str(record.get("filename", ""))
        camera["view_category"] = category
        camera["step13_registered"] = True
        camera["camera_model"] = "SIMPLE_RADIAL"
        camera["coordinate_path"] = "display_only_pinhole_approximation"
        camera["exact_metric_path"] = "raw_source_exact_simple_radial_regular_python"
        camera["sweep"] = sweep_name
        cameras.append(camera)

        reference = bpy.data.objects.new(f"REF_{index:03d}", None)
        reference.empty_display_type = "IMAGE"
        reference.empty_display_size = 0.5
        source_value = record.get("source_path")
        if source_value:
            source_path = project_root / str(source_value)
            if source_path.is_file():
                source_sha256 = sha256_file(source_path)
                try:
                    reference.data = bpy.data.images.load(str(source_path), check_existing=True)
                except RuntimeError:
                    pass
            else:
                source_sha256 = ""
        else:
            source_sha256 = ""
        collections["COL_REFERENCE"].objects.link(reference)
        reference.parent = root
        reference.hide_render = True
        reference.hide_set(True)
        reference["selected_index"] = index
        reference["source_filename"] = str(record.get("filename", ""))
        reference["view_category"] = category
        reference["step13_registered"] = True
        reference["source_path"] = str(source_value or "")
        reference["source_sha256"] = source_sha256
    return cameras


def _build_base_geometry(
    profiles: Mapping[str, Sequence[tuple[float, float]]],
    collections: Mapping[str, Any],
    root,
) -> list[Any]:
    clay = _new_material(
        "MAT_V2_NeutralClay", (0.54, 0.43, 0.27, 1.0), metallic=0.0, roughness=0.62
    )
    blockout = collections["COL_BLOCKOUT"]
    low = collections["COL_LOW"]
    for name in REQUIRED_PROFILES:
        _create_profile_curve(
            "CRV_" + "".join(part.title() for part in name.split("_")) + "Profile",
            profiles[name],
            blockout,
            root,
        )

    pedestal = _create_mesh_object(
        "SM_Pedestal", _solid_profile(profiles["pedestal"]), low, root, clay
    )
    bowl = _create_mesh_object(
        "SM_Bowl",
        closed_shell_profile(profiles["bowl_outer"], profiles["bowl_inner"]),
        low,
        root,
        clay,
    )
    body_envelope = outer_envelope_profile(profiles["globe"], profiles["shoulder"])
    body = _create_mesh_object(
        "SM_VesselBody", _solid_profile(body_envelope), low, root, clay
    )
    neck = _create_mesh_object(
        "SM_VesselNeck",
        closed_shell_profile(profiles["neck_outer"], profiles["neck_inner"]),
        low,
        root,
        clay,
    )
    lid = _create_mesh_object(
        "SM_Lid", _solid_profile(profiles["lid"]), low, root, clay
    )
    finial = _create_mesh_object(
        "SM_Finial", _solid_profile(profiles["finial"]), low, root, clay
    )

    rings = []
    ring_specs: list[tuple[str, str, float, float]] = [
        ("SM_Pedestal_FootRoll", "pedestal", 0.018, 0.0060),
        ("SM_Pedestal_UpperRoll", "pedestal", 0.184, 0.0045),
        ("SM_Bowl_LowerSeatRing", "bowl_outer", 0.214, 0.0035),
        ("SM_Bowl_RolledRim", "bowl_outer", 0.443, 0.0048),
        ("SM_Globe_LowerConstructionBand", "globe", 0.397, 0.0030),
        ("SM_Shoulder_Ring_01", "shoulder", 0.568, 0.0026),
        ("SM_Shoulder_Ring_02", "shoulder", 0.592, 0.0024),
        ("SM_Shoulder_Ring_03", "shoulder", 0.618, 0.0022),
        ("SM_Neck_LowerCollar", "neck_outer", 0.646, 0.0025),
        ("SM_Neck_UpperCollar", "neck_outer", 0.799, 0.0028),
    ]
    for tier, z in enumerate((0.827, 0.845, 0.863, 0.881, 0.899, 0.917, 0.935), 1):
        ring_specs.append((f"SM_Lid_TierRing_{tier:02d}", "lid", z, 0.0020))
    for name, profile_name, z, minor in ring_specs:
        outer_radius = _interpolate_radius(profiles[profile_name], z)
        rings.append(
            _add_torus(
                name,
                outer_radius=outer_radius,
                z=z,
                minor_radius=min(minor, outer_radius * 0.18),
                collection=low,
                root=root,
                material=clay,
            )
        )

    for object_value in (pedestal, bowl, body, neck, lid, finial):
        object_value["physical_part"] = True
        object_value["normalized_axis"] = "+Z"
    bowl["real_interior_cavity"] = True
    bowl["fake_black_cavity"] = False
    neck["real_opening"] = True
    lid["removable_part"] = True
    finial["attached_to"] = "SM_Lid"
    return [pedestal, bowl, body, neck, lid, finial, *rings]


def _scene_bounds(objects: Iterable[Any]):
    _, _, Vector = _bpy_modules()
    minimum = Vector((float("inf"),) * 3)
    maximum = Vector((float("-inf"),) * 3)
    for obj in objects:
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            minimum.x = min(minimum.x, world.x)
            minimum.y = min(minimum.y, world.y)
            minimum.z = min(minimum.z, world.z)
            maximum.x = max(maximum.x, world.x)
            maximum.y = max(maximum.y, world.y)
            maximum.z = max(maximum.z, world.z)
    return minimum, maximum


def _setup_neutral_review(collections: Mapping[str, Any], model_objects: Sequence[Any]):
    bpy, _, Vector = _bpy_modules()
    camera = bpy.data.objects.get("CAM_Hero_Review")
    if camera is None:
        bpy.ops.object.camera_add(location=(1.48, -2.55, 0.78))
        camera = bpy.context.object
        camera.name = "CAM_Hero_Review"
        camera.data.name = "CAM_Hero_Review_Data"
        _move_object_to_collection(camera, collections["COL_CAMERAS"])
    camera.data.lens = 66.0
    camera.data.sensor_width = 36.0
    _look_at(camera, (0.0, 0.0, 0.50))
    bpy.context.scene.camera = camera

    neutral_collection = collections["COL_LIGHTS_NEUTRAL"]
    if not bpy.data.objects.get("LGT_Neutral_Key"):
        for name, location, energy, size in (
            ("LGT_Neutral_Key", (-1.6, -2.4, 2.8), 900.0, 2.0),
            ("LGT_Neutral_Fill", (2.2, -1.0, 1.6), 500.0, 2.4),
            ("LGT_Neutral_Rim", (0.4, 2.0, 2.4), 700.0, 1.5),
        ):
            data = bpy.data.lights.new(name + "_Data", "AREA")
            data.energy = energy
            data.shape = "DISK"
            data.size = size
            light = bpy.data.objects.new(name, data)
            neutral_collection.objects.link(light)
            light.location = location
            _look_at(light, (0.0, 0.0, 0.52))

    world = bpy.context.scene.world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        0.055,
        0.055,
        0.055,
        1.0,
    )
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.40

    floor = bpy.data.objects.get("SM_DiagnosticFloor")
    if floor is None:
        bpy.ops.mesh.primitive_plane_add(size=6.0, location=(0.0, 0.0, -0.003))
        floor = bpy.context.object
        floor.name = "SM_DiagnosticFloor"
        floor.data.name = "SM_DiagnosticFloor_Mesh"
        _move_object_to_collection(floor, collections["COL_DIAGNOSTICS"])
        floor.data.materials.append(
            _new_material(
                "MAT_DiagnosticFloor", (0.11, 0.11, 0.11, 1.0), metallic=0.0, roughness=0.75
            )
        )
    floor.hide_render = False
    return camera, floor


def _render_review_views(
    output_dir: Path,
    camera,
    floor,
    model_objects: Sequence[Any],
    *,
    prefix: str,
) -> dict[str, str]:
    bpy, _, _ = _bpy_modules()
    scene = bpy.context.scene
    output_dir.mkdir(parents=True, exist_ok=True)
    views = {
        "front": ((0.0, -2.8, 0.52), (0.0, 0.0, 0.50)),
        "quarter": ((1.55, -2.55, 0.78), (0.0, 0.0, 0.50)),
        "side": ((2.8, 0.0, 0.52), (0.0, 0.0, 0.50)),
        "top": ((1.25, -1.55, 2.35), (0.0, 0.0, 0.40)),
        "low": ((1.35, -2.75, 0.28), (0.0, 0.0, 0.48)),
    }
    rendered: dict[str, str] = {}
    for name, (location, target) in views.items():
        camera.location = location
        camera.data.type = "PERSP"
        _look_at(camera, target)
        path = output_dir / f"{prefix}_{name}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        rendered[name] = str(path)

    camera.location = (1.55, -2.55, 0.78)
    _look_at(camera, (0.0, 0.0, 0.50))
    wire_path = output_dir / f"{prefix}_wire.png"
    _render_true_wireframe(wire_path, model_objects, floor)
    rendered["wire"] = str(wire_path)
    return rendered


def _render_true_wireframe(
    output_path: Path,
    model_objects: Sequence[Any],
    floor,
) -> None:
    """Render actual topology through temporary Wireframe geometry modifiers."""

    bpy, _, _ = _bpy_modules()
    scene = bpy.context.scene
    wire_material = _new_material(
        "MAT_QA_TopologyWire",
        (0.075, 0.55, 0.95, 1.0),
        metallic=0.0,
        roughness=0.38,
    )
    temporary: list[tuple[Any, Any, int]] = []
    floor_was_hidden = bool(floor.hide_render)
    floor.hide_render = True
    try:
        for obj in model_objects:
            if getattr(obj, "type", None) != "MESH":
                continue
            material_slot = len(obj.data.materials)
            obj.data.materials.append(wire_material)
            modifier = obj.modifiers.new("TMP_QA_RenderableWireframe", "WIREFRAME")
            modifier.thickness = 0.00048
            modifier.use_replace = True
            modifier.use_even_offset = True
            modifier.material_offset = material_slot
            temporary.append((obj, modifier, material_slot))
        scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
    finally:
        floor.hide_render = floor_was_hidden
        for obj, modifier, material_slot in reversed(temporary):
            if modifier.name in obj.modifiers:
                obj.modifiers.remove(modifier)
            if len(obj.data.materials) > material_slot:
                obj.data.materials.pop(index=material_slot)


def _mesh_statistics(objects: Sequence[Any]) -> dict[str, Any]:
    bpy, _, _ = _bpy_modules()
    bpy.context.view_layer.update()
    entries: list[dict[str, Any]] = []
    total_vertices = 0
    total_polygons = 0
    for obj in objects:
        if obj.type != "MESH":
            continue
        vertices = len(obj.data.vertices)
        polygons = len(obj.data.polygons)
        total_vertices += vertices
        total_polygons += polygons
        entries.append(
            {
                "name": obj.name,
                "vertices": vertices,
                "polygons": polygons,
                "materials": [material.name for material in obj.data.materials],
                "source": obj.get("source"),
            }
        )
    return {
        "mesh_count": len(entries),
        "vertices": total_vertices,
        "polygons": total_polygons,
        "objects": entries,
    }


def _render_detail_views(
    output_dir: Path,
    camera,
    views: Mapping[str, tuple[tuple[float, float, float], tuple[float, float, float], float]],
) -> dict[str, dict[str, str]]:
    """Render the small source-comparison set used for visible Plan-3 decisions."""

    bpy, _, _ = _bpy_modules()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    rendered: dict[str, dict[str, str]] = {}
    for name, (location, target, lens) in views.items():
        camera.location = location
        camera.data.type = "PERSP"
        camera.data.lens = float(lens)
        _look_at(camera, target)
        path = output_dir / f"{name}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        rendered[name] = {"path": str(path), "sha256": sha256_file(path)}
    return rendered


def _clear_collection_objects(collection) -> None:
    bpy, _, _ = _bpy_modules()
    for obj in tuple(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _load_stage_blend(v2_root: Path, stage: str) -> Path:
    bpy, _, _ = _bpy_modules()
    checkpoints = stage_checkpoint_paths(v2_root)
    predecessors = {
        "geometry-validate": "base",
        "ornament": "geometry-validate",
        "cleanup": "ornament",
        "uv-bake": "cleanup",
        "lookdev": "uv-bake",
        "final-validate": "lookdev",
    }
    source = checkpoints[predecessors[stage]]
    if not source.is_file():
        raise FileNotFoundError(f"missing Blender stage input: {source}")
    bpy.ops.wm.open_mainfile(filepath=str(source))
    return source


def run_base_stage(v2_root: Path) -> dict[str, Any]:
    bpy, _, _ = _bpy_modules()
    v2_root = Path(v2_root).resolve()
    project_root = v2_root.parents[1]
    profile_path = v2_root / "reports" / "final_profiles.json"
    fit_path = v2_root / "reports" / "final_cv_fit.json"
    profiles_payload = read_json(profile_path)
    fit_payload = read_json(fit_path)
    if fit_payload.get("accepted") is not True:
        raise ValueError("Plan-1 final_cv_fit.json is not accepted")
    profiles = validate_profile_payload(profiles_payload)
    profile_hash = sha256_file(profile_path)
    candidate_hash = str(fit_payload.get("candidate_sha256", ""))
    if not candidate_hash:
        raise ValueError("accepted Plan-1 report lacks candidate SHA-256")

    scene, collections, root = setup_v2_scene(v2_root, profile_hash, candidate_hash)
    setup_path = v2_root / "work" / "00_scene_setup.blend"
    setup_path.parent.mkdir(parents=True, exist_ok=True)
    if not setup_path.exists():
        bpy.ops.wm.save_as_mainfile(filepath=str(setup_path))

    cameras = _create_reference_objects(
        profiles_payload, fit_payload, collections, root, project_root
    )
    model_objects = _build_base_geometry(profiles, collections, root)
    camera, floor = _setup_neutral_review(collections, model_objects)
    diagnostics = _render_review_views(
        v2_root / "diagnostics" / "20_base_geometry",
        camera,
        floor,
        model_objects,
        prefix="base_clay",
    )

    prevalidation = v2_root / "work" / "20_base_geometry_pre_validation.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(prevalidation))
    output = stage_checkpoint_paths(v2_root)["base"]
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    report = {
        "schema_version": 1,
        "stage": "base",
        "accepted": False,
        "validation_pending": True,
        "blend_path": output.relative_to(project_root).as_posix(),
        "blend_sha256": sha256_file(output),
        "plan1_candidate_sha256": candidate_hash,
        "profiles_path": profile_path.relative_to(project_root).as_posix(),
        "profiles_sha256": profile_hash,
        "camera_count": len(cameras),
        "camera_metric_contract": "raw_source_exact_simple_radial_regular_python",
        "camera_display_contract": "blender_pinhole_display_only",
        "canonical_selected_indices": [
            int(record["selected_index"]) for record in fit_payload.get("views", [])
        ],
        "dimensions_normalized": profile_measurements(profiles),
        "statistics": _mesh_statistics(model_objects),
        "diagnostics": diagnostics,
        "wireframe_diagnostic_method": (
            "temporary_renderable_WIREFRAME_modifiers_removed_after_render"
        ),
        "ornament_started": False,
    }
    write_json(v2_root / "reports" / "base_geometry_blender_report.json", report)
    return report


def run_geometry_validate_stage(v2_root: Path) -> dict[str, Any]:
    bpy, _, _ = _bpy_modules()
    v2_root = Path(v2_root).resolve()
    project_root = v2_root.parents[1]
    source = _load_stage_blend(v2_root, "geometry-validate")
    collections = {name: bpy.data.collections.get(name) for name in COLLECTIONS}
    missing = [name for name, value in collections.items() if value is None]
    if missing:
        raise ValueError("base scene is missing collections: " + ", ".join(missing))
    model_objects = [bpy.data.objects[name] for name in BASE_SURFACE_OBJECTS]
    model_objects.extend(
        obj
        for obj in collections["COL_LOW"].objects
        if obj.type == "MESH" and obj.name not in BASE_SURFACE_OBJECTS
    )
    camera, floor = _setup_neutral_review(collections, model_objects)
    diagnostics = _render_review_views(
        v2_root / "diagnostics" / "20_base_geometry" / "validation",
        camera,
        floor,
        model_objects,
        prefix="geometry_validation",
    )
    output = stage_checkpoint_paths(v2_root)["geometry-validate"]
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    report = {
        "schema_version": 1,
        "stage": "geometry-validate-blender",
        "accepted": False,
        "regular_python_cv_validation_pending": True,
        "source_blend": source.relative_to(project_root).as_posix(),
        "blend_path": output.relative_to(project_root).as_posix(),
        "blend_sha256": sha256_file(output),
        "statistics": _mesh_statistics(model_objects),
        "diagnostics": diagnostics,
        "wireframe_diagnostic_method": (
            "temporary_renderable_WIREFRAME_modifiers_removed_after_render"
        ),
    }
    write_json(v2_root / "reports" / "base_geometry_blender_report.json", report)
    return report


def run_ornament_stage(v2_root: Path) -> dict[str, Any]:
    """Build the photographed ornament families on the accepted CV geometry."""

    bpy, _, _ = _bpy_modules()
    v2_root = Path(v2_root).resolve()
    project_root = v2_root.parents[1]
    source = _load_stage_blend(v2_root, "ornament")
    source_sha256 = sha256_file(source)
    accepted_base = read_json(v2_root / "reports" / "base_geometry_report.json")
    if accepted_base.get("accepted") is not True:
        raise ValueError("Plan-2 base geometry report is not accepted")
    if str(accepted_base.get("blend_sha256")) != source_sha256:
        raise ValueError("Plan-2 base report does not bind the ornament source blend")

    rollback = v2_root / "work" / "31_before_ornament.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(rollback))
    profiles = validate_profile_payload(read_json(v2_root / "reports" / "final_profiles.json"))
    manifest_path = v2_root / "reports" / "ornament_manifest.json"
    manifest = read_json(manifest_path)
    families = validate_ornament_manifest(manifest)
    blueprint = build_ornament_blueprint(families)
    collections = {name: bpy.data.collections.get(name) for name in COLLECTIONS}
    missing_collections = [name for name, value in collections.items() if value is None]
    if missing_collections:
        raise ValueError("ornament scene is missing collections: " + ", ".join(missing_collections))
    root = bpy.data.objects.get("ROOT_ThaiLibationV2")
    if root is None:
        raise ValueError("ornament scene lacks ROOT_ThaiLibationV2")
    _clear_collection_objects(collections["COL_ORNAMENT_HIGH"])
    ornament_objects = _build_ornament_sources(profiles, blueprint, collections, root)

    # Keep the candidate editable: profile surfaces remain separate meshes and
    # all relief stays as named curve/torus sources until the cleanup checkpoint.
    base_objects = [obj for obj in collections["COL_LOW"].objects if obj.type == "MESH"]
    camera, floor = _setup_neutral_review(collections, base_objects)
    detail_dir = v2_root / "diagnostics" / "30_ornament"
    render_records = _render_detail_views(
        detail_dir,
        camera,
        {
            "globe_front_clay": ((0.0, -1.05, 0.52), (0.0, 0.0, 0.505), 78.0),
            "globe_quarter_clay": ((0.70, -0.88, 0.58), (0.0, 0.0, 0.505), 78.0),
            "neck_lid_clay": ((0.0, -0.90, 0.79), (0.0, 0.0, 0.785), 82.0),
            "bowl_flame_clay": ((0.0, -1.02, 0.34), (0.0, 0.0, 0.335), 76.0),
            "pedestal_clay": ((0.0, -0.96, 0.12), (0.0, 0.0, 0.115), 80.0),
            "full_quarter_clay": ((1.55, -2.55, 0.78), (0.0, 0.0, 0.50), 66.0),
        },
    )

    family_render = {
        "ORB_GLOBE_CROSSHATCH": "globe_front_clay",
        "ORB_GLOBE_HERO_MOTIF": "globe_front_clay",
        "ORB_GLOBE_SCROLL_BAND": "globe_quarter_clay",
        "ORB_GLOBE_LOWER_BAND": "globe_quarter_clay",
        "ORB_SHOULDER_RINGS": "globe_quarter_clay",
        "ORB_NECK_FIELD": "neck_lid_clay",
        "ORB_NECK_LOTUS": "neck_lid_clay",
        "ORB_NECK_UPPER_BAND": "neck_lid_clay",
        "ORB_LID_TIERS": "neck_lid_clay",
        "ORB_LID_DECOR_BAND": "neck_lid_clay",
        "ORB_BOWL_FLAME_BAND": "bowl_flame_clay",
        "ORB_PEDESTAL_RINGS": "pedestal_clay",
    }
    diagnostics: dict[str, dict[str, Any]] = {}
    for family_id, record in families.items():
        family_dir = detail_dir / family_id
        family_dir.mkdir(parents=True, exist_ok=True)
        source_crop = project_root / str(record["anchor_crop"]["crop_path"])
        copied_source = family_dir / "source_crop.png"
        shutil.copy2(source_crop, copied_source)
        render_name = family_render[family_id]
        render_path = Path(render_records[render_name]["path"])
        copied_render = family_dir / "model_clay.png"
        shutil.copy2(render_path, copied_render)
        diagnostics[family_id] = {
            "source_crop": {
                "path": copied_source.relative_to(project_root).as_posix(),
                "sha256": sha256_file(copied_source),
            },
            "model_render": {
                "path": copied_render.relative_to(project_root).as_posix(),
                "sha256": sha256_file(copied_render),
            },
            "source_views": record["primary_view_indices"],
            "representation": record["representation"],
        }

    represented = {
        "ORB_GLOBE_CROSSHATCH": ["HP_ORN_GlobeCrosshatchSource"],
        "ORB_GLOBE_HERO_MOTIF": ["HP_ORN_GlobeHeroMotif"],
        "ORB_GLOBE_SCROLL_BAND": ["HP_ORN_GlobeScrollBand"],
        "ORB_GLOBE_LOWER_BAND": ["SM_Globe_LowerConstructionBand"],
        "ORB_SHOULDER_RINGS": [
            "SM_Shoulder_Ring_01", "SM_Shoulder_Ring_02", "SM_Shoulder_Ring_03",
            "HP_ORN_ShoulderRing_04",
        ],
        "ORB_NECK_FIELD": ["HP_ORN_NeckFieldSource"],
        "ORB_NECK_LOTUS": ["HP_ORN_NeckLotus"],
        "ORB_NECK_UPPER_BAND": ["HP_ORN_NeckUpperBand"],
        "ORB_LID_TIERS": [f"SM_Lid_TierRing_{index:02d}" for index in range(1, 8)],
        "ORB_LID_DECOR_BAND": ["HP_ORN_LidDecorBand"],
        "ORB_BOWL_FLAME_BAND": ["HP_ORN_BowlFlameBand"],
        "ORB_PEDESTAL_RINGS": [
            "SM_Pedestal_FootRoll", "SM_Pedestal_UpperRoll",
            "HP_ORN_PedestalRing_03", "HP_ORN_PedestalRing_04",
        ],
    }
    missing_objects = {
        family_id: [name for name in names if bpy.data.objects.get(name) is None]
        for family_id, names in represented.items()
    }
    missing_objects = {key: value for key, value in missing_objects.items() if value}
    forbidden_chain = sorted(
        obj.name for obj in bpy.data.objects if "chain" in obj.name.lower()
    )
    forbidden_v1 = sorted(
        obj.name for obj in bpy.data.objects if "reference_assisted" in obj.name.lower()
    )
    output = stage_checkpoint_paths(v2_root)["ornament"]
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    accepted = not missing_objects and not forbidden_chain and not forbidden_v1
    report = {
        "schema_version": 1,
        "stage": "ornament",
        "accepted": accepted,
        "source_blend": source.relative_to(project_root).as_posix(),
        "source_blend_sha256": source_sha256,
        "blend_path": output.relative_to(project_root).as_posix(),
        "blend_sha256": sha256_file(output),
        "manifest_path": manifest_path.relative_to(project_root).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "source_policy": "project_photographs_and_existing_cv_feature_evidence_only",
        "family_count": len(represented),
        "represented_families": represented,
        "missing_objects": missing_objects,
        "ornament_source_object_count": len(ornament_objects),
        "editable_high_sources_preserved": True,
        "chain_decision": "omit_from_v2",
        "forbidden_chain_objects": forbidden_chain,
        "forbidden_v1_objects": forbidden_v1,
        "detail_renders": {
            name: {
                "path": Path(record["path"]).relative_to(project_root).as_posix(),
                "sha256": record["sha256"],
            }
            for name, record in render_records.items()
        },
        "family_diagnostics": diagnostics,
        "visual_review_scope": [
            "globe_hero_and_crosshatch", "globe_scroll_and_shoulder_rings",
            "neck_lotus_and_field", "lid_tiers", "bowl_flame_band", "pedestal_rings",
        ],
        "failure_reasons": (
            [] if accepted else ["missing_or_forbidden_ornament_objects"]
        ),
    }
    write_json(v2_root / "reports" / "ornament_build_report.json", report)
    return report


def _args(argv: Sequence[str] | None = None) -> tuple[str, Path]:
    values = list(sys.argv if argv is None else argv)
    if "--" not in values:
        raise SystemExit("expected -- <stage> <v2_root>")
    arguments = values[values.index("--") + 1 :]
    if len(arguments) != 2:
        raise SystemExit("expected exactly two script arguments")
    stage, root = arguments
    if stage not in BLENDER_STAGES:
        raise SystemExit(f"unsupported Blender stage: {stage}")
    return stage, Path(root).resolve()


def main() -> int:
    stage, v2_root = _args()
    if stage == "base":
        result = run_base_stage(v2_root)
    elif stage == "geometry-validate":
        result = run_geometry_validate_stage(v2_root)
    elif stage == "ornament":
        result = run_ornament_stage(v2_root)
    else:
        script_dir = str(Path(__file__).resolve().parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from final_blender_finish import (
            run_cleanup_stage,
            run_final_validate_stage,
            run_lookdev_stage,
            run_uv_bake_stage,
        )

        finish_stages = {
            "cleanup": run_cleanup_stage,
            "uv-bake": run_uv_bake_stage,
            "lookdev": run_lookdev_stage,
            "final-validate": run_final_validate_stage,
        }
        result = finish_stages[stage](v2_root)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
