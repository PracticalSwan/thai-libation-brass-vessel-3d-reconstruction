"""Deadline-focused Blender finishing stages for the CSX4213 Final V2 asset.

These stages deliberately preserve the accepted CV-constrained component meshes and
editable ornament sources.  They perform only practical cleanup, one shared UV set
with real geometry bakes, polished-brass lookdev, and representative pre-export QA.
There is intentionally no export code here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import build_final_model_blender as builder


# The accepted Plan-1/2 macro geometry remains the foundation. Strong top-oblique
# source views and explicit user review reveal one localized exception: the upper
# receiving-bowl lip is too narrow relative to the globe. This bounded Plan-4
# correction flares only the upper bowl lip/cavity; lower bowl geometry, globe
# geometry, axis, heights, and component positions remain unchanged. Values are
# normalized to the project's 1.0 set height.
BOWL_RIM_TO_GLOBE_RADIUS_RATIO = 1.14
BOWL_UPPER_FLARE_START_Z = 0.385
BOWL_GLOBE_CLEARANCE_AT_FLARE_START = 0.0060
BOWL_GLOBE_CLEARANCE_AT_RIM = 0.0180
BOWL_MIN_WALL_THICKNESS = 0.0015
BOWL_CLEARANCE_BLEND_DEPTH = 0.045
BOWL_RIM_MINOR_RADIUS = 0.0032


def _bpy_modules():
    import bpy  # type: ignore
    import bmesh  # type: ignore
    return bpy, bmesh


def _require_accepted(v2_root: Path, report_name: str) -> dict[str, Any]:
    payload = builder.read_json(Path(v2_root) / "reports" / report_name)
    if payload.get("accepted") is not True:
        raise ValueError(f"required report is not accepted: {report_name}")
    return payload


def _collections() -> dict[str, Any]:
    bpy, _ = _bpy_modules()
    result = {name: bpy.data.collections.get(name) for name in builder.COLLECTIONS}
    missing = [name for name, value in result.items() if value is None]
    if missing:
        raise ValueError("Final V2 scene is missing collections: " + ", ".join(missing))
    return result


def _mesh_objects(collections: dict[str, Any]) -> list[Any]:
    seen: set[str] = set()
    objects: list[Any] = []
    for collection_name in ("COL_LOW", "COL_ORNAMENT_HIGH"):
        for obj in collections[collection_name].objects:
            if obj.type == "MESH" and obj.name not in seen:
                seen.add(obj.name)
                objects.append(obj)
    return objects


def _render_objects(collections: dict[str, Any]) -> list[Any]:
    seen: set[str] = set()
    objects: list[Any] = []
    for collection_name in ("COL_LOW", "COL_ORNAMENT_HIGH"):
        for obj in collections[collection_name].objects:
            if obj.type in {"MESH", "CURVE"} and obj.name not in seen:
                seen.add(obj.name)
                objects.append(obj)
    return objects


def _forbidden_scene_objects() -> list[str]:
    bpy, _ = _bpy_modules()
    return sorted(
        obj.name
        for obj in bpy.data.objects
        if "chain" in obj.name.lower() or "reference_assisted" in obj.name.lower()
    )


def source_supported_bowl_outer_profile(
    bowl_outer: Iterable[tuple[float, float]],
    globe: Iterable[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Apply the source-supported upper receiving-bowl flare only."""

    outer = tuple((float(z), float(radius)) for z, radius in bowl_outer)
    globe_profile = tuple((float(z), float(radius)) for z, radius in globe)
    if not outer or not globe_profile:
        raise ValueError("bowl lip flare requires non-empty profiles")
    top_z, original_top_radius = outer[-1]
    globe_top_sample_z = min(max(top_z, globe_profile[0][0]), globe_profile[-1][0])
    globe_radius_at_top = builder._interpolate_radius(globe_profile, globe_top_sample_z)
    target_top_radius = max(
        original_top_radius,
        globe_radius_at_top * BOWL_RIM_TO_GLOBE_RADIUS_RATIO,
    )
    denominator = max(top_z - BOWL_UPPER_FLARE_START_Z, 1e-9)
    corrected: list[tuple[float, float]] = []
    for z, radius in outer:
        if z <= BOWL_UPPER_FLARE_START_Z:
            corrected.append((z, radius))
            continue
        t = max(0.0, min(1.0, (z - BOWL_UPPER_FLARE_START_Z) / denominator))
        t = t * t * (3.0 - 2.0 * t)
        corrected.append((z, radius + (target_top_radius - original_top_radius) * t))
    return tuple(corrected)


def source_supported_bowl_inner_profile(
    bowl_inner: Iterable[tuple[float, float]],
    bowl_outer: Iterable[tuple[float, float]],
    globe: Iterable[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Open the inferred upper bowl cavity to the source-supported annular gap."""

    inner = tuple((float(z), float(radius)) for z, radius in bowl_inner)
    outer = tuple((float(z), float(radius)) for z, radius in bowl_outer)
    globe_profile = tuple((float(z), float(radius)) for z, radius in globe)
    if not inner or not outer or not globe_profile:
        raise ValueError("bowl clearance correction requires non-empty profiles")

    globe_bottom = globe_profile[0][0]
    blend_start = globe_bottom - BOWL_CLEARANCE_BLEND_DEPTH
    top_z = inner[-1][0]
    flare_span = max(top_z - BOWL_UPPER_FLARE_START_Z, 1e-9)
    corrected: list[tuple[float, float]] = []
    for z, radius in inner:
        if z < blend_start:
            corrected.append((z, radius))
            continue

        outer_radius = builder._interpolate_radius(outer, z)
        globe_sample_z = min(max(z, globe_profile[0][0]), globe_profile[-1][0])
        globe_radius = builder._interpolate_radius(globe_profile, globe_sample_z)
        flare_t = max(0.0, min(1.0, (z - BOWL_UPPER_FLARE_START_Z) / flare_span))
        flare_t = flare_t * flare_t * (3.0 - 2.0 * flare_t)
        desired_gap = (
            BOWL_GLOBE_CLEARANCE_AT_FLARE_START
            + (BOWL_GLOBE_CLEARANCE_AT_RIM - BOWL_GLOBE_CLEARANCE_AT_FLARE_START) * flare_t
        )
        clearance_target = min(
            outer_radius - BOWL_MIN_WALL_THICKNESS,
            globe_radius + desired_gap,
        )
        if clearance_target <= 0.0:
            raise ValueError("bowl clearance correction produced a non-positive cavity radius")

        if z < globe_bottom:
            t = (z - blend_start) / max(globe_bottom - blend_start, 1e-9)
            t = max(0.0, min(1.0, t))
            t = t * t * (3.0 - 2.0 * t)
        else:
            t = 1.0
        corrected_radius = max(radius, radius + (clearance_target - radius) * t)
        corrected.append((z, corrected_radius))
    return tuple(corrected)


def _replace_lathe_mesh(obj: Any, profile: Iterable[tuple[float, float]]) -> None:
    bpy, bmesh = _bpy_modules()
    old_mesh = obj.data
    vertices, faces = builder.lathe_mesh_data(tuple(profile), segments=160, closed=True)
    mesh = bpy.data.meshes.new(obj.name + "_Mesh_Clearance")
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
    materials = [material for material in old_mesh.materials]
    obj.data = mesh
    for material in materials:
        mesh.materials.append(material)
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)


def _rebuild_bowl_rim(
    rim: Any,
    *,
    outer_radius: float,
    z: float,
    minor_radius: float,
) -> None:
    bpy, _ = _bpy_modules()
    old_mesh = rim.data
    materials = [material for material in old_mesh.materials]
    major_radius = max(outer_radius - minor_radius, minor_radius * 1.5)
    bpy.ops.mesh.primitive_torus_add(
        major_radius=major_radius,
        minor_radius=minor_radius,
        major_segments=128,
        minor_segments=12,
        location=(0.0, 0.0, z),
    )
    temporary = bpy.context.object
    new_mesh = temporary.data
    new_mesh.name = rim.name + "_Mesh_Clearance"
    rim.data = new_mesh
    rim.location = (0.0, 0.0, z)
    for material in materials:
        new_mesh.materials.append(material)
    for polygon in new_mesh.polygons:
        polygon.use_smooth = True
    bpy.data.objects.remove(temporary, do_unlink=True)
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)


def _apply_receiving_bowl_clearance(v2_root: Path) -> dict[str, Any]:
    bpy, _ = _bpy_modules()
    profiles = builder.validate_profile_payload(
        builder.read_json(v2_root / "reports" / "final_profiles.json")
    )
    bowl = bpy.data.objects.get("SM_Bowl")
    rim = bpy.data.objects.get("SM_Bowl_RolledRim")
    if bowl is None or rim is None:
        raise ValueError("receiving-bowl clearance correction requires bowl and rolled rim")

    corrected_outer = source_supported_bowl_outer_profile(
        profiles["bowl_outer"], profiles["globe"]
    )
    corrected_inner = source_supported_bowl_inner_profile(
        profiles["bowl_inner"], corrected_outer, profiles["globe"]
    )
    corrected_shell = builder.closed_shell_profile(corrected_outer, corrected_inner)
    _replace_lathe_mesh(bowl, corrected_shell)

    rim_z = 0.443
    rim_outer = builder._interpolate_radius(corrected_outer, rim_z)
    _rebuild_bowl_rim(
        rim,
        outer_radius=rim_outer,
        z=rim_z,
        minor_radius=BOWL_RIM_MINOR_RADIUS,
    )

    overlap_samples: list[dict[str, float]] = []
    globe_bottom = profiles["globe"][0][0]
    for z, inner_radius in corrected_inner:
        if z < globe_bottom:
            continue
        globe_radius = builder._interpolate_radius(profiles["globe"], z)
        overlap_samples.append(
            {
                "z": float(z),
                "inner_radius": float(inner_radius),
                "globe_radius": float(globe_radius),
                "radial_clearance": float(inner_radius - globe_radius),
            }
        )
    rim_globe_radius = builder._interpolate_radius(profiles["globe"], rim_z)
    rim_inner_radius = rim_outer - 2.0 * BOWL_RIM_MINOR_RADIUS
    rim_clearance = rim_inner_radius - rim_globe_radius
    min_shell_clearance = min(sample["radial_clearance"] for sample in overlap_samples)
    if min_shell_clearance <= 0.0 or rim_clearance <= 0.0:
        raise ValueError("receiving-bowl clearance correction still intersects the globe")

    bowl["source_supported_clearance_correction"] = True
    bowl["clearance_evidence"] = "top_oblique_globe_reference_views_267_268_278_288"
    rim["source_supported_clearance_correction"] = True
    rim["clearance_evidence"] = "top_oblique_globe_reference_views_267_268_278_288"
    outer_max_delta = max(
        abs(corrected_radius - original_radius)
        for (_, corrected_radius), (_, original_radius) in zip(corrected_outer, profiles["bowl_outer"])
    )
    top_z, top_outer_radius = corrected_outer[-1]
    top_globe_radius = builder._interpolate_radius(profiles["globe"], top_z)
    return {
        "policy": "localized_top_oblique_evidence_correction_upper_receiving_bowl_lip_only",
        "source_views": [267, 268, 278, 288],
        "evidence_reason": "top_only_step6_mismatch_plus_user_rejected_fused_globe_bowl_clearance",
        "target_rim_to_globe_radius_ratio": BOWL_RIM_TO_GLOBE_RADIUS_RATIO,
        "clearance_at_flare_start": BOWL_GLOBE_CLEARANCE_AT_FLARE_START,
        "clearance_at_rim_target": BOWL_GLOBE_CLEARANCE_AT_RIM,
        "minimum_wall_thickness": BOWL_MIN_WALL_THICKNESS,
        "minimum_shell_radial_clearance": min_shell_clearance,
        "rolled_rim_outer_radius": rim_outer,
        "rolled_rim_minor_radius": BOWL_RIM_MINOR_RADIUS,
        "rolled_rim_inner_radius": rim_inner_radius,
        "rolled_rim_radial_clearance": rim_clearance,
        "top_outer_radius": top_outer_radius,
        "top_globe_radius": top_globe_radius,
        "top_outer_to_globe_radius_ratio": top_outer_radius / top_globe_radius,
        "outer_profile_changed": True,
        "outer_profile_change_scope": "bowl_outer_z_above_0.385_only",
        "outer_profile_max_radius_delta": outer_max_delta,
        "lower_bowl_profile_unchanged": True,
        "globe_profile_changed": False,
        "component_positions_changed": False,
        "samples": overlap_samples,
    }


def _cleanup_mesh(obj: Any) -> dict[str, Any]:
    bpy, bmesh = _bpy_modules()
    mesh = obj.data
    before_vertices = len(mesh.vertices)
    before_polygons = len(mesh.polygons)
    mesh.validate(clean_customdata=False)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    degenerate = sum(1 for face in bm.faces if face.calc_area() <= 1e-12)
    nonmanifold = sum(1 for edge in bm.edges if not edge.is_manifold)
    loose_vertices = sum(1 for vertex in bm.verts if not vertex.link_edges)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    # Existing Plan-2 micro bevels are retained.  Add one only if an imported
    # explicit ring lacks it; do not globally remesh or decimate.
    if not any(mod.type == "BEVEL" for mod in obj.modifiers) and obj.name.startswith("SM_"):
        bevel = obj.modifiers.new("MOD_FinalMicroBevel", "BEVEL")
        bevel.width = 0.00075
        bevel.segments = 2
    return {
        "name": obj.name,
        "vertices": before_vertices,
        "polygons": before_polygons,
        "degenerate_faces": degenerate,
        "nonmanifold_edges": nonmanifold,
        "loose_vertices": loose_vertices,
    }


def run_cleanup_stage(v2_root: Path) -> dict[str, Any]:
    bpy, _ = _bpy_modules()
    v2_root = Path(v2_root).resolve()
    project_root = v2_root.parents[1]
    source = builder._load_stage_blend(v2_root, "cleanup")
    ornament = _require_accepted(v2_root, "ornament_build_report.json")
    if ornament.get("blend_sha256") != builder.sha256_file(source):
        raise ValueError("ornament report does not bind the cleanup source blend")
    collections = _collections()
    clearance_correction = _apply_receiving_bowl_clearance(v2_root)
    meshes = _mesh_objects(collections)
    cleanup = [_cleanup_mesh(obj) for obj in meshes]
    required = [name for name in builder.BASE_SURFACE_OBJECTS if bpy.data.objects.get(name) is None]
    forbidden = _forbidden_scene_objects()
    accepted = not required and not forbidden and bool(meshes)
    output = builder.stage_checkpoint_paths(v2_root)["cleanup"]
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    report = {
        "schema_version": 1,
        "stage": "cleanup",
        "accepted": accepted,
        "source_blend": source.relative_to(project_root).as_posix(),
        "source_blend_sha256": builder.sha256_file(source),
        "blend_path": output.relative_to(project_root).as_posix(),
        "blend_sha256": builder.sha256_file(output),
        "policy": "preserve_separate_CV_profile_parts_no_global_remesh_or_decimation",
        "receiving_bowl_clearance_correction": clearance_correction,
        "mesh_cleanup": cleanup,
        "required_base_objects_missing": required,
        "forbidden_objects": forbidden,
        "editable_ornament_sources_preserved": True,
        "failure_reasons": [] if accepted else ["required_objects_or_scene_policy_failed"],
    }
    builder.write_json(v2_root / "reports" / "cleanup_report.json", report)
    return report


def _select_meshes(meshes: Iterable[Any]) -> list[Any]:
    bpy, _ = _bpy_modules()
    bpy.ops.object.mode_set(mode="OBJECT") if bpy.context.object and bpy.context.object.mode != "OBJECT" else None
    bpy.ops.object.select_all(action="DESELECT")
    selected = list(meshes)
    for obj in selected:
        obj.hide_set(False)
        obj.select_set(True)
    if selected:
        bpy.context.view_layer.objects.active = selected[0]
    return selected


def _unwrap_shared_uv(meshes: list[Any]) -> None:
    bpy, _ = _bpy_modules()
    if not meshes:
        raise ValueError("no mesh objects available for UV unwrap")
    _select_meshes(meshes)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.012)
    bpy.ops.uv.pack_islands(margin=0.008)
    bpy.ops.object.mode_set(mode="OBJECT")


def _attach_bake_image(material: Any, image: Any) -> Any:
    material.use_nodes = True
    nodes = material.node_tree.nodes
    node = nodes.new("ShaderNodeTexImage")
    node.name = "TMP_FINAL_BAKE_TARGET"
    node.image = image
    nodes.active = node
    node.select = True
    return node


def _remove_bake_nodes(materials: Iterable[Any]) -> None:
    for material in materials:
        if not material or not material.use_nodes:
            continue
        node = material.node_tree.nodes.get("TMP_FINAL_BAKE_TARGET")
        if node is not None:
            material.node_tree.nodes.remove(node)


def _ensure_material_slots(meshes: list[Any]) -> list[Any]:
    bpy, _ = _bpy_modules()
    fallback = builder._new_material(
        "MAT_V2_BakeBase", (0.52, 0.32, 0.08, 1.0), metallic=0.0, roughness=0.45
    )
    materials: list[Any] = []
    seen: set[str] = set()
    for obj in meshes:
        if not obj.data.materials:
            obj.data.materials.append(fallback)
        for material in obj.data.materials:
            if material and material.name not in seen:
                seen.add(material.name)
                materials.append(material)
    return materials


def _save_bake(image: Any, path: Path) -> None:
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()


def _bake_map(meshes: list[Any], materials: list[Any], *, name: str, path: Path, bake_type: str) -> dict[str, Any]:
    bpy, _ = _bpy_modules()
    image = bpy.data.images.get(name) or bpy.data.images.new(name, width=2048, height=2048, alpha=False)
    image.generated_color = (0.5, 0.5, 1.0, 1.0) if bake_type == "NORMAL" else (1.0, 1.0, 1.0, 1.0)
    for material in materials:
        _attach_bake_image(material, image)
    _select_meshes(meshes)
    scene = bpy.context.scene
    # Blender's bake operator requires Cycles even when the final renderer is Eevee.
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 8
    scene.render.bake.margin = 8
    if bake_type == "NORMAL":
        bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT", use_clear=True)
    else:
        bpy.ops.object.bake(type=bake_type, use_clear=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_bake(image, path)
    _remove_bake_nodes(materials)
    return {"path": path.as_posix(), "sha256": builder.sha256_file(path), "resolution": [2048, 2048]}


def _curvature_bake(meshes: list[Any], path: Path) -> dict[str, Any]:
    bpy, _ = _bpy_modules()
    saved_slots = {obj.name: [mat for mat in obj.data.materials] for obj in meshes}
    material = bpy.data.materials.get("MAT_TMP_CurvatureBake") or bpy.data.materials.new("MAT_TMP_CurvatureBake")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    geometry = nodes.new("ShaderNodeNewGeometry")
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[1].position = 0.65
    material.node_tree.links.new(geometry.outputs["Pointiness"], ramp.inputs["Fac"])
    material.node_tree.links.new(ramp.outputs["Color"], emission.inputs["Color"])
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    image = bpy.data.images.get("T_ThaiLibation_Curvature") or bpy.data.images.new("T_ThaiLibation_Curvature", width=2048, height=2048, alpha=False)
    target = nodes.new("ShaderNodeTexImage")
    target.name = "TMP_FINAL_BAKE_TARGET"
    target.image = image
    nodes.active = target
    for obj in meshes:
        obj.data.materials.clear()
        obj.data.materials.append(material)
    try:
        _select_meshes(meshes)
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 8
        scene.render.bake.margin = 8
        bpy.ops.object.bake(type="EMIT", use_clear=True)
        _save_bake(image, path)
    finally:
        for obj in meshes:
            obj.data.materials.clear()
            for saved in saved_slots[obj.name]:
                obj.data.materials.append(saved)
    return {"path": path.as_posix(), "sha256": builder.sha256_file(path), "resolution": [2048, 2048]}


def run_uv_bake_stage(v2_root: Path) -> dict[str, Any]:
    bpy, _ = _bpy_modules()
    v2_root = Path(v2_root).resolve()
    project_root = v2_root.parents[1]
    source = builder._load_stage_blend(v2_root, "uv-bake")
    cleanup = _require_accepted(v2_root, "cleanup_report.json")
    if cleanup.get("blend_sha256") != builder.sha256_file(source):
        raise ValueError("cleanup report does not bind the UV source blend")
    collections = _collections()
    meshes = _mesh_objects(collections)
    _unwrap_shared_uv(meshes)
    materials = _ensure_material_slots(meshes)
    texture_dir = v2_root / "textures" / "final"
    bake_failures: list[str] = []
    bakes: dict[str, Any] = {}
    for key, filename, bake_type in (
        ("ao", "T_ThaiLibation_AO.png", "AO"),
        ("normal", "T_ThaiLibation_Normal.png", "NORMAL"),
    ):
        try:
            bakes[key] = _bake_map(meshes, materials, name=f"BAKE_{key}", path=texture_dir / filename, bake_type=bake_type)
        except Exception as exc:  # keep the asset recoverable and report exact failure
            _remove_bake_nodes(materials)
            bake_failures.append(f"{key}: {type(exc).__name__}: {exc}")
    try:
        bakes["curvature"] = _curvature_bake(meshes, texture_dir / "T_ThaiLibation_Curvature.png")
    except Exception as exc:
        bake_failures.append(f"curvature: {type(exc).__name__}: {exc}")
    uv_objects = [obj.name for obj in meshes if obj.data.uv_layers and obj.data.uv_layers.active]
    accepted = len(uv_objects) == len(meshes) and "ao" in bakes and "normal" in bakes
    scene = bpy.context.scene
    scene.render.engine = builder.RENDER_ENGINE
    output = builder.stage_checkpoint_paths(v2_root)["uv-bake"]
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    report = {
        "schema_version": 1,
        "stage": "uv-bake",
        "accepted": accepted,
        "source_blend": source.relative_to(project_root).as_posix(),
        "source_blend_sha256": builder.sha256_file(source),
        "blend_path": output.relative_to(project_root).as_posix(),
        "blend_sha256": builder.sha256_file(output),
        "uv_method": "shared_multi_object_smart_uv_then_pack",
        "uv_mesh_count": len(uv_objects),
        "mesh_count": len(meshes),
        "bakes": bakes,
        "bake_failures": bake_failures,
        "failure_reasons": [] if accepted else ["shared_uv_or_required_AO_normal_bake_failed"],
    }
    builder.write_json(v2_root / "reports" / "uv_bake_report.json", report)
    return report


def _set_input(node: Any, names: tuple[str, ...], value: Any) -> None:
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            socket.default_value = value
            return


def _load_image(path: Path, *, non_color: bool = False) -> Any | None:
    bpy, _ = _bpy_modules()
    if not path.is_file():
        return None
    image = bpy.data.images.load(str(path), check_existing=True)
    if non_color:
        image.colorspace_settings.name = "Non-Color"
    return image


def _build_brass_material(v2_root: Path) -> Any:
    bpy, _ = _bpy_modules()
    material = bpy.data.materials.get("MAT_FINAL_PolishedThaiBrass") or bpy.data.materials.new("MAT_FINAL_PolishedThaiBrass")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.name = "Final Polished Brass"
    _set_input(principled, ("Metallic",), 1.0)
    _set_input(principled, ("Roughness",), 0.205)
    _set_input(principled, ("Coat Weight", "Coat IOR Level"), 0.08)
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    gold = (0.83, 0.49, 0.105, 1.0)
    base_path = v2_root / "textures" / "final" / "T_ThaiLibation_BaseColor.png"
    base_image = _load_image(base_path)
    if base_image is not None:
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = base_image
        tex.interpolation = "Linear"
        mix = nodes.new("ShaderNodeMixRGB")
        mix.blend_type = "MIX"
        mix.inputs["Fac"].default_value = 0.26
        mix.inputs[1].default_value = gold
        links.new(tex.outputs["Color"], mix.inputs[2])
        links.new(mix.outputs["Color"], principled.inputs["Base Color"])
    else:
        principled.inputs["Base Color"].default_value = gold

    rough_image = _load_image(v2_root / "textures" / "final" / "T_ThaiLibation_Roughness.png", non_color=True)
    if rough_image is not None:
        rough_tex = nodes.new("ShaderNodeTexImage")
        rough_tex.image = rough_image
        rough_tex.interpolation = "Linear"
        map_range = nodes.new("ShaderNodeMapRange")
        map_range.inputs["From Min"].default_value = 0.0
        map_range.inputs["From Max"].default_value = 1.0
        map_range.inputs["To Min"].default_value = 0.14
        map_range.inputs["To Max"].default_value = 0.32
        links.new(rough_tex.outputs["Color"], map_range.inputs["Value"])
        links.new(map_range.outputs["Result"], principled.inputs["Roughness"])

    normal_image = _load_image(v2_root / "textures" / "final" / "T_ThaiLibation_Normal.png", non_color=True)
    normal_output = None
    if normal_image is not None:
        normal_tex = nodes.new("ShaderNodeTexImage")
        normal_tex.image = normal_image
        normal = nodes.new("ShaderNodeNormalMap")
        normal.inputs["Strength"].default_value = 0.35
        links.new(normal_tex.outputs["Color"], normal.inputs["Color"])
        normal_output = normal.outputs["Normal"]

    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 190.0
    noise.inputs["Detail"].default_value = 2.2
    noise.inputs["Roughness"].default_value = 0.58
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.028
    bump.inputs["Distance"].default_value = 0.00045
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    if normal_output is not None:
        links.new(normal_output, bump.inputs["Normal"])
    links.new(bump.outputs["Normal"], principled.inputs["Normal"])
    material["photo_informed_basecolor"] = str(base_path)
    material["metal_family"] = "polished_warm_brass"
    return material


def _assign_brass(material: Any, objects: Iterable[Any]) -> None:
    for obj in objects:
        data = getattr(obj, "data", None)
        if data is None or not hasattr(data, "materials"):
            continue
        data.materials.clear()
        data.materials.append(material)


def _setup_beauty_lighting(collections: dict[str, Any]) -> tuple[Any, Any]:
    bpy, _ = _bpy_modules()
    camera, floor = builder._setup_neutral_review(collections, _render_objects(collections))
    beauty = collections["COL_LIGHTS_BEAUTY"]
    for obj in tuple(beauty.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for name, location, energy, size, color in (
        ("LGT_Beauty_Key", (-1.7, -2.2, 2.55), 980.0, 2.6, (1.0, 0.82, 0.60)),
        ("LGT_Beauty_Fill", (2.3, -1.1, 1.55), 820.0, 3.0, (0.82, 0.91, 1.0)),
        ("LGT_Beauty_FrontSoft", (0.0, -2.85, 1.25), 930.0, 3.2, (1.0, 0.88, 0.70)),
        ("LGT_Beauty_Rim", (0.5, 2.2, 2.25), 820.0, 2.0, (1.0, 0.72, 0.46)),
        ("LGT_Beauty_Top", (0.0, 0.0, 3.2), 520.0, 1.8, (1.0, 0.95, 0.82)),
    ):
        data = bpy.data.lights.new(name + "_Data", "AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        data.color = color
        light = bpy.data.objects.new(name, data)
        beauty.objects.link(light)
        light.location = location
        builder._look_at(light, (0.0, 0.0, 0.50))
    # Neutral lights are kept for reproducibility but disabled for beauty renders.
    collections["COL_LIGHTS_NEUTRAL"].hide_render = True
    beauty.hide_render = False
    world = bpy.context.scene.world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.085, 0.095, 0.115, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.46
    floor.data.materials.clear()
    floor.data.materials.append(builder._new_material("MAT_FinalFloor", (0.055, 0.06, 0.072, 1.0), metallic=0.0, roughness=0.42))
    camera.data.lens = 72.0
    camera.location = (1.48, -2.45, 0.78)
    builder._look_at(camera, (0.0, 0.0, 0.50))
    bpy.context.scene.camera = camera
    return camera, floor


def run_lookdev_stage(v2_root: Path) -> dict[str, Any]:
    bpy, _ = _bpy_modules()
    v2_root = Path(v2_root).resolve()
    project_root = v2_root.parents[1]
    source = builder._load_stage_blend(v2_root, "lookdev")
    uv_report = _require_accepted(v2_root, "uv_bake_report.json")
    if uv_report.get("blend_sha256") != builder.sha256_file(source):
        raise ValueError("UV report does not bind the lookdev source blend")
    collections = _collections()
    render_objects = _render_objects(collections)
    brass = _build_brass_material(v2_root)
    _assign_brass(brass, render_objects)
    camera, floor = _setup_beauty_lighting(collections)
    scene = bpy.context.scene
    scene.render.engine = builder.RENDER_ENGINE
    scene.render.resolution_x = 720
    scene.render.resolution_y = 900
    scene.view_settings.look = "AgX - Medium High Contrast"
    views = builder._render_detail_views(
        v2_root / "diagnostics" / "50_lookdev",
        camera,
        {
            "beauty_front": ((0.0, -2.65, 0.56), (0.0, 0.0, 0.50), 72.0),
            "beauty_quarter": ((1.48, -2.45, 0.78), (0.0, 0.0, 0.50), 72.0),
            "beauty_side": ((2.65, 0.0, 0.56), (0.0, 0.0, 0.50), 72.0),
            "globe_detail": ((0.48, -1.10, 0.54), (0.0, 0.0, 0.51), 88.0),
            "assembly_clearance_detail": ((0.52, -0.78, 1.12), (0.0, 0.0, 0.43), 78.0),
            "neck_lid_detail": ((0.28, -0.96, 0.80), (0.0, 0.0, 0.79), 92.0),
            "bowl_pedestal_detail": ((0.42, -1.18, 0.28), (0.0, 0.0, 0.27), 88.0),
        },
    )
    camera.location = (1.48, -2.45, 0.78)
    camera.data.lens = 72.0
    builder._look_at(camera, (0.0, 0.0, 0.50))
    scene.camera = camera
    output = builder.stage_checkpoint_paths(v2_root)["lookdev"]
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    forbidden = _forbidden_scene_objects()
    accepted = not forbidden and all(Path(record["path"]).is_file() for record in views.values())
    report = {
        "schema_version": 1,
        "stage": "lookdev",
        "accepted": accepted,
        "source_blend": source.relative_to(project_root).as_posix(),
        "source_blend_sha256": builder.sha256_file(source),
        "blend_path": output.relative_to(project_root).as_posix(),
        "blend_sha256": builder.sha256_file(output),
        "material": brass.name,
        "material_contract": {"metallic": 1.0, "roughness_target": "0.14-0.32", "photo_informed_basecolor_blend": 0.26},
        "beauty_lighting": [obj.name for obj in collections["COL_LIGHTS_BEAUTY"].objects],
        "renders": {name: {"path": Path(record["path"]).relative_to(project_root).as_posix(), "sha256": record["sha256"]} for name, record in views.items()},
        "forbidden_objects": forbidden,
        "failure_reasons": [] if accepted else ["lookdev_render_or_scene_policy_failed"],
    }
    builder.write_json(v2_root / "reports" / "lookdev_report.json", report)
    return report


def _render_uv_checker(v2_root: Path, camera: Any, floor: Any, mesh_objects: list[Any]) -> Path:
    bpy, _ = _bpy_modules()
    saved = {obj.name: [mat for mat in obj.data.materials] for obj in mesh_objects}
    checker = bpy.data.materials.get("MAT_QA_UVChecker") or bpy.data.materials.new("MAT_QA_UVChecker")
    checker.use_nodes = True
    nodes = checker.node_tree.nodes
    links = checker.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    checker_node = nodes.new("ShaderNodeTexChecker")
    checker_node.inputs["Scale"].default_value = 28.0
    checker_node.inputs["Color1"].default_value = (0.05, 0.12, 0.42, 1.0)
    checker_node.inputs["Color2"].default_value = (0.92, 0.92, 0.92, 1.0)
    links.new(texcoord.outputs["UV"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], checker_node.inputs["Vector"])
    links.new(checker_node.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    for obj in mesh_objects:
        obj.data.materials.clear()
        obj.data.materials.append(checker)
    path = v2_root / "diagnostics" / "60_final_validation" / "uv_checker.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    camera.location = (1.48, -2.45, 0.78)
    camera.data.lens = 72.0
    builder._look_at(camera, (0.0, 0.0, 0.50))
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    for obj in mesh_objects:
        obj.data.materials.clear()
        for material in saved[obj.name]:
            obj.data.materials.append(material)
    return path


def run_final_validate_stage(v2_root: Path) -> dict[str, Any]:
    bpy, _ = _bpy_modules()
    v2_root = Path(v2_root).resolve()
    project_root = v2_root.parents[1]
    source = builder._load_stage_blend(v2_root, "final-validate")
    lookdev = _require_accepted(v2_root, "lookdev_report.json")
    if lookdev.get("blend_sha256") != builder.sha256_file(source):
        raise ValueError("lookdev report does not bind the final-validation source blend")
    collections = _collections()
    meshes = _mesh_objects(collections)
    render_objects = _render_objects(collections)
    camera = bpy.data.objects.get("CAM_Hero_Review")
    floor = bpy.data.objects.get("SM_DiagnosticFloor")
    if camera is None or floor is None:
        camera, floor = _setup_beauty_lighting(collections)
    final_dir = v2_root / "diagnostics" / "60_final_validation"
    review = builder._render_review_views(final_dir, camera, floor, meshes, prefix="final")
    details = builder._render_detail_views(
        final_dir,
        camera,
        {
            "final_globe_close": ((0.48, -1.10, 0.54), (0.0, 0.0, 0.51), 88.0),
            "final_assembly_clearance_close": ((0.52, -0.78, 1.12), (0.0, 0.0, 0.43), 78.0),
            "final_neck_lid_close": ((0.28, -0.96, 0.80), (0.0, 0.0, 0.79), 92.0),
            "final_bowl_pedestal_close": ((0.42, -1.18, 0.28), (0.0, 0.0, 0.27), 88.0),
        },
    )
    uv_checker = _render_uv_checker(v2_root, camera, floor, meshes)
    forbidden = _forbidden_scene_objects()
    missing_uv = [obj.name for obj in meshes if not obj.data.uv_layers or not obj.data.uv_layers.active]
    # Keep references/debug data organized but out of the startup presentation.
    for name in ("COL_REFERENCE", "COL_BLOCKOUT"):
        collections[name].hide_render = True
        collections[name].hide_viewport = True
    collections["COL_EXPORT"].hide_viewport = True
    collections["COL_EXPORT"].hide_render = True
    camera.location = (1.48, -2.45, 0.78)
    camera.data.lens = 72.0
    builder._look_at(camera, (0.0, 0.0, 0.50))
    bpy.context.scene.camera = camera
    output = builder.stage_checkpoint_paths(v2_root)["final-validate"]
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    accepted = not forbidden and not missing_uv and len(render_objects) >= len(builder.BASE_SURFACE_OBJECTS)
    report = {
        "schema_version": 1,
        "stage": "final-validate",
        "accepted": accepted,
        "user_review_required_before_export": True,
        "export_performed": False,
        "source_blend": source.relative_to(project_root).as_posix(),
        "source_blend_sha256": builder.sha256_file(source),
        "blend_path": output.relative_to(project_root).as_posix(),
        "blend_sha256": builder.sha256_file(output),
        "render_object_count": len(render_objects),
        "mesh_statistics": builder._mesh_statistics(meshes),
        "missing_uv_objects": missing_uv,
        "forbidden_objects": forbidden,
        "chain_decision": "omit_from_v2",
        "review_renders": {name: {"path": Path(path).relative_to(project_root).as_posix(), "sha256": builder.sha256_file(Path(path))} for name, path in review.items()},
        "detail_renders": {name: {"path": Path(record["path"]).relative_to(project_root).as_posix(), "sha256": record["sha256"]} for name, record in details.items()},
        "uv_checker": {"path": uv_checker.relative_to(project_root).as_posix(), "sha256": builder.sha256_file(uv_checker)},
        "failure_reasons": [] if accepted else ["final_scene_policy_or_uv_gate_failed"],
    }
    builder.write_json(v2_root / "reports" / "final_validation_report.json", report)
    return report
