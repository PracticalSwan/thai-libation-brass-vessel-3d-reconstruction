"""Build a fresh, scan-preserving Blender/GLB candidate from V4 Poisson.

The script is deliberately technical: it imports the exact versioned Poisson
mesh, preserves an untouched raw object, performs only reproducible smoothing,
decimation, UV, AO, material, and export operations, and records every source
hash and known anatomy limitation.  It never creates vessel geometry.

Run through Blender 5.2 with arguments after ``--``::

    blender.exe --background --python scripts/run_v4_best_defensible_blender.py -- \
      --mesh ... --output-dir ... --appearance-stats ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import bpy
from mathutils import Vector


COLLECTION_NAMES = ("COL_V4_Source", "COL_V4_Work", "COL_V4_Final", "COL_V4_Lookdev")
RAW_NAME = "SM_V4_Poisson_Raw"
CLEAN_NAME = "SM_V4_Scan_CleanHigh"
LOD0_NAME = "SM_V4_Vessel_LOD0"
MATERIAL_NAME = "MAT_V4_Brass"
AO_NAME = "T_V4_BestDefensible_AO"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def ensure_collection(name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
    if collection.name not in {child.name for child in bpy.context.scene.collection.children}:
        bpy.context.scene.collection.children.link(collection)
    return collection


def move_to_collection(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    for current in list(obj.users_collection):
        current.objects.unlink(obj)
    collection.objects.link(obj)


def activate_only(obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.hide_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def apply_transform(obj: bpy.types.Object) -> None:
    activate_only(obj)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def object_bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    minimum = Vector((min(point.x for point in points), min(point.y for point in points), min(point.z for point in points)))
    maximum = Vector((max(point.x for point in points), max(point.y for point in points), max(point.z for point in points)))
    return minimum, maximum


def center_and_floor(obj: bpy.types.Object) -> dict[str, Any]:
    minimum, maximum = object_bounds(obj)
    obj.location += Vector((-(minimum.x + maximum.x) / 2.0, -(minimum.y + maximum.y) / 2.0, -minimum.z))
    apply_transform(obj)
    minimum, maximum = object_bounds(obj)
    return {
        "min": [float(value) for value in minimum],
        "max": [float(value) for value in maximum],
        "dimensions": [float(value) for value in (maximum - minimum)],
    }


def duplicate_mesh(source: bpy.types.Object, name: str, collection: bpy.types.Collection) -> bpy.types.Object:
    duplicate = source.copy()
    duplicate.data = source.data.copy()
    duplicate.name = name
    duplicate.data.name = f"{name}_Mesh"
    collection.objects.link(duplicate)
    return duplicate


def apply_decimate(obj: bpy.types.Object, ratio: float, label: str) -> dict[str, Any]:
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"invalid decimate ratio for {label}: {ratio}")
    before = {"vertices": len(obj.data.vertices), "faces": len(obj.data.polygons)}
    activate_only(obj)
    modifier = obj.modifiers.new(name=f"DECIMATE_{label}", type="DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = ratio
    modifier.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    after = {"vertices": len(obj.data.vertices), "faces": len(obj.data.polygons)}
    return {"ratio": ratio, "before": before, "after": after}


def smooth_mesh(obj: bpy.types.Object) -> None:
    for polygon in obj.data.polygons:
        polygon.use_smooth = True


def create_material(appearance: dict[str, Any], ao_image: bpy.types.Image) -> bpy.types.Material:
    linear = appearance.get("base_color_linear_median")
    if not isinstance(linear, list) or len(linear) != 3:
        raise ValueError("appearance statistics lack a 3-channel linear median")
    metallic = float(appearance.get("metallic", 0.85))
    roughness = float(appearance.get("roughness"))
    material = bpy.data.materials.new(MATERIAL_NAME)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (420, 0)
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.location = (80, 0)
    shader.inputs["Base Color"].default_value = (*[float(value) for value in linear], 1.0)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    ao_node = nodes.new("ShaderNodeTexImage")
    ao_node.name = AO_NAME
    ao_node.label = "Geometry AO baked from fresh best-defensible LOD0"
    ao_node.image = ao_image
    ao_node.location = (-300, -180)
    # AO is connected to roughness only as a portable, non-colour-changing
    # detail signal; the measured median colour remains the base appearance.
    links.new(ao_node.outputs.get("Color"), shader.inputs["Roughness"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def ensure_uv(obj: bpy.types.Object) -> int:
    activate_only(obj)
    if obj.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(island_margin=0.02, area_weight=0.0, correct_aspect=True, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    return len(obj.data.uv_layers)


def bake_ao(obj: bpy.types.Object, material: bpy.types.Material, output_dir: Path, size: int = 1024) -> tuple[bpy.types.Image, dict[str, Any]]:
    image = bpy.data.images.get(AO_NAME)
    if image is None:
        image = bpy.data.images.new(AO_NAME, width=size, height=size, alpha=False, float_buffer=False)
    image.generated_color = (0.78, 0.78, 0.78, 1.0)
    image.colorspace_settings.name = "Non-Color"
    image.filepath_raw = str((output_dir / f"{AO_NAME}.png").resolve())
    material.node_tree.nodes.get(AO_NAME).image = image
    material.node_tree.nodes.active = material.node_tree.nodes.get(AO_NAME)
    activate_only(obj)
    scene = bpy.context.scene
    previous_engine = scene.render.engine
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    result: dict[str, Any] = {"status": "failed", "size": size, "path": str(Path(image.filepath_raw).resolve())}
    links = material.node_tree.links
    roughness_socket = material.node_tree.nodes.get("Principled BSDF").inputs["Roughness"] if material.node_tree.nodes.get("Principled BSDF") else None
    ao_links = [link for link in list(roughness_socket.links) if link.from_node == material.node_tree.nodes.get(AO_NAME)] if roughness_socket else []
    for link in ao_links:
        links.remove(link)
    try:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 16
        scene.cycles.use_denoising = False
        bpy.ops.object.bake(type="AO", width=size, height=size, margin=8, use_clear=True)
        image.pack()
        image.save_render(image.filepath_raw)
        result["status"] = "passed"
        result["sha256"] = sha256_file(Path(image.filepath_raw))
    except Exception as error:  # pragma: no cover - Blender runtime only
        result["error"] = f"{type(error).__name__}: {error}"
        image.generated_color = (0.78, 0.78, 0.78, 1.0)
        try:
            image.save_render(image.filepath_raw)
            image.pack()
        except Exception as fallback_error:
            result["fallback_error"] = f"{type(fallback_error).__name__}: {fallback_error}"
    finally:
        if roughness_socket is not None and not any(link.from_node == material.node_tree.nodes.get(AO_NAME) for link in roughness_socket.links):
            links.new(material.node_tree.nodes.get(AO_NAME).outputs.get("Color"), roughness_socket)
        scene.render.engine = previous_engine
    return image, result


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def setup_render_scene(lod: bpy.types.Object, bounds: dict[str, Any]) -> tuple[bpy.types.Object, list[bpy.types.Object]]:
    scene = bpy.context.scene
    minimum = Vector(bounds["min"])
    maximum = Vector(bounds["max"])
    center = (minimum + maximum) / 2.0
    dimensions = Vector(bounds["dimensions"])
    radius = max(dimensions) * 1.8
    camera_data = bpy.data.cameras.new("CAM_V4_BestDefensible_QA")
    camera = bpy.data.objects.new("CAM_V4_BestDefensible_QA", camera_data)
    ensure_collection("COL_V4_Lookdev").objects.link(camera)
    camera_data.lens = 58.0
    camera_data.sensor_width = 36.0
    scene.camera = camera

    lights: list[bpy.types.Object] = []
    for name, location, energy, size in (
        ("LGT_V4_BestDefensible_Key", (2.5, -3.0, 3.5), 1300.0, 3.0),
        ("LGT_V4_BestDefensible_Fill", (-2.0, 1.5, 2.2), 850.0, 4.0),
        ("LGT_V4_BestDefensible_Rim", (0.5, 2.5, 3.0), 1100.0, 2.5),
    ):
        data = bpy.data.lights.new(name, type="AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        light = bpy.data.objects.new(name, data)
        light.location = location
        look_at(light, center)
        ensure_collection("COL_V4_Lookdev").objects.link(light)
        lights.append(light)
    if scene.world is None:
        scene.world = bpy.data.worlds.new("WORLD_V4_BestDefensible")
    scene.world.color = (0.035, 0.035, 0.035)
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 700
    scene.render.resolution_y = 700
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    return camera, lights


def render_views(scene: bpy.types.Scene, camera: bpy.types.Object, bounds: dict[str, Any], output_dir: Path) -> dict[str, str]:
    minimum = Vector(bounds["min"])
    maximum = Vector(bounds["max"])
    center = (minimum + maximum) / 2.0
    dimensions = Vector(bounds["dimensions"])
    radius = max(dimensions) * 1.8
    output_dir.mkdir(parents=True, exist_ok=True)
    renders: dict[str, str] = {}
    for index, degrees in enumerate(range(0, 360, 45)):
        radians = math.radians(float(degrees))
        elevation = math.radians(10.0 if index % 2 == 0 else 16.0)
        camera.location = Vector(
            (
                radius * math.cos(radians) * math.cos(elevation),
                radius * math.sin(radians) * math.cos(elevation),
                center.z + radius * math.sin(elevation),
            )
        )
        look_at(camera, center)
        name = f"view_{index:02d}_{degrees:03d}.png"
        path = (output_dir / name).resolve()
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        renders[name] = sha256_file(path)
    return renders


def object_summary(obj: bpy.types.Object, bounds: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": obj.name,
        "vertices": len(obj.data.vertices),
        "faces": len(obj.data.polygons),
        "uv_layers": len(obj.data.uv_layers),
        "materials": [slot.material.name for slot in obj.material_slots if slot.material],
        "bounds": bounds,
        "finite_vertices": all(all(math.isfinite(float(value)) for value in vertex.co) for vertex in obj.data.vertices),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    mesh_path = args.mesh.resolve()
    output_dir = args.output_dir.resolve()
    appearance_path = args.appearance_stats.resolve()
    dense_selection_path = args.dense_selection.resolve()
    raw_diagnostic_path = args.raw_diagnostic.resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    if output_dir.exists():
        existing = [path for path in output_dir.iterdir() if path.resolve() != appearance_path]
        if existing:
            raise FileExistsError(f"refusing to overwrite non-empty versioned Blender output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    appearance = read_json(appearance_path)
    dense_selection = read_json(dense_selection_path)
    raw_diagnostic = read_json(raw_diagnostic_path)
    selected = dense_selection.get("selected", {})
    dense_sha = str(selected.get("fused_sha256", "")).lower()
    raw_sha = str(raw_diagnostic.get("poisson", {}).get("mesh_sha256", "")).lower()
    actual_raw_sha = sha256_file(mesh_path)
    if raw_sha and raw_sha != actual_raw_sha:
        raise ValueError(f"Poisson diagnostic hash does not match mesh bytes: {raw_sha} != {actual_raw_sha}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    source_collection = ensure_collection("COL_V4_Source")
    work_collection = ensure_collection("COL_V4_Work")
    final_collection = ensure_collection("COL_V4_Final")
    lookdev_collection = ensure_collection("COL_V4_Lookdev")
    del lookdev_collection  # retained by name for scene organization

    bpy.ops.wm.ply_import(filepath=str(mesh_path), global_scale=1.0, forward_axis="Y", up_axis="Z", merge_verts=False, import_colors="SRGB", import_attributes=True)
    imported = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if len(imported) != 1:
        raise RuntimeError(f"expected one imported Poisson mesh, found {len(imported)}")
    raw = imported[0]
    raw.name = RAW_NAME
    raw.data.name = f"{RAW_NAME}_Mesh"
    move_to_collection(raw, source_collection)
    # COLMAP's recovered vessel vertical is +Y.  Rotate the entire scan into
    # Blender's +Z convention, then center/floor it without changing shape.
    raw.rotation_euler = (math.pi / 2.0, 0.0, 0.0)
    apply_transform(raw)
    raw_bounds = center_and_floor(raw)
    smooth_mesh(raw)
    raw.hide_render = True
    raw.hide_set(True)

    clean = duplicate_mesh(raw, CLEAN_NAME, work_collection)
    clean.hide_render = True
    clean.hide_set(True)
    clean_transform = apply_decimate(clean, float(args.clean_ratio), "CLEAN_HIGH")
    smooth_mesh(clean)

    lod = duplicate_mesh(clean, LOD0_NAME, final_collection)
    lod.hide_render = False
    lod.hide_set(False)
    lod_transform = apply_decimate(lod, float(args.lod_ratio), "LOD0")
    smooth_mesh(lod)
    lod_bounds = center_and_floor(lod)
    uv_layers = ensure_uv(lod)

    ao_image = bpy.data.images.new(AO_NAME, width=int(args.ao_size), height=int(args.ao_size), alpha=False)
    material = create_material(appearance, ao_image)
    lod.data.materials.append(material)
    ao_image, ao_result = bake_ao(lod, material, output_dir, int(args.ao_size))
    # Ensure the image remains attached after baking/fallback.
    material.node_tree.nodes.get(AO_NAME).image = ao_image

    camera, lights = setup_render_scene(lod, lod_bounds)
    renders = render_views(bpy.context.scene, camera, lod_bounds, output_dir / "renders")
    for light in lights:
        light.hide_render = True
    camera.hide_render = True

    blend_path = output_dir / "Thai_Libation_Vessel_V4_BEST_DEFENSIBLE_TRIM5.blend"
    glb_path = output_dir / "Thai_Libation_Vessel_V4_BEST_DEFENSIBLE_TRIM5.glb"
    activate_only(lod)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    bpy.ops.export_scene.gltf(
        filepath=str(glb_path),
        export_format="GLB",
        use_selection=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
        export_texcoords=True,
        export_normals=True,
        export_vertex_color="NONE",
        export_cameras=False,
        export_lights=False,
        export_apply=True,
        export_yup=True,
    )
    # Saving after export records the final scene state and packed AO image.
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    report = {
        "schema_version": 1,
        "status": "complete_with_documented_anatomy_limitation",
        "stage": "fresh_best_defensible_blender_and_glb",
        "source": {
            "mesh_path": str(mesh_path),
            "mesh_sha256": actual_raw_sha,
            "dense_selection_path": str(dense_selection_path),
            "dense_selection_sha256": sha256_file(dense_selection_path),
            "selected_fused_sha256": dense_sha,
            "raw_diagnostic_path": str(raw_diagnostic_path),
            "raw_diagnostic_sha256": sha256_file(raw_diagnostic_path),
        },
        "appearance": {
            "stats_path": str(appearance_path),
            "stats_sha256": sha256_file(appearance_path),
            "image_count": appearance.get("image_count"),
            "source_manifest_sha256": appearance.get("source_manifest_sha256"),
            "photographic_projection_verified": appearance.get("photographic_projection_verified"),
            "base_color_srgb_median": appearance.get("base_color_srgb_median"),
            "roughness": appearance.get("roughness"),
            "metallic": appearance.get("metallic"),
        },
        "operations": {
            "orientation": "COLMAP +Y vertical rotated to Blender +Z; translation only centers XY and floors Z",
            "clean_high": clean_transform,
            "lod0": lod_transform,
            "smooth_shading": True,
            "uv_layers": uv_layers,
            "ao": ao_result,
            "geometry_creation": False,
        },
        "raw": object_summary(raw, raw_bounds),
        "clean_high": object_summary(clean, object_bounds(clean)),
        "lod0": object_summary(lod, lod_bounds),
        "scene": {
            "collections": sorted(collection.name for collection in bpy.context.scene.collection.children),
            "objects": sorted(obj.name for obj in bpy.context.scene.objects),
            "raw_preserved": raw.name == RAW_NAME and raw.hide_get(),
            "final_object": lod.name,
            "sane_transforms": all(
                abs(float(value) - expected) < 1e-6
                for obj in (raw, clean, lod)
                for value, expected in zip((*obj.location, *obj.rotation_euler, *obj.scale), (0, 0, 0, 0, 0, 0, 1, 1, 1))
            ),
        },
        "render_views": renders,
        "known_limitations": list(raw_diagnostic.get("blocking_reasons", []))
        + list(raw_diagnostic.get("anatomy_evidence", {}).get("failures", []))
        + ["raw Poisson component-fraction gate passes for trim-5, but anatomy gate remains failed; no vessel-scale geometry was fabricated"],
        "artifacts": {
            "blend": str(blend_path.resolve()),
            "blend_sha256": sha256_file(blend_path),
            "glb": str(glb_path.resolve()),
            "glb_sha256": sha256_file(glb_path),
            "ao": str((output_dir / f"{AO_NAME}.png").resolve()),
            "ao_sha256": sha256_file(output_dir / f"{AO_NAME}.png") if (output_dir / f"{AO_NAME}.png").is_file() else None,
        },
    }
    report_path = output_dir / "blender_best_defensible_v1_trim5.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--appearance-stats", type=Path, required=True)
    parser.add_argument("--dense-selection", type=Path, required=True)
    parser.add_argument("--raw-diagnostic", type=Path, required=True)
    parser.add_argument("--clean-ratio", type=float, default=0.12)
    parser.add_argument("--lod-ratio", type=float, default=0.25)
    parser.add_argument("--ao-size", type=int, default=1024)
    if argv is not None:
        return parser.parse_args(list(argv))
    raw = list(sys.argv)
    if "--" in raw:
        raw = raw[raw.index("--") + 1 :]
    else:
        raw = raw[1:]
    return parser.parse_args(raw)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = build(args)
    print(json.dumps({"status": report["status"], "blend": report["artifacts"]["blend"], "glb": report["artifacts"]["glb"], "lod0": report["lod0"], "ao": report["operations"]["ao"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
