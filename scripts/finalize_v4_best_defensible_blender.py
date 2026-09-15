"""Finalize the preserved V4 Blender authoring scene without changing anatomy.

This post-Blender continuation is deliberately technical rather than reconstructive:
- preserve the exact raw Poisson, clean-high and LOD0 geometry/transforms;
- eliminate the authoring viewport overlap by hiding raw/clean-high sources;
- derive final visible appearance only from the frozen 158-image uncoated-project
  appearance statistics (no manual/reference-assisted colour or photo projection);
- preserve/export the existing scan-derived AO, bake clean-high -> LOD0 tangent
  normal/detail, and export only LOD0 to GLB;
- bind the final artifact to the frozen pre-Blender Poisson handoff and recovery
  evidence while carrying unresolved anatomy limitations forward unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import sys
import zlib
from pathlib import Path
from typing import Any, Iterable, Sequence

import bpy


RAW_NAME = "SM_V4_Poisson_Raw"
CLEAN_NAME = "SM_V4_Scan_CleanHigh"
LOD0_NAME = "SM_V4_Vessel_LOD0"
MATERIAL_NAME = "MAT_V4_Brass"
AO_NAME = "T_V4_BestDefensible_AO"
BASE_COLOR_NAME = "T_V4_BestDefensible_BaseColor"
ROUGHNESS_NAME = "T_V4_BestDefensible_Roughness"
NORMAL_NAME = "T_V4_BestDefensible_Normal"
NORMAL_MAP_NODE = "N_V4_BestDefensible_NormalMap"
GLTF_OUTPUT_GROUP = "glTF Material Output"
GLTF_OUTPUT_NODE = "N_V4_glTF_Material_Output"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def geometry_signature(obj: bpy.types.Object) -> dict[str, Any]:
    dimensions = tuple(float(value) for value in obj.dimensions)
    return {
        "vertices": len(obj.data.vertices),
        "edges": len(obj.data.edges),
        "faces": len(obj.data.polygons),
        "dimensions": list(dimensions),
        "location": [float(value) for value in obj.location],
        "rotation_euler": [float(value) for value in obj.rotation_euler],
        "scale": [float(value) for value in obj.scale],
        "finite_vertices": all(
            all(math.isfinite(float(value)) for value in vertex.co)
            for vertex in obj.data.vertices
        ),
    }


def uv_metrics(obj: bpy.types.Object) -> dict[str, Any]:
    layer = obj.data.uv_layers.active
    if layer is None:
        return {"present": False}
    areas: list[float] = []
    zero = 0
    minimum_u = minimum_v = float("inf")
    maximum_u = maximum_v = float("-inf")
    occupied: set[tuple[int, int]] = set()
    bins = 128
    for polygon in obj.data.polygons:
        points: list[tuple[float, float]] = []
        for loop_index in polygon.loop_indices:
            uv = layer.data[loop_index].uv
            u, v = float(uv.x), float(uv.y)
            points.append((u, v))
            minimum_u, maximum_u = min(minimum_u, u), max(maximum_u, u)
            minimum_v, maximum_v = min(minimum_v, v), max(maximum_v, v)
            occupied.add((max(0, min(bins - 1, int(u * bins))), max(0, min(bins - 1, int(v * bins)))))
        area = 0.0
        for index, (x1, y1) in enumerate(points):
            x2, y2 = points[(index + 1) % len(points)]
            area += x1 * y2 - x2 * y1
        area = abs(area) * 0.5
        areas.append(area)
        zero += area < 1.0e-12
    return {
        "present": True,
        "name": layer.name,
        "polygon_count": len(areas),
        "bounds": [minimum_u, minimum_v, maximum_u, maximum_v],
        "uv_area_sum": sum(areas),
        "zero_area_faces": zero,
        "mean_face_uv_area": sum(areas) / len(areas) if areas else 0.0,
        "max_face_uv_area": max(areas) if areas else 0.0,
        "occupied_bins_128": len(occupied),
        "occupied_fraction_128": len(occupied) / float(bins * bins),
    }


def generate_box_atlas_uv(obj: bpy.types.Object, *, padding: float = 0.03) -> dict[str, Any]:
    """Generate a deterministic six-way geometry-derived box atlas.

    A conventional smart projection fragments the noisy scan into almost one UV
    island per triangle. This signed-dominant-normal atlas instead projects each
    face into one of six fixed atlas cells using only the reconstructed geometry.
    It changes UV coordinates only; mesh vertices/faces/transforms are untouched.
    """
    mesh = obj.data
    for layer in list(mesh.uv_layers):
        mesh.uv_layers.remove(layer)
    layer = mesh.uv_layers.new(name="UVMap")
    mesh.uv_layers.active = layer
    layer.active_render = True

    vertices = [vertex.co for vertex in mesh.vertices]
    minimum = [min(float(vertex[index]) for vertex in vertices) for index in range(3)]
    maximum = [max(float(vertex[index]) for vertex in vertices) for index in range(3)]
    span = [max(1.0e-9, maximum[index] - minimum[index]) for index in range(3)]
    tiles = {"+X": (0, 0), "-X": (1, 0), "+Y": (2, 0), "-Y": (0, 1), "+Z": (1, 1), "-Z": (2, 1)}

    for polygon in mesh.polygons:
        normal = polygon.normal
        magnitudes = [abs(float(normal.x)), abs(float(normal.y)), abs(float(normal.z))]
        axis = magnitudes.index(max(magnitudes))
        key = ("+" if float(normal[axis]) >= 0.0 else "-") + "XYZ"[axis]
        column, row = tiles[key]
        for loop_index in polygon.loop_indices:
            coordinate = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            x = (float(coordinate.x) - minimum[0]) / span[0]
            y = (float(coordinate.y) - minimum[1]) / span[1]
            z = (float(coordinate.z) - minimum[2]) / span[2]
            if key == "+X":
                s, t = y, z
            elif key == "-X":
                s, t = 1.0 - y, z
            elif key == "+Y":
                s, t = 1.0 - x, z
            elif key == "-Y":
                s, t = x, z
            elif key == "+Z":
                s, t = x, y
            else:
                s, t = 1.0 - x, y
            s = padding + s * (1.0 - 2.0 * padding)
            t = padding + t * (1.0 - 2.0 * padding)
            layer.data[loop_index].uv = ((column + s) / 3.0, (row + t) / 2.0)
    result = uv_metrics(obj)
    result.update({"method": "signed_dominant_normal_six_way_box_atlas", "padding_per_cell": padding, "geometry_creation": False})
    if result.get("uv_area_sum", 0.0) < 0.25 or result.get("occupied_fraction_128", 0.0) < 0.25:
        raise RuntimeError(f"generated UV atlas has insufficient usable coverage: {result}")
    return result


def require_scene_objects() -> tuple[bpy.types.Object, bpy.types.Object, bpy.types.Object, bpy.types.Material]:
    raw = bpy.data.objects.get(RAW_NAME)
    clean = bpy.data.objects.get(CLEAN_NAME)
    lod = bpy.data.objects.get(LOD0_NAME)
    if raw is None or raw.type != "MESH":
        raise RuntimeError(f"missing raw mesh: {RAW_NAME}")
    if clean is None or clean.type != "MESH":
        raise RuntimeError(f"missing clean-high mesh: {CLEAN_NAME}")
    if lod is None or lod.type != "MESH":
        raise RuntimeError(f"missing LOD0 mesh: {LOD0_NAME}")
    material = bpy.data.materials.get(MATERIAL_NAME)
    if material is None or not material.use_nodes or material.node_tree is None:
        raise RuntimeError(f"missing node material: {MATERIAL_NAME}")
    if len(lod.data.uv_layers) < 1:
        raise RuntimeError("LOD0 has no UV layer for texture/detail baking")
    return raw, clean, lod, material


def _principled(material: bpy.types.Material) -> bpy.types.Node:
    for node in material.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    raise RuntimeError("brass material has no Principled BSDF node")


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def write_solid_rgb_png(path: Path, rgb: Sequence[float], *, size: int = 32) -> None:
    """Write a deterministic sRGB/non-colour 8-bit RGB PNG without external packages."""
    if len(rgb) != 3:
        raise ValueError("RGB value must have exactly three channels")
    channels = [max(0, min(255, int(round(float(value) * 255.0)))) for value in rgb]
    row = b"\x00" + bytes(channels) * size
    raw = row * size
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    payload = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", header) + _png_chunk(b"IDAT", zlib.compress(raw, 9)) + _png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def _replace_image(name: str, path: Path, colorspace: str) -> bpy.types.Image:
    existing = bpy.data.images.get(name)
    if existing is not None:
        bpy.data.images.remove(existing)
    image = bpy.data.images.load(str(path.resolve()), check_existing=False)
    image.name = name
    image.colorspace_settings.name = colorspace
    image.pack()
    return image


def _ensure_texture_node(material: bpy.types.Material, name: str, image: bpy.types.Image, x: float, y: float) -> bpy.types.Node:
    nodes = material.node_tree.nodes
    node = nodes.get(name)
    if node is not None and node.type != "TEX_IMAGE":
        nodes.remove(node)
        node = None
    if node is None:
        node = nodes.new("ShaderNodeTexImage")
        node.name = name
    node.label = name
    node.image = image
    node.location = (x, y)
    return node


def _remove_input_links(material: bpy.types.Material, socket: bpy.types.NodeSocket) -> None:
    for link in list(socket.links):
        material.node_tree.links.remove(link)


def prepare_gltf_occlusion(material: bpy.types.Material, ao_node: bpy.types.Node) -> bpy.types.Node:
    """Expose the scan-derived AO through Blender's glTF exporter occlusion hook."""
    tree = material.node_tree
    nodes = tree.nodes
    group_tree = bpy.data.node_groups.get(GLTF_OUTPUT_GROUP)
    if group_tree is None:
        group_tree = bpy.data.node_groups.new(GLTF_OUTPUT_GROUP, "ShaderNodeTree")
        group_tree.interface.new_socket(name="Occlusion", in_out="INPUT", socket_type="NodeSocketFloat")
    elif "Occlusion" not in [item.name for item in group_tree.interface.items_tree if getattr(item, "item_type", None) == "SOCKET"]:
        group_tree.interface.new_socket(name="Occlusion", in_out="INPUT", socket_type="NodeSocketFloat")

    node = nodes.get(GLTF_OUTPUT_NODE)
    if node is not None and node.type != "GROUP":
        nodes.remove(node)
        node = None
    if node is None:
        node = nodes.new("ShaderNodeGroup")
        node.name = GLTF_OUTPUT_NODE
    node.label = GLTF_OUTPUT_GROUP
    node.node_tree = group_tree
    node.location = (40, -520)
    occlusion = node.inputs.get("Occlusion")
    if occlusion is None:
        raise RuntimeError("glTF Material Output node does not expose Occlusion")
    _remove_input_links(material, occlusion)
    tree.links.new(ao_node.outputs["Color"], occlusion)
    return node


def prepare_project_appearance(
    material: bpy.types.Material,
    output_dir: Path,
    appearance_stats_path: Path,
) -> dict[str, Any]:
    stats = read_json(appearance_stats_path)
    if int(stats.get("image_count", 0)) != 158:
        raise RuntimeError("appearance statistics do not bind the complete 158-image uncoated set")
    if stats.get("source_role") != "appearance_reference":
        raise RuntimeError("appearance statistics source role is not appearance_reference")
    if stats.get("photographic_projection_verified") is not False:
        raise RuntimeError("this finalizer expects the fail-closed non-projected appearance route")

    srgb = [float(value) for value in stats["base_color_srgb_median"]]
    linear = [float(value) for value in stats["base_color_linear_median"]]
    roughness = float(stats["roughness"])
    metallic = float(stats["metallic"])
    if len(srgb) != 3 or len(linear) != 3:
        raise RuntimeError("appearance BaseColor must have three channels")
    if not all(0.0 <= value <= 1.0 for value in srgb + linear + [roughness, metallic]):
        raise RuntimeError("appearance statistics contain an out-of-range material value")

    base_path = output_dir / f"{BASE_COLOR_NAME}.png"
    rough_path = output_dir / f"{ROUGHNESS_NAME}.png"
    write_solid_rgb_png(base_path, srgb)
    write_solid_rgb_png(rough_path, (roughness, roughness, roughness))
    base_image = _replace_image(BASE_COLOR_NAME, base_path, "sRGB")
    rough_image = _replace_image(ROUGHNESS_NAME, rough_path, "Non-Color")

    tree = material.node_tree
    shader = _principled(material)
    base_node = _ensure_texture_node(material, BASE_COLOR_NAME, base_image, -620, 250)
    rough_node = _ensure_texture_node(material, ROUGHNESS_NAME, rough_image, -620, -80)
    # The historical authoring scene accidentally used AO as Roughness. Remove
    # inherited appearance links and build explicit project-derived
    # BaseColor/Roughness. AO is rebaked after the replacement UV atlas exists.
    _remove_input_links(material, shader.inputs["Base Color"])
    _remove_input_links(material, shader.inputs["Roughness"])
    tree.links.new(base_node.outputs["Color"], shader.inputs["Base Color"])
    tree.links.new(rough_node.outputs["Color"], shader.inputs["Roughness"])
    shader.inputs["Metallic"].default_value = metallic

    copied_stats = output_dir / "appearance_stats.json"
    shutil.copy2(appearance_stats_path, copied_stats)
    return {
        "appearance_stats": str(appearance_stats_path.resolve()),
        "appearance_stats_sha256": sha256_file(appearance_stats_path),
        "copied_appearance_stats": str(copied_stats.resolve()),
        "copied_appearance_stats_sha256": sha256_file(copied_stats),
        "source_manifest": stats.get("source_manifest"),
        "source_manifest_sha256": stats.get("source_manifest_sha256"),
        "source_role": stats.get("source_role"),
        "image_count": int(stats["image_count"]),
        "selected_pixel_count": int(stats["selected_pixel_count"]),
        "base_color_srgb_median": srgb,
        "base_color_linear_median": linear,
        "roughness": roughness,
        "roughness_provenance": stats.get("roughness_provenance"),
        "metallic": metallic,
        "metallic_provenance": stats.get("metallic_provenance"),
        "photographic_projection_verified": False,
        "base_color_texture": {"path": str(base_path.resolve()), "sha256": sha256_file(base_path)},
        "roughness_texture": {"path": str(rough_path.resolve()), "sha256": sha256_file(rough_path)},
        "coated_geometry_vertex_colors_used_for_final_appearance": False,
        "manual_reference_matching_used": False,
        "artist_authored_color_or_texture_used": False,
    }


def prepare_ao_node(material: bpy.types.Material, output_dir: Path, *, size: int) -> tuple[bpy.types.Image, bpy.types.Node]:
    nodes = material.node_tree.nodes
    node = nodes.get(AO_NAME)
    if node is not None and node.type != "TEX_IMAGE":
        nodes.remove(node)
        node = None
    old_image = bpy.data.images.get(AO_NAME)
    if old_image is not None:
        bpy.data.images.remove(old_image)
    image = bpy.data.images.new(AO_NAME, width=size, height=size, alpha=False, float_buffer=False)
    image.generated_color = (1.0, 1.0, 1.0, 1.0)
    image.colorspace_settings.name = "Non-Color"
    image.file_format = "PNG"
    image.filepath_raw = str((output_dir / f"{AO_NAME}.png").resolve())
    if node is None:
        node = nodes.new("ShaderNodeTexImage")
        node.name = AO_NAME
    node.label = "LOD0 geometry-derived ambient occlusion"
    node.image = image
    node.location = (-620, -520)
    nodes.active = node
    return image, node


def bake_scan_ao(
    lod: bpy.types.Object,
    material: bpy.types.Material,
    output_dir: Path,
    *,
    size: int,
) -> dict[str, Any]:
    image, node = prepare_ao_node(material, output_dir, size=size)
    scene = bpy.context.scene
    previous_engine = scene.render.engine
    lod.hide_viewport = False
    lod.hide_set(False)
    bpy.ops.object.select_all(action="DESELECT")
    lod.select_set(True)
    bpy.context.view_layer.objects.active = lod
    material.node_tree.nodes.active = node
    result: dict[str, Any] = {
        "status": "failed",
        "source_object": lod.name,
        "target_object": lod.name,
        "size": size,
        "path": str(Path(image.filepath_raw).resolve()),
        "geometry_creation": False,
        "method": "cycles_ambient_occlusion_bake_on_final_lod0_uv",
    }
    try:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 16
        bake_result = bpy.ops.object.bake(
            type="AO",
            width=size,
            height=size,
            margin=16,
            use_selected_to_active=False,
            target="IMAGE_TEXTURES",
            save_mode="INTERNAL",
            use_clear=True,
        )
        if "FINISHED" not in bake_result:
            raise RuntimeError(f"AO bake did not finish: {bake_result}")
        image.save()
        image.pack()
        prepare_gltf_occlusion(material, node)
        result["status"] = "passed"
        result["sha256"] = sha256_file(Path(image.filepath_raw))
        result["packed"] = image.packed_file is not None
    finally:
        scene.render.engine = previous_engine
    return result


def prepare_normal_nodes(
    material: bpy.types.Material,
    output_dir: Path,
    *,
    size: int,
) -> tuple[bpy.types.Image, bpy.types.Node, bpy.types.Node]:
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    texture = nodes.get(NORMAL_NAME)
    if texture is not None and texture.type != "TEX_IMAGE":
        nodes.remove(texture)
        texture = None
    image = bpy.data.images.get(NORMAL_NAME)
    if image is not None and tuple(image.size) != (size, size):
        bpy.data.images.remove(image)
        image = None
    if image is None:
        image = bpy.data.images.new(NORMAL_NAME, width=size, height=size, alpha=False, float_buffer=False)
    image.generated_color = (0.5, 0.5, 1.0, 1.0)
    image.colorspace_settings.name = "Non-Color"
    image.file_format = "PNG"
    image.filepath_raw = str((output_dir / f"{NORMAL_NAME}.png").resolve())
    if texture is None:
        texture = nodes.new("ShaderNodeTexImage")
        texture.name = NORMAL_NAME
    texture.label = "Scan-derived clean-high to LOD0 tangent normal/detail"
    texture.image = image
    texture.location = (-620, -300)

    normal_node = nodes.get(NORMAL_MAP_NODE)
    if normal_node is not None and normal_node.type != "NORMAL_MAP":
        nodes.remove(normal_node)
        normal_node = None
    if normal_node is None:
        normal_node = nodes.new("ShaderNodeNormalMap")
        normal_node.name = NORMAL_MAP_NODE
    normal_node.label = "Scan-derived tangent normal/detail"
    normal_node.location = (-300, -280)
    normal_node.space = "TANGENT"
    normal_node.inputs["Strength"].default_value = 1.0

    shader = _principled(material)
    _remove_input_links(material, normal_node.inputs["Color"])
    _remove_input_links(material, shader.inputs["Normal"])
    # Keep the target normal image disconnected while baking to avoid a Cycles
    # circular-dependency warning. The material connection is restored only
    # after the selected-to-active bake completes successfully.
    nodes.active = texture
    return image, texture, normal_node


def bake_scan_normal(
    clean: bpy.types.Object,
    lod: bpy.types.Object,
    material: bpy.types.Material,
    output_dir: Path,
    *,
    size: int,
) -> dict[str, Any]:
    image, texture, normal_node = prepare_normal_nodes(material, output_dir, size=size)
    scene = bpy.context.scene
    previous_engine = scene.render.engine
    max_dimension = max(float(value) for value in lod.dimensions)
    cage_extrusion = max_dimension * 0.002
    max_ray_distance = max_dimension * 0.01

    clean_collection = bpy.data.collections.get("COL_V4_Work")
    if clean_collection is not None:
        clean_collection.hide_viewport = False
    clean.hide_viewport = False
    clean.hide_set(False)
    lod.hide_viewport = False
    lod.hide_set(False)
    bpy.ops.object.select_all(action="DESELECT")
    clean.select_set(True)
    lod.select_set(True)
    bpy.context.view_layer.objects.active = lod
    material.node_tree.nodes.active = texture

    result: dict[str, Any] = {
        "status": "failed",
        "source_object": clean.name,
        "target_object": lod.name,
        "space": "TANGENT",
        "size": size,
        "cage_extrusion_relative_to_lod_max_dimension": 0.002,
        "max_ray_distance_relative_to_lod_max_dimension": 0.01,
        "cage_extrusion": cage_extrusion,
        "max_ray_distance": max_ray_distance,
        "path": str(Path(image.filepath_raw).resolve()),
        "geometry_creation": False,
    }
    try:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 1
        bake_result = bpy.ops.object.bake(
            type="NORMAL",
            width=size,
            height=size,
            margin=16,
            use_selected_to_active=True,
            max_ray_distance=max_ray_distance,
            cage_extrusion=cage_extrusion,
            normal_space="TANGENT",
            target="IMAGE_TEXTURES",
            save_mode="INTERNAL",
            use_clear=True,
        )
        if "FINISHED" not in bake_result:
            raise RuntimeError(f"normal bake did not finish: {bake_result}")
        image.save()
        image.pack()
        material.node_tree.links.new(texture.outputs["Color"], normal_node.inputs["Color"])
        material.node_tree.links.new(normal_node.outputs["Normal"], _principled(material).inputs["Normal"])
        result["status"] = "passed"
        result["sha256"] = sha256_file(Path(image.filepath_raw))
        result["packed"] = image.packed_file is not None
    finally:
        scene.render.engine = previous_engine
    return result


def set_presentation_visibility(raw: bpy.types.Object, clean: bpy.types.Object, lod: bpy.types.Object) -> dict[str, Any]:
    for obj in (raw, clean):
        obj.hide_render = True
        obj.hide_viewport = True
        obj.hide_set(True)
        obj.select_set(False)
    lod.hide_render = False
    lod.hide_viewport = False
    lod.hide_set(False)

    for collection_name in ("COL_V4_Source", "COL_V4_Work"):
        collection = bpy.data.collections.get(collection_name)
        if collection is not None:
            collection.hide_render = True
            collection.hide_viewport = True
    final_collection = bpy.data.collections.get("COL_V4_Final")
    if final_collection is not None:
        final_collection.hide_render = False
        final_collection.hide_viewport = False

    bpy.ops.object.select_all(action="DESELECT")
    lod.select_set(True)
    bpy.context.view_layer.objects.active = lod
    return {
        "raw_hidden_viewport": raw.hide_viewport and raw.hide_get(),
        "raw_hidden_render": raw.hide_render,
        "clean_high_hidden_viewport": clean.hide_viewport and clean.hide_get(),
        "clean_high_hidden_render": clean.hide_render,
        "lod0_visible_viewport": not lod.hide_viewport and not lod.hide_get(),
        "lod0_visible_render": not lod.hide_render,
        "source_collection_hidden_viewport": bool(bpy.data.collections.get("COL_V4_Source") and bpy.data.collections["COL_V4_Source"].hide_viewport),
        "work_collection_hidden_viewport": bool(bpy.data.collections.get("COL_V4_Work") and bpy.data.collections["COL_V4_Work"].hide_viewport),
    }


def export_lod0(lod: bpy.types.Object, glb_path: Path) -> dict[str, Any]:
    bpy.ops.object.select_all(action="DESELECT")
    lod.hide_viewport = False
    lod.hide_set(False)
    lod.hide_render = False
    lod.select_set(True)
    bpy.context.view_layer.objects.active = lod
    result = bpy.ops.export_scene.gltf(
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
    if "FINISHED" not in result:
        raise RuntimeError(f"GLB export did not finish: {result}")
    return {"status": "passed", "path": str(glb_path.resolve()), "sha256": sha256_file(glb_path)}


def _evidence_record(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = read_json(resolved)
    return {
        "report": str(resolved),
        "report_sha256": sha256_file(resolved),
        "status": payload.get("status"),
        "decision": payload.get("decision"),
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    input_blend = args.input_blend.resolve()
    output_dir = args.output_dir.resolve()
    appearance_stats_path = args.appearance_stats.resolve()
    if not input_blend.is_file():
        raise FileNotFoundError(input_blend)
    if not appearance_stats_path.is_file():
        raise FileNotFoundError(appearance_stats_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty final Blender directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.open_mainfile(filepath=str(input_blend))
    raw, clean, lod, material = require_scene_objects()
    before = {name: geometry_signature(obj) for name, obj in ((RAW_NAME, raw), (CLEAN_NAME, clean), (LOD0_NAME, lod))}

    uv_before = uv_metrics(lod)
    uv_atlas = generate_box_atlas_uv(lod)
    appearance_result = prepare_project_appearance(material, output_dir, appearance_stats_path)
    ao_result = bake_scan_ao(lod, material, output_dir, size=args.ao_size)
    if ao_result["status"] != "passed":
        raise RuntimeError("scan-derived AO bake failed")
    appearance_result["ao_texture"] = {"path": ao_result["path"], "sha256": ao_result["sha256"]}
    normal_result = bake_scan_normal(clean, lod, material, output_dir, size=args.normal_size)
    if normal_result["status"] != "passed":
        raise RuntimeError("scan-derived normal/detail bake failed")
    uv_final = uv_metrics(lod)
    visibility = set_presentation_visibility(raw, clean, lod)
    if not all(visibility.values()):
        raise RuntimeError(f"final authoring visibility is ambiguous: {visibility}")

    after = {name: geometry_signature(obj) for name, obj in ((RAW_NAME, raw), (CLEAN_NAME, clean), (LOD0_NAME, lod))}
    if before != after:
        raise RuntimeError("finalization unexpectedly changed mesh geometry or transforms")

    required_images = (AO_NAME, BASE_COLOR_NAME, ROUGHNESS_NAME, NORMAL_NAME)
    for name in required_images:
        image = bpy.data.images.get(name)
        if image is None or image.packed_file is None:
            raise RuntimeError(f"required final packed image is missing or unpacked: {name}")

    blend_path = output_dir / args.blend_name
    glb_path = output_dir / args.glb_name
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    export_result = export_lod0(lod, glb_path)
    visibility = set_presentation_visibility(raw, clean, lod)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    upper_recovery = _evidence_record(args.upper_recovery_report)
    poisson_handoff = _evidence_record(args.poisson_handoff_report)
    packed_images = sorted(image.name for image in bpy.data.images if image.packed_file is not None)
    texture_files = {
        name: {
            "path": str((output_dir / f"{name}.png").resolve()),
            "sha256": sha256_file(output_dir / f"{name}.png"),
        }
        for name in required_images
    }
    shader = _principled(material)
    material_links = sorted(
        f"{link.from_node.name}.{link.from_socket.name}->{link.to_node.name}.{link.to_socket.name}"
        for link in material.node_tree.links
    )
    report = {
        "schema_version": 3,
        "status": "complete_with_documented_anatomy_limitation",
        "stage": "final_scan_preserving_blender_cleanup_uv_project_appearance_ao_detail_bake_export",
        "input_blend": str(input_blend),
        "input_blend_sha256": sha256_file(input_blend),
        "blend": str(blend_path.resolve()),
        "blend_sha256": sha256_file(blend_path),
        "glb": str(glb_path.resolve()),
        "glb_sha256": export_result["sha256"],
        "geometry_before": before,
        "geometry_after": after,
        "geometry_unchanged": before == after,
        "visibility": visibility,
        "duplicate_appearance_root_cause": "historical authoring viewport showed overlapping CleanHigh and LOD0; final source/work collections are hidden and only LOD0 is exported",
        "uv_before": uv_before,
        "uv_atlas": uv_atlas,
        "uv_final": uv_final,
        "appearance": appearance_result,
        "ao_bake": ao_result,
        "normal_detail_bake": normal_result,
        "textures": texture_files,
        "packed_images": packed_images,
        "required_packed_images_present": all(name in packed_images for name in required_images),
        "materials": [slot.material.name for slot in lod.material_slots if slot.material],
        "material_links": material_links,
        "principled_metallic": float(shader.inputs["Metallic"].default_value),
        "uv_layers": len(lod.data.uv_layers),
        "final_object": lod.name,
        "vertex_color_export": "NONE",
        "raw_preserved": RAW_NAME in bpy.data.objects,
        "clean_high_preserved": CLEAN_NAME in bpy.data.objects,
        "upper_recovery": upper_recovery,
        "poisson_handoff": poisson_handoff,
        "known_limitations": [
            "dense strict post-fusion anatomy gate remains authoritative",
            "finial/narrow-top anatomy is unresolved and is not manually repaired",
            "no_major_vessel_scale_holes remains false in the reconstruction evidence",
            "semantic support for all detached Poisson components is not proven",
            "photographic_projection_verified=false; BaseColor/Roughness use deterministic 158-image uncoated-project statistics instead",
            "normal/detail is baked only from the preserved scan-derived clean-high mesh",
        ],
    }
    report_path = output_dir / "blender_final_completion_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report"] = str(report_path.resolve())
    report["report_sha256"] = sha256_file(report_path)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-blend", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--appearance-stats", type=Path, required=True)
    parser.add_argument("--ao-size", type=int, default=2048)
    parser.add_argument("--normal-size", type=int, default=2048)
    parser.add_argument("--upper-recovery-report", type=Path)
    parser.add_argument("--poisson-handoff-report", type=Path, required=True)
    parser.add_argument("--blend-name", default="Thai_Libation_Vessel_V4_BEST_DEFENSIBLE_TRIM5_FINAL.blend")
    parser.add_argument("--glb-name", default="Thai_Libation_Vessel_V4_BEST_DEFENSIBLE_TRIM5_FINAL.glb")
    if argv is not None:
        return parser.parse_args(list(argv))
    raw = list(sys.argv)
    raw = raw[raw.index("--") + 1 :] if "--" in raw else raw[1:]
    return parser.parse_args(raw)


def main(argv: Iterable[str] | None = None) -> int:
    report = finalize(parse_args(argv))
    print(json.dumps({
        "status": report["status"],
        "blend_sha256": report["blend_sha256"],
        "glb_sha256": report["glb_sha256"],
        "geometry_unchanged": report["geometry_unchanged"],
        "normal_detail_bake": report["normal_detail_bake"],
        "appearance": report["appearance"],
        "visibility": report["visibility"],
        "report_sha256": report["report_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
