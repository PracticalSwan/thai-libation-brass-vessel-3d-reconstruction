"""Build the complete reference-assisted Thai libation vessel in Blender.

Run with Blender, not regular Python:
    blender --background --python build_reference_model_blender.py -- <report.json> <output_dir>

The script creates new presentation geometry from measured reference profiles.  It
never imports, edits, or overwrites the Steps 14-17 Poisson reconstruction.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector


def _args() -> tuple[Path, Path]:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("expected -- <reference_report.json> <output_dir>")
    values = argv[argv.index("--") + 1 :]
    if len(values) != 2:
        raise SystemExit("expected exactly two script arguments")
    return Path(values[0]).resolve(), Path(values[1]).resolve()


def srgb_to_linear(value: float) -> float:
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def ensure_collection(name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def move_to_collection(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    for current in tuple(obj.users_collection):
        current.objects.unlink(obj)
    collection.objects.link(obj)


def make_brass_material(name: str, srgb: list[float], roughness: float, relief: float = 0.12) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    texcoord = nodes.new("ShaderNodeTexCoord")
    noise_color = nodes.new("ShaderNodeTexNoise")
    noise_color.inputs["Scale"].default_value = 3.5
    noise_color.inputs["Detail"].default_value = 4.0
    noise_color.inputs["Roughness"].default_value = 0.65
    ramp = nodes.new("ShaderNodeValToRGB")
    base = [srgb_to_linear(float(v)) for v in srgb]
    dark = [max(0.0, channel * 0.48) for channel in base]
    bright = [min(1.0, channel * 1.45 + 0.025) for channel in base]
    ramp.color_ramp.elements[0].position = 0.20
    ramp.color_ramp.elements[0].color = (*dark, 1.0)
    ramp.color_ramp.elements[1].position = 0.82
    ramp.color_ramp.elements[1].color = (*bright, 1.0)

    noise_bump = nodes.new("ShaderNodeTexNoise")
    noise_bump.inputs["Scale"].default_value = 38.0
    noise_bump.inputs["Detail"].default_value = 3.0
    noise_bump.inputs["Roughness"].default_value = 0.72
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = relief
    bump.inputs["Distance"].default_value = 0.018

    principled.inputs["Metallic"].default_value = 1.0
    principled.inputs["Roughness"].default_value = roughness
    if "Coat Weight" in principled.inputs:
        principled.inputs["Coat Weight"].default_value = 0.18
    if "Coat Roughness" in principled.inputs:
        principled.inputs["Coat Roughness"].default_value = 0.12

    links.new(texcoord.outputs["Generated"], noise_color.inputs["Vector"])
    links.new(noise_color.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], principled.inputs["Base Color"])
    links.new(texcoord.outputs["Generated"], noise_bump.inputs["Vector"])
    links.new(noise_bump.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], principled.inputs["Normal"])
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return material


def make_dark_material() -> bpy.types.Material:
    material = bpy.data.materials.new("Dark_Recess")
    material.use_nodes = True
    p = material.node_tree.nodes.get("Principled BSDF")
    p.inputs["Base Color"].default_value = (0.018, 0.012, 0.006, 1.0)
    p.inputs["Metallic"].default_value = 0.75
    p.inputs["Roughness"].default_value = 0.34
    return material


def lathe_object(name: str, profile: list[list[float]], material: bpy.types.Material, collection: bpy.types.Collection, segments: int = 192) -> bpy.types.Object:
    """Revolve one closed r/z polygon around Z with explicit shared axis vertices."""
    mesh = bpy.data.meshes.new(name + "_Mesh")
    vertices: list[tuple[float, float, float]] = []
    rings: list[list[int]] = []
    for radius, z in profile:
        radius = float(radius)
        z = float(z)
        if abs(radius) < 1e-9:
            rings.append([len(vertices)])
            vertices.append((0.0, 0.0, z))
        else:
            ring = []
            for segment in range(segments):
                angle = 2.0 * math.pi * segment / segments
                ring.append(len(vertices))
                vertices.append((radius * math.cos(angle), radius * math.sin(angle), z))
            rings.append(ring)

    faces: list[tuple[int, ...]] = []
    for first, second in zip(rings, rings[1:] + rings[:1]):
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
    obj.data.materials.append(material)
    bevel = obj.modifiers.new("Micro_Bevel", "BEVEL")
    bevel.width = 0.006
    bevel.segments = 2
    return obj


def add_torus(name: str, major_radius: float, minor_radius: float, z: float, material: bpy.types.Material, collection: bpy.types.Collection) -> bpy.types.Object:
    bpy.ops.mesh.primitive_torus_add(major_radius=major_radius, minor_radius=minor_radius, major_segments=128, minor_segments=16, location=(0, 0, z))
    obj = bpy.context.object
    obj.name = name
    move_to_collection(obj, collection)
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def add_relief_ring(prefix: str, radius: float, z: float, count: int, height: float, width: float, depth: float, material: bpy.types.Material, collection: bpy.types.Collection, angle_offset: float = 0.0) -> list[bpy.types.Object]:
    objects = []
    for index in range(count):
        angle = angle_offset + 2.0 * math.pi * index / count
        bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, location=(radius * math.cos(angle), radius * math.sin(angle), z))
        obj = bpy.context.object
        obj.name = f"{prefix}_{index + 1:02d}"
        move_to_collection(obj, collection)
        # Local X points radially; local Y is tangent and Z stays vertical.
        obj.rotation_euler[2] = angle
        obj.scale = (depth, width, height)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        obj.data.materials.append(material)
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        objects.append(obj)
    return objects


def add_diamond_relief_ring(prefix: str, radius: float, z: float, count: int, height: float, width: float, material: bpy.types.Material, collection: bpy.types.Collection, angle_offset: float = 0.0) -> list[bpy.types.Object]:
    """Add thin closed diamond-line motifs on tangent planes around the form."""
    objects = []
    for index in range(count):
        angle = angle_offset + 2.0 * math.pi * index / count
        curve = bpy.data.curves.new(f"{prefix}_{index + 1:02d}_Curve", "CURVE")
        curve.dimensions = "3D"
        curve.bevel_depth = 0.0065
        curve.bevel_resolution = 3
        curve.use_fill_caps = True
        spline = curve.splines.new("POLY")
        spline.points.add(3)
        local_points = [(0.0, 0.0, height), (0.0, width, 0.0), (0.0, 0.0, -height), (0.0, -width, 0.0)]
        for point, coordinate in zip(spline.points, local_points):
            point.co = (*coordinate, 1.0)
        spline.use_cyclic_u = True
        obj = bpy.data.objects.new(f"{prefix}_{index + 1:02d}", curve)
        collection.objects.link(obj)
        obj.location = (radius * math.cos(angle), radius * math.sin(angle), z)
        obj.rotation_euler[2] = angle
        obj.data.materials.append(material)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.convert(target="MESH")
        obj.select_set(False)
        objects.append(obj)
    return objects


def add_chain(material: bpy.types.Material, collection: bpy.types.Collection) -> bpy.types.Object:
    """Build an inferred hanging chain from closed torus links so export stays manifold."""
    parent = bpy.data.objects.new("Reference_Chain", None)
    collection.objects.link(parent)
    controls = [
        Vector((0.31, -0.04, 2.73)),
        Vector((0.62, -0.15, 2.42)),
        Vector((0.78, -0.20, 1.92)),
        Vector((0.92, -0.12, 1.31)),
        Vector((0.87, 0.02, 0.84)),
    ]
    link_count = 48
    for index in range(link_count):
        t = index / (link_count - 1)
        scaled = t * (len(controls) - 1)
        segment = min(int(math.floor(scaled)), len(controls) - 2)
        local_t = scaled - segment
        location = controls[segment].lerp(controls[segment + 1], local_t)
        bpy.ops.mesh.primitive_torus_add(
            major_radius=0.026,
            minor_radius=0.006,
            major_segments=24,
            minor_segments=8,
            location=location,
        )
        link = bpy.context.object
        link.name = f"Chain_Link_{index + 1:02d}"
        move_to_collection(link, collection)
        link.parent = parent
        link.rotation_euler[0] = math.pi / 2.0
        if index % 2:
            link.rotation_euler[2] = math.pi / 2.0
        link.data.materials.append(material)
        for polygon in link.data.polygons:
            polygon.use_smooth = True
    return parent


def add_model(report: dict, collection: bpy.types.Collection) -> dict[str, bpy.types.Object]:
    measured = report["measurements"]
    base_color = list(measured["photo_median_brass_srgb"])
    roughness = float(measured["roughness_estimate"])
    brass = make_brass_material("Photo_Referenced_Brass", base_color, roughness, relief=0.11)
    relief_brass = make_brass_material("Photo_Referenced_Relief_Brass", base_color, min(0.34, roughness + 0.05), relief=0.18)
    dark = make_dark_material()
    profiles = report["profiles"]
    radial_scale = float(profiles["radial_scale"])

    bowl = lathe_object("Receiving_Bowl_Pedestal", profiles["bowl"], brass, collection)
    vessel = lathe_object("Water_Vessel", profiles["vessel"], brass, collection)
    lid = lathe_object("Lid", profiles["lid"], relief_brass, collection)

    # Dark cavity disk just below the vessel makes the intentionally separate bowl readable.
    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=0.38 * radial_scale, depth=0.012, location=(0, 0, 0.705))
    cavity = bpy.context.object
    cavity.name = "Bowl_Cavity_Shadow"
    move_to_collection(cavity, collection)
    cavity.data.materials.append(dark)

    # Foot, pedestal, bowl-rim, shoulder, neck, and lid rings follow the photographed construction.
    ring_specs = [
        ("Foot_Lower", 0.48, 0.018, 0.035), ("Foot_Upper", 0.55, 0.022, 0.16),
        ("Pedestal_Band", 0.47, 0.018, 0.34), ("Bowl_Lower_Band", 0.61, 0.016, 0.51),
        ("Bowl_Rim_Inner", 0.82, 0.025, 0.86), ("Bowl_Rim_Outer", 0.90, 0.022, 0.88),
        ("Globe_Lower", 0.58, 0.015, 0.82), ("Globe_Upper", 0.52, 0.014, 1.70),
        ("Neck_Base", 0.35, 0.018, 1.90), ("Neck_Band_1", 0.32, 0.014, 2.06),
        ("Neck_Band_2", 0.32, 0.014, 2.30), ("Neck_Band_3", 0.33, 0.018, 2.50),
        ("Lid_Band_1", 0.39, 0.018, 2.62), ("Lid_Band_2", 0.35, 0.017, 2.75),
        ("Lid_Band_3", 0.32, 0.016, 2.86),
    ]
    for name, radius, minor, z in ring_specs:
        add_torus(name, radius * radial_scale, minor, z, relief_brass, collection)

    # Finial: photographed rounded top with a small stem and stepped collar.
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=0.135 * radial_scale, location=(0, 0, 3.055))
    finial = bpy.context.object
    finial.name = "Finial"
    move_to_collection(finial, collection)
    finial.scale.z = 1.16
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    finial.data.materials.append(relief_brass)
    add_torus("Finial_Collar", 0.14 * radial_scale, 0.018, 2.965, relief_brass, collection)

    # Estimated repeated relief.  Placement and density follow visible motifs; unseen repetitions are explicit inference.
    add_relief_ring("Globe_Lotus_A", 0.755 * radial_scale, 1.18, 12, 0.115, 0.060, 0.030, relief_brass, collection)
    add_relief_ring("Globe_Lotus_B", 0.735 * radial_scale, 1.43, 12, 0.105, 0.055, 0.028, relief_brass, collection, math.pi / 12.0)
    add_diamond_relief_ring("Globe_Diamond", 0.774 * radial_scale, 1.31, 8, 0.105, 0.060, relief_brass, collection, math.pi / 8.0)
    add_relief_ring("Bowl_Floral", 0.825 * radial_scale, 0.66, 18, 0.055, 0.052, 0.020, relief_brass, collection, math.pi / 18.0)
    add_diamond_relief_ring("Bowl_Diamond", 0.805 * radial_scale, 0.67, 9, 0.048, 0.040, relief_brass, collection)
    add_relief_ring("Neck_Motif", 0.315 * radial_scale, 2.22, 8, 0.12, 0.045, 0.018, relief_brass, collection, math.pi / 8.0)
    chain = add_chain(relief_brass, collection)

    return {"bowl": bowl, "vessel": vessel, "lid": lid, "finial": finial, "chain": chain, "cavity": cavity}


def object_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    points = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    low = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    high = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return low, high


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_render_scene(model_objects: list[bpy.types.Object]) -> tuple[bpy.types.Object, list[bpy.types.Object]]:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 900
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    try:
        scene.view_settings.view_transform = "AgX"
    except Exception:
        pass

    low, high = object_bounds(model_objects)
    center = (low + high) * 0.5
    height = high.z - low.z

    bpy.ops.object.camera_add(location=(4.8, -7.4, center.z + 0.35))
    camera = bpy.context.object
    camera.name = "Reference_Presentation_Camera"
    camera.data.lens = 58
    camera.data.sensor_width = 36
    look_at(camera, center + Vector((0, 0, 0.15)))
    scene.camera = camera

    lights = []
    for name, location, energy, size, color in [
        ("Key", (4.5, -4.5, 5.4), 1050, 4.0, (1.0, 0.80, 0.58)),
        ("Fill", (-4.0, -2.2, 3.2), 620, 5.0, (0.72, 0.82, 1.0)),
        ("Rim", (1.0, 4.5, 4.8), 900, 3.0, (1.0, 0.58, 0.26)),
        ("Top", (0.0, 0.0, 7.0), 520, 3.5, (1.0, 0.92, 0.78)),
    ]:
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        data.color = color
        light = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(light)
        light.location = location
        look_at(light, center)
        lights.append(light)

    bpy.ops.mesh.primitive_plane_add(size=14, location=(0, 0, low.z - 0.012))
    floor = bpy.context.object
    floor.name = "Presentation_Floor"
    material = bpy.data.materials.new("Presentation_Floor_Material")
    material.use_nodes = True
    p = material.node_tree.nodes.get("Principled BSDF")
    p.inputs["Base Color"].default_value = (0.035, 0.04, 0.05, 1.0)
    p.inputs["Roughness"].default_value = 0.32
    floor.data.materials.append(material)

    world = bpy.data.worlds.new("Reference_World") if bpy.context.scene.world is None else bpy.context.scene.world
    bpy.context.scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.018, 0.022, 0.03, 1.0)
    background.inputs["Strength"].default_value = 0.34
    return camera, lights + [floor]


def render_views(output_dir: Path, camera: bpy.types.Object, helpers: list[bpy.types.Object], model_objects: list[bpy.types.Object]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    low, high = object_bounds(model_objects)
    center = (low + high) * 0.5
    views = {
        "front": (0.0, -7.8, center.z + 0.15),
        "quarter": (5.1, -6.2, center.z + 0.35),
        "side": (7.8, 0.0, center.z + 0.15),
        "top_oblique": (4.7, -5.4, center.z + 3.2),
    }
    rendered = {}
    for name, location in views.items():
        camera.location = location
        look_at(camera, center + Vector((0, 0, 0.10)))
        path = output_dir / f"reference_{name}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        rendered[name] = str(path)

    # Transparent front silhouette compares only the primary measured forms. Decorative
    # relief/rings and the chain are intentionally excluded because the reviewed masks
    # measure the vessel-set envelope rather than inferred ornament protrusions.
    floor = next((obj for obj in helpers if obj.name == "Presentation_Floor"), None)
    if floor is not None:
        floor.hide_render = True
    primary_names = {"Receiving_Bowl_Pedestal", "Water_Vessel", "Lid", "Finial", "Bowl_Cavity_Shadow"}
    hidden_for_silhouette = []
    for obj in model_objects:
        if obj.name not in primary_names:
            hidden_for_silhouette.append((obj, obj.hide_render))
            obj.hide_render = True
    scene.render.film_transparent = True
    camera.location = (0.0, -8.0, center.z)
    look_at(camera, center)
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = (high.z - low.z) * 1.10
    silhouette = output_dir / "reference_render_silhouette.png"
    scene.render.filepath = str(silhouette)
    bpy.ops.render.render(write_still=True)
    rendered["silhouette"] = str(silhouette)
    camera.data.type = "PERSP"
    scene.render.film_transparent = False
    for obj, previous in hidden_for_silhouette:
        obj.hide_render = previous
    if floor is not None:
        floor.hide_render = False
    return rendered


def mesh_stats(collection: bpy.types.Collection) -> dict:
    objects = list(collection.all_objects)
    meshes = [obj for obj in objects if obj.type == "MESH"]
    vertices = sum(len(obj.data.vertices) for obj in meshes)
    polygons = sum(len(obj.data.polygons) for obj in meshes)
    nonmanifold = {}
    for obj in meshes:
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bad = sum(1 for edge in bm.edges if len(edge.link_faces) != 2)
        bm.free()
        if bad:
            nonmanifold[obj.name] = bad
    return {
        "object_count": len(objects),
        "mesh_object_count": len(meshes),
        "vertex_count": vertices,
        "polygon_count": polygons,
        "nonmanifold_edges_by_object": nonmanifold,
    }


def export_assets(output_dir: Path, collection: bpy.types.Collection) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    blend_path = output_dir / "Thai_Libation_Vessel_REFERENCE_ASSISTED_PRE_CLEAN.blend"
    glb_path = output_dir / "Thai_Libation_Vessel_REFERENCE_ASSISTED_PRE_CLEAN.glb"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    bpy.ops.object.select_all(action="DESELECT")
    for obj in collection.all_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = next((obj for obj in collection.all_objects if obj.type == "MESH"), None)
    bpy.ops.export_scene.gltf(filepath=str(glb_path), export_format="GLB", use_selection=True, export_apply=True)
    return {"blend": str(blend_path), "glb": str(glb_path)}


def main() -> None:
    report_path, output_dir = _args()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("manual_cleanup_started") is not False:
        raise ValueError("reference report does not preserve the pre-clean boundary")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    model_collection = ensure_collection("REFERENCE_ASSISTED_MODEL_PRE_CLEAN")
    primary = add_model(report, model_collection)
    model_objects = list(model_collection.all_objects)
    camera, helpers = setup_render_scene(model_objects)
    rendered = render_views(output_dir / "previews", camera, helpers, model_objects)
    stats = mesh_stats(model_collection)
    exports = export_assets(output_dir, model_collection)

    build_report = {
        "method": "reference_assisted_rotational_model_from_reviewed_masks_and_photos",
        "manual_cleanup_started": False,
        "geometry_source": "new parametric geometry; Steps 14-17 Poisson mesh not imported or modified",
        "major_components": sorted(primary),
        "stats": stats,
        "renders": rendered,
        "exports": exports,
        "inference_disclosure": [
            "rotational symmetry is used for unseen sides of bowl, globe, neck, lid, and finial",
            "repeated relief motifs on unseen sides are estimated from visible repeated ornament",
            "photo-derived brass color is combined with procedural metallic roughness/bump rather than baking specular highlights into albedo",
        ],
    }
    (output_dir / "build_report.json").write_text(json.dumps(build_report, indent=2), encoding="utf-8")
    print(json.dumps(build_report, indent=2))


if __name__ == "__main__":
    main()
