"""Fresh factory-empty Blender GLB import and eight-view verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import bpy
from mathutils import Vector


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    minimum = Vector((min(point.x for point in points), min(point.y for point in points), min(point.z for point in points)))
    maximum = Vector((max(point.x for point in points), max(point.y for point in points), max(point.z for point in points)))
    return minimum, maximum


def material_summary(material: bpy.types.Material) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    for node in material.node_tree.nodes if material.use_nodes else []:
        if node.type != "TEX_IMAGE" or node.image is None:
            continue
        image = node.image
        images.append(
            {
                "name": image.name,
                "filepath": str(image.filepath),
                "packed": image.packed_file is not None,
                "resolved": image.packed_file is not None or (bool(image.filepath) and Path(bpy.path.abspath(image.filepath)).is_file()),
            }
        )
    return {"name": material.name, "images": images}


def render_eight_views(obj: bpy.types.Object, output_dir: Path) -> dict[str, str]:
    scene = bpy.context.scene
    minimum, maximum = bounds(obj)
    center = (minimum + maximum) / 2.0
    dimensions = maximum - minimum
    radius = max(dimensions) * 1.8
    camera_data = bpy.data.cameras.new("CAM_GL B_REIMPORT_QA")
    camera = bpy.data.objects.new("CAM_GLB_REIMPORT_QA", camera_data)
    scene.collection.objects.link(camera)
    camera_data.lens = 58.0
    camera_data.sensor_width = 36.0
    scene.camera = camera
    for name, location, energy, size in (
        ("LGT_GLB_KEY", (2.5, -3.0, 3.5), 1300.0, 3.0),
        ("LGT_GLB_FILL", (-2.0, 1.5, 2.2), 850.0, 4.0),
        ("LGT_GLB_RIM", (0.5, 2.5, 3.0), 1100.0, 2.5),
    ):
        data = bpy.data.lights.new(name, type="AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        light = bpy.data.objects.new(name, data)
        light.location = location
        look_at(light, center)
        scene.collection.objects.link(light)
    if scene.world is None:
        scene.world = bpy.data.worlds.new("WORLD_GLB_REIMPORT_QA")
    scene.world.color = (0.035, 0.035, 0.035)
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 700
    scene.render.resolution_y = 700
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for index, degrees in enumerate(range(0, 360, 45)):
        radians = math.radians(float(degrees))
        elevation = math.radians(10.0 if index % 2 == 0 else 16.0)
        camera.location = Vector((radius * math.cos(radians) * math.cos(elevation), radius * math.sin(radians) * math.cos(elevation), center.z + radius * math.sin(elevation)))
        look_at(camera, center)
        path = (output_dir / f"view_{index:02d}_{degrees:03d}.png").resolve()
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        paths[path.name] = sha256_file(path)
    return paths


def verify(args: argparse.Namespace) -> dict[str, Any]:
    glb = args.glb.resolve()
    output_dir = args.output_dir.resolve()
    if not glb.is_file():
        raise FileNotFoundError(glb)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    result = bpy.ops.import_scene.gltf(filepath=str(glb))
    if "FINISHED" not in result:
        raise RuntimeError(f"GLB import did not finish: {result}")
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    minimum: Vector | None = None
    maximum: Vector | None = None
    object_rows: list[dict[str, Any]] = []
    for obj in meshes:
        obj_min, obj_max = bounds(obj)
        minimum = obj_min if minimum is None else Vector((min(minimum[i], obj_min[i]) for i in range(3)))
        maximum = obj_max if maximum is None else Vector((max(maximum[i], obj_max[i]) for i in range(3)))
        mesh = obj.data
        normals_ok = all(all(math.isfinite(float(value)) for value in loop.normal) for loop in mesh.loops)
        positions_ok = all(all(math.isfinite(float(value)) for value in vertex.co) for vertex in mesh.vertices)
        materials = [material_summary(slot.material) for slot in obj.material_slots if slot.material]
        object_rows.append(
            {
                "name": obj.name,
                "type": obj.type,
                "verts": len(mesh.vertices),
                "faces": len(mesh.polygons),
                "uv_layers": len(mesh.uv_layers),
                "finite_positions": positions_ok,
                "finite_normals": normals_ok,
                "materials": materials,
                "bounds_world": [float(value) for value in (obj_max - obj_min)],
            }
        )
    materials = [material for row in object_rows for material in row["materials"]]
    texture_rows = [image for material in materials for image in material["images"]]
    dims = (maximum - minimum) if minimum is not None and maximum is not None else Vector((0.0, 0.0, 0.0))
    only_final = len(meshes) == 1 and meshes[0].name == args.expected_object
    report = {
        "schema_version": 1,
        "status": "passed" if meshes and only_final else "failed",
        "stage": "fresh_factory_empty_glb_reimport",
        "glb": str(glb),
        "glb_sha256": sha256_file(glb),
        "import_operator": str(result),
        "mesh_object_count": len(meshes),
        "mesh_name": meshes[0].name if len(meshes) == 1 else None,
        "objects": object_rows,
        "world_bounds": [[float(value) for value in minimum], [float(value) for value in maximum]] if minimum is not None and maximum is not None else [],
        "world_dims": [float(value) for value in dims],
        "finite_positions": all(row["finite_positions"] for row in object_rows),
        "finite_normals": all(row["finite_normals"] for row in object_rows),
        "materials": materials,
        "texture_refs": texture_rows,
        "unexpected_debug_objects": sorted(obj.name for obj in bpy.context.scene.objects if obj.type in {"CAMERA", "LIGHT"}),
        "checks": {
            "mesh_nonzero": len(meshes) == 1 and len(meshes[0].data.vertices) > 0 and len(meshes[0].data.polygons) > 0 if meshes else False,
            "only_final_exported": only_final,
            "material_present": any(material["name"] == args.expected_material for material in materials),
            "textures_resolve": bool(texture_rows) and all(item["resolved"] for item in texture_rows),
            "sane_bounds": all(math.isfinite(float(value)) and float(value) > 0.0 for value in dims),
            "normals_ok": all(row["finite_normals"] for row in object_rows),
            "uvs_present": all(row["uv_layers"] > 0 for row in object_rows),
        },
    }
    report["checks"]["passed"] = bool(all(report["checks"].values()))
    report["renders"] = render_eight_views(meshes[0], output_dir / "renders") if meshes else {}
    report["status"] = "passed" if report["checks"]["passed"] else "failed"
    report_path = output_dir / "glb_reimport_verification.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["checks"]["passed"]:
        raise RuntimeError("fresh GLB re-import verification failed: " + json.dumps(report["checks"], sort_keys=True))
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-object", default="SM_V4_Vessel_LOD0")
    parser.add_argument("--expected-material", default="MAT_V4_Brass")
    if argv is not None:
        return parser.parse_args(list(argv))
    raw = list(sys.argv)
    if "--" in raw:
        raw = raw[raw.index("--") + 1 :]
    else:
        raw = raw[1:]
    return parser.parse_args(raw)


def main(argv: Iterable[str] | None = None) -> int:
    report = verify(parse_args(argv))
    print(json.dumps({"status": report["status"], "glb_sha256": report["glb_sha256"], "checks": report["checks"], "render_count": len(report["renders"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
