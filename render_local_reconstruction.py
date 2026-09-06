"""Headless Blender validator for a real COLMAP photo-textured local mesh.

The module-level helpers are standard-Python testable. Blender is imported only
inside ``main`` so unit tests never start Blender or change the current scene.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def camera_matrix_world(cam_from_world):
    """Convert COLMAP/OpenCV world-to-camera coordinates to Blender pose."""
    pose = np.asarray(cam_from_world, dtype=np.float64)
    if pose.shape == (3, 4):
        homogeneous = np.eye(4, dtype=np.float64)
        homogeneous[:3, :] = pose
        pose = homogeneous
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("camera pose must be a finite 3x4 or 4x4 matrix")
    cv_to_blender = np.diag([1., -1., -1., 1.])
    return np.linalg.inv(cv_to_blender @ pose)


def load_geometry_contract(path: Path):
    try:
        archive = np.load(path, allow_pickle=False)
        xyz = np.asarray(archive["xyz"], dtype=np.float64)
        faces = np.asarray(archive["faces"], dtype=np.int64)
        uvs = np.asarray(archive["uvs"], dtype=np.float64) if "uvs" in archive.files else None
    except Exception as error:
        raise ValueError("Unable to load geometry contract") from error
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("geometry coordinates must be finite Nx3")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("geometry faces must be triangular Mx3")
    if np.any(faces < 0) or np.any(faces >= len(xyz)):
        raise ValueError("geometry face index outside vertex range")
    if uvs is not None and uvs.size:
        if uvs.shape != (len(faces), 3, 2) or not np.isfinite(uvs).all():
            raise ValueError("geometry UVs must be finite Mx3x2")
    else:
        uvs = None
    return dict(xyz=xyz, faces=faces.astype(np.int32), uvs=uvs)


def load_camera_contract(path: Path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError("Unable to load camera contract") from error
    if not isinstance(payload, list) or not payload:
        raise ValueError("camera contract must contain calibrated cameras")
    cameras = []
    for camera in payload:
        try:
            name = str(camera["name"])
            width, height = int(camera["width"]), int(camera["height"])
            params = np.asarray(camera["params"], dtype=np.float64)
            pose = np.asarray(camera["cam_from_world"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("camera contract is missing calibrated camera fields") from error
        if not name or width <= 0 or height <= 0 or params.size < 4 or pose.shape != (3, 4):
            raise ValueError("camera contract has invalid camera calibration")
        if not np.isfinite(params).all() or not np.isfinite(pose).all():
            raise ValueError("camera calibration is non-finite")
        cameras.append(dict(name=name, width=width, height=height,
                            params=params.tolist(), cam_from_world=pose.tolist()))
    return cameras


def _sha256(path: Path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--texture", type=Path, required=True)
    parser.add_argument("--save-blend", type=Path)
    return parser.parse_args(argv)


def _make_camera(bpy, mathutils, camera, name, render_width):
    data = bpy.data.cameras.new(name + "_data")
    data.type = "PERSP"
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 36.0
    data.lens = float(camera["params"][0]) * data.sensor_width / camera["width"]
    data.shift_x = -(float(camera["params"][2]) - camera["width"] / 2.0) / camera["width"]
    data.shift_y = (float(camera["params"][3]) - camera["height"] / 2.0) / camera["height"]
    data.clip_start = 0.001
    data.clip_end = 100000.0
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.matrix_world = mathutils.Matrix(camera_matrix_world(camera["cam_from_world"]).tolist())
    return obj


def _make_material(bpy, texture_path):
    material = bpy.data.materials.new("COLMAP_Photo_Texture")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    image_node = nodes.new("ShaderNodeTexImage")
    image = bpy.data.images.load(str(texture_path), check_existing=False)
    image_node.image = image
    shader.inputs["Roughness"].default_value = 0.38
    shader.inputs["Metallic"].default_value = 0.20
    links.new(image_node.outputs["Color"], shader.inputs["Base Color"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material, image


def _render_triptych(bpy, np, scene, cameras, output, render_width=640):
    paths = []
    selected = [cameras[0], cameras[len(cameras) // 2], cameras[-1]]
    for index, camera in enumerate(selected):
        scene.camera = camera["object"]
        scene.render.resolution_x = render_width
        scene.render.resolution_y = max(1, round(render_width * camera["height"] / camera["width"]))
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        path = output.with_name(output.stem + f"_view{index + 1}.png")
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        paths.append(path)
    images = [bpy.data.images.load(str(path), check_existing=False) for path in paths]
    width = max(int(image.size[0]) for image in images)
    height = max(int(image.size[1]) for image in images)
    combined = bpy.data.images.new("photo_textured_triptych", width=width * len(images), height=height, alpha=False)
    pixels = np.zeros((height, width * len(images), 4), dtype=np.float32)
    pixels[:, :, 3] = 1.0
    for index, image in enumerate(images):
        array = np.asarray(image.pixels[:], dtype=np.float32).reshape((int(image.size[1]), int(image.size[0]), 4))
        h, w = array.shape[:2]
        pixels[:h, index * width:index * width + w] = array
    combined.pixels = pixels.reshape(-1).tolist()
    combined.filepath_raw = str(output)
    combined.file_format = "PNG"
    combined.save()
    non_background = [float(np.mean(np.any(np.abs(pixels[:, i * width:(i + 1) * width, :3] - .04) > .03, axis=2)))
                      for i in range(len(images))]
    for image in images:
        bpy.data.images.remove(image)
    for path in paths:
        path.unlink()
    return len(paths), non_background


def main():
    args = _parse_args()
    geometry = load_geometry_contract(args.geometry)
    cameras = load_camera_contract(args.cameras)
    if not args.texture.is_file():
        raise ValueError("photo texture image is missing")
    import bpy  # type: ignore
    import mathutils  # type: ignore

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mesh = bpy.data.meshes.new("COLMAP_Photo_Textured_Mesh")
    mesh.from_pydata(geometry["xyz"].tolist(), [], geometry["faces"].tolist())
    mesh.update()
    object_ = bpy.data.objects.new("COLMAP_Photo_Textured_Local_73_View", mesh)
    bpy.context.scene.collection.objects.link(object_)
    material, image = _make_material(bpy, args.texture)
    object_.data.materials.append(material)
    if geometry["uvs"] is not None:
        layer = mesh.uv_layers.new(name="COLMAP_Photo_UV")
        for polygon in mesh.polygons:
            for loop_index, uv in zip(polygon.loop_indices, geometry["uvs"][polygon.index]):
                layer.data[loop_index].uv = (float(uv[0]), float(uv[1]))
    cameras_bpy = []
    for camera in cameras:
        camera = dict(camera)
        camera["object"] = _make_camera(bpy, mathutils, camera, camera["name"], 640)
        cameras_bpy.append(camera)
    low = geometry["xyz"].min(axis=0)
    high = geometry["xyz"].max(axis=0)
    center = (low + high) / 2.0
    radius = max(float(np.max(high - low)), 1.0)
    # Two neutral fill lights only make the imported photo texture readable;
    # they do not alter geometry or substitute a procedural material.
    for name, location, energy, size in (("Validation_Key", center + [radius, -radius, radius], 700, radius),
                                          ("Validation_Fill", center + [-radius, radius, radius], 350, radius)):
        light_data = bpy.data.lights.new(name, "AREA")
        light_data.energy = energy
        light_data.shape = "DISK"
        light_data.size = size
        light = bpy.data.objects.new(name, light_data)
        bpy.context.scene.collection.objects.link(light)
        light.location = location
        direction = mathutils.Vector(center) - light.location
        light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = False
    if scene.world is None:
        scene.world = bpy.data.worlds.new("Validation_World")
    scene.world.color = (0.04, 0.04, 0.04)
    rendered_view_count, non_background = _render_triptych(bpy, np, scene, cameras_bpy, args.output)
    if args.save_blend:
        args.save_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(args.save_blend))
    report = dict(status="completed", vertex_count=len(mesh.vertices), face_count=len(mesh.polygons),
                  uv_layer_count=len(mesh.uv_layers), uv_loop_count=len(mesh.uv_layers[0].data) if mesh.uv_layers else 0,
                  material_name=material.name, texture_path=str(args.texture), texture_sha256=_sha256(args.texture),
                  texture_dimensions=[int(image.size[0]), int(image.size[1])],
                  rendered_view_count=rendered_view_count, temporary_view_files_removed=True,
                  non_background_fraction=non_background, saved_blend=str(args.save_blend) if args.save_blend else None)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
