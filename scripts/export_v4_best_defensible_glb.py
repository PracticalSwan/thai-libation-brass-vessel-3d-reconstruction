"""Export the already-built fresh V4 Blender authoring scene to GLB.

This continuation exists so a GLB operator/API failure cannot force a second
import/decimation pass.  It opens only the versioned best-defensible blend,
selects its LOD0 object, exports that object and its packed material texture,
and writes a hash-bound authoring report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import bpy


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dense-selection", type=Path, required=True)
    parser.add_argument("--raw-diagnostic", type=Path, required=True)
    parser.add_argument("--appearance-stats", type=Path, required=True)
    parser.add_argument("--expected-object", default="SM_V4_Vessel_LOD0")
    if argv is not None:
        return parser.parse_args(list(argv))
    raw = list(sys.argv)
    raw = raw[raw.index("--") + 1 :] if "--" in raw else raw[1:]
    return parser.parse_args(raw)


def export(args: argparse.Namespace) -> dict[str, Any]:
    blend = args.blend.resolve()
    output_dir = args.output_dir.resolve()
    if not blend.is_file():
        raise FileNotFoundError(blend)
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    lod = bpy.data.objects.get(args.expected_object)
    if lod is None or lod.type != "MESH":
        raise RuntimeError(f"expected LOD0 mesh is missing: {args.expected_object}")
    bpy.ops.object.select_all(action="DESELECT")
    lod.hide_set(False)
    lod.hide_render = True
    lod.select_set(True)
    bpy.context.view_layer.objects.active = lod
    glb = output_dir / "Thai_Libation_Vessel_V4_BEST_DEFENSIBLE_TRIM5.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(glb),
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
    scene_objects = sorted(obj.name for obj in bpy.context.scene.objects)
    materials = sorted({slot.material.name for slot in lod.material_slots if slot.material})
    dense = json.loads(args.dense_selection.read_text(encoding="utf-8"))
    raw = json.loads(args.raw_diagnostic.read_text(encoding="utf-8"))
    appearance = json.loads(args.appearance_stats.read_text(encoding="utf-8"))
    if appearance.get("image_count") != 158:
        raise RuntimeError("appearance statistics must be bound to exactly 158 reference images")
    if appearance.get("source_role") != "appearance_reference":
        raise RuntimeError("appearance statistics are not bound to the appearance_reference source role")
    if appearance.get("photographic_projection_verified") is not False:
        raise RuntimeError("photographic projection must remain explicitly unverified")
    report = {
        "schema_version": 1,
        "status": "complete_with_documented_anatomy_limitation",
        "stage": "fresh_best_defensible_blender_glb_export",
        "blend": str(blend),
        "blend_sha256": sha256_file(blend),
        "glb": str(glb.resolve()),
        "glb_sha256": sha256_file(glb),
        "final_object": lod.name,
        "final_vertex_count": len(lod.data.vertices),
        "final_face_count": len(lod.data.polygons),
        "uv_layers": len(lod.data.uv_layers),
        "materials": materials,
        "scene_objects": scene_objects,
        "packed_images": [image.name for image in bpy.data.images if image.packed_file is not None],
        "source_dense_selection": str(args.dense_selection.resolve()),
        "source_dense_selection_sha256": sha256_file(args.dense_selection.resolve()),
        "selected_fused_sha256": dense.get("selected", {}).get("fused_sha256"),
        "source_raw_diagnostic": str(args.raw_diagnostic.resolve()),
        "source_raw_diagnostic_sha256": sha256_file(args.raw_diagnostic.resolve()),
        "raw_poisson_sha256": raw.get("poisson", {}).get("mesh_sha256"),
        "appearance": {
            "stats_path": str(args.appearance_stats.resolve()),
            "stats_sha256": sha256_file(args.appearance_stats.resolve()),
            "image_count": appearance.get("image_count"),
            "source_manifest_sha256": appearance.get("source_manifest_sha256"),
            "photographic_projection_verified": appearance.get("photographic_projection_verified"),
            "base_color_srgb_median": appearance.get("base_color_srgb_median"),
            "roughness": appearance.get("roughness"),
            "metallic": appearance.get("metallic"),
        },
        "photographic_projection_verified": False,
        "known_limitations": list(raw.get("blocking_reasons", []))
        + list(raw.get("anatomy_evidence", {}).get("failures", []))
        + ["raw trim-5 component fractions pass, but the unresolved finial and no-major-hole anatomy result remain explicitly failed; no geometry was invented"],
    }
    report_path = output_dir / "blender_best_defensible_v1_trim5.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Iterable[str] | None = None) -> int:
    report = export(parse_args(argv))
    print(json.dumps({"status": report["status"], "blend_sha256": report["blend_sha256"], "glb_sha256": report["glb_sha256"], "final_faces": report["final_face_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
