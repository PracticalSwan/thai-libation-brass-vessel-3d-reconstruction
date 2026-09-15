"""Fresh Blender 5.2 verification of the final V4 authoring master.

This verifier is intentionally non-mutating: it opens the supplied .blend, checks
that the preserved raw/clean-high sources and final LOD0 have the expected
technical state, renders eight deterministic LOD0 views, and writes a compact
JSON report beside those renders. It never saves the opened .blend.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_v4_best_defensible_glb import material_summary, render_eight_views, sha256_file


RAW_NAME = "SM_V4_Poisson_Raw"
CLEAN_NAME = "SM_V4_Scan_CleanHigh"
LOD0_NAME = "SM_V4_Vessel_LOD0"
MATERIAL_NAME = "MAT_V4_Brass"
EXPECTED_COUNTS = {
    RAW_NAME: (4_952_940, 9_899_268),
    CLEAN_NAME: (597_208, 1_187_912),
    LOD0_NAME: (151_547, 294_713),
}
REQUIRED_IMAGES = {
    "T_V4_BestDefensible_AO",
    "T_V4_BestDefensible_BaseColor",
    "T_V4_BestDefensible_Normal",
    "T_V4_BestDefensible_Roughness",
}


def _finite_mesh(obj: bpy.types.Object) -> bool:
    return all(
        all(math.isfinite(float(value)) for value in vertex.co)
        for vertex in obj.data.vertices
    )


def _uv_metrics(obj: bpy.types.Object) -> dict[str, Any]:
    layer = obj.data.uv_layers.active
    if layer is None:
        return {"present": False, "uv_area_sum": 0.0, "zero_area_faces": len(obj.data.polygons)}
    area_sum = 0.0
    zero = 0
    minimum_u = minimum_v = float("inf")
    maximum_u = maximum_v = float("-inf")
    for polygon in obj.data.polygons:
        points: list[tuple[float, float]] = []
        for loop_index in polygon.loop_indices:
            uv = layer.data[loop_index].uv
            u, v = float(uv.x), float(uv.y)
            points.append((u, v))
            minimum_u = min(minimum_u, u)
            maximum_u = max(maximum_u, u)
            minimum_v = min(minimum_v, v)
            maximum_v = max(maximum_v, v)
        area = 0.0
        for index, (x1, y1) in enumerate(points):
            x2, y2 = points[(index + 1) % len(points)]
            area += x1 * y2 - x2 * y1
        area = abs(area) * 0.5
        area_sum += area
        zero += area < 1.0e-12
    return {
        "present": True,
        "name": layer.name,
        "bounds": [minimum_u, minimum_v, maximum_u, maximum_v],
        "uv_area_sum": area_sum,
        "zero_area_faces": zero,
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    blend = args.blend.resolve()
    output_dir = args.output_dir.resolve()
    if not blend.is_file():
        raise FileNotFoundError(blend)
    bpy.ops.wm.open_mainfile(filepath=str(blend))

    rows: dict[str, Any] = {}
    for name, (expected_vertices, expected_faces) in EXPECTED_COUNTS.items():
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "MESH":
            rows[name] = {"present": False}
            continue
        rows[name] = {
            "present": True,
            "vertices": len(obj.data.vertices),
            "faces": len(obj.data.polygons),
            "expected_vertices": expected_vertices,
            "expected_faces": expected_faces,
            "counts_match": len(obj.data.vertices) == expected_vertices and len(obj.data.polygons) == expected_faces,
            "finite_vertices": _finite_mesh(obj),
            "dimensions": [float(value) for value in obj.dimensions],
            "location": [float(value) for value in obj.location],
            "rotation_euler": [float(value) for value in obj.rotation_euler],
            "scale": [float(value) for value in obj.scale],
            "hide_viewport": bool(obj.hide_viewport),
            "hide_render": bool(obj.hide_render),
            "hide_get": bool(obj.hide_get()),
        }

    lod = bpy.data.objects.get(LOD0_NAME)
    if lod is None or lod.type != "MESH":
        raise RuntimeError(f"missing final LOD0: {LOD0_NAME}")
    material = bpy.data.materials.get(MATERIAL_NAME)
    material_row = material_summary(material) if material is not None else None
    observed_images = {
        image["name"] for image in (material_row or {}).get("images", [])
    }
    uv = _uv_metrics(lod)

    source_collection = bpy.data.collections.get("COL_V4_Source")
    work_collection = bpy.data.collections.get("COL_V4_Work")
    final_collection = bpy.data.collections.get("COL_V4_Final")
    checks = {
        "all_objects_present": all(row.get("present") for row in rows.values()),
        "all_counts_match": all(row.get("counts_match") for row in rows.values()),
        "all_vertices_finite": all(row.get("finite_vertices") for row in rows.values()),
        "raw_hidden": bool(rows[RAW_NAME].get("hide_viewport") and rows[RAW_NAME].get("hide_render") and rows[RAW_NAME].get("hide_get")),
        "clean_high_hidden": bool(rows[CLEAN_NAME].get("hide_viewport") and rows[CLEAN_NAME].get("hide_render") and rows[CLEAN_NAME].get("hide_get")),
        "source_collection_hidden": bool(source_collection is not None and source_collection.hide_viewport and source_collection.hide_render),
        "work_collection_hidden": bool(work_collection is not None and work_collection.hide_viewport and work_collection.hide_render),
        "final_collection_visible": bool(final_collection is not None and not final_collection.hide_viewport and not final_collection.hide_render),
        "lod0_visible": bool(not rows[LOD0_NAME].get("hide_viewport") and not rows[LOD0_NAME].get("hide_render") and not rows[LOD0_NAME].get("hide_get")),
        "material_present": material is not None,
        "required_images_present": REQUIRED_IMAGES.issubset(observed_images),
        "required_images_packed": bool(material_row) and all(
            image["packed"] for image in material_row["images"] if image["name"] in REQUIRED_IMAGES
        ),
        "base_color_connected": "T_V4_BestDefensible_BaseColor" in (material_row or {}).get("base_color_images", []),
        "roughness_connected": "T_V4_BestDefensible_Roughness" in (material_row or {}).get("roughness_images", []),
        "normal_connected": "T_V4_BestDefensible_Normal" in (material_row or {}).get("normal_map_images", []),
        "ao_occlusion_connected": "T_V4_BestDefensible_AO" in (material_row or {}).get("occlusion_images", []),
        "uv_present": bool(uv.get("present")),
        "uv_usable_area": float(uv.get("uv_area_sum", 0.0)) >= 0.25,
    }
    checks["passed"] = bool(all(checks.values()))

    output_dir.mkdir(parents=True, exist_ok=True)
    renders = render_eight_views(lod, output_dir / "renders")
    report = {
        "schema_version": 1,
        "status": "passed" if checks["passed"] else "failed",
        "stage": "fresh_final_blend_authoring_verification",
        "blend": str(blend),
        "blend_sha256": sha256_file(blend),
        "objects": rows,
        "material": material_row,
        "uv": uv,
        "checks": checks,
        "renders": renders,
    }
    report_path = output_dir / "blend_authoring_verification.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not checks["passed"]:
        raise RuntimeError("final Blend verification failed: " + json.dumps(checks, sort_keys=True))
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    if argv is not None:
        return parser.parse_args(list(argv))
    raw = list(sys.argv)
    raw = raw[raw.index("--") + 1 :] if "--" in raw else raw[1:]
    return parser.parse_args(raw)


def main(argv: Iterable[str] | None = None) -> int:
    report = verify(parse_args(argv))
    print(json.dumps({"status": report["status"], "checks": report["checks"], "render_count": len(report["renders"]), "blend_sha256": report["blend_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
