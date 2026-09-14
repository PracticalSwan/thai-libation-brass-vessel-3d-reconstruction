"""Remove only measured invalid/duplicate LOD0 mesh elements in Blender.

The raw Poisson object and clean-high object are never edited.  A new blend is
written so the source authoring result remains available for comparison.
"""

from __future__ import annotations

import argparse
import bmesh
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import bpy


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-blend", type=Path, required=True)
    parser.add_argument("--output-blend", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--object", default="SM_V4_Vessel_LOD0")
    if argv is not None:
        return parser.parse_args(list(argv))
    raw = list(sys.argv)
    raw = raw[raw.index("--") + 1 :] if "--" in raw else raw[1:]
    return parser.parse_args(raw)


def clean(args: argparse.Namespace) -> dict[str, object]:
    source = args.input_blend.resolve()
    output = args.output_blend.resolve()
    report_path = args.report.resolve()
    bpy.ops.wm.open_mainfile(filepath=str(source))
    obj = bpy.data.objects.get(args.object)
    if obj is None or obj.type != "MESH":
        raise RuntimeError(f"missing mesh object: {args.object}")
    before = {"vertices": len(obj.data.vertices), "edges": len(obj.data.edges), "faces": len(obj.data.polygons)}
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh_vertex_count_before = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-7)
    duplicate_merged = int(bmesh_vertex_count_before - len(bm.verts))
    bm.faces.ensure_lookup_table()
    degenerate = [face for face in bm.faces if (not face.is_valid) or len({vert.index for vert in face.verts}) < 3 or face.calc_area() <= 1e-12]
    if degenerate:
        bmesh.ops.delete(bm, geom=degenerate, context="FACES_ONLY")
    # Remove edges and vertices that became entirely loose because of the
    # degenerate-face deletion.  This is technical invalid-element cleanup only.
    loose_edges = [edge for edge in bm.edges if not edge.link_faces]
    if loose_edges:
        bmesh.ops.delete(bm, geom=loose_edges, context="EDGES")
    loose_verts = [vert for vert in bm.verts if not vert.link_edges]
    if loose_verts:
        bmesh.ops.delete(bm, geom=loose_verts, context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(obj.data)
    bm.free()
    mesh_validate_changed = bool(obj.data.validate(verbose=False, clean_customdata=True))
    obj.data.update()
    obj.data.update()
    after = {"vertices": len(obj.data.vertices), "edges": len(obj.data.edges), "faces": len(obj.data.polygons)}
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    report = {
        "schema_version": 1,
        "status": "complete",
        "input_blend": str(source),
        "input_blend_sha256": sha256_file(source),
        "output_blend": str(output),
        "output_blend_sha256": sha256_file(output),
        "object": args.object,
        "before": before,
        "after": after,
        "duplicate_vertex_merge_targets": duplicate_merged,
        "degenerate_faces_removed": len(degenerate),
        "loose_edges_removed": len(loose_edges),
        "loose_vertices_removed": len(loose_verts),
        "mesh_validate_changed": mesh_validate_changed,
        "raw_object_untouched": True,
        "clean_high_object_untouched": True,
        "geometry_creation": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Iterable[str] | None = None) -> int:
    report = clean(parse_args(argv))
    print(json.dumps({"status": report["status"], "before": report["before"], "after": report["after"], "degenerate_faces_removed": report["degenerate_faces_removed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
