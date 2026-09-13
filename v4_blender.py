"""Blender-side naming and acceptance contracts for the V4 scan-derived asset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from v4_config import RECONSTRUCTION_V4_ROOT, assert_output_path, write_json


COLLECTIONS = (
    "COL_V4_Source",
    "COL_V4_Work",
    "COL_V4_Final",
    "COL_V4_Lookdev",
)
RAW_OBJECT = "SM_V4_Poisson_Raw"
CLEAN_OBJECT = "SM_V4_Scan_CleanHigh"
FINAL_OBJECT = "SM_V4_Vessel_LOD0"
MATERIAL_NAME = "MAT_V4_Brass"
BLEND_PATH = RECONSTRUCTION_V4_ROOT / "blender" / "Thai_Libation_Vessel_V4_FINAL.blend"
GLB_PATH = RECONSTRUCTION_V4_ROOT / "blender" / "Thai_Libation_Vessel_V4_FINAL.glb"


def blender_paths() -> dict[str, Path]:
    return {
        "blend": assert_output_path(BLEND_PATH),
        "glb": assert_output_path(GLB_PATH),
        "raw_mesh": assert_output_path(RECONSTRUCTION_V4_ROOT / "mesh" / "poisson_raw.ply"),
    }


def scene_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate a structured Blender MCP scene summary without guessing state."""

    objects = set(str(value) for value in summary.get("objects", ()))
    collections = set(str(value) for value in summary.get("collections", ()))
    materials = set(str(value) for value in summary.get("materials", ()))
    checks = {
        "collections": set(COLLECTIONS).issubset(collections),
        "raw_preserved": RAW_OBJECT in objects,
        "clean_high_present": CLEAN_OBJECT in objects,
        "final_present": FINAL_OBJECT in objects,
        "brass_material": MATERIAL_NAME in materials,
        "nonzero_mesh": int(summary.get("final_vertex_count", 0)) > 0 and int(summary.get("final_face_count", 0)) > 0,
        "sane_transforms": bool(summary.get("sane_transforms", False)),
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def glb_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "mesh_nonzero": int(summary.get("vertex_count", 0)) > 0 and int(summary.get("face_count", 0)) > 0,
        "material_present": MATERIAL_NAME in set(str(value) for value in summary.get("materials", ())),
        "textures_resolve": bool(summary.get("textures_resolve", False)),
        "sane_bounds": bool(summary.get("sane_bounds", False)),
        "normals_ok": bool(summary.get("normals_ok", False)),
        "only_final_exported": bool(summary.get("only_final_exported", False)),
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def write_blender_report(payload: Mapping[str, Any], path: Path | None = None) -> Path:
    return write_json(path or RECONSTRUCTION_V4_ROOT / "reports" / "blender_gate.json", dict(payload))


__all__ = [
    "BLEND_PATH",
    "CLEAN_OBJECT",
    "COLLECTIONS",
    "FINAL_OBJECT",
    "GLB_PATH",
    "MATERIAL_NAME",
    "RAW_OBJECT",
    "blender_paths",
    "glb_gate",
    "scene_gate",
    "write_blender_report",
]
