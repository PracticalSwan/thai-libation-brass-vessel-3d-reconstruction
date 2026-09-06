"""Small, dependency-light PLY and textured-asset validation helpers.

The parser intentionally supports the PLY contracts emitted by COLMAP 4.2:
ASCII/binary point clouds and triangle meshes, including per-corner ``texcoord``
lists in the binary textured PLY. It rejects ambiguous or unsafe geometry.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from pathlib import Path
import struct
from typing import Any

import numpy as np
from PIL import Image


_TYPES = {
    "char": ("b", np.int8), "int8": ("b", np.int8),
    "uchar": ("B", np.uint8), "uint8": ("B", np.uint8),
    "short": ("h", np.int16), "int16": ("h", np.int16),
    "ushort": ("H", np.uint16), "uint16": ("H", np.uint16),
    "int": ("i", np.int32), "int32": ("i", np.int32),
    "uint": ("I", np.uint32), "uint32": ("I", np.uint32),
    "int64": ("q", np.int64), "uint64": ("Q", np.uint64),
    "float": ("f", np.float32), "float32": ("f", np.float32),
    "double": ("d", np.float64), "float64": ("d", np.float64),
}


def _type_info(name: str):
    try:
        return _TYPES[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported PLY scalar type: {name}") from error


def _read_exact(handle, count: int, what: str) -> bytes:
    data = handle.read(count)
    if len(data) != count:
        raise ValueError(f"truncated PLY {what}")
    return data


def _header(handle):
    first = handle.readline()
    if first.strip() != b"ply":
        raise ValueError("Not a PLY file")
    format_name = None
    elements = []
    comments = []
    current = None
    while True:
        line = handle.readline()
        if not line:
            raise ValueError("truncated PLY header")
        stripped = line.strip()
        if stripped == b"end_header":
            break
        try:
            tokens = stripped.decode("ascii").split()
        except UnicodeDecodeError as error:
            raise ValueError("PLY header is not ASCII") from error
        if not tokens:
            continue
        keyword = tokens[0].lower()
        if keyword == "format" and len(tokens) >= 2:
            format_name = tokens[1].lower()
        elif keyword == "comment":
            comments.append(" ".join(tokens[1:]))
        elif keyword == "element" and len(tokens) == 3:
            try:
                count = int(tokens[2])
            except ValueError as error:
                raise ValueError("Invalid PLY element count") from error
            if count < 0:
                raise ValueError("Negative PLY element count")
            current = {"name": tokens[1], "count": count, "properties": []}
            elements.append(current)
        elif keyword == "property" and current is not None:
            if len(tokens) == 3:
                _type_info(tokens[1])
                current["properties"].append(("scalar", tokens[2], tokens[1]))
            elif len(tokens) == 5 and tokens[1].lower() == "list":
                _type_info(tokens[2])
                _type_info(tokens[3])
                current["properties"].append(("list", tokens[4], tokens[2], tokens[3]))
            else:
                raise ValueError("Malformed PLY property")
    if format_name not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"Unsupported PLY format: {format_name}")
    return format_name, elements, comments


def _scalar_ascii(token: str, type_name: str):
    code, dtype = _type_info(type_name)
    try:
        return float(token) if dtype in (np.float32, np.float64) else int(token)
    except ValueError as error:
        raise ValueError(f"Invalid PLY scalar: {token}") from error


def _row_ascii(tokens: list[str], properties):
    values = {}
    cursor = 0
    for prop in properties:
        if prop[0] == "scalar":
            if cursor >= len(tokens):
                raise ValueError("truncated PLY ASCII row")
            values[prop[1]] = _scalar_ascii(tokens[cursor], prop[2])
            cursor += 1
        else:
            _, name, count_type, value_type = prop
            if cursor >= len(tokens):
                raise ValueError("truncated PLY ASCII list")
            count = int(_scalar_ascii(tokens[cursor], count_type))
            cursor += 1
            if count < 0 or cursor + count > len(tokens):
                raise ValueError("Invalid PLY ASCII list length")
            values[name] = [_scalar_ascii(token, value_type) for token in tokens[cursor:cursor + count]]
            cursor += count
    if cursor != len(tokens):
        raise ValueError("Extra values in PLY ASCII row")
    return values


def _scalar_binary(handle, type_name: str, endian: str):
    code, _ = _type_info(type_name)
    size = struct.calcsize(endian + code)
    return struct.unpack(endian + code, _read_exact(handle, size, "scalar"))[0]


def _row_binary(handle, properties, endian: str):
    values = {}
    for prop in properties:
        if prop[0] == "scalar":
            values[prop[1]] = _scalar_binary(handle, prop[2], endian)
        else:
            _, name, count_type, value_type = prop
            count = int(_scalar_binary(handle, count_type, endian))
            if count < 0 or count > 10_000_000:
                raise ValueError("Invalid PLY binary list length")
            values[name] = [_scalar_binary(handle, value_type, endian) for _ in range(count)]
    return values


def _read_rows(handle, format_name: str, element: dict):
    rows = []
    endian = "<" if format_name == "binary_little_endian" else ">"
    for _ in range(element["count"]):
        if format_name == "ascii":
            line = handle.readline()
            if not line:
                raise ValueError("truncated PLY ASCII element")
            try:
                tokens = line.decode("ascii").split()
            except UnicodeDecodeError as error:
                raise ValueError("Non-ASCII PLY row") from error
            rows.append(_row_ascii(tokens, element["properties"]))
        else:
            rows.append(_row_binary(handle, element["properties"], endian))
    return rows


def _as_array(rows, names, dtype, label):
    if not rows:
        return np.empty((0, len(names)), dtype=dtype)
    if any(any(name not in row for name in names) for row in rows):
        raise ValueError(f"PLY vertex properties missing {label}")
    values = np.asarray([[row[name] for name in names] for row in rows], dtype=dtype)
    if not np.isfinite(values).all():
        raise ValueError(f"PLY {label} contains non-finite values")
    return values


def read_ply(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"PLY is missing: {path}")
    with path.open("rb") as handle:
        format_name, elements, comments = _header(handle)
        vertex_rows = []
        face_rows = []
        for element in elements:
            rows = _read_rows(handle, format_name, element)
            name = element["name"].lower()
            if name == "vertex":
                vertex_rows = rows
            elif name == "face":
                face_rows = rows
    xyz = _as_array(vertex_rows, ("x", "y", "z"), np.float64, "coordinates")
    normal_names = ("nx", "ny", "nz")
    normals = None
    if vertex_rows and all(name in vertex_rows[0] for name in normal_names):
        normals = _as_array(vertex_rows, normal_names, np.float64, "normals")
    color_names = ("red", "green", "blue")
    if vertex_rows and not all(name in vertex_rows[0] for name in color_names):
        color_names = ("r", "g", "b")
    colors = None
    if vertex_rows and all(name in vertex_rows[0] for name in color_names):
        colors = np.asarray([[row[name] for name in color_names] for row in vertex_rows], dtype=np.int64)
        if np.any(colors < 0) or np.any(colors > 255):
            raise ValueError("PLY colors outside 0..255")
        colors = colors.astype(np.uint8)

    faces_list = []
    uv_list = []
    has_uv = False
    for row in face_rows:
        index_key = next((key for key in ("vertex_indices", "vertex_index") if key in row), None)
        if index_key is None:
            raise ValueError("PLY face has no vertex index list")
        indices = list(row[index_key])
        if len(indices) != 3:
            raise ValueError("PLY mesh must contain triangular faces")
        indices = [int(value) for value in indices]
        if any(value < 0 or value >= len(xyz) for value in indices):
            raise ValueError("PLY face index outside vertex range")
        faces_list.append(indices)
        uv_value = row.get("texcoord")
        if uv_value is None:
            uv_list.append(None)
        else:
            has_uv = True
            if len(uv_value) not in (0, 6):
                raise ValueError("PLY texcoord list must contain six values per triangle")
            uv_list.append(np.asarray(uv_value, dtype=np.float64).reshape(3, 2) if len(uv_value) else None)
    faces = np.asarray(faces_list, dtype=np.int32).reshape((-1, 3)) if faces_list else None
    uvs = None
    if has_uv:
        uvs = np.full((len(uv_list), 3, 2), np.nan, dtype=np.float64)
        for index, value in enumerate(uv_list):
            if value is not None:
                uvs[index] = value
        if np.isinf(uvs).any():
            raise ValueError("PLY texture coordinates are non-finite")
    texture_file = None
    for comment in comments:
        if comment.lower().startswith("texturefile "):
            texture_file = comment.split(None, 1)[1].strip()
            break
    return dict(path=path, xyz=xyz, colors=colors, normals=normals,
                faces=faces, uvs=uvs, texture_file=texture_file,
                comments=comments, format=format_name)


def _bounds(xyz):
    if len(xyz) == 0:
        return None, None, None, 0
    low = xyz.min(axis=0)
    high = xyz.max(axis=0)
    extent = high - low
    rank = int(np.linalg.matrix_rank(xyz - xyz.mean(axis=0))) if len(xyz) > 1 else 0
    return low.tolist(), high.tolist(), extent.tolist(), rank


def ply_metrics(path: Path, reference_bounds: dict | None = None) -> dict[str, Any]:
    data = read_ply(path)
    xyz = data["xyz"]
    low, high, extent, rank = _bounds(xyz)
    faces = data["faces"]
    finite = np.isfinite(xyz).all(axis=1)
    result = dict(point_count=int(len(xyz)), vertex_count=int(len(xyz)),
                  face_count=int(0 if faces is None else len(faces)),
                  finite_xyz_count=int(finite.sum()),
                  finite_xyz_fraction=float(finite.mean()) if len(finite) else 1.0,
                  has_color=data["colors"] is not None,
                  has_normals=data["normals"] is not None,
                  has_uv=data["uvs"] is not None,
                  bounding_box_min=low, bounding_box_max=high,
                  bounding_box_extents=extent, rank=rank,
                  file_size_bytes=Path(path).stat().st_size)
    if len(xyz):
        result["robust_bounding_box_min"] = np.quantile(xyz, .001, axis=0).tolist()
        result["robust_bounding_box_max"] = np.quantile(xyz, .999, axis=0).tolist()
    if reference_bounds is not None and len(xyz):
        ref_low = np.asarray(reference_bounds["bounding_box_min"], dtype=np.float64)
        ref_high = np.asarray(reference_bounds["bounding_box_max"], dtype=np.float64)
        span = ref_high - ref_low
        inside = np.all((xyz >= ref_low - span) & (xyz <= ref_high + span), axis=1)
        result["reference_expanded_fraction"] = float(inside.mean())
        result["reference_expanded_outlier_count"] = int((~inside).sum())
    if data["uvs"] is not None:
        valid = np.isfinite(data["uvs"]).all(axis=(1, 2))
        result["uv_face_count"] = int(valid.sum())
        result["meaningful_uv_fraction"] = float(valid.mean()) if len(valid) else 0.0
    return result


def component_summary(data: dict[str, Any], *, include_labels: bool = False) -> dict[str, Any]:
    faces = data.get("faces")
    if faces is None or len(faces) == 0:
        return dict(component_count=0, face_counts=[], face_labels=[])
    parent = list(range(len(faces)))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    by_vertex = defaultdict(list)
    for face_index, face in enumerate(faces):
        for vertex in face:
            by_vertex[int(vertex)].append(face_index)
    for indices in by_vertex.values():
        for index in indices[1:]:
            union(indices[0], index)
    groups = defaultdict(list)
    for index in range(len(faces)):
        groups[find(index)].append(index)
    ordered = sorted(groups.values(), key=lambda group: (-len(group), group[0]))
    labels = [0] * len(faces)
    for label, group in enumerate(ordered):
        for index in group:
            labels[index] = label
    result = dict(component_count=len(ordered), face_counts=[len(group) for group in ordered],
                  dominant_face_fraction=len(ordered[0]) / len(faces))
    if include_labels:
        result["face_labels"] = labels
    return result


def filter_small_components(data: dict[str, Any], minimum_face_fraction: float = .005):
    """Remove only components below a face-share rule and remap used vertices."""
    if not 0 < minimum_face_fraction < 1:
        raise ValueError("Component threshold must be a fraction between zero and one")
    summary = component_summary(data, include_labels=True)
    faces = data.get("faces")
    if faces is None or len(faces) == 0:
        raise ValueError("Component filtering requires a triangular mesh")
    threshold = max(1, int(np.ceil(len(faces) * minimum_face_fraction)))
    keep_components = sum(count >= threshold for count in summary["face_counts"])
    if keep_components == 0:
        raise ValueError("Component rule would remove the entire mesh")
    labels = np.asarray(summary.pop("face_labels"), dtype=np.int32)
    kept_faces = np.asarray(faces)[labels < keep_components]
    used = np.unique(kept_faces.reshape(-1))
    remap = np.full(len(data["xyz"]), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    result = dict(xyz=np.asarray(data["xyz"])[used], faces=remap[kept_faces],
                  colors=None if data.get("colors") is None else np.asarray(data["colors"])[used],
                  normals=None if data.get("normals") is None else np.asarray(data["normals"])[used])
    report = dict(used=True, minimum_face_fraction=minimum_face_fraction,
                  minimum_faces=threshold, source_components=summary["component_count"],
                  kept_components=keep_components,
                  removed_components=summary["component_count"] - keep_components,
                  source_faces=len(faces), kept_faces=len(kept_faces),
                  removed_faces=len(faces) - len(kept_faces),
                  retained_face_fraction=len(kept_faces) / len(faces))
    return result, report


def write_ply(path: Path, xyz, *, faces=None, colors=None, normals=None, uvs=None, texture_file=None):
    path = Path(path)
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("PLY writer requires finite Nx3 coordinates")
    faces = None if faces is None else np.asarray(faces, dtype=np.int32)
    if faces is not None and (faces.ndim != 2 or faces.shape[1] != 3 or np.any(faces < 0) or np.any(faces >= len(xyz))):
        raise ValueError("PLY writer requires in-range triangular faces")
    colors = None if colors is None else np.asarray(colors, dtype=np.uint8)
    normals = None if normals is None else np.asarray(normals, dtype=np.float64)
    uvs = None if uvs is None else np.asarray(uvs, dtype=np.float64)
    if colors is not None and colors.shape != (len(xyz), 3):
        raise ValueError("PLY colors must be Nx3")
    if normals is not None and (normals.shape != (len(xyz), 3) or not np.isfinite(normals).all()):
        raise ValueError("PLY normals must be finite Nx3")
    if uvs is not None and (faces is None or uvs.shape != (len(faces), 3, 2) or not np.isfinite(uvs).all()):
        raise ValueError("PLY UVs must be finite Mx3x2")
    lines = ["ply", "format ascii 1.0"]
    if texture_file:
        lines.append(f"comment TextureFile {Path(texture_file).name}")
    lines.extend([f"element vertex {len(xyz)}", "property double x", "property double y", "property double z"])
    if normals is not None:
        lines.extend(["property double nx", "property double ny", "property double nz"])
    if colors is not None:
        lines.extend(["property uchar red", "property uchar green", "property uchar blue"])
    if faces is not None:
        lines.extend([f"element face {len(faces)}", "property list uchar int vertex_indices"])
        if uvs is not None:
            lines.append("property list uchar float texcoord")
    lines.append("end_header")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
        for index, point in enumerate(xyz):
            values = [f"{value:.17g}" for value in point]
            if normals is not None:
                values.extend(f"{value:.17g}" for value in normals[index])
            if colors is not None:
                values.extend(str(int(value)) for value in colors[index])
            handle.write(" ".join(values) + "\n")
        if faces is not None:
            for index, face in enumerate(faces):
                row = "3 " + " ".join(str(int(value)) for value in face)
                if uvs is not None:
                    row += " 6 " + " ".join(f"{value:.17g}" for value in uvs[index].reshape(-1))
                handle.write(row + "\n")


def _file_manifest(path: Path, role: str) -> dict[str, Any]:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    entry = dict(relative_path=path.name, role=role, size_bytes=path.stat().st_size,
                 sha256=digest)
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        with Image.open(path) as image:
            image.verify()
            entry["image_dimensions"] = [int(image.width), int(image.height)]
    return entry


def validate_textured_asset(output_dir: Path, reference_mesh: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    mesh_path = output_dir / "mesh.ply"
    if not mesh_path.is_file():
        raise ValueError("Textured mesh.ply is missing")
    data = read_ply(mesh_path)
    reference = read_ply(reference_mesh)
    if len(data["xyz"]) != len(reference["xyz"]):
        raise ValueError("Textured mesh vertex count changed")
    if (0 if data["faces"] is None else len(data["faces"])) != (0 if reference["faces"] is None else len(reference["faces"])):
        raise ValueError("Textured mesh face count changed")
    if data["faces"] is None or reference["faces"] is None or not np.array_equal(data["faces"], reference["faces"]):
        raise ValueError("Textured mesh topology changed")
    if not np.allclose(data["xyz"], reference["xyz"], rtol=1e-5, atol=1e-7):
        raise ValueError("Textured mesh coordinates changed")
    texture_name = data.get("texture_file")
    if not texture_name or Path(texture_name).name != texture_name:
        raise ValueError("Textured mesh has no safe TextureFile reference")
    texture_path = output_dir / texture_name
    if not texture_path.is_file():
        raise ValueError("Referenced texture image is missing")
    try:
        with Image.open(texture_path) as image:
            image.verify()
            dimensions = [int(image.width), int(image.height)]
    except Exception as error:
        raise ValueError("Referenced texture image is unreadable") from error
    if data["uvs"] is None or data["faces"] is None:
        raise ValueError("Textured mesh has no UV-bearing faces")
    valid = np.isfinite(data["uvs"]).all(axis=(1, 2))
    finite_uvs = data["uvs"][valid]
    if len(finite_uvs) and np.any((finite_uvs < 0.0) | (finite_uvs > 1.0)):
        raise ValueError("Textured mesh UV coordinates are outside the normalized atlas range")
    areas = np.zeros(len(data["uvs"]), dtype=np.float64)
    for index, uv in enumerate(data["uvs"]):
        if valid[index]:
            delta_a = uv[1] - uv[0]
            delta_b = uv[2] - uv[0]
            areas[index] = abs(float(delta_a[0] * delta_b[1] - delta_a[1] * delta_b[0])) / 2
    meaningful_fraction = float(np.mean(areas > 1e-8)) if len(areas) else 0.0
    minimum_meaningful_fraction = .5
    if meaningful_fraction < minimum_meaningful_fraction:
        raise ValueError("Textured mesh does not provide substantial UV coverage")
    entries = [_file_manifest(mesh_path, "textured_mesh"), _file_manifest(texture_path, "photo_texture")]
    return dict(accepted=True, mesh_path=mesh_path.as_posix(), texture_path=texture_path.as_posix(),
                texture_dimensions=dimensions, vertex_count=len(data["xyz"]), face_count=len(data["faces"]),
                meaningful_uv_fraction=meaningful_fraction,
                minimum_meaningful_uv_fraction=minimum_meaningful_fraction,
                normalized_uv_range=True,
                uv_min=float(finite_uvs.min()) if len(finite_uvs) else None,
                uv_max=float(finite_uvs.max()) if len(finite_uvs) else None,
                topology_preserved=True, output_manifest=entries,
                geometry_bounds=ply_metrics(mesh_path)["bounding_box_extents"])
