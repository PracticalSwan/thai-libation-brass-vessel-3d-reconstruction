"""Render deterministic technical projections of an accepted raw Poisson mesh.

This is a read-only visual-evidence helper.  It projects the actual triangle
surface into eight fixed camera directions; it does not smooth, close holes,
edit topology, or manufacture geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Running this helper as ``py scripts/...`` sets ``sys.path[0]`` to the
# scripts directory; add the repository root so the project PLY contract is
# imported from the current checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_reconstruction_io import read_ply
from v4_postfusion import ANATOMY_VIEW_KEYS, anatomy_region_evidence


VIEWS: dict[str, tuple[float, float, float]] = {
    "bowl_interior": (0.0, 0.0, 1.0),
    "rim": (0.7071, 0.0, 0.7071),
    "globe_shoulder": (1.0, 0.0, 0.0),
    "continuous_neck": (0.7071, 0.0, -0.7071),
    "lid_tiers": (0.0, 0.0, -1.0),
    "finial": (0.35, 0.38, 0.86),
    "pedestal_transitions": (-0.35, -0.30, 0.88),
    "base": (-0.62, 0.24, 0.74),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("view basis is degenerate")
    return value / norm


def _render(
    xyz: np.ndarray,
    faces: np.ndarray,
    *,
    view_direction: tuple[float, float, float],
    screen_up: np.ndarray,
    title: str,
    target: Path,
    max_faces: int,
) -> None:
    if len(xyz) == 0 or len(faces) == 0:
        raise ValueError("raw mesh has no renderable triangles")
    vertical = _unit(np.asarray(screen_up, dtype=np.float64))
    direction = _unit(np.asarray(view_direction, dtype=np.float64))
    horizontal = _unit(np.cross(vertical, direction))
    vertical = _unit(np.cross(direction, horizontal))
    center = np.mean(xyz, axis=0)

    # Deterministic evenly spaced face sampling keeps the evidence bounded while
    # retaining the original face ordering and all major components.
    if len(faces) > max_faces:
        face_indices = np.linspace(0, len(faces) - 1, num=max_faces, dtype=np.int64)
        selected = faces[face_indices]
    else:
        selected = faces
    triangles = xyz[selected]
    screen_x = (triangles - center) @ horizontal
    screen_y = (triangles - center) @ vertical
    depth = (triangles - center) @ direction
    x_low, x_high = np.quantile(screen_x, [0.002, 0.998])
    y_low, y_high = np.quantile(screen_y, [0.002, 0.998])
    x_span = max(float(x_high - x_low), 1e-12)
    y_span = max(float(y_high - y_low), 1e-12)

    width, height = 1000, 800
    canvas = Image.new("RGB", (width, height), (246, 246, 243))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((24, 38, width - 24, height - 24), outline=(120, 120, 115), width=2)
    # Draw far-to-near so the sampled surface remains visually occlusion-aware.
    order = np.argsort(np.mean(depth, axis=1))
    for index in order:
        projected = []
        for point_x, point_y in zip(screen_x[index], screen_y[index]):
            px = int(30 + (float(point_x) - x_low) / x_span * (width - 60))
            py = int(height - 32 - (float(point_y) - y_low) / y_span * (height - 92))
            projected.append((px, py))
        if all(28 <= px < width - 28 and 42 <= py < height - 28 for px, py in projected):
            # Technical brass-like neutral is only a display color; it is not an
            # appearance claim and does not derive a final material.
            draw.polygon(projected, fill=(174, 126, 45))
    draw.text((34, 14), title, fill=(20, 20, 20))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, format="PNG", optimize=True)


def render_raw_mesh_views(
    mesh_path: Path,
    output_dir: Path,
    *,
    basis_vectors: dict[str, list[float]],
    basis_source: Path,
    max_faces: int = 180_000,
) -> dict[str, object]:
    data = read_ply(mesh_path)
    faces = data.get("faces")
    if faces is None:
        raise ValueError("raw Poisson mesh has no faces")
    xyz = np.asarray(data["xyz"], dtype=np.float64)
    faces_array = np.asarray(faces, dtype=np.int64)
    region_evidence = anatomy_region_evidence(
        xyz,
        basis_vectors=basis_vectors,
        inside_support=np.ones(len(xyz), dtype=bool),
        inside_view_counts=np.full(len(xyz), 8, dtype=np.int32),
        support_view_count=8,
    )
    center = np.asarray(basis_vectors["center"], dtype=np.float64)
    vertical = _unit(np.asarray(basis_vectors["vertical"], dtype=np.float64))
    front = _unit(np.asarray(basis_vectors["front"], dtype=np.float64))
    right = _unit(np.asarray(basis_vectors["right"], dtype=np.float64))
    # Directions are expressed in the accepted basis, then transformed into
    # mesh coordinates.  Every crop therefore shares the same measured
    # vertical convention as the dense audit.
    views = {
        "bowl_interior": _unit(front - vertical * 0.65),
        "rim": _unit(front - vertical * 0.85 + right * 0.45),
        "globe_shoulder": front,
        "continuous_neck": _unit(front + right * 0.25),
        "lid_tiers": _unit(front - vertical * 0.25 + right * 0.35),
        "finial": _unit(-right + front * 0.15),
        "pedestal_transitions": _unit(front + vertical * 0.40),
        "base": _unit(front + vertical * 0.70 - right * 0.45),
    }
    normalized_height = (xyz - center) @ vertical
    low, high = np.quantile(normalized_height, [0.01, 0.99])
    normalized_height = np.clip((normalized_height - low) / max(float(high - low), 1e-12), 0.0, 1.0)
    records: dict[str, dict[str, object]] = {}
    for key in ANATOMY_VIEW_KEYS:
        lower, upper = (float(value) for value in region_evidence["regions"][key]["normalized_height_bounds"])
        face_centers = np.mean(xyz[faces_array], axis=1)
        face_height = (face_centers - center) @ vertical
        face_normalized = np.clip((face_height - low) / max(float(high - low), 1e-12), 0.0, 1.0)
        selected_faces = faces_array[(face_normalized >= lower) & (face_normalized <= upper)]
        if len(selected_faces) == 0:
            raise ValueError(f"raw mesh region has no triangles: {key}")
        target = output_dir / f"raw_poisson_{key}.png"
        _render(
            xyz,
            selected_faces,
            view_direction=views[key],
            screen_up=vertical,
            title=f"V4 raw Poisson region crop: {key}",
            target=target,
            max_faces=max_faces,
        )
        records[key] = {
            "path": str(target.resolve()),
            "sha256": _sha256(target),
            "anatomy_label": key,
            "region_crop": {
                "region_name": key,
                "projection_scope": "region_only_face_subset",
                "whole_object_projection": False,
                "basis_sha256": _sha256(basis_source),
                "face_count": int(len(selected_faces)),
                "vertex_count": int(len(np.unique(selected_faces))),
                "normalized_height_bounds": [lower, upper],
            },
        }
    return {
        "views": records,
        "anatomy_region_evidence": region_evidence,
        "basis_vectors": basis_vectors,
        "basis_source": str(basis_source.resolve()),
        "basis_sha256": _sha256(basis_source),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--basis-report", type=Path, required=True)
    parser.add_argument("--max-faces", type=int, default=180_000)
    args = parser.parse_args()
    if args.max_faces < 1:
        raise SystemExit("--max-faces must be positive")
    basis_report = json.loads(args.basis_report.resolve().read_text(encoding="utf-8"))
    basis_vectors = basis_report.get("contamination", {}).get("basis_vectors")
    if not isinstance(basis_vectors, dict):
        raise SystemExit("basis report does not contain contamination.basis_vectors")
    records = render_raw_mesh_views(
        args.mesh.resolve(),
        args.output_dir.resolve(),
        basis_vectors=basis_vectors,
        basis_source=args.basis_report.resolve(),
        max_faces=args.max_faces,
    )
    report = {
        "schema_version": 2,
        "mesh_path": str(args.mesh.resolve()),
        "mesh_sha256": _sha256(args.mesh.resolve()),
        "max_faces": args.max_faces,
        **records,
        "status": "rendered",
    }
    report_path = args.output_dir.resolve().parent / f"{args.output_dir.name}_evidence.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
