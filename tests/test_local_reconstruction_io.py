"""PLY and textured-output contracts using tiny hand-built fixtures."""
from pathlib import Path
import struct

import numpy as np
import pytest
from PIL import Image

from local_reconstruction_io import (
    component_summary,
    filter_small_components,
    ply_metrics,
    read_ply,
    validate_textured_asset,
    write_ply,
)


def test_ascii_cloud_reads_xyz_normals_colors_and_metrics(tmp_path):
    path = tmp_path / "cloud.ply"
    path.write_text(
        "ply\nformat ascii 1.0\ncomment fixture\n"
        "element vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        "0 0 0 0 0 1 255 0 1\n1 0 0 0 0 1 0 255 2\n0 1 0 0 0 1 2 3 4\n",
        encoding="ascii",
    )
    data = read_ply(path)
    assert data["xyz"].shape == (3, 3)
    assert data["colors"].tolist() == [[255, 0, 1], [0, 255, 2], [2, 3, 4]]
    assert np.allclose(data["normals"], [[0, 0, 1]] * 3)
    metrics = ply_metrics(path)
    assert metrics["point_count"] == 3 and metrics["face_count"] == 0
    assert metrics["rank"] == 2 and metrics["has_color"] and metrics["has_normals"]
    assert metrics["finite_xyz_fraction"] == 1.0


def test_binary_little_endian_mesh_reads_faces_and_rejects_bad_index(tmp_path):
    path = tmp_path / "mesh.ply"
    header = (b"ply\nformat binary_little_endian 1.0\n"
              b"element vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
              b"element face 1\nproperty list uchar int vertex_indices\nend_header\n")
    body = b"".join(struct.pack("<fff", *xyz) for xyz in ((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    body += struct.pack("<Biii", 3, 0, 1, 2)
    path.write_bytes(header + body)
    data = read_ply(path)
    assert data["faces"].tolist() == [[0, 1, 2]]
    assert ply_metrics(path)["face_count"] == 1
    bad = path.with_name("bad.ply")
    bad.write_bytes(header + body[:-4] + struct.pack("<i", 9))
    with pytest.raises(ValueError, match="index"):
        read_ply(bad)


def test_binary_big_endian_and_nontriangular_face_are_explicit(tmp_path):
    path = tmp_path / "big.ply"
    header = (b"ply\nformat binary_big_endian 1.0\n"
              b"element vertex 4\nproperty double x\nproperty double y\nproperty double z\n"
              b"element face 1\nproperty list uchar int vertex_indices\nend_header\n")
    body = b"".join(struct.pack(">ddd", *xyz) for xyz in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)))
    path.write_bytes(header + body + struct.pack(">Biiii", 4, 0, 1, 2, 3))
    with pytest.raises(ValueError, match="triang"):
        read_ply(path)


def test_write_roundtrip_component_summary_and_texture_validation(tmp_path):
    mesh = tmp_path / "source.ply"
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0], [10, 1, 0], [11, 0, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    write_ply(mesh, xyz, faces=faces)
    components = component_summary(read_ply(mesh))
    assert components["component_count"] == 2
    assert components["face_counts"] == [1, 1]

    output = tmp_path / "texture"
    output.mkdir()
    textured = output / "mesh.ply"
    uvs = np.array([[[0, 0], [1, 0], [0, 1]], [[0, 0], [1, 0], [0, 1]]], dtype=np.float32)
    write_ply(textured, xyz, faces=faces, uvs=uvs, texture_file="texture.png")
    Image.new("RGB", (8, 8), (100, 80, 20)).save(output / "texture.png")
    report = validate_textured_asset(output, mesh)
    assert report["accepted"] and report["texture_dimensions"] == [8, 8]
    assert report["meaningful_uv_fraction"] == 1.0
    (output / "texture.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="texture"):
        validate_textured_asset(output, mesh)


def test_texturing_must_preserve_face_connectivity(tmp_path):
    mesh = tmp_path / "source.ply"
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    write_ply(mesh, xyz, faces=np.array([[0, 1, 2]]))
    output = tmp_path / "texture"
    output.mkdir()
    write_ply(output / "mesh.ply", xyz, faces=np.array([[0, 2, 1]]),
              uvs=np.array([[[0, 0], [1, 0], [0, 1]]]), texture_file="texture.png")
    Image.new("RGB", (8, 8)).save(output / "texture.png")
    with pytest.raises(ValueError, match="topology"):
        validate_textured_asset(output, mesh)


def test_texturing_rejects_tiny_uv_coverage(tmp_path):
    mesh = tmp_path / "source.ply"
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.repeat(np.array([[0, 1, 2]]), 4, axis=0)
    write_ply(mesh, xyz, faces=faces)
    output = tmp_path / "texture"
    output.mkdir()
    uvs = np.zeros((4, 3, 2), dtype=float)
    uvs[0] = [[0, 0], [1, 0], [0, 1]]
    write_ply(output / "mesh.ply", xyz, faces=faces, uvs=uvs, texture_file="texture.png")
    Image.new("RGB", (8, 8)).save(output / "texture.png")
    with pytest.raises(ValueError, match="substantial"):
        validate_textured_asset(output, mesh)


def test_texturing_rejects_uvs_outside_normalized_atlas_range(tmp_path):
    mesh = tmp_path / "source.ply"
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]])
    write_ply(mesh, xyz, faces=faces)
    output = tmp_path / "texture"
    output.mkdir()
    uvs = np.array([[[0, 0], [1.01, 0], [0, 1]]], dtype=float)
    write_ply(output / "mesh.ply", xyz, faces=faces, uvs=uvs, texture_file="texture.png")
    Image.new("RGB", (8, 8)).save(output / "texture.png")
    with pytest.raises(ValueError, match="normalized atlas range"):
        validate_textured_asset(output, mesh)


def test_metrics_report_robust_bounds_and_reference_coverage(tmp_path):
    cloud = tmp_path / "cloud.ply"
    rng = np.random.default_rng(4213)
    xyz = rng.uniform(-1, 1, size=(1000, 3))
    xyz = np.vstack(([-100, -100, 100], xyz))
    write_ply(cloud, xyz)
    metrics = ply_metrics(cloud, reference_bounds={"bounding_box_min": [-1, -1, -1],
                                                   "bounding_box_max": [1, 1, 1]})
    assert metrics["bounding_box_max"][2] == 100
    assert max(metrics["robust_bounding_box_max"]) <= 1
    assert metrics["reference_expanded_fraction"] > .99


def test_component_cleanup_uses_face_share_rule_and_remaps_vertices():
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                    [10, 0, 0], [11, 0, 0], [10, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]] * 199 + [[3, 4, 5]])
    cleaned, report = filter_small_components({"xyz": xyz, "faces": faces,
                                                "colors": None, "normals": None}, .01)
    assert report["minimum_faces"] == 2
    assert report["kept_components"] == 1 and report["removed_components"] == 1
    assert report["retained_face_fraction"] == 199 / 200
    assert cleaned["xyz"].shape == (3, 3)
    assert cleaned["faces"].max() == 2


def test_nonfinite_geometry_and_truncated_binary_are_rejected(tmp_path):
    path = tmp_path / "nan.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\nnan 0 0\n", encoding="ascii")
    with pytest.raises(ValueError, match="finite"):
        read_ply(path)
    truncated = tmp_path / "truncated.ply"
    truncated.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n" + b"\x00")
    with pytest.raises(ValueError, match="truncated"):
        read_ply(truncated)
