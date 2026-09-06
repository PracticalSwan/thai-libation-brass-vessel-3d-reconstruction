"""Pure contracts for the headless Blender validation renderer."""
import json
import numpy as np
import pytest

from render_local_reconstruction import (
    camera_matrix_world,
    load_geometry_contract,
    load_camera_contract,
)


def test_opencv_pose_maps_to_blender_camera_axes():
    pose = np.eye(4, dtype=float)
    matrix = camera_matrix_world(pose)
    expected = np.diag([1., -1., -1., 1.])
    assert np.allclose(matrix, expected)
    pose[:3, 3] = [1., 2., 3.]
    translated = camera_matrix_world(pose)
    assert np.allclose(translated[:3, 3], [-1., -2., -3.])


def test_geometry_contract_rejects_nontriangles_and_nonfinite(tmp_path):
    path = tmp_path / "geometry.npz"
    np.savez(path, xyz=np.zeros((3, 3)), faces=np.array([[0, 1, 2]], dtype=np.int32), uvs=np.zeros((1, 3, 2)))
    data = load_geometry_contract(path)
    assert data["xyz"].shape == (3, 3) and data["faces"].shape == (1, 3)
    np.savez(path, xyz=np.full((3, 3), np.nan), faces=np.array([[0, 1, 2]], dtype=np.int32))
    with pytest.raises(ValueError, match="finite"):
        load_geometry_contract(path)
    np.savez(path, xyz=np.zeros((4, 3)), faces=np.array([[0, 1, 3, 2]], dtype=np.int32))
    with pytest.raises(ValueError, match="triang"):
        load_geometry_contract(path)


def test_camera_contract_requires_calibrated_fields(tmp_path):
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps([{"name": "a.jpg", "width": 10, "height": 8,
                                 "params": [5, 5, 5, 4], "cam_from_world": np.eye(4)[:3].tolist()}]), encoding="utf-8")
    cameras = load_camera_contract(path)
    assert cameras[0]["name"] == "a.jpg"
    path.write_text(json.dumps([{"name": "a.jpg", "width": 10, "height": 8}]), encoding="utf-8")
    with pytest.raises(ValueError, match="camera"):
        load_camera_contract(path)
