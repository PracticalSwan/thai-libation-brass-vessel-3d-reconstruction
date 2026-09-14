from __future__ import annotations

import numpy as np
import pycolmap
import pytest

from scripts.run_v4_v50_solved_center_native_diagnostic import (
    _build_joint_ba_config,
    _validate_bundle_adjustment_summary,
)


def _tiny_reconstruction() -> pycolmap.Reconstruction:
    reconstruction = pycolmap.Reconstruction()
    camera = pycolmap.Camera.create_from_model_name(
        1, "SIMPLE_PINHOLE", 80.0, 100, 100
    )
    reconstruction.add_camera_with_trivial_rig(camera)
    for image_id in (1, 2):
        image = pycolmap.Image()
        image.image_id = image_id
        image.camera_id = 1
        image.name = f"{image_id}.jpg"
        image.points2D.append(pycolmap.Point2D(np.array([50.0, 50.0])))
        reconstruction.add_image_with_trivial_frame(image, pycolmap.Rigid3d())

    track = pycolmap.Track()
    track.add_element(pycolmap.TrackElement(1, 0))
    track.add_element(pycolmap.TrackElement(2, 0))
    point_id = reconstruction.add_point3D(np.array([0.0, 0.0, 5.0]), track)
    for image_id in (1, 2):
        reconstruction.image(image_id).set_point3D_for_point2D(0, point_id)
    return reconstruction


def test_joint_ba_config_registers_every_registered_image_and_has_residuals() -> None:
    reconstruction = _tiny_reconstruction()

    config, anchor_image_id, residual_count = _build_joint_ba_config(reconstruction)

    assert anchor_image_id == 1
    assert all(config.has_image(image_id) for image_id in reconstruction.reg_image_ids())
    assert config.num_residuals(reconstruction) == residual_count
    assert residual_count > 0


def test_bundle_adjustment_validation_rejects_empty_or_failed_summary() -> None:
    with pytest.raises(RuntimeError, match="nonzero residual"):
        _validate_bundle_adjustment_summary(
            {"termination_type": "BundleAdjustmentTerminationType.FAILURE", "num_residuals": 0},
            configured_residual_count=4,
        )

    with pytest.raises(RuntimeError, match="termination"):
        _validate_bundle_adjustment_summary(
            {"termination_type": "BundleAdjustmentTerminationType.FAILURE", "num_residuals": 4},
            configured_residual_count=4,
        )


def test_bundle_adjustment_validation_accepts_convergence_with_residuals() -> None:
    validated = _validate_bundle_adjustment_summary(
        {"termination_type": "BundleAdjustmentTerminationType.CONVERGENCE", "num_residuals": 4},
        configured_residual_count=4,
    )

    assert validated["termination_type"] == "CONVERGENCE"
    assert validated["num_residuals"] == 4
