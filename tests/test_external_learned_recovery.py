from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pycolmap
import pytest
import torch

from analysis_common import SelectedImageRecord
import external_learned_recovery as external
from sparse_bridging import BridgePairMetrics, BridgeSearchConfig
from sparse_reconstruction import ModelMetrics, SparseRunConfig


def _records(count: int = 288) -> tuple[SelectedImageRecord, ...]:
    return tuple(
        SelectedImageRecord(
            index=index,
            filename=f"image{index:03d}.jpg",
            variant="PREPROCESSED",
            width=3072,
            height=4080,
            size_bytes=100,
            sha256=f"{index:064x}"[-64:],
            decision="ACCEPT",
            reasons="",
        )
        for index in range(1, count + 1)
    )


def _features(count: int = 3) -> external.ImageFeatures:
    return external.ImageFeatures(
        keypoints=np.asarray([[10.0 + i, 20.0 + i] for i in range(count)], dtype=np.float32),
        descriptors=np.ones((count, 128), dtype=np.float32),
        scores=np.linspace(1.0, 0.5, count, dtype=np.float32),
        image_size=np.asarray([3072.0, 4080.0], dtype=np.float32),
    )


def _model(path: str, *, registered: int = 73, points: int = 6099, error: float = 1.2) -> ModelMetrics:
    return ModelMetrics(
        model_path=Path(path),
        registered_images=registered,
        total_images=288,
        sparse_points=points,
        observations=max(points * 3, 1),
        mean_track_length=3.0,
        mean_reprojection_error=error,
        camera_count=1,
        camera_model="SIMPLE_RADIAL",
        camera_params=(3069.0, 1536.0, 2040.0, 0.0),
    )


def test_config_freezes_one_pinned_external_frontend():
    config = external.ExternalLearnedConfig()

    assert external.PINNED_LIGHTGLUE_COMMIT == "eb42fee2d71449efb0aa5c10549752b5d75384d8"
    assert config.frontend_name == "aliked_lightglue"
    assert config.aliked_model_name == "aliked-n16rot"
    assert config.max_image_size == 1600
    assert config.max_keypoints == 4096
    assert config.sequential_overlap == 20


def test_resolve_device_prefers_cuda_only_when_available():
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

    class FakeTorch:
        cuda = FakeCuda()

    assert external.resolve_device(FakeTorch()) == "cuda"

    FakeCuda.is_available = staticmethod(lambda: False)
    assert external.resolve_device(FakeTorch()) == "cpu"


def test_image_features_are_strict_and_colmap_origin_is_plus_half_pixel():
    features = _features(3)
    external.validate_image_features(features)

    keypoints = external.colmap_keypoints(features)
    np.testing.assert_allclose(keypoints, features.keypoints + 0.5)
    assert keypoints.dtype == np.float32

    with pytest.raises(ValueError, match="descriptor count"):
        external.validate_image_features(
            replace(features, descriptors=np.ones((2, 128), dtype=np.float32))
        )
    with pytest.raises(ValueError, match="finite"):
        external.validate_image_features(
            replace(features, keypoints=np.asarray([[np.nan, 0]], dtype=np.float32), descriptors=np.ones((1, 128), dtype=np.float32), scores=np.ones(1, dtype=np.float32))
        )


def test_validate_matches_requires_n_by_two_uint_indices_in_bounds():
    matches = external.validate_matches(np.asarray([[0, 1], [2, 0]]), 3, 2)
    assert matches.dtype == np.uint32
    assert matches.shape == (2, 2)

    with pytest.raises(ValueError, match="N x 2"):
        external.validate_matches(np.asarray([0, 1, 2]), 3, 3)
    with pytest.raises(ValueError, match="out of bounds"):
        external.validate_matches(np.asarray([[3, 0]]), 3, 3)
    with pytest.raises(ValueError, match="nonnegative"):
        external.validate_matches(np.asarray([[-1, 0]]), 3, 3)


def test_sequential_pair_schedule_is_deterministic_and_exactly_5550():
    pairs = external.generate_sequential_pairs(_records(), overlap=20)

    assert len(pairs) == 5550
    assert pairs[0] == ("image001.jpg", "image002.jpg")
    assert pairs[-1] == ("image287.jpg", "image288.jpg")
    assert len(set(pairs)) == len(pairs)
    assert all(left < right for left, right in pairs)


def test_bridge_merge_adds_only_new_normalized_pairs():
    sequential = (("a.jpg", "b.jpg"), ("b.jpg", "c.jpg"))
    bridges = (("b.jpg", "a.jpg"), ("a.jpg", "d.jpg"), ("c.jpg", "e.jpg"))

    merged = external.merge_pair_schedule(sequential, bridges)

    assert merged == (
        ("a.jpg", "b.jpg"),
        ("b.jpg", "c.jpg"),
        ("a.jpg", "d.jpg"),
        ("c.jpg", "e.jpg"),
    )


def test_local_fallback_ranking_prefers_registered_then_points_then_error():
    step10 = _model("step10", registered=73, points=6099, error=1.2373)
    step11 = _model("step11", registered=73, points=3443, error=1.1989)

    label, selected = external.choose_local_fallback(("step10", step10), ("step11", step11))
    assert label == "step10"
    assert selected is step10

    improved = replace(step11, registered_images=74)
    assert external.choose_local_fallback(("step10", step10), ("step11", improved))[0] == "step11"


def test_global_acceptance_reuses_frozen_single_model_gate():
    accepted = _model("accepted", registered=274, points=1000, error=0.9)
    assert external.external_model_metric_accepted(accepted)
    assert not external.external_model_metric_accepted(replace(accepted, registered_images=273))
    assert not external.external_model_metric_accepted(replace(accepted, sparse_points=999))
    assert not external.external_model_metric_accepted(replace(accepted, camera_count=2))
    assert not external.external_model_metric_accepted(replace(accepted, camera_model="OPENCV"))
    assert not external.external_model_metric_accepted(replace(accepted, mean_reprojection_error=float("nan")))


def test_feature_cache_round_trip_requires_exact_provenance(tmp_path: Path):
    records = _records(2)
    features = {record.filename: _features(index + 2) for index, record in enumerate(records)}
    config = replace(external.ExternalLearnedConfig(), expected_images=2)

    external.write_feature_cache(
        tmp_path,
        records,
        features,
        selection_manifest_sha256="manifest-sha",
        config=config,
        torch_version="2.13.0+cu130",
        device="cuda",
    )

    loaded = external.load_feature_cache(
        tmp_path,
        records,
        selection_manifest_sha256="manifest-sha",
        config=config,
        torch_version="2.13.0+cu130",
        device="cuda",
    )
    assert loaded is not None
    assert tuple(loaded) == tuple(record.filename for record in records)
    np.testing.assert_allclose(loaded[records[0].filename].keypoints, features[records[0].filename].keypoints)

    assert external.load_feature_cache(
        tmp_path,
        records,
        selection_manifest_sha256="changed",
        config=config,
        torch_version="2.13.0+cu130",
        device="cuda",
    ) is None
    assert not (tmp_path / "features_complete.json.tmp").exists()


def test_feature_cache_rejects_non_source_coordinate_frame(tmp_path: Path):
    records = _records(2)
    bad = replace(
        _features(3),
        image_size=np.asarray([1600.0, 1200.0], dtype=np.float32),
    )
    features = {records[0].filename: bad, records[1].filename: _features(3)}
    config = replace(external.ExternalLearnedConfig(), expected_images=2)

    with pytest.raises(ValueError, match="source image dimensions"):
        external.write_feature_cache(
            tmp_path,
            records,
            features,
            selection_manifest_sha256="manifest-sha",
            config=config,
            torch_version="2.13.0+cu130",
            device="cuda",
        )


def test_external_database_imports_single_camera_keypoints_matches_and_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    names = ("a.jpg", "b.jpg")
    # import_images only needs readable images; keep fixtures tiny.
    from PIL import Image

    for name in names:
        Image.new("RGB", (32, 24), (128, 128, 128)).save(image_dir / name)

    sparse_config = replace(
        SparseRunConfig(),
        expected_images=2,
        image_width=32,
        image_height=24,
        focal_35mm=26.0,
    )
    feature_map = {name: _features(3) for name in names}
    pair_matches = {(names[0], names[1]): np.asarray([[0, 0], [1, 1]], dtype=np.uint32)}
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    pair_file = work_dir / "pairs.txt"
    database = work_dir / "database.db"
    verified: list[tuple[Path, Path]] = []

    def fake_verify(database_path, pairs_path, options):
        verified.append((Path(database_path), Path(pairs_path)))

    monkeypatch.setattr(external.pycolmap, "verify_matches", fake_verify)

    external.build_verified_match_database(
        image_dir,
        database,
        feature_map,
        pair_matches,
        pair_file,
        sparse_config,
    )

    with pycolmap.Database.open(database) as db:
        assert db.num_cameras() == 1
        assert db.num_images() == 2
        assert db.num_keypoints() == 6
        images = {image.name: image for image in db.read_all_images()}
        matches = db.read_matches(images["a.jpg"].image_id, images["b.jpg"].image_id)
        assert np.asarray(matches).shape == (2, 2)
    assert verified == [(database, pair_file)]
    assert pair_file.read_text(encoding="utf-8") == "a.jpg b.jpg\n"


def test_external_module_does_not_expose_native_exhaustive_or_dense_execution():
    source = Path(external.__file__).read_text(encoding="utf-8")
    assert "match_exhaustive" not in source
    assert "patch_match_stereo" not in source
    assert "stereo_fusion" not in source
