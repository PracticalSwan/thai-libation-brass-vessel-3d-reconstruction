from __future__ import annotations

from contextlib import closing
import json
from dataclasses import replace
from pathlib import Path
import sqlite3

import numpy as np
import pycolmap
import pytest

from analysis_common import SelectedImageRecord, VerifiedSelectedSet, load_selected_manifest
import learned_sparse_recovery as learned
from sparse_bridging import (
    BridgeCandidate,
    BridgePairMetrics,
    BridgeSearchConfig,
    generate_candidate_pairs,
)
from sparse_reconstruction import DatabaseMetrics, ModelMetrics, SparseRunConfig


def _write_feature_database(
    path: Path,
    *,
    names: tuple[str, ...] = ("a.jpg", "b.jpg"),
    camera_ids: tuple[int, ...] = (1, 1),
    keypoint_rows: tuple[int, ...] = (3, 4),
    descriptor_rows: tuple[int | None, ...] = (3, 4),
) -> None:
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            """
            CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER);
            CREATE TABLE keypoints(image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB);
            CREATE TABLE descriptors(image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB);
            CREATE TABLE matches(pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
            CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
            """
        )
        connection.executemany(
            "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)",
            [
                (index, name, camera_id)
                for index, (name, camera_id) in enumerate(
                    zip(names, camera_ids, strict=True), start=1
                )
            ],
        )
        connection.executemany(
            "INSERT INTO keypoints(image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
            [
                (index, rows, 4, b"k")
                for index, rows in enumerate(keypoint_rows, start=1)
            ],
        )
        connection.executemany(
            "INSERT INTO descriptors(image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
            [
                (index, rows, 128, b"d")
                for index, rows in enumerate(descriptor_rows, start=1)
                if rows is not None
            ],
        )
        connection.commit()


def _records(count: int = 288) -> tuple[SelectedImageRecord, ...]:
    return tuple(
        SelectedImageRecord(
            index=index,
            filename=f"image{index:03d}.jpg",
            variant="PREPROCESSED",
            width=3072,
            height=4080,
            size_bytes=1,
            sha256="0" * 64,
            decision="ACCEPT",
            reasons="",
        )
        for index in range(1, count + 1)
    )


def _bridge_metric(boundary: tuple[int, int]) -> BridgePairMetrics:
    left, right = boundary
    candidate = BridgeCandidate(
        boundary_left=left,
        boundary_right=right,
        left_index=left - 39,
        right_index=right + 1,
        left_filename=f"image{left - 39:03d}.jpg",
        right_filename=f"image{right + 1:03d}.jpg",
    )
    return BridgePairMetrics(candidate, 100, 25, 0.25, True)


def _model(path: Path, *, registered: int = 274) -> ModelMetrics:
    return ModelMetrics(
        model_path=path,
        registered_images=registered,
        total_images=288,
        sparse_points=2000,
        observations=6000,
        mean_track_length=3.0,
        mean_reprojection_error=0.8,
        camera_count=1,
        camera_model="SIMPLE_RADIAL",
        camera_params=(3070.0, 1536.0, 2040.0, 0.0),
    )


def test_frontend_registry_contains_only_the_two_approved_native_paths():
    assert tuple(learned.FRONTENDS) == ("aliked", "loma")
    assert learned.ALIKED_FRONTEND.extractor_type == pycolmap.FeatureExtractorType.ALIKED_N16ROT
    assert learned.ALIKED_FRONTEND.matcher_type == pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE
    assert learned.LOMA_FRONTEND.extractor_type == pycolmap.FeatureExtractorType.LOMA_B
    assert learned.LOMA_FRONTEND.matcher_type == pycolmap.FeatureMatcherType.LOMA_L
    assert learned.ALIKED_FRONTEND.max_image_size == 1600
    assert learned.LOMA_FRONTEND.max_image_size == 1600


@pytest.mark.parametrize("frontend", ("aliked", "loma"))
def test_learned_options_freeze_frontend_identity_and_cpu_policy(frontend: str):
    spec = learned.FRONTENDS[frontend]
    extraction = learned.build_learned_extraction_options(spec)
    matching = learned.build_learned_matching_options(spec)

    assert extraction.type == spec.extractor_type
    assert extraction.max_image_size == 1600
    assert extraction.use_gpu is False
    assert extraction.check() is True
    assert matching.type == spec.matcher_type
    assert matching.use_gpu is False
    assert matching.check() is True


def test_capability_snapshot_is_json_safe_and_records_live_native_contract():
    snapshot = learned.build_capability_snapshot()

    assert snapshot["python_version"]
    assert snapshot["pycolmap_version"] == str(pycolmap.__version__)
    assert snapshot["has_cuda"] is False
    assert snapshot["device_policy"] == "cpu"
    assert "ALIKED_N16ROT" in snapshot["extractor_members"]
    assert "LOMA_B" in snapshot["extractor_members"]
    assert "ALIKED_LIGHTGLUE" in snapshot["matcher_members"]
    assert "LOMA_L" in snapshot["matcher_members"]
    assert snapshot["frontends"]["aliked"]["effective_max_image_size"] == 1600
    assert snapshot["frontends"]["aliked"]["requires_rgb"] is True
    assert snapshot["frontends"]["aliked"]["requires_opengl"] is False
    assert snapshot["frontends"]["loma"]["matching_options_valid"] is True
    json.dumps(snapshot, allow_nan=False)


def test_smoke_frontend_passes_rgb_uint8_arrays_and_validates_match_shape():
    images_seen: list[np.ndarray] = []

    class FakeExtractor:
        def extract_from_uint8_array(self, image: np.ndarray):
            images_seen.append(image)
            return np.zeros((3, 4), dtype=np.float32), np.zeros((3, 128), dtype=np.float32)

    class FakeMatcher:
        def match(self, *args):
            assert len(args) == 4
            return np.asarray([[0, 0], [1, 1]], dtype=np.uint32)

    def extractor_factory(*, options, device):
        assert options.type == pycolmap.FeatureExtractorType.ALIKED_N16ROT
        assert device == pycolmap.Device.cpu
        return FakeExtractor()

    def matcher_factory(*, options, device):
        assert options.type == pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE
        assert device == pycolmap.Device.cpu
        return FakeMatcher()

    image = np.zeros((12, 10, 3), dtype=np.uint8)
    result = learned.smoke_frontend(
        learned.ALIKED_FRONTEND,
        image,
        image.copy(),
        extractor_factory=extractor_factory,
        matcher_factory=matcher_factory,
    )

    assert result == learned.FrontendSmokeResult("aliked", "passed", 3, 3, 2)
    assert len(images_seen) == 2
    assert all(item.dtype == np.uint8 and item.shape == (12, 10, 3) for item in images_seen)


def test_smoke_frontend_records_exact_blocker_when_feature_counts_disagree():
    class FakeExtractor:
        calls = 0

        def extract_from_uint8_array(self, image: np.ndarray):
            self.calls += 1
            descriptor_count = 3 if self.calls == 1 else 2
            return (
                np.zeros((3, 4), dtype=np.float32),
                np.zeros((descriptor_count, 128), dtype=np.float32),
            )

    result = learned.smoke_frontend(
        learned.ALIKED_FRONTEND,
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.uint8),
        extractor_factory=lambda **kwargs: FakeExtractor(),
        matcher_factory=lambda **kwargs: pytest.fail("invalid features must not be matched"),
    )

    assert result.status == "blocked"
    assert result.exception_type == "ValueError"
    assert "keypoint and descriptor counts" in result.exception_message


def test_feature_database_identity_requires_exact_names_camera_and_positive_rows(
    tmp_path: Path,
):
    path = tmp_path / "features.db"
    _write_feature_database(path)

    identity = learned.validate_learned_feature_database(
        path, ("a.jpg", "b.jpg"), expected_camera_ids=(1,)
    )

    assert identity.image_count == 2
    assert identity.feature_count == 7
    assert identity.camera_ids == (1,)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("wrong-name", "image names"),
        ("wrong-camera", "camera IDs"),
        ("missing-descriptor", "missing a keypoint or descriptor"),
        ("row-mismatch", "keypoint and descriptor counts"),
        ("zero-features", "no learned features"),
    ),
)
def test_feature_database_identity_rejects_stale_or_partial_layouts(
    tmp_path: Path, case: str, message: str
):
    path = tmp_path / "features.db"
    kwargs: dict[str, object] = {}
    expected_names = ("a.jpg", "b.jpg")
    if case == "wrong-name":
        kwargs["names"] = ("a.jpg", "other.jpg")
    elif case == "wrong-camera":
        kwargs["camera_ids"] = (1, 2)
    elif case == "missing-descriptor":
        kwargs["descriptor_rows"] = (3, None)
    elif case == "row-mismatch":
        kwargs["descriptor_rows"] = (3, 2)
    elif case == "zero-features":
        kwargs["keypoint_rows"] = (0, 0)
        kwargs["descriptor_rows"] = (0, 0)
    _write_feature_database(path, **kwargs)

    with pytest.raises(RuntimeError, match=message):
        learned.validate_learned_feature_database(
            path, expected_names, expected_camera_ids=(1,)
        )


def test_feature_cache_marker_is_atomic_and_reused_only_with_full_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    manifest = tmp_path / "selection_manifest.csv"
    manifest.write_text("fixture", encoding="utf-8")
    records = _records(2)
    verified = VerifiedSelectedSet(records, "manifest-sha", image_dir)
    monkeypatch.setattr(learned, "verify_selected_images", lambda *args, **kwargs: verified)
    calls = 0

    def extractor(image_path, database_path, frontend, sparse_config):
        nonlocal calls
        calls += 1
        assert frontend is learned.ALIKED_FRONTEND
        _write_feature_database(
            database_path,
            names=(records[0].filename, records[1].filename),
        )
        return DatabaseMetrics(2, 7, 0, 0)

    sparse_config = replace(SparseRunConfig(), expected_images=2)
    database = learned.prepare_learned_feature_cache(
        image_dir,
        manifest,
        tmp_path / "work",
        learned.ALIKED_FRONTEND,
        sparse_config,
        feature_extractor=extractor,
    )
    marker_path = tmp_path / "work" / "aliked" / "features_complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))

    assert database == tmp_path / "work" / "aliked" / "features.db"
    assert calls == 1
    assert marker["frontend"] == "aliked"
    assert marker["extractor_type"] == "ALIKED_N16ROT"
    assert marker["matcher_type"] == "ALIKED_LIGHTGLUE"
    assert marker["device"] == "cpu"
    assert marker["camera_ids"] == [1]
    assert marker["selection_manifest_sha256"] == "manifest-sha"
    assert not marker_path.with_suffix(".json.tmp").exists()

    reused = learned.prepare_learned_feature_cache(
        image_dir,
        manifest,
        tmp_path / "work",
        learned.ALIKED_FRONTEND,
        sparse_config,
        feature_extractor=lambda *args: pytest.fail("valid cache must be reused"),
    )
    assert reused == database


def test_feature_cache_rejects_changed_per_image_layout_with_same_total_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    manifest = tmp_path / "selection_manifest.csv"
    manifest.write_text("fixture", encoding="utf-8")
    records = _records(2)
    verified = VerifiedSelectedSet(records, "manifest-sha", image_dir)
    monkeypatch.setattr(learned, "verify_selected_images", lambda *args, **kwargs: verified)
    calls = 0

    def extractor(image_path, database_path, frontend, sparse_config):
        nonlocal calls
        calls += 1
        _write_feature_database(
            database_path,
            names=(records[0].filename, records[1].filename),
            keypoint_rows=(3, 4),
            descriptor_rows=(3, 4),
        )
        return DatabaseMetrics(2, 7, 0, 0)

    sparse_config = replace(SparseRunConfig(), expected_images=2)
    database = learned.prepare_learned_feature_cache(
        image_dir,
        manifest,
        tmp_path / "work",
        learned.ALIKED_FRONTEND,
        sparse_config,
        feature_extractor=extractor,
    )

    connection = sqlite3.connect(database)
    try:
        connection.execute("UPDATE keypoints SET rows = 4 WHERE image_id = 1")
        connection.execute("UPDATE descriptors SET rows = 4 WHERE image_id = 1")
        connection.execute("UPDATE keypoints SET rows = 3 WHERE image_id = 2")
        connection.execute("UPDATE descriptors SET rows = 3 WHERE image_id = 2")
        connection.commit()
    finally:
        connection.close()

    learned.prepare_learned_feature_cache(
        image_dir,
        manifest,
        tmp_path / "work",
        learned.ALIKED_FRONTEND,
        sparse_config,
        feature_extractor=extractor,
    )

    assert calls == 2


def test_step12_candidates_exactly_match_the_authoritative_step11_order():
    root = Path(__file__).resolve().parents[1]
    records = load_selected_manifest(root / "preprocessing" / "reports" / "selection_manifest.csv")
    candidates = generate_candidate_pairs(records, BridgeSearchConfig())

    learned.verify_step11_candidate_identity(
        candidates,
        root / "reconstruction" / "bridging" / "reports" / "step11_candidates.csv",
    )

    assert len(candidates) == 2340
    assert all(candidate.sequence_gap >= 41 for candidate in candidates)
    assert {
        boundary: sum(
            (candidate.boundary_left, candidate.boundary_right) == boundary
            for candidate in candidates
        )
        for boundary in BridgeSearchConfig().boundaries
    } == {(73, 74): 780, (145, 146): 780, (203, 204): 780}


def test_candidate_identity_rejects_a_single_changed_pair(tmp_path: Path):
    csv_path = tmp_path / "step11.csv"
    csv_path.write_text(
        "boundary,boundary_left,boundary_right,left_index,right_index,left_filename,right_filename\n"
        "73-74,73,74,34,75,a.jpg,b.jpg\n",
        encoding="utf-8",
    )
    changed = BridgeCandidate(73, 74, 34, 76, "a.jpg", "c.jpg")

    with pytest.raises(RuntimeError, match="candidate identity/order"):
        learned.verify_step11_candidate_identity((changed,), csv_path)


def test_diagnostics_use_the_frontend_matcher_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    features = tmp_path / "features.db"
    features.write_bytes(b"features")
    workspace = tmp_path / "work" / "aliked"
    candidate = BridgeCandidate(73, 74, 34, 75, "a.jpg", "b.jpg")
    expected = (BridgePairMetrics(candidate, 100, 25, 0.25, True),)
    calls: list[str] = []

    def match_image_pairs(**kwargs):
        calls.append("match")
        assert kwargs["matching_options"].type == pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE
        assert kwargs["matching_options"].use_gpu is False
        assert kwargs["device"] == pycolmap.Device.cpu
        assert Path(str(kwargs["pairing_options"].match_list_path)) == workspace / "diagnostic_pairs.txt"

    monkeypatch.setattr(learned.pycolmap, "match_image_pairs", match_image_pairs)
    monkeypatch.setattr(learned, "summarize_bridge_pairs", lambda *args: expected)

    result = learned.run_learned_diagnostics(
        features, workspace, (candidate,), learned.ALIKED_FRONTEND
    )

    assert result == expected
    assert calls == ["match"]


@pytest.mark.parametrize(
    ("frontend_name", "matcher_type"),
    (
        ("aliked", pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE),
        ("loma", pycolmap.FeatureMatcherType.LOMA_L),
    ),
)
def test_targeted_attempt_uses_explicit_learned_matcher_for_both_match_phases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frontend_name,
    matcher_type,
):
    frontend = learned.FRONTENDS[frontend_name]
    features = tmp_path / "features.db"
    features.write_bytes(b"features")
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / frontend.name
    selected = tuple(_bridge_metric(boundary) for boundary in BridgeSearchConfig().boundaries)
    model = _model(output_dir / "0", registered=200)
    calls: list[str] = []

    def sequential(**kwargs):
        calls.append("sequential")
        assert kwargs["matching_options"].type == matcher_type
        assert kwargs["matching_options"].use_gpu is False
        assert kwargs["pairing_options"].overlap == 20
        assert kwargs["pairing_options"].quadratic_overlap is True
        assert kwargs["pairing_options"].loop_detection is False
        assert kwargs["device"] == pycolmap.Device.cpu

    def imported(**kwargs):
        calls.append("imported")
        assert kwargs["matching_options"].type == matcher_type
        assert kwargs["matching_options"].use_gpu is False
        assert kwargs["device"] == pycolmap.Device.cpu
        assert Path(str(kwargs["pairing_options"].match_list_path)) == (
            tmp_path / "work" / frontend.name / "targeted_bridge_pairs.txt"
        )

    def mapping(database_path, actual_image_dir, actual_output_dir, config):
        calls.append("mapping")
        assert database_path == tmp_path / "work" / frontend.name / "targeted.db"
        assert actual_image_dir == image_dir
        assert actual_output_dir == output_dir
        return (model,)

    monkeypatch.setattr(learned.pycolmap, "match_sequential", sequential)
    monkeypatch.setattr(learned.pycolmap, "match_image_pairs", imported)
    monkeypatch.setattr(learned, "map_sparse_database", mapping)
    monkeypatch.setattr(
        learned, "summarize_database", lambda path: DatabaseMetrics(288, 1000, 500, 400)
    )

    attempt = learned.run_learned_targeted_attempt(
        image_dir, features, output_dir, selected, frontend
    )

    assert calls == ["sequential", "imported", "mapping"]
    assert attempt.name == f"{frontend.name}_targeted"
    assert attempt.best_model == model


def test_learned_metric_acceptance_requires_every_single_model_condition():
    accepted = _model(Path("model"))

    assert learned.learned_model_metric_accepted(accepted)
    assert not learned.learned_model_metric_accepted(replace(accepted, registered_images=273))
    assert not learned.learned_model_metric_accepted(replace(accepted, sparse_points=999))
    assert not learned.learned_model_metric_accepted(replace(accepted, camera_count=2))
    assert not learned.learned_model_metric_accepted(replace(accepted, camera_model="OPENCV"))
    assert not learned.learned_model_metric_accepted(
        replace(accepted, mean_reprojection_error=float("nan"))
    )
