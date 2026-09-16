from __future__ import annotations

from pathlib import Path

import pytest

from v4_config import StageStateStore, sha256_file, validate_output_path


def _roots(tmp_path: Path):
    writable = (tmp_path / "capture_v4", tmp_path / "reconstruction" / "v4")
    protected = (
        tmp_path / "CSX4213_Project_V4_Images",
        tmp_path / ".codegraph",
    )
    return writable, protected


def test_output_boundary_resolves_traversal_and_protects_raw_roots(tmp_path: Path):
    writable, protected = _roots(tmp_path)
    good = validate_output_path(
        tmp_path / "capture_v4" / "manifests" / "source_manifest.csv",
        project_root=tmp_path,
        writable_roots=writable,
        protected_roots=protected,
    )
    assert good == (tmp_path / "capture_v4" / "manifests" / "source_manifest.csv").resolve()

    bad = (
        tmp_path / "capture_v4" / ".." / "CSX4213_Project_V4_Images" / "copy.jpg",
        tmp_path / ".codegraph" / "cache.json",
        tmp_path / "unapproved" / "result.ply",
        tmp_path,
    )
    for candidate in bad:
        with pytest.raises(ValueError):
            validate_output_path(
                candidate,
                project_root=tmp_path,
                writable_roots=writable,
                protected_roots=protected,
            )


def test_stage_state_reuses_identity_and_invalidates_only_downstream(tmp_path: Path):
    writable, protected = _roots(tmp_path)
    state_path = tmp_path / "reconstruction" / "v4" / "work" / "stage_state.json"
    store = StageStateStore(
        state_path,
        project_root=tmp_path,
        writable_roots=writable,
        protected_roots=protected,
    )
    output = tmp_path / "capture_v4" / "manifest.json"
    output.parent.mkdir(parents=True)
    output.write_text("{}\n", encoding="utf-8")
    store.begin("manifest", input_hashes={"source": "abc"}, config={"version": 1}, tools={"tool": "x"})
    store.complete("manifest", output_paths=[output], output_hashes={str(output.resolve()): sha256_file(output)})
    assert store.is_reusable("manifest", input_hashes={"source": "abc"}, config={"version": 1})

    store.begin("isolation", input_hashes={"manifest": "abc"}, config={"version": 1})
    store.complete("isolation", output_paths=[], details={"ok": True})
    invalidated = store.invalidate_from("isolation", reason="mask_identity_changed")
    assert invalidated[0] == "isolation"
    assert "manifest" not in invalidated
    payload = store.load()
    assert payload["stages"]["manifest"]["status"] == "complete"
    assert payload["stages"]["isolation"]["status"] == "pending"
    assert payload["stages"]["matching"]["status"] == "pending"
