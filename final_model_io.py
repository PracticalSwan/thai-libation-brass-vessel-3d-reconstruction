"""Safe file helpers for the final V2 reconstruction workflow."""

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from collections.abc import Iterable, Mapping, Sequence


V2_RELATIVE_ROOT = Path("reconstruction/reference_assisted_v2")

REQUIRED_PROMOTION_REPORTS = (
    "final_cv_fit.json",
    "base_geometry_report.json",
    "ornament_build_report.json",
    "cleanup_report.json",
    "uv_bake_report.json",
    "texture_projection_report.json",
    "lookdev_report.json",
    "final_validation_report.json",
)


def ensure_under_v2_root(path: Path, v2_root: Path) -> Path:
    """Resolve *path* and reject outputs outside the owned V2 directory."""

    resolved_root = v2_root.resolve()
    resolved_path = path.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise ValueError(f"output path is outside V2 root: {resolved_path}")
    return resolved_path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file without changing it."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text_path(path: Path, v2_root: Path) -> tuple[Path, Path]:
    resolved = ensure_under_v2_root(path, v2_root)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=resolved.parent,
        prefix=f".{resolved.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return resolved, Path(temporary_name)


def write_json_atomic(path: Path, payload: object, v2_root: Path) -> None:
    """Write deterministic JSON through a same-directory temporary file."""

    resolved, temporary = _atomic_text_path(path, v2_root)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(resolved)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv_atomic(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
    v2_root: Path,
) -> None:
    """Write deterministic CSV through a same-directory temporary file."""

    resolved, temporary = _atomic_text_path(path, v2_root)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(resolved)
    finally:
        temporary.unlink(missing_ok=True)


def owned_work_path(path: Path, v2_root: Path) -> bool:
    """Return whether *path* belongs to the V2 transient work subtree."""

    work_root = (v2_root.resolve() / "work").resolve()
    resolved = path.resolve()
    return resolved == work_root or work_root in resolved.parents


def _promotion_report_accepted(name: str, payload: Mapping[str, object]) -> bool:
    """Return whether one report is truthful and sufficient for final promotion."""

    if payload.get("accepted") is True:
        return True
    if name != "texture_projection_report.json":
        return False

    # Plan 5 reached a disclosed component-level photo-informed fallback rather
    # than exact per-texel projection. That is sufficient for the accepted final
    # lookdev/material asset only when the report preserves the exact limitation;
    # it must never be promoted as direct texture projection.
    fallback = payload.get("fallback")
    blocked = payload.get("blocked_reasons")
    try:
        direct_percent = float(payload.get("direct_projection_percent", -1.0))
        inferred_percent = float(payload.get("inferred_fill_percent", -1.0))
    except (TypeError, ValueError):
        return False
    return (
        isinstance(fallback, Mapping)
        and fallback.get("active") is True
        and fallback.get("texel_direct_projection") is False
        and payload.get("projection_method") == "component_level_photo_informed_fusion_fallback"
        and payload.get("claim_scope") == "photo_informed_component_fusion_fallback_no_visual_qa"
        and direct_percent == 0.0
        and inferred_percent == 100.0
        and isinstance(blocked, list)
        and "per_texel_depth_masked_projection_not_run" in blocked
        and "component_level_photo_informed_fallback_only" in blocked
    )


def required_reports_ready(v2_root: Path) -> tuple[Path, ...]:
    """Return truthful promotion-ready reports or fail closed on the first gap."""

    root = v2_root.resolve()
    reports = root / "reports"
    ready: list[Path] = []
    for name in REQUIRED_PROMOTION_REPORTS:
        path = reports / name
        if not path.is_file():
            raise ValueError(f"required promotion report is missing: {name}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"required promotion report is unreadable: {name}") from exc
        if not _promotion_report_accepted(name, payload):
            raise ValueError(f"required promotion report {name} is not accepted")
        ready.append(path)
    return tuple(ready)


def build_file_manifest(path: Path, v2_root: Path) -> dict[str, object]:
    """Return a stable hash/size record for one owned final artifact."""

    resolved = ensure_under_v2_root(path, v2_root)
    if not resolved.is_file():
        raise ValueError(f"manifest artifact is missing: {resolved}")
    return {
        "path": resolved.relative_to(v2_root.resolve()).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
