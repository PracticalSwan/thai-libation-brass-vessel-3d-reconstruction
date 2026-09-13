"""V4 path, provenance, restart-state, and runtime identity contracts.

The V4 source photographs are a protected evidence boundary.  This module is
deliberately dependency-light so that path and restart checks can run before
CUDA, COLMAP, Blender, or learned-model imports are attempted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
V4_SOURCE_ROOT = PROJECT_ROOT / "CSX4213_Project_V4_Images"
HISTORICAL_RAW_ROOT = PROJECT_ROOT / "IMG20260826122949"
CAPTURE_V4_ROOT = PROJECT_ROOT / "capture_v4"
RECONSTRUCTION_V4_ROOT = PROJECT_ROOT / "reconstruction" / "v4"
V4_STAGE_STATE_PATH = RECONSTRUCTION_V4_ROOT / "work" / "stage_state.json"
CODEGRAPH_ROOT = PROJECT_ROOT / ".codegraph"
PRIVATE_CHECKPOINT_ROOT = PROJECT_ROOT / "analysis" / "ml" / "checkpoints"

V4_WRITABLE_ROOTS = (CAPTURE_V4_ROOT, RECONSTRUCTION_V4_ROOT)
V4_PROTECTED_ROOTS = (
    V4_SOURCE_ROOT,
    HISTORICAL_RAW_ROOT,
    CODEGRAPH_ROOT,
    PRIVATE_CHECKPOINT_ROOT,
)

STAGE_ORDER = (
    "paths",
    "manifest",
    "isolation",
    "matching",
    "pairing",
    "database",
    "sparse",
    "dense_preflight",
    "dense",
    "mesh",
    "blender",
    "production",
    "material",
    "export",
    "verification",
)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp suitable for JSON provenance."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_path(path: str | os.PathLike[str]) -> Path:
    """Resolve a path without requiring it to exist yet."""

    return Path(path).expanduser().resolve(strict=False)


def _is_within(candidate: Path, root: Path) -> bool:
    candidate = candidate.resolve(strict=False)
    root = root.resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def validate_output_path(
    path: str | os.PathLike[str],
    *,
    project_root: Path = PROJECT_ROOT,
    writable_roots: Sequence[Path] | None = None,
    protected_roots: Sequence[Path] | None = None,
) -> Path:
    """Validate a generated-output path against the V4 writable boundary.

    The optional roots make the contract unit-testable in an isolated
    temporary project while the production defaults remain canonical.
    ``resolve`` is performed before every comparison, preventing ``..`` and
    symlink/reparse aliases from bypassing the boundary.
    """

    root = resolve_path(project_root)
    writable = tuple(
        resolve_path(value)
        for value in (writable_roots if writable_roots is not None else (
            root / "capture_v4",
            root / "reconstruction" / "v4",
        ))
    )
    protected = tuple(
        resolve_path(value)
        for value in (protected_roots if protected_roots is not None else (
            root / "CSX4213_Project_V4_Images",
            root / "IMG20260826122949",
            root / ".codegraph",
            root / "analysis" / "ml" / "checkpoints",
        ))
    )
    candidate = resolve_path(path)
    if not any(_is_within(candidate, allowed) for allowed in writable):
        raise ValueError(
            f"V4 output must be inside capture_v4/ or reconstruction/v4/: {candidate}"
        )
    if any(_is_within(candidate, forbidden) for forbidden in protected):
        raise ValueError(f"V4 output overlaps a protected path: {candidate}")
    return candidate


def assert_output_path(path: str | os.PathLike[str]) -> Path:
    """Canonical production alias for :func:`validate_output_path`."""

    return validate_output_path(path)


def ensure_v4_directories() -> dict[str, Path]:
    """Create only the approved V4 derived/output directories."""

    paths = {
        "capture_manifests": CAPTURE_V4_ROOT / "manifests",
        "capture_derived": CAPTURE_V4_ROOT / "derived",
        "capture_previews": CAPTURE_V4_ROOT / "derived" / "previews",
        "reconstruction_reports": RECONSTRUCTION_V4_ROOT / "reports",
        "reconstruction_previews": RECONSTRUCTION_V4_ROOT / "previews",
        "reconstruction_work": RECONSTRUCTION_V4_ROOT / "work",
        "reconstruction_sparse": RECONSTRUCTION_V4_ROOT / "sparse",
        "reconstruction_dense": RECONSTRUCTION_V4_ROOT / "dense",
        "reconstruction_mesh": RECONSTRUCTION_V4_ROOT / "mesh",
        "reconstruction_blender": RECONSTRUCTION_V4_ROOT / "blender",
    }
    for path in paths.values():
        assert_output_path(path).mkdir(parents=True, exist_ok=True)
    return paths


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash a file with bounded memory use."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint(value: Any) -> str:
    """Create a stable SHA-256 fingerprint for JSON-compatible provenance."""

    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256_bytes(encoded)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def write_json(
    path: str | os.PathLike[str],
    payload: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
    writable_roots: Sequence[Path] | None = None,
    protected_roots: Sequence[Path] | None = None,
) -> Path:
    """Atomically write JSON inside the approved V4 output boundary."""

    target = validate_output_path(
        path,
        project_root=project_root,
        writable_roots=writable_roots,
        protected_roots=protected_roots,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _command_identity(command: str, args: Sequence[str]) -> dict[str, Any]:
    executable = shutil.which(command)
    result: dict[str, Any] = {"command": command, "path": executable or "", "status": "missing"}
    if not executable:
        return result
    try:
        completed = subprocess.run(
            [executable, *args],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        result.update(status="error", error_type=type(error).__name__, error=str(error))
        return result
    output = (completed.stdout or completed.stderr or "").strip()
    result.update(
        status="available" if completed.returncode == 0 else "error",
        returncode=int(completed.returncode),
        version=output.splitlines()[0] if output else "",
    )
    return result


def _colmap_subcommand_probe() -> dict[str, Any]:
    """Record whether the installed COLMAP advertises the V4 commands."""

    executable = shutil.which("colmap")
    required = (
        "geometric_verifier",
        "mapper",
        "bundle_adjuster",
        "image_undistorter",
        "patch_match_stereo",
        "stereo_fusion",
        "poisson_mesher",
    )
    result: dict[str, Any] = {name: False for name in required}
    result["status"] = "missing"
    if not executable:
        return result
    try:
        completed = subprocess.run(
            [executable, "-h"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError) as error:
        result.update(status="error", error_type=type(error).__name__, error=str(error))
        return result
    output = f"{completed.stdout}\n{completed.stderr}"
    result.update(
        status="available" if completed.returncode == 0 else "error",
        returncode=int(completed.returncode),
    )
    for name in required:
        result[name] = bool(re.search(rf"\b{re.escape(name)}\b", output))
    return result


def runtime_snapshot() -> dict[str, Any]:
    """Collect measured local runtime/tool identities without installing anything."""

    snapshot: dict[str, Any] = {
        "captured_at": utc_now(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "tools": {
            "exiftool": _command_identity("exiftool", ["-ver"]),
            "ffmpeg": _command_identity("ffmpeg", ["-version"]),
            "colmap": _command_identity("colmap", ["version"]),
            "blender": _command_identity("blender", ["--version"]),
        },
        "colmap_subcommands": _colmap_subcommand_probe(),
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        snapshot["ram"] = {
            "total_bytes": int(memory.total),
            "available_bytes": int(memory.available),
            "used_bytes": int(memory.used),
        }
    except Exception as error:
        snapshot["ram"] = {
            "status": "unavailable",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    try:
        import importlib.metadata

        packages: dict[str, str] = {}
        for name in ("numpy", "Pillow", "opencv-python", "torch", "pycolmap", "lightglue"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = "missing"
        snapshot["packages"] = packages
    except Exception as error:  # pragma: no cover - defensive runtime probe
        snapshot["packages_error"] = f"{type(error).__name__}: {error}"
    try:
        import torch

        cuda: dict[str, Any] = {
            "available": bool(torch.cuda.is_available()),
            "torch_version": str(torch.__version__),
            "runtime": str(torch.version.cuda or ""),
            "device_count": int(torch.cuda.device_count()),
        }
        if cuda["device_count"]:
            cuda["device_0"] = {
                "name": str(torch.cuda.get_device_name(0)),
                "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            }
            try:
                free, total = torch.cuda.mem_get_info(0)
                cuda["free_memory_bytes"] = int(free)
                cuda["total_memory_bytes"] = int(total)
            except Exception as error:  # pragma: no cover - driver-dependent
                cuda["mem_get_info_error"] = f"{type(error).__name__}: {error}"
        snapshot["cuda"] = cuda
    except Exception as error:
        snapshot["cuda"] = {"status": "unavailable", "error": f"{type(error).__name__}: {error}"}
    try:
        import pycolmap

        snapshot["pycolmap"] = str(pycolmap.__version__)
    except Exception as error:
        snapshot["pycolmap"] = {"status": "unavailable", "error": f"{type(error).__name__}: {error}"}
    for name in ("groundingdino", "sam2"):
        try:
            __import__(name)
            snapshot.setdefault("learned_models", {})[name] = "importable"
        except Exception as error:
            snapshot.setdefault("learned_models", {})[name] = {
                "status": "missing",
                "error_type": type(error).__name__,
                "error": str(error),
            }
    return snapshot


def record_runtime_identities(path: str | os.PathLike[str] | None = None) -> Path:
    target = Path(path) if path is not None else RECONSTRUCTION_V4_ROOT / "work" / "runtime_identities.json"
    return write_json(target, runtime_snapshot())


@dataclass
class StageStateStore:
    """Small persistent stage-state store with downstream-only invalidation."""

    path: Path = V4_STAGE_STATE_PATH
    project_root: Path = PROJECT_ROOT
    writable_roots: Sequence[Path] | None = None
    protected_roots: Sequence[Path] | None = None

    def __post_init__(self) -> None:
        self.path = validate_output_path(
            self.path,
            project_root=self.project_root,
            writable_roots=self.writable_roots,
            protected_roots=self.protected_roots,
        )

    def _validate_output(self, path: str | os.PathLike[str]) -> Path:
        return validate_output_path(
            path,
            project_root=self.project_root,
            writable_roots=self.writable_roots,
            protected_roots=self.protected_roots,
        )

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"schema_version": 1, "updated_at": None, "stages": {}}
        payload = read_json(self.path)
        payload.setdefault("schema_version", 1)
        payload.setdefault("stages", {})
        return payload

    def save(self, payload: Mapping[str, Any]) -> None:
        enriched = dict(payload)
        enriched["schema_version"] = 1
        enriched["updated_at"] = utc_now()
        write_json(
            self.path,
            enriched,
            project_root=self.project_root,
            writable_roots=self.writable_roots,
            protected_roots=self.protected_roots,
        )

    def _record(self, stage: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if stage not in STAGE_ORDER:
            raise ValueError(f"unknown V4 stage: {stage}")
        if payload is None:
            payload = self.load()
        stages = payload.setdefault("stages", {})
        value = stages.setdefault(stage, {"stage": stage, "status": "pending"})
        if not isinstance(value, dict):
            raise ValueError(f"invalid state record for stage: {stage}")
        return value

    def begin(
        self,
        stage: str,
        *,
        input_hashes: Mapping[str, str] | None = None,
        config: Any = None,
        tools: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self.load()
        record = self._record(stage, payload)
        record.update(
            {
                "stage": stage,
                "status": "running",
                "input_hashes": dict(input_hashes or {}),
                "config_fingerprint": fingerprint(config) if config is not None else "",
                "tool_model_identities": _json_safe(tools or {}),
                "started_at": utc_now(),
                "completed_at": None,
                "failure_category": None,
                "failure_message": None,
                "retry_decision": "resume_or_run",
            }
        )
        self.save(payload)
        return record

    def complete(
        self,
        stage: str,
        *,
        output_paths: Sequence[str | os.PathLike[str]] = (),
        output_hashes: Mapping[str, str] | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self.load()
        record = self._record(stage, payload)
        resolved = [str(self._validate_output(path)) for path in output_paths]
        record.update(
            {
                "stage": stage,
                "status": "complete",
                "output_paths": resolved,
                "output_hashes": dict(output_hashes or {}),
                "completed_at": utc_now(),
                "failure_category": None,
                "failure_message": None,
                "retry_decision": "reusable_if_identity_matches",
            }
        )
        if details:
            record["details"] = _json_safe(details)
        self.save(payload)
        return record

    def fail(
        self,
        stage: str,
        *,
        category: str,
        message: str,
        retry_decision: str = "diagnose_before_retry",
    ) -> dict[str, Any]:
        payload = self.load()
        record = self._record(stage, payload)
        record.update(
            {
                "stage": stage,
                "status": "failed",
                "failure_category": category,
                "failure_message": message,
                "retry_decision": retry_decision,
                "completed_at": utc_now(),
            }
        )
        self.save(payload)
        return record

    def is_reusable(
        self,
        stage: str,
        *,
        input_hashes: Mapping[str, str] | None = None,
        config: Any = None,
    ) -> bool:
        record = self._record(stage)
        if record.get("status") != "complete":
            return False
        if record.get("input_hashes", {}) != dict(input_hashes or {}):
            return False
        expected_config = fingerprint(config) if config is not None else ""
        if record.get("config_fingerprint", "") != expected_config:
            return False
        paths = record.get("output_paths", [])
        hashes = record.get("output_hashes", {})
        if not isinstance(paths, list) or not isinstance(hashes, dict):
            return False
        for path in paths:
            resolved = self._validate_output(path)
            if not resolved.is_file() and not resolved.is_dir():
                return False
            expected = hashes.get(str(resolved))
            if expected and resolved.is_file() and sha256_file(resolved) != expected:
                return False
        return True

    def invalidate_from(self, stage: str, *, reason: str = "upstream_identity_changed") -> list[str]:
        if stage not in STAGE_ORDER:
            raise ValueError(f"unknown V4 stage: {stage}")
        payload = self.load()
        start = STAGE_ORDER.index(stage)
        invalidated = list(STAGE_ORDER[start:])
        stages = payload.setdefault("stages", {})
        for name in invalidated:
            current = stages.setdefault(name, {"stage": name})
            current.update(
                {
                    "stage": name,
                    "status": "pending",
                    "invalidated_at": utc_now(),
                    "invalidation_reason": reason,
                    "retry_decision": "rerun_after_identity_check",
                }
            )
        self.save(payload)
        return invalidated


__all__ = [
    "CAPTURE_V4_ROOT",
    "HISTORICAL_RAW_ROOT",
    "PROJECT_ROOT",
    "RECONSTRUCTION_V4_ROOT",
    "STAGE_ORDER",
    "StageStateStore",
    "V4_PROTECTED_ROOTS",
    "V4_SOURCE_ROOT",
    "V4_STAGE_STATE_PATH",
    "V4_WRITABLE_ROOTS",
    "assert_output_path",
    "ensure_v4_directories",
    "fingerprint",
    "read_json",
    "record_runtime_identities",
    "runtime_snapshot",
    "sha256_file",
    "utc_now",
    "validate_output_path",
    "write_json",
]
