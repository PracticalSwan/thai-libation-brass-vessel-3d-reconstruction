"""Small deterministic contracts for the local Step 10 dense-to-texture flow.

No sparse reconstruction or automatic tool download is performed by this module.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np

SOURCE = Path("reconstruction/sparse/best")
OUTPUT = Path("reconstruction/local_dense")
SELECTION_HASH = "79408d59b022803e1acc42d4c1e118c209a8120e76e689ea9b60909015f37a91"
DENSE_COMMANDS = {"image_undistorter", "patch_match_stereo", "stereo_fusion",
                  "poisson_mesher", "delaunay_mesher", "mesh_simplifier", "mesh_texturer"}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, payload: dict) -> None:
    """Serialize before touching the prior report, then replace atomically."""
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(data, encoding="utf-8", newline="\n")
    temporary.replace(path)


def validate_source(root: Path, path: Path, metrics) -> None:
    if path.resolve() != (root / SOURCE).resolve():
        raise ValueError("Only the frozen Step 10 source is permitted")
    expected = (73, 6099, 21351, 1, "SIMPLE_RADIAL")
    actual = (metrics.registered_images, metrics.sparse_points, metrics.observations,
              metrics.camera_count, metrics.camera_model)
    calibration = (3542.7206959261907, 1536., 2040., -0.01641661056124677)
    if (actual != expected
            or not math.isclose(metrics.mean_reprojection_error, 1.2373052447638215, rel_tol=0, abs_tol=1e-10)
            or len(metrics.camera_params) != 4
            or not np.allclose(metrics.camera_params, calibration, rtol=0, atol=1e-10)):
        raise ValueError("Step 10 source metrics/calibration differ from the frozen baseline")


def registered_manifest(records, names, expected_count: int = 73) -> list[dict]:
    names = list(names)
    if len(names) != expected_count or len(set(names)) != expected_count:
        raise ValueError("Registered names must be unique and have the expected count")
    rows = [dict(selected_index=r.index, filename=r.filename, sha256=r.sha256,
                 size_bytes=r.size_bytes) for r in records if r.filename in set(names)]
    if len(rows) != expected_count:
        raise ValueError("Registered view missing from the selected manifest")
    return rows


def names_hash(rows: list[dict]) -> str:
    return hashlib.sha256(("\n".join(r["filename"] for r in rows) + "\n").encode("utf-8")).hexdigest()


def safe_output(root: Path, path: Path) -> Path:
    """Reject traversal and every existing junction/symlink on an output path."""
    root = root.absolute()
    path = path.absolute()
    allowed = root / OUTPUT
    if path == allowed or not path.is_relative_to(allowed):
        raise ValueError("Output must be a child of reconstruction/local_dense")
    for parent in (path, *path.parents):
        if parent.is_symlink() or (parent.exists() and parent.is_junction()):
            raise ValueError(f"Linked output boundary is forbidden: {parent}")
        if parent == root:
            break
    resolved = path.resolve()
    if resolved == allowed.resolve() or not resolved.is_relative_to(allowed.resolve()):
        raise ValueError("Output escapes the local reconstruction boundary")
    return resolved


def cleanup_target(root: Path, path: Path) -> Path:
    path = safe_output(root, path)
    work = (root / OUTPUT / "work").resolve()
    if path == work or not path.is_relative_to(work):
        raise ValueError("Cleanup is restricted to explicit children of local work")
    return path


def colmap_command(executable: Path, command: str, options: dict) -> list[str]:
    """Use the executable directly: never pass data through a Windows batch shell."""
    if executable.suffix.lower() in {".bat", ".cmd"}:
        raise ValueError("Use colmap.exe directly with its library environment")
    if command not in DENSE_COMMANDS:
        raise ValueError("Command outside the approved dense command surface")
    args = [str(executable), command]
    for name, value in options.items():
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_." for c in name):
            raise ValueError("Invalid option name")
        value = str(int(value)) if isinstance(value, bool) else str(value)
        if any(c in value for c in "\x00\r\n"):
            raise ValueError("Invalid command argument")
        args.extend(["--" + name, value])
    return args


def process_tree_pids(root_pid: int) -> list[int]:
    """Return a live Windows process tree without reading command lines."""
    root_pid = int(root_pid)
    if root_pid <= 0:
        return []
    if os.name != "nt":
        try:
            os.kill(root_pid, 0)
        except OSError:
            return []
        return [root_pid]

    import ctypes
    from ctypes import wintypes

    snapshot_flag = 0x00000002
    invalid_handle = ctypes.c_void_p(-1).value

    class ProcessEntry(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_void_p),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    snapshot = kernel32.CreateToolhelp32Snapshot(snapshot_flag, 0)
    if snapshot == invalid_handle:
        raise OSError(ctypes.get_last_error(), "Could not enumerate Windows processes")
    parent_by_pid = {}
    try:
        entry = ProcessEntry()
        entry.dwSize = ctypes.sizeof(entry)
        ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            parent_by_pid[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)

    live = []
    frontier = [root_pid]
    seen = set()
    while frontier:
        parent = frontier.pop()
        if parent in seen:
            continue
        seen.add(parent)
        if parent in parent_by_pid:
            live.append(parent)
        frontier.extend(pid for pid, parent_pid in parent_by_pid.items()
                        if parent_pid == parent and pid not in seen)
    return sorted(live)


def terminate_process_tree(process) -> None:
    """Terminate the launched process and all descendants before returning."""
    if os.name == "nt":
        known_pids = process_tree_pids(process.pid)
        targets = [process.pid] if process.pid in known_pids else []
        targets.extend(pid for pid in reversed(known_pids) if pid != process.pid)
        for pid in targets:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW, timeout=30)
    elif process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_command(args: list[str], log_path: Path, *, timeout: float = 43200,
                env: dict | None = None, on_start=None) -> dict:
    """Capture a capped 32 MiB log and bounded tail, including real process failures."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    tail = deque(maxlen=160)
    status = "completed"
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, env=env, shell=False,
                                   creationflags=flags)
    except OSError as error:
        return dict(args=args, started_at=started, runtime_seconds=time.monotonic()-start,
                    returncode=None, status="failed", output_tail=str(error), log_path=str(log_path))

    try:
        if on_start is not None:
            on_start(process.pid)
    except BaseException:
        terminate_process_tree(process)
        process.stdout.close()
        raise

    def consume():
        size = 0
        with log_path.open("wb") as log:
            for line in iter(process.stdout.readline, b""):
                tail.append(line.decode("utf-8", errors="replace"))
                if size < 32 * 1024 * 1024:
                    log.write(line[:32 * 1024 * 1024-size])
                    log.flush()
                    size += len(line)

    reader = threading.Thread(target=consume, daemon=True)
    reader.start()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        status = "timeout"
        terminate_process_tree(process)
    except BaseException:
        terminate_process_tree(process)
        raise
    finally:
        reader.join(timeout=10)
        process.stdout.close()
    if status != "timeout" and process.returncode != 0:
        status = "failed"
    return dict(args=args, started_at=started, runtime_seconds=time.monotonic()-start,
                returncode=process.returncode, status=status, output_tail="".join(tail)[-24000:],
                log_path=str(log_path), log_sha256=sha256(log_path))


def failure_category(result: dict) -> str | None:
    if result["status"] == "completed":
        return None
    message = result.get("output_tail", "").lower()
    if any(word in message for word in ("out of memory", "cudaerrormemoryallocation", "launch timed out", "tdr", "cudaerrormemory", "bad_alloc")):
        return "resource"
    if any(word in message for word in ("no kernel image", "requires cuda", "cuda driver version", "not compiled", "unknown command")):
        return "capability"
    return "timeout" if result["status"] == "timeout" else "subprocess"


def dense_fallback(attempts: list[dict]) -> dict:
    allowed_statuses = {"failed", "timeout", "interrupted"}
    allowed_categories = {"resource", "timeout", "runtime"}
    if (len(attempts) != 1 or attempts[0]["status"] not in allowed_statuses
            or attempts[0]["failure_category"] not in allowed_categories):
        raise ValueError("Only one measured resource/runtime failure permits one dense fallback")
    config = dict(attempts[0]["config"])
    config["max_image_size"] = 1600 if config["max_image_size"] > 1600 else 1200
    return config


def plausible_bounds(metrics: dict, reference: dict) -> bool:
    low = np.asarray(metrics.get("robust_bounding_box_min", metrics["bounding_box_min"]))
    high = np.asarray(metrics.get("robust_bounding_box_max", metrics["bounding_box_max"]))
    ref_low = np.asarray(reference.get("robust_bounding_box_min", reference["bounding_box_min"]))
    ref_high = np.asarray(reference.get("robust_bounding_box_max", reference["bounding_box_max"]))
    span = ref_high-ref_low
    coverage = float(metrics.get("reference_expanded_fraction", 1.0))
    return bool(np.all(np.isfinite([low, high])) and np.all(span > 0) and coverage >= .99
                and np.all(high > low) and np.all(low >= ref_low-span)
                and np.all(high <= ref_high+span))


def dense_gate(metrics: dict, sparse_metrics: dict, visual_status: str) -> dict:
    checks = dict(more_than_sparse=metrics["point_count"] > 6099,
                  finite=metrics["finite_xyz_fraction"] == 1.,
                  spatial_rank_3=metrics["rank"] == 3,
                  plausible_bounds=plausible_bounds(metrics, sparse_metrics),
                  reference_coverage=float(metrics.get("reference_expanded_fraction", 1.0)) >= .99,
                  color=metrics["has_color"], visual=visual_status == "passed")
    return dict(passed=all(checks.values()), checks=checks,
                quality_target_met=metrics["point_count"] >= 60990)


def integrated_success(reports: dict) -> bool:
    return all(reports.get(str(stage), {}).get("acceptance", {}).get("passed") is True
               for stage in range(14, 18))


def mesh_gate(metrics: dict, dense_metrics: dict, dominant_fraction: float, visual_status: str) -> dict:
    checks = dict(vertices=metrics["vertex_count"] > 0, faces=metrics["face_count"] > 0,
                  finite=metrics["finite_xyz_fraction"] == 1., spatial_rank_3=metrics["rank"] == 3,
                  plausible_bounds=plausible_bounds(metrics, dense_metrics),
                  dominant_component=dominant_fraction >= .3, visual=visual_status == "passed")
    return dict(passed=all(checks.values()), checks=checks)


def choose_mesh(candidates: list[dict]) -> dict | None:
    if len(candidates) > 2:
        raise ValueError("At most two meaningful mesh candidates are allowed")
    # An alternative is only generated after rejecting the primary. If both are
    # accepted on reinspection, preserve the first rather than chasing face count.
    return next((c for c in candidates if c["acceptance"]["passed"]), None)


def simplification_plan(metrics: dict) -> dict:
    """Choose simplification only from the measured mesh burden."""
    faces = int(metrics["face_count"])
    size = int(metrics["file_size_bytes"])
    if faces <= 750_000 and size <= 512 * 1024 * 1024:
        return dict(used=False, target_faces=None,
                    reason="Measured mesh is practical for texturing and headless Blender validation")
    target = min(500_000, max(100_000, faces // 2))
    return dict(used=True, target_faces=target,
                source_faces=faces, target_ratio=target / faces,
                reason="Measured face count or file size exceeds the practical texturing threshold")
