"""Restartable, gated Steps 14–17 using the frozen local Step 10 model.

Run each stage, inspect its real preview, then record a visual review with
--visual-status and --visual-note. `all` only sequences the stage-owned gates.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import pycolmap

from analysis_common import verify_selected_images, load_selected_manifest
from run_preprocessing import verify_raw_manifest
from sparse_reconstruction import summarize_reconstruction
from local_reconstruction import (
    SOURCE, OUTPUT, SELECTION_HASH, DENSE_COMMANDS, sha256, write_json,
    validate_source, registered_manifest, names_hash, safe_output,
    colmap_command, run_command, failure_category, dense_fallback,
    dense_gate, integrated_success, mesh_gate, simplification_plan,
    process_tree_pids,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_EXE = Path("C:/Tools/COLMAP-4.2.0/bin/colmap.exe")
DEFAULT_BLENDER = Path("C:/Program Files/Blender Foundation/Blender 5.2/blender.exe")
INITIAL_DENSE = dict(max_image_size=1600, geom_consistency=True, filter=True,
                     source_views=20, gpu_index="0", cache_size=1, num_threads=2)


def run_sequence(stages, finalize):
    for stage in stages:
        if not stage().get("acceptance", {}).get("passed", False):
            break
    return finalize()


def capability_gate(version, commands, cuda_device):
    return bool("COLMAP 4.2.0 " in version and "with CUDA" in version
                and cuda_device and all(commands.get(name) for name in DENSE_COMMANDS))


def inspect_map(path: Path, channels: int) -> dict:
    with path.open("rb") as handle:
        header = bytearray()
        while header.count(b"&") < 3 and len(header) < 100:
            value = handle.read(1)
            if not value:
                raise ValueError(f"Truncated dense map header: {path}")
            header.extend(value)
        width, height, count = map(int, header.decode("ascii").rstrip("&").split("&"))
        if min(width, height) <= 0 or count != channels:
            raise ValueError(f"Invalid dense map dimensions/channels: {path}")
        remaining = path.stat().st_size-handle.tell()
        if remaining != width*height*count*4:
            raise ValueError(f"Truncated or excess dense map payload: {path}")
        values = np.fromfile(handle, dtype="<f4")
    return dict(width=width, height=height, channels=count,
                finite=bool(np.isfinite(values).all()), positive_count=int((values > 0).sum()))


def visual_review(report, artifact, preview, status, note):
    identity = dict(artifact_sha256=sha256(artifact), preview_sha256=sha256(preview))
    previous = report.get("visual_review", {})
    if status is not None:
        if status not in {"passed", "failed"} or not note or not note.strip():
            raise ValueError("A visual decision requires substantive observations")
        return dict(status=status, note=note, **identity)
    if all(previous.get(k) == v for k, v in identity.items()):
        return previous
    return dict(status="pending", note="Real preview awaits inspection", **identity)


def texture_asset_identity(output_dir: Path, validation: dict, preview: Path) -> dict:
    """Bind cached rendering to every current asset plus the rendered preview."""
    output_dir = Path(output_dir)
    manifest = []
    for entry in validation["output_manifest"]:
        path = output_dir / entry["relative_path"]
        manifest.append(dict(relative_path=entry["relative_path"], sha256=sha256(path)))
    return dict(output_dir=str(output_dir.resolve()), output_manifest=manifest,
                preview_sha256=sha256(preview) if Path(preview).is_file() else None)


def restorable_source_identity(recorded, current):
    if recorded is not None and recorded != current:
        raise ValueError("Texture attempt source identity mismatch")
    return current


def mesh_preview_evidence(metrics):
    face_count = int(metrics.get("face_count", 0))
    vertex_count = int(metrics.get("vertex_count", metrics.get("point_count", 0)))
    return dict(
        mode="triangle_surface" if 0 < face_count <= 180000 else "sampled_vertices",
        face_render_limit=180000,
        sampled_vertex_limit=100000,
        sampled_vertex_count=min(vertex_count, 100000),
        full_coordinate_bounds=True,
    )


def render_geometry(path: Path, xyz, colors, title: str, cameras=None, faces=None):
    """Three honest full-bound views; no outlier cropping or synthetic geometry."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    xyz = np.asarray(xyz)
    finite = np.isfinite(xyz).all(axis=1)
    if not finite.all():
        raise ValueError("Preview refuses non-finite geometry")
    indices = np.linspace(0, len(xyz)-1, min(len(xyz), 100000), dtype=int)
    color = np.asarray(colors)[indices]/255. if colors is not None else "#b4974d"
    low, high = xyz.min(axis=0), xyz.max(axis=0)
    center, radius = (low+high)/2, max(high-low)/2
    fig = plt.figure(figsize=(16, 6), facecolor="white")
    for index, azimuth in enumerate((25, 115, 205), 1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        if faces is not None and len(faces) <= 180000:
            polys = Poly3DCollection(xyz[faces], facecolors="#b7a473", edgecolors="none", alpha=1.)
            ax.add_collection3d(polys)
        else:
            ax.scatter(*xyz[indices].T, c=color, s=.3 if len(xyz) > 20000 else 1., depthshade=False)
        if cameras is not None:
            ax.scatter(*np.asarray(cameras).T, color="#176ea0", s=8, label="73 calibrated cameras")
            low_c, high_c = np.minimum(low, np.min(cameras, axis=0)), np.maximum(high, np.max(cameras, axis=0))
            center, radius = (low_c+high_c)/2, max(high_c-low_c)/2
        ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
               zlim=(center[2]-radius, center[2]+radius), xlabel="X", ylabel="Y", zlabel="Z")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=azimuth)
    fig.suptitle(title, fontsize=15)
    fig.text(.5, .035, "LOCAL 73-view capture arc | full coordinate bounds | arbitrary SfM scale", ha="center")
    fig.subplots_adjust(top=.83, bottom=.10, left=.02, right=.98)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


@contextmanager
def pipeline_lock(work: Path):
    """OS advisory lock releases on interruption; lockfile carries no stage state."""
    work.mkdir(parents=True, exist_ok=True)
    with (work / "pipeline.lock").open("a+b") as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class Pipeline:
    def __init__(self, root=ROOT, executable=DEFAULT_EXE, blender=DEFAULT_BLENDER):
        self.root, self.executable = root.resolve(), executable.resolve()
        self.blender = blender.resolve()
        self.output = self.root / OUTPUT
        self.reports = safe_output(self.root, self.output / "reports")
        self.work = safe_output(self.root, self.output / "work")
        self.workspace = safe_output(self.root, self.work / "dense_workspace")
        self.previews = safe_output(self.root, self.output / "previews")
        self.selected = self.root / "preprocessing/pycolmap_input/images"
        self.selection_manifest = self.root / "preprocessing/reports/selection_manifest.csv"
        self.source = self.root / SOURCE
        self.env = dict(os.environ)
        self.env["PATH"] = str(executable.parent) + os.pathsep + self.env.get("PATH", "")
        self.env["QT_PLUGIN_PATH"] = str(executable.parent.parent / "plugins")

    def load(self, filename):
        path = self.reports / filename
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def save(self, filename, report):
        write_json(safe_output(self.root, self.reports / filename), report)
        return report

    def command(self, name, options, log_name, timeout=43200):
        options = dict(options, log_target="stdout", log_color=False)
        args = colmap_command(self.executable, name, options)
        print(f"Running {name}; log: {log_name}", flush=True)
        active = self.load("active_process.json")
        if active and active.get("status") == "running":
            live_tree = process_tree_pids(active["pid"])
            if live_tree:
                raise ValueError(
                    "Prior reconstruction process tree is still running; "
                    f"resume after these PIDs exit: {live_tree}"
                )
        result = run_command(args, safe_output(self.root, self.work / "logs" / log_name), env=self.env, timeout=timeout,
                             on_start=lambda pid: self.save("active_process.json", dict(pid=pid, args=args, status="running")))
        self.save("active_process.json", dict(status=result["status"], args=args))
        return result

    def integrity(self, full=False):
        baseline = self.load("step14_protection_baseline.json")
        if not baseline:
            raise ValueError("Freeze the protection baseline before preparing dense outputs")
        mismatches = [name for name, digest in baseline["protected_files"].items()
                      if not (self.root / name).is_file() or sha256(self.root / name) != digest]
        if mismatches:
            raise ValueError("Protected reconstruction evidence changed: " + ", ".join(mismatches))
        model = summarize_reconstruction(self.source, 288)
        validate_source(self.root, self.source, model)
        if sha256(self.selection_manifest) != SELECTION_HASH:
            raise ValueError("Frozen selected manifest changed")
        rows = registered_manifest(load_selected_manifest(self.selection_manifest), model.registered_image_names)
        if names_hash(rows) != baseline["ordered_names_sha256"]:
            raise ValueError("Frozen local registered-name manifest changed")
        for row in rows:
            if sha256(self.selected / row["filename"]) != row["sha256"]:
                raise ValueError("Local source image changed: " + row["filename"])
        result = dict(protected_files_checked=len(baseline["protected_files"]), protected_unchanged=True,
                      source_manifest_hash=names_hash(rows), local_images_verified=len(rows))
        if full:
            raw = verify_raw_manifest(self.root / "IMG20260826122949", self.root / "preprocessing/reports/raw_manifest_before.json")
            verified = verify_selected_images(self.selected, self.selection_manifest, expected_count=288)
            if not raw["unchanged"] or raw["actual_count"] != 297 or verified.manifest_sha256 != SELECTION_HASH:
                raise ValueError("Raw/selected source integrity failed")
            result.update(raw=raw, selected_count=len(verified.records))
        return result, model, rows

    def capability(self):
        version = run_command([str(self.executable), "-h"], self.work / "logs/capability-version.log", timeout=30, env=self.env)
        probes = {name: run_command([str(self.executable), name, "-h"], self.work / f"logs/capability-{name}.log", timeout=30, env=self.env)
                  for name in sorted(DENSE_COMMANDS)}
        gpu = run_command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv"], self.work / "logs/gpu.log", timeout=30)
        device = run_command([sys.executable, "-B", "-c", "import torch; x=torch.ones(8,device='cuda:0'); print(torch.cuda.get_device_name(0)); print(float(x.sum())); print(torch.cuda.get_device_capability(0))"], self.work / "logs/cuda-device.log", timeout=60)
        blender = run_command([str(self.blender), "--version"], self.work / "logs/blender-version.log", timeout=30)
        commands = {name: probe["status"] == "completed" and "Options can either" in probe["output_tail"] for name, probe in probes.items()}
        accepted = capability_gate(version["output_tail"], commands, device["status"] == "completed")
        report = dict(backend_kind="official_colmap_4.2_cuda_cli", backend_reason="Installed pyCOLMAP wheel has no CUDA; official CLI provides the full dense and texture chain",
                      executable=str(self.executable), executable_sha256=sha256(self.executable),
                      version=version["output_tail"].splitlines()[0], command_capabilities=commands,
                      command_help={name: probe["output_tail"] for name, probe in probes.items()},
                      pycolmap_version=pycolmap.__version__, pycolmap_has_cuda=pycolmap.has_cuda,
                      pycolmap_dense_functions=[name for name in ("undistort_images", "patch_match_stereo", "stereo_fusion", "poisson_meshing", "simplify_mesh") if hasattr(pycolmap, name)],
                      gpu=gpu, cuda_device=device, cuda_dense_supported=accepted,
                      blender=dict(path=str(self.blender), executable_sha256=sha256(self.blender),
                                   status=blender["status"],
                                   version=blender["output_tail"].splitlines()[0] if blender["status"] == "completed" else None,
                                   output_tail=blender["output_tail"]),
                      patchmatch_kernel_execution="not_run_until_step15",
                      release_url="https://github.com/colmap/colmap/releases/tag/4.2.0",
                      archive_sha256="991e0bae403a496fcc4de0c1f1f428619bf12f8000978f77bc6799d9bfeac23e")
        self.save("step14_capability.json", report)
        return report

    def inspect_workspace(self, rows):
        model = pycolmap.Reconstruction(self.workspace / "sparse")
        source = pycolmap.Reconstruction(self.source)
        expected = {row["filename"] for row in rows}
        actual = {image.name for image in model.images.values() if image.has_pose}
        files = {p.name for p in (self.workspace / "images").iterdir()}
        if actual != expected or files != expected or model.num_reg_images() != 73:
            raise ValueError("Dense workspace does not contain exactly the registered 73 views")
        options = pycolmap.UndistortCameraOptions()
        options.max_image_size = INITIAL_DENSE["max_image_size"]
        calibrated = pycolmap.undistort_camera(options, source.camera(next(iter(source.cameras))))
        for image_id, image in model.images.items():
            camera = model.camera(image.camera_id)
            original = source.image(image_id)
            if image.name != original.name or not np.allclose(image.cam_from_world().matrix(), original.cam_from_world().matrix(), atol=1e-10, rtol=0):
                raise ValueError("Undistortion changed calibrated camera poses")
            if camera.model_name != calibrated.model_name or (camera.width, camera.height) != (calibrated.width, calibrated.height) or not np.allclose(camera.params, calibrated.params, atol=1e-8, rtol=0):
                raise ValueError("Undistorted intrinsics differ from the exact pyCOLMAP transformation")
            frame = cv2.imread(str(self.workspace / "images" / image.name))
            if frame is None or frame.shape[:2] != (camera.height, camera.width):
                raise ValueError("Unreadable or incorrectly sized undistorted image")
        if model.num_points3D() != source.num_points3D() or any(not np.array_equal(p.xyz, source.point3D(pid).xyz) for pid, p in model.points3D.items()):
            raise ValueError("Undistortion changed sparse geometry")
        for filename in ("patch-match.cfg", "fusion.cfg"):
            if not (self.workspace / "stereo" / filename).is_file():
                raise ValueError("Missing stereo configuration")
        return dict(registered_images=73, readable_images=73, camera_model=calibrated.model_name,
                    camera_params=calibrated.params.tolist(), width=calibrated.width, height=calibrated.height,
                    exact_intrinsics_transformation=True, unchanged_poses=True, unchanged_sparse_points=True)

    def prepare(self):
        integrity, metrics, rows = self.integrity(full=True)
        self.reports.mkdir(parents=True, exist_ok=True)
        manifest = self.reports / "step14_source_manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        capability = self.capability()
        report = dict(source_sparse_model=SOURCE.as_posix(), source_model_metrics=metrics.to_dict(),
                      source_manifest_hash=names_hash(rows), source_csv_sha256=sha256(manifest),
                      capability=capability, integrity=integrity, acceptance={"passed": False})
        if not capability["cuda_dense_supported"]:
            report["status"] = "blocked_capability"
            return self.save("step14_summary.json", report)
        old = self.load("step14_summary.json")
        if not old or not old.get("undistortion", {}).get("status") == "completed":
            self.workspace.mkdir(parents=True, exist_ok=True)
            result = self.command("image_undistorter", dict(image_path=self.selected, input_path=self.source,
                                  output_path=self.workspace, output_type="COLMAP", max_image_size=1600,
                                  num_patch_match_src_images=20, num_threads=2), "step14-undistort.log", timeout=1800)
            report["undistortion"] = result
        else:
            report["undistortion"] = old["undistortion"]
        if report["undistortion"]["status"] != "completed":
            report["status"] = "failed_undistortion"
            return self.save("step14_summary.json", report)
        report["workspace"] = self.inspect_workspace(rows)
        model = pycolmap.Reconstruction(self.source)
        xyz = np.asarray([p.xyz for p in model.points3D.values()])
        colors = np.asarray([p.color for p in model.points3D.values()])
        camera_centers = np.asarray([model.image(i).projection_center() for i in model.reg_image_ids()])
        render_geometry(self.previews / "step14_01_local_sparse_source.png", xyz, colors,
                        "Step 14: frozen Step 10 LOCAL source — 73 views / 6,099 points", cameras=camera_centers)
        report["sparse_bounds"] = dict(bounding_box_min=xyz.min(axis=0).tolist(), bounding_box_max=xyz.max(axis=0).tolist(), bounding_box_extents=np.ptp(xyz, axis=0).tolist())
        report["integrity_after"] = self.integrity(full=True)[0]
        report.update(status="completed", acceptance={"passed": True})
        return self.save("step14_summary.json", report)

    def require(self, stage, filename):
        report = self.load(filename)
        if not report or report.get("acceptance", {}).get("passed") is not True:
            raise ValueError(f"Step {stage} hard gate has not passed")
        self.integrity()
        return report

    def stereo(self, visual_status=None, visual_note=None):
        prepared = self.require(14, "step14_summary.json")
        rows = self.integrity()[2]
        self.inspect_workspace(rows)
        capability = self.load("step14_capability.json")
        if sha256(self.executable) != capability["executable_sha256"]:
            raise ValueError("Dense backend executable changed since capability verification")
        ledger = self.load("step15_dense_attempts.json") or {"attempts": [], "source_manifest_hash": names_hash(rows)}
        if ledger["source_manifest_hash"] != names_hash(rows):
            raise ValueError("Dense attempt source identity mismatch")
        attempts = ledger["attempts"]
        if not attempts:
            attempts.append(dict(config=dict(INITIAL_DENSE), workspace=str(self.workspace), status="prepared"))
        while True:
            current = attempts[-1]
            workspace = safe_output(self.root, Path(current["workspace"]))
            config = current["config"]
            if current["status"] in {"prepared", "running", "interrupted"}:
                current["status"] = "running"
                self.save("step15_dense_attempts.json", ledger)
                result = self.command("patch_match_stereo", dict(workspace_path=workspace, workspace_format="COLMAP",
                                      **{"PatchMatchStereo."+key: value for key, value in config.items() if key != "source_views"}),
                                      f"step15-patchmatch-{len(attempts)}.log")
                current.update(status=result["status"], process=result, failure_category=failure_category(result))
                self.save("step15_dense_attempts.json", ledger)
            if current["status"] == "completed":
                break
            if (len(attempts) == 1
                    and current["status"] in {"failed", "timeout", "interrupted"}
                    and current.get("failure_category") in {"resource", "timeout", "runtime"}):
                fallback = dense_fallback(attempts)
                next_workspace = safe_output(self.root, self.work / "dense_fallback_workspace")
                if next_workspace.exists():
                    raise ValueError("Unowned existing fallback workspace; inspect before overwriting")
                for folder in ("images", "sparse"):
                    shutil.copytree(self.workspace / folder, next_workspace / folder)
                (next_workspace / "stereo").mkdir()
                for filename in ("patch-match.cfg", "fusion.cfg"):
                    shutil.copy2(self.workspace / "stereo" / filename, next_workspace / "stereo" / filename)
                for folder in ("depth_maps", "normal_maps", "consistency_graphs"):
                    (next_workspace / "stereo" / folder).mkdir()
                attempts.append(dict(config=fallback, workspace=str(next_workspace), status="prepared",
                                     reason="One measured resource failure; reduce internal resolution only, isolate maps"))
                self.save("step15_dense_attempts.json", ledger)
                continue
            return self.save("step15_summary.json", dict(status="failed_stereo", attempts=attempts,
                             acceptance={"passed": False}, reason=current.get("failure_category")))
        input_type = "geometric" if config["geom_consistency"] else "photometric"
        maps = []
        for row in rows:
            name = row["filename"]
            depth = inspect_map(workspace / f"stereo/depth_maps/{name}.{input_type}.bin", 1)
            normal = inspect_map(workspace / f"stereo/normal_maps/{name}.{input_type}.bin", 3)
            if not depth["finite"] or not normal["finite"] or depth["positive_count"] == 0:
                raise ValueError("Invalid depth/normal output for " + name)
            maps.append(dict(filename=name, depth=depth, normal=normal))
        fused = safe_output(self.root, self.output / "dense/fused.ply")
        fused.parent.mkdir(parents=True, exist_ok=True)
        previous = self.load("step15_summary.json") or {}
        fusion = current.get("fusion")
        if not fusion or fusion["status"] != "completed":
            self.save("step15_summary.json", dict(status="fusion_running", selected_attempt=len(attempts),
                      acceptance={"passed": False}))
            fusion = self.command("stereo_fusion", dict(workspace_path=workspace, workspace_format="COLMAP",
                                  input_type=input_type, output_path=fused, output_type="PLY",
                                  **{"StereoFusion.max_image_size": config["max_image_size"],
                                     "StereoFusion.num_threads": 2, "StereoFusion.cache_size": 4,
                                     "StereoFusion.use_cache": True}), "step15-fusion.log")
            current["fusion"] = fusion
            self.save("step15_dense_attempts.json", ledger)
        if fusion["status"] != "completed":
            return self.save("step15_summary.json", dict(status="failed_fusion", acceptance={"passed": False}, fusion=fusion))
        capability["patchmatch_kernel_execution"] = dict(
            status=current["process"]["status"], runtime_seconds=current["process"]["runtime_seconds"],
            log_sha256=current["process"].get("log_sha256"), gpu_index=config["gpu_index"],
            completed_depth_maps=len(rows))
        self.save("step14_capability.json", capability)
        from local_reconstruction_io import ply_metrics, read_ply
        metrics = ply_metrics(fused, reference_bounds=prepared["sparse_bounds"])
        metrics["dense_to_sparse_ratio"] = metrics["point_count"]/6099
        preview = self.previews / "step15_01_dense_cloud.png"
        if not preview.exists() or previous.get("fused_sha256") != sha256(fused):
            data = read_ply(fused)
            render_geometry(preview, data["xyz"], data["colors"], f"Step 15: real local dense cloud — {metrics['point_count']:,} points")
        review = visual_review(previous, fused, preview, visual_status, visual_note)
        acceptance = dense_gate(metrics, prepared["sparse_bounds"], review["status"])
        report = dict(status="completed" if acceptance["passed"] else "pending_visual" if review["status"] == "pending" else "failed_gate",
                      source_manifest_hash=names_hash(rows), selected_attempt=len(attempts),
                      selected_config=config, workspace=str(workspace), metrics=metrics, maps=maps,
                      readable_depth_maps=len(maps), readable_normal_maps=len(maps), fusion=fusion,
                      fused_path=fused.relative_to(self.root).as_posix(), fused_sha256=sha256(fused),
                      visual_review=review, acceptance=acceptance, backend=capability["version"], integrity=self.integrity()[0])
        return self.save("step15_summary.json", report)

    def _mesh_candidate(self, name, path, process=None):
        from local_reconstruction_io import component_summary, ply_metrics, read_ply
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"Mesh candidate is missing: {path}")
        data = read_ply(path)
        metrics = ply_metrics(path)
        components = component_summary(data)
        dominant = float(components.get("dominant_face_fraction", 0.0))
        return dict(name=name, path=str(path), metrics=metrics, components=components,
                    dominant_face_fraction=dominant, process=process,
                    preview_path=str(self.previews / f"step16_{name}.png"),
                    preview_evidence=mesh_preview_evidence(metrics))

    def _render_mesh_candidate(self, candidate):
        from local_reconstruction_io import read_ply
        data = read_ply(Path(candidate["path"]))
        colors = data.get("colors")
        render_geometry(Path(candidate["preview_path"]), data["xyz"], colors,
                        f"Step 16: {candidate['name']} mesh — LOCAL 73-view reconstruction",
                        faces=data.get("faces"))

    def _comparison_preview(self, candidates):
        from PIL import Image, ImageDraw
        images = [Image.open(Path(candidate["preview_path"])).convert("RGB") for candidate in candidates]
        width = max(image.width for image in images)
        height = max(image.height for image in images)
        canvas = Image.new("RGB", (width * len(images), height), "white")
        for index, image in enumerate(images):
            canvas.paste(image, (index * width, 0))
        out = self.previews / "step16_02_mesh_comparison.png"
        canvas.save(out)
        for image in images:
            image.close()
        return out

    def _record_mesh_attempt(self, name, algorithm, path, process):
        record = dict(name=name, algorithm=algorithm, path=str(path), process=process,
                      preview_path=str(self.previews / f"step16_{name}.png"))
        if process.get("status") != "completed":
            record["output_error"] = "Mesher subprocess did not complete"
            return record
        try:
            record.update(self._mesh_candidate(name, path, process))
            record["algorithm"] = algorithm
        except (OSError, ValueError) as error:
            record["output_error"] = str(error)
        return record

    @staticmethod
    def _mesh_hard_acceptance(candidate, dense_metrics):
        if "metrics" not in candidate:
            return dict(passed=False, checks={"process_and_output": False})
        checks = mesh_gate(candidate["metrics"], dense_metrics,
                           candidate["dominant_face_fraction"], "passed")["checks"]
        checks["process_and_output"] = candidate.get("process", {}).get("status") == "completed"
        return dict(passed=all(value for key, value in checks.items() if key != "visual"), checks=checks)

    def mesh(self, visual_status=None, visual_note=None):
        dense = self.require(15, "step15_summary.json")
        rows = self.integrity()[2]
        from local_reconstruction_io import ply_metrics
        fused = self.root / dense["fused_path"]
        if not fused.is_file():
            raise ValueError("Accepted dense report does not point to a fused PLY")
        ledger = self.load("step16_mesh_attempts.json") or {
            "source_manifest_hash": names_hash(rows), "attempts": []
        }
        if ledger["source_manifest_hash"] != names_hash(rows):
            raise ValueError("Mesh attempt source identity mismatch")
        attempts = ledger["attempts"]
        previous = self.load("step16_summary.json") or {}
        mesh_dir = safe_output(self.root, self.output / "mesh")
        mesh_dir.mkdir(parents=True, exist_ok=True)
        if not attempts:
            primary_path = safe_output(self.root, mesh_dir / "primary_mesh.ply")
            if primary_path.exists():
                raise ValueError("Unrecorded primary mesh exists; inspect before overwriting")
            process = self.command("poisson_mesher", dict(input_path=fused, output_path=primary_path,
                                    **{"PoissonMeshing.point_weight": 1,
                                       "PoissonMeshing.depth": 13,
                                       "PoissonMeshing.color": True,
                                       "PoissonMeshing.trim": 10,
                                       "PoissonMeshing.num_threads": 2}), "step16-poisson.log", timeout=7200)
            attempts.append(self._record_mesh_attempt("primary_mesh", "poisson", primary_path, process))
            self.save("step16_mesh_attempts.json", ledger)
        candidates = attempts
        for candidate in candidates:
            if isinstance(candidate.get("components"), dict):
                candidate["components"].pop("face_labels", None)
            if ("metrics" not in candidate and candidate.get("process", {}).get("status") == "completed"
                    and Path(candidate["path"]).is_file()):
                refreshed = self._mesh_candidate(candidate["name"], candidate["path"], candidate.get("process"))
                candidate.update(refreshed)
            if "metrics" in candidate:
                candidate["preview_evidence"] = mesh_preview_evidence(candidate["metrics"])
            if "metrics" in candidate and not Path(candidate["preview_path"]).is_file():
                self._render_mesh_candidate(candidate)
        dense_metrics = dense["metrics"]
        for candidate in candidates:
            candidate["hard_acceptance"] = self._mesh_hard_acceptance(candidate, dense_metrics)
        primary = candidates[0]
        visual_consumed = False
        if primary["hard_acceptance"]["passed"]:
            primary_review = visual_review(primary, Path(primary["path"]), Path(primary["preview_path"]),
                                           visual_status if len(candidates) == 1 else None,
                                           visual_note if len(candidates) == 1 else None)
            primary["visual_review"] = primary_review
            visual_consumed = visual_status is not None and len(candidates) == 1
        else:
            primary_review = primary.get("visual_review", {"status": "failed", "note": primary.get("output_error", "Primary hard gate failed")})
        primary_rejected = not primary["hard_acceptance"]["passed"] or primary_review.get("status") == "failed"
        alternative_created = False
        if primary_rejected:
            if len(candidates) == 1:
                workspace = self.root / dense["workspace"]
                workspace_fused = workspace / "fused.ply"
                workspace_vis = workspace / "fused.ply.vis"
                durable_vis = Path(str(fused) + ".vis")
                if not durable_vis.is_file():
                    raise ValueError("Delaunay alternative requires fusion visibility output fused.ply.vis")
                if not workspace_fused.is_file():
                    shutil.copy2(fused, workspace_fused)
                if not workspace_vis.is_file():
                    shutil.copy2(durable_vis, workspace_vis)
                alternative_path = safe_output(self.root, mesh_dir / "optional_alternative_mesh.ply")
                process = self.command("delaunay_mesher", dict(input_path=workspace, input_type="dense",
                                          output_path=alternative_path,
                                          **{"DelaunayMeshing.num_threads": 2}), "step16-delaunay.log", timeout=7200)
                candidates.append(self._record_mesh_attempt("alternative_mesh", "delaunay", alternative_path, process))
                alternative_created = True
                self.save("step16_mesh_attempts.json", ledger)
                if "metrics" in candidates[-1]:
                    self._render_mesh_candidate(candidates[-1])
                    self._comparison_preview([candidate for candidate in candidates if "metrics" in candidate])
            if len(candidates) > 2:
                raise ValueError("At most one mesh alternative is permitted")
        selected = primary if not primary_rejected else candidates[-1]
        selected["hard_acceptance"] = self._mesh_hard_acceptance(selected, dense_metrics)
        if not selected["hard_acceptance"]["passed"]:
            self.save("step16_mesh_attempts.json", ledger)
            report = dict(status="failed_alternative" if selected is not primary else "failed_primary",
                          source_manifest_hash=names_hash(rows), selected_mesh=selected,
                          attempts=candidates, final_mesh_path=None, final_metrics=None,
                          simplification={"used": False, "target_faces": None,
                                          "reason": "No valid mesh candidate was available"},
                          acceptance={"passed": False, "checks": selected["hard_acceptance"]["checks"]},
                          reason=selected.get("output_error", "Mesh hard gate failed"), integrity=self.integrity()[0])
            return self.save("step16_summary.json", report)
        selected_already_passed = selected.get("visual_review", {}).get("status") == "passed"
        selected_status = None if visual_consumed or alternative_created or selected_already_passed else visual_status
        selected_note = None if visual_consumed or alternative_created or selected_already_passed else visual_note
        review = visual_review(selected, Path(selected["path"]), Path(selected["preview_path"]), selected_status, selected_note)
        selected["visual_review"] = review
        checks = mesh_gate(selected["metrics"], dense_metrics, selected["dominant_face_fraction"], review["status"])
        selected["acceptance"] = checks
        visual_consumed = visual_consumed or selected_status is not None
        final_source = Path(selected["path"])
        processing_metrics = selected["metrics"]
        processing_components = selected["components"]
        component_cleanup = dict(used=False, reason="No clear small-component cleanup rule was triggered")
        if checks["passed"] and processing_components["component_count"] > 1:
            face_counts = processing_components["face_counts"]
            minimum_fraction = .005
            minimum_faces = max(1, int(np.ceil(processing_metrics["face_count"] * minimum_fraction)))
            kept_faces = sum(count for count in face_counts if count >= minimum_faces)
            retained_fraction = kept_faces / processing_metrics["face_count"]
            if kept_faces < processing_metrics["face_count"] and retained_fraction >= .95:
                from local_reconstruction_io import filter_small_components, read_ply, write_ply
                cleaned_path = safe_output(self.root, mesh_dir / "component_filtered_mesh.ply")
                old_cleanup = previous.get("component_cleanup", {})
                source_sha = sha256(final_source)
                if old_cleanup.get("source_sha256") != source_sha or not cleaned_path.is_file():
                    if cleaned_path.exists():
                        raise ValueError("Unbound component-filtered mesh exists; inspect before overwriting")
                    cleaned, cleanup_metrics = filter_small_components(read_ply(final_source), minimum_fraction)
                    write_ply(cleaned_path, cleaned["xyz"], faces=cleaned["faces"],
                              colors=cleaned["colors"], normals=cleaned["normals"])
                    component_cleanup = dict(cleanup_metrics, source_sha256=source_sha,
                                             output_path=str(cleaned_path), output_sha256=sha256(cleaned_path))
                else:
                    component_cleanup = old_cleanup
                cleaned_candidate = self._mesh_candidate("component_filtered_mesh", cleaned_path,
                                                         {"status": "completed", "kind": "deterministic_component_filter"})
                component_cleanup.update(output_metrics=cleaned_candidate["metrics"],
                                         components=cleaned_candidate["components"])
                final_source = cleaned_path
                processing_metrics = cleaned_candidate["metrics"]
                processing_components = cleaned_candidate["components"]
        simplification = simplification_plan(processing_metrics)
        final_metrics = None
        final_review = review
        if checks["passed"] and simplification["used"]:
            simplified_path = safe_output(self.root, mesh_dir / "simplified_mesh.ply")
            old_simplification = previous.get("simplification", {})
            source_sha = sha256(final_source)
            process = old_simplification.get("process")
            if (old_simplification.get("source_sha256") != source_sha
                    or not process or process.get("status") != "completed"
                    or not simplified_path.is_file()):
                if simplified_path.exists():
                    raise ValueError("Unbound simplified mesh exists; inspect before overwriting")
                process = self.command("mesh_simplifier", dict(input_path=final_source, output_path=simplified_path,
                                       **{"MeshSimplification.target_face_ratio": simplification["target_ratio"],
                                          "MeshSimplification.interpolate_colors": True,
                                          "MeshSimplification.num_threads": 2}), "step16-simplify.log", timeout=3600)
                old_simplification = dict(simplification, source_sha256=source_sha, process=process)
            if process["status"] != "completed" or not simplified_path.is_file():
                checks = {"passed": False, "checks": {"simplification_process": False}}
                simplification = dict(old_simplification, acceptance=checks)
            else:
                simplified = self._mesh_candidate("simplified_mesh", simplified_path, process)
                simplified_preview = self.previews / "step16_simplified_mesh.png"
                simplified["preview_path"] = str(simplified_preview)
                if not simplified_preview.is_file() or old_simplification.get("output_sha256") != sha256(simplified_path):
                    self._render_mesh_candidate(simplified)
                simplified_review = visual_review(old_simplification, simplified_path, simplified_preview,
                                                  None if visual_consumed else visual_status,
                                                  None if visual_consumed else visual_note)
                simplified_gate = mesh_gate(simplified["metrics"], dense_metrics,
                                            simplified["dominant_face_fraction"], simplified_review["status"])
                simplification = dict(old_simplification, **simplification,
                                      output_path=str(simplified_path), output_sha256=sha256(simplified_path),
                                      output_metrics=simplified["metrics"], components=simplified["components"],
                                      preview_evidence=simplified["preview_evidence"],
                                      visual_review=simplified_review, acceptance=simplified_gate)
                checks = simplified_gate
                final_review = simplified_review
                if checks["passed"]:
                    final_source = simplified_path
        elif checks["passed"] and component_cleanup["used"]:
            cleaned_candidate = self._mesh_candidate("component_filtered_mesh", final_source,
                                                     {"status": "completed", "kind": "deterministic_component_filter"})
            cleaned_preview = self.previews / "step16_component_filtered_mesh.png"
            cleaned_candidate["preview_path"] = str(cleaned_preview)
            if not cleaned_preview.is_file() or component_cleanup.get("reviewed_output_sha256") != sha256(final_source):
                self._render_mesh_candidate(cleaned_candidate)
            final_review = visual_review(component_cleanup, final_source, cleaned_preview,
                                         None if visual_consumed else visual_status,
                                         None if visual_consumed else visual_note)
            checks = mesh_gate(cleaned_candidate["metrics"], dense_metrics,
                               cleaned_candidate["dominant_face_fraction"], final_review["status"])
            component_cleanup.update(visual_review=final_review, acceptance=checks,
                                     reviewed_output_sha256=sha256(final_source))
        self.save("step16_mesh_attempts.json", ledger)
        if checks["passed"]:
            final_path = safe_output(self.root, mesh_dir / "final_mesh.ply")
            final_source_sha = sha256(final_source)
            if final_path.is_file() and sha256(final_path) != final_source_sha:
                recorded_final_sha = previous.get("final_mesh_sha256")
                if not recorded_final_sha or sha256(final_path) != recorded_final_sha:
                    raise ValueError("Unbound final mesh exists; inspect before overwriting")
            if not final_path.is_file() or sha256(final_path) != final_source_sha:
                shutil.copy2(final_source, final_path)
            final_metrics = ply_metrics(final_path)
            final_mesh_sha = sha256(final_path)
            status = "completed"
        else:
            final_path = mesh_dir / "final_mesh.ply"
            final_source_sha = None
            final_mesh_sha = None
            status = "pending_visual" if final_review["status"] == "pending" else "failed_gate"
        report = dict(status=status, source_manifest_hash=names_hash(rows), selected_mesh=selected,
                      attempts=candidates, final_mesh_path=final_path.relative_to(self.root).as_posix() if final_metrics else None,
                      final_mesh_source_sha256=final_source_sha, final_mesh_sha256=final_mesh_sha,
                      final_metrics=final_metrics, component_cleanup=component_cleanup,
                      simplification=simplification, visual_review=final_review,
                      acceptance=checks, integrity=self.integrity()[0])
        return self.save("step16_summary.json", report)

    def _preview_inputs(self, mesh_path, texture_path):
        from local_reconstruction_io import read_ply
        data = read_ply(mesh_path)
        geometry = safe_output(self.root, self.work / "texture_preview_geometry.npz")
        np.savez_compressed(geometry, xyz=data["xyz"], faces=data["faces"], uvs=data["uvs"] if data["uvs"] is not None else np.empty((0, 3, 2)))
        reconstruction = pycolmap.Reconstruction(Path(self.load("step15_summary.json")["workspace"]) / "sparse")
        cameras = []
        for image_id in reconstruction.reg_image_ids():
            image = reconstruction.image(image_id)
            camera = reconstruction.camera(image.camera_id)
            cameras.append(dict(name=image.name, width=camera.width, height=camera.height,
                                params=[float(value) for value in camera.params],
                                cam_from_world=image.cam_from_world().matrix().tolist()))
        camera_json = safe_output(self.root, self.work / "texture_preview_cameras.json")
        camera_json.write_text(json.dumps(cameras, indent=2) + "\n", encoding="utf-8")
        return geometry, camera_json

    def _run_blender_preview(self, mesh_path, texture_path, output, report_path):
        script = self.root / "render_local_reconstruction.py"
        if not script.is_file():
            raise ValueError("Headless Blender validation renderer is missing")
        geometry, cameras = self._preview_inputs(mesh_path, texture_path)
        for target in (Path(output), Path(report_path),
                       *(Path(output).with_name(Path(output).stem + f"_view{index}.png") for index in range(1, 4))):
            safe_target = safe_output(self.root, target)
            if safe_target.is_file():
                safe_target.unlink()
        args = [str(self.blender), "--background", "--python-exit-code", "1", "--python", str(script), "--", "--geometry", str(geometry),
                "--cameras", str(cameras), "--output", str(output), "--report", str(report_path),
                "--texture", str(texture_path)]
        result = run_command(args, safe_output(self.root, self.work / "logs/step17-blender.log"), timeout=1800)
        if result["status"] == "completed":
            try:
                from PIL import Image
                payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
                if payload.get("status") != "completed":
                    raise ValueError("render report did not complete")
                with Image.open(output) as image:
                    image.verify()
            except (OSError, ValueError, json.JSONDecodeError) as error:
                result = dict(result, status="failed", returncode=result.get("returncode"),
                              output_tail=result.get("output_tail", "") + f"\nMissing or invalid Blender render contract: {error}")
        return result

    def texture_attempt_dir(self, attempt_number):
        if attempt_number not in (1, 2):
            raise ValueError("Only two isolated texture attempts are permitted")
        return safe_output(self.root, self.output / "texture" / f"attempt_{attempt_number}")

    def cleanup_texture_attempt(self, attempt_number):
        path = self.texture_attempt_dir(attempt_number)
        expected_parent = safe_output(self.root, self.output / "texture" / "attempt_guard").parent
        if path.parent != expected_parent:
            raise ValueError("Texture cleanup escaped the attempt directory boundary")
        if path.exists():
            if path.is_symlink() or path.is_junction():
                raise ValueError("Refusing linked texture attempt cleanup")
            shutil.rmtree(path)
        return path

    def texture(self, visual_status=None, visual_note=None):
        mesh = self.require(16, "step16_summary.json")
        rows = self.integrity()[2]
        from local_reconstruction_io import validate_textured_asset
        final_mesh = self.root / mesh["final_mesh_path"]
        texture_root = safe_output(self.root, self.output / "texture")
        texture_root.mkdir(parents=True, exist_ok=True)
        ledger = self.load("step17_texture_summary.json") or {"attempts": [], "source_manifest_hash": names_hash(rows)}
        current_source_hash = names_hash(rows)
        ledger["source_manifest_hash"] = restorable_source_identity(
            ledger.get("source_manifest_hash"), current_source_hash)
        attempts = ledger["attempts"]
        if not attempts:
            first_output = self.texture_attempt_dir(1)
            if first_output.exists():
                raise ValueError("Unrecorded texture attempt exists; inspect before overwriting")
            attempts.append(dict(config={"texture_scale_factor": 1.0, "num_threads": 2}, status="prepared",
                                 output_dir=str(first_output)))
        while attempts[-1]["status"] in {"prepared", "running"}:
            current = attempts[-1]
            attempt_number = len(attempts)
            output_dir = self.texture_attempt_dir(attempt_number)
            if Path(current.get("output_dir", output_dir)).resolve() != output_dir:
                raise ValueError("Texture attempt output identity mismatch")
            current["output_dir"] = str(output_dir)
            if current["status"] == "running":
                self.cleanup_texture_attempt(attempt_number)
                current["restart_cleanup"] = "Removed only the interrupted attempt's isolated output"
            elif output_dir.exists():
                raise ValueError("Unbound prepared texture attempt exists; inspect before overwriting")
            output_dir.mkdir(parents=True, exist_ok=True)
            current["status"] = "running"
            self.save("step17_texture_summary.json", ledger)
            process = self.command("mesh_texturer", dict(workspace_path=Path(mesh.get("workspace", self.load("step15_summary.json")["workspace"])),
                                      input_path=final_mesh, output_path=output_dir, output_type="BIN",
                                      **{"MeshTextureMapping.texture_scale_factor": current["config"]["texture_scale_factor"],
                                         "MeshTextureMapping.num_threads": current["config"]["num_threads"]}),
                                      f"step17-texture-{len(attempts)}.log", timeout=7200)
            category = failure_category(process)
            message = process.get("output_tail", "").lower()
            if category == "subprocess" and any(term in message for term in ("atlas", "packing", "texture scale", "too large")):
                category = "configuration"
            current.update(status=process["status"], process=process, failure_category=category)
            self.save("step17_texture_summary.json", ledger)
            if (current["status"] != "completed" and len(attempts) == 1
                    and category in {"resource", "timeout", "runtime", "configuration"}):
                self.cleanup_texture_attempt(1)
                current["failed_output_cleanup"] = "Removed only reconstruction/local_dense/texture/attempt_1"
                attempts.append(dict(config={"texture_scale_factor": .5, "num_threads": 2}, status="prepared",
                                     output_dir=str(self.texture_attempt_dir(2)),
                                     reason=f"One measured {category} failure; halve atlas scale in an isolated output"))
                self.save("step17_texture_summary.json", ledger)
                continue
            break
        current = attempts[-1]
        if current["status"] != "completed":
            return self.save("step17_texture_summary.json", dict(status="failed_texturing", attempts=attempts,
                      acceptance={"passed": False}, source_manifest_hash=names_hash(rows)))
        output_dir = self.texture_attempt_dir(len(attempts))
        validation = validate_textured_asset(output_dir, final_mesh)
        preview = self.previews / "step17_01_textured_preview.png"
        render_report = self.reports / "step17_blender_render.json"
        current_identity = texture_asset_identity(output_dir, validation, preview)
        current_identity["renderer_sha256"] = sha256(self.root / "render_local_reconstruction.py")
        current_identity["blender_sha256"] = sha256(self.blender)
        cached_identity = ledger.get("render_identity", {})
        needs_render = (not preview.is_file() or not render_report.is_file()
                        or cached_identity.get("output_manifest") != current_identity["output_manifest"]
                        or cached_identity.get("preview_sha256") != current_identity["preview_sha256"]
                        or cached_identity.get("renderer_sha256") != current_identity["renderer_sha256"]
                        or cached_identity.get("blender_sha256") != current_identity["blender_sha256"])
        if needs_render:
            texture_path = Path(validation["texture_path"])
            render = self._run_blender_preview(output_dir / "mesh.ply", texture_path, preview, render_report)
            if render["status"] != "completed":
                return self.save("step17_texture_summary.json", dict(status="failed_preview", attempts=attempts,
                          source_manifest_hash=current_source_hash, validation=validation,
                          render=render, acceptance={"passed": False}))
            ledger["render_report"] = json.loads(render_report.read_text(encoding="utf-8"))
            ledger["render_identity"] = texture_asset_identity(output_dir, validation, preview)
            ledger["render_identity"]["renderer_sha256"] = sha256(self.root / "render_local_reconstruction.py")
            ledger["render_identity"]["blender_sha256"] = sha256(self.blender)
            self.save("step17_texture_summary.json", ledger)
        else:
            ledger["render_report"] = json.loads(render_report.read_text(encoding="utf-8"))
        review = visual_review(ledger, output_dir / "mesh.ply", preview, visual_status, visual_note)
        acceptance = dict(passed=bool(validation["accepted"] and review["status"] == "passed"),
                          checks=dict(asset=validation["accepted"], visual=review["status"] == "passed"))
        report = dict(status="completed" if acceptance["passed"] else "pending_visual" if review["status"] == "pending" else "failed_gate",
                      attempts=attempts, source_manifest_hash=names_hash(rows), validation=validation,
                      texture_directory=output_dir.relative_to(self.root).as_posix(),
                      render=ledger["render_report"], render_identity=ledger["render_identity"],
                      visual_review=review, acceptance=acceptance,
                      texture_output=validation["output_manifest"], integrity=self.integrity()[0])
        return self.save("step17_texture_summary.json", report)

    def finalize(self):
        reports = {
            "14": self.load("step14_summary.json") or {},
            "15": self.load("step15_summary.json") or {},
            "16": self.load("step16_summary.json") or {},
            "17": self.load("step17_texture_summary.json") or {},
        }
        source = reports["14"].get("source_model_metrics", {})
        source_hash = reports["14"].get("source_manifest_hash")
        summary = dict(source_sparse_model=SOURCE.as_posix(), source_registered_images=73,
                       source_model_metrics=source, source_manifest_hash=source_hash,
                       capability=self.load("step14_capability.json") or reports["14"].get("capability", {}),
                       selected_dense_attempt=dict(number=reports["15"].get("selected_attempt"),
                                                   config=reports["15"].get("selected_config"),
                                                   metrics=reports["15"].get("metrics"),
                                                   fusion=reports["15"].get("fusion")),
                       dense_acceptance=reports["15"].get("acceptance", {"passed": False}),
                       selected_mesh=dict(candidate=reports["16"].get("selected_mesh"),
                                          final_mesh_path=reports["16"].get("final_mesh_path"),
                                          final_metrics=reports["16"].get("final_metrics"),
                                          component_cleanup=reports["16"].get("component_cleanup"),
                                          simplification=reports["16"].get("simplification")),
                       mesh_acceptance=reports["16"].get("acceptance", {"passed": False}),
                       texture_output=reports["17"].get("texture_output"),
                       texture_validation=reports["17"].get("validation"),
                       texture_render=reports["17"].get("render"),
                       texture_acceptance=reports["17"].get("acceptance", {"passed": False}),
                       visual_statuses={str(stage): reports[str(stage)].get("visual_review", {}).get("status")
                                        for stage in range(14, 18)},
                       steps14_17_success=integrated_success(reports),
                       known_limitations=["Local 73-view capture arc; not a complete 288-view global reconstruction",
                                          "Reflective brass produces seams, highlight inconsistencies, and view-dependent coverage"],
                       next_phase="Separately authorize any Blender manual cleanup and final presentation work",
                       blender_manual_cleanup_started=False,
                       generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        return self.save("steps14_17_summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare", "stereo", "mesh", "texture", "finalize", "all"))
    parser.add_argument("--colmap", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--visual-status", choices=("passed", "failed"))
    parser.add_argument("--visual-note")
    args = parser.parse_args()
    pipeline = Pipeline(executable=args.colmap)
    with pipeline_lock(pipeline.work):
        if args.stage == "prepare":
            report = pipeline.prepare()
        elif args.stage == "stereo":
            report = pipeline.stereo(args.visual_status, args.visual_note)
        elif args.stage == "mesh":
            report = pipeline.mesh(args.visual_status, args.visual_note)
        elif args.stage == "texture":
            report = pipeline.texture(args.visual_status, args.visual_note)
        elif args.stage == "finalize":
            report = pipeline.finalize()
        else:
            report = run_sequence([
                pipeline.prepare,
                lambda: pipeline.stereo(args.visual_status, args.visual_note),
                lambda: pipeline.mesh(args.visual_status, args.visual_note),
                lambda: pipeline.texture(args.visual_status, args.visual_note),
            ], pipeline.finalize)
    print(json.dumps({"stage": args.stage, "status": report.get("status", "completed"),
                      "acceptance": report.get("acceptance", {"passed": report.get("steps14_17_success", False)})}, indent=2))
    return 0 if report.get("acceptance", {"passed": report.get("steps14_17_success", False)})["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
