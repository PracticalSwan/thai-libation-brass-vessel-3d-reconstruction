"""COLMAP 4.2 CUDA dense-reconstruction contracts for V4."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np

from v4_config import (
    RECONSTRUCTION_V4_ROOT,
    assert_output_path,
    fingerprint,
    sha256_file,
    write_json,
)

CONTAMINATION_METRIC_KEYS = (
    "board_point_fraction",
    "pedestal_board_webbing_point_fraction",
    "cloth_or_background_point_fraction",
)
CONTAMINATION_FINDING_KEYS = (
    "board_slab_detected",
    "pedestal_board_webbing_detected",
    "cloth_or_background_structure_detected",
    "vessel_identity_confirmed",
)
SEMANTIC_VIEW_KEYS = ("front", "quarter", "side", "top_oblique")
REQUIRED_REVIEW_REGION_KEYS = (
    "bowl_and_interior",
    "globe_or_shoulder",
    "neck_lid_or_finial",
    "pedestal_and_base",
)


@dataclass(frozen=True)
class V4DenseConfig:
    max_image_size: int = 2000
    gpu_index: int = 0
    geom_consistency: bool = True
    filter: bool = True
    input_type: str = "geometric"
    workspace_format: str = "COLMAP"

    def validate(self) -> "V4DenseConfig":
        if self.max_image_size < 1:
            raise ValueError("dense max_image_size must be positive")
        if self.gpu_index < 0:
            raise ValueError("dense GPU index must be nonnegative")
        if self.input_type != "geometric":
            raise ValueError("V4 fusion is frozen to geometric depth maps")
        if self.workspace_format != "COLMAP":
            raise ValueError("V4 dense workspace must use COLMAP format")
        return self


def dense_stage_fingerprint(
    *,
    sparse_sha256: str,
    image_manifest_sha256: str,
    mask_manifest_sha256: str,
    config: V4DenseConfig = V4DenseConfig(),
) -> str:
    config.validate()
    return fingerprint(
        {
            "sparse_sha256": sparse_sha256,
            "image_manifest_sha256": image_manifest_sha256,
            "mask_manifest_sha256": mask_manifest_sha256,
            "config": asdict(config),
        }
    )


def _colmap_executable() -> str:
    path = shutil.which("colmap")
    if not path:
        raise RuntimeError("COLMAP executable is unavailable")
    # The Windows distribution puts a batch launcher on PATH.  Passing a
    # project path containing spaces through that launcher loses quoting in
    # ``%*``; prefer its sibling executable for all V4 subprocesses.
    candidate = Path(path)
    if candidate.suffix.lower() in {".bat", ".cmd"}:
        direct = candidate.parent / "bin" / "colmap.exe"
        if direct.is_file():
            return str(direct)
    return path


def colmap_help(command: str) -> str:
    """Return current installed help; do not rely on stale option spelling."""

    if command not in {"image_undistorter", "patch_match_stereo", "stereo_fusion", "poisson_mesher"}:
        raise ValueError(f"unsupported V4 COLMAP help command: {command}")
    completed = subprocess.run(
        [_colmap_executable(), command, "-h"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    output = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode != 0:
        raise RuntimeError(f"COLMAP {command} help failed: {output[-2000:]}")
    return output


def build_image_undistorter_command(
    *,
    image_path: Path,
    input_path: Path,
    output_path: Path,
    image_list_path: Path | None = None,
    max_image_size: int = 2000,
) -> list[str]:
    if max_image_size < 1:
        raise ValueError("max_image_size must be positive")
    return [
        _colmap_executable(),
        "image_undistorter",
        "--image_path",
        str(image_path),
        "--input_path",
        str(input_path),
        "--output_path",
        str(output_path),
        "--output_type",
        "COLMAP",
        "--max_image_size",
        str(max_image_size),
        *( ["--image_list_path", str(image_list_path)] if image_list_path else [] ),
    ]


def build_patch_match_command(
    workspace_path: Path,
    *,
    config: V4DenseConfig = V4DenseConfig(),
    config_path: Path | None = None,
    allow_missing_files: bool = False,
) -> list[str]:
    config.validate()
    command = [
        _colmap_executable(),
        "patch_match_stereo",
        "--workspace_path",
        str(workspace_path),
        "--workspace_format",
        config.workspace_format,
        "--PatchMatchStereo.max_image_size",
        str(config.max_image_size),
        "--PatchMatchStereo.gpu_index",
        str(config.gpu_index),
    ]
    # COLMAP 4.2 defaults both switches to true.  Always pass an explicit
    # numeric value so the bounded photometric pre-pass can really disable
    # them, and so the approved geometric + filtered pass is auditable in the
    # runtime log.
    command.extend([
        "--PatchMatchStereo.geom_consistency",
        "1" if config.geom_consistency else "0",
        "--PatchMatchStereo.filter",
        "1" if config.filter else "0",
    ])
    if allow_missing_files:
        # Only the intentionally bounded smoke may omit non-selected source
        # images from a sparse model; the production run remains strict.
        # COLMAP 4.2 requires an explicit value for this option (unlike the
        # bool-switch form accepted by geom_consistency/filter).
        command.extend(["--PatchMatchStereo.allow_missing_files", "1"])
    if config_path is not None:
        command.extend(["--config_path", str(config_path)])
    return command


def build_stereo_fusion_command(
    workspace_path: Path,
    output_path: Path,
    *,
    mask_path: Path,
    max_image_size: int = 2000,
) -> list[str]:
    if max_image_size < 1:
        raise ValueError("max_image_size must be positive")
    return [
        _colmap_executable(),
        "stereo_fusion",
        "--workspace_path",
        str(workspace_path),
        "--workspace_format",
        "COLMAP",
        "--input_type",
        "geometric",
        "--output_type",
        "PLY",
        "--output_path",
        str(output_path),
        "--StereoFusion.mask_path",
        str(mask_path),
        "--StereoFusion.max_image_size",
        str(max_image_size),
    ]


def dense_failure_category(output: str, *, status: str = "failed") -> str:
    text = str(output).lower()
    if any(token in text for token in ("out of memory", "cudaerrormemoryallocation", "bad_alloc", "memory allocation")):
        return "cuda_oom"
    if any(token in text for token in ("no kernel image", "requires cuda", "not compiled", "cuda driver")):
        return "capability"
    if status == "timeout":
        return "timeout"
    return "runtime"


def resource_fallback(
    config: V4DenseConfig,
    *,
    failure_category: str,
) -> V4DenseConfig:
    """Permit only the approved one-step CUDA OOM resolution."""

    config.validate()
    if failure_category != "cuda_oom" or config.max_image_size != 2000:
        raise ValueError("V4 dense fallback requires one diagnosed CUDA OOM at max_image_size=2000")
    return V4DenseConfig(
        max_image_size=1600,
        gpu_index=config.gpu_index,
        geom_consistency=config.geom_consistency,
        filter=config.filter,
        input_type=config.input_type,
        workspace_format=config.workspace_format,
    )


def _measured_fraction(metric: Any, name: str) -> float:
    """Validate one explicit measured contamination fraction."""

    if not isinstance(metric, Mapping):
        raise ValueError(f"{name} must be an explicit measured-metric object")
    value = metric.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}.value must be numeric")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name}.value must be a finite fraction in [0, 1]")
    method = str(metric.get("measurement_method", "")).strip()
    evidence = str(metric.get("evidence_sha256", "")).strip().lower()
    if not method:
        raise ValueError(f"{name}.measurement_method is required")
    if len(evidence) != 64 or any(char not in "0123456789abcdef" for char in evidence):
        raise ValueError(f"{name}.evidence_sha256 must be a SHA-256 digest")
    return value


def _exact_bool(value: Any, expected: bool) -> bool:
    return type(value) is bool and value is expected


def _fraction_input_matches(value: Any, measured: float | None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return measured is not None and math.isfinite(numeric) and numeric == measured


def _canonical_sha256(payload: Any) -> str:
    """Hash JSON evidence using the same canonical encoding as post-fusion."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: Any) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def postfusion_evidence_gate(
    report: Mapping[str, Any] | None,
    *,
    fused_path: Path | None = None,
    expected_fused_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the evidence contract emitted after geometric fusion.

    This is deliberately separate from :func:`dense_gate`, which evaluates
    numerical PLY metrics and a human review object.  A dense visual acceptance
    must also prove that those metrics came from the current fused cloud,
    projected through multiple registered camera/mask views, and that the
    source graph and ring-transition depth checks were actually evaluated.
    Missing, stale, or status-only evidence fails closed.
    """

    checks = {
        "report_object_present": isinstance(report, Mapping),
        "postfusion_report_status_passed": False,
        "fused_cloud_hash_verified": False,
        "semantic_preview_evidence_verified": False,
        "fused_cloud_multiview_contamination": False,
        "fusion_mask_resolution_verified": False,
        "contamination_metrics_provenance_verified": False,
        "contamination_findings_explicit_and_clean": False,
        "source_selection_evidence_verified": False,
        "ring_transition_evidence_verified": False,
    }
    reasons: list[str] = []
    if not isinstance(report, Mapping):
        reasons.append("post-fusion evidence report is missing or is not an object")
        return {"passed": False, "checks": checks, "reasons": reasons}
    if report.get("status") != "passed":
        reasons.append("post-fusion evidence report status is not passed")
    checks["postfusion_report_status_passed"] = report.get("status") == "passed"

    report_hash = str(report.get("fused_sha256", "")).strip().lower()
    actual_hash = ""
    if not _valid_sha256(report_hash):
        reasons.append("post-fusion fused_sha256 is missing or invalid")
    if fused_path is not None:
        target = Path(fused_path)
        if not target.is_file():
            reasons.append(f"post-fusion fused cloud is missing: {target}")
        elif _valid_sha256(report_hash):
            actual_hash = sha256_file(target)
            if actual_hash != report_hash:
                reasons.append("post-fusion evidence hash does not match the fused cloud bytes")
    if expected_fused_sha256 is not None:
        expected = str(expected_fused_sha256).strip().lower()
        if not _valid_sha256(expected) or report_hash != expected:
            reasons.append("post-fusion evidence hash does not match the expected fused-cloud hash")
    checks["fused_cloud_hash_verified"] = bool(
        _valid_sha256(report_hash)
        and (fused_path is None or actual_hash == report_hash)
        and (expected_fused_sha256 is None or report_hash == str(expected_fused_sha256).strip().lower())
    )

    semantic = report.get("semantic_previews")
    semantic_paths: set[str] = set()
    semantic_hashes: set[str] = set()
    if not isinstance(semantic, Mapping) or set(semantic) != set(SEMANTIC_VIEW_KEYS):
        reasons.append("post-fusion evidence must contain exactly four semantic fused-cloud previews")
    else:
        for key in SEMANTIC_VIEW_KEYS:
            item = semantic.get(key)
            if not isinstance(item, Mapping):
                reasons.append(f"post-fusion semantic preview evidence is missing for {key}")
                continue
            path = Path(str(item.get("path", "")))
            digest = str(item.get("sha256", "")).strip().lower()
            if not path.is_file() or not _valid_sha256(digest):
                reasons.append(f"post-fusion semantic preview evidence is invalid for {key}")
                continue
            try:
                actual = sha256_file(path)
            except OSError as error:
                reasons.append(f"post-fusion semantic preview cannot be read for {key}: {error}")
                continue
            if actual != digest:
                reasons.append(f"post-fusion semantic preview hash mismatch for {key}")
                continue
            semantic_paths.add(str(path.resolve()))
            semantic_hashes.add(digest)
        if len(semantic_paths) != len(SEMANTIC_VIEW_KEYS) or len(semantic_hashes) != len(SEMANTIC_VIEW_KEYS):
            reasons.append("post-fusion semantic previews must be four distinct hash-verified files")
    checks["semantic_preview_evidence_verified"] = len(semantic_paths) == len(SEMANTIC_VIEW_KEYS) and len(semantic_hashes) == len(SEMANTIC_VIEW_KEYS)

    contamination = report.get("contamination")
    contamination_metrics = contamination.get("metrics") if isinstance(contamination, Mapping) else None
    contamination_evidence = contamination.get("evidence") if isinstance(contamination, Mapping) else None
    contamination_findings = contamination.get("findings") if isinstance(contamination, Mapping) else None
    metric_hash = _canonical_sha256(contamination_evidence) if isinstance(contamination_evidence, Mapping) else ""
    metric_errors: list[str] = []
    if not isinstance(contamination, Mapping) or contamination.get("status") != "measured":
        reasons.append("post-fusion contamination evidence is missing or not measured")
    if not isinstance(contamination_metrics, Mapping):
        reasons.append("post-fusion contamination metrics are missing")
    else:
        for key in CONTAMINATION_METRIC_KEYS:
            item = contamination_metrics.get(key)
            try:
                value = _measured_fraction(item, key)
            except ValueError as error:
                metric_errors.append(str(error))
                continue
            if not isinstance(item, Mapping) or not _nonnegative_int(item.get("count")) or not _nonnegative_int(item.get("denominator")):
                metric_errors.append(f"{key} must record integer count and denominator provenance")
            elif int(item["denominator"]) < 1 or int(item["count"]) > int(item["denominator"]):
                metric_errors.append(f"{key} count/denominator provenance is invalid")
            elif abs(float(item["count"]) / float(item["denominator"]) - value) > 1e-12:
                metric_errors.append(f"{key} value does not match its count/denominator provenance")
            if not metric_hash or str(item.get("evidence_sha256", "")).strip().lower() != metric_hash:
                metric_errors.append(f"{key} evidence hash does not bind to the contamination evidence object")
    if metric_errors:
        reasons.extend(metric_errors)
    if not isinstance(contamination_evidence, Mapping):
        reasons.append("post-fusion contamination evidence details are missing")
    else:
        evidence_hash = str(contamination_evidence.get("fused_sha256", "")).strip().lower()
        if not _valid_sha256(evidence_hash) or evidence_hash != report_hash:
            reasons.append("contamination evidence is not bound to the reported fused cloud")
    camera_count = contamination.get("camera_count") if isinstance(contamination, Mapping) else None
    ring_counts = contamination.get("ring_counts") if isinstance(contamination, Mapping) else None
    multiview = contamination_evidence.get("multi_view_projection") if isinstance(contamination_evidence, Mapping) else None
    sample_count = contamination_evidence.get("sample_count") if isinstance(contamination_evidence, Mapping) else None
    visible_count = contamination_evidence.get("visible_count") if isinstance(contamination_evidence, Mapping) else None
    mask_count = contamination_evidence.get("mask_count") if isinstance(contamination_evidence, Mapping) else None
    mask_resolution = contamination_evidence.get("mask_resolution") if isinstance(contamination_evidence, Mapping) else None
    quantiles = contamination_evidence.get("inside_fraction_quantiles") if isinstance(contamination_evidence, Mapping) else None
    resolution_counts = mask_resolution.get("resolution_counts") if isinstance(mask_resolution, Mapping) else None
    mask_resolution_valid = (
        isinstance(mask_resolution, Mapping)
        and mask_resolution.get("resolver") == "colmap_image_name_plus_png"
        and type(mask_resolution.get("resolved_count")) is int
        and mask_resolution.get("resolved_count") == camera_count
        and type(mask_resolution.get("missing_count")) is int
        and mask_resolution.get("missing_count") == 0
        and type(mask_resolution.get("legacy_fallback_count")) is int
        and mask_resolution.get("legacy_fallback_count") == 0
        and isinstance(resolution_counts, Mapping)
        and type(resolution_counts.get("colmap_image_name_plus_png")) is int
        and resolution_counts.get("colmap_image_name_plus_png") == camera_count
    )
    if not mask_resolution_valid:
        reasons.append("post-fusion masks are not proven to use COLMAP's image-name-plus-.png resolver")
    checks["fusion_mask_resolution_verified"] = bool(mask_resolution_valid)
    multiview_valid = (
        type(camera_count) is int
        and camera_count >= 4
        and isinstance(multiview, Mapping)
        and multiview.get("aggregation") == "all_registered_camera_views"
        and type(multiview.get("view_count")) is int
        and multiview.get("view_count") == camera_count
        and type(multiview.get("mask_count")) is int
        and multiview.get("mask_count") == camera_count
        and type(mask_count) is int
        and mask_count == camera_count
        and mask_resolution_valid
        and type(sample_count) is int
        and sample_count > 0
        and type(visible_count) is int
        and 0 < visible_count <= sample_count
        and isinstance(quantiles, list)
        and len(quantiles) == 3
        and all(_finite_number(value) for value in quantiles)
        and isinstance(ring_counts, Mapping)
        and len(ring_counts) >= 2
        and all(_nonnegative_int(value) and value > 0 for value in ring_counts.values())
        and sum(int(value) for value in ring_counts.values()) == camera_count
    )
    if not multiview_valid:
        reasons.append("post-fusion contamination must be measured from all registered camera/mask views with non-empty samples")
    checks["fused_cloud_multiview_contamination"] = bool(multiview_valid)
    checks["contamination_metrics_provenance_verified"] = bool(
        contamination.get("status") == "measured" if isinstance(contamination, Mapping) else False
    ) and not metric_errors and isinstance(contamination_evidence, Mapping) and _valid_sha256(metric_hash)

    clean_findings = {
        "board_slab_detected": False,
        "pedestal_board_webbing_detected": False,
        "cloth_or_background_structure_detected": False,
        "vessel_identity_confirmed": True,
    }
    findings_valid = (
        isinstance(contamination_findings, Mapping)
        and set(contamination_findings) == set(CONTAMINATION_FINDING_KEYS)
        and all(type(contamination_findings[key]) is bool for key in CONTAMINATION_FINDING_KEYS)
        and dict(contamination_findings) == clean_findings
    )
    if not findings_valid:
        reasons.append("post-fusion contamination findings must be explicit and clean")
    checks["contamination_findings_explicit_and_clean"] = findings_valid

    source = report.get("source_selection_audit")
    source_observed = source.get("observed") if isinstance(source, Mapping) else None
    source_configs = source.get("configs") if isinstance(source, Mapping) else None
    source_errors = source.get("errors") if isinstance(source, Mapping) else None
    source_config_hashes_valid = isinstance(source_configs, list) and bool(source_configs)
    if source_config_hashes_valid:
        for item in source_configs:
            if not isinstance(item, Mapping):
                source_config_hashes_valid = False
                continue
            path = Path(str(item.get("path", "")))
            digest = str(item.get("sha256", "")).strip().lower()
            if not path.is_file() or not _valid_sha256(digest):
                source_config_hashes_valid = False
                continue
            try:
                if sha256_file(path) != digest:
                    source_config_hashes_valid = False
            except OSError:
                source_config_hashes_valid = False
    observed_counts_valid = (
        isinstance(source_observed, Mapping)
        and _nonnegative_int(source_observed.get("reference_count"))
        and int(source_observed.get("reference_count", 0)) > 0
        and _nonnegative_int(source_observed.get("registered_image_count"))
        and int(source_observed.get("registered_image_count", 0)) > 0
        and int(source_observed.get("reference_count")) == int(source_observed.get("registered_image_count"))
        and _nonnegative_int(source_observed.get("configured_reference_count_total"))
        and int(source_observed.get("configured_reference_count_total", 0)) > 0
        and _nonnegative_int(source_observed.get("directed_source_count"))
        and int(source_observed.get("directed_source_count", 0)) > 0
        and _nonnegative_int(source_observed.get("cross_ring_directed_source_count"))
        and _nonnegative_int(source_observed.get("references_without_cross_ring_source_count"))
    )
    deviation = source.get("deviation_from_original_source_selection") if isinstance(source, Mapping) else None
    deviation_valid = (
        isinstance(deviation, Mapping)
        and deviation.get("original_plan") == "COLMAP automatic/default source selection"
        and deviation.get("deviation_is_intentional") is True
        and type(deviation.get("max_sources")) is int
        and int(deviation.get("max_sources")) > 0
    )
    source_valid = (
        isinstance(source, Mapping)
        and source.get("status") == "passed"
        and isinstance(source_errors, list)
        and not source_errors
        and source_config_hashes_valid
        and observed_counts_valid
        and deviation_valid
    )
    if not source_valid:
        reasons.append("final tile source-selection evidence is missing, stale, or not hash-verified")
    checks["source_selection_evidence_verified"] = bool(source_valid)

    ring = report.get("ring_transition_audit")
    ring_evidence = ring.get("evidence") if isinstance(ring, Mapping) else None
    transitions = ring.get("transitions") if isinstance(ring, Mapping) else None
    ring_valid = (
        isinstance(ring, Mapping)
        and ring.get("status") == "passed"
        and ring.get("passed") is True
        and type(ring.get("transition_count")) is int
        and int(ring.get("transition_count")) > 0
        and isinstance(transitions, Mapping)
        and len(transitions) == int(ring.get("transition_count"))
        and isinstance(ring_evidence, Mapping)
        and ring_evidence.get("mask_resolver") == "colmap_image_name_plus_png"
        and str(ring_evidence.get("pair_measurement_method", "")).strip() == "project_reference_geometric_depth_into_source_and_compare_masked_geometric_depth"
        and type(ring_evidence.get("evaluated_pair_count")) is int
        and int(ring_evidence.get("evaluated_pair_count")) > 0
        and Path(str(ring_evidence.get("depth_dir", ""))).is_dir()
        and Path(str(ring_evidence.get("mask_dir", ""))).is_dir()
        and Path(str(ring_evidence.get("sparse_model_path", ""))).is_dir()
    )
    measured_pair_count = 0
    if isinstance(transitions, Mapping):
        for transition_name, item in transitions.items():
            if not isinstance(item, Mapping):
                ring_valid = False
                reasons.append(f"ring transition evidence is invalid for {transition_name}")
                continue
            pair_count = item.get("pair_count")
            evaluated = item.get("evaluated_pairs")
            measurements = item.get("measurements")
            selected_pairs = item.get("selected_pairs")
            transition_ok = (
                item.get("status") == "passed"
                and _nonnegative_int(pair_count)
                and int(pair_count) > 0
                and _nonnegative_int(evaluated)
                and int(evaluated) > 0
                and int(evaluated) <= int(pair_count)
                and isinstance(measurements, list)
                and len(measurements) == int(evaluated)
                and isinstance(selected_pairs, list)
                and len(selected_pairs) == int(evaluated)
            )
            if transition_ok:
                for measurement in measurements:
                    hashes_ok = isinstance(measurement, Mapping) and all(
                        _valid_sha256(measurement.get(key))
                        for key in (
                            "reference_depth_sha256",
                            "source_depth_sha256",
                            "reference_mask_sha256",
                            "source_mask_sha256",
                        )
                    )
                    files_match = True
                    if isinstance(ring_evidence, Mapping) and isinstance(measurement, Mapping):
                        depth_root = Path(str(ring_evidence.get("depth_dir", "")))
                        mask_root = Path(str(ring_evidence.get("mask_dir", "")))
                        reference_name = Path(str(measurement.get("reference", ""))).name
                        source_name = Path(str(measurement.get("source", ""))).name
                        expected_mask_paths = (
                            mask_root / f"{reference_name}.png",
                            mask_root / f"{source_name}.png",
                        )
                        expected_files = (
                            (depth_root / f"{reference_name}.geometric.bin", measurement.get("reference_depth_sha256")),
                            (depth_root / f"{source_name}.geometric.bin", measurement.get("source_depth_sha256")),
                            (expected_mask_paths[0], measurement.get("reference_mask_sha256")),
                            (expected_mask_paths[1], measurement.get("source_mask_sha256")),
                        )
                        try:
                            files_match = all(path.is_file() and sha256_file(path) == str(expected).strip().lower() for path, expected in expected_files)
                            files_match = files_match and measurement.get("reference_mask_resolution") == "colmap_image_name_plus_png"
                            files_match = files_match and measurement.get("source_mask_resolution") == "colmap_image_name_plus_png"
                            files_match = files_match and Path(str(measurement.get("reference_mask_path", ""))).resolve() == expected_mask_paths[0].resolve()
                            files_match = files_match and Path(str(measurement.get("source_mask_path", ""))).resolve() == expected_mask_paths[1].resolve()
                        except OSError:
                            files_match = False
                    values_ok = isinstance(measurement, Mapping) and all(
                        _finite_number(measurement.get(key))
                        for key in (
                            "coverage_fraction",
                            "consistent_fraction_at_1pct",
                            "relative_depth_median",
                            "relative_depth_p95",
                        )
                    )
                    ranges_ok = (
                        isinstance(measurement, Mapping)
                        and 0.0 <= float(measurement.get("coverage_fraction", -1.0)) <= 1.0
                        and 0.0 <= float(measurement.get("consistent_fraction_at_1pct", -1.0)) <= 1.0
                        and float(measurement.get("relative_depth_median", -1.0)) >= 0.0
                        and float(measurement.get("relative_depth_p95", -1.0)) >= 0.0
                    ) if values_ok else False
                    counts_ok = (
                        isinstance(measurement, Mapping)
                        and _nonnegative_int(measurement.get("sampled"))
                        and _nonnegative_int(measurement.get("overlap_count"))
                        and int(measurement.get("sampled", 0)) > 0
                        and 0 < int(measurement.get("overlap_count", 0)) <= int(measurement.get("sampled", 0))
                    )
                    if not (isinstance(measurement, Mapping) and measurement.get("status") == "measured" and hashes_ok and files_match and values_ok and ranges_ok and counts_ok):
                        transition_ok = False
                        break
                measured_pair_count += int(evaluated) if transition_ok else 0
            if not transition_ok:
                ring_valid = False
                reasons.append(f"ring transition evidence is incomplete for {transition_name}")
    evaluated_pair_count = ring_evidence.get("evaluated_pair_count") if isinstance(ring_evidence, Mapping) else None
    if not isinstance(ring_evidence, Mapping) or not _nonnegative_int(evaluated_pair_count) or measured_pair_count != int(evaluated_pair_count):
        ring_valid = False
    if not ring_valid:
        reasons.append("fused ring-transition continuity evidence is missing or not measured from geometric depth pairs")
    checks["ring_transition_evidence_verified"] = bool(ring_valid)

    return {"passed": bool(all(checks.values())), "checks": checks, "reasons": reasons}


def contamination_assessment(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate contamination from explicit measurements and exact booleans."""

    raw_metrics = metrics.get("contamination_metrics")
    measured: dict[str, float | None] = {key: None for key in CONTAMINATION_METRIC_KEYS}
    metric_errors: list[str] = []
    if isinstance(raw_metrics, Mapping):
        for key in CONTAMINATION_METRIC_KEYS:
            try:
                measured[key] = _measured_fraction(raw_metrics.get(key), key)
            except ValueError as error:
                metric_errors.append(str(error))
    else:
        metric_errors.append("contamination_metrics is missing or is not an object")

    findings_input = metrics.get("contamination_findings")
    findings: dict[str, bool | None] = {key: None for key in CONTAMINATION_FINDING_KEYS}
    finding_errors: list[str] = []
    if isinstance(findings_input, Mapping):
        for key in CONTAMINATION_FINDING_KEYS:
            value = findings_input.get(key)
            if type(value) is not bool:
                finding_errors.append(f"{key} must be an explicit boolean")
            else:
                findings[key] = value
    else:
        finding_errors.append("contamination_findings is missing or is not an object")

    board = measured["board_point_fraction"]
    webbing = measured["pedestal_board_webbing_point_fraction"]
    cloth = measured["cloth_or_background_point_fraction"]
    top_level_board = metrics.get("board_point_fraction")
    top_level_cloth = metrics.get("cloth_point_fraction")
    board_inputs_consistent = "board_point_fraction" not in metrics or _fraction_input_matches(top_level_board, board)
    cloth_inputs_consistent = "cloth_point_fraction" not in metrics or _fraction_input_matches(top_level_cloth, cloth)
    checks = {
        "contamination_metrics_explicit": not metric_errors,
        "board_not_dominant": board is not None and board < 0.20,
        "pedestal_board_webbing_not_dominant": webbing is not None and webbing < 0.05,
        "cloth_or_background_not_dominant": cloth is not None and cloth < 0.10,
        "board_fraction_inputs_consistent": board_inputs_consistent,
        "cloth_fraction_inputs_consistent": cloth_inputs_consistent,
        "board_slab_absent": findings["board_slab_detected"] is False,
        "pedestal_board_webbing_absent": findings["pedestal_board_webbing_detected"] is False,
        "cloth_or_background_absent": findings["cloth_or_background_structure_detected"] is False,
        "vessel_identity_confirmed": findings["vessel_identity_confirmed"] is True,
        "contamination_findings_explicit": not finding_errors,
    }
    reasons = [*metric_errors, *finding_errors]
    if not board_inputs_consistent:
        reasons.append("board_point_fraction does not match the measured board fraction")
    if not cloth_inputs_consistent:
        reasons.append("cloth_point_fraction does not match the measured cloth/background fraction")
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "measured_fractions": measured,
        "reasons": reasons,
    }


def reviewed_evidence_gate(review: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate independent, hash-anchored visual review evidence."""

    if not isinstance(review, Mapping):
        return {
            "passed": False,
            "checks": {"review_object_present": False},
            "reasons": ["reviewed evidence object is missing"],
            "preview_evidence": {},
        }

    reasons: list[str] = []
    previews_input = review.get("previews")
    preview_evidence: dict[str, dict[str, str]] = {}
    previews_valid = isinstance(previews_input, Mapping) and set(previews_input) == set(SEMANTIC_VIEW_KEYS)
    if previews_valid:
        for key in SEMANTIC_VIEW_KEYS:
            item = previews_input[key]
            if not isinstance(item, Mapping):
                reasons.append(f"{key} preview evidence is missing")
                continue
            path = str(item.get("path", ""))
            expected_hash = str(item.get("sha256", "")).strip().lower()
            try:
                if not path:
                    raise ValueError(f"{key} preview path is empty")
                if len(expected_hash) != 64 or any(char not in "0123456789abcdef" for char in expected_hash):
                    raise ValueError(f"{key} preview SHA-256 is invalid")
                actual_hash = sha256_file(path)
                if actual_hash != expected_hash:
                    raise ValueError(f"{key} preview bytes do not match the recorded SHA-256")
                preview_evidence[key] = {
                    "path": str(Path(path).resolve()),
                    "sha256": actual_hash,
                }
            except (OSError, ValueError) as error:
                reasons.append(str(error))
    else:
        reasons.append("previews must contain exactly front, quarter, side, and top_oblique evidence")
    previews_distinct = len({item["path"] for item in preview_evidence.values()}) == len(SEMANTIC_VIEW_KEYS) and len(
        {item["sha256"] for item in preview_evidence.values()}
    ) == len(SEMANTIC_VIEW_KEYS)
    if previews_valid and not previews_distinct:
        reasons.append("four semantic preview views must be distinct files with distinct bytes")

    regions = review.get("regions")
    regions_valid = (
        isinstance(regions, Mapping)
        and set(regions) == set(REQUIRED_REVIEW_REGION_KEYS)
        and all(type(value) is bool for value in regions.values())
        and all(value is True for value in regions.values())
    )
    if not regions_valid:
        reasons.append(
            "regions must contain exactly bowl_and_interior, globe_or_shoulder, neck_lid_or_finial, and pedestal_and_base as explicit true booleans"
        )

    findings = review.get("contamination_findings")
    findings_valid = (
        isinstance(findings, Mapping)
        and set(findings) == set(CONTAMINATION_FINDING_KEYS)
        and _exact_bool(findings.get("board_slab_detected"), False)
        and _exact_bool(findings.get("pedestal_board_webbing_detected"), False)
        and _exact_bool(findings.get("cloth_or_background_structure_detected"), False)
        and _exact_bool(findings.get("vessel_identity_confirmed"), True)
    )
    if not findings_valid:
        reasons.append("review contamination booleans must explicitly exclude contamination and confirm identity")

    reviewer = str(review.get("reviewer", "")).strip()
    approver = str(review.get("approver", "")).strip()
    review_status = str(review.get("review_status", "")).strip()
    approval_status = str(review.get("approval_status", "")).strip()
    reviewer_valid = bool(reviewer)
    approver_valid = bool(approver)
    independent_approval = reviewer_valid and approver_valid and reviewer.casefold() != approver.casefold()
    review_status_valid = review_status == "passed"
    approval_status_valid = approval_status == "passed"
    if not reviewer_valid:
        reasons.append("reviewer identity is missing")
    if not approver_valid:
        reasons.append("approver identity is missing")
    if reviewer_valid and approver_valid and not independent_approval:
        reasons.append("self-approval is not accepted")
    if not review_status_valid:
        reasons.append("review status must be passed, not missing or pending")
    if not approval_status_valid:
        reasons.append("approval status must be passed, not missing or pending")

    checks = {
        "review_object_present": True,
        "four_semantic_views": previews_valid and len(preview_evidence) == len(SEMANTIC_VIEW_KEYS),
        "preview_hashes_verified": previews_valid and len(preview_evidence) == len(SEMANTIC_VIEW_KEYS),
        "preview_views_distinct": previews_distinct,
        "required_regions_confirmed": regions_valid,
        "contamination_findings_explicit": findings_valid,
        "reviewer_identified": reviewer_valid,
        "approval_independent": independent_approval,
        "review_status_passed": review_status_valid,
        "approval_status_passed": approval_status_valid,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "reasons": reasons,
        "preview_evidence": preview_evidence,
        "reviewer": reviewer,
        "approver": approver,
        "regions": dict(regions) if isinstance(regions, Mapping) else {},
    }


def dense_gate(
    metrics: Mapping[str, Any],
    *,
    smoke: Mapping[str, Any],
    visual_status: str,
    review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contamination = contamination_assessment(metrics)
    evidence = reviewed_evidence_gate(review)
    checks = {
        "finite_xyz": float(metrics.get("finite_xyz_fraction", 0.0)) == 1.0,
        "nonzero_points": int(metrics.get("point_count", 0)) > 0,
        "rank_3": int(metrics.get("rank", 0)) == 3,
        "cuda_smoke_passed": smoke.get("status") == "passed" and int(smoke.get("depth_normal_count", 0)) > 0,
        "visual_review": visual_status == "passed" and evidence["passed"],
    }
    checks.update(contamination["checks"])
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "metrics": dict(metrics),
        "smoke": dict(smoke),
        "contamination": contamination,
        "review": evidence,
    }


def write_dense_report(payload: Mapping[str, Any], path: Path | None = None) -> Path:
    return write_json(path or RECONSTRUCTION_V4_ROOT / "reports" / "dense_gate.json", dict(payload))


def dense_workspace_paths(
    tag: str = "workspace",
    *,
    workspace_root: Path | None = None,
) -> dict[str, Path]:
    """Return the bounded V4 dense workspace layout for one attempt.

    The normal default is the project-local ``reconstruction/v4/dense``
    boundary.  A caller may provide the explicitly authorized D: scratch root
    for large CUDA intermediates; reports and final products remain project
    local, and the caller is responsible for validating that fixed root before
    invoking this helper.
    """

    if not tag or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in tag):
        raise ValueError("dense workspace tag contains unsupported characters")
    if workspace_root is None:
        root = assert_output_path(RECONSTRUCTION_V4_ROOT / "dense" / tag)
    else:
        base = Path(workspace_root).expanduser()
        if not base.is_absolute():
            raise ValueError("external dense workspace root must be absolute")
        root = (base / tag).resolve(strict=False)
    return {
        "root": root,
        "images": root / "images",
        "sparse": root / "sparse" / "0",
        "masks": root / "masks",
        "stereo": root / "stereo",
        "depth_maps": root / "stereo" / "depth_maps",
        "normal_maps": root / "stereo" / "normal_maps",
        "fused": RECONSTRUCTION_V4_ROOT / "dense" / f"fused_{tag}.ply",
    }


def write_dense_image_list(names: Sequence[str], path: Path) -> Path:
    """Write an explicit COLMAP image-list boundary, rejecting traversal."""

    values = [str(name).replace("\\", "/") for name in names]
    if not values or any(not value or Path(value).name != value for value in values):
        raise ValueError("dense image list must contain non-empty basenames only")
    if len(values) != len(set(values)):
        raise ValueError("dense image list contains duplicate image names")
    target = assert_output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")
    return target


def write_dense_smoke_config(
    path: Path,
    image_names: Sequence[str],
    *,
    source_count: int = 3,
) -> Path:
    """Write a bounded PatchMatch config whose dependencies are all in the smoke set.

    COLMAP's image undistorter keeps the complete sparse model even when an
    image list selects only a small smoke subset. An explicit cyclic neighbor
    list lets the smoke exercise both the photometric pre-pass and the approved
    geometric/filter pass without silently depending on unselected maps.
    """

    if source_count < 1:
        raise ValueError("dense smoke source_count must be positive")
    normalized: list[str] = []
    for name in image_names:
        value = str(name).replace("\\", "/")
        candidate = Path(value)
        if candidate.name != value or candidate.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError(f"invalid smoke image basename: {name!r}")
        normalized.append(candidate.name)
    if len(normalized) < 2:
        raise ValueError("dense smoke config requires at least two images")
    if len(set(normalized)) != len(normalized):
        raise ValueError("dense smoke image names must be unique")
    target = assert_output_path(path)
    if target.exists():
        raise FileExistsError(f"dense smoke config already exists; preserve it before retrying: {target}")
    count = min(int(source_count), len(normalized) - 1)
    lines: list[str] = []
    for index, reference in enumerate(normalized):
        sources = [normalized[(index + offset) % len(normalized)] for offset in range(1, count + 1)]
        lines.extend([reference, ",".join(sources)])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def build_dense_pair_adjacency(
    image_names: Sequence[str],
    pairs: Sequence[Sequence[str]],
    *,
    max_sources: int | None = None,
) -> dict[str, tuple[str, ...]]:
    """Build a deterministic bidirectional PatchMatch source graph.

    The V4 pairing schedule is the acquisition-aware source of truth for
    dense matching.  Expanding each undirected pair in both directions keeps
    the dense stage independent of COLMAP's ``__auto__`` selection (which
    otherwise refers to the entire sparse model, including images omitted by
    a smoke list) while retaining all scheduled nearby/cross-ring links.
    """

    normalized: list[str] = []
    for name in image_names:
        value = str(name).replace("\\", "/")
        candidate = Path(value)
        if candidate.name != value or candidate.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError(f"invalid dense image basename: {name!r}")
        normalized.append(candidate.name)
    if len(normalized) < 2 or len(set(normalized)) != len(normalized):
        raise ValueError("dense image names must be unique and contain at least two images")
    if max_sources is not None and max_sources < 1:
        raise ValueError("dense max_sources must be positive when provided")
    allowed = set(normalized)
    adjacency: dict[str, list[str]] = {name: [] for name in normalized}
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"dense pair must contain exactly two image names: {pair!r}")
        left, right = (str(value).replace("\\", "/") for value in pair)
        if left not in allowed or right not in allowed or left == right:
            raise ValueError(f"dense pair contains an unknown or self image: {pair!r}")
        if right not in adjacency[left]:
            adjacency[left].append(right)
        if left not in adjacency[right]:
            adjacency[right].append(left)
    # Preserve the manifest order rather than pair-file ordering.  This makes
    # chunking and cleanup reproducible across retries.
    order = {name: index for index, name in enumerate(normalized)}
    result = {
        name: tuple(sorted(sources, key=order.__getitem__)[:max_sources])
        for name, sources in adjacency.items()
    }
    isolated = [name for name, sources in result.items() if not sources]
    if isolated:
        raise ValueError(f"dense pair graph has isolated images: {isolated[:5]}")
    return result


def build_prioritized_dense_pair_adjacency(
    image_names: Sequence[str],
    pairs: Sequence[Sequence[str]],
    *,
    ring_by_name: Mapping[str, str],
    phase_by_name: Mapping[str, float] | None = None,
    max_sources: int = 6,
    min_cross_sources: int = 1,
) -> dict[str, tuple[str, ...]]:
    """Build a bounded graph with explicit local and cross-ring guarantees.

    The original V4 graph is retained for provenance and for the healthy
    candidate.  A refinement rerun must not rely on manifest order when the
    source cap is applied: if local and cross-ring candidates exist, one local
    source and ``min_cross_sources`` cross-ring sources are reserved before the
    remaining slots are filled by phase-nearest candidates.  This helper is
    intentionally separate so an affected-only rerun cannot silently alter the
    accepted source graph in place.
    """

    if max_sources < 2:
        raise ValueError("prioritized dense source cap must allow local and cross-ring support")
    if min_cross_sources < 1:
        raise ValueError("prioritized dense graph requires at least one cross-ring source")
    if max_sources < 1 + min_cross_sources:
        raise ValueError("prioritized dense source cap is too small for the requested local and cross-ring guarantees")
    normalized: list[str] = []
    for name in image_names:
        value = str(name).replace("\\", "/")
        candidate = Path(value)
        if candidate.name != value or candidate.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError(f"invalid prioritized dense image basename: {name!r}")
        normalized.append(candidate.name)
    if len(normalized) < 2 or len(set(normalized)) != len(normalized):
        raise ValueError("prioritized dense image names must be unique and contain at least two images")
    if any(name not in ring_by_name for name in normalized):
        missing = [name for name in normalized if name not in ring_by_name]
        raise ValueError(f"prioritized dense ring metadata is missing: {missing[:5]}")

    allowed = set(normalized)
    adjacency: dict[str, set[str]] = {name: set() for name in normalized}
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"dense pair must contain exactly two image names: {pair!r}")
        left, right = (str(value).replace("\\", "/") for value in pair)
        left, right = Path(left).name, Path(right).name
        if left not in allowed or right not in allowed or left == right:
            raise ValueError(f"dense pair contains an unknown or self image: {pair!r}")
        adjacency[left].add(right)
        adjacency[right].add(left)

    order = {name: index for index, name in enumerate(normalized)}
    phases = {name: float(value) for name, value in (phase_by_name or {}).items() if name in allowed}

    def phase_distance(reference: str, source: str) -> float:
        left, right = phases.get(reference), phases.get(source)
        if left is None or right is None or not math.isfinite(left) or not math.isfinite(right):
            return float(abs(order[reference] - order[source]))
        delta = abs((right - left) % 1.0)
        return float(min(delta, 1.0 - delta))

    result: dict[str, tuple[str, ...]] = {}
    for reference in normalized:
        candidates = sorted(adjacency[reference], key=lambda value: (phase_distance(reference, value), order[value]))
        same = [value for value in candidates if str(ring_by_name[value]) == str(ring_by_name[reference])]
        cross = [value for value in candidates if str(ring_by_name[value]) != str(ring_by_name[reference])]
        if not candidates:
            raise ValueError(f"prioritized dense graph has an isolated image: {reference}")
        selected: list[str] = []
        if same:
            selected.append(same.pop(0))
        if cross:
            if len(cross) < min_cross_sources:
                raise ValueError(
                    f"prioritized dense graph has fewer than {min_cross_sources} cross-ring candidates: {reference}"
                )
            selected.extend(cross[:min_cross_sources])
            cross = cross[min_cross_sources:]
        remaining = sorted(
            same + cross,
            key=lambda value: (
                0 if str(ring_by_name[value]) == str(ring_by_name[reference]) else 1,
                phase_distance(reference, value),
                order[value],
            ),
        )
        selected.extend(remaining)
        selected = selected[:max_sources]
        had_same = any(str(ring_by_name[value]) == str(ring_by_name[reference]) for value in candidates)
        had_cross = any(str(ring_by_name[value]) != str(ring_by_name[reference]) for value in candidates)
        has_same = any(str(ring_by_name[value]) == str(ring_by_name[reference]) for value in selected)
        has_cross = any(str(ring_by_name[value]) != str(ring_by_name[reference]) for value in selected)
        if had_same and not has_same:
            raise ValueError(f"prioritized dense cap dropped local support: {reference}")
        if had_cross and not has_cross:
            raise ValueError(f"prioritized dense cap dropped cross-ring support: {reference}")
        result[reference] = tuple(selected)
    return result


def write_prioritized_dense_pair_config(
    path: Path,
    image_names: Sequence[str],
    adjacency: Mapping[str, Sequence[str]],
    *,
    reference_names: Sequence[str],
) -> Path:
    """Write a tile config from an already-audited prioritized adjacency."""

    order = {Path(str(name).replace("\\", "/")).name: index for index, name in enumerate(image_names)}
    references = [Path(str(name).replace("\\", "/")).name for name in reference_names]
    if not references or len(set(references)) != len(references):
        raise ValueError("prioritized dense references must be non-empty and unique")
    if any(name not in order or name not in adjacency for name in references):
        raise ValueError("prioritized dense references must belong to the audited adjacency")
    lines: list[str] = []
    for reference in sorted(references, key=order.__getitem__):
        sources = [Path(str(value).replace("\\", "/")).name for value in adjacency[reference]]
        if not sources or reference in sources or len(set(sources)) != len(sources):
            raise ValueError(f"invalid prioritized source list for {reference}")
        lines.extend([reference, ",".join(sources)])
    target = assert_output_path(path)
    if target.exists():
        raise FileExistsError(f"prioritized dense config already exists; preserve it before retrying: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def write_dense_pair_config(
    path: Path,
    image_names: Sequence[str],
    pairs: Sequence[Sequence[str]],
    *,
    reference_names: Sequence[str] | None = None,
    max_sources: int | None = None,
) -> Path:
    """Write explicit COLMAP PatchMatch reference/source lines.

    ``reference_names`` is used for bounded geometric chunks; source names
    may come from any image in the complete undistorted workspace because the
    photometric pre-pass has produced maps for the whole graph first.
    """

    adjacency = build_dense_pair_adjacency(image_names, pairs, max_sources=max_sources)
    order = {str(name).replace("\\", "/"): index for index, name in enumerate(image_names)}
    references = list(adjacency) if reference_names is None else [
        Path(str(name).replace("\\", "/")).name for name in reference_names
    ]
    if not references:
        raise ValueError("dense pair config requires at least one reference image")
    if len(set(references)) != len(references) or any(name not in adjacency for name in references):
        raise ValueError("dense pair config references must be unique members of the image set")
    references.sort(key=order.__getitem__)
    lines: list[str] = []
    for reference in references:
        sources = adjacency[reference]
        if not sources:
            raise ValueError(f"dense reference has no source images: {reference}")
        lines.extend([reference, ",".join(sources)])
    target = assert_output_path(path)
    if target.exists():
        raise FileExistsError(f"dense pair config already exists; preserve it before retrying: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def dense_typed_file_counts(workspace_path: Path) -> dict[str, int]:
    """Count photometric/geometric depth and normal maps separately."""

    stereo = Path(workspace_path) / "stereo"
    depth = stereo / "depth_maps"
    normals = stereo / "normal_maps"
    return {
        "photometric_depth_count": len(tuple(depth.glob("*.photometric.bin"))),
        "photometric_normal_count": len(tuple(normals.glob("*.photometric.bin"))),
        "geometric_depth_count": len(tuple(depth.glob("*.geometric.bin"))),
        "geometric_normal_count": len(tuple(normals.glob("*.geometric.bin"))),
    }


def prune_dense_photometric_maps(
    workspace_path: Path,
    *,
    keep_names: Sequence[str],
    manifest_path: Path,
) -> dict[str, Any]:
    """Remove no-longer-needed photometric inputs with hash/size provenance.

    COLMAP's geometric controller retains the complete photometric pass because
    it is used as the input depth/normal set.  V4's explicit local graph lets
    us release maps only after their last future geometric dependency, keeping
    the approved 2000px run within the available disk budget.  The manifest is
    an auditable record of every removed binary; geometric maps are untouched.
    """

    workspace = Path(workspace_path)
    depth_dir = workspace / "stereo" / "depth_maps"
    normal_dir = workspace / "stereo" / "normal_maps"
    keep = {Path(str(name).replace("\\", "/")).name for name in keep_names}
    manifest_target = assert_output_path(manifest_path)
    payload: dict[str, Any]
    if manifest_target.is_file():
        from v4_config import read_json

        payload = read_json(manifest_target)
    else:
        payload = {
            "schema_version": 1,
            "workspace": str(workspace.resolve()),
            "removed": [],
        }
    removed: list[dict[str, Any]] = []
    for depth_path in sorted(depth_dir.glob("*.photometric.bin")):
        image_name = depth_path.name[: -len(".photometric.bin")]
        if image_name in keep:
            continue
        normal_path = normal_dir / f"{image_name}.photometric.bin"
        if not normal_path.is_file():
            raise RuntimeError(f"photometric depth/normal pair is incomplete: {image_name}")
        record = {
            "image_name": image_name,
            "depth_sha256": sha256_file(depth_path),
            "depth_size_bytes": depth_path.stat().st_size,
            "normal_sha256": sha256_file(normal_path),
            "normal_size_bytes": normal_path.stat().st_size,
        }
        depth_path.unlink()
        normal_path.unlink()
        removed.append(record)
    payload["removed"].extend(removed)
    payload["last_keep_count"] = len(keep)
    payload["removed_count"] = len(payload["removed"])
    payload["remaining_counts"] = dense_typed_file_counts(workspace)
    write_json(manifest_target, payload)
    return {
        "removed_count": len(removed),
        "keep_count": len(keep),
        "remaining_counts": payload["remaining_counts"],
        "manifest_path": str(manifest_target.resolve()),
    }


def _load_mask_array(path: Path) -> np.ndarray:
    from PIL import Image

    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError(f"V4 mask is not single-channel: {path}")
    return np.where(array > 127, 255, 0).astype(np.uint8)


def undistort_v4_masks(
    *,
    sparse_model_path: Path,
    source_mask_dir: Path,
    output_mask_dir: Path,
    image_names: Sequence[str],
    max_image_size: int = 2000,
    external_workspace_root: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Warp vessel masks with the same camera transform as COLMAP images.

    ``image_undistorter`` changes both the image size and camera model.  A plain
    resize of the masks would therefore mis-register pixels near the radial
    boundary.  pyCOLMAP performs the exact COLMAP warp; masks are re-thresholded
    after a nearest-neighbour warp and written with COLMAP's required
    ``<image_name>.png`` mask names so ``StereoFusion.mask_path`` cannot silently
    fall back to an unmasked fusion.
    """

    if max_image_size < 1:
        raise ValueError("max_image_size must be positive")
    sparse_model_path = Path(sparse_model_path)
    source_mask_dir = Path(source_mask_dir)
    if external_workspace_root is None:
        target_dir = assert_output_path(output_mask_dir)
    else:
        scratch_root = Path(external_workspace_root).expanduser()
        if not scratch_root.is_absolute():
            raise ValueError("external dense workspace root must be absolute")
        scratch_root = scratch_root.resolve(strict=False)
        target_dir = Path(output_mask_dir).expanduser().resolve(strict=False)
        try:
            target_dir.relative_to(scratch_root)
        except ValueError as error:
            raise ValueError("undistorted masks must stay inside the external dense workspace") from error
    target_dir.mkdir(parents=True, exist_ok=True)
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(sparse_model_path.resolve()))
    camera_ids = sorted(int(value) for value in reconstruction.cameras.keys())
    if len(camera_ids) != 1:
        raise ValueError(f"V4 dense requires one measured shared camera, found {len(camera_ids)}")
    camera = reconstruction.camera(camera_ids[0])
    options = pycolmap.UndistortCameraOptions()
    options.max_image_size = int(max_image_size)
    options.warp_options.interpolation = pycolmap.WarpImageOptions.Interpolation.NEAREST_NEIGHBOR
    output_camera = pycolmap.undistort_camera(options, camera)
    written: list[dict[str, Any]] = []
    from PIL import Image

    for name in image_names:
        basename = Path(str(name)).name
        source = source_mask_dir / basename
        if not source.is_file():
            raise FileNotFoundError(f"V4 dense source mask is missing: {source}")
        array = _load_mask_array(source)
        bitmap = pycolmap.Bitmap.from_array(array)
        warped, warped_camera = pycolmap.undistort_image(options, bitmap, camera)
        warped_array = np.where(np.asarray(warped.to_array()) > 127, 255, 0).astype(np.uint8)
        target = target_dir / f"{basename}.png"
        # COLMAP resolves a JPEG image mask as ``mask_path / (image_name +
        # '.png')``.  Keep PNG payloads and the explicit appended name so the
        # fusion command actually applies the vessel mask.
        Image.fromarray(warped_array, mode="L").save(target, format="PNG")
        written.append(
            {
                "name": basename,
                "output_name": target.name,
                "source_sha256": sha256_file(source),
                "output_sha256": sha256_file(target),
                "source_shape": [int(value) for value in array.shape],
                "output_shape": [int(value) for value in warped_array.shape],
                "output_nonzero_fraction": float(np.mean(warped_array > 0)),
            }
        )
        if (warped_camera.width, warped_camera.height) != (output_camera.width, output_camera.height):
            raise RuntimeError("pyCOLMAP returned inconsistent undistorted mask camera geometry")
    summary = {
        "status": "complete",
        "sparse_model_path": str(sparse_model_path.resolve()),
        "source_mask_dir": str(source_mask_dir.resolve()),
        "output_mask_dir": str(target_dir.resolve()),
        "image_count": len(written),
        "camera": {
            "model": str(output_camera.model_name),
            "width": int(output_camera.width),
            "height": int(output_camera.height),
            "params": [float(value) for value in np.asarray(output_camera.params).reshape(-1)],
        },
        "records": written,
        "manifest_sha256": fingerprint(written),
    }
    report_path = manifest_path or target_dir.parent / f"{target_dir.name}_manifest.json"
    write_json(report_path, summary)
    return summary


def dense_depth_file_counts(workspace_path: Path) -> dict[str, int]:
    """Count real PatchMatch outputs without treating a directory as proof."""

    stereo = Path(workspace_path) / "stereo"
    depth = stereo / "depth_maps"
    normals = stereo / "normal_maps"
    consistency = stereo / "consistency_graphs"
    return {
        "depth_map_count": len(tuple(depth.glob("*.geometric.bin"))) + len(tuple(depth.glob("*.photometric.bin"))),
        "normal_map_count": len(tuple(normals.glob("*.geometric.bin"))) + len(tuple(normals.glob("*.photometric.bin"))),
        "consistency_graph_count": len(tuple(consistency.glob("*.geometric.bin"))),
    }


def render_dense_contact_sheet(path: Path, output_path: Path, *, max_points: int = 120_000) -> Path:
    """Render PCA projections of the actual fused cloud for the visual gate."""

    if max_points < 100:
        raise ValueError("max_points is too small for a useful dense preview")
    from local_reconstruction_io import read_ply
    from PIL import Image, ImageDraw

    data = read_ply(Path(path))
    xyz = np.asarray(data["xyz"], dtype=np.float64)
    colors = data.get("colors")
    if colors is None:
        colors = np.full((len(xyz), 3), 170, dtype=np.uint8)
    else:
        colors = np.asarray(colors, dtype=np.uint8)
    finite = np.isfinite(xyz).all(axis=1)
    xyz, colors = xyz[finite], colors[finite]
    if len(xyz) == 0:
        raise ValueError("cannot render an empty fused cloud")
    if len(xyz) > max_points:
        indices = np.linspace(0, len(xyz) - 1, num=max_points, dtype=np.int64)
        xyz, colors = xyz[indices], colors[indices]
    center = np.median(xyz, axis=0)
    _, _, vt = np.linalg.svd(xyz - center, full_matrices=False)
    projected = (xyz - center) @ vt[:3].T
    canvas = Image.new("RGB", (1200, 900), (245, 245, 242))
    draw = ImageDraw.Draw(canvas)
    views = ((0, 1, "fused PCA-XY"), (0, 2, "fused PCA-XZ"), (1, 2, "fused PCA-YZ"), (0, 1, "fused PCA-XY depth"))
    for view_index, (x_axis, y_axis, label) in enumerate(views):
        column, row = view_index % 2, view_index // 2
        left, top = column * 600 + 25, row * 450 + 25
        right, bottom = left + 550, top + 400
        draw.rectangle((left, top, right, bottom), outline=(150, 150, 145), width=1)
        x_values, y_values = projected[:, x_axis], projected[:, y_axis]
        x_low, x_high = np.quantile(x_values, [0.005, 0.995])
        y_low, y_high = np.quantile(y_values, [0.005, 0.995])
        x_span = max(float(x_high - x_low), 1e-9)
        y_span = max(float(y_high - y_low), 1e-9)
        depth = projected[:, 2] if view_index == 3 else projected[:, (3 - x_axis - y_axis)]
        for index in np.argsort(depth):
            px = int(left + 8 + (float(x_values[index]) - x_low) / x_span * (right - left - 16))
            py = int(bottom - 8 - (float(y_values[index]) - y_low) / y_span * (bottom - top - 28))
            if left + 2 <= px <= right - 2 and top + 22 <= py <= bottom - 2:
                draw.point((px, py), fill=tuple(int(value) for value in colors[index]))
        draw.text((left + 8, top + 6), label, fill=(25, 25, 25))
    target = assert_output_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, format="PNG", optimize=True)
    return target


def finalize_dense_visual_gate(
    report_path: Path,
    *,
    visual_status: str,
    review_basis: Sequence[str],
    review: Mapping[str, Any] | None = None,
    postfusion_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record a visual review only after complete post-fusion evidence.

    ``postfusion_report`` is required for a passed visual review.  Keeping it
    explicit prevents callers from turning a bare set of numerical metrics and
    four arbitrary images into an accepted dense result.
    """

    from v4_config import read_json

    report = read_json(report_path)
    evidence = reviewed_evidence_gate(review)
    fused_value = report.get("fused_path")
    fused_path = Path(str(fused_value)) if isinstance(fused_value, str) and fused_value.strip() else None
    postfusion = postfusion_evidence_gate(postfusion_report, fused_path=fused_path)
    basis = [str(value).strip() for value in review_basis if str(value).strip()]
    if visual_status != "passed" or not evidence["passed"]:
        raise ValueError("; ".join(evidence["reasons"]) or "dense visual review is not passed")
    if not basis:
        raise ValueError("dense visual review_basis must contain non-empty evidence statements")
    if not postfusion["passed"]:
        raise ValueError("; ".join(postfusion["reasons"]) or "post-fusion evidence is not complete")
    metrics = report.get("metrics", {})
    smoke = report.get("smoke", {})
    gate = dense_gate(metrics, smoke=smoke, visual_status=str(visual_status), review=review)
    if not gate["passed"]:
        reasons = [
            *gate["contamination"]["reasons"],
            *gate["review"]["reasons"],
        ]
        raise ValueError("; ".join(reasons) or "dense quantitative gate failed")
    report["visual_gate"] = {
        "status": str(visual_status),
        "previews": evidence["preview_evidence"],
        "reviewer": evidence["reviewer"],
        "approver": evidence["approver"],
        "regions": evidence["regions"],
        "contamination_findings": dict(review["contamination_findings"]),  # type: ignore[index]
        "review_basis": basis,
    }
    report["gate"] = gate
    report["postfusion_evidence_gate"] = postfusion
    report["status"] = "complete"
    write_dense_report(report, Path(report_path))
    return report


__all__ = [
    "V4DenseConfig",
    "build_dense_pair_adjacency",
    "build_prioritized_dense_pair_adjacency",
    "build_image_undistorter_command",
    "build_patch_match_command",
    "build_stereo_fusion_command",
    "colmap_help",
    "contamination_assessment",
    "dense_depth_file_counts",
    "dense_failure_category",
    "dense_gate",
    "dense_stage_fingerprint",
    "dense_typed_file_counts",
    "dense_workspace_paths",
    "finalize_dense_visual_gate",
    "postfusion_evidence_gate",
    "prune_dense_photometric_maps",
    "render_dense_contact_sheet",
    "reviewed_evidence_gate",
    "resource_fallback",
    "undistort_v4_masks",
    "write_dense_image_list",
    "write_dense_pair_config",
    "write_prioritized_dense_pair_config",
    "write_dense_smoke_config",
    "write_dense_report",
]
