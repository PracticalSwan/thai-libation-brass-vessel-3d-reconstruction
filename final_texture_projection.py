"""Deterministic multi-view texture projection contracts for the final V2 model.

This module intentionally does not own Blender operations or claim visual QA.  It
consumes a validated, camera-ready geometry package exported by the final V2
Blender stage and provides the deterministic CV-side projection and fusion
helpers used by the orchestrator.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np
from PIL import Image

from final_model_io import ensure_under_v2_root, sha256_file, write_json_atomic


_HASH_256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_MATRIX = tuple(float(value) for value in np.eye(4).reshape(-1))
_SPECULAR_CONFIDENCE_CUTOFF = 0.5
_COVERAGE_CLASS_ORDER = (
    "direct_multi_view",
    "reviewed_single_or_detail",
    "symmetry_repetition",
    "hidden_generic_fill",
)
_COVERAGE_CLASSES = {
    coverage_class: True
    for coverage_class in _COVERAGE_CLASS_ORDER
}
_REQUIRED_TEXTURE_COLOR_SPACES = {
    "T_ThaiLibation_BaseColor.png": "sRGB",
    "T_ThaiLibation_Roughness.png": "Non-Color",
    "T_ThaiLibation_Wear.png": "Non-Color",
    "T_ThaiLibation_Metallic.png": "Non-Color",
    "T_ThaiLibation_Normal.png": "Non-Color",
    "T_ThaiLibation_AO.png": "Non-Color",
    "T_ThaiLibation_InferredRegionMask.png": "Non-Color",
}


@dataclass(frozen=True)
class ProjectionConfig:
    """Fail-closed numerical settings for texture projection and fusion."""

    texture_size: int = 4096
    minimum_cos_view_angle: float = 0.20
    highlight_value_cutoff: float = 0.97
    deep_shadow_value_cutoff: float = 0.08
    minimum_mask_confidence: float = 0.5
    minimum_samples_per_texel: int = 2

    def __post_init__(self) -> None:
        if self.texture_size < 1:
            raise ValueError("texture_size must be positive")
        if not -1.0 <= self.minimum_cos_view_angle <= 1.0:
            raise ValueError("minimum_cos_view_angle must be in [-1, 1]")
        if not 0.0 <= self.deep_shadow_value_cutoff < self.highlight_value_cutoff <= 1.0:
            raise ValueError("value cutoffs must satisfy 0 <= shadow < highlight <= 1")
        if not 0.0 <= self.minimum_mask_confidence <= 1.0:
            raise ValueError("minimum_mask_confidence must be in [0, 1]")
        if self.minimum_samples_per_texel < 1:
            raise ValueError("minimum_samples_per_texel must be positive")


@dataclass(frozen=True)
class CameraReference:
    """One registered Step 13 camera in COLMAP's SIMPLE_RADIAL convention."""

    camera_id: int
    selected_index: int
    width: int
    height: int
    focal_length: float
    principal_x: float
    principal_y: float
    radial_distortion: float = 0.0
    camera_model: str = "SIMPLE_RADIAL"
    world_to_camera: tuple[float, ...] = _IDENTITY_MATRIX

    def __post_init__(self) -> None:
        if self.camera_model != "SIMPLE_RADIAL":
            raise ValueError("final projection requires the frozen SIMPLE_RADIAL camera model")
        if self.width < 1 or self.height < 1:
            raise ValueError("camera image dimensions must be positive")
        if self.focal_length <= 0.0:
            raise ValueError("camera focal_length must be positive")
        if len(self.world_to_camera) not in {12, 16}:
            raise ValueError(
                "world_to_camera must contain 12 or 16 row-major matrix values"
            )
        rows = 3 if len(self.world_to_camera) == 12 else 4
        matrix = np.asarray(self.world_to_camera, dtype=np.float64).reshape(rows, 4)
        if not np.isfinite(matrix).all():
            raise ValueError("world_to_camera must be finite")
        if rows == 4 and not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
            raise ValueError("world_to_camera bottom row must be [0, 0, 0, 1]")

    @property
    def matrix(self) -> np.ndarray:
        """Return a fresh read-only 4x4 world-to-camera matrix."""

        values = np.asarray(self.world_to_camera, dtype=np.float64)
        if len(values) == 12:
            matrix = np.vstack((values.reshape(3, 4), (0.0, 0.0, 0.0, 1.0)))
        else:
            matrix = values.reshape(4, 4)
        matrix = matrix.copy()
        matrix.setflags(write=False)
        return matrix

    @property
    def center(self) -> np.ndarray:
        """Return the camera center in world coordinates."""

        matrix = self.matrix
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
        return -rotation.T @ translation


@dataclass(frozen=True)
class ProjectionView:
    """One registered source image and the whole-object mask used for sampling."""

    selected_index: int
    image_path: Path
    whole_mask_path: Path
    camera: CameraReference | None
    quality_condition: str
    source_sha256: str
    view_category: str | None = None
    mask_source: str = "reconstruction"
    step13_registered: bool = True

    def __post_init__(self) -> None:
        if not self.step13_registered:
            if self.camera is not None:
                raise ValueError("an unregistered view must not carry a camera")
        elif self.camera is None:
            raise ValueError("a registered projection view requires a camera")
        if self.camera is not None and self.camera.selected_index != self.selected_index:
            raise ValueError("projection view and camera selected indices disagree")
        if not _HASH_256.fullmatch(self.source_sha256):
            raise ValueError("source_sha256 must be a lowercase 64-character SHA-256")
        if self.mask_source not in {"reviewed", "reconstruction"}:
            raise ValueError("mask_source must be 'reviewed' or 'reconstruction'")


@dataclass(frozen=True)
class ProjectionSample:
    """One source-color observation at one UV texel."""

    texel_x: int
    texel_y: int
    view_index: int
    selected_index: int
    triangle_index: int
    source_x: float
    source_y: float
    color_srgb: tuple[float, float, float]
    weight: float
    valid: bool


@dataclass(frozen=True)
class FusedTexel:
    """Robust multi-view result and the confidence retained for a texel."""

    color_srgb: tuple[float, float, float]
    confidence: float
    accepted_count: int


@dataclass(frozen=True)
class OverlapObservation:
    """One trustworthy candidate in a geometric surface correspondence."""

    surface_id: str
    view_index: int
    selected_index: int
    color_srgb: tuple[float, float, float]
    luminance: float | None = None
    saturation: float | None = None
    specular_confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.view_index < 0 or self.selected_index < 0:
            raise ValueError("overlap observation indices must be non-negative")
        color = np.asarray(self.color_srgb, dtype=np.float64)
        if color.shape != (3,) or not np.isfinite(color).all():
            raise ValueError("overlap color must contain three finite values")
        if np.any(color < 0.0) or np.any(color > 255.0):
            raise ValueError("overlap color values must be in [0, 255]")
        maximum = float(color.max())
        minimum = float(color.min())
        if self.luminance is None:
            object.__setattr__(self, "luminance", maximum / 255.0)
        if self.saturation is None:
            object.__setattr__(
                self, "saturation", 0.0 if maximum <= 0.0 else (maximum - minimum) / maximum
            )
        if not 0.0 <= float(self.luminance) <= 1.0:
            raise ValueError("overlap luminance must be in [0, 1]")
        if not 0.0 <= float(self.saturation) <= 1.0:
            raise ValueError("overlap saturation must be in [0, 1]")
        if not 0.0 <= self.specular_confidence <= 1.0:
            raise ValueError("specular_confidence must be in [0, 1]")

    @property
    def color_array(self) -> np.ndarray:
        return np.asarray(self.color_srgb, dtype=np.float64)


@dataclass(frozen=True)
class PhotometricCorrection:
    """Bounded relative RGB gain and fail-closed diagnostics for one view."""

    view_index: int
    selected_index: int
    gains: tuple[float, float, float]
    accepted: bool
    sample_count: int
    rejected_clipped_count: int
    rejected_deep_shadow_count: int
    rejected_specular_count: int
    residual_before: float
    residual_after: float
    unreliable_reason: str | None


@dataclass(frozen=True)
class PhotometricHarmonization:
    """Low-DOF overlap normalization relative to one deterministic anchor."""

    anchor_view_index: int
    corrections: tuple[PhotometricCorrection, ...]
    method: str = "anchor_median_rgb_gain"
    color_calibrated: bool = False

    def report_dict(self) -> dict[str, Any]:
        return {
            "anchor_view_index": self.anchor_view_index,
            "method": self.method,
            "color_calibrated": self.color_calibrated,
            "claim": "relative_photometric_harmonization_not_reflectance_recovery",
            "corrections": [
                {
                    "view_index": correction.view_index,
                    "selected_index": correction.selected_index,
                    "gains_srgb": list(correction.gains),
                    "accepted": correction.accepted,
                    "sample_count": correction.sample_count,
                    "rejection_counts": {
                        "clipped_highlight": correction.rejected_clipped_count,
                        "deep_shadow": correction.rejected_deep_shadow_count,
                        "specular": correction.rejected_specular_count,
                    },
                    "robust_residual_before": correction.residual_before,
                    "robust_residual_after": correction.residual_after,
                    "unreliable_reason": correction.unreliable_reason,
                }
                for correction in self.corrections
            ],
        }


@dataclass(frozen=True)
class GapFillPlan:
    """Coverage-class authorization before any unsupported texel is filled."""

    projection_defect_mask: np.ndarray
    reviewed_source_gap_mask: np.ndarray
    symmetry_fill_mask: np.ndarray
    generic_fill_mask: np.ndarray


@dataclass(frozen=True)
class GapFillResult:
    """Conservatively filled color and the provenance left after filling."""

    base_color: np.ndarray
    fill_origin: np.ndarray
    unresolved_mask: np.ndarray
    symmetry_filled_count: int
    generic_filled_count: int


@dataclass(frozen=True)
class ProjectionGeometry:
    """Camera-ready, world-space triangulated geometry exported by Blender."""

    vertices: np.ndarray
    triangles: np.ndarray
    loop_uvs: np.ndarray
    loop_normals: np.ndarray
    material_ids: np.ndarray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float32)
        triangles = np.asarray(self.triangles, dtype=np.int32)
        loop_uvs = np.asarray(self.loop_uvs, dtype=np.float32)
        loop_normals = np.asarray(self.loop_normals, dtype=np.float32)
        material_ids = np.asarray(self.material_ids, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            raise ValueError("vertices must have shape (vertex_count, 3)")
        if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
            raise ValueError("triangles must have shape (triangle_count, 3)")
        if int(triangles.min()) < 0 or int(triangles.max()) >= len(vertices):
            raise ValueError("triangle index is outside the vertex array")
        if loop_uvs.shape != (len(triangles) * 3, 2):
            raise ValueError("loop_uvs must contain exactly three coordinates per triangle")
        if loop_normals.shape != (len(triangles) * 3, 3):
            raise ValueError("loop_normals must contain exactly three normals per triangle")
        if material_ids.shape != (len(triangles),):
            raise ValueError("material_ids must contain one value per triangle")
        if not (
            np.isfinite(vertices).all()
            and np.isfinite(loop_uvs).all()
            and np.isfinite(loop_normals).all()
        ):
            raise ValueError("geometry arrays must contain only finite values")
        if np.any(np.linalg.norm(loop_normals, axis=1) <= 1e-8):
            raise ValueError("loop normals must be non-degenerate")
        if loop_uvs.min() < 0.0 or loop_uvs.max() > 1.0:
            raise ValueError("loop UVs must lie in [0, 1]")
        object.__setattr__(self, "vertices", _read_only(vertices))
        object.__setattr__(self, "triangles", _read_only(triangles))
        object.__setattr__(self, "loop_uvs", _read_only(loop_uvs))
        object.__setattr__(self, "loop_normals", _read_only(loop_normals))
        object.__setattr__(self, "material_ids", _read_only(material_ids))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProjectionGeometry):
            return NotImplemented
        return bool(
            np.array_equal(self.vertices, other.vertices)
            and np.array_equal(self.triangles, other.triangles)
            and np.array_equal(self.loop_uvs, other.loop_uvs)
            and np.array_equal(self.loop_normals, other.loop_normals)
            and np.array_equal(self.material_ids, other.material_ids)
        )


@dataclass(frozen=True)
class ProjectionPackage:
    """Validated geometry package and its accepted Blender source provenance."""

    geometry: ProjectionGeometry
    source_blend_path: str
    source_blend_sha256: str
    coordinate_system: str
    format_version: int


@dataclass(frozen=True)
class RasterizedUV:
    """Deterministic triangle and barycentric assignment for each texel center."""

    triangle_index: np.ndarray
    barycentric: np.ndarray


@dataclass(frozen=True)
class ProjectionSelection:
    """Bounded registered view set and its provenance manifest rows."""

    views: tuple[ProjectionView, ...]
    manifest_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CoverageSummary:
    """Aggregated direct-projection coverage over the UV texture grid."""

    texture_size: int
    supported_texel_count: int
    direct_projection_percent: float
    low_confidence_percent: float
    zero_sample_percent: float


@dataclass(frozen=True)
class FinalTextureSet:
    """Hashed final texture files and their color-space contract."""

    texture_records: tuple[dict[str, Any], ...]
    color_spaces: dict[str, str]


def _read_only(array: np.ndarray) -> np.ndarray:
    result = array.copy()
    result.setflags(write=False)
    return result


def _quality_penalty(quality_condition: str) -> float:
    """Map frozen image diagnostics to a conservative contribution penalty."""

    normalized = quality_condition.strip().lower().replace(" ", "_")
    if normalized in {"normal", "accept"}:
        return 1.0
    if normalized in {"bright_clipping", "high_brightness"}:
        return 0.55
    if normalized == "low_sharpness_low_features":
        return 0.70
    if normalized == "warn":
        return 0.85
    return 0.75


def view_weight(
    cos_view_angle: float,
    mask_confidence: float,
    luminance: float,
    saturation: float,
    visible: bool,
    quality_condition: str = "normal",
    reprojection_confidence: float = 1.0,
    config: ProjectionConfig | None = None,
) -> float:
    """Return the multiplicative contribution weight for one source observation.

    Camera distance is deliberately absent: it is not an image-quality proxy.
    Invalid evidence raises ``ValueError``; rejected highlights and shadows return
    zero so callers can preserve provenance rather than silently dropping a view.
    """

    if not -1.0 <= cos_view_angle <= 1.0:
        raise ValueError("cos_view_angle must be in [-1, 1]")
    if not 0.0 <= mask_confidence <= 1.0:
        raise ValueError("mask_confidence must be in [0, 1]")
    if not 0.0 <= luminance <= 1.0:
        raise ValueError("luminance must be in [0, 1]")
    if not 0.0 <= saturation <= 1.0:
        raise ValueError("saturation must be in [0, 1]")
    if not 0.0 <= reprojection_confidence <= 1.0:
        raise ValueError("reprojection_confidence must be in [0, 1]")
    if not visible:
        return 0.0
    settings = ProjectionConfig() if config is None else config
    if cos_view_angle < settings.minimum_cos_view_angle:
        return 0.0
    if (
        luminance >= settings.highlight_value_cutoff
        or luminance <= settings.deep_shadow_value_cutoff
    ):
        return 0.0
    return float(
        cos_view_angle
        * mask_confidence
        * reprojection_confidence
        * _quality_penalty(quality_condition)
    )


def barycentric_coordinates(
    point: Sequence[float],
    triangle: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Compute barycentric coordinates without changing UV orientation."""

    query = np.asarray(point, dtype=np.float64)
    corners = np.asarray(triangle, dtype=np.float64)
    if query.shape != (2,) or corners.shape != (3, 2):
        raise ValueError("point must be 2D and triangle must have shape (3, 2)")
    edge_1 = corners[1] - corners[0]
    edge_2 = corners[2] - corners[0]
    local = query - corners[0]
    d00 = float(edge_1 @ edge_1)
    d01 = float(edge_1 @ edge_2)
    d11 = float(edge_2 @ edge_2)
    d20 = float(local @ edge_1)
    d21 = float(local @ edge_2)
    denominator = d00 * d11 - d01 * d01
    if abs(denominator) < epsilon:
        raise ValueError("degenerate UV triangle")
    beta = (d11 * d20 - d01 * d21) / denominator
    gamma = (d00 * d21 - d01 * d20) / denominator
    return np.asarray([1.0 - beta - gamma, beta, gamma], dtype=np.float32)


def interpolate_barycentric(values: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    """Interpolate three corner values with barycentric weights."""

    coefficients = np.asarray(weights, dtype=np.float64)
    array = np.asarray(values)
    if coefficients.shape != (3,) or not np.isfinite(coefficients).all():
        raise ValueError("barycentric weights must be three finite values")
    if array.shape[0] != 3:
        raise ValueError("values must contain exactly three corners")
    return np.asarray(coefficients @ array)


def _geometry_corner_slice(geometry: ProjectionGeometry, triangle_index: int) -> tuple[slice, np.ndarray]:
    if triangle_index < 0 or triangle_index >= len(geometry.triangles):
        raise ValueError("triangle_index is outside the geometry package")
    corner = slice(triangle_index * 3, triangle_index * 3 + 3)
    return corner, geometry.triangles[triangle_index]


def interpolate_surface(
    geometry: ProjectionGeometry, triangle_index: int, weights: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate world position and loop normal for one surface point."""

    corner, _ = _geometry_corner_slice(geometry, triangle_index)
    point = interpolate_barycentric(geometry.vertices[corner], weights)
    normal = interpolate_barycentric(geometry.loop_normals[corner], weights)
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= 1e-8:
        raise ValueError("interpolated surface normal is degenerate")
    return point.astype(np.float32), (normal / normal_length).astype(np.float32)


def rasterize_uv_triangles(
    geometry: ProjectionGeometry, *, texture_size: int
) -> RasterizedUV:
    """Rasterize UV triangles onto square texel centers in deterministic order.

    Array row 0 corresponds to UV v near 1; this matches the usual image-row to
    OpenGL/Blender UV conversion used when writing PNG textures.
    """

    if texture_size < 1:
        raise ValueError("texture_size must be positive")
    triangle_index = np.full((texture_size, texture_size), -1, dtype=np.int32)
    barycentric = np.zeros((texture_size, texture_size, 3), dtype=np.float32)
    ys, xs = np.indices((texture_size, texture_size), dtype=np.float32)
    query_uvs = np.stack(
        ((xs + 0.5) / texture_size, 1.0 - (ys + 0.5) / texture_size), axis=-1
    )

    for triangle_id in range(len(geometry.triangles)):
        corners = geometry.loop_uvs[triangle_id * 3 : triangle_id * 3 + 3]
        min_uv = corners.min(axis=0)
        max_uv = corners.max(axis=0)
        min_pixel = np.ceil(min_uv * texture_size - 0.5).astype(int)
        max_pixel = np.ceil(max_uv * texture_size - 0.5).astype(int) - 1
        x0, y0 = np.maximum(0, min_pixel)
        x1, y1 = np.minimum(texture_size - 1, max_pixel)
        if x0 > x1 or y0 > y1:
            continue
        candidates = query_uvs[y0 : y1 + 1, x0 : x1 + 1].reshape(-1, 2)
        for local_index, candidate in enumerate(candidates):
            weights = barycentric_coordinates(candidate, corners)
            if np.all(weights >= -1e-7):
                weights = np.maximum(weights, 0.0)
                weights /= weights.sum()
                row = y0 + local_index // (x1 - x0 + 1)
                column = x0 + local_index % (x1 - x0 + 1)
                triangle_index[row, column] = triangle_id
                barycentric[row, column] = weights

    return RasterizedUV(
        triangle_index=_read_only(triangle_index), barycentric=_read_only(barycentric)
    )


def project_simple_radial(
    camera: CameraReference, world_point: Sequence[float]
) -> tuple[float, float, float]:
    """Project a world point through the exact SIMPLE_RADIAL mapping."""

    point = np.asarray(world_point, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("world_point must be three finite coordinates")
    camera_point = camera.matrix @ np.asarray([*point, 1.0], dtype=np.float64)
    x, y, depth = camera_point[:3]
    if depth <= 1e-12:
        raise ValueError("world point is behind the camera")
    normalized_x = x / depth
    normalized_y = y / depth
    radius_sq = normalized_x * normalized_x + normalized_y * normalized_y
    distorted = 1.0 + camera.radial_distortion * radius_sq
    pixel_x = camera.focal_length * normalized_x * distorted + camera.principal_x
    pixel_y = camera.focal_length * normalized_y * distorted + camera.principal_y
    return float(pixel_x), float(pixel_y), float(depth)


def _direction_between(start: np.ndarray, finish: np.ndarray) -> np.ndarray:
    direction = np.asarray(finish, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        raise ValueError("surface point coincides with the camera center")
    return direction / length


def sample_projection_view(
    *,
    view: ProjectionView,
    geometry: ProjectionGeometry,
    triangle_index: int,
    barycentric: Sequence[float],
    source_rgb: np.ndarray,
    mask: np.ndarray,
    depth_map: np.ndarray,
    texel_x: int,
    texel_y: int,
    view_index: int,
    config: ProjectionConfig,
    depth_tolerance: float = 1e-4,
) -> ProjectionSample:
    """Sample one view with an explicit depth-map visibility decision."""

    if view.camera is None:
        raise ValueError("projection sampling requires a registered camera")
    image = np.asarray(source_rgb)
    object_mask = np.asarray(mask)
    depths = np.asarray(depth_map)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("source_rgb must have shape (height, width, 3)")
    if object_mask.shape != image.shape[:2] or depths.shape != image.shape[:2]:
        raise ValueError("mask and depth map must match the source image dimensions")
    if not np.isfinite(depths).all():
        raise ValueError("depth map contains non-finite values")
    if not np.isfinite(object_mask).all():
        raise ValueError("whole-object mask contains non-finite values")
    mask_minimum = float(object_mask.min())
    mask_maximum = float(object_mask.max())
    mask_is_normalized = mask_minimum >= 0.0 and mask_maximum <= 1.0
    mask_is_byte_scale = (
        mask_minimum >= 0.0
        and mask_maximum <= 255.0
        and np.array_equal(object_mask, np.rint(object_mask))
    )
    if not (mask_is_normalized or mask_is_byte_scale):
        raise ValueError("whole-object mask must use [0, 1] or integer [0, 255] values")

    point, normal = interpolate_surface(geometry, triangle_index, barycentric)
    try:
        source_x, source_y, projected_depth = project_simple_radial(view.camera, point)
    except ValueError:
        return ProjectionSample(
            texel_x, texel_y, view_index, view.selected_index, triangle_index,
            -1.0, -1.0, (0.0, 0.0, 0.0), 0.0, False,
        )
    source_column = int(math.floor(source_x))
    source_row = int(math.floor(source_y))
    inside = (
        0 <= source_row < image.shape[0]
        and 0 <= source_column < image.shape[1]
    )
    if not inside:
        return ProjectionSample(
            texel_x, texel_y, view_index, view.selected_index, triangle_index,
            source_x, source_y, (0.0, 0.0, 0.0), 0.0, False,
        )
    color = image[source_row, source_column].astype(np.float64)
    maximum = float(color.max())
    minimum = float(color.min())
    luminance = maximum / 255.0
    saturation = 0.0 if maximum <= 0.0 else (maximum - minimum) / maximum
    mask_confidence = float(object_mask[source_row, source_column])
    if mask_confidence > 1.0 and mask_confidence <= 255.0:
        mask_confidence /= 255.0
    if not 0.0 <= mask_confidence <= 1.0:
        raise ValueError("whole-object mask values must be in [0, 1] or [0, 255]")
    visible = bool(
        mask_confidence >= config.minimum_mask_confidence
        and projected_depth <= float(depths[source_row, source_column]) + depth_tolerance
    )
    direction = _direction_between(point, view.camera.center)
    cos_view_angle = float(np.dot(normal.astype(np.float64), direction))
    weight = view_weight(
        cos_view_angle=cos_view_angle,
        mask_confidence=mask_confidence,
        luminance=luminance,
        saturation=saturation,
        visible=visible,
        quality_condition=view.quality_condition,
        config=config,
    )
    valid = bool(visible and cos_view_angle >= config.minimum_cos_view_angle and weight > 0.0)
    return ProjectionSample(
        texel_x, texel_y, view_index, view.selected_index, triangle_index,
        source_x, source_y, tuple(float(value) for value in color), weight, valid,
    )


def _weighted_median_1d(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    return float(sorted_values[int(np.searchsorted(cumulative, cumulative[-1] / 2.0))])


def _weighted_median_color(colors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.asarray(
        [_weighted_median_1d(colors[:, channel], weights) for channel in range(3)],
        dtype=np.float64,
    )


def robust_fuse_colors(
    colors: np.ndarray, weights: np.ndarray, config: ProjectionConfig
) -> FusedTexel | None:
    """Fuse sRGB colors with a weighted channel median and local consistency.

    Inputs use 8-bit-style sRGB values (0-255).  A transient highlight is removed
    only when other geometrically consistent observations support the texel; a
    dark value repeated across the accepted observations remains the median.
    """

    values = np.asarray(colors, dtype=np.float64)
    contribution = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        raise ValueError("colors must have shape (sample_count, 3)")
    if contribution.shape != (len(values),):
        raise ValueError("one weight is required per color sample")
    if not np.isfinite(values).all() or not np.isfinite(contribution).all():
        raise ValueError("fusion inputs must be finite")
    if np.any(values < 0.0) or np.any(values > 255.0):
        raise ValueError("sRGB color values must be in [0, 255]")
    if np.any(contribution < 0.0):
        raise ValueError("fusion weights must be non-negative")

    luminance = values.max(axis=1) / 255.0
    eligible = (
        (contribution > 0.0)
        & (luminance < config.highlight_value_cutoff)
        & (luminance > config.deep_shadow_value_cutoff)
    )
    if int(eligible.sum()) < config.minimum_samples_per_texel:
        return None
    first_pass = _weighted_median_color(values[eligible], contribution[eligible])
    distance = np.linalg.norm(values - first_pass, axis=1) / 255.0
    consistent = eligible & (distance <= 0.25)
    if int(consistent.sum()) < config.minimum_samples_per_texel:
        consistent = eligible
    fused = _weighted_median_color(values[consistent], contribution[consistent])
    total_weight = float(contribution.sum())
    accepted_weight = float(contribution[consistent].sum())
    weight_agreement = accepted_weight / total_weight if total_weight > 0.0 else 0.0
    confidence = min(
        1.0,
        weight_agreement
        * min(1.0, int(consistent.sum()) / config.minimum_samples_per_texel),
    )
    return FusedTexel(
        color_srgb=tuple(float(value) for value in fused),
        confidence=float(confidence),
        accepted_count=int(consistent.sum()),
    )


def fuse_texel_samples(
    samples: Iterable[ProjectionSample], config: ProjectionConfig
) -> FusedTexel | None:
    """Fuse all valid observations for one texel using the configured consensus."""

    valid = [sample for sample in samples if sample.valid and sample.weight > 0.0]
    if not valid:
        return None
    return robust_fuse_colors(
        np.asarray([sample.color_srgb for sample in valid], dtype=np.float64),
        np.asarray([sample.weight for sample in valid], dtype=np.float64),
        config,
    )


def _camera_bin(view: ProjectionView, azimuth_bins: int, elevation_bins: int) -> tuple[int, int]:
    assert view.camera is not None
    center = view.camera.center
    radius = float(np.linalg.norm(center))
    if radius <= 1e-12:
        raise ValueError("cannot derive coverage direction for a camera at the origin")
    azimuth = int(math.floor(((math.atan2(center[1], center[0]) + math.pi) / (2.0 * math.pi)) * azimuth_bins))
    elevation = int(math.floor(((center[2] / radius + 1.0) / 2.0) * elevation_bins))
    return (
        min(azimuth_bins - 1, max(0, azimuth)),
        min(elevation_bins - 1, max(0, elevation)),
    )


def select_projection_views(
    views: Iterable[ProjectionView],
    *,
    maximum_views: int = 48,
    azimuth_bins: int = 12,
    elevation_bins: int = 3,
) -> ProjectionSelection:
    """Select a bounded set while cycling through all occupied direction bins."""

    if maximum_views < 1:
        raise ValueError("maximum_views must be positive")
    if azimuth_bins < 1 or elevation_bins < 1:
        raise ValueError("coverage bin counts must be positive")
    eligible: list[ProjectionView] = []
    seen_indices: set[int] = set()
    seen_cameras: set[int] = set()
    for view in views:
        if not view.step13_registered or view.camera is None:
            continue
        if view.selected_index in seen_indices or view.camera.camera_id in seen_cameras:
            raise ValueError("duplicate registered projection view")
        seen_indices.add(view.selected_index)
        seen_cameras.add(view.camera.camera_id)
        eligible.append(view)

    bins: dict[tuple[int, int], list[ProjectionView]] = {}
    for view in eligible:
        bins.setdefault(_camera_bin(view, azimuth_bins, elevation_bins), []).append(view)
    for candidates in bins.values():
        candidates.sort(
            key=lambda view: (
                -_quality_penalty(view.quality_condition),
                view.selected_index,
            )
        )

    selected: list[ProjectionView] = []
    occupied_bins = sorted(bins)
    while len(selected) < maximum_views and any(bins.values()):
        for coverage_bin in occupied_bins:
            if len(selected) >= maximum_views:
                break
            if bins[coverage_bin]:
                selected.append(bins[coverage_bin].pop(0))

    rows: list[dict[str, Any]] = []
    for view in selected:
        assert view.camera is not None
        azimuth_bin, elevation_bin = _camera_bin(view, azimuth_bins, elevation_bins)
        rows.append(
            {
                "selected_index": view.selected_index,
                "camera_id": view.camera.camera_id,
                "quality_condition": view.quality_condition,
                "mask_source": view.mask_source,
                "view_category": view.view_category,
                "source_sha256": view.source_sha256,
                "image_path": view.image_path.as_posix(),
                "whole_mask_path": view.whole_mask_path.as_posix(),
                "azimuth_bin": azimuth_bin,
                "elevation_bin": elevation_bin,
                "inclusion_reason": "registered_coverage_bin",
            }
        )
    return ProjectionSelection(tuple(selected), tuple(rows))


def _trusted_overlap_observation(
    observation: OverlapObservation, config: ProjectionConfig
) -> bool:
    luminance = float(observation.luminance or 0.0)
    return bool(
        luminance < config.highlight_value_cutoff
        and luminance > config.deep_shadow_value_cutoff
        and observation.specular_confidence < _SPECULAR_CONFIDENCE_CUTOFF
        and float(observation.color_array.min()) > 1.0
    )


def _overlap_rejection_counts(
    observations: Sequence[OverlapObservation], config: ProjectionConfig
) -> tuple[int, int, int]:
    clipped = sum(
        float(observation.luminance or 0.0) >= config.highlight_value_cutoff
        for observation in observations
    )
    shadow = sum(
        float(observation.luminance or 0.0) <= config.deep_shadow_value_cutoff
        for observation in observations
    )
    specular = sum(
        observation.specular_confidence >= _SPECULAR_CONFIDENCE_CUTOFF
        for observation in observations
    )
    return int(clipped), int(shadow), int(specular)


def _pair_residual(target: np.ndarray, corrected: np.ndarray) -> float:
    if len(target) == 0:
        return 1.0
    normalized_distance = np.linalg.norm(target - corrected, axis=1) / 255.0
    return float(np.sqrt(np.mean(normalized_distance**2)))


def estimate_photometric_harmonization(
    observations: Iterable[OverlapObservation],
    *,
    config: ProjectionConfig | None = None,
    minimum_overlap_samples: int = 24,
    minimum_gain: float = 0.88,
    maximum_gain: float = 1.12,
) -> PhotometricHarmonization:
    """Estimate conservative per-view RGB gains from geometric overlap pairs.

    The solve is deliberately limited to one gain per channel.  Clipped
    highlights, deep shadows, likely moving speculars, and near-zero channels are
    excluded.  A view needing a correction outside the documented bounds keeps
    identity gains and is marked photometrically unreliable.
    """

    settings = ProjectionConfig() if config is None else config
    if minimum_overlap_samples < 2:
        raise ValueError("minimum_overlap_samples must be at least 2")
    if not 0.0 < minimum_gain <= 1.0 <= maximum_gain:
        raise ValueError("gain bounds must include unity gain")
    if maximum_gain <= minimum_gain:
        raise ValueError("maximum_gain must exceed minimum_gain")

    grouped: dict[str, list[OverlapObservation]] = {}
    selected_by_view: dict[int, int] = {}
    all_by_view: dict[int, list[OverlapObservation]] = {}
    for observation in observations:
        if observation.view_index in selected_by_view:
            if selected_by_view[observation.view_index] != observation.selected_index:
                raise ValueError("one view index maps to multiple selected indices")
        else:
            selected_by_view[observation.view_index] = observation.selected_index
        grouped.setdefault(observation.surface_id, []).append(observation)
        all_by_view.setdefault(observation.view_index, []).append(observation)
    if not grouped or not selected_by_view:
        raise ValueError("photometric overlap samples are required")

    trusted_by_surface: dict[str, dict[int, OverlapObservation]] = {}
    for surface_id, surface_observations in grouped.items():
        trusted: dict[int, OverlapObservation] = {}
        for observation in surface_observations:
            if observation.view_index in trusted:
                raise ValueError("a surface has duplicate observations for one view")
            if _trusted_overlap_observation(observation, settings):
                trusted[observation.view_index] = observation
        if trusted:
            trusted_by_surface[surface_id] = trusted

    trusted_counts = {
        view_index: sum(
            view_index in trusted for trusted in trusted_by_surface.values()
        )
        for view_index in selected_by_view
    }
    anchor_view = min(
        selected_by_view,
        key=lambda view_index: (-trusted_counts[view_index], view_index),
    )
    corrections: list[PhotometricCorrection] = []
    for view_index in sorted(selected_by_view):
        view_observations = all_by_view[view_index]
        clipped, shadow, specular = _overlap_rejection_counts(
            view_observations, settings
        )
        targets: list[np.ndarray] = []
        sources: list[np.ndarray] = []
        for trusted in trusted_by_surface.values():
            anchor = trusted.get(anchor_view)
            source = trusted.get(view_index)
            if anchor is not None and source is not None:
                targets.append(anchor.color_array)
                sources.append(source.color_array)

        sample_count = len(targets)
        if view_index == anchor_view:
            gains = (1.0, 1.0, 1.0)
            accepted = sample_count >= minimum_overlap_samples
            residual_before = 0.0 if accepted else 1.0
            reason = None if accepted else "insufficient_overlap_samples"
            corrections.append(
                PhotometricCorrection(
                    view_index=view_index,
                    selected_index=selected_by_view[view_index],
                    gains=gains,
                    accepted=accepted,
                    sample_count=sample_count,
                    rejected_clipped_count=clipped,
                    rejected_deep_shadow_count=shadow,
                    rejected_specular_count=specular,
                    residual_before=residual_before,
                    residual_after=residual_before,
                    unreliable_reason=reason,
                )
            )
            continue

        if sample_count == 0:
            raw_gains = np.asarray([1.0, 1.0, 1.0])
        else:
            target_matrix = np.stack(targets)
            source_matrix = np.stack(sources)
            raw_gains = np.median(target_matrix / source_matrix, axis=0)
        before = _pair_residual(np.stack(targets), np.stack(sources)) if sample_count else 1.0
        outside_bounds = bool(
            np.any(raw_gains < minimum_gain) or np.any(raw_gains > maximum_gain)
        )
        sufficient = sample_count >= minimum_overlap_samples
        if sufficient and not outside_bounds:
            gains = tuple(float(value) for value in raw_gains)
            corrected = np.stack(sources) * np.asarray(gains)
            after = _pair_residual(np.stack(targets), corrected)
            accepted = True
            reason = None
        else:
            gains = (1.0, 1.0, 1.0)
            after = before
            accepted = False
            reason = (
                "gain_outside_conservative_bounds"
                if outside_bounds
                else "insufficient_overlap_samples"
            )
        corrections.append(
            PhotometricCorrection(
                view_index=view_index,
                selected_index=selected_by_view[view_index],
                gains=gains,
                accepted=accepted,
                sample_count=sample_count,
                rejected_clipped_count=clipped,
                rejected_deep_shadow_count=shadow,
                rejected_specular_count=specular,
                residual_before=before,
                residual_after=after,
                unreliable_reason=reason,
            )
        )

    return PhotometricHarmonization(
        anchor_view_index=anchor_view,
        corrections=tuple(corrections),
    )


def apply_photometric_correction(
    colors: np.ndarray, correction: PhotometricCorrection
) -> np.ndarray:
    """Apply one accepted bounded RGB gain without extrapolating bad views."""

    values = np.asarray(colors, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError("colors must have a final RGB channel")
    if not np.isfinite(values).all():
        raise ValueError("colors must be finite")
    if np.any(values < 0.0) or np.any(values > 255.0):
        raise ValueError("sRGB colors must be in [0, 255]")
    if not correction.accepted:
        raise ValueError("cannot apply a photometrically unreliable correction")
    corrected = values * np.asarray(correction.gains, dtype=np.float64)
    return np.rint(np.clip(corrected, 0.0, 255.0)).astype(np.uint8)


def coverage_gap_plan(
    sample_count: np.ndarray,
    confidence: np.ndarray,
    coverage_class_map: np.ndarray,
    *,
    minimum_samples: int,
) -> GapFillPlan:
    """Classify unsupported texels without silently promoting their evidence."""

    counts = np.asarray(sample_count)
    confidences = np.asarray(confidence, dtype=np.float64)
    classes = np.asarray(coverage_class_map, dtype=object)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("sample_count must be a square texture grid")
    if confidences.shape != counts.shape or classes.shape != counts.shape:
        raise ValueError("confidence and coverage_class_map must match sample_count")
    if not np.isfinite(confidences).all():
        raise ValueError("confidence must be finite")
    if np.any(confidences < 0.0) or np.any(confidences > 1.0):
        raise ValueError("confidence values must be in [0, 1]")
    unknown = ~np.isin(classes, list(_COVERAGE_CLASSES))
    if np.any(unknown):
        raise ValueError(f"unsupported coverage class: {classes[unknown][0]!r}")

    unsupported = ~(
        (counts >= minimum_samples) & (confidences > 0.0)
    )
    return GapFillPlan(
        projection_defect_mask=_read_only(
            unsupported & (classes == "direct_multi_view")
        ),
        reviewed_source_gap_mask=_read_only(
            unsupported & (classes == "reviewed_single_or_detail")
        ),
        symmetry_fill_mask=_read_only(
            unsupported & (classes == "symmetry_repetition")
        ),
        generic_fill_mask=_read_only(
            unsupported & (classes == "hidden_generic_fill")
        ),
    )


def apply_conservative_gap_fill(
    plan: GapFillPlan,
    *,
    base_color: np.ndarray,
    symmetry_source_color: np.ndarray | None,
    generic_brass_color: tuple[float, float, float],
) -> GapFillResult:
    """Fill only coverage-authorized texels and retain unresolved defects."""

    color = np.asarray(base_color, dtype=np.float32)
    if color.ndim != 3 or color.shape[2] != 3 or not np.isfinite(color).all():
        raise ValueError("base_color must have finite shape (height, width, 3)")
    if np.any(color < 0.0) or np.any(color > 1.0):
        raise ValueError("normalized base_color values must be in [0, 1]")
    shape = color.shape[:2]
    masks = (
        plan.projection_defect_mask,
        plan.reviewed_source_gap_mask,
        plan.symmetry_fill_mask,
        plan.generic_fill_mask,
    )
    if any(np.asarray(mask).shape != shape for mask in masks):
        raise ValueError("gap-fill masks must match base_color dimensions")
    if np.any(np.asarray(plan.symmetry_fill_mask, dtype=np.uint8) + np.asarray(plan.generic_fill_mask, dtype=np.uint8) > 1):
        raise ValueError("gap-fill authorization masks must not overlap")

    symmetry = np.asarray(symmetry_source_color, dtype=np.float32) if symmetry_source_color is not None else None
    if symmetry is not None:
        if symmetry.shape != color.shape or not np.isfinite(symmetry).all():
            raise ValueError("symmetry_source_color must match base_color and be finite")
        if np.any(symmetry < 0.0) or np.any(symmetry > 1.0):
            raise ValueError("normalized symmetry_source_color values must be in [0, 1]")
    generic = np.asarray(generic_brass_color, dtype=np.float32)
    if generic.shape != (3,) or not np.isfinite(generic).all() or np.any(generic < 0.0) or np.any(generic > 1.0):
        raise ValueError("generic_brass_color must contain normalized RGB values")

    filled = color.copy()
    origin = np.zeros(shape, dtype=np.uint8)
    origin[plan.projection_defect_mask] = 4
    origin[plan.reviewed_source_gap_mask] = 1
    origin[plan.symmetry_fill_mask] = 2
    origin[plan.generic_fill_mask] = 3
    unresolved = (
        plan.projection_defect_mask
        | plan.reviewed_source_gap_mask
    )
    symmetry_mask = np.asarray(plan.symmetry_fill_mask, dtype=bool)
    if symmetry is None:
        unresolved |= symmetry_mask
        symmetry_count = 0
    else:
        filled[symmetry_mask] = symmetry[symmetry_mask]
        symmetry_count = int(symmetry_mask.sum())

    generic_mask = np.asarray(plan.generic_fill_mask, dtype=bool)
    filled[generic_mask] = generic
    return GapFillResult(
        base_color=_read_only(filled),
        fill_origin=_read_only(origin),
        unresolved_mask=_read_only(unresolved),
        symmetry_filled_count=symmetry_count,
        generic_filled_count=int(generic_mask.sum()),
    )


def load_projection_package(
    package_dir: Path,
    *,
    geometry_filename: str = "projection_geometry.npz",
    manifest_filename: str = "projection_manifest.json",
) -> ProjectionPackage:
    """Load and validate the compact Blender geometry export package."""

    directory = Path(package_dir)
    geometry_path = directory / geometry_filename
    manifest_path = directory / manifest_filename
    if not geometry_path.is_file() or not manifest_path.is_file():
        raise ValueError("projection geometry package is incomplete")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported projection package format version")
    source = manifest.get("source_blend")
    if not isinstance(source, dict):
        raise ValueError("projection package is missing source_blend provenance")
    blend_hash = str(source.get("sha256", ""))
    if not _HASH_256.fullmatch(blend_hash):
        raise ValueError("projection package has an invalid source blend SHA-256")
    blend_path = str(source.get("path", ""))
    if not blend_path:
        raise ValueError("projection package is missing the source blend path")
    coordinate_system = str(manifest.get("coordinate_system", ""))
    if not coordinate_system:
        raise ValueError("projection package is missing its coordinate system")

    with np.load(geometry_path, allow_pickle=False) as payload:
        required = {"vertices", "triangles", "loop_uvs", "loop_normals", "material_ids"}
        if not required.issubset(payload.files):
            raise ValueError("projection geometry arrays are incomplete")
        geometry = ProjectionGeometry(
            vertices=payload["vertices"],
            triangles=payload["triangles"],
            loop_uvs=payload["loop_uvs"],
            loop_normals=payload["loop_normals"],
            material_ids=payload["material_ids"],
        )
    return ProjectionPackage(
        geometry=geometry,
        source_blend_path=blend_path,
        source_blend_sha256=blend_hash,
        coordinate_system=coordinate_system,
        format_version=int(manifest["format_version"]),
    )


def coverage_summary(
    sample_count: np.ndarray,
    confidence: np.ndarray,
    *,
    minimum_samples: int,
) -> CoverageSummary:
    """Summarize direct sample coverage without inventing unsupported texels."""

    counts = np.asarray(sample_count)
    confidences = np.asarray(confidence, dtype=np.float64)
    if not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("sample_count must contain integer counts")
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("sample_count must be a square texture grid")
    if confidences.shape != counts.shape or not np.isfinite(confidences).all():
        raise ValueError("confidence must match sample_count and be finite")
    if np.any(confidences < 0.0) or np.any(confidences > 1.0):
        raise ValueError("confidence values must be in [0, 1]")
    if np.any(counts < 0):
        raise ValueError("sample counts cannot be negative")
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be positive")
    supported = (counts >= minimum_samples) & (confidences > 0.0)
    zero = counts == 0
    low_confidence = (counts > 0) & ~supported
    texel_count = counts.size
    return CoverageSummary(
        texture_size=int(counts.shape[0]),
        supported_texel_count=int(supported.sum()),
        direct_projection_percent=float(100.0 * supported.sum() / texel_count),
        low_confidence_percent=float(100.0 * low_confidence.sum() / texel_count),
        zero_sample_percent=float(100.0 * zero.sum() / texel_count),
    )


def inference_region_mask(
    sample_count: np.ndarray,
    confidence: np.ndarray,
    *,
    minimum_samples: int,
) -> np.ndarray:
    """Mark every texel lacking direct multi-view support for conservative fill."""

    counts = np.asarray(sample_count)
    confidences = np.asarray(confidence)
    return ~((counts >= minimum_samples) & (confidences > 0.0))


def derive_roughness_and_metallic(
    base_color: np.ndarray,
    *,
    highlight_variance: np.ndarray | None = None,
    recess_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive relative, evidence-bounded roughness while keeping brass metallic."""

    color = np.asarray(base_color, dtype=np.float32)
    if color.ndim != 3 or color.shape[2] != 3 or not np.isfinite(color).all():
        raise ValueError("base_color must have finite shape (height, width, 3)")
    if np.any(color < 0.0) or np.any(color > 1.0):
        raise ValueError("normalized base_color values must be in [0, 1]")
    variance = (
        np.zeros(color.shape[:2], dtype=np.float32)
        if highlight_variance is None
        else np.asarray(highlight_variance, dtype=np.float32)
    )
    if variance.shape != color.shape[:2] or not np.isfinite(variance).all():
        raise ValueError("highlight_variance must match base_color and be finite")
    if np.any(variance < 0.0) or np.any(variance > 1.0):
        raise ValueError("highlight_variance must be normalized to [0, 1]")
    recess = (
        np.zeros(color.shape[:2], dtype=bool)
        if recess_mask is None
        else np.asarray(recess_mask, dtype=bool)
    )
    if recess.shape != color.shape[:2]:
        raise ValueError("recess_mask must match base_color dimensions")

    luminance = (0.2126 * color[..., 0] + 0.7152 * color[..., 1] + 0.0722 * color[..., 2]).astype(np.float32)
    polished = np.clip(0.18 + 0.07 * variance, 0.14, 0.25).astype(np.float32)
    engraved = np.clip(0.32 + 0.08 * (1.0 - luminance) + 0.06 * variance, 0.30, 0.48).astype(np.float32)
    roughness = np.where(recess, engraved, polished).astype(np.float32)
    metallic = np.ones(color.shape[:2], dtype=np.float32)
    return roughness, metallic


def derive_wear_mask(
    base_color: np.ndarray,
    *,
    highlight_variance: np.ndarray | None = None,
    recess_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Derive a normalized, relative wear cue from repeatable photo evidence."""

    color = np.asarray(base_color, dtype=np.float32)
    if color.ndim != 3 or color.shape[2] != 3 or not np.isfinite(color).all():
        raise ValueError("base_color must have finite shape (height, width, 3)")
    if np.any(color < 0.0) or np.any(color > 1.0):
        raise ValueError("normalized base_color values must be in [0, 1]")
    variance = (
        np.zeros(color.shape[:2], dtype=np.float32)
        if highlight_variance is None
        else np.asarray(highlight_variance, dtype=np.float32)
    )
    if variance.shape != color.shape[:2] or not np.isfinite(variance).all():
        raise ValueError("highlight_variance must match base_color and be finite")
    if np.any(variance < 0.0) or np.any(variance > 1.0):
        raise ValueError("highlight_variance must be normalized to [0, 1]")
    recess = (
        np.zeros(color.shape[:2], dtype=np.float32)
        if recess_mask is None
        else np.asarray(recess_mask, dtype=np.float32)
    )
    if recess.shape != color.shape[:2]:
        raise ValueError("recess_mask must match base_color dimensions")
    if not np.isin(recess, (0.0, 1.0)).all():
        raise ValueError("recess_mask must be boolean-equivalent")

    luminance = (
        0.2126 * color[..., 0]
        + 0.7152 * color[..., 1]
        + 0.0722 * color[..., 2]
    ).astype(np.float32)
    wear = np.clip(
        0.20
        + 0.45 * (1.0 - luminance)
        + 0.25 * variance
        + 0.20 * recess,
        0.0,
        1.0,
    ).astype(np.float32)
    return wear


def _validate_texture_array(
    array: np.ndarray, size: int, name: str, channels: int | None = None
) -> np.ndarray:
    value = np.asarray(array)
    expected_shape = (size, size, channels) if channels is not None else (size, size)
    if value.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def _atomic_image_save(image: Image.Image, path: Path, v2_root: Path) -> None:
    resolved = ensure_under_v2_root(path, v2_root)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=resolved.parent, prefix=f".{resolved.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        image.save(temporary, format=resolved.suffix[1:].upper())
        temporary.replace(resolved)
    finally:
        temporary.unlink(missing_ok=True)


def assemble_final_texture_set(
    *,
    output_dir: Path,
    v2_root: Path,
    config: ProjectionConfig,
    base_color: np.ndarray,
    roughness: np.ndarray,
    wear: np.ndarray,
    metallic: np.ndarray,
    normal: np.ndarray,
    ambient_occlusion: np.ndarray,
    inferred_region_mask: np.ndarray,
    height: np.ndarray | None = None,
) -> FinalTextureSet:
    """Write the final texture set atomically and return hash provenance.

    AO is kept separate and is never multiplied permanently into BaseColor.
    """

    size = config.texture_size
    color = _validate_texture_array(base_color, size, "base_color", 3)
    if color.dtype != np.uint8:
        raise ValueError("base_color must already be uint8 sRGB")
    rough = _validate_texture_array(roughness, size, "roughness")
    wear_map = _validate_texture_array(wear, size, "wear")
    metal = _validate_texture_array(metallic, size, "metallic")
    normal_map = _validate_texture_array(normal, size, "normal", 3)
    ao = _validate_texture_array(ambient_occlusion, size, "ambient_occlusion")
    inferred = _validate_texture_array(inferred_region_mask, size, "inferred_region_mask")
    if inferred.dtype != np.bool_:
        raise ValueError("inferred_region_mask must be boolean")
    for name, values, low, high in (
        ("roughness", rough, 0.0, 1.0),
        ("wear", wear_map, 0.0, 1.0),
        ("metallic", metal, 0.0, 1.0),
        ("ambient_occlusion", ao, 0.0, 1.0),
    ):
        if np.any(values < low) or np.any(values > high):
            raise ValueError(f"{name} values must be in [0, 1]")
    if np.any(normal_map < -1.0) or np.any(normal_map > 1.0):
        raise ValueError("normal values must be in [-1, 1]")
    normal_lengths = np.linalg.norm(normal_map, axis=2)
    if np.any(normal_lengths <= 1e-6):
        raise ValueError("normal map contains degenerate vectors")

    maps: dict[str, tuple[np.ndarray, str, str]] = {
        "T_ThaiLibation_BaseColor.png": (color, "sRGB", "RGB"),
        "T_ThaiLibation_Roughness.png": (np.rint(rough * 255.0).astype(np.uint8), "Non-Color", "L"),
        "T_ThaiLibation_Wear.png": (
            np.rint(wear_map * 255.0).astype(np.uint8),
            "Non-Color",
            "L",
        ),
        "T_ThaiLibation_Metallic.png": (np.rint(metal * 255.0).astype(np.uint8), "Non-Color", "L"),
        "T_ThaiLibation_Normal.png": (
            np.rint((normal_map * 0.5 + 0.5) * 255.0).astype(np.uint8),
            "Non-Color",
            "RGB",
        ),
        "T_ThaiLibation_AO.png": (np.rint(ao * 255.0).astype(np.uint8), "Non-Color", "L"),
        "T_ThaiLibation_InferredRegionMask.png": (
            np.where(inferred, 255, 0).astype(np.uint8),
            "Non-Color",
            "L",
        ),
    }
    if height is not None:
        height_map = _validate_texture_array(height, size, "height")
        if np.any(height_map < 0.0) or np.any(height_map > 1.0):
            raise ValueError("height values must be in [0, 1]")
        maps["T_ThaiLibation_Height.png"] = (
            np.rint(height_map * 255.0).astype(np.uint8),
            "Non-Color",
            "L",
        )

    root = v2_root.resolve()
    records: list[dict[str, Any]] = []
    color_spaces: dict[str, str] = {}
    for filename, (values, color_space, mode) in maps.items():
        path = Path(output_dir) / filename
        _atomic_image_save(Image.fromarray(values, mode=mode), path, v2_root)
        resolved = path.resolve()
        records.append(
            {
                "path": resolved.relative_to(root).as_posix(),
                "sha256": sha256_file(resolved),
                "width": size,
                "height": size,
                "color_space": color_space,
            }
        )
        color_spaces[filename] = color_space
    return FinalTextureSet(tuple(records), color_spaces)


def build_texture_projection_report(
    *,
    source_blend_sha256: str,
    selection: ProjectionSelection,
    coverage: CoverageSummary,
    texture_records: Sequence[dict[str, Any]],
    config: ProjectionConfig,
    surface_evidence_sha256: str | None = None,
    photometric_harmonization: PhotometricHarmonization | None = None,
    coverage_class_percentages: Mapping[str, float] | None = None,
    gap_fill_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a truthful contract report without asserting visual QA or SHIP."""

    if not _HASH_256.fullmatch(source_blend_sha256):
        raise ValueError("source_blend_sha256 must be a lowercase 64-character SHA-256")
    direct = float(coverage.direct_projection_percent)
    inferred = max(0.0, 100.0 - direct)
    if not math.isclose(direct + inferred, 100.0, abs_tol=1e-9):
        raise ValueError("direct and inferred texture percentages must total 100")
    if direct < 0.0 or direct > 100.0:
        raise ValueError("direct projection percentage is outside [0, 100]")
    if surface_evidence_sha256 is not None and not _HASH_256.fullmatch(
        surface_evidence_sha256
    ):
        raise ValueError("surface_evidence_sha256 must be a lowercase 64-character SHA-256")
    if coverage.texture_size != config.texture_size:
        raise ValueError("coverage and texture configuration sizes disagree")
    if coverage.supported_texel_count > config.texture_size**2:
        raise ValueError("supported texel count exceeds the texture grid")
    expected_direct_percent = (
        100.0 * coverage.supported_texel_count / config.texture_size**2
    )
    if not math.isclose(
        coverage.direct_projection_percent, expected_direct_percent, abs_tol=1e-5
    ):
        raise ValueError("coverage direct percentage disagrees with supported texels")
    normalized_class_percentages: dict[str, float] | None = None
    if coverage_class_percentages is not None:
        if set(coverage_class_percentages) != set(_COVERAGE_CLASSES):
            raise ValueError("coverage class provenance must use the exact four classes")
        normalized_class_percentages = {
            coverage_class: float(coverage_class_percentages[coverage_class])
            for coverage_class in _COVERAGE_CLASS_ORDER
        }
        if any(
            not math.isfinite(value)
            or value < 0.0
            or value > 100.0
            for value in normalized_class_percentages.values()
        ):
            raise ValueError("coverage class percentages must be finite values in [0, 100]")
        if not math.isclose(
            sum(normalized_class_percentages.values()), 100.0, abs_tol=1e-6
        ):
            raise ValueError("coverage class percentages must total 100")
    normalized_gap_summary: dict[str, Any] | None = None
    if gap_fill_summary is not None:
        required_gap_fields = {
            "texture_size",
            "symmetry_filled_texel_count",
            "generic_filled_texel_count",
            "projection_defect_texel_count",
            "reviewed_source_gap_texel_count",
            "unresolved_texel_count",
        }
        if set(gap_fill_summary) != required_gap_fields:
            raise ValueError("gap-fill provenance fields are incomplete or unexpected")
        normalized_gap_summary = dict(gap_fill_summary)
        if int(normalized_gap_summary["texture_size"]) != config.texture_size:
            raise ValueError("gap-fill and texture configuration sizes disagree")
        for field in required_gap_fields - {"texture_size"}:
            value = int(normalized_gap_summary[field])
            if value < 0:
                raise ValueError("gap-fill counts cannot be negative")
            normalized_gap_summary[field] = value
    reasons: list[str] = []
    if len(selection.views) < 24:
        reasons.append("fewer_than_24_projection_views")
    if len(selection.views) > 64:
        reasons.append("more_than_64_projection_views")
    if config.texture_size < 4096:
        reasons.append("texture_size_below_4096")
    if not texture_records:
        reasons.append("final_texture_records_missing")
    if any(
        not _HASH_256.fullmatch(str(record.get("sha256", "")))
        for record in texture_records
    ):
        reasons.append("texture_hash_missing")
    records_by_name: dict[str, dict[str, Any]] = {}
    for record in texture_records:
        filename = Path(str(record.get("path", ""))).name
        if filename in records_by_name:
            reasons.append("duplicate_texture_record")
            break
        records_by_name[filename] = record
    if set(_REQUIRED_TEXTURE_COLOR_SPACES).difference(records_by_name):
        reasons.append("final_texture_set_incomplete")
    for filename, color_space in _REQUIRED_TEXTURE_COLOR_SPACES.items():
        if records_by_name.get(filename, {}).get("color_space") != color_space:
            reasons.append("texture_color_space_mismatch")
            break
    if any(
        int(record.get("width", -1)) != config.texture_size
        or int(record.get("height", -1)) != config.texture_size
        for record in texture_records
    ):
        reasons.append("texture_dimension_mismatch")
    if photometric_harmonization is None:
        reasons.append("photometric_harmonization_missing")
    else:
        correction_by_selected_index = {
            correction.selected_index: correction
            for correction in photometric_harmonization.corrections
        }
        selected_indices = {view.selected_index for view in selection.views}
        if not selected_indices.issubset(correction_by_selected_index):
            reasons.append("photometric_selected_view_missing")
        if any(
            not correction_by_selected_index[selected_index].accepted
            for selected_index in selected_indices.intersection(correction_by_selected_index)
        ):
            reasons.append("photometric_selected_view_unreliable")
    if normalized_class_percentages is None:
        reasons.append("coverage_class_provenance_missing")
    if surface_evidence_sha256 is None:
        reasons.append("surface_evidence_provenance_missing")
    if normalized_gap_summary is None:
        reasons.append("gap_fill_provenance_missing")
    elif int(normalized_gap_summary["unresolved_texel_count"]) > 0:
        reasons.append("unresolved_gap_fill_texels")
    status = "READY_FOR_MATERIALS" if not reasons else "BLOCKED"
    return {
        "schema_version": 1,
        "stage": "texture_projection",
        "status": status,
        "blocked_reasons": reasons,
        "claim_scope": "candidate_bound_projection_contract_no_visual_qa",
        "candidate": {
            "source_blend_sha256": source_blend_sha256,
            "surface_evidence_sha256": surface_evidence_sha256,
            "binding": "exact_projection_geometry_source",
        },
        "source_blend_sha256": source_blend_sha256,
        "selected_view_count": len(selection.views),
        "selected_views": list(selection.manifest_rows),
        "texture_size": config.texture_size,
        "projection_method": "SIMPLE_RADIAL_depth_masked_UV_projection",
        "visibility_method": "per_view_depth_map_comparison",
        "coordinate_model": "SIMPLE_RADIAL",
        "fusion_method": "weighted_per_channel_median_local_consistency",
        "derived_maps_claim": "roughness_and_wear_inferred_not_directly_measured",
        "highlight_rejection": {
            "highlight_value_cutoff": config.highlight_value_cutoff,
            "deep_shadow_value_cutoff": config.deep_shadow_value_cutoff,
            "specular_confidence_cutoff": _SPECULAR_CONFIDENCE_CUTOFF,
        },
        "photometric_harmonization": (
            None
            if photometric_harmonization is None
            else photometric_harmonization.report_dict()
        ),
        "coverage_class_percentages": normalized_class_percentages,
        "gap_fill_summary": normalized_gap_summary,
        "coverage": {
            "supported_texel_count": coverage.supported_texel_count,
            "direct_projection_percent": direct,
            "low_confidence_percent": coverage.low_confidence_percent,
            "zero_sample_percent": coverage.zero_sample_percent,
        },
        "direct_projection_percent": direct,
        "inferred_fill_percent": inferred,
        "textures": list(texture_records),
    }


def _required_report(path: Path, *, accepted: bool = False) -> dict[str, Any]:
    """Load one Final V2 JSON gate without weakening its acceptance contract."""

    if not path.is_file():
        raise ValueError(f"missing required Final V2 report: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable required report: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    if accepted and payload.get("accepted") is not True:
        raise ValueError(f"{path.name} is not accepted")
    return payload


def _trusted_component_median(
    source_rgb: np.ndarray,
    component_mask: np.ndarray,
    config: ProjectionConfig,
    row_span: tuple[int, int] | None = None,
    *,
    minimum_pixels: int = 32,
) -> tuple[float, float, float] | None:
    """Median trusted source color inside one reviewed component region."""

    image = np.asarray(source_rgb)
    mask = np.asarray(component_mask)
    if image.ndim != 3 or image.shape[2] != 3 or mask.ndim != 2:
        raise ValueError("component sampling expects RGB image and grayscale mask")
    if mask.shape != image.shape[:2]:
        raise ValueError("component mask and source image dimensions disagree")
    binary = mask > 127
    if row_span is not None:
        low, high = row_span
        if low < 0 or high < low or high >= binary.shape[0]:
            raise ValueError("component row span is outside the source image")
        row_selection = np.zeros(binary.shape, dtype=bool)
        row_selection[low : high + 1] = True
        binary = binary & row_selection
    luminance = image.max(axis=2) / 255.0
    trusted = binary & (
        luminance > config.deep_shadow_value_cutoff
    ) & (luminance < config.highlight_value_cutoff)
    if int(trusted.sum()) < minimum_pixels:
        return None
    median = np.median(image[trusted], axis=0)
    return tuple(float(value) for value in median)


def _component_gradient_stops(
    source_rgb: np.ndarray,
    component_mask: np.ndarray,
    config: ProjectionConfig,
) -> list[tuple[float, float, float] | None]:
    """Measure lower/middle/upper component colors from one source photograph."""

    mask = np.asarray(component_mask)
    foreground_rows = np.flatnonzero((mask > 127).any(axis=1))
    if len(foreground_rows) == 0:
        return [None, None, None]
    top, bottom = int(foreground_rows[0]), int(foreground_rows[-1])
    span = max(bottom - top + 1, 3)
    edges = (
        top,
        top + span // 3,
        top + (2 * span) // 3,
        bottom,
    )
    return [
        _trusted_component_median(source_rgb, mask, config, (edges[index], edges[index + 1]))
        for index in range(3)
    ]


def _profile_boundaries(profiles: Mapping[str, Sequence[Sequence[float]]]) -> dict[str, float]:
    """Derive normalized-height component boundaries from accepted Plan-1 profiles."""

    def extrema(name: str) -> tuple[float, float]:
        samples = profiles[name]
        heights = [float(sample[0]) for sample in samples]
        return min(heights), max(heights)

    return {
        "receiving_top": extrema("bowl_outer")[1],
        "globe_bottom": extrema("globe")[0],
        "globe_top": extrema("shoulder")[1],
        "neck_top": extrema("neck_outer")[1],
    }


def project_final_textures(project_root: Path, v2_root: Path) -> dict[str, Any]:
    """Build the Plan-5 texture checkpoint from accepted V2 evidence.

    The exact per-texel path is intentionally fail-closed: it requires the
    Blender geometry export package and per-view depth maps.  Until that
    contract exists, this entry point records the missing prerequisite and uses
    a bounded, disclosed fallback: accepted Plan-1 component masks provide one
    coarse geometric correspondence per component per view, trusted source
    medians feed the existing overlap harmonization and robust fusion helpers,
    and the resulting component-level brass colors are assembled into the
    required 4096 texture set.  The report never claims per-texel projection.
    """

    root = Path(project_root).resolve()
    v2 = Path(v2_root).resolve()
    ensure_under_v2_root(v2, v2)
    report_dir = ensure_under_v2_root(v2 / "reports", v2)
    config = ProjectionConfig()

    fit_report = _required_report(report_dir / "final_cv_fit.json", accepted=True)
    profiles_report = _required_report(report_dir / "final_profiles.json")
    reference_report = _required_report(
        report_dir / "final_reference_evidence.json", accepted=True
    )
    surface_report = _required_report(
        report_dir / "surface_evidence_coverage.json", accepted=True
    )
    profiles = profiles_report.get("profiles")
    if not isinstance(profiles, dict) or not {
        "bowl_outer", "globe", "shoulder", "neck_outer"
    }.issubset(profiles):
        raise ValueError("accepted final_profiles.json lacks the required Plan-1 profiles")
    boundaries = _profile_boundaries(profiles)

    package_dir = v2 / "work" / "projection_geometry"
    package_present = (package_dir / "projection_geometry.npz").is_file() and (
        package_dir / "projection_manifest.json"
    ).is_file()

    from final_reference_evidence import load_reviewed_reference_views

    reviewed_by_index = {
        view.selected_index: view for view in load_reviewed_reference_views(root)
    }
    component_rows = {
        int(row["selected_index"]): row
        for row in reference_report["component_masks"]["views"]
    }
    fit_views = [view for view in fit_report["views"] if view.get("fit_included") is True]
    if len(fit_views) < 2:
        raise ValueError("accepted fit has fewer than two reliable component views")

    components = ("receiving_bowl_pedestal", "globe", "neck", "lid_finial")
    stop_observations: dict[
        tuple[str, int], list[tuple[int, int, tuple[float, float, float]]]
    ] = {}
    middle_observations: list[OverlapObservation] = []
    view_manifest: list[dict[str, Any]] = []
    for view_index, fit_view in enumerate(fit_views):
        selected_index = int(fit_view["selected_index"])
        reviewed = reviewed_by_index.get(selected_index)
        component_record = component_rows.get(selected_index)
        if reviewed is None or component_record is None:
            raise ValueError(f"missing reviewed provenance for fit view {selected_index:03d}")
        source_path = root / str(fit_view["source_path"])
        if not source_path.is_file():
            raise ValueError(f"fit source photograph is missing: {source_path.name}")
        with Image.open(source_path) as source_image:
            source_rgb = np.asarray(source_image.convert("RGB"))
        for component in components:
            component_path = root / str(component_record["components"][component]["path"])
            with Image.open(component_path) as component_image:
                component_mask = np.asarray(component_image.convert("L"))
            if component_mask.shape != source_rgb.shape[:2]:
                raise ValueError(
                    f"component mask dimensions disagree with source: view {selected_index:03d}"
                )
            stops = _component_gradient_stops(source_rgb, component_mask, config)
            for stop_index, color in enumerate(stops):
                if color is not None:
                    stop_observations.setdefault((component, stop_index), []).append(
                        (view_index, selected_index, color)
                    )
            middle = stops[1]
            if middle is not None:
                middle_observations.append(
                    OverlapObservation(
                        surface_id=component,
                        view_index=view_index,
                        selected_index=selected_index,
                        color_srgb=middle,
                    )
                )
        view_manifest.append(
            {
                "view_index": view_index,
                "selected_index": selected_index,
                "filename": str(fit_view["filename"]),
                "view_category": str(fit_view["view_category"]),
                "quality_condition": reviewed.quality_condition,
                "fit_included": True,
                "source_path": str(fit_view["source_path"]),
                "source_sha256": reviewed.source_sha256,
                "component_mask_provenance": {
                    name: str(record["path"])
                    for name, record in component_record["components"].items()
                },
                "inclusion_reason": "accepted_plan1_fit_component_color_fusion",
            }
        )

    minimum_overlap_samples = max(2, min(4, len(components)))
    harmonization = estimate_photometric_harmonization(
        middle_observations,
        config=config,
        minimum_overlap_samples=minimum_overlap_samples,
    )
    correction_by_view = {
        correction.view_index: correction
        for correction in harmonization.corrections
    }

    fused_stops: dict[str, list[tuple[float, float, float]]] = {}
    fusion_diagnostics: dict[str, list[dict[str, Any]]] = {}
    for component in components:
        fused_stops[component] = []
        fusion_diagnostics[component] = []
        for stop_index in range(3):
            observations = stop_observations.get((component, stop_index), [])
            colors: list[tuple[float, float, float]] = []
            weights: list[float] = []
            for observation_view_index, observation_selected_index, observation in observations:
                color_array = np.asarray(observation, dtype=np.float64).reshape(1, 3)
                correction = correction_by_view.get(observation_view_index)
                fused_color = color_array
                if correction is not None and correction.accepted:
                    fused_color = apply_photometric_correction(
                        color_array, correction
                    ).astype(np.float64)
                maximum = float(fused_color.max())
                minimum = float(fused_color.min())
                luminance = maximum / 255.0
                saturation = 0.0 if maximum <= 0.0 else (maximum - minimum) / maximum
                weight = view_weight(
                    cos_view_angle=1.0,
                    mask_confidence=1.0,
                    luminance=luminance,
                    saturation=saturation,
                    visible=True,
                    quality_condition=str(
                        reviewed_by_index[observation_selected_index].quality_condition
                    ),
                    config=config,
                )
                colors.append(tuple(float(value) for value in fused_color[0]))
                weights.append(weight)
            if len(colors) < config.minimum_samples_per_texel:
                raise ValueError(
                    f"insufficient trusted component observations: {component} stop {stop_index}"
                )
            fused = robust_fuse_colors(
                np.asarray(colors, dtype=np.float64),
                np.asarray(weights, dtype=np.float64),
                config,
            )
            if fused is None:
                raise ValueError(
                    f"robust component fusion rejected all samples: {component} stop {stop_index}"
                )
            fused_stops[component].append(fused.color_srgb)
            fusion_diagnostics[component].append(
                {
                    "stop": ("lower", "middle", "upper")[stop_index],
                    "view_count": len(colors),
                    "color_srgb_fused": list(fused.color_srgb),
                    "confidence": fused.confidence,
                    "accepted_count": fused.accepted_count,
                }
            )

    size = config.texture_size
    z_values = 1.0 - (np.arange(size, dtype=np.float64) + 0.5) / size

    def stop_interpolation(component: str, heights: np.ndarray) -> np.ndarray:
        low, high = (
            (0.0, boundaries["receiving_top"])
            if component == "receiving_bowl_pedestal"
            else (
                (boundaries["globe_bottom"], boundaries["globe_top"])
                if component == "globe"
                else (
                    (boundaries["globe_top"], boundaries["neck_top"])
                    if component == "neck"
                    else (boundaries["neck_top"], 1.0)
                )
            )
        )
        relative = np.clip((heights - low) / max(high - low, 1e-9), 0.0, 1.0)
        stops = np.asarray(fused_stops[component], dtype=np.float64)
        return np.stack(
            [np.interp(relative, (0.0, 0.5, 1.0), stops[:, channel]) for channel in range(3)],
            axis=1,
        )

    receiving_colors = stop_interpolation("receiving_bowl_pedestal", z_values)
    globe_colors = stop_interpolation("globe", z_values)
    neck_colors = stop_interpolation("neck", z_values)
    lid_colors = stop_interpolation("lid_finial", z_values)
    crossfade = np.clip(
        (z_values - boundaries["globe_bottom"])
        / max(boundaries["receiving_top"] - boundaries["globe_bottom"], 1e-9),
        0.0,
        1.0,
    )
    row_colors = (
        receiving_colors * (1.0 - crossfade)[:, None]
        + globe_colors * crossfade[:, None]
    )
    row_colors = np.where(
        z_values[:, None] <= boundaries["receiving_top"], row_colors, globe_colors
    )
    row_colors = np.where(
        z_values[:, None] > boundaries["globe_top"], neck_colors, row_colors
    )
    row_colors = np.where(
        z_values[:, None] > boundaries["neck_top"], lid_colors, row_colors
    )

    view_spread = np.zeros(size, dtype=np.float32)
    for component, (low, high) in (
        ("receiving_bowl_pedestal", (0.0, boundaries["receiving_top"])),
        ("globe", (boundaries["globe_bottom"], boundaries["globe_top"])),
        ("neck", (boundaries["globe_top"], boundaries["neck_top"])),
        ("lid_finial", (boundaries["neck_top"], 1.0)),
    ):
        observations = np.asarray(
            [record[2] for record in stop_observations.get((component, 1), [])],
            dtype=np.float64,
        )
        spread = (
            float(np.std(observations.max(axis=1) / 255.0, ddof=1))
            if len(observations) > 1
            else 0.0
        )
        selected = (z_values >= low) & (z_values <= high)
        view_spread[selected] = np.clip(spread, 0.0, 1.0)

    base_color = np.rint(np.clip(row_colors, 0.0, 255.0)).astype(np.uint8)
    base_color = np.repeat(base_color[:, None, :], size, axis=1)
    highlight_variance = np.repeat(view_spread[:, None], size, axis=1)
    roughness, metallic = derive_roughness_and_metallic(
        base_color.astype(np.float32) / 255.0,
        highlight_variance=highlight_variance,
    )
    wear = derive_wear_mask(
        base_color.astype(np.float32) / 255.0,
        highlight_variance=highlight_variance,
    )
    normal = np.zeros((size, size, 3), dtype=np.float32)
    normal[:, :, 2] = 1.0
    ambient_occlusion = np.ones((size, size), dtype=np.float32)
    inferred_region = np.ones((size, size), dtype=bool)

    texture_dir = v2 / "textures" / "final"
    texture_set = assemble_final_texture_set(
        output_dir=texture_dir,
        v2_root=v2,
        config=config,
        base_color=base_color,
        roughness=roughness,
        wear=wear,
        metallic=metallic,
        normal=normal,
        ambient_occlusion=ambient_occlusion,
        inferred_region_mask=inferred_region,
    )

    blocked_reasons = []
    if not package_present:
        blocked_reasons.append("exact_projection_geometry_package_missing")
    blocked_reasons.extend(
        (
            "per_texel_depth_masked_projection_not_run",
            "component_level_photo_informed_fallback_only",
        )
    )
    texel_count = size * size
    report = {
        "schema_version": 1,
        "stage": "texture_projection",
        "status": "BLOCKED" if blocked_reasons else "READY_FOR_MATERIALS",
        "blocked_reasons": blocked_reasons,
        "claim_scope": "photo_informed_component_fusion_fallback_no_visual_qa",
        "fallback": {
            "active": True,
            "reason": (
                "Blender UV geometry export and per-view depth maps are not yet "
                "available for exact per-texel projection"
            ),
            "exact_projection_package_present": package_present,
            "exact_projection_package_path": package_dir.relative_to(root).as_posix(),
            "component_correspondence": "accepted_plan1_reviewed_component_masks",
            "texel_direct_projection": False,
            "azimuthal_ornament_detail_projected": False,
            "vertical_component_gradients": True,
            "normal_map": "neutral_flat_pending_plan4_bake_integration",
            "ambient_occlusion_map": "neutral_pending_plan4_bake_integration",
        },
        "projection_method": "component_level_photo_informed_fusion_fallback",
        "visibility_method": "reviewed_component_masks_no_uv_depth_visibility",
        "coordinate_model": "plan1_normalized_height_bands",
        "fusion_method": "per_component_weighted_median_local_consistency",
        "derived_maps_claim": "roughness_and_wear_inferred_not_directly_measured",
        "highlight_rejection": {
            "highlight_value_cutoff": config.highlight_value_cutoff,
            "deep_shadow_value_cutoff": config.deep_shadow_value_cutoff,
        },
        "accepted_input_provenance": {
            "final_cv_fit_sha256": sha256_file(report_dir / "final_cv_fit.json"),
            "final_profiles_sha256": sha256_file(report_dir / "final_profiles.json"),
            "final_reference_evidence_sha256": sha256_file(
                report_dir / "final_reference_evidence.json"
            ),
            "surface_evidence_coverage_sha256": sha256_file(
                report_dir / "surface_evidence_coverage.json"
            ),
        },
        "component_height_boundaries": boundaries,
        "selected_view_count": len(view_manifest),
        "selected_views": view_manifest,
        "component_fusion": fusion_diagnostics,
        "photometric_harmonization": harmonization.report_dict(),
        "photometric_correspondence_note": (
            "One overlap observation per accepted component mask and view; "
            f"minimum_overlap_samples={minimum_overlap_samples}; relative "
            "harmonization only, not reflectance recovery"
        ),
        "texture_size": size,
        "coverage": {
            "supported_texel_count": 0,
            "direct_projection_percent": 0.0,
            "low_confidence_percent": 0.0,
            "zero_sample_percent": 100.0,
        },
        "direct_projection_percent": 0.0,
        "inferred_fill_percent": 100.0,
        "inferred_texel_count": texel_count,
        "textures": list(texture_set.texture_records),
    }
    report_path = report_dir / "texture_projection_report.json"
    write_json_atomic(report_path, report, v2)
    return {
        "stage": "texture_projection",
        "status": report["status"],
        "blocked_reasons": blocked_reasons,
        "method": report["projection_method"],
        "selected_view_count": len(view_manifest),
        "report": report_path.relative_to(root).as_posix(),
        "textures": [record["path"] for record in texture_set.texture_records],
    }
