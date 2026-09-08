"""Validated computer-vision reference catalog for the final V2 model."""

import csv
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from analysis_common import load_selected_manifest
from external_learned_recovery import (
    ExternalLearnedConfig,
    FrontendBundle,
    ImageFeatures,
    extract_image_features,
    load_frontend,
    match_feature_pair,
)
from final_model_io import (
    V2_RELATIVE_ROOT,
    ensure_under_v2_root,
    sha256_file,
    write_json_atomic,
)
from geometry_detection import (
    GeometryMatchResult,
    ImageScale,
    SiftConfig,
    SiftFeatures,
    estimate_fundamental_geometry,
    extract_sift,
    match_sift,
    sampson_errors,
    select_display_matches,
)
from reconstruction_masks import load_mask_manifest
from segmentation_data import load_segmentation_manifest


SELECTION_MANIFEST = Path("preprocessing/reports/selection_manifest.csv")
SELECTED_IMAGES = Path("preprocessing/pycolmap_input/images")
REVIEWED_MANIFEST = Path("ml_dataset/manifest.csv")
MASK_MANIFEST = Path("analysis/reports/reconstruction_mask_manifest.csv")
STEP13_REGISTRATION = Path(
    "reconstruction/external_learned_recovery/reports/step13_registered_images.csv"
)
STEP13_BEST_MODEL = Path("reconstruction/external_learned_recovery/best")
V1_PREVIEWS = (
    Path("reconstruction/reference_assisted/previews/reference_front.png"),
    Path("reconstruction/reference_assisted/previews/reference_quarter.png"),
)

DETAIL_REFERENCE_INDICES = {
    "neck_lid_detail": (148, 151, 154, 165),
    "globe_ornament_detail": (267, 268, 278, 288),
    "bowl_pedestal_detail": (90, 114, 128, 142),
}

PREFERRED_GEOMETRY_INDICES = {
    "normal_side": (3, 19, 50, 72),
    "low_angle_pedestal": (90, 114, 128, 142),
    "elevated_oblique": (148, 165, 188, 200),
    "top_down_rim": (206, 221, 243, 255),
}

V1_REJECTION_CATEGORIES = (
    "wrong globe/body profile",
    "wrong bowl/pedestal profile",
    "wrong neck proportions",
    "unsupported generic decoration",
    "unsupported chain unless later evidence found",
    "material/value mismatch",
    "insufficient single-aggregate validation",
)


@dataclass(frozen=True)
class FinalReferenceView:
    selected_index: int
    filename: str
    split: str
    view_category: str
    quality_condition: str
    source_sha256: str
    reviewed_mask_path: Path
    reviewed_mask_sha256: str
    cnn_prediction_path: Path
    reconstruction_mask_path: Path
    step13_registered: bool
    step13_image_id: int | None


@dataclass(frozen=True)
class PairAlignmentDecision:
    method: str
    match_count: int
    inlier_count: int
    homography: tuple[float, ...] | None
    reason: str


@dataclass(frozen=True)
class Landmark:
    name: str
    x: float
    y: float
    normalized_x: float
    normalized_y: float
    confidence: str
    annotation_method: str


@dataclass(frozen=True)
class OrnamentFamily:
    family_id: str
    host_component: str
    representation: str
    primary_view_indices: tuple[int, ...]
    repeat_mode: str
    repeat_count: int | None
    confidence: str
    notes: str


BASE_LANDMARK_NAMES = (
    "axis_top",
    "axis_bottom",
    "finial_top",
    "finial_bottom",
    "lid_max_left",
    "lid_max_right",
    "lid_lower_left",
    "lid_lower_right",
    "neck_top_left",
    "neck_top_right",
    "neck_base_left",
    "neck_base_right",
    "globe_max_left",
    "globe_max_right",
    "globe_top_transition",
    "globe_bottom_transition",
    "bowl_rim_left",
    "bowl_rim_right",
    "bowl_bottom_transition_left",
    "bowl_bottom_transition_right",
    "pedestal_waist_left",
    "pedestal_waist_right",
    "foot_left",
    "foot_right",
)

TOP_DOWN_LANDMARK_NAMES = (
    "bowl_rim_center",
    "bowl_rim_major_axis_a",
    "bowl_rim_major_axis_b",
    "globe_center",
)

ORNAMENT_FAMILIES = (
    OrnamentFamily(
        "ORB_GLOBE_CROSSHATCH",
        "vessel_globe",
        "normal_height",
        (267, 268, 278, 288),
        "surface_field",
        None,
        "high",
        "Dense cross-hatched and pebbled field visible across close oblique globe views.",
    ),
    OrnamentFamily(
        "ORB_GLOBE_HERO_MOTIF",
        "vessel_globe",
        "highpoly_bake",
        (267, 268, 288),
        "radial_repetition",
        6,
        "medium",
        "Large Thai flame/floral medallion; six repeats are an evidence-bounded radial inference.",
    ),
    OrnamentFamily(
        "ORB_GLOBE_SCROLL_BAND",
        "vessel_globe",
        "normal_height",
        (148, 165, 267, 268),
        "radial_band",
        None,
        "high",
        "Scrolling floral band around the lower globe.",
    ),
    OrnamentFamily(
        "ORB_GLOBE_LOWER_BAND",
        "vessel_globe",
        "explicit_geometry",
        (3, 72, 148, 165),
        "continuous_ring",
        1,
        "high",
        "Horizontal construction seam and lower textured band.",
    ),
    OrnamentFamily(
        "ORB_SHOULDER_RINGS",
        "vessel_shoulder",
        "explicit_geometry",
        (148, 165, 267, 268),
        "continuous_rings",
        4,
        "high",
        "Concentric stepped shoulder rings visible in elevated and close oblique photographs.",
    ),
    OrnamentFamily(
        "ORB_NECK_FIELD",
        "vessel_neck",
        "normal_height",
        (3, 19, 148, 165),
        "surface_field",
        None,
        "medium",
        "Fine cross-hatched neck background field.",
    ),
    OrnamentFamily(
        "ORB_NECK_LOTUS",
        "vessel_neck",
        "highpoly_bake",
        (3, 19, 148, 165),
        "radial_repetition",
        6,
        "medium",
        "Tall lotus/flame panels visible around the tapered neck.",
    ),
    OrnamentFamily(
        "ORB_NECK_UPPER_BAND",
        "vessel_neck",
        "normal_height",
        (3, 19, 148, 165),
        "radial_band",
        None,
        "high",
        "Decorated upper collar below the lid.",
    ),
    OrnamentFamily(
        "ORB_LID_TIERS",
        "lid",
        "explicit_geometry",
        (148, 165, 206, 221),
        "continuous_rings",
        6,
        "high",
        "Six progressively smaller stepped lid rings plus finial seat.",
    ),
    OrnamentFamily(
        "ORB_LID_DECOR_BAND",
        "lid",
        "normal_height",
        (3, 19, 148, 165),
        "radial_band",
        None,
        "medium",
        "Narrow decorated band immediately below the conical tiers.",
    ),
    OrnamentFamily(
        "ORB_BOWL_FLAME_BAND",
        "receiving_bowl",
        "highpoly_bake",
        (90, 114, 128, 142),
        "radial_repetition",
        8,
        "high",
        "Repeated raised flame/lotus motif around the receiving bowl.",
    ),
    OrnamentFamily(
        "ORB_PEDESTAL_RINGS",
        "pedestal",
        "explicit_geometry",
        (90, 114, 128, 142),
        "continuous_rings",
        4,
        "high",
        "Foot and waist construction rings.",
    ),
)

# Normalized crop boxes are deliberately broad.  They preserve context needed to
# distinguish a photographed motif from glare, neighboring rings, or background.
ORNAMENT_CROP_BOXES = {
    "ORB_GLOBE_CROSSHATCH": (267, (0.16, 0.18, 0.84, 0.70)),
    "ORB_GLOBE_HERO_MOTIF": (267, (0.16, 0.20, 0.84, 0.70)),
    "ORB_GLOBE_SCROLL_BAND": (268, (0.14, 0.34, 0.86, 0.72)),
    "ORB_GLOBE_LOWER_BAND": (148, (0.16, 0.48, 0.84, 0.72)),
    "ORB_SHOULDER_RINGS": (267, (0.25, 0.02, 0.75, 0.33)),
    "ORB_NECK_FIELD": (165, (0.32, 0.04, 0.68, 0.39)),
    "ORB_NECK_LOTUS": (165, (0.30, 0.03, 0.70, 0.38)),
    "ORB_NECK_UPPER_BAND": (165, (0.28, 0.01, 0.72, 0.22)),
    "ORB_LID_TIERS": (148, (0.33, 0.00, 0.67, 0.23)),
    "ORB_LID_DECOR_BAND": (148, (0.30, 0.03, 0.70, 0.28)),
    "ORB_BOWL_FLAME_BAND": (90, (0.12, 0.38, 0.88, 0.78)),
    "ORB_PEDESTAL_RINGS": (90, (0.22, 0.68, 0.78, 0.99)),
}


def _label_font(cell_height: int):
    font_size = max(12, min(22, cell_height // 18))
    windows_arial = Path("C:/Windows/Fonts/arial.ttf")
    if windows_arial.is_file():
        return ImageFont.truetype(str(windows_arial), font_size)
    return ImageFont.load_default()


def make_labeled_contact_sheet(
    image_paths: Sequence[Path],
    labels: Sequence[str],
    output_path: Path,
    *,
    cell_size: tuple[int, int] = (512, 512),
    columns: int = 4,
) -> Path:
    """Create a deterministic, aspect-preserving labeled image grid."""

    if len(image_paths) != len(labels):
        raise ValueError("contact sheet requires one label per image")
    if not image_paths:
        raise ValueError("contact sheet requires at least one image")
    cell_width, cell_height = cell_size
    if cell_width < 32 or cell_height < 48 or columns < 1:
        raise ValueError("invalid contact-sheet layout")

    font = _label_font(cell_height)
    label_height = max(28, min(44, cell_height // 6))
    rows = math.ceil(len(image_paths) / columns)
    sheet = Image.new(
        "RGB",
        (cell_width * columns, cell_height * rows),
        (18, 18, 18),
    )
    draw = ImageDraw.Draw(sheet)

    for position, (image_path, label) in enumerate(
        zip(image_paths, labels, strict=True)
    ):
        row, column = divmod(position, columns)
        cell_x = column * cell_width
        cell_y = row * cell_height
        with Image.open(image_path) as opened:
            source = ImageOps.exif_transpose(opened).convert("RGB")
            thumbnail = ImageOps.contain(
                source,
                (cell_width, cell_height - label_height),
                method=Image.Resampling.LANCZOS,
            )
        paste_x = cell_x + (cell_width - thumbnail.width) // 2
        paste_y = cell_y + (cell_height - label_height - thumbnail.height) // 2
        sheet.paste(thumbnail, (paste_x, paste_y))
        label_top = cell_y + cell_height - label_height
        draw.rectangle(
            (cell_x, label_top, cell_x + cell_width - 1, cell_y + cell_height - 1),
            fill=(10, 10, 10),
        )
        draw.text(
            (cell_x + 8, label_top + 6),
            str(label),
            fill=(245, 245, 245),
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG", compress_level=9)
    return output_path


def build_reference_boards(
    project_root: Path,
    views: Sequence[FinalReferenceView] | None = None,
) -> dict[str, dict[str, object]]:
    """Build the eight source-reference boards required by the V2 plan."""

    root = project_root.resolve()
    v2_root = root / V2_RELATIVE_ROOT
    board_dir = ensure_under_v2_root(
        v2_root / "evidence" / "reference_boards", v2_root
    )
    reviewed_views = tuple(views) if views is not None else load_reviewed_reference_views(root)
    selected = load_selected_manifest(root / SELECTION_MANIFEST)
    selected_by_index = {record.index: record for record in selected}
    category_names = (
        "normal_side",
        "low_angle_pedestal",
        "elevated_oblique",
        "top_down_rim",
        "oblique_detail",
    )
    indices_by_board = {
        category: tuple(
            view.selected_index
            for view in reviewed_views
            if view.view_category == category
        )
        for category in category_names
    }
    indices_by_board.update(DETAIL_REFERENCE_INDICES)

    reports: dict[str, dict[str, object]] = {}
    for board_name, indices in indices_by_board.items():
        records = tuple(selected_by_index[index] for index in indices)
        image_paths = tuple(root / SELECTED_IMAGES / record.filename for record in records)
        labels = tuple(f"{record.index:03d} | {record.filename}" for record in records)
        output_path = ensure_under_v2_root(board_dir / f"{board_name}.png", v2_root)
        make_labeled_contact_sheet(image_paths, labels, output_path)
        reports[board_name] = {
            "path": output_path.relative_to(root).as_posix(),
            "sha256": sha256_file(output_path),
            "selected_indices": list(indices),
            "filenames": [record.filename for record in records],
        }
    return reports


def build_v1_rejection_board(project_root: Path) -> dict[str, object]:
    """Place the visually rejected V1 beside representative source evidence."""

    root = project_root.resolve()
    v2_root = root / V2_RELATIVE_ROOT
    board_dir = ensure_under_v2_root(
        v2_root / "evidence" / "reference_boards", v2_root
    )
    selected = load_selected_manifest(root / SELECTION_MANIFEST)
    representative_indices = (3, 72, 148, 165)
    representative = tuple(selected[index - 1] for index in representative_indices)
    source_paths = tuple(root / SELECTED_IMAGES / record.filename for record in representative)
    v1_paths = tuple(root / relative for relative in V1_PREVIEWS)
    image_paths = source_paths + v1_paths
    labels = tuple(
        f"SOURCE {record.index:03d} | {record.filename}" for record in representative
    ) + ("REJECTED V1 | front", "REJECTED V1 | quarter")
    output_path = ensure_under_v2_root(board_dir / "v1_rejection.png", v2_root)
    make_labeled_contact_sheet(image_paths, labels, output_path, columns=3)
    return {
        "path": output_path.relative_to(root).as_posix(),
        "sha256": sha256_file(output_path),
        "representative_source_indices": list(representative_indices),
        "v1_preview_paths": [path.relative_to(root).as_posix() for path in v1_paths],
        "mismatch_categories": list(V1_REJECTION_CATEGORIES),
    }


def _scale_report(scale: ImageScale) -> dict[str, object]:
    return {
        "original_size": list(scale.original_size),
        "analysis_size": list(scale.analysis_size),
        "scale_x_to_original": float(scale.scale_x_to_original),
        "scale_y_to_original": float(scale.scale_y_to_original),
    }


def build_sift_pair_diagnostic(
    first_path: Path,
    second_path: Path,
    output_path: Path,
    *,
    config: SiftConfig = SiftConfig(maximum_width=1200),
) -> dict[str, object]:
    """Measure and render one pair through the existing Step 6 SIFT pipeline."""

    first_image = cv2.imread(str(first_path), cv2.IMREAD_COLOR)
    second_image = cv2.imread(str(second_path), cv2.IMREAD_COLOR)
    if first_image is None:
        raise ValueError(f"unreadable SIFT diagnostic image: {first_path}")
    if second_image is None:
        raise ValueError(f"unreadable SIFT diagnostic image: {second_path}")

    first = extract_sift(first_image, config)
    second = extract_sift(second_image, config)
    matches = match_sift(first, second, ratio_threshold=config.ratio_threshold)
    geometry = estimate_fundamental_geometry(
        first,
        second,
        matches,
        ransac_threshold=config.ransac_threshold,
        confidence=config.confidence,
        rng_seed=config.rng_seed,
        minimum_correspondences=config.minimum_correspondences,
    )

    median_sampson_error: float | None = None
    if geometry.status == "ok" and geometry.fundamental_matrix is not None:
        residuals = sampson_errors(
            geometry.fundamental_matrix,
            geometry.points_a[geometry.inlier_mask],
            geometry.points_b[geometry.inlier_mask],
        )
        finite = residuals[np.isfinite(residuals)]
        if finite.size:
            median_sampson_error = float(np.median(finite))

    display_matches = select_display_matches(
        first,
        second,
        geometry.inlier_matches or geometry.candidate_matches,
        max_matches=80,
    )
    rendered = cv2.drawMatches(
        first.analysis_image,
        list(first.keypoints),
        second.analysis_image,
        list(second.keypoints),
        list(display_matches),
        None,
        matchColor=(40, 205, 40) if geometry.inlier_matches else (0, 150, 255),
        singlePointColor=(180, 180, 180),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    header = np.full((92, rendered.shape[1], 3), 245, dtype=np.uint8)
    lines = (
        f"Step 6 SIFT/RANSAC: {first_path.name} | {second_path.name}",
        f"candidates={geometry.candidate_count} inliers={geometry.inlier_count} "
        f"ratio={geometry.inlier_ratio:.3f} status={geometry.status}",
    )
    for line_index, line in enumerate(lines):
        cv2.putText(
            header,
            line,
            (20, 33 + line_index * 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (24, 24, 24),
            2,
            cv2.LINE_AA,
        )
    diagnostic = np.vstack((header, rendered))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), diagnostic):
        raise OSError(f"failed to write SIFT diagnostic: {output_path}")

    return {
        "first_path": str(first_path),
        "second_path": str(second_path),
        "output_path": str(output_path),
        "candidate_matches": geometry.candidate_count,
        "ransac_inliers": geometry.inlier_count,
        "inlier_ratio": geometry.inlier_ratio,
        "fundamental_matrix_status": geometry.status,
        "median_sampson_error": median_sampson_error,
        "first_scale": _scale_report(first.scale),
        "second_scale": _scale_report(second.scale),
    }


def _estimate_homography(
    first_points: np.ndarray,
    second_points: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray]:
    first = np.asarray(first_points, dtype=np.float32)
    second = np.asarray(second_points, dtype=np.float32)
    if first.shape != second.shape or first.ndim != 2 or first.shape[1:] != (2,):
        raise ValueError("homography points must be matching finite Nx2 arrays")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("homography points must be finite")
    if len(first) < 4:
        return None, np.zeros(len(first), dtype=bool)
    cv2.setRNGSeed(4213)
    try:
        homography, mask = cv2.findHomography(
            first,
            second,
            cv2.RANSAC,
            3.0,
            maxIters=5000,
            confidence=0.995,
        )
    except cv2.error:
        homography, mask = None, None
    if (
        homography is None
        or np.asarray(homography).shape != (3, 3)
        or not np.isfinite(homography).all()
        or mask is None
        or np.asarray(mask).size != len(first)
    ):
        return None, np.zeros(len(first), dtype=bool)
    return np.asarray(homography, dtype=np.float64), np.asarray(mask).reshape(-1).astype(bool)


def _homography_to_original(
    homography: np.ndarray,
    first_scale: ImageScale,
    second_scale: ImageScale,
) -> np.ndarray:
    first_transform = np.diag(
        (first_scale.scale_x_to_original, first_scale.scale_y_to_original, 1.0)
    )
    second_transform = np.diag(
        (second_scale.scale_x_to_original, second_scale.scale_y_to_original, 1.0)
    )
    converted = second_transform @ homography @ np.linalg.inv(first_transform)
    return converted / converted[2, 2]


def _resize_with_points(
    image: np.ndarray, points: np.ndarray, maximum_width: int = 900
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    scale = min(1.0, maximum_width / float(width))
    if scale < 1.0:
        resized = cv2.resize(
            image,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = image.copy()
    return resized, np.asarray(points, dtype=np.float32) * scale


def _render_alignment_matches(
    first_image: np.ndarray,
    second_image: np.ndarray,
    first_points: np.ndarray,
    second_points: np.ndarray,
    inlier_mask: np.ndarray,
    output_path: Path,
    *,
    method: str,
    raw_count: int,
    sift_failure: str | None,
) -> None:
    first_display, first_scaled = _resize_with_points(first_image, first_points)
    second_display, second_scaled = _resize_with_points(second_image, second_points)
    indices = np.flatnonzero(inlier_mask)
    if len(indices) == 0:
        indices = np.arange(len(first_scaled))
    if len(indices) > 80:
        positions = np.linspace(0, len(indices) - 1, 80, dtype=int)
        indices = indices[positions]
    first_keypoints = tuple(cv2.KeyPoint(float(x), float(y), 1.0) for x, y in first_scaled)
    second_keypoints = tuple(cv2.KeyPoint(float(x), float(y), 1.0) for x, y in second_scaled)
    matches = tuple(cv2.DMatch(int(index), int(index), 0.0) for index in indices)
    rendered = cv2.drawMatches(
        first_display,
        list(first_keypoints),
        second_display,
        list(second_keypoints),
        list(matches),
        None,
        matchColor=(40, 205, 40),
        singlePointColor=(180, 180, 180),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    header_height = 92 if sift_failure is None else 126
    header = np.full((header_height, rendered.shape[1], 3), 245, dtype=np.uint8)
    lines = [
        f"Detail alignment: {method}",
        f"raw correspondences={raw_count} RANSAC inliers={int(np.count_nonzero(inlier_mask))}",
    ]
    if sift_failure is not None:
        lines.append(sift_failure)
    for line_index, line in enumerate(lines):
        cv2.putText(
            header,
            line,
            (20, 31 + line_index * 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (24, 24, 24),
            2,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), np.vstack((header, rendered))):
        raise OSError(f"failed to write detail alignment diagnostic: {output_path}")


def align_detail_pair(
    first_path: Path,
    second_path: Path,
    output_path: Path,
    *,
    sift_config: SiftConfig = SiftConfig(maximum_width=1200),
    learned_config: ExternalLearnedConfig = ExternalLearnedConfig(),
    frontend_bundle: FrontendBundle | None = None,
) -> PairAlignmentDecision:
    """Align one detail pair with SIFT first and frozen ALIKED/LightGlue on failure."""

    first_image = cv2.imread(str(first_path), cv2.IMREAD_COLOR)
    second_image = cv2.imread(str(second_path), cv2.IMREAD_COLOR)
    if first_image is None or second_image is None:
        raise ValueError("detail alignment requires two readable images")

    first_sift = extract_sift(first_image, sift_config)
    second_sift = extract_sift(second_image, sift_config)
    sift_matches = match_sift(
        first_sift, second_sift, ratio_threshold=sift_config.ratio_threshold
    )
    sift_first_points = np.asarray(
        [first_sift.keypoints[match.queryIdx].pt for match in sift_matches],
        dtype=np.float32,
    ).reshape(-1, 2)
    sift_second_points = np.asarray(
        [second_sift.keypoints[match.trainIdx].pt for match in sift_matches],
        dtype=np.float32,
    ).reshape(-1, 2)
    sift_homography, sift_inliers = _estimate_homography(
        sift_first_points, sift_second_points
    )
    sift_inlier_count = int(np.count_nonzero(sift_inliers))
    sift_ratio = sift_inlier_count / len(sift_matches) if sift_matches else 0.0
    sift_accepted = (
        len(sift_matches) >= 30
        and sift_inlier_count >= 15
        and sift_ratio >= 0.35
        and sift_homography is not None
    )
    if sift_accepted:
        original_first = first_sift.scale.points_to_original(sift_first_points)
        original_second = second_sift.scale.points_to_original(sift_second_points)
        _render_alignment_matches(
            first_image,
            second_image,
            original_first,
            original_second,
            sift_inliers,
            output_path,
            method="sift",
            raw_count=len(sift_matches),
            sift_failure=None,
        )
        original_homography = _homography_to_original(
            sift_homography, first_sift.scale, second_sift.scale
        )
        return PairAlignmentDecision(
            method="sift",
            match_count=len(sift_matches),
            inlier_count=sift_inlier_count,
            homography=tuple(float(value) for value in original_homography.reshape(-1)),
            reason="SIFT met the bounded local patch-alignment thresholds",
        )

    sift_failure = (
        f"SIFT rejected: matches={len(sift_matches)} inliers={sift_inlier_count} "
        f"ratio={sift_ratio:.3f}"
    )
    bundle = frontend_bundle or load_frontend(learned_config)
    first_learned = extract_image_features(first_path, bundle, learned_config)
    second_learned = extract_image_features(second_path, bundle, learned_config)
    learned_matches = match_feature_pair(first_learned, second_learned, bundle)
    learned_first_points = np.asarray(
        first_learned.keypoints[learned_matches[:, 0]], dtype=np.float32
    ).reshape(-1, 2)
    learned_second_points = np.asarray(
        second_learned.keypoints[learned_matches[:, 1]], dtype=np.float32
    ).reshape(-1, 2)
    learned_homography, learned_inliers = _estimate_homography(
        learned_first_points, learned_second_points
    )
    _render_alignment_matches(
        first_image,
        second_image,
        learned_first_points,
        learned_second_points,
        learned_inliers,
        output_path,
        method="aliked_lightglue",
        raw_count=len(learned_matches),
        sift_failure=sift_failure,
    )
    return PairAlignmentDecision(
        method="aliked_lightglue",
        match_count=int(len(learned_matches)),
        inlier_count=int(np.count_nonzero(learned_inliers)),
        homography=(
            tuple(float(value) for value in learned_homography.reshape(-1))
            if learned_homography is not None
            else None
        ),
        reason=f"{sift_failure}; used frozen ALIKED-N16Rot + LightGlue fallback",
    )


def load_step13_registration(project_root: Path) -> dict[int, int | None]:
    """Cross-check the frozen Step 13 registration CSV against its best model."""

    import pycolmap

    root = project_root.resolve()
    selected = load_selected_manifest(root / SELECTION_MANIFEST)
    csv_path = root / STEP13_REGISTRATION
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != (
            "selected_index",
            "filename",
            "registered",
            "selected_frontend",
        ):
            raise ValueError("unexpected Step 13 registration columns")
        rows = list(reader)

    if len(rows) != len(selected):
        raise ValueError(
            f"expected {len(selected)} Step 13 registration rows, found {len(rows)}"
        )

    reconstruction = pycolmap.Reconstruction(str(root / STEP13_BEST_MODEL))
    image_id_by_name: dict[str, int] = {}
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        if image.name in image_id_by_name:
            raise ValueError(f"duplicate registered Step 13 filename: {image.name}")
        image_id_by_name[image.name] = int(image_id)

    registration: dict[int, int | None] = {}
    for selected_record, row in zip(selected, rows, strict=True):
        selected_index = int(row["selected_index"])
        filename = str(row["filename"])
        if selected_index != selected_record.index or filename != selected_record.filename:
            raise ValueError(
                "Step 13 registration rows do not preserve the selected-image manifest"
            )
        if row["registered"] not in {"0", "1"}:
            raise ValueError(f"invalid Step 13 registration flag: {row['registered']}")
        if row["selected_frontend"] != "aliked_lightglue":
            raise ValueError("unexpected Step 13 frontend provenance")

        model_image_id = image_id_by_name.get(filename)
        csv_registered = row["registered"] == "1"
        if csv_registered != (model_image_id is not None):
            raise ValueError(
                f"Step 13 CSV/model registration disagreement: {filename}"
            )
        registration[selected_index] = model_image_id

    if len(registration) != len(selected):
        raise ValueError("duplicate selected indices in Step 13 registration CSV")
    return registration


def select_canonical_geometry_views(
    views: Sequence[FinalReferenceView],
) -> tuple[dict[str, object], ...]:
    """Choose four registered references per required geometry category."""

    by_index = {view.selected_index: view for view in views}
    if len(by_index) != len(views):
        raise ValueError("reference views contain duplicate selected indices")

    selections: list[dict[str, object]] = []
    selected_indices: set[int] = set()
    for category, preferred_indices in PREFERRED_GEOMETRY_INDICES.items():
        selected_splits: set[str] = set()
        category_views = tuple(view for view in views if view.view_category == category)
        if len(category_views) < 4:
            raise ValueError(f"fewer than four reference views for {category}")

        for preferred_index in preferred_indices:
            preferred = by_index.get(preferred_index)
            if (
                preferred is not None
                and preferred.view_category == category
                and preferred.step13_registered
                and preferred.step13_image_id is not None
            ):
                chosen = preferred
                replacement_for = None
                quality = preferred.quality_condition.replace(" ", "_")
                reason = f"preferred_registered_{quality}_quality"
            else:
                candidates = tuple(
                    view
                    for view in category_views
                    if view.selected_index not in selected_indices
                    and view.selected_index not in preferred_indices
                    and view.step13_registered
                    and view.step13_image_id is not None
                )
                if not candidates:
                    raise ValueError(
                        f"no registered same-category replacement for {preferred_index}"
                    )
                chosen = min(
                    candidates,
                    key=lambda view: (
                        view.quality_condition != "normal",
                        view.split in selected_splits,
                        view.selected_index,
                    ),
                )
                replacement_for = preferred_index
                quality = chosen.quality_condition.replace(" ", "_")
                reason = f"replacement_registered_{quality}_quality_same_category"

            selected_indices.add(chosen.selected_index)
            selected_splits.add(chosen.split)
            selections.append(
                {
                    "selected_index": chosen.selected_index,
                    "filename": chosen.filename,
                    "view_category": chosen.view_category,
                    "quality_condition": chosen.quality_condition,
                    "split": chosen.split,
                    "step13_registered": True,
                    "step13_image_id": chosen.step13_image_id,
                    "selection_reason": reason,
                    "replacement_for": replacement_for,
                }
            )

    return tuple(selections)


def _largest_binary_component(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("landmark mask must be a 2D array")
    binary = np.where(mask > 0, 1, 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if component_count <= 1:
        raise ValueError("landmark mask has no foreground")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _row_extents(binary: np.ndarray, requested_y: int) -> tuple[int, int, int]:
    height = binary.shape[0]
    for distance in range(height):
        candidates = (requested_y - distance, requested_y + distance)
        for y in candidates:
            if 0 <= y < height:
                columns = np.flatnonzero(binary[y])
                if columns.size:
                    return int(columns[0]), int(columns[-1]), y
    raise ValueError("landmark mask has no usable rows")


def _fraction_row(top: int, bottom: int, fraction: float) -> int:
    return int(round(top + fraction * (bottom - top)))


def _landmark(
    name: str,
    x: float,
    y: float,
    image_size: tuple[int, int],
    *,
    confidence: str,
    annotation_method: str = "cv_seeded",
) -> Landmark:
    width, height = image_size
    return Landmark(
        name=name,
        x=float(x),
        y=float(y),
        normalized_x=float(x) / float(max(width - 1, 1)),
        normalized_y=float(y) / float(max(height - 1, 1)),
        confidence=confidence,
        annotation_method=annotation_method,
    )


def seed_landmarks_from_mask(
    whole_mask: np.ndarray,
    *,
    view_category: str,
) -> tuple[Landmark, ...]:
    """Propose deterministic component landmarks from one reviewed silhouette."""

    binary = _largest_binary_component(whole_mask)
    height, width = binary.shape
    foreground_y, _ = np.nonzero(binary)
    top, bottom = int(foreground_y.min()), int(foreground_y.max())
    if bottom - top < 20:
        raise ValueError("landmark mask foreground is vertically degenerate")

    widths = np.count_nonzero(binary, axis=1).astype(np.float64)
    smooth = cv2.GaussianBlur(widths.reshape(-1, 1), (1, 31), 0).reshape(-1)
    derivative = np.gradient(smooth)

    def extremum(low: float, high: float, *, maximum: bool) -> int:
        start = _fraction_row(top, bottom, low)
        stop = max(start + 1, _fraction_row(top, bottom, high))
        values = smooth[start : stop + 1]
        offset = int(np.argmax(values) if maximum else np.argmin(values))
        return start + offset

    neck_search_start = _fraction_row(top, bottom, 0.20)
    neck_search_stop = _fraction_row(top, bottom, 0.52)
    neck_base_y = neck_search_start + int(
        np.argmax(derivative[neck_search_start : neck_search_stop + 1])
    )
    lid_max_y = extremum(0.04, 0.18, maximum=True)
    lid_lower_y = max(lid_max_y, _fraction_row(top, bottom, 0.17))
    neck_top_y = max(lid_lower_y, _fraction_row(top, bottom, 0.19))
    neck_base_y = max(neck_top_y + 1, neck_base_y)
    bowl_rim_y = extremum(0.30, 0.72, maximum=True)
    globe_max_y = extremum(
        (neck_base_y - top) / (bottom - top),
        max((neck_base_y - top) / (bottom - top) + 0.02, (bowl_rim_y - top) / (bottom - top)),
        maximum=True,
    )
    globe_bottom_y = max(
        globe_max_y,
        min(_fraction_row(top, bottom, 0.70), _fraction_row(top, bottom, 0.76)),
    )
    bowl_bottom_y = max(globe_bottom_y, _fraction_row(top, bottom, 0.78))
    pedestal_waist_y = extremum(0.76, 0.91, maximum=False)
    pedestal_waist_y = max(bowl_bottom_y, pedestal_waist_y)
    foot_y = extremum(
        max(0.88, (pedestal_waist_y - top) / (bottom - top)),
        0.995,
        maximum=True,
    )
    foot_y = max(pedestal_waist_y, foot_y)

    row_levels = {
        "lid_max": lid_max_y,
        "lid_lower": lid_lower_y,
        "neck_top": neck_top_y,
        "neck_base": neck_base_y,
        "globe_max": globe_max_y,
        "bowl_rim": bowl_rim_y,
        "bowl_bottom_transition": bowl_bottom_y,
        "pedestal_waist": pedestal_waist_y,
        "foot": foot_y,
    }
    extents = {
        name: _row_extents(binary, row_y) for name, row_y in row_levels.items()
    }
    top_left, top_right, top_y = _row_extents(binary, top)
    _bottom_left, _bottom_right, bottom_y = _row_extents(binary, bottom)
    axis_x = float(
        np.median(
            [
                (columns[0] + columns[-1]) / 2.0
                for y in range(top, bottom + 1)
                if (columns := np.flatnonzero(binary[y])).size
            ]
        )
    )

    landmarks = [
        _landmark("axis_top", axis_x, top_y, (width, height), confidence="high"),
        _landmark(
            "axis_bottom",
            axis_x,
            bottom_y,
            (width, height),
            confidence="low" if bottom_y == height - 1 else "high",
        ),
        _landmark(
            "finial_top",
            (top_left + top_right) / 2.0,
            top_y,
            (width, height),
            confidence="high",
        ),
        _landmark(
            "finial_bottom",
            axis_x,
            _fraction_row(top, bottom, 0.065),
            (width, height),
            confidence="medium",
        ),
    ]
    for prefix in (
        "lid_max",
        "lid_lower",
        "neck_top",
        "neck_base",
        "globe_max",
        "bowl_rim",
        "bowl_bottom_transition",
        "pedestal_waist",
        "foot",
    ):
        left, right, y = extents[prefix]
        confidence = "high" if prefix in {"globe_max", "bowl_rim", "foot"} else "medium"
        landmarks.extend(
            (
                _landmark(
                    f"{prefix}_left", left, y, (width, height), confidence=confidence
                ),
                _landmark(
                    f"{prefix}_right", right, y, (width, height), confidence=confidence
                ),
            )
        )
    globe_top_left, globe_top_right, globe_top_y = _row_extents(binary, neck_base_y)
    globe_bottom_left, globe_bottom_right, globe_bottom_y = _row_extents(
        binary, globe_bottom_y
    )
    landmarks.extend(
        (
            _landmark(
                "globe_top_transition",
                (globe_top_left + globe_top_right) / 2.0,
                globe_top_y,
                (width, height),
                confidence="medium",
            ),
            _landmark(
                "globe_bottom_transition",
                (globe_bottom_left + globe_bottom_right) / 2.0,
                globe_bottom_y,
                (width, height),
                confidence="medium",
            ),
        )
    )

    by_name = {landmark.name: landmark for landmark in landmarks}
    ordered = [by_name[name] for name in BASE_LANDMARK_NAMES]
    if view_category == "top_down_rim":
        bowl_left = by_name["bowl_rim_left"]
        bowl_right = by_name["bowl_rim_right"]
        globe_left = by_name["globe_max_left"]
        globe_right = by_name["globe_max_right"]
        ordered.extend(
            (
                _landmark(
                    "bowl_rim_center",
                    (bowl_left.x + bowl_right.x) / 2.0,
                    bowl_left.y,
                    (width, height),
                    confidence="high",
                ),
                _landmark(
                    "bowl_rim_major_axis_a",
                    bowl_left.x,
                    bowl_left.y,
                    (width, height),
                    confidence="high",
                ),
                _landmark(
                    "bowl_rim_major_axis_b",
                    bowl_right.x,
                    bowl_right.y,
                    (width, height),
                    confidence="high",
                ),
                _landmark(
                    "globe_center",
                    (globe_left.x + globe_right.x) / 2.0,
                    globe_left.y,
                    (width, height),
                    confidence="medium",
                ),
            )
        )
    result = tuple(ordered)
    validate_landmark_set(result, (width, height), view_category=view_category)
    return result


def validate_landmark_set(
    landmarks: Sequence[Landmark],
    image_size: tuple[int, int],
    *,
    view_category: str,
) -> None:
    """Fail closed on missing, ambiguous, out-of-bounds, or impossible landmarks."""

    width, height = image_size
    if width < 2 or height < 2:
        raise ValueError("landmark image size is invalid")
    by_name = {landmark.name: landmark for landmark in landmarks}
    if len(by_name) != len(landmarks):
        raise ValueError("landmark names must be unique")
    required = set(BASE_LANDMARK_NAMES)
    if view_category == "top_down_rim":
        required.update(TOP_DOWN_LANDMARK_NAMES)
    missing = sorted(required - set(by_name))
    if missing:
        raise ValueError("missing required landmarks: " + ", ".join(missing))

    for landmark in landmarks:
        if not (0.0 <= landmark.x <= width - 1 and 0.0 <= landmark.y <= height - 1):
            raise ValueError(f"landmark outside image bounds: {landmark.name}")
        if not (
            abs(landmark.normalized_x - landmark.x / (width - 1)) <= 1e-9
            and abs(landmark.normalized_y - landmark.y / (height - 1)) <= 1e-9
        ):
            raise ValueError(f"landmark normalized coordinates disagree: {landmark.name}")
        if landmark.confidence not in {"low", "medium", "high"}:
            raise ValueError(f"invalid landmark confidence: {landmark.name}")
        if not landmark.annotation_method:
            raise ValueError(f"empty landmark annotation method: {landmark.name}")

    for prefix in (
        "lid_max",
        "lid_lower",
        "neck_top",
        "neck_base",
        "globe_max",
        "bowl_rim",
        "bowl_bottom_transition",
        "pedestal_waist",
        "foot",
    ):
        if by_name[f"{prefix}_left"].x >= by_name[f"{prefix}_right"].x:
            raise ValueError(f"landmark left/right ordering reversed: {prefix}")

    if by_name["axis_bottom"].y - by_name["axis_top"].y < 2.0:
        raise ValueError("landmark axis endpoints are degenerate")
    vertical = (
        by_name["finial_top"].y,
        by_name["finial_bottom"].y,
        by_name["lid_max_left"].y,
        by_name["lid_lower_left"].y,
        by_name["neck_top_left"].y,
        by_name["neck_base_left"].y,
        by_name["globe_top_transition"].y,
        by_name["globe_max_left"].y,
        by_name["globe_bottom_transition"].y,
        by_name["bowl_bottom_transition_left"].y,
        by_name["pedestal_waist_left"].y,
        by_name["foot_left"].y,
    )
    if any(second < first for first, second in zip(vertical, vertical[1:])):
        raise ValueError("landmark vertical component ordering is impossible")


def render_landmark_overlay(
    image_path: Path,
    landmarks: Sequence[Landmark],
    output_path: Path,
    *,
    maximum_width: int = 1600,
) -> Path:
    """Render source-pixel landmarks on a resized diagnostic copy."""

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unreadable landmark source image: {image_path}")
    height, width = image.shape[:2]
    scale = min(1.0, maximum_width / float(width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    palette = {
        "axis": (255, 255, 255),
        "finial": (255, 80, 80),
        "lid": (255, 180, 50),
        "neck": (50, 220, 255),
        "globe": (80, 255, 120),
        "bowl": (255, 80, 220),
        "pedestal": (180, 100, 255),
        "foot": (80, 180, 255),
    }
    for index, landmark in enumerate(landmarks):
        prefix = landmark.name.split("_", 1)[0]
        color = palette.get(prefix, (230, 230, 230))
        point = (int(round(landmark.x * scale)), int(round(landmark.y * scale)))
        cv2.circle(image, point, 6, color, -1, cv2.LINE_AA)
        label_y = point[1] - 9 if index % 2 == 0 else point[1] + 20
        cv2.putText(
            image,
            landmark.name,
            (point[0] + 8, max(18, min(image.shape[0] - 8, label_y))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"failed to write landmark overlay: {output_path}")
    return output_path


def split_component_masks(
    whole_mask: np.ndarray,
    landmarks: Sequence[Landmark],
) -> dict[str, np.ndarray]:
    """Split a reviewed whole-object mask into deterministic V2 annotation regions.

    These masks are geometry annotations constrained by the reviewed whole mask;
    they are not additional CNN predictions.  The two broad assembly masks form a
    disjoint partition.  Finer masks are allowed to overlap their broad parent.
    """

    binary = np.asarray(whole_mask) > 0
    if binary.ndim != 2 or not np.any(binary):
        raise ValueError("component source mask must contain 2D foreground")
    by_name = {landmark.name: landmark for landmark in landmarks}
    required = {
        "finial_top",
        "lid_lower_left",
        "neck_top_left",
        "neck_base_left",
        "globe_top_transition",
        "globe_bottom_transition",
        "bowl_rim_left",
        "axis_bottom",
    }
    missing = sorted(required - set(by_name))
    if missing:
        raise ValueError("component-mask landmarks missing: " + ", ".join(missing))

    height, _ = binary.shape
    globe_bottom = float(by_name["globe_bottom_transition"].y)
    bowl_rim = float(by_name["bowl_rim_left"].y)
    assembly_split = int(round((globe_bottom + bowl_rim) * 0.5))
    assembly_split = min(height - 1, max(1, assembly_split))
    rows = np.arange(height, dtype=np.int32)[:, None]

    def clipped_region(top: float, bottom: float) -> np.ndarray:
        low = min(height - 1, max(0, int(math.floor(top))))
        high = min(height - 1, max(low, int(math.ceil(bottom))))
        return np.where(binary & (rows >= low) & (rows <= high), 255, 0).astype(
            np.uint8
        )

    main_vessel = np.where(binary & (rows <= assembly_split), 255, 0).astype(np.uint8)
    receiving = np.where(binary & (rows > assembly_split), 255, 0).astype(np.uint8)
    lid_bottom = max(
        by_name["lid_lower_left"].y,
        by_name.get("lid_lower_right", by_name["lid_lower_left"]).y,
    )
    neck_top = min(
        by_name["neck_top_left"].y,
        by_name.get("neck_top_right", by_name["neck_top_left"]).y,
    )
    neck_bottom = max(
        by_name["neck_base_left"].y,
        by_name.get("neck_base_right", by_name["neck_base_left"]).y,
    )
    return {
        "main_vessel": main_vessel,
        "receiving_bowl_pedestal": receiving,
        "lid_finial": clipped_region(by_name["finial_top"].y, lid_bottom),
        "neck": clipped_region(neck_top, neck_bottom),
        "globe": clipped_region(
            by_name["globe_top_transition"].y,
            by_name["globe_bottom_transition"].y,
        ),
    }


def refine_geometry_mask(
    source_image: np.ndarray,
    reviewed_mask: np.ndarray,
    *,
    maximum_dimension: int = 768,
) -> np.ndarray:
    """Refine a coarse reviewed training mask into V2 geometry evidence.

    The immutable 36 reviewed masks remain the historical ML labels.  This helper
    uses each one only as a bounded GrabCut seed against the actual source photo so
    that V2 profile fitting does not silently treat a coarse training silhouette as
    precision geometry.  The result stays inside a narrow dilation of the reviewed
    mask and keeps the connected component with strongest overlap to its core.
    """

    image = np.asarray(source_image)
    mask = np.asarray(reviewed_mask)
    if image.ndim != 3 or image.shape[2] != 3 or mask.ndim != 2:
        raise ValueError("geometry-mask refinement expects BGR image and grayscale mask")
    if image.shape[:2] != mask.shape:
        raise ValueError("geometry-mask source and reviewed mask sizes disagree")
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    if np.count_nonzero(binary) < 64:
        raise ValueError("reviewed geometry-mask seed is empty or too small")

    scale = min(1.0, maximum_dimension / float(max(binary.shape)))
    target_size = (
        max(2, int(round(binary.shape[1] * scale))),
        max(2, int(round(binary.shape[0] * scale))),
    )
    small_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    small_binary = cv2.resize(binary, target_size, interpolation=cv2.INTER_NEAREST)

    foreground = small_binary > 0
    ys, xs = np.nonzero(foreground)
    bbox_extent = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    band = max(3, int(round(bbox_extent * 0.018)))
    core_radius = max(2, int(round(bbox_extent * 0.010)))
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1)
    )
    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * core_radius + 1, 2 * core_radius + 1)
    )
    allowed = cv2.dilate(small_binary, dilate_kernel) > 0
    eroded = cv2.erode(small_binary, erode_kernel) > 0

    # The reviewed CNN-training masks can contain connected table/background
    # regions.  Do not force every eroded seed pixel to definite foreground: the
    # real artifact is brass and supplies a strong chroma core, while the common
    # connected table/desk errors are nearly achromatic.  Keep low-chroma regions
    # as probable foreground so GrabCut can reject them from source-photo evidence.
    saturation = cv2.cvtColor(small_image, cv2.COLOR_BGR2HSV)[..., 1]
    foreground_saturation = saturation[foreground]
    adaptive_saturation = max(
        24.0,
        float(np.percentile(foreground_saturation, 35.0)) * 0.65,
    )
    core = eroded & (saturation >= adaptive_saturation)
    if np.count_nonzero(core) < 32:
        core = eroded
    if np.count_nonzero(core) < 32:
        core = foreground.copy()

    support_radius = max(4, int(round(bbox_extent * 0.045)))
    support_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * support_radius + 1, 2 * support_radius + 1)
    )
    chroma_support = cv2.dilate(core.astype(np.uint8), support_kernel) > 0
    probable_foreground = foreground & (
        (saturation >= adaptive_saturation) | chroma_support
    )

    grabcut_mask = np.full(small_binary.shape, cv2.GC_BGD, dtype=np.uint8)
    grabcut_mask[allowed] = cv2.GC_PR_BGD
    grabcut_mask[probable_foreground] = cv2.GC_PR_FGD
    grabcut_mask[core] = cv2.GC_FGD
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.setRNGSeed(4213)
    cv2.grabCut(
        small_image,
        grabcut_mask,
        None,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_MASK,
    )
    refined = np.isin(grabcut_mask, (cv2.GC_FGD, cv2.GC_PR_FGD)) & allowed

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        refined.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        raise ValueError("geometry-mask refinement produced no foreground component")
    best_label = None
    best_score = -1
    for label in range(1, count):
        component = labels == label
        overlap = int(np.count_nonzero(component & foreground))
        area = int(stats[label, cv2.CC_STAT_AREA])
        score = overlap * 4 + area
        if score > best_score:
            best_label = label
            best_score = score
    selected = np.where(labels == int(best_label), 255, 0).astype(np.uint8)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, close_kernel)

    # A few low-angle training masks contain a narrow table/reflection tail that
    # reaches the bottom image border below the actual brass foot.  Remove only
    # low-chroma foreground connected to that bottom border; legitimate cropped
    # brass remains protected by the chroma-supported core above.
    low_chroma = saturation < adaptive_saturation
    border_candidate = (selected > 0) & low_chroma
    border_count, border_labels, border_stats, _ = cv2.connectedComponentsWithStats(
        border_candidate.astype(np.uint8), connectivity=8
    )
    minimum_border_area = max(12, int(round(selected.size * 0.00025)))
    for label in range(1, border_count):
        if (
            int(border_stats[label, cv2.CC_STAT_AREA]) >= minimum_border_area
            and np.any(border_labels[-1, :] == label)
        ):
            selected[border_labels == label] = 0

    # Removing the achromatic bridge can leave tiny reflected fragments below the
    # foot.  Re-select the connected component with strongest overlap to the
    # definite brass core rather than choosing by raw area alone.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (selected > 0).astype(np.uint8), connectivity=8
    )
    if count <= 1:
        raise ValueError("geometry-mask border cleanup removed all foreground")
    best_label = max(
        range(1, count),
        key=lambda label: (
            int(np.count_nonzero((labels == label) & core)) * 4
            + int(stats[label, cv2.CC_STAT_AREA])
        ),
    )
    selected = np.where(labels == best_label, 255, 0).astype(np.uint8)
    selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, close_kernel)
    return cv2.resize(
        selected, (binary.shape[1], binary.shape[0]), interpolation=cv2.INTER_NEAREST
    )


def load_source_review(project_root: Path, view: FinalReferenceView) -> dict | None:
    """Load an independently reviewed annotation, bound to its exact source photo."""
    path = project_root / V2_RELATIVE_ROOT / "evidence/annotations/source_review.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = [row for row in payload["views"] if row["selected_index"] == view.selected_index]
    if len(records) != 1:
        raise ValueError(f"source review must identify view {view.selected_index} exactly once")
    record = records[0]
    if record["filename"] != view.filename or record["source_sha256"] != view.source_sha256:
        raise ValueError(f"source review hash/name mismatch: {view.filename}")
    return {**record, "display_size": payload["display_size"], "review_sha256": sha256_file(path)}


def load_reference_visual_review(
    project_root: Path,
    geometry_mask_report: Mapping[str, object],
    landmark_board_path: Path,
) -> dict[str, object] | None:
    """Load a visual review only when it is bound to the exact current boards."""

    root = project_root.resolve()
    review_path = root / V2_RELATIVE_ROOT / "reports" / "reference_visual_review.json"
    if not review_path.is_file():
        return None
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    if (
        payload.get("accepted") is not True
        or payload.get("geometry_mask_board_sha256")
        != str(geometry_mask_report["overlay_board_sha256"])
        or payload.get("landmark_board_sha256") != sha256_file(landmark_board_path)
    ):
        return None
    return {
        **payload,
        "path": review_path.relative_to(root).as_posix(),
        "sha256": sha256_file(review_path),
    }


def apply_reviewed_mask_patches(mask: np.ndarray, review: dict) -> np.ndarray:
    """Apply only photo-traced row intervals; never mutate the input ML/geometry mask."""
    output = mask.copy()
    height, width = output.shape
    display_width, display_height = review["display_size"]
    scale = np.asarray((width / display_width, height / display_height))
    for patch in review.get("mask_patches", []):
        polygon = np.asarray(patch["polygon"], dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            raise ValueError("invalid reviewed contour polygon")
        if np.any(polygon < 0) or np.any(polygon >= (display_width, display_height)):
            raise ValueError("reviewed contour polygon outside source")
        low, high = patch["rows"]
        if not 0 <= low <= high < display_height:
            raise ValueError("reviewed contour interval outside source")
        row_start = int(round(low * scale[1]))
        row_end = min(height, int(round((high + 1) * scale[1])))
        replacement = np.zeros_like(mask)
        cv2.fillPoly(replacement, [np.rint(polygon * scale).astype(np.int32)], 255)
        output[row_start:row_end] = replacement[row_start:row_end]
    return output


def reviewed_landmarks(review: dict, image_size: tuple[int, int], *, view_category: str) -> tuple[Landmark, ...]:
    """Convert reviewed visible feature coordinates, not fitted model points, to source pixels."""
    width, height = image_size
    scale = np.asarray((width / review["display_size"][0], height / review["display_size"][1]))
    points = review["points"]
    low_keys = set(points.get("low", []))
    records = []

    def add(name: str, xy, confidence="high"):
        key = name.removesuffix("_left").removesuffix("_right")
        if key in low_keys:
            confidence = "low"
        x, y = np.asarray(xy) * scale
        records.append(_landmark(name, x, y, image_size, confidence=confidence,
                                 annotation_method="cv_seeded_visually_reviewed"))

    # Image extrema support normalization/silhouette QA, not point triangulation.
    add("axis_top", points["top"], "low")
    add("axis_bottom", points["bottom"], "low")
    for name, key in (("finial_top", "top"), ("finial_bottom", "finial_bottom"),
                      ("globe_top_transition", "globe_top"), ("globe_bottom_transition", "globe_bottom")):
        add(name, points[key], "medium" if name.startswith("globe") else "high")
    pairs = {"lid_max": "lid", "lid_lower": "lid", "neck_top": "neck_top",
             "neck_base": "neck_base", "globe_max": "globe", "bowl_rim": "bowl_rim",
             "bowl_bottom_transition": "bowl_bottom", "pedestal_waist": "waist", "foot": "foot"}
    for name, key in pairs.items():
        value = points[key]
        add(name + "_left", value[:2])
        add(name + "_right", value[2:])
    if view_category == "top_down_rim":
        ring = points["bowl_rim"]
        add("bowl_rim_center", ((ring[0]+ring[2])/2, (ring[1]+ring[3])/2))
        add("bowl_rim_major_axis_a", ring[:2])
        add("bowl_rim_major_axis_b", ring[2:])
        globe = points["globe"]
        add("globe_center", ((globe[0]+globe[2])/2, (globe[1]+globe[3])/2), "low")
    by_name = {record.name: record for record in records}
    names = BASE_LANDMARK_NAMES + (TOP_DOWN_LANDMARK_NAMES if view_category == "top_down_rim" else ())
    return tuple(by_name[name] for name in names)


def build_geometry_mask_evidence(
    project_root: Path,
    views: Sequence[FinalReferenceView],
    canonical_views: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Persist source-photo-refined geometry masks without modifying ML labels."""

    root = project_root.resolve()
    v2_root = root / V2_RELATIVE_ROOT
    output_dir = ensure_under_v2_root(v2_root / "evidence" / "geometry_masks", v2_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_index = {view.selected_index: view for view in views}
    records: list[dict[str, object]] = []
    overlay_paths: list[Path] = []
    labels: list[str] = []
    for canonical in canonical_views:
        selected_index = int(canonical["selected_index"])
        view = by_index[selected_index]
        source_path = root / SELECTED_IMAGES / view.filename
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        reviewed = cv2.imread(str(view.reviewed_mask_path), cv2.IMREAD_GRAYSCALE)
        if source is None or reviewed is None:
            raise ValueError(f"unreadable geometry-mask source: {view.filename}")
        refined = refine_geometry_mask(source, reviewed)
        source_review = load_source_review(root, view)
        if source_review is not None:
            refined = apply_reviewed_mask_patches(refined, source_review)
        mask_path = ensure_under_v2_root(
            output_dir / f"{selected_index:03d}_geometry_mask.png", v2_root
        )
        if not cv2.imwrite(str(mask_path), refined):
            raise OSError(f"failed to write geometry mask: {mask_path}")

        scale = min(1.0, 900.0 / max(source.shape[:2]))
        display_size = (
            int(round(source.shape[1] * scale)),
            int(round(source.shape[0] * scale)),
        )
        display = cv2.resize(source, display_size, interpolation=cv2.INTER_AREA)
        reviewed_small = cv2.resize(reviewed, display_size, interpolation=cv2.INTER_NEAREST)
        refined_small = cv2.resize(refined, display_size, interpolation=cv2.INTER_NEAREST)
        reviewed_contours, _ = cv2.findContours(
            (reviewed_small > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        refined_contours, _ = cv2.findContours(
            (refined_small > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(display, reviewed_contours, -1, (0, 165, 255), 2)
        cv2.drawContours(display, refined_contours, -1, (80, 220, 80), 2)
        overlay_path = ensure_under_v2_root(
            output_dir / f"{selected_index:03d}_geometry_overlay.png", v2_root
        )
        if not cv2.imwrite(str(overlay_path), display):
            raise OSError(f"failed to write geometry-mask overlay: {overlay_path}")
        overlay_paths.append(overlay_path)
        labels.append(f"{selected_index:03d} | {view.view_category}")

        reviewed_area = int(np.count_nonzero(reviewed))
        refined_area = int(np.count_nonzero(refined))
        records.append(
            {
                "selected_index": selected_index,
                "filename": view.filename,
                "view_category": view.view_category,
                "annotation_method": "source_photo_grabcut_refined_from_reviewed_ml_mask",
                "source_review_sha256": source_review["review_sha256"] if source_review else None,
                "photo_traced_patch_count": len(source_review["mask_patches"]) if source_review else 0,
                "reviewed_ml_mask_path": view.reviewed_mask_path.relative_to(root).as_posix(),
                "reviewed_ml_mask_sha256": view.reviewed_mask_sha256,
                "geometry_mask_path": mask_path.relative_to(root).as_posix(),
                "geometry_mask_sha256": sha256_file(mask_path),
                "overlay_path": overlay_path.relative_to(root).as_posix(),
                "overlay_sha256": sha256_file(overlay_path),
                "reviewed_foreground_pixels": reviewed_area,
                "geometry_foreground_pixels": refined_area,
                "area_ratio_vs_reviewed": refined_area / max(reviewed_area, 1),
            }
        )
    board_path = ensure_under_v2_root(
        output_dir / "geometry_mask_overlays_board.png", v2_root
    )
    make_labeled_contact_sheet(
        tuple(overlay_paths), tuple(labels), board_path, cell_size=(360, 480), columns=4
    )
    return {
        "schema_version": 1,
        "annotation_method": "source_photo_grabcut_refined_from_reviewed_ml_mask",
        "historical_ml_masks_modified": False,
        "visual_review_status": "pending",
        "view_count": len(records),
        "views": records,
        "overlay_board_path": board_path.relative_to(root).as_posix(),
        "overlay_board_sha256": sha256_file(board_path),
    }


def build_component_mask_evidence(
    project_root: Path,
    views: Sequence[FinalReferenceView],
    landmark_report: dict[str, object],
) -> dict[str, object]:
    """Persist component annotations and overlays for all canonical views."""

    root = project_root.resolve()
    v2_root = root / V2_RELATIVE_ROOT
    output_dir = ensure_under_v2_root(v2_root / "evidence" / "component_masks", v2_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    views_by_index = {view.selected_index: view for view in views}
    records: list[dict[str, object]] = []
    for landmark_view in landmark_report["views"]:
        selected_index = int(landmark_view["selected_index"])
        view = views_by_index[selected_index]
        mask_path = root / str(
            landmark_view.get(
                "geometry_mask_path", view.reviewed_mask_path.relative_to(root).as_posix()
            )
        )
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"unreadable component source mask: {view.filename}")
        landmarks = tuple(Landmark(**record) for record in landmark_view["landmarks"])
        components = split_component_masks(mask, landmarks)
        component_records: dict[str, object] = {}
        overlay = np.zeros((*mask.shape, 3), dtype=np.uint8)
        palette = {
            "main_vessel": (55, 210, 255),
            "receiving_bowl_pedestal": (225, 80, 225),
            "lid_finial": (70, 120, 255),
            "neck": (255, 210, 70),
            "globe": (80, 230, 100),
        }
        for name, component in components.items():
            path = ensure_under_v2_root(
                output_dir / f"{selected_index:03d}_{name}.png", v2_root
            )
            if not cv2.imwrite(str(path), component):
                raise OSError(f"failed to write component mask: {path}")
            overlay[component > 0] = palette[name]
            component_records[name] = {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "foreground_pixels": int(np.count_nonzero(component)),
            }
        overlay_path = ensure_under_v2_root(
            output_dir / f"{selected_index:03d}_components.png", v2_root
        )
        source = cv2.imread(str(root / SELECTED_IMAGES / view.filename), cv2.IMREAD_COLOR)
        if source is None:
            raise ValueError(f"unreadable component overlay source: {view.filename}")
        blended = cv2.addWeighted(source, 0.62, overlay, 0.38, 0.0)
        if not cv2.imwrite(str(overlay_path), blended):
            raise OSError(f"failed to write component overlay: {overlay_path}")
        records.append(
            {
                "selected_index": selected_index,
                "filename": view.filename,
                "annotation_method": "reviewed_whole_mask_plus_cv_seeded_landmark_split",
                "components": component_records,
                "overlay_path": overlay_path.relative_to(root).as_posix(),
                "overlay_sha256": sha256_file(overlay_path),
            }
        )
    return {"views": records, "view_count": len(records)}


def build_ornament_inventory() -> dict[str, object]:
    """Return the source-traceable V2 motif inventory and chain decision."""

    return {
        "families": ORNAMENT_FAMILIES,
        "chain": {
            "supported": False,
            "decision": "omit_from_v2",
            "evidence_requirement": "two_independent_source_views",
        },
    }


def extract_ornament_crop(
    source_path: Path,
    bbox_xyxy: tuple[int, int, int, int],
    output_path: Path,
) -> dict[str, object]:
    """Extract one lossless derived crop while preserving source provenance."""

    with Image.open(source_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        left, top, right, bottom = (int(value) for value in bbox_xyxy)
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("ornament bbox must stay inside source image")
        crop = image.crop((left, top, right, bottom))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path, format="PNG", compress_level=9)
    return {
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "crop_path": str(output_path),
        "crop_sha256": sha256_file(output_path),
        "source_bbox": [left, top, right, bottom],
        "source_dimensions": [width, height],
        "crop_dimensions": [right - left, bottom - top],
    }


def build_ornament_evidence(project_root: Path) -> dict[str, object]:
    """Extract documented motif crops and persist the ornament manifest."""

    root = project_root.resolve()
    v2_root = root / V2_RELATIVE_ROOT
    crop_dir = ensure_under_v2_root(v2_root / "evidence" / "ornament_crops", v2_root)
    report_dir = ensure_under_v2_root(v2_root / "reports", v2_root)
    selected = load_selected_manifest(root / SELECTION_MANIFEST)
    selected_by_index = {record.index: record for record in selected}
    inventory = build_ornament_inventory()
    family_reports: list[dict[str, object]] = []
    for family in inventory["families"]:
        crop_index, normalized_box = ORNAMENT_CROP_BOXES[family.family_id]
        record = selected_by_index[crop_index]
        source_path = root / SELECTED_IMAGES / record.filename
        with Image.open(source_path) as opened:
            width, height = ImageOps.exif_transpose(opened).size
        left, top, right, bottom = normalized_box
        bbox = (
            int(round(left * width)),
            int(round(top * height)),
            int(round(right * width)),
            int(round(bottom * height)),
        )
        output_path = ensure_under_v2_root(
            crop_dir / f"{family.family_id}_{crop_index:03d}.png", v2_root
        )
        crop_report = extract_ornament_crop(source_path, bbox, output_path)
        crop_report["source_path"] = source_path.relative_to(root).as_posix()
        crop_report["crop_path"] = output_path.relative_to(root).as_posix()
        family_reports.append(
            {
                **asdict(family),
                "directly_observed": True,
                "hidden_repetition_inferred": family.repeat_mode.startswith("radial"),
                "anchor_crop": crop_report,
                "support_count": len(family.primary_view_indices),
            }
        )
    payload = {
        "schema_version": 1,
        "source_policy": "immutable_project_photographs_only",
        "families": family_reports,
        "chain": inventory["chain"],
        "accepted": True,
    }
    path = ensure_under_v2_root(report_dir / "ornament_manifest.json", v2_root)
    write_json_atomic(path, payload, v2_root)
    payload["path"] = path.relative_to(root).as_posix()
    payload["sha256"] = sha256_file(path)
    return payload


def build_landmark_evidence(
    project_root: Path,
    views: Sequence[FinalReferenceView],
    canonical_views: Sequence[dict[str, object]],
    *,
    geometry_mask_report: dict[str, object] | None = None,
    visual_review_status: str = "pending",
) -> dict[str, object]:
    """Seed, validate, render, and persist landmarks for all canonical views."""

    root = project_root.resolve()
    v2_root = root / V2_RELATIVE_ROOT
    annotation_dir = ensure_under_v2_root(
        v2_root / "evidence" / "annotations", v2_root
    )
    by_index = {view.selected_index: view for view in views}
    geometry_by_index = {
        int(record["selected_index"]): record
        for record in (geometry_mask_report or {}).get("views", [])
    }
    records: list[dict[str, object]] = []
    for canonical in canonical_views:
        selected_index = int(canonical["selected_index"])
        view = by_index[selected_index]
        geometry_record = geometry_by_index.get(selected_index)
        mask_path = (
            root / str(geometry_record["geometry_mask_path"])
            if geometry_record is not None
            else view.reviewed_mask_path
        )
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"unreadable landmark geometry mask: {view.filename}")
        height, width = mask.shape
        source_review = load_source_review(root, view)
        if source_review is not None:
            landmarks = reviewed_landmarks(source_review, (width, height), view_category=view.view_category)
        else:
            landmarks = seed_landmarks_from_mask(mask, view_category=view.view_category)
        validate_landmark_set(
            landmarks, (width, height), view_category=view.view_category
        )
        overlay_path = ensure_under_v2_root(
            annotation_dir / f"{selected_index:03d}_landmarks.png", v2_root
        )
        render_landmark_overlay(
            root / SELECTED_IMAGES / view.filename,
            landmarks,
            overlay_path,
        )
        records.append(
            {
                "selected_index": selected_index,
                "filename": view.filename,
                "view_category": view.view_category,
                "source_image_size": [width, height],
                "geometry_mask_path": mask_path.relative_to(root).as_posix(),
                "geometry_mask_sha256": sha256_file(mask_path),
                "geometry_mask_method": (
                    str(geometry_record["annotation_method"])
                    if geometry_record is not None
                    else "historical_reviewed_ml_mask_fallback"
                ),
                "overlay_path": overlay_path.relative_to(root).as_posix(),
                "overlay_sha256": sha256_file(overlay_path),
                "landmarks": [asdict(landmark) for landmark in landmarks],
                "source_review_sha256": source_review["review_sha256"] if source_review else None,
            }
        )

    payload = {
        "schema_version": 1,
        "coordinate_space": "source_image_pixels_and_normalized_0_1",
        "annotation_method": "cv_seeded",
        "visual_review_status": visual_review_status,
        "views": records,
    }
    output_path = ensure_under_v2_root(annotation_dir / "landmarks.json", v2_root)
    write_json_atomic(output_path, payload, v2_root)
    payload["path"] = output_path.relative_to(root).as_posix()
    payload["sha256"] = sha256_file(output_path)
    return payload


def load_reviewed_reference_views(project_root: Path) -> tuple[FinalReferenceView, ...]:
    """Load the 36 frozen reviewed views with fail-closed provenance checks."""

    root = project_root.resolve()
    selected = load_selected_manifest(root / SELECTION_MANIFEST)
    selected_by_index = {record.index: record for record in selected}
    reviewed = load_segmentation_manifest(
        root / REVIEWED_MANIFEST,
        selected,
        root / SELECTED_IMAGES,
    )
    masks = load_mask_manifest(
        root / MASK_MANIFEST,
        project_root=root,
        expected_count=len(selected),
    )
    masks_by_index = {record.selected_index: record for record in masks}
    registration = load_step13_registration(root)

    views: list[FinalReferenceView] = []
    for record in reviewed.records:
        selected_record = selected_by_index[record.selected_index]
        source_path = root / SELECTED_IMAGES / record.filename
        if sha256_file(source_path) != record.source_sha256:
            raise ValueError(f"reviewed source hash mismatch: {record.filename}")

        mask_record = masks_by_index.get(record.selected_index)
        if mask_record is None:
            raise ValueError(
                f"missing CNN/reconstruction masks for index {record.selected_index}"
            )
        if (
            mask_record.filename != record.filename
            or mask_record.source_width != selected_record.width
            or mask_record.source_height != selected_record.height
        ):
            raise ValueError(
                f"mask catalog disagrees with selected image: {record.filename}"
            )

        step13_image_id = registration[record.selected_index]
        views.append(
            FinalReferenceView(
                selected_index=record.selected_index,
                filename=record.filename,
                split=record.split,
                view_category=record.view_category,
                quality_condition=record.quality_condition,
                source_sha256=record.source_sha256,
                reviewed_mask_path=record.mask_path,
                reviewed_mask_sha256=record.mask_sha256,
                cnn_prediction_path=mask_record.raw_prediction_path,
                reconstruction_mask_path=mask_record.reconstruction_mask_path,
                step13_registered=step13_image_id is not None,
                step13_image_id=step13_image_id,
            )
        )

    return tuple(views)


def build_final_reference_evidence(project_root: Path) -> dict[str, object]:
    """Build the complete restartable Gate-A evidence package."""

    root = project_root.resolve()
    v2_root = root / V2_RELATIVE_ROOT
    report_dir = ensure_under_v2_root(v2_root / "reports", v2_root)
    match_dir = ensure_under_v2_root(v2_root / "evidence" / "cv_matches", v2_root)
    views = load_reviewed_reference_views(root)
    canonical = select_canonical_geometry_views(views)
    boards = build_reference_boards(root, views)
    v1_rejection = build_v1_rejection_board(root)
    geometry_masks = build_geometry_mask_evidence(root, views, canonical)
    landmarks = build_landmark_evidence(
        root,
        views,
        canonical,
        geometry_mask_report=geometry_masks,
        visual_review_status="pending_geometry_mask_and_landmark_review",
    )
    landmark_board_paths = tuple(
        root / str(record["overlay_path"]) for record in landmarks["views"]
    )
    landmark_board = ensure_under_v2_root(
        v2_root / "evidence" / "annotations" / "landmark_overlays_board.png",
        v2_root,
    )
    make_labeled_contact_sheet(
        landmark_board_paths,
        tuple(f"{int(record['selected_index']):03d}_landmarks" for record in landmarks["views"]),
        landmark_board,
        columns=4,
    )
    reference_visual_review = load_reference_visual_review(
        root, geometry_masks, landmark_board
    )
    if reference_visual_review is not None:
        geometry_masks["visual_review_status"] = "accepted_exact_board_hashes"
        landmarks["visual_review_status"] = "accepted_exact_board_hashes"
        landmark_output = root / str(landmarks["path"])
        landmark_payload = {
            key: value for key, value in landmarks.items() if key not in {"path", "sha256"}
        }
        write_json_atomic(landmark_output, landmark_payload, v2_root)
        landmarks["sha256"] = sha256_file(landmark_output)
    components = build_component_mask_evidence(root, views, landmarks)

    selected = load_selected_manifest(root / SELECTION_MANIFEST)
    selected_by_index = {record.index: record for record in selected}

    def source_path(index: int) -> Path:
        return root / SELECTED_IMAGES / selected_by_index[index].filename

    sift_pairs = (
        (3, 4, "normal_side"),
        (90, 91, "low_angle_pedestal"),
        (148, 149, "elevated_oblique"),
        (206, 207, "top_down_rim"),
    )
    sift_reports: list[dict[str, object]] = []
    for first_index, second_index, category in sift_pairs:
        output_path = ensure_under_v2_root(
            match_dir / f"sift_{category}_{first_index:03d}_{second_index:03d}.png",
            v2_root,
        )
        report = build_sift_pair_diagnostic(
            source_path(first_index), source_path(second_index), output_path
        )
        report.update(
            {
                "first_index": first_index,
                "second_index": second_index,
                "view_category": category,
                "first_path": source_path(first_index).relative_to(root).as_posix(),
                "second_path": source_path(second_index).relative_to(root).as_posix(),
                "output_path": output_path.relative_to(root).as_posix(),
                "output_sha256": sha256_file(output_path),
            }
        )
        sift_reports.append(report)

    detail_pairs = ((266, 267), (266, 268), (268, 278), (278, 288))
    detail_reports: list[dict[str, object]] = []
    for first_index, second_index in detail_pairs:
        output_path = ensure_under_v2_root(
            match_dir / f"detail_{first_index:03d}_{second_index:03d}.png", v2_root
        )
        decision = align_detail_pair(
            source_path(first_index), source_path(second_index), output_path
        )
        detail_reports.append(
            {
                **asdict(decision),
                "first_index": first_index,
                "second_index": second_index,
                "first_path": source_path(first_index).relative_to(root).as_posix(),
                "second_path": source_path(second_index).relative_to(root).as_posix(),
                "output_path": output_path.relative_to(root).as_posix(),
                "output_sha256": sha256_file(output_path),
            }
        )

    ornament = build_ornament_evidence(root)
    payload = {
        "schema_version": 1,
        "method": "cv_constrained_reference_reconstruction",
        "reviewed_view_count": len(views),
        "cnn_prediction_count": len(views),
        "step13_registered_reviewed_count": sum(view.step13_registered for view in views),
        "canonical_geometry_views": list(canonical),
        "reference_boards": boards,
        "v1_rejection": v1_rejection,
        "geometry_masks": geometry_masks,
        "landmarks": {
            "path": str(landmarks["path"]),
            "sha256": str(landmarks["sha256"]),
            "view_count": len(landmarks["views"]),
            "visual_review_status": landmarks["visual_review_status"],
            "board_path": landmark_board.relative_to(root).as_posix(),
            "board_sha256": sha256_file(landmark_board),
        },
        "component_masks": components,
        "reference_visual_review": reference_visual_review,
        "sift_ransac_diagnostics": sift_reports,
        "detail_alignment_diagnostics": detail_reports,
        "ornament_manifest": {
            "path": ornament["path"],
            "sha256": ornament["sha256"],
            "family_count": len(ornament["families"]),
            "chain": ornament["chain"],
        },
        "protected_inputs_modified": False,
        "accepted": (
            len(views) == 36
            and len(canonical) == 16
            and all(record["step13_registered"] for record in canonical)
            and len(sift_reports) == 4
            and len(detail_reports) == 4
            and components["view_count"] == 16
            and len(ornament["families"]) >= 12
            and reference_visual_review is not None
            and geometry_masks["visual_review_status"] == "accepted_exact_board_hashes"
            and landmarks["visual_review_status"] == "accepted_exact_board_hashes"
        ),
    }
    path = ensure_under_v2_root(report_dir / "final_reference_evidence.json", v2_root)
    write_json_atomic(path, payload, v2_root)
    payload["path"] = path.relative_to(root).as_posix()
    payload["sha256"] = sha256_file(path)
    return payload
