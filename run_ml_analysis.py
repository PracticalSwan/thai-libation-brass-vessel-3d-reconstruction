"""Frozen held-out evaluation and Step 8 SIFT feature-mask analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, median

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cnn_segmentation import SmallSegCNN, binary_metrics, logits_to_binary
from ml_feature_analysis import analyze_image_features
from segmentation_data import validate_binary_mask
from train_cnn_segmentation import CHECKPOINT_PATH, load_real_dataset

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "IMG20260826122949"
SELECTED_IMAGES = ROOT / "preprocessing" / "pycolmap_input" / "images"
PREDICTIONS_DIR = ROOT / "analysis" / "ml" / "predictions"
TEST_METRICS_PATH = ROOT / "analysis" / "reports" / "cnn_test_metrics.csv"
SUMMARY_PATH = ROOT / "analysis" / "reports" / "cnn_summary.json"
FEATURE_COUNTS_PATH = ROOT / "analysis" / "reports" / "masked_feature_counts.csv"
PRESENTATION_DIR = ROOT / "analysis" / "previews" / "presentation"
EXAMPLES_PATH = PRESENTATION_DIR / "ml_02_segmentation_examples.png"
CONTACT_SHEET_PATH = PRESENTATION_DIR / "ml_03_test_mask_contact_sheet.png"
MASKED_FEATURES_PATH = PRESENTATION_DIR / "ml_04_masked_features.png"
FEATURE_SUMMARY_PATH = PRESENTATION_DIR / "ml_05_feature_mask_summary.png"
ML_SUMMARY_PATH = PRESENTATION_DIR / "ml_06_summary.png"


def restore_prediction_mask(mask: np.ndarray, source_size: tuple[int, int]) -> np.ndarray:
    """Restore a binary model-space mask to source (width, height)."""
    if mask.ndim != 2:
        raise ValueError("prediction mask must be 2D")
    values = set(int(value) for value in np.unique(mask))
    if not values.issubset({0, 255}):
        raise ValueError("prediction mask must contain only 0 and 255")
    width, height = source_size
    if width < 1 or height < 1:
        raise ValueError("source size must be positive")
    restored = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.where(restored > 0, 255, 0).astype(np.uint8)


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def validate_prediction_output_path(path: Path, raw_dir: Path, selected_dir: Path) -> None:
    if _is_within(path, raw_dir) or _is_within(path, selected_dir):
        raise ValueError("prediction output cannot be written inside source directories")


def verify_checkpoint_provenance(
    checkpoint: dict[str, object], manifest_sha256: str, test_indices: list[int]
) -> None:
    if checkpoint.get("model_name") != "SmallSegCNN":
        raise ValueError("unexpected checkpoint model")
    if checkpoint.get("random_initialization") is not True:
        raise ValueError("checkpoint does not prove random initialization")
    if checkpoint.get("pretrained_weights") is not False:
        raise ValueError("pretrained checkpoint is not allowed")
    if checkpoint.get("manifest_sha256") != manifest_sha256:
        raise ValueError("checkpoint manifest hash does not match frozen labels")
    stored_test = [int(value) for value in checkpoint.get("test_indices", [])]
    if stored_test != list(test_indices):
        raise ValueError("checkpoint test split does not match frozen held-out split")
    config = checkpoint.get("config")
    if not isinstance(config, dict) or float(config.get("threshold", -1.0)) != 0.5:
        raise ValueError("checkpoint threshold must remain frozen at 0.5")


def _prepare_image_tensor(image: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    height, width = size
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0)


def _status_for_metrics(
    dice: float, iou: float, precision: float, recall: float
) -> str:
    if precision < 0.90 and recall >= 0.90:
        return "background_false_positive"
    if recall < 0.90 and precision >= 0.90:
        return "partial_vessel_mask"
    if dice >= 0.95 and iou >= 0.90:
        return "ok"
    if dice >= 0.90:
        return "minor_boundary_error"
    if dice >= 0.80:
        return "visible_boundary_error"
    return "mixed_segmentation_error"


def _rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    shown = image.copy()
    tint = np.zeros_like(shown)
    tint[:, :, 1] = 220
    inside = mask > 0
    shown[inside] = (0.62 * shown[inside] + 0.38 * tint[inside]).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(shown, contours, -1, (0, 0, 240), 4, cv2.LINE_AA)
    return shown


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _figure_examples(results: list[dict[str, object]]) -> None:
    chosen = [row for row in results if int(row["selected_index"]) in {165, 255}]
    fig, axes = plt.subplots(len(chosen), 4, figsize=(13, 7.4))
    if len(chosen) == 1:
        axes = np.asarray([axes])
    for row_index, row in enumerate(chosen):
        image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(row["ground_truth_path"]), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(str(ROOT / str(row["prediction_path"])), cv2.IMREAD_GRAYSCALE)
        panels = [
            (_rgb(image), "Original"),
            (gt, "Ground truth"),
            (pred, "CNN prediction"),
            (_rgb(_overlay(image, pred)), "Prediction overlay"),
        ]
        for col, (panel, title) in enumerate(panels):
            axes[row_index, col].imshow(panel, cmap="gray" if panel.ndim == 2 else None)
            axes[row_index, col].axis("off")
            axes[row_index, col].set_title(title, fontsize=10)
        axes[row_index, 0].set_ylabel(
            f"Index {row['selected_index']}\nDice {float(row['dice']):.3f} | IoU {float(row['iou']):.3f}",
            fontsize=10,
        )
    fig.suptitle("Held-out SmallSegCNN segmentation examples", fontsize=14)
    fig.tight_layout()
    fig.savefig(EXAMPLES_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _figure_contact_sheet(results: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 9.5))
    for axis, row in zip(axes.flat, results, strict=True):
        image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        pred = cv2.imread(str(ROOT / str(row["prediction_path"])), cv2.IMREAD_GRAYSCALE)
        axis.imshow(_rgb(_overlay(image, pred)))
        axis.axis("off")
        axis.set_title(
            f"{int(row['selected_index']):03d} | {row['status']}\nDice {float(row['dice']):.3f} | IoU {float(row['iou']):.3f}",
            fontsize=9,
        )
    fig.suptitle("All six held-out CNN predictions", fontsize=14)
    fig.tight_layout()
    fig.savefig(CONTACT_SHEET_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _draw_sampled_keypoints(
    image: np.ndarray,
    keypoints: tuple[cv2.KeyPoint, ...],
    indices: tuple[int, ...] | None = None,
    *,
    color: tuple[int, int, int] = (255, 0, 255),
) -> np.ndarray:
    canvas = image.copy()
    selected = tuple(range(len(keypoints))) if indices is None else indices
    if len(selected) > 450:
        step = max(1, len(selected) // 450)
        selected = selected[::step][:450]
    for index in selected:
        x, y = keypoints[index].pt
        cv2.circle(canvas, (round(x), round(y)), 4, color, 2, cv2.LINE_AA)
    return canvas


def _figure_masked_features(
    feature_rows: list[dict[str, object]],
    feature_cache: dict[int, tuple[object, object]],
    results_by_index: dict[int, dict[str, object]],
) -> None:
    sample_index = 165 if 165 in feature_cache else int(feature_rows[0]["selected_index"])
    features, classification = feature_cache[sample_index]
    result = results_by_index[sample_index]
    pred = cv2.imread(str(ROOT / str(result["prediction_path"])), cv2.IMREAD_GRAYSCALE)
    analysis_image = features.analysis_image
    aw, ah = features.scale.analysis_size
    pred_analysis = cv2.resize(pred, (aw, ah), interpolation=cv2.INTER_NEAREST)
    overlay = _overlay(analysis_image, pred_analysis)
    all_points = _draw_sampled_keypoints(
        analysis_image, features.keypoints, color=(255, 255, 0)
    )
    inside = _draw_sampled_keypoints(
        analysis_image,
        features.keypoints,
        classification.vessel_indices,
        color=(255, 0, 255),
    )
    outside = analysis_image.copy()
    bg_indices = classification.background_indices
    if len(bg_indices) > 450:
        step = max(1, len(bg_indices) // 450)
        bg_indices = bg_indices[::step][:450]
    for idx in bg_indices:
        x, y = features.keypoints[idx].pt
        cv2.circle(outside, (round(x), round(y)), 4, (0, 0, 255), 2, cv2.LINE_AA)
    panels = [
        (_rgb(all_points), f"All SIFT keypoints ({classification.total_keypoints})"),
        (_rgb(overlay), "CNN-predicted vessel mask"),
        (_rgb(inside), f"Inside predicted vessel ({classification.vessel_keypoints})"),
        (_rgb(outside), f"Background ({classification.background_keypoints})"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.4))
    for axis, (panel, title) in zip(axes, panels, strict=True):
        axis.imshow(panel)
        axis.axis("off")
        axis.set_title(title, fontsize=10)
    fig.suptitle(
        f"Step 8 feature-mask analysis | held-out index {sample_index} | Dice {float(result['dice']):.3f}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(MASKED_FEATURES_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _figure_feature_summary(feature_rows: list[dict[str, object]]) -> None:
    labels = [f"{int(row['selected_index']):03d}" for row in feature_rows]
    x = np.arange(len(labels))
    vessel = np.asarray([int(row["vessel_keypoints"]) for row in feature_rows])
    background = np.asarray([int(row["background_keypoints"]) for row in feature_rows])
    dice = np.asarray([float(row["segmentation_dice"]) for row in feature_rows])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    axes[0].bar(x, vessel, label="Vessel keypoints")
    axes[0].bar(x, background, bottom=vessel, label="Background keypoints")
    axes[0].set_xticks(x, labels)
    axes[0].set_xlabel("Held-out selected index")
    axes[0].set_ylabel("SIFT keypoints")
    axes[0].set_title("Predicted vessel vs background features")
    axes[0].legend(fontsize=8)
    fractions = np.asarray([float(row["vessel_feature_fraction"]) for row in feature_rows])
    axes[1].bar(x, fractions, label="Vessel feature fraction")
    axes[1].plot(x, dice, marker="o", label="Segmentation Dice")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xlabel("Held-out selected index")
    axes[1].set_title("Feature fraction with mask quality")
    axes[1].legend(fontsize=8)
    fig.suptitle("SIFT distribution using CNN-predicted masks", fontsize=14)
    fig.text(
        0.5,
        0.015,
        "Index 072 retains a visible background false positive; its feature fraction is reported, not hidden, and should be read beside mask quality.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))
    fig.savefig(FEATURE_SUMMARY_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _figure_summary(summary: dict[str, object]) -> None:
    test = summary["test_aggregate"]
    feature = summary["feature_aggregate"]
    fig, ax = plt.subplots(figsize=(13.5, 5.0))
    ax.axis("off")
    boxes = [
        (0.07, "36 reviewed masks\n24 train / 6 val / 6 test"),
        (0.27, "SmallSegCNN\n487,297 parameters\nrandom initialization"),
        (
            0.47,
            f"Held-out prediction\nmean Dice {test['mean_dice']:.3f}\nmean IoU {test['mean_iou']:.3f}",
        ),
        (0.67, "Existing Step 6 SIFT\nexact extraction/scale contract"),
        (
            0.87,
            f"Predicted-mask feature split\nmean vessel fraction {feature['mean_vessel_feature_fraction']:.3f}",
        ),
    ]
    for x, text in boxes:
        ax.text(
            x,
            0.52,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.55", facecolor="white", edgecolor="black"),
        )
    for start, end in [(0.14, 0.20), (0.34, 0.40), (0.54, 0.60), (0.74, 0.80)]:
        ax.annotate(
            "",
            xy=(end, 0.52),
            xytext=(start, 0.52),
            xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", lw=1.7),
        )
    ax.text(
        0.5,
        0.88,
        "Custom CNN segmentation + SIFT feature-mask analysis",
        transform=ax.transAxes,
        ha="center",
        fontsize=16,
        weight="bold",
    )
    ax.text(
        0.5,
        0.12,
        "Index 072 retains a background false positive (Dice 0.894). Feature counts are descriptive only; no reconstruction-improvement claim and no pyCOLMAP run.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(ML_SUMMARY_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_analysis() -> dict[str, object]:
    dataset = load_real_dataset(verify_source_hashes=True)
    test_records = dataset.split("test")
    test_indices = [record.selected_index for record in test_records]
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    verify_checkpoint_provenance(checkpoint, dataset.manifest_sha256, test_indices)

    config = checkpoint["config"]
    input_size = (int(config["input_height"]), int(config["input_width"]))
    threshold = float(config["threshold"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallSegCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    feature_cache: dict[int, tuple[object, object]] = {}

    with torch.no_grad():
        for record in test_records:
            image_path = dataset.images_dir / record.filename
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"unreadable held-out source image: {record.filename}")
            gt = validate_binary_mask(
                record.mask_path, (record.source_width, record.source_height)
            )
            tensor = _prepare_image_tensor(image, input_size).to(device)
            logits = model(tensor)
            small_binary = (
                logits_to_binary(logits, threshold)[0, 0].cpu().numpy().astype(np.uint8)
                * 255
            )
            prediction = restore_prediction_mask(
                small_binary, (record.source_width, record.source_height)
            )
            output_path = (
                PREDICTIONS_DIR
                / f"{record.selected_index:03d}_{Path(record.filename).stem}_pred.png"
            )
            validate_prediction_output_path(output_path, RAW_DIR, SELECTED_IMAGES)
            if not cv2.imwrite(str(output_path), prediction):
                raise RuntimeError(f"failed to write prediction: {output_path}")

            metric = binary_metrics(
                torch.from_numpy((prediction > 0).astype(np.uint8)),
                torch.from_numpy((gt > 0).astype(np.uint8)),
            )
            status = _status_for_metrics(
                metric.dice, metric.iou, metric.precision, metric.recall
            )
            row = {
                "selected_index": record.selected_index,
                "filename": record.filename,
                "view_category": record.view_category,
                "quality_condition": record.quality_condition,
                "dice": metric.dice,
                "iou": metric.iou,
                "precision": metric.precision,
                "recall": metric.recall,
                "pixel_accuracy": metric.pixel_accuracy,
                "foreground_fraction": metric.foreground_fraction,
                "status": status,
                "prediction_path": output_path.relative_to(ROOT).as_posix(),
                "image_path": image_path,
                "ground_truth_path": record.mask_path,
            }
            results.append(row)

            features, feature_result = analyze_image_features(image, prediction)
            feature_cache[record.selected_index] = (features, feature_result)
            feature_rows.append(
                {
                    "selected_index": record.selected_index,
                    "filename": record.filename,
                    "segmentation_dice": metric.dice,
                    "segmentation_iou": metric.iou,
                    "total_sift_keypoints": feature_result.total_keypoints,
                    "vessel_keypoints": feature_result.vessel_keypoints,
                    "background_keypoints": feature_result.background_keypoints,
                    "vessel_feature_fraction": feature_result.vessel_feature_fraction,
                    "background_feature_fraction": feature_result.background_feature_fraction,
                    "mask_foreground_fraction": feature_result.mask_foreground_fraction,
                    "segmentation_status": status,
                }
            )

    csv_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"image_path", "ground_truth_path"}
        }
        for row in results
    ]
    _write_csv(TEST_METRICS_PATH, csv_rows)
    _write_csv(FEATURE_COUNTS_PATH, feature_rows)

    dices = [float(row["dice"]) for row in results]
    ious = [float(row["iou"]) for row in results]
    precisions = [float(row["precision"]) for row in results]
    recalls = [float(row["recall"]) for row in results]
    vessel_fractions = [
        float(row["vessel_feature_fraction"]) for row in feature_rows
    ]
    summary: dict[str, object] = {
        "model": {
            "name": checkpoint["model_name"],
            "parameter_count": checkpoint["parameter_count"],
            "random_initialization": checkpoint["random_initialization"],
            "pretrained_weights": checkpoint["pretrained_weights"],
            "manifest_sha256": checkpoint["manifest_sha256"],
            "best_epoch": checkpoint["best_epoch"],
            "best_val_dice": checkpoint["best_val_dice"],
            "best_val_iou": checkpoint["best_val_iou"],
            "epochs_completed": checkpoint["epochs_completed"],
            "runtime_seconds": checkpoint["runtime_seconds"],
            "training_environment": checkpoint["environment"],
            "training_config": checkpoint["config"],
        },
        "test_indices": test_indices,
        "test_aggregate": {
            "mean_dice": mean(dices),
            "median_dice": median(dices),
            "mean_iou": mean(ious),
            "median_iou": median(ious),
            "mean_precision": mean(precisions),
            "mean_recall": mean(recalls),
        },
        "feature_aggregate": {
            "mean_vessel_feature_fraction": mean(vessel_fractions),
            "mean_background_feature_fraction": 1.0 - mean(vessel_fractions),
            "total_sift_keypoints": sum(
                int(row["total_sift_keypoints"]) for row in feature_rows
            ),
            "total_vessel_keypoints": sum(
                int(row["vessel_keypoints"]) for row in feature_rows
            ),
            "total_background_keypoints": sum(
                int(row["background_keypoints"]) for row in feature_rows
            ),
        },
        "test_results": csv_rows,
        "feature_results": feature_rows,
        "interpretation_boundary": (
            "SIFT counts are descriptive only; no reconstruction experiment was run."
        ),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    results_by_index = {int(row["selected_index"]): row for row in results}
    _figure_examples(results)
    _figure_contact_sheet(results)
    _figure_masked_features(feature_rows, feature_cache, results_by_index)
    _figure_feature_summary(feature_rows)
    _figure_summary(summary)
    return summary


def main() -> None:
    summary = run_analysis()
    aggregate = summary["test_aggregate"]
    print(
        f"held_out=6 mean_dice={aggregate['mean_dice']:.4f} "
        f"mean_iou={aggregate['mean_iou']:.4f}"
    )


if __name__ == "__main__":
    main()
