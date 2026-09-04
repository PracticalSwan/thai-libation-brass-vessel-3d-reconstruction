"""Deterministic train/validation workflow for the from-scratch SmallSegCNN."""

from __future__ import annotations

import csv
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from analysis_common import load_selected_manifest
from cnn_segmentation import (
    SmallSegCNN,
    binary_metrics,
    logits_to_binary,
    segmentation_loss,
    trainable_parameter_count,
)
from segmentation_data import (
    DEFAULT_INPUT_SIZE,
    SegmentationDataset,
    ValidatedSegmentationSet,
    load_segmentation_manifest,
)

ROOT = Path(__file__).resolve().parent
SELECTION_MANIFEST = ROOT / "preprocessing" / "reports" / "selection_manifest.csv"
SELECTED_IMAGES = ROOT / "preprocessing" / "pycolmap_input" / "images"
SEGMENTATION_MANIFEST = ROOT / "ml_dataset" / "manifest.csv"
CHECKPOINT_PATH = ROOT / "analysis" / "ml" / "checkpoints" / "best_small_seg_cnn.pt"
HISTORY_PATH = ROOT / "analysis" / "reports" / "cnn_training_history.csv"
CURVES_PATH = ROOT / "analysis" / "previews" / "presentation" / "ml_01_training_curves.png"


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 4213
    learning_rate: float = 1e-3
    batch_size: int = 8
    max_epochs: int = 60
    patience: int = 10
    threshold: float = 0.5
    input_height: int = DEFAULT_INPUT_SIZE[0]
    input_width: int = DEFAULT_INPUT_SIZE[1]


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    epochs_completed: int
    best_val_dice: float
    best_val_iou: float
    runtime_seconds: float
    checkpoint_path: Path
    history_path: Path
    curves_path: Path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def runtime_environment(device: torch.device) -> dict[str, object]:
    try:
        import torchvision

        torchvision_version = torchvision.__version__
    except Exception:
        torchvision_version = "unavailable"
    result: dict[str, object] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torchvision": torchvision_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        result["gpu"] = torch.cuda.get_device_name(device)
        result["gpu_memory_bytes"] = int(torch.cuda.get_device_properties(device).total_memory)
    return result


def build_loaders(
    dataset: ValidatedSegmentationSet,
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    input_size = (config.input_height, config.input_width)
    train_dataset = SegmentationDataset(
        dataset.split("train"), dataset.images_dir, training=True, input_size=input_size
    )
    val_dataset = SegmentationDataset(
        dataset.split("val"), dataset.images_dir, training=False, input_size=input_size
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def run_epoch(
    model: SmallSegCNN,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: Adam | None,
    threshold: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    dices: list[float] = []
    ious: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = segmentation_loss(logits, masks)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite segmentation loss")
            if training:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

            predictions = logits_to_binary(logits.detach(), threshold)
            for i in range(predictions.shape[0]):
                metrics = binary_metrics(predictions[i], masks[i])
                dices.append(metrics.dice)
                ious.append(metrics.iou)
                precisions.append(metrics.precision)
                recalls.append(metrics.recall)

    return {
        "loss": mean(losses),
        "dice": mean(dices),
        "iou": mean(ious),
        "precision": mean(precisions),
        "recall": mean(recalls),
    }


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _write_curves(path: Path, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train loss")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE + Dice loss")
    axes[0].set_title("Training loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(epochs, [row["val_dice"] for row in history], label="Validation Dice")
    axes[1].plot(epochs, [row["val_iou"] for row in history], label="Validation IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Validation segmentation")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle("SmallSegCNN training from random initialization")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def train_model(
    dataset: ValidatedSegmentationSet,
    *,
    config: TrainingConfig = TrainingConfig(),
    checkpoint_path: Path = CHECKPOINT_PATH,
    history_path: Path = HISTORY_PATH,
    curves_path: Path = CURVES_PATH,
    device: torch.device | None = None,
) -> TrainingResult:
    seed_everything(config.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallSegCNN().to(device)
    parameter_count = trainable_parameter_count(model)
    if parameter_count >= 2_000_000:
        raise RuntimeError(f"SmallSegCNN is too large: {parameter_count} trainable parameters")
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    train_loader, val_loader = build_loaders(dataset, config)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_dice = -1.0
    best_val_iou = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, device, optimizer=optimizer, threshold=config.threshold
        )
        val_metrics = run_epoch(
            model, val_loader, device, optimizer=None, threshold=config.threshold
        )
        row = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_dice={row['val_dice']:.4f} "
            f"val_iou={row['val_iou']:.4f}"
        )

        improved = val_metrics["dice"] > best_val_dice + 1e-6
        if improved:
            best_val_dice = val_metrics["dice"]
            best_val_iou = val_metrics["iou"]
            best_epoch = epoch
            stale_epochs = 0
            checkpoint = {
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "model_name": "SmallSegCNN",
                "random_initialization": True,
                "pretrained_weights": False,
                "parameter_count": parameter_count,
                "config": asdict(config),
                "manifest_sha256": dataset.manifest_sha256,
                "train_indices": [record.selected_index for record in dataset.split("train")],
                "val_indices": [record.selected_index for record in dataset.split("val")],
                "test_indices": [record.selected_index for record in dataset.split("test")],
                "best_epoch": best_epoch,
                "best_val_dice": best_val_dice,
                "best_val_iou": best_val_iou,
                "environment": runtime_environment(device),
            }
            torch.save(checkpoint, checkpoint_path)
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    runtime_seconds = time.perf_counter() - started
    _write_history(history_path, history)
    _write_curves(curves_path, history)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["epochs_completed"] = len(history)
    checkpoint["runtime_seconds"] = runtime_seconds
    torch.save(checkpoint, checkpoint_path)
    return TrainingResult(
        best_epoch=best_epoch,
        epochs_completed=len(history),
        best_val_dice=best_val_dice,
        best_val_iou=best_val_iou,
        runtime_seconds=runtime_seconds,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        curves_path=curves_path,
    )


def load_real_dataset(*, verify_source_hashes: bool = True) -> ValidatedSegmentationSet:
    selected_records = load_selected_manifest(SELECTION_MANIFEST)
    return load_segmentation_manifest(
        SEGMENTATION_MANIFEST,
        selected_records,
        SELECTED_IMAGES,
        verify_source_hashes=verify_source_hashes,
    )


def main() -> None:
    dataset = load_real_dataset(verify_source_hashes=True)
    config = TrainingConfig()
    result = train_model(dataset, config=config)
    print(
        f"best_epoch={result.best_epoch} epochs_completed={result.epochs_completed} "
        f"best_val_dice={result.best_val_dice:.4f} "
        f"best_val_iou={result.best_val_iou:.4f} runtime_seconds={result.runtime_seconds:.1f}"
    )


if __name__ == "__main__":
    main()
