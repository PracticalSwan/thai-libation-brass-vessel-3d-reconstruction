from pathlib import Path

from analysis_common import load_selected_manifest
from segmentation_data import load_segmentation_manifest
from train_cnn_segmentation import TrainingConfig, build_loaders

ROOT = Path(__file__).resolve().parents[1]


def test_training_loaders_exclude_held_out_test_records():
    selected = load_selected_manifest(ROOT / "preprocessing/reports/selection_manifest.csv")
    dataset = load_segmentation_manifest(
        ROOT / "ml_dataset/manifest.csv",
        selected,
        ROOT / "preprocessing/pycolmap_input/images",
        verify_source_hashes=False,
    )
    train_loader, val_loader = build_loaders(dataset, TrainingConfig(batch_size=8))
    train_indices = {record.selected_index for record in train_loader.dataset.records}
    val_indices = {record.selected_index for record in val_loader.dataset.records}
    test_indices = {record.selected_index for record in dataset.split("test")}
    assert len(train_indices) == 24
    assert len(val_indices) == 6
    assert not train_indices.intersection(test_indices)
    assert not val_indices.intersection(test_indices)
    assert train_loader.dataset.training is True
    assert val_loader.dataset.training is False
