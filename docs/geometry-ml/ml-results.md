# Steps 7 and 8 CNN Segmentation and Feature-Mask Results

Updated: 2026-09-05

## Outcome

Steps 7+8 are implemented and measured on the real project images. The final workflow:

```text
36 reviewed vessel masks
-> SmallSegCNN trained from random initialization
-> validation-only checkpoint selection
-> one frozen held-out test evaluation
-> source-size CNN-predicted masks
-> existing Step 6 SIFT extraction
-> keypoints classified inside/outside the predicted mask
```

No pretrained segmentation model, pretrained backbone, SAM checkpoint, transfer learning, external segmentation API, pyCOLMAP run, reconstruction, meshing, texturing, or Blender operation was used in this phase.

## Dataset and split

Dataset details are in `docs/geometry-ml/cnn-dataset.md`.

- initial/final labeled total: **36**;
- train: **24**;
- validation: **6**;
- held-out test: **6**;
- no training-only expansion was needed;
- label manifest SHA-256: `9925bccf367221472e2301d7c360bd7ea4f5f947981d81b5da22f71fe5b02e0f`;
- test indices: **72, 142, 165, 200, 255, 288**.

The six held-out labels were not used for architecture selection, augmentation tuning, learning-rate tuning, threshold tuning, early stopping, or checkpoint selection.

## Model

`SmallSegCNN` is a project-defined compact U-Net-like binary semantic-segmentation CNN:

```text
Encoder 1:   3 -> 16
Encoder 2:  16 -> 32
Encoder 3:  32 -> 64
Bottleneck: 64 -> 128
Decoder:    128 -> 64 -> 32 -> 16
Skip connections at all three encoder/decoder scales
Output:     1 logit channel
```

Actual trainable parameters: **487,297**.

The model is instantiated from PyTorch default random initialization. The checkpoint records:

```text
random_initialization = true
pretrained_weights    = false
```

## Runtime and training configuration

Measured training environment:

| Item | Measured value |
|---|---|
| Python | 3.14.2 |
| PyTorch | 2.13.0+cu130 |
| torchvision | 0.28.0+cu130 |
| CUDA available | yes |
| CUDA runtime | 13.0 |
| Device | cuda |
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU |
| GPU memory | 8,518,041,600 bytes |

Fixed training configuration:

| Setting | Value |
|---|---:|
| Input geometry | 384 x 288 (H x W) |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 8 |
| Maximum epochs | 60 |
| Early-stopping patience | 10 validation epochs |
| Seed | 4213 |
| Threshold | 0.5 |
| Loss | BCEWithLogitsLoss + soft Dice loss |
| Model selection | highest validation Dice |

Training finished after **49 epochs** because the validation-Dice best at epoch 39 was not exceeded during the next 10 epochs.

- runtime: **332.112 s**;
- best epoch: **39**;
- best validation Dice: **0.968066**;
- best validation IoU: **0.938252**.

The optional data expansion was not used because the original baseline already exceeded the plan targets of Dice 0.85 and IoU 0.75.

## Held-out test results

The model and threshold were frozen before the following six images were evaluated.

| Index | View | Dice | IoU | Precision | Recall | Visual status |
|---:|---|---:|---:|---:|---:|---|
| 72 | normal side | 0.8942 | 0.8087 | 0.8318 | 0.9668 | `background_false_positive` |
| 142 | low angle / pedestal | 0.9617 | 0.9262 | 0.9516 | 0.9720 | `ok` |
| 165 | elevated oblique | 0.9701 | 0.9420 | 0.9725 | 0.9677 | `ok` |
| 200 | elevated oblique | 0.9416 | 0.8896 | 0.9065 | 0.9794 | `minor_boundary_error` |
| 255 | top-down / rim | 0.9651 | 0.9325 | 0.9433 | 0.9879 | `ok` |
| 288 | oblique detail | 0.9825 | 0.9656 | 0.9796 | 0.9854 | `ok` |

Aggregate held-out metrics:

```text
Mean Dice:       0.952521
Median Dice:     0.963377
Mean IoU:        0.910745
Median IoU:      0.929347
Mean precision:  0.930884
Mean recall:     0.976529
```

### Visible failure analysis

All six predictions were inspected after the frozen evaluation.

The important weak case is **index 72**. Its high recall hides a substantial false positive: the CNN labels a large portion of the yellow classroom wall/background above and beside the vessel as foreground. This prediction is retained unchanged in the test evidence and is labeled `background_false_positive`; it was not repaired and the model was not tuned after seeing it.

Index **200** has smaller excess/misaligned boundary regions and is retained as `minor_boundary_error`. The other four predictions align closely enough with their reviewed masks to be reported as `ok`, while still containing normal small boundary differences.

The held-out metrics are also affected by the ground-truth annotation limitation described in `cnn-dataset.md`: some narrow rear-side brass slivers were conservatively excluded from a few labels. Test masks were not changed after evaluation.

## Step 8 SIFT feature-mask analysis

Step 8 directly reuses:

```python
geometry_detection.extract_sift
```

No second SIFT implementation was created. The existing Step 6 analysis size and explicit analysis-to-original scale metadata are used to map each keypoint to the source-size CNN-predicted mask.

Primary results use **CNN-predicted masks**, not ground truth.

| Index | Dice | Total SIFT | Vessel keypoints | Background keypoints | Vessel feature fraction | Predicted-mask foreground fraction |
|---:|---:|---:|---:|---:|---:|---:|
| 72 | 0.8942 | 5,824 | 5,483 | 341 | 0.9414 | 0.3042 |
| 142 | 0.9617 | 4,359 | 4,098 | 261 | 0.9401 | 0.2988 |
| 165 | 0.9701 | 4,653 | 4,375 | 278 | 0.9403 | 0.2243 |
| 200 | 0.9416 | 4,704 | 4,414 | 290 | 0.9384 | 0.2101 |
| 255 | 0.9651 | 1,233 | 1,182 | 51 | 0.9586 | 0.1586 |
| 288 | 0.9825 | 7,900 | 7,879 | 21 | 0.9973 | 0.4702 |

Across the six images:

```text
Total SIFT keypoints:       28,673
Inside predicted vessel:    27,431
Background:                   1,242
Mean vessel feature fraction across images: 0.952693
```

These counts describe where the Step 6 SIFT detector finds features relative to the CNN mask. They do **not** prove that masking improves Structure from Motion or reconstruction. In particular, image 72's background false positive can also cause background features to be misclassified as vessel features, so segmentation quality must be read beside the feature statistics.

## Evidence files

Training and evaluation:

- `analysis/reports/cnn_training_history.csv`;
- `analysis/reports/cnn_test_metrics.csv`;
- `analysis/reports/cnn_summary.json`;
- local final checkpoint: `analysis/ml/checkpoints/best_small_seg_cnn.pt`;
- source-size predictions: `analysis/ml/predictions/`.

Step 8:

- `analysis/reports/masked_feature_counts.csv`.

Presentation figures:

- `analysis/previews/presentation/ml_01_training_curves.png`;
- `analysis/previews/presentation/ml_02_segmentation_examples.png`;
- `analysis/previews/presentation/ml_03_test_mask_contact_sheet.png`;
- `analysis/previews/presentation/ml_04_masked_features.png`;
- `analysis/previews/presentation/ml_05_feature_mask_summary.png`;
- `analysis/previews/presentation/ml_06_summary.png`.

All six ML presentation figures were visually inspected. The contact sheet retains the weak image-72 prediction rather than hiding it.

## Verification

Final ML-focused verification:

```text
13 passed
```

Complete project test suite:

```text
66 passed
```

Changed Python modules and tests also completed `python -m py_compile` successfully.

Final source-integrity verification:

```text
Raw photographs:       297 / 297, 0 mismatches, 0 unexpected files
Selected PREPROCESSED: 288 / 288 verified against selection_manifest.csv
```

## Limits and next boundary

- The training dataset is deliberately small and consists of views of one physical vessel in one capture environment.
- The annotation workflow is OpenCV-assisted and visually reviewed rather than exhaustive sub-pixel hand tracing.
- The yellow classroom wall remains a demonstrated failure mode for the CNN on held-out index 72.
- A high vessel-feature fraction is descriptive and may be inflated by segmentation false positives.
- No claim is made that predicted masking improves reconstruction quality.
- **Stop here.** pyCOLMAP, camera pose estimation, triangulation, sparse/dense reconstruction, meshing, texturing, and Blender remain separate future work requiring explicit authorization.
