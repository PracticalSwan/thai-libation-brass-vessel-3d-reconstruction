# Steps 7 and 8 CNN Segmentation and Feature-Mask Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Updated: 2026-09-01

**Status:** Architecture revised and approved for planning only. No CNN code, labels, training, dependency changes, masks, or Step 8 outputs are authorized by this document update itself.

**Goal:** Train a small binary semantic-segmentation CNN from scratch to identify visible Thai brass vessel pixels, evaluate it on a leakage-controlled held-out test set, then reuse the completed Step 6 SIFT interface to measure vessel-versus-background feature distributions from the CNN-predicted masks.

**Architecture:** Use a compact U-Net-like encoder-decoder built entirely from project-defined convolutional blocks with random initialization and no pretrained weights. Build a small manually labeled dataset that references the existing verified selected JPEGs without copying or modifying them, split labels by separated viewpoints rather than random neighboring frames, train only on the training split, select the model using validation Dice/IoU, evaluate the locked model once on the held-out test split, then feed predicted masks into the existing Step 6 SIFT coordinate/scale contract.

**Tech Stack:** Python, PyTorch, torchvision where useful for deterministic image transforms, OpenCV, NumPy, CSV/JSON, matplotlib/OpenCV for figures, pytest. No pretrained segmentation model, no SAM checkpoint, and no model-assisted labeling is required.

**Spec:** `docs/superpowers/specs/2026-08-27-geometry-ml-integration-design.md`

## Global Constraints

- `IMG20260826122949/` remains immutable.
- `preprocessing/pycolmap_input/images/` remains the verified 288-image PREPROCESSED source set and must never be modified.
- Step 6 is complete. Reuse `analysis_common.py` for verified selected-image access and `geometry_detection.extract_sift` for SIFT. Do not duplicate those pipelines.
- The segmentation model must be trained from random initialization. No pretrained backbone, transfer learning, SAM, foundation-model checkpoint, or external segmentation API is allowed in the planned baseline.
- The task is **binary semantic segmentation**: visible brass vessel pixels are foreground; background, holes/openings through which background is visible, hands, table, and unrelated objects are background.
- Source photographs are never resized or rewritten on disk. Training/inference may create in-memory resized tensors and derived mask files only.
- Use a manually annotated initial dataset of **36 images: 24 train, 6 validation, 6 test**. If validation evidence shows a real coverage gap, the **training split may expand from 24 to at most 36 images** by adding up to 12 new training-only labels, giving at most 48 labeled images total. Validation/test membership must not change after the split is locked.
- Avoid sequence leakage. Do not randomly split adjacent capture frames. Labeled indices should be distributed across middle, low, elevated, top-down, and detail viewpoints, with train/validation/test examples separated in the capture sequence where practical.
- The held-out test set is not used for architecture choice, augmentation choice, threshold tuning, early stopping, or hyperparameter selection.
- Preferred tensor geometry is `(height=384, width=288)`, matching the source photographs' approximately 4:3 portrait aspect ratio without square-image distortion. If runtime evidence requires a smaller shape, preserve the same aspect ratio and document the change before training.
- Masks used for training resize with nearest-neighbor interpolation only. Image and mask geometric augmentation must remain exactly synchronized.
- Use deterministic seed `4213` wherever the library permits deterministic control.
- Step 8 reports feature distribution only. It must not claim that segmentation improves 3D reconstruction unless a later reconstruction experiment actually tests that hypothesis.
- No task in this plan may invoke pyCOLMAP, COLMAP, sparse/dense reconstruction, meshing, texturing, or Blender.
- Planning filenames below are preferred boundaries, not permission for unrelated refactors. Preserve the smallest coherent structure when implementation begins.

---

## Preferred future file structure

```text
ml_dataset/
  manifest.csv                         labeled-image index, filename, split, mask path/hash
  masks/                               manually created source-size binary PNG masks

segmentation_data.py                  mask/manifest validation, split loading, transforms
cnn_segmentation.py                   SmallSegCNN model, loss, metrics, prediction helpers
train_cnn_segmentation.py             deterministic training/validation/checkpoint workflow
ml_feature_analysis.py                Step 8 SIFT-inside/outside-mask measurements
run_ml_analysis.py                    held-out evaluation, reports, presentation figures

tests/
  test_segmentation_data.py
  test_cnn_segmentation.py
  test_train_cnn_segmentation.py
  test_ml_feature_analysis.py
  test_run_ml_analysis.py

analysis/
  ml/
    checkpoints/                       local generated model checkpoint; untracked by default
    predictions/                       final derived source-size binary masks
  reports/
  previews/
    presentation/
```

The implementation may combine files if that is demonstrably clearer, but these responsibilities should remain separate: data/splits, model math, training, feature-mask analysis, and orchestration/presentation.

---

### Task 1: Freeze the labeled segmentation dataset and leakage-safe split

**Files:**
- Create: `ml_dataset/manifest.csv`
- Create: `ml_dataset/masks/*.png`
- Create: `segmentation_data.py`
- Test: `tests/test_segmentation_data.py`
- Create after the split is locked: `docs/geometry-ml/cnn-dataset.md`

**Interfaces:**
- Consumes: verified selected-image records from `analysis_common.py`.
- Produces: validated image/mask pairs with fixed `train`, `val`, or `test` membership.
- Preferred API: `load_segmentation_manifest(path, selected_records) -> list[SegmentationRecord]`.
- `SegmentationRecord` should contain at least selected index, filename, split, mask path, source size, source SHA-256, and mask SHA-256.

- [ ] **Step 1: Select 36 annotation candidates before drawing masks**

Choose the candidates from the verified 288-image sequence so the labeled set spans the real capture conditions:

```text
middle/side views
low-angle/base views
elevated views
top-down/rim views
oblique/detail views
stronger and weaker reflections/background contrast
```

Use the existing representative indices as anchors, but do not force a visually unsuitable frame only to preserve an old list. Preferred held-out test anchors include `165` and `255` because they connect directly to the completed geometry demonstration. Freeze the exact list in `ml_dataset/manifest.csv` before training.

- [ ] **Step 2: Lock a 24/6/6 group-aware split**

Rules:

```text
24 train
6 validation
6 test
```

Do not use random image-level splitting across near-neighbor frames. Test and validation examples should be capture-sequence-separated from training examples where practical. Record the reason/view category for each selected label so coverage can be reviewed before training.

- [ ] **Step 3: Manually annotate source-resolution binary masks**

Annotation rule:

```text
255 = visible brass vessel surface
0   = background or visible opening/hole/background through the vessel
```

Do not use the future CNN prediction to create its own ground truth. A normal polygon/brush annotation tool is acceptable. Record the annotation tool and mask convention in `docs/geometry-ml/cnn-dataset.md`.

- [ ] **Step 4: Write failing data-contract tests**

Tests must reject:

- missing labeled indices;
- a labeled filename that disagrees with `selection_manifest.csv`;
- duplicated indices;
- invalid split names;
- any split count other than 24/6/6 for the initial dataset;
- missing/unreadable masks;
- mask dimensions that differ from the source photograph;
- mask values other than `0` and `255`;
- source or mask hash mismatches;
- the same index appearing in multiple splits.

- [ ] **Step 5: Implement only the manifest/mask validation boundary**

No training logic belongs in this task. The loader should return validated immutable records and derived paths without modifying image or mask bytes.

- [ ] **Step 6: Verify the dataset contract**

Run the focused data tests and a real manifest/mask verification. Confirm 36/36 labeled source images still match the existing selected-image manifest.

- [ ] **Step 7: Freeze the test set**

After the split passes review, treat the six test labels as evaluation-only. Later validation failure may add **training-only** examples, but must not move or replace test images because of model performance.

---

### Task 2: Define the from-scratch SmallSegCNN architecture

**Files:**
- Create: `cnn_segmentation.py`
- Test: `tests/test_cnn_segmentation.py`

**Interfaces:**
- Consumes: normalized image tensor `[N, 3, 384, 288]`.
- Produces: one-channel segmentation logits `[N, 1, 384, 288]`.
- Preferred model name: `SmallSegCNN`.
- No pretrained weights or external backbone constructor is permitted.

- [ ] **Step 1: Write model-shape and initialization tests**

Verify:

```text
input:  N x 3 x 384 x 288
output: N x 1 x 384 x 288 logits
```

Also verify the model contains project-defined convolution/upsampling blocks only and has no code path that downloads or loads pretrained weights.

- [ ] **Step 2: Implement the preferred compact encoder-decoder**

Recommended baseline:

```text
Encoder 1: 3 -> 16 channels, two 3x3 Conv + ReLU, MaxPool
Encoder 2: 16 -> 32 channels, two 3x3 Conv + ReLU, MaxPool
Encoder 3: 32 -> 64 channels, two 3x3 Conv + ReLU, MaxPool
Bottleneck: 64 -> 128 channels, two 3x3 Conv + ReLU, optional small Dropout2d
Decoder 3: upsample, concatenate encoder-3 skip, convs -> 64
Decoder 2: upsample, concatenate encoder-2 skip, convs -> 32
Decoder 1: upsample, concatenate encoder-1 skip, convs -> 16
Output: 1x1 Conv -> 1 logit channel
```

Use either bilinear upsampling plus convolution or transposed convolution, whichever stays simpler and avoids shape ambiguity. Skip connections are retained because vessel boundaries/rims are important and they are straightforward to explain in a computer-vision presentation.

- [ ] **Step 3: Keep model size bounded**

Target fewer than approximately 2 million trainable parameters. Record the actual parameter count rather than claiming a guessed value. Do not make the network deeper unless validation evidence shows a clear underfitting problem.

- [ ] **Step 4: Define segmentation loss and metrics**

Training loss:

```text
BCEWithLogitsLoss + soft Dice loss
```

Primary evaluation metrics:

```text
Dice coefficient
IoU / Jaccard index
foreground precision
foreground recall
```

Pixel accuracy may be reported but is secondary because background pixels can dominate it.

- [ ] **Step 5: Verify synthetic forward/backward behavior**

Use tiny deterministic tensors to confirm finite logits, finite loss, a backward pass, and metric behavior for perfect, empty, and mismatched masks.

---

### Task 3: Build deterministic training and validation

**Files:**
- Modify/Create: `segmentation_data.py`
- Create: `train_cnn_segmentation.py`
- Test: `tests/test_train_cnn_segmentation.py`
- Generated locally: `analysis/ml/checkpoints/best_small_seg_cnn.pt`
- Generated: `analysis/reports/cnn_training_history.csv`
- Generated: `analysis/previews/presentation/ml_01_training_curves.png`

**Interfaces:**
- Consumes: the frozen train/validation records and `SmallSegCNN`.
- Produces: best-validation checkpoint, training history, exact run configuration, and training curves.

- [ ] **Step 1: Implement leakage-safe transforms**

Preferred input tensor size: `384 x 288` `(H x W)`.

Training-only augmentation may include:

```text
horizontal flip p=0.5
small rotation, approximately +/-5 degrees
small translation/scale
mild brightness/contrast variation, approximately +/-15%
```

Rules:

- geometric transforms apply identically to image and mask;
- photometric transforms apply only to the image;
- mask interpolation is nearest-neighbor;
- no vertical flip;
- no arbitrary perspective warp;
- no generative augmentation or synthetic reflection removal.

Validation and test use deterministic resize/normalization only.

- [ ] **Step 2: Use a simple fixed baseline training configuration**

Preferred first run:

```text
optimizer: Adam
learning rate: 1e-3
batch size: 8, reduced only if measured GPU memory requires it
maximum epochs: 60
early stopping patience: 10 validation epochs
prediction threshold: 0.5
seed: 4213
model selection: highest validation Dice, with validation IoU as supporting metric
```

Do not touch the test set during this process.

- [ ] **Step 3: Write miniature training-loop tests**

Use a tiny synthetic dataset to verify:

- train and validation loaders stay separate;
- only training applies random augmentation;
- loss decreases or remains finite over a tiny deterministic smoke run;
- checkpoint selection uses validation metrics, not test metrics;
- early stopping state is deterministic under the fixed seed;
- run configuration and split-manifest hash are written with the checkpoint report.

- [ ] **Step 4: Run the real baseline training once**

Record actual hardware/device, PyTorch version, parameter count, epochs completed, best epoch, best validation Dice/IoU, runtime, and batch size.

- [ ] **Step 5: Apply one bounded data-expansion decision if needed**

Validation target for a strong coursework result:

```text
mean Dice >= 0.85
mean IoU  >= 0.75
```

These are targets, not numbers to fabricate. If validation quality is clearly weak or a specific viewpoint category fails, add up to **12 new training-only masks** from the failing categories and retrain the same baseline. Do not modify validation/test membership and do not increase network complexity at the same time as adding data, because that would make the cause of improvement unclear.

- [ ] **Step 6: Freeze the final model before test evaluation**

Once model/data decisions are complete from training+validation evidence, lock the checkpoint and threshold. No more architecture, augmentation, or hyperparameter changes after viewing held-out test metrics.

---

### Task 4: Evaluate held-out CNN segmentation and create presentation evidence

**Files:**
- Modify: `cnn_segmentation.py` as needed for inference helpers only
- Create/Modify: `run_ml_analysis.py`
- Test: `tests/test_run_ml_analysis.py`
- Generated: `analysis/ml/predictions/*.png`
- Generated: `analysis/reports/cnn_test_metrics.csv`
- Generated: `analysis/reports/cnn_summary.json`
- Generated: `analysis/previews/presentation/ml_02_segmentation_examples.png`
- Generated: `analysis/previews/presentation/ml_03_test_mask_contact_sheet.png`

**Interfaces:**
- Consumes: frozen checkpoint and six held-out test records.
- Produces: source-size binary predicted masks and held-out segmentation metrics.

- [ ] **Step 1: Write inference/output-contract tests**

Verify:

- model input is derived from, but never overwrites, the verified selected JPEG;
- prediction is thresholded to binary `uint8 {0,255}`;
- predicted mask is restored to exact source width/height using nearest-neighbor interpolation;
- output writes cannot occur inside raw or selected-image directories;
- every reported test metric corresponds to the locked test split.

- [ ] **Step 2: Evaluate the held-out test set once**

For all six test images report per-image and aggregate:

```text
Dice
IoU
precision
recall
foreground area fraction
```

Include mean and median Dice/IoU. Do not hide a weak image from the aggregate.

- [ ] **Step 3: Classify visible failures honestly**

Useful failure labels include:

```text
missed_rim_or_neck
background_false_positive
reflection_boundary_error
opening_filled_in
partial_vessel_mask
```

A weak prediction remains part of the test evidence. Do not manually edit a CNN prediction and then report it as model output.

- [ ] **Step 4: Generate professor-facing segmentation figures**

`ml_02_segmentation_examples.png` should show at least two held-out examples, preferably including the continuity images 165 and/or 255 when they are in the frozen test split:

```text
Original | Ground-truth mask | CNN prediction | Prediction overlay
```

Include test Dice/IoU for each shown example.

`ml_03_test_mask_contact_sheet.png` should show all six test predictions with concise index/status/Dice labels.

- [ ] **Step 5: Visually inspect every test prediction**

Check the actual vessel rim, neck, body, base, top opening, and background boundary. The written failure label must agree with what is visible.

---

### Task 5: Step 8 SIFT feature-mask analysis

**Files:**
- Create: `ml_feature_analysis.py`
- Test: `tests/test_ml_feature_analysis.py`
- Generated: `analysis/reports/masked_feature_counts.csv`
- Generated: `analysis/previews/presentation/ml_04_masked_features.png`
- Generated: `analysis/previews/presentation/ml_05_feature_mask_summary.png`

**Interfaces:**
- Consumes: `geometry_detection.extract_sift` and source-size CNN predicted masks from the held-out test set.
- Preferred API: `classify_keypoints_by_mask(features: SiftFeatures, full_resolution_mask: np.ndarray) -> FeatureMaskResult`.
- Produces: total, vessel-inside, and background keypoint counts plus aligned keypoint indices and fractions.

- [ ] **Step 1: Write coordinate-alignment tests before integration**

Synthetic features should prove that Step 6 analysis-pixel keypoints map to the correct mask pixels through explicit size metadata rather than guessed scaling.

Test:

- known inside/outside points;
- border points;
- source-size mismatch;
- all-background/all-foreground masks;
- descriptor/keypoint count alignment.

- [ ] **Step 2: Align predicted masks to Step 6 SIFT analysis space**

Use nearest-neighbor resizing of the **derived binary mask only**. Never resize or rewrite the source JPEG for Step 8.

- [ ] **Step 3: Compute feature-distribution measurements**

For each held-out test image record:

```text
index
filename
segmentation Dice/IoU
total SIFT keypoints
vessel keypoints
background keypoints
vessel feature fraction
background feature fraction
mask foreground fraction
segmentation status
```

Counts are descriptive. A high vessel-feature fraction does not by itself prove better reconstruction.

- [ ] **Step 4: Generate `ml_04_masked_features.png`**

Use a held-out example and show:

```text
all SIFT keypoints
CNN mask overlay
vessel-inside keypoints
background keypoints + counts
```

Use visibly distinct markers while keeping the source image readable.

- [ ] **Step 5: Generate `ml_05_feature_mask_summary.png`**

Across the six held-out test images, show total versus vessel-inside keypoint counts and vessel/background fractions. Keep segmentation quality visible beside feature statistics so a poor mask is not presented as trustworthy measurement.

- [ ] **Step 6: Visually cross-check feature classification**

Sample visible keypoints near the rim, silhouette, inner opening, pedestal, and background. Confirm the mask classification matches the overlay.

---

### Task 6: Integrate the CNN and feature-mask demonstration

**Files:**
- Modify: `run_ml_analysis.py`
- Test: `tests/test_run_ml_analysis.py`
- Generated: `analysis/previews/presentation/ml_06_summary.png`
- Create after real run: `docs/geometry-ml/ml-results.md`
- Modify after real run: `README.md`
- Modify after real run: `CHANGELOG.md`
- Modify after real run: `docs/memory-bank/active-context.md`
- Modify after real run: `docs/memory-bank/progress.md`

**Interfaces:**
- Orchestrator should verify source/label/checkpoint provenance before final output creation.
- It should consume the frozen model and held-out test set, then invoke Step 8 using the already generated predictions.

- [ ] **Step 1: Write orchestration tests**

Verify:

- Step 6 interfaces exist before Step 8 starts;
- test split and checkpoint provenance are verified before evaluation;
- no training occurs inside the evaluation orchestrator;
- failed/weak segmentations remain in reports;
- source/output path separation is enforced;
- completion does not depend on pyCOLMAP artifacts.

- [ ] **Step 2: Generate `ml_06_summary.png`**

Professor-ready flow:

```text
manual ground-truth masks
→ SmallSegCNN trained from scratch
→ held-out CNN vessel prediction
→ Dice / IoU evaluation
→ existing Step 6 SIFT features
→ keypoints inside vs outside predicted vessel mask
```

The figure must not mention SAM or imply reconstruction improvement.

- [ ] **Step 3: Run final ML verification**

Future implementation verification should include:

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_segmentation_data.py tests/test_cnn_segmentation.py tests/test_train_cnn_segmentation.py tests/test_ml_feature_analysis.py tests/test_run_ml_analysis.py
```

Then run the real frozen-model held-out evaluation and Step 8 analysis.

- [ ] **Step 4: Verify source immutability**

Re-hash all 297 raw photographs and all 288 selected images against existing manifests. Expected: zero mismatches.

- [ ] **Step 5: Write measured results documentation**

`docs/geometry-ml/ml-results.md` should record:

- exact 24/6/6 initial split and any training-only expansion;
- annotation convention/tool and labeled-manifest hash;
- SmallSegCNN architecture and actual parameter count;
- training configuration and seed;
- training/validation curves and best epoch;
- held-out Dice/IoU/precision/recall;
- visible failure cases;
- Step 8 SIFT inside/outside counts;
- explicit limitation that no reconstruction experiment has been run.

- [ ] **Step 6: Cleanup generated residue**

Remove caches, temporary checkpoints, failed partial runs, scratch masks, and debug exports. Preserve intentional labeled masks, frozen split manifest, final reports, and presentation figures. Keep only the final best checkpoint locally by default unless the user explicitly chooses to publish model weights.

---

## Completion gate

Steps 7+8 are complete only when future implementation evidence proves all of the following:

- the segmentation model was trained from random initialization with no pretrained backbone/checkpoint;
- the labeled dataset/split is reproducible and leakage-controlled;
- 24 train / 6 validation / 6 held-out test masks exist for the initial run, with only training-only expansion allowed afterward;
- the test split remained untouched during model/hyperparameter selection;
- real training curves and best-validation checkpoint provenance exist;
- all six held-out test images have unedited CNN predictions and measured Dice/IoU/precision/recall;
- weak test predictions are retained and documented rather than hidden or manually repaired;
- Step 8 reuses the completed Step 6 SIFT extraction/scale metadata;
- feature-mask figures use CNN-predicted masks, not ground truth, for the primary Step 8 result;
- all presentation figures are generated from real outputs and visually inspected;
- all 297 raw and 288 selected photographs remain hash-identical;
- no claim is made that CNN masking improves reconstruction;
- no pyCOLMAP/reconstruction, meshing, texturing, or Blender work was started as part of this plan.

## Plan self-review

- The model is now a custom CNN segmentation network rather than a pretrained model.
- The segmentation objective is pixel-level binary masking, not classification.
- Manual labels are separated into train/validation/test with explicit leakage controls for sequential near-neighbor photographs.
- Architecture/model selection uses only train+validation evidence; the test set is held out until the model is frozen.
- The plan starts small at 36 labels and permits only one bounded training-data expansion instead of immediately labeling all 288 images.
- The recommended U-Net-like skips improve boundary localization while remaining explainable as ordinary convolutional encoder/decoder operations.
- Dice and IoU are primary because pixel accuracy can be misleading on background-heavy segmentation.
- Step 8 uses predictions from held-out images and the existing Step 6 SIFT contract.
- No SAM, pretrained backbone, reconstruction, or generative image manipulation remains in the planned architecture.
