# Geometry and ML Implementation Plan Index

> **For agentic workers:** This file is an index only. Do not execute it as a combined implementation plan.

Updated: 2026-09-05

## Current implementation scope

The coursework extension is complete through Step 8. **Step 6 geometry analysis and Steps 7+8 custom CNN segmentation/SIFT feature-mask analysis are implemented, measured, visually reviewed, and verified. pyCOLMAP and reconstruction remain outside the authorized scope.**

### Step 6 — Geometry Detection / Analysis

Plan:

`docs/superpowers/plans/2026-08-27-step-6-geometry-detection-analysis.md`

Completed scope:

- SIFT keypoints and candidate matches;
- Fundamental Matrix estimation;
- RANSAC geometric inliers;
- epipolar-line visualization and Sampson residuals;
- Canny edges and contour detection;
- ellipse fitting where valid;
- centroid, bounding box, and principal/symmetry axis;
- real presentation figures and measured geometry reports;
- reusable verified selected-image/SIFT scale contract;
- Python popup visualizer that reuses the real Step 6 implementation.

Measured Step 6 results are in `docs/geometry-ml/geometry-results.md`.

### Steps 7 + 8 — Custom CNN Segmentation + Feature-Mask Analysis

Plan:

`docs/superpowers/plans/2026-08-27-steps-7-8-ml-segmentation-feature-mask-analysis.md`

Completed scope:

- froze 36 reviewed source-size masks using a sequence-aware 24 train / 6 validation / 6 held-out test split;
- trained a project-defined compact U-Net-like `SmallSegCNN` from random initialization with no pretrained weights/backbone;
- used 384 x 288 `(H x W)` model input geometry, BCE-with-logits + Dice loss, Adam, seed 4213, validation-based early stopping, and fixed 0.5 threshold;
- selected the checkpoint from validation only, then evaluated all six frozen held-out images without changing model or threshold;
- measured held-out mean Dice 0.952521 and mean IoU 0.910745 while retaining the image-72 yellow-wall `background_false_positive`;
- reused `geometry_detection.extract_sift` and the Step 6 scale contract to classify SIFT keypoints inside/outside CNN-predicted test masks;
- retained all weak predictions and feature-mask limitations in machine-readable reports and presentation figures;
- generated and visually reviewed six ML presentation figures from real outputs.

Measured dataset and ML results are in:

- `docs/geometry-ml/cnn-dataset.md`;
- `docs/geometry-ml/ml-results.md`.

The CNN performs binary semantic segmentation, not image classification. It outputs a one-channel vessel/background mask rather than a single class label.

Hard boundary: no pretrained segmentation model, no SAM, no external segmentation API, no pyCOLMAP, no sparse/dense reconstruction, no meshing, no texturing, no Blender.

## Current order

```text
Verified preprocessing (complete)
        ↓
Step 6: Geometry Detection / Analysis (complete and verified)
        ↓
Steps 7 + 8: From-scratch CNN Segmentation + Feature-Mask Analysis (complete and verified)
        ↓
STOP before pyCOLMAP/reconstruction
```

A reconstruction implementation plan will be created only when the project is explicitly authorized to move beyond Steps 6-8.

## Shared design source

Both implementation plans use:

`docs/superpowers/specs/2026-08-27-geometry-ml-integration-design.md`

The design document now records the completed geometry contract and the implemented CNN-based ML architecture. Current measured evidence, rather than the old planning-only state, is authoritative for the project status.
