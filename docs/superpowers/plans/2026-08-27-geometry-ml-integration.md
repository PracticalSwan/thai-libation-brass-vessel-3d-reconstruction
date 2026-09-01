# Geometry and ML Implementation Plan Index

> **For agentic workers:** This file is an index only. Do not execute it as a combined implementation plan.

Updated: 2026-09-01

## Current implementation scope

The project is split along the current coursework boundary. **Step 6 is implemented and verified. Steps 7+8 now have an approved CNN-based planning architecture but remain unimplemented and require separate execution authorization. pyCOLMAP and reconstruction are not part of the current implementation scope.**

### Step 6 — Geometry Detection / Analysis

Use:

`docs/superpowers/plans/2026-08-27-step-6-geometry-detection-analysis.md`

Scope:

- SIFT keypoints and candidate matches;
- Fundamental Matrix estimation;
- RANSAC geometric inliers;
- epipolar-line visualization and Sampson/geometric residuals;
- Canny edges;
- contour detection;
- ellipse fitting where valid;
- centroid, bounding box, and principal/symmetry axis;
- real presentation figures and measured geometry reports;
- reusable verified selected-image/SIFT scale contract;
- Python popup visualizer that reuses the real Step 6 implementation.

Step 6 is a completed historical implementation boundary. It does not need to be redesigned around the future CNN.

### Steps 7 + 8 — Custom CNN Segmentation + Feature-Mask Analysis

Planning source:

`docs/superpowers/plans/2026-08-27-steps-7-8-ml-segmentation-feature-mask-analysis.md`

Current scope:

- manually annotate an initial 36-image segmentation dataset without modifying the verified source JPEGs;
- use a leakage-controlled 24 train / 6 validation / 6 held-out test split based on separated capture positions/view groups rather than a random neighboring-frame shuffle;
- train a small U-Net-like `SmallSegCNN` from random initialization with no pretrained backbone/checkpoint;
- preferred input tensor geometry `384 x 288` `(H x W)` to preserve the source portrait aspect ratio;
- train with BCE-with-logits + Dice loss, Adam, fixed seed `4213`, validation-based early stopping, and no test-set tuning;
- evaluate held-out predictions with Dice, IoU, foreground precision, recall, and visible failure analysis;
- reuse Step 6 SIFT extraction/scale metadata to count keypoints inside versus outside CNN-predicted test masks;
- create training, segmentation, feature-mask, and final summary figures from real outputs;
- retain weak predictions in the measured evidence rather than hiding or manually repairing them.

The planned CNN is semantic segmentation, not image classification. It outputs a one-channel vessel/background mask rather than a single class label.

Prerequisite: Step 6 must continue to provide deterministic manifest access, verified selected-image loading, public SIFT keypoints/descriptors, original/analysis sizes, and explicit coordinate-scale metadata. The CNN code consumes those interfaces but does not change them.

Hard boundary: no pretrained segmentation model, no SAM, no external segmentation API, no pyCOLMAP, no sparse/dense reconstruction, no meshing, no texturing, no Blender.

## Current order

```text
Verified preprocessing (complete)
        ↓
Step 6: Geometry Detection / Analysis (complete and verified)
        ↓
Steps 7 + 8: From-scratch CNN Segmentation + Feature-Mask Analysis (planned)
        ↓
STOP
```

A reconstruction implementation plan will be created only when the project is explicitly ready to move beyond Steps 6-8.

## Shared design source

Both implementation plans use:

`docs/superpowers/specs/2026-08-27-geometry-ml-integration-design.md`

The design document records the completed geometry contract and the current CNN-based ML architecture. The Steps 7+8 plan is detailed enough for a future executor, but this planning-only revision does not create labels, code, dependencies, training runs, model weights, or ML outputs.
