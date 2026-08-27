# Geometry and ML Implementation Plan Index

> **For agentic workers:** This file is an index only. Do not execute it as a combined implementation plan.

Updated: 2026-08-27

## Current implementation scope

The previous combined geometry/ML/reconstruction plan has been split into two independent implementation plans matching the current project pipeline. **pyCOLMAP and reconstruction are not part of the current implementation scope.**

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
- real presentation figures and measured geometry reports.

Hard boundary: no SAM 2, no ML masks, no pyCOLMAP.

### Steps 7 + 8 — Machine Learning + Feature-Mask Analysis

Use:

`docs/superpowers/plans/2026-08-27-steps-7-8-ml-segmentation-feature-mask-analysis.md`

Scope:

- isolated SAM 2.1 environment;
- `sam2.1_hiera_small` feasibility check;
- prompt-based vessel segmentation on ten representative selected images;
- binary mask QA and presentation overlays;
- SIFT features inside versus outside the vessel masks;
- feature-mask counts, summary figures, and measured ML results.

Prerequisite: Step 6 must provide the verified input loader and public SIFT extraction interface.

Hard boundary: no pyCOLMAP, no sparse/dense reconstruction, no meshing, no texturing, no Blender.

## Current order

```text
Verified preprocessing ✅
        ↓
Step 6: Geometry Detection / Analysis
        ↓
Steps 7 + 8: SAM 2.1 Segmentation + Feature-Mask Analysis
        ↓
STOP
```

A reconstruction implementation plan will be created only when the project is explicitly ready to move beyond Steps 6-8.

## Shared design source

Both implementation plans use:

`docs/superpowers/specs/2026-08-27-geometry-ml-integration-design.md`

The design document describes the intended geometry and ML behavior. When implementation starts, the narrower Step 6 and Steps 7+8 plans above control task sequencing and scope.
