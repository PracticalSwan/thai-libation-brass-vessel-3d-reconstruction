# Geometry and ML Implementation Plan Index

> **For agentic workers:** This file is an index only. Do not execute it as a combined implementation plan.

Updated: 2026-08-28

## Current implementation scope

The previous combined geometry/ML/reconstruction plan has been split along the
current project boundary. **Step 6 is implemented and verified. Steps 7+8 are
provisional and must be redesigned/reconfirmed before execution. pyCOLMAP and
reconstruction are not part of the current implementation scope.**

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
- a reusable verified selected-image/SIFT scale contract;
- a Python popup visualizer that reuses the real Step 6 implementation.

Preferred indices are defaults subject to measured/visual substitution with a
documented reason. Hard boundary: no SAM 2, no ML masks, no pyCOLMAP.

### Steps 7 + 8 — Machine Learning + Feature-Mask Analysis

Provisional planning source:

`docs/superpowers/plans/2026-08-27-steps-7-8-ml-segmentation-feature-mask-analysis.md`

Scope:

- isolated SAM 2.1 environment;
- `sam2.1_hiera_small` feasibility check;
- prompt-based vessel segmentation on ten representative selected images;
- binary mask QA and presentation overlays;
- SIFT features inside versus outside the vessel masks;
- feature-mask counts, summary figures, and measured ML results.

Prerequisite: Step 6 must provide deterministic manifest access, verified
selected-image loading, public SIFT keypoints/descriptors, original/analysis
sizes, and explicit coordinate-scale metadata. The later ML file/class layout is
not fixed and must not shape Step 6 beyond this clean dependency contract.

Hard boundary: no pyCOLMAP, no sparse/dense reconstruction, no meshing, no texturing, no Blender.

## Current order

```text
Verified preprocessing (complete)
        ↓
Step 6: Geometry Detection / Analysis (complete and verified)
        ↓
Steps 7 + 8: SAM 2.1 Segmentation + Feature-Mask Analysis (provisional)
        ↓
STOP
```

A reconstruction implementation plan will be created only when the project is explicitly ready to move beyond Steps 6-8.

## Shared design source

Both implementation plans use:

`docs/superpowers/specs/2026-08-27-geometry-ml-integration-design.md`

The design document describes the geometry and intended ML behavior. The narrow
Step 6 plan now records the implemented contract and links to measured results.
The Steps 7+8 document remains an outcome-oriented placeholder that must be
revised using the completed Step 6 evidence.
